"""Shared backend-local value extraction for radar and rose charts."""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import matplotlib.pyplot as plt

from gemini_calls import FAILURE_TEXT, chat_with_gemini, get_last_modal_call_id, reset_modal_call_context, set_modal_call_context

from ..common.chart_io import ensure_dir, filter_chart_configs, image_to_data_url, read_json
from ..common.json_utils import parse_model_json, unwrap_openai_content
from ..common.model_config import get_chat_completion_urls, get_headers, get_model_name, use_legacy_pixtral
from ..common.paths import RESULTS_ROOT
from ..common.gt_grid_renderer import render_gt_grid_image
from ..common.runtime import (
    get_amplifier_rounds,
    get_feedback_rounds,
    get_prompt_rounds,
    get_repeat_times,
    numeric_axis_span,
)
from .polar_visual import add_target_color_swatch, crop_polar_amplifier, draw_polar_feedback, safe_name
from .circular_model_config import (
    get_chat_completion_urls as get_circular_chat_completion_urls,
    get_headers as get_circular_headers,
    get_model_name as get_circular_model_name,
)


POLAR_ABSOLUTE_STABILITY_TOLERANCE = 2.0


@dataclass(frozen=True)
class PolarTarget:
    chart_id: str
    point_name: str
    series_name: str
    theta_label: str
    color: str | None


def load_backend_polar_datasets(
    chart_type: str,
    chart_ids: Iterable[str] | None = None,
    config_paths: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if not config_paths:
        return []
    datasets = [_load_backend_polar_dataset(path, chart_type) for path in config_paths]
    return filter_chart_configs(datasets, chart_ids)


def iter_polar_targets(dataset: dict[str, Any], chart_type: str) -> list[PolarTarget]:
    chart_id = str(dataset["chart_id"])
    theta_labels = _theta_labels(dataset)
    colors = _series_color(dataset)

    if chart_type == "radar":
        targets: list[PolarTarget] = []
        for series_name, color in colors.items():
            for theta_label in theta_labels:
                targets.append(
                    PolarTarget(
                        chart_id=chart_id,
                        point_name=f"{series_name}, {theta_label}",
                        series_name=series_name,
                        theta_label=theta_label,
                        color=color,
                    )
                )
        return targets

    names = theta_labels or list(colors)
    return [
        PolarTarget(
            chart_id=chart_id,
            point_name=name,
            series_name="",
            theta_label=name,
            color=colors.get(name),
        )
        for name in names
    ]


def image_path(dataset: dict[str, Any], image_type: str) -> Path:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    if image_type == "grid_with_grid":
        value = image_paths.get("grid_with_grid") or image_paths.get("with_grid") or dataset.get("encrypted_grid_path")
    elif image_type == "with_grid":
        value = image_paths.get("with_grid") or image_paths.get("grid_with_grid") or dataset.get("encrypted_grid_path")
    else:
        value = image_paths.get("no_grid") or dataset.get("image_path")
    if not value:
        raise KeyError(f"Missing image path for {image_type!r} in {dataset.get('chart_id')!r}")
    return Path(value).resolve()


def build_prompt(
    dataset: dict[str, Any],
    target: PolarTarget,
    chart_type: str,
    prompt_type: str,
    *,
    prev_r: float | None = None,
    visible_ticks: list[float] | None = None,
) -> str:
    r_ticks = dataset.get("r_ticks", [])
    r_pixels = dataset.get("r_pixels") or dataset.get("radius") or []
    theta_ticks = _theta_labels(dataset)
    theta_angles = dataset.get("theta_angles") or dataset.get("axes_angles") or []
    color_hint = f"The target color is {target.color}." if target.color else ""
    if prompt_type == "grid":
        grid_hint = "Use the encrypted radial grid and interpolate between the nearest radial ticks."
    elif prompt_type == "feedback":
        grid_hint = (
            f"The image contains a red visual marker for the previous prediction r={prev_r}. "
            "Compare that marker with the true target boundary/point and refine the radial value."
        )
    elif prompt_type == "amplifier":
        if chart_type == "radar":
            grid_hint = (
                "The image is a 3x zoom-in crop along the target spoke. It preserves the full radial range from the center to the outer ring, "
                "so you must find the colored vertex/intersection on the named spoke rather than trust the previous red marker. "
                f"Visible local radial tick values are: {json.dumps(visible_ticks or [], ensure_ascii=False)}. "
                "The red circular guide labels include original radial tick values and midpoint values between adjacent ticks. "
                "Use the two nearest red rings around the target vertex and interpolate the fractional radial position between them. "
                "Do not return crop pixel coordinates or image-relative distances."
            )
        else:
            grid_hint = (
                "The image is a 3x local zoom-in crop generated around the previous model prediction, not around GT. "
                f"Visible local radial tick values are: {json.dumps(visible_ticks or [], ensure_ascii=False)}. "
                "The red circular guide labels include original radial tick values and midpoint values between adjacent ticks. "
                "For a rose chart, the two red radial boundary lines bracket the target sector; only read the bar inside those boundaries. "
                "Use the two nearest red rings around the target point or boundary and interpolate the fractional radial position between them. "
                "Do not return crop pixel coordinates or image-relative distances."
            )
    else:
        grid_hint = "Estimate from the original chart without relying on ground-truth data."

    if chart_type == "radar":
        return f"""
You are analyzing a radar chart.
The radial tick values are: {json.dumps(r_ticks, ensure_ascii=False)}.
The GT radial tick-to-pixel-radius mapping is: {json.dumps(_tick_pixel_pairs(r_ticks, r_pixels), ensure_ascii=False)}.
The angular axis labels are: {json.dumps(theta_ticks, ensure_ascii=False)}.
Their angle positions are: {json.dumps(theta_angles, ensure_ascii=False)} degrees.
The chart entities and colors are: {json.dumps(_series_color(dataset), ensure_ascii=False)}.

Task: locate entity "{target.series_name}" on axis "{target.theta_label}" and estimate its radial data value.
{color_hint}
{grid_hint}

Read the value on the named spoke only:
- Find the spoke whose label is exactly "{target.theta_label}".
- Follow that spoke from the center outward.
- For entity "{target.series_name}", use the colored polyline vertex or filled-boundary intersection on that spoke.
- Do not use a neighboring spoke, the filled polygon area away from the spoke, or the outer label position.
- Interpolate the vertex radius between the two nearest radial tick rings. If the vertex is between rings, return the fractional value, not just a tick label.

Return only JSON in this exact shape:
{{"datapoints":[{{"{target.point_name}":[r_value,null]}}]}}
""".strip()

    return f"""
You are analyzing a rose chart.
The radial tick values are: {json.dumps(r_ticks, ensure_ascii=False)}.
The GT radial tick-to-pixel-radius mapping is: {json.dumps(_tick_pixel_pairs(r_ticks, r_pixels), ensure_ascii=False)}.
The sector labels are: {json.dumps(theta_ticks, ensure_ascii=False)}.
Their angle positions are: {json.dumps(theta_angles, ensure_ascii=False)} degrees.
The sector colors are: {json.dumps(_series_color(dataset), ensure_ascii=False)}.

Task: locate sector "{target.theta_label}" and estimate the radial value represented by the sector's outer boundary.
{color_hint}
{grid_hint}

Return only JSON in this exact shape:
{{"datapoints":[{{"{target.point_name}":[r_value,null]}}]}}
""".strip()


class PolarModelClient:
    def __init__(self, legacy_urls: list[str] | None = None, *, max_retries: int = 3) -> None:
        if _use_circular_model_fallback():
            self.urls = get_circular_chat_completion_urls()
            self.headers = get_circular_headers()
            self.model_name = get_circular_model_name()
        else:
            self.urls = get_chat_completion_urls(legacy_urls)
            self.headers = get_headers()
            self.model_name = os.getenv("POLAR_CHART_MODEL_NAME") or get_model_name()
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_connect=30, sock_read=240)
        self._session: aiohttp.ClientSession | None = None
        self._index = 0

    async def __aenter__(self) -> "PolarModelClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    def _next_url(self) -> str:
        url = self.urls[self._index]
        self._index = (self._index + 1) % len(self.urls)
        return url

    def _payload(self, prompt: str, used_image: Path) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(used_image)}},
                    ],
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.0,
        }

    async def predict_value(self, prompt: str, used_image: Path, point_name: str) -> float | None:
        content = await self._call_text(prompt, used_image, point_name)
        if not content:
            return None
        parsed = parse_model_json(content)
        return extract_radial_value(parsed, point_name)

    async def predict_all_values(self, prompt: str, used_image: Path, chart_id: str) -> list[dict[str, Any]]:
        content = await self._call_text(prompt, used_image, f"{chart_id}:all")
        if not content:
            return []
        parsed = parse_model_json(content)
        return extract_whole_chart_predictions(parsed)

    async def _call_text(self, prompt: str, used_image: Path, label: str) -> str | None:
        url = self._next_url()
        print(f"[polar model] {label} -> {url}")
        request_urls = [url] + [item for item in self.urls if item != url]
        content = await chat_with_gemini(
            self._payload(prompt, used_image)["messages"],
            model=self.model_name,
            max_tokens=8192,
            temperature=0.0,
            urls=request_urls,
            headers=self.headers,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return None if content == FAILURE_TEXT else content


def _use_circular_model_fallback() -> bool:
    raw = os.getenv("POLAR_USE_CIRCULAR_MODEL", "").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if os.getenv("POLAR_CHART_MODEL_NAME") or os.getenv("CHART_MODEL_NAME") or use_legacy_pixtral():
        return False
    return get_model_name().strip().lower() == "gpt-5.4"


def extract_radial_value(parsed: Any, point_name: str) -> float | None:
    items = parsed.get("datapoints") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return _number_or_none(parsed.get(point_name) if isinstance(parsed, dict) else parsed)

    point_key = _key(point_name)
    fallback = None
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if _key(key) == point_key:
                return _value_to_number(value)
            if fallback is None:
                fallback = _value_to_number(value)
        name = item.get("name") or item.get("label")
        if _key(name) == point_key:
            return _value_to_number(item.get("value", item.get("r_value", item.get("r"))))
    return fallback


def extract_whole_chart_predictions(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        for key in ("datapoints", "data_points", "predictions", "segments", "sectors", "objects"):
            items = parsed.get(key)
            if isinstance(items, list):
                return _normalize_whole_items(items)
        if parsed:
            item = _normalize_whole_item(parsed, 0)
            return [item] if item else []
    if isinstance(parsed, list):
        return _normalize_whole_items(parsed)
    return []


def _normalize_whole_items(items: list[Any]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        normalized = _normalize_whole_item(item, index)
        if normalized:
            predictions.append(normalized)
    return predictions


def _normalize_whole_item(item: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    common_keys = {
        "id",
        "name",
        "label",
        "series_name",
        "series",
        "theta_label",
        "axis",
        "category",
        "sector",
        "value",
        "r",
        "r_value",
        "radius",
        "color",
    }
    data = dict(item)
    if not any(key in data for key in common_keys):
        for key, value in data.items():
            number = _value_to_number(value)
            if number is None:
                continue
            series_name, theta_label = _split_point_name(str(key))
            return {
                "id": str(key),
                "series_name": series_name,
                "theta_label": theta_label,
                "label": theta_label or str(key),
                "value": number,
                "color": None,
            }

    value = _value_to_number(
        data.get("value", data.get("r", data.get("r_value", data.get("radius"))))
    )
    if value is None:
        return None

    raw_series = data.get("series_name", data.get("series", ""))
    raw_label = data.get(
        "theta_label",
        data.get("axis", data.get("category", data.get("sector", data.get("label", data.get("name", ""))))),
    )
    point_id = data.get("id") or data.get("name")
    if point_id and (not raw_series or not raw_label):
        split_series, split_label = _split_point_name(str(point_id))
        raw_series = raw_series or split_series
        raw_label = raw_label or split_label
    label = str(raw_label or point_id or f"Object {index + 1}").strip()
    series_name = str(raw_series or "").strip()
    if point_id:
        point_id = str(point_id).strip()
    else:
        point_id = f"{series_name}, {label}" if series_name else label

    return {
        "id": point_id,
        "series_name": series_name,
        "theta_label": label,
        "label": label,
        "value": value,
        "color": data.get("color"),
    }


def build_whole_chart_prompt(dataset: dict[str, Any], chart_type: str, prompt_type: str) -> str:
    r_ticks = dataset.get("r_ticks", [])
    theta_ticks = _theta_labels(dataset)
    theta_angles = dataset.get("theta_angles") or dataset.get("axes_angles") or []
    series_color = _series_color(dataset)
    grid_hint = (
        "Use the encrypted radial grid and interpolate between the nearest radial ticks."
        if prompt_type == "grid"
        else "Estimate from the visible chart without using any ground-truth JSON."
    )

    if chart_type == "radar":
        return f"""
You are analyzing a radar chart from a user-uploaded image.
The radial tick values are: {json.dumps(r_ticks, ensure_ascii=False)}.
The angular axis labels are: {json.dumps(theta_ticks, ensure_ascii=False)}.
Their angle positions are: {json.dumps(theta_angles, ensure_ascii=False)} degrees.
Known series colors, if any: {json.dumps(series_color, ensure_ascii=False)}.

Task: extract every visible radar data object. For each visible series and each angular axis, estimate the radial data value.
If the chart has only one visible series, use an empty series_name.
{grid_hint}

Return only JSON in this exact shape:
{{"datapoints":[{{"series_name":"", "axis":"A", "value":80.0}}]}}
""".strip()

    return f"""
You are analyzing a rose chart from a user-uploaded image.
The radial tick values, if detected, are: {json.dumps(r_ticks, ensure_ascii=False)}.
The sector labels, if detected, are: {json.dumps(theta_ticks, ensure_ascii=False)}.
Their angle positions, if detected, are: {json.dumps(theta_angles, ensure_ascii=False)} degrees.
Known sector colors, if any: {json.dumps(series_color, ensure_ascii=False)}.

Task: extract every visible rose sector. For each sector, report its label/name and the radial value represented by its outer boundary.
If there are no visible labels, use stable labels such as Sector 1, Sector 2 in clockwise order.
{grid_hint}

Return only JSON in this exact shape:
{{"datapoints":[{{"label":"Sector 1", "value":80.0}}]}}
""".strip()


def select_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = _number_or_none(record.get("pred_r"))
        if value is not None:
            by_point.setdefault(str(record["point"]), []).append(record)

    predictions: list[dict[str, Any]] = []
    for point, point_records in by_point.items():
        chosen = _choose_record(point_records)
        if chosen is None:
            continue
        value = _number_or_none(chosen.get("pred_r"))
        if value is None:
            continue
        predictions.append(
            {
                "id": point,
                "series_name": chosen.get("series_name", ""),
                "theta_label": chosen.get("theta_label") or point,
                "label": chosen.get("theta_label") or point,
                "axis": "r",
                "value": value,
                "r": value,
                "prompt_type": chosen.get("prompt_type"),
                "image_type": chosen.get("image_type"),
                "image_path": chosen.get("image_path"),
                "selection_reason": chosen.get("selection_reason"),
            }
        )
    return predictions


async def run_polar_experiment(
    chart_type: str,
    *,
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
    legacy_urls: list[str] | None = None,
) -> list[dict[str, Any]]:
    datasets = load_backend_polar_datasets(chart_type, chart_ids, config_paths)
    if not datasets:
        print(f"[{chart_type}] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times(default=1)
    all_records: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    async with PolarModelClient(legacy_urls=legacy_urls) as client:
        for start in range(0, len(datasets), batch_size or len(datasets)):
            batch = datasets[start : start + (batch_size or len(datasets))]
            for dataset in batch:
                targets = iter_polar_targets(dataset, chart_type)
                records: list[dict[str, Any]] = []
                if targets:
                    if os.getenv("CHART_EXPERIMENT_MODE", "").strip().lower() == "gt":
                        for target in targets:
                            records.extend(await _run_target(client, dataset, target, chart_type, repeat_times))
                    else:
                        tasks = [
                            _run_target(client, dataset, target, chart_type, repeat_times)
                            for target in targets
                        ]
                        for result in await asyncio.gather(*tasks):
                            records.extend(result)
                expected_points = {target.point_name for target in targets}
                predictions = select_predictions(records)
                predicted_points = {str(item.get("id")) for item in predictions}
                needs_fallback = (
                    not predictions
                    or bool(expected_points and not expected_points.issubset(predicted_points))
                )
                if needs_fallback:
                    fallback_records = await _run_whole_chart(client, dataset, chart_type, repeat_times)
                    records.extend(fallback_records)
                    predictions = select_predictions(records)
                all_records.extend(records)
                all_summaries.append(_chart_summary(dataset, chart_type, records, predictions))

    by_chart: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_chart.setdefault(str(record["chart_id"]), []).append(record)

    summaries: list[dict[str, Any]] = []
    for chart_id, records in by_chart.items():
        result_dir = ensure_dir(RESULTS_ROOT / chart_type / chart_id)
        _save_records(records, result_dir)
        predictions = select_predictions(records)
        summaries.append(_chart_summary({"chart_id": chart_id}, chart_type, records, predictions, result_dir))
    if summaries:
        return summaries
    return all_summaries


def _chart_summary(
    dataset: dict[str, Any],
    chart_type: str,
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    result_dir: Path | None = None,
) -> dict[str, Any]:
    chart_id = str(dataset["chart_id"])
    result_dir = result_dir or ensure_dir(RESULTS_ROOT / chart_type / chart_id)
    if records:
        _save_run_artifacts(records, predictions, result_dir, chart_type)
    return {
        "chart_id": chart_id,
        "result_dir": str(result_dir),
        "record_count": len(records),
        "object_count": len(predictions),
        "predictions": predictions,
    }



async def _run_target(
    client: PolarModelClient,
    dataset: dict[str, Any],
    target: PolarTarget,
    chart_type: str,
    repeat_times: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    result_dir = ensure_dir(RESULTS_ROOT / chart_type / str(dataset["chart_id"]))

    baseline_image = image_path(dataset, "no_grid")
    baseline_pred = await _predict_target_stage(
        client=client,
        dataset=dataset,
        target=target,
        chart_type=chart_type,
        prompt_type="baseline",
        image_type="no_grid",
        run_idx=1,
        used_image=baseline_image,
    )
    records.append(baseline_pred)

    try:
        grid_image = image_path(dataset, "grid_with_grid")
    except KeyError:
        grid_image = image_path(dataset, "with_grid")
    grid_image = add_target_color_swatch(
        image_path=grid_image,
        output_path=ensure_dir(result_dir / "grid_img") / f"{safe_name(target.point_name)}_grid_target.png",
        point_name=target.point_name,
        color=target.color,
    )
    grid_record = await _predict_target_stage(
        client=client,
        dataset=dataset,
        target=target,
        chart_type=chart_type,
        prompt_type="grid",
        image_type="grid_with_grid",
        run_idx=1,
        used_image=grid_image,
    )
    records.append(grid_record)

    last_pred = _number_or_none(grid_record.get("pred_r"))
    feedback_pred = last_pred
    for feedback_round in range(1, get_feedback_rounds(2) + 1):
        if feedback_pred is None:
            records.append(_empty_stage_record(dataset, target, "feedback", "feedback", feedback_round, grid_image))
            break
        previous_feedback_pred = feedback_pred
        feedback_image = draw_polar_feedback(
            dataset=dataset,
            chart_type=chart_type,
            source_image=grid_image,
            result_dir=result_dir,
            point_name=target.point_name,
            theta_label=target.theta_label,
            pred_r=feedback_pred,
            round_index=feedback_round,
        )
        feedback_image = add_target_color_swatch(
            image_path=feedback_image,
            point_name=target.point_name,
            color=target.color,
        )
        feedback_record = await _predict_target_stage(
            client=client,
            dataset=dataset,
            target=target,
            chart_type=chart_type,
            prompt_type="feedback",
            image_type="feedback",
            run_idx=feedback_round,
            used_image=feedback_image,
            prev_r=feedback_pred,
        )
        records.append(feedback_record)
        last_pred = _number_or_none(feedback_record.get("pred_r"))
        is_stable = _polar_prediction_stable(
            last_pred,
            previous_feedback_pred,
            axis_span=_polar_axis_span(dataset),
        )
        if last_pred is not None:
            feedback_pred = last_pred
        if is_stable:
            print(f"[{chart_type}] Feedback prediction stabilized at round {feedback_round}; enter amplifier.")
            break

    amp_pred = feedback_pred if feedback_pred is not None else last_pred
    for amp_round in range(1, get_amplifier_rounds(3) + 1):
        if amp_pred is None:
            records.append(_empty_stage_record(dataset, target, "amplifier", "amplifier", amp_round, grid_image))
            break
        previous_amp_pred = amp_pred
        try:
            crop_image, visible_ticks = crop_polar_amplifier(
                dataset=dataset,
                chart_type=chart_type,
                source_image=image_path(dataset, "no_grid"),
                result_dir=result_dir,
                point_name=target.point_name,
                theta_label=target.theta_label,
                pred_r=amp_pred,
                round_index=amp_round,
            )
            crop_image = add_target_color_swatch(
                image_path=crop_image,
                point_name=target.point_name,
                color=target.color,
            )
        except Exception as exc:
            print(f"[{chart_type}] Amplifier crop failed for {target.point_name}, round {amp_round}: {exc}")
            records.append(_empty_stage_record(dataset, target, "amplifier", "amplifier", amp_round, grid_image))
            break
        amp_record = await _predict_target_stage(
            client=client,
            dataset=dataset,
            target=target,
            chart_type=chart_type,
            prompt_type="amplifier",
            image_type="amplifier",
            run_idx=amp_round,
            used_image=crop_image,
            prev_r=amp_pred,
            visible_ticks=visible_ticks,
        )
        records.append(amp_record)
        amp_pred = _number_or_none(amp_record.get("pred_r"))
        if _polar_prediction_stable(
            amp_pred,
            previous_amp_pred,
            axis_span=_polar_axis_span(dataset),
        ):
            print(f"[{chart_type}] Amplifier prediction stabilized at round {amp_round}; stop amplifier.")
            break
    return records


async def _predict_target_stage(
    *,
    client: PolarModelClient,
    dataset: dict[str, Any],
    target: PolarTarget,
    chart_type: str,
    prompt_type: str,
    image_type: str,
    run_idx: int,
    used_image: Path,
    prev_r: float | None = None,
    visible_ticks: list[float] | None = None,
) -> dict[str, Any]:
    prompt = build_prompt(
        dataset,
        target,
        chart_type,
        prompt_type,
        prev_r=prev_r,
        visible_ticks=visible_ticks,
    )
    print(f"[{chart_type}] Round {run_idx} | Point: {target.point_name} | {prompt_type} - {image_type}")
    token = set_modal_call_context(
        _modal_context(dataset, target, prompt_type, run_idx, used_image)
    )
    try:
        pred = await client.predict_value(prompt, used_image, target.point_name)
        call_id = get_last_modal_call_id()
    finally:
        reset_modal_call_context(token)
    return _record(
        dataset=dataset,
        target=target,
        prompt_type=prompt_type,
        image_type=image_type,
        run_idx=run_idx,
        used_image=used_image,
        pred=pred,
        call_id=call_id,
    )


def _record(
    *,
    dataset: dict[str, Any],
    target: PolarTarget,
    prompt_type: str,
    image_type: str,
    run_idx: int,
    used_image: Path,
    pred: Any,
    call_id: str | None = None,
) -> dict[str, Any]:
    gt = _target_gt(dataset, target)
    pred_value = _number_or_none(pred)
    gt_value = _number_or_none(gt)
    mae = abs(pred_value - gt_value) if pred_value is not None and gt_value is not None else None
    r_re = abs(pred_value - gt_value) / abs(gt_value) if pred_value is not None and gt_value not in (None, 0) else None
    return {
        "chart_id": dataset["chart_id"],
        "call_id": call_id,
        "point": target.point_name,
        "series_name": target.series_name,
        "theta_label": target.theta_label,
        "prompt_type": prompt_type,
        "image_type": image_type,
        "run": run_idx,
        "image_path": str(used_image),
        "gt_r": gt,
        "pred_r": pred_value,
        "mae": mae,
        "r_re": r_re,
        "r_axis_span": numeric_axis_span(dataset.get("r_ticks")),
    }


def _empty_stage_record(
    dataset: dict[str, Any],
    target: PolarTarget,
    prompt_type: str,
    image_type: str,
    run_idx: int,
    used_image: Path,
) -> dict[str, Any]:
    return _record(
        dataset=dataset,
        target=target,
        prompt_type=prompt_type,
        image_type=image_type,
        run_idx=run_idx,
        used_image=used_image,
        pred=None,
    )


async def _run_whole_chart(
    client: PolarModelClient,
    dataset: dict[str, Any],
    chart_type: str,
    repeat_times: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_type, image_type in (("grid", "grid_with_grid"),):
        for run_idx in range(1, get_prompt_rounds(prompt_type, repeat_times) + 1):
            try:
                used_image = image_path(dataset, image_type)
            except KeyError:
                if image_type != "grid_with_grid":
                    raise
                used_image = image_path(dataset, "with_grid")
            prompt = build_whole_chart_prompt(dataset, chart_type, prompt_type)
            print(f"[{chart_type}] Round {run_idx} | Whole chart fallback | {prompt_type} - {image_type}")
            token = set_modal_call_context(
                {
                    "chart_name": dataset.get("chart_id"),
                    "processing_object": "__whole_chart__",
                    "object_category": chart_type,
                    "gt": _polar_gt_map(dataset),
                    "stage": prompt_type,
                    "round": run_idx,
                    "image_path": str(used_image),
                }
            )
            try:
                predictions = await client.predict_all_values(prompt, used_image, str(dataset["chart_id"]))
                call_id = get_last_modal_call_id()
            finally:
                reset_modal_call_context(token)
            if not predictions:
                records.append(
                    {
                        "chart_id": dataset["chart_id"],
                        "call_id": call_id,
                        "point": "__whole_chart__",
                        "series_name": "",
                        "theta_label": "",
                        "prompt_type": prompt_type,
                        "image_type": image_type,
                        "run": run_idx,
                        "image_path": str(used_image),
                        "gt_r": None,
                        "pred_r": None,
                        "mae": None,
                        "r_re": None,
                    }
                )
            for item in predictions:
                point = str(item.get("id") or item.get("label") or f"Object {len(records) + 1}")
                records.append(
                    {
                        "chart_id": dataset["chart_id"],
                        "call_id": call_id,
                        "point": point,
                        "series_name": item.get("series_name", ""),
                        "theta_label": item.get("theta_label") or item.get("label") or point,
                        "prompt_type": prompt_type,
                        "image_type": image_type,
                        "run": run_idx,
                        "image_path": str(used_image),
                        "gt_r": None,
                        "pred_r": item.get("value"),
                        "mae": None,
                        "r_re": None,
                    }
                )
            if predictions:
                return records
    return records


def _load_backend_polar_dataset(config_path: str | Path, chart_type: str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    base = _read_json_dict(path)
    if path.stem.endswith("_axes"):
        stem = path.stem.removesuffix("_axes")
        base = _read_json_dict(path.with_name(f"{stem}_image.json")) or _read_json_dict(path.with_name(f"{stem}.json"))
    axes = _read_json_dict(_sibling_axes_path(path))

    merged = dict(base)
    merged.update({key: value for key, value in axes.items() if key not in {"chart_id"}})
    merged["chart_type"] = chart_type
    merged["chart_id"] = str(base.get("chart_id") or path.stem.removesuffix("_image").removesuffix("_axes"))
    _strip_reference_data(merged)
    merged["series_color"] = _series_color(merged)
    merged["theta_ticks"] = _theta_labels(merged)
    merged["image_paths"] = _image_paths(merged, path.parent, chart_type, path.stem.removesuffix("_image").removesuffix("_axes"), path)
    return merged


def _strip_reference_data(dataset: dict[str, Any]) -> None:
    if os.getenv("CHART_EXPERIMENT_PRESERVE_GT", "").strip().lower() in {"1", "true", "yes"}:
        for key in ("reference_config_path", "reference_chart_id"):
            dataset.pop(key, None)
        return
    for key in ("data", "data_points", "ground_truth", "labels", "reference_config_path", "reference_chart_id"):
        dataset.pop(key, None)


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _sibling_axes_path(path: Path) -> Path:
    stem = path.stem.removesuffix("_image").removesuffix("_axes")
    for candidate in (path.with_name(f"{stem}_axes.json"), path.with_name(f"{stem}_image_axes.json")):
        if candidate.exists():
            return candidate
    return path.with_name(f"{stem}_axes.json")


def _image_paths(dataset: dict[str, Any], base_dir: Path, chart_type: str, stem: str, config_path: Path) -> dict[str, str]:
    raw = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    inferred_no_grid = base_dir / f"{stem}.png"
    no_grid = raw.get("no_grid") or dataset.get("image_path")
    if not no_grid and inferred_no_grid.exists():
        no_grid = str(inferred_no_grid)
    with_grid = (
        raw.get("grid_with_grid")
        or raw.get("with_grid")
        or dataset.get("encrypted_grid_path")
        or dataset.get("basic_grid_path")
    )
    encode_candidate = base_dir / f"{stem}_image_encode.png"
    if not encode_candidate.exists():
        encode_candidate = base_dir / f"{stem}_encode.png"
    if encode_candidate.exists() and (not with_grid or not _resolve_path(with_grid, base_dir).exists()):
        with_grid = str(encode_candidate)
    if not with_grid or not _resolve_path(with_grid, base_dir).exists():
        rendered = render_gt_grid_image(config_path, dataset=dataset)
        if rendered is not None:
            with_grid = str(rendered)
    values = {"no_grid": no_grid, "with_grid": with_grid, "grid_with_grid": with_grid}
    return {key: str(_resolve_path(value, base_dir)) for key, value in values.items() if isinstance(value, str) and value}


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    dataset_root = base_dir.parent
    candidates = [
        (base_dir / path).resolve(),
        (base_dir / path.name).resolve(),
        (base_dir / "chart" / path.name).resolve(),
        (base_dir / "charts" / path.name).resolve(),
        (base_dir / "charts" / path.parent.name / path.name).resolve(),
        (dataset_root / path).resolve(),
        (dataset_root / "charts" / path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _series_color(dataset: dict[str, Any]) -> dict[str, str]:
    raw = dataset.get("series_color")
    if isinstance(raw, dict) and raw:
        return {str(key): str(value) for key, value in raw.items() if value}

    result: dict[str, str] = {}
    colors = dataset.get("colors")
    if isinstance(colors, list):
        for item in colors:
            if isinstance(item, dict) and item.get("name") and item.get("color"):
                result[str(item["name"])] = str(item["color"])
    return result


def _polar_gt_map(dataset: dict[str, Any]) -> dict[str, Any]:
    raw = dataset.get("data_points") or dataset.get("ground_truth") or dataset.get("data")
    return raw if isinstance(raw, dict) else {}


def _target_gt(dataset: dict[str, Any], target: PolarTarget) -> Any:
    values = _polar_gt_map(dataset)
    candidates = [
        target.point_name,
        target.theta_label,
        f"{target.series_name}, {target.theta_label}" if target.series_name else "",
    ]
    for candidate in candidates:
        if candidate in values:
            return values[candidate]
    if target.series_name and target.series_name in values:
        nested = values[target.series_name]
        if isinstance(nested, dict):
            return nested.get(target.theta_label)
        return nested
    for key, value in values.items():
        if str(key).strip().casefold() in {str(item).strip().casefold() for item in candidates if item}:
            return value
        if not target.series_name and isinstance(value, dict) and target.theta_label in value:
            return value[target.theta_label]
    return None


def _modal_context(
    dataset: dict[str, Any],
    target: PolarTarget,
    stage: str,
    round_index: int,
    used_image: Path,
) -> dict[str, Any]:
    return {
        "chart_name": dataset.get("chart_id"),
        "processing_object": target.point_name,
        "object_category": target.series_name or target.theta_label,
        "gt": _target_gt(dataset, target),
        "stage": stage,
        "round": round_index,
        "image_path": str(used_image),
    }


def _theta_labels(dataset: dict[str, Any]) -> list[str]:
    raw = dataset.get("theta_ticks")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw]
    axis_labels = dataset.get("axis_labels")
    if isinstance(axis_labels, dict) and axis_labels:
        def angle_key(item: tuple[Any, Any]) -> float:
            try:
                return float(item[0])
            except Exception:
                return 0.0
        return [str(value) for _, value in sorted(axis_labels.items(), key=angle_key)]
    return list(_series_color(dataset))


def _choose_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected, _audit = _select_polar_full_flow_record(records)
    return selected


def _select_polar_full_flow_record(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    full_records = [
        record
        for record in records
        if record.get("prompt_type") in {"grid", "feedback", "amplifier"}
        and _number_or_none(record.get("pred_r")) is not None
    ]
    if not full_records:
        return None, {"reason": "no_readable_full_flow_prediction", "readable_full_flow_count": 0}

    amp_records = [record for record in full_records if record.get("prompt_type") == "amplifier"]
    feedback_records = [record for record in full_records if record.get("prompt_type") == "feedback"]
    grid_records = [record for record in full_records if record.get("prompt_type") == "grid"]
    axis_span = numeric_axis_span([record.get("gt_r") for record in full_records]) or _number_or_none(full_records[0].get("r_axis_span"))
    selected: dict[str, Any] | None
    reason: str
    stable_grid_feedback = _stable_polar_grid_feedback(grid_records, feedback_records, axis_span)
    stable_amp = _stable_polar_tail(amp_records, axis_span)
    full_view_median = _median(
        [
            value
            for record in (grid_records + feedback_records)
            for value in [_number_or_none(record.get("pred_r"))]
            if value is not None
        ]
    )
    outer_drift_reference = _normalized_polar_outer_drift_reference(
        grid_records,
        feedback_records,
        stable_amp,
        axis_span,
    )
    if outer_drift_reference is not None:
        selected, selected_value, reason = outer_drift_reference
    elif (
        stable_amp is not None
        and _normalized_outer_ring_amplifier_drift(stable_amp, full_view_median, axis_span)
        and (feedback_records or grid_records)
    ):
        selected = feedback_records[-1] if feedback_records else grid_records[-1]
        selected_value = full_view_median
        reason = "normalized_outer_ring_amplifier_drift_use_full_view_median"
    elif stable_amp is not None:
        selected = amp_records[-1]
        selected_value = stable_amp
        reason = "stable_amplifier_tail"
    elif stable_grid_feedback is not None:
        selected = feedback_records[-1]
        selected_value = stable_grid_feedback
        reason = "stable_grid_feedback"
    elif amp_records and len(amp_records) == 1 and (feedback_records or grid_records):
        selected = feedback_records[-1] if feedback_records else grid_records[-1]
        selected_value = _median(
            [
                value
                for record in (grid_records + feedback_records)
                for value in [_number_or_none(record.get("pred_r"))]
                if value is not None
            ]
        )
        reason = "weak_single_amplifier_use_full_view_median"
    elif len(amp_records) >= 2 and (feedback_records or grid_records):
        selected = feedback_records[-1] if feedback_records else grid_records[-1]
        selected_value = _median(
            [
                value
                for record in (grid_records + feedback_records)
                for value in [_number_or_none(record.get("pred_r"))]
                if value is not None
            ]
        )
        reason = "unstable_amplifier_use_full_view_median"
    elif amp_records:
        selected = amp_records[-1]
        selected_value = _number_or_none(selected.get("pred_r"))
        reason = "latest_amplifier_prediction"
    elif feedback_records:
        selected = feedback_records[-1]
        selected_value = _number_or_none(selected.get("pred_r"))
        reason = "latest_feedback_prediction"
    elif grid_records:
        selected = grid_records[-1]
        selected_value = _number_or_none(selected.get("pred_r"))
        reason = "grid_only_prediction"
    else:
        selected = None
        selected_value = None
        reason = "no_readable_full_flow_prediction"

    if selected is None or selected_value is None:
        return None, {"reason": reason, "readable_full_flow_count": len(full_records)}

    output = dict(selected)
    output["pred_r"] = selected_value
    output["prompt_type"] = "full_flow_final"
    output["image_type"] = "selected"
    output["run"] = "final"
    output["call_id"] = None
    output["selection_source_prompt_type"] = selected.get("prompt_type")
    output["selection_source_image_type"] = selected.get("image_type")
    output["selection_source_run"] = selected.get("run")
    output["selection_reason"] = reason
    gt_value = _number_or_none(output.get("gt_r"))
    if gt_value is not None:
        output["mae"] = abs(selected_value - gt_value)
        output["r_re"] = None if gt_value == 0 else abs(selected_value - gt_value) / abs(gt_value)
    return output, {
        "reason": reason,
        "source_prompt_type": selected.get("prompt_type"),
        "source_image_type": selected.get("image_type"),
        "source_run": selected.get("run"),
        "readable_full_flow_count": len(full_records),
        "absolute_stability_tolerance": POLAR_ABSOLUTE_STABILITY_TOLERANCE,
    }


def _polar_prediction_stable(current: Any, reference: Any, *, axis_span: float | None = None) -> bool:
    current_value = _number_or_none(current)
    reference_value = _number_or_none(reference)
    if current_value is None or reference_value is None:
        return False
    tolerance = _polar_stability_tolerance(axis_span)
    return abs(current_value - reference_value) < tolerance


def _polar_stability_tolerance(axis_span: float | None) -> float:
    if axis_span is None or axis_span <= 0:
        return POLAR_ABSOLUTE_STABILITY_TOLERANCE
    return min(POLAR_ABSOLUTE_STABILITY_TOLERANCE, max(axis_span * 0.04, axis_span / 40.0))


def _polar_axis_span(dataset: dict[str, Any]) -> float | None:
    values = [_number_or_none(item) for item in dataset.get("r_ticks", [])]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return max(values) - min(values)


def _polar_selection_close(current: float, reference: float, axis_span: float | None) -> bool:
    scale = axis_span if axis_span is not None and axis_span > 0 else max(abs(current), abs(reference), 1.0)
    return abs(current - reference) / scale <= 0.06


def _stable_polar_tail(records: list[dict[str, Any]], axis_span: float | None) -> float | None:
    values = [_number_or_none(record.get("pred_r")) for record in records]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    if _polar_selection_close(values[-1], values[-2], axis_span):
        return _median(values[-2:])
    return None


def _stable_polar_grid_feedback(
    grid_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    axis_span: float | None,
) -> float | None:
    if not grid_records or not feedback_records:
        return None
    grid_value = _number_or_none(grid_records[-1].get("pred_r"))
    feedback_value = _number_or_none(feedback_records[-1].get("pred_r"))
    if grid_value is None or feedback_value is None:
        return None
    if _polar_selection_close(grid_value, feedback_value, axis_span):
        return _median([grid_value, feedback_value])
    return None


def _normalized_outer_ring_amplifier_drift(
    amp_value: float,
    full_view_value: float | None,
    axis_span: float | None,
) -> bool:
    if axis_span is None or axis_span > 1.5 or full_view_value is None:
        return False
    if amp_value < 0.90:
        return False
    return amp_value - full_view_value > max(axis_span * 0.16, 0.08)


def _normalized_polar_outer_drift_reference(
    grid_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    stable_amp: float | None,
    axis_span: float | None,
) -> tuple[dict[str, Any], float, str] | None:
    if axis_span is None or axis_span > 1.5 or stable_amp is None or stable_amp < 0.90:
        return None
    if not grid_records:
        return None
    grid_value = _number_or_none(grid_records[-1].get("pred_r"))
    if grid_value is None:
        return None

    drift_tolerance = max(axis_span * 0.16, 0.08)
    if stable_amp - grid_value <= drift_tolerance:
        return None

    stable_feedback = _stable_polar_tail(feedback_records, axis_span)
    if stable_feedback is not None:
        if stable_feedback < 0.85 and abs(stable_feedback - grid_value) > drift_tolerance:
            return feedback_records[-1], stable_feedback, "stable_feedback_outer_amplifier_drift"
        if stable_feedback >= 0.85 and grid_value <= 0.70:
            return grid_records[-1], grid_value, "grid_reference_outer_feedback_amplifier_drift"

    stable_grid_feedback = _stable_polar_grid_feedback(grid_records, feedback_records, axis_span)
    if stable_grid_feedback is not None:
        return feedback_records[-1], stable_grid_feedback, "stable_grid_feedback_outer_amplifier_drift"

    if stable_amp - grid_value > max(axis_span * 0.24, 0.12):
        return grid_records[-1], grid_value, "grid_reference_outer_amplifier_drift"
    return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def _axis_span_from_records(records: list[dict[str, Any]]) -> float | None:
    for record in records:
        value = _number_or_none(record.get("r_axis_span"))
        if value is not None and value > 0:
            return value
    return None


def _split_point_name(value: str) -> tuple[str, str]:
    if "," not in value:
        return "", value.strip()
    series_name, theta_label = value.rsplit(",", 1)
    return series_name.strip(), theta_label.strip()


def _save_records(records: list[dict[str, Any]], result_dir: Path) -> None:
    if not records:
        return
    columns: list[str] = []
    for record in records:
        for key in record:
            if key not in columns:
                columns.append(key)
    with (result_dir / "experiment_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _save_run_artifacts(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    result_dir: Path,
    chart_type: str,
) -> None:
    _save_records(records, result_dir)
    _save_csv(records, result_dir / "full_results_with_rre.csv")
    _save_csv(_full_flow_final_rows(records), result_dir / "full_flow_final_predictions.csv")
    _save_csv(_full_flow_selection_audit_rows(records), result_dir / "full_flow_final_selection_audit.csv")
    _save_csv(
        predictions,
        result_dir / "selected_predictions.csv",
        columns=["id", "series_name", "theta_label", "label", "axis", "value", "r", "prompt_type", "image_type", "image_path"],
    )
    with (result_dir / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)

    summary_rows = _prediction_summary_rows(records)
    _save_csv(summary_rows, result_dir / "r_level_summary.csv")
    valid_count = sum(1 for record in records if _number_or_none(record.get("pred_r")) is not None)
    run_summary = {
        "chart_type": chart_type,
        "chart_id": records[0].get("chart_id") if records else None,
        "record_count": len(records),
        "object_count": len(predictions),
        "valid_prediction_count": valid_count,
        "prompt_image_summary": summary_rows,
        "note": "GT values are preserved for experiment metrics. Full flow uses grid, prediction-driven feedback overlays, and prediction-driven local amplifier crops.",
    }
    with (result_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, ensure_ascii=False, indent=2)
    _save_summary_plot(summary_rows, result_dir / "r_level_prediction_summary.png")


def _save_csv(rows: list[dict[str, Any]], path: Path, columns: list[str] | None = None) -> None:
    columns = list(columns or [])
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _full_flow_final_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _point, point_records in _records_by_point(records).items():
        selected, _audit = _select_polar_full_flow_record(point_records)
        if selected is not None:
            rows.append(selected)
    return rows


def _full_flow_selection_audit_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point, point_records in _records_by_point(records).items():
        _selected, audit = _select_polar_full_flow_record(point_records)
        rows.append({"point": point, **audit})
    return rows


def _records_by_point(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        point = record.get("point")
        if point in (None, "__whole_chart__"):
            continue
        by_point.setdefault(str(point), []).append(record)
    return by_point


def _prediction_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[float]] = {}
    total_counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (str(record.get("prompt_type") or ""), str(record.get("image_type") or ""))
        total_counts[key] = total_counts.get(key, 0) + 1
        value = _number_or_none(record.get("pred_r"))
        if value is not None:
            groups.setdefault(key, []).append(value)

    rows: list[dict[str, Any]] = []
    for key in sorted(total_counts):
        values = groups.get(key, [])
        rows.append(
            {
                "prompt_type": key[0],
                "image_type": key[1],
                "record_count": total_counts[key],
                "valid_r_count": len(values),
                "avg_pred_r": round(sum(values) / len(values), 6) if values else None,
                "min_pred_r": min(values) if values else None,
                "max_pred_r": max(values) if values else None,
            }
        )
    return rows


def _save_summary_plot(summary_rows: list[dict[str, Any]], path: Path) -> None:
    if not summary_rows:
        return
    labels = [f"{row['prompt_type']}\n{row['image_type']}" for row in summary_rows]
    counts = [int(row.get("valid_r_count") or 0) for row in summary_rows]
    avg_values = [
        float(row["avg_pred_r"]) if row.get("avg_pred_r") is not None else 0.0
        for row in summary_rows
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    x_values = list(range(len(summary_rows)))
    ax1.bar(x_values, counts, color="#4C78A8", alpha=0.85, label="Valid R predictions")
    ax1.set_ylabel("Valid prediction count")
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(labels, rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x_values, avg_values, color="#F58518", marker="o", linewidth=2, label="Average predicted R")
    ax2.set_ylabel("Average predicted R")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.title("Polar Prediction Summary")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def _value_to_number(value: Any) -> float | None:
    if isinstance(value, (list, tuple)) and value:
        return _number_or_none(value[0])
    if isinstance(value, dict):
        for key in ("r_value", "r", "value", "y"):
            number = _number_or_none(value.get(key))
            if number is not None:
                return number
    return _number_or_none(value)


def _number_or_none(value: Any) -> float | None:
    try:
        if isinstance(value, str):
            value = re.sub(r"[%，,]", "", value.strip())
        number = float(value)
        return number if number == number else None
    except Exception:
        return None


def _key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _tick_pixel_pairs(ticks: Any, pixels: Any) -> list[dict[str, Any]]:
    if not isinstance(ticks, list):
        return []
    if not isinstance(pixels, list):
        pixels = [pixels] if pixels not in (None, "") else []
    return [{"tick": tick, "pixel_radius": pixel} for tick, pixel in zip(ticks, pixels)]
