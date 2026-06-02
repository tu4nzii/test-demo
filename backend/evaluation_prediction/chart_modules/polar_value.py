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

from ..common.chart_io import ensure_dir, filter_chart_configs, image_to_data_url, read_json
from ..common.json_utils import parse_model_json, unwrap_openai_content
from ..common.model_config import get_chat_completion_urls, get_headers, get_model_name, use_legacy_pixtral
from ..common.paths import RESULTS_ROOT
from ..common.runtime import get_repeat_times
from .circular_model_config import (
    get_chat_completion_urls as get_circular_chat_completion_urls,
    get_headers as get_circular_headers,
    get_model_name as get_circular_model_name,
)


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


def build_prompt(dataset: dict[str, Any], target: PolarTarget, chart_type: str, prompt_type: str) -> str:
    r_ticks = dataset.get("r_ticks", [])
    theta_ticks = _theta_labels(dataset)
    theta_angles = dataset.get("theta_angles") or dataset.get("axes_angles") or []
    color_hint = f"The target color is {target.color}." if target.color else ""
    grid_hint = (
        "Use the encrypted radial grid and interpolate between the nearest radial ticks."
        if prompt_type == "grid"
        else "Estimate from the original chart without relying on ground-truth data."
    )

    if chart_type == "radar":
        return f"""
You are analyzing a radar chart.
The radial tick values are: {json.dumps(r_ticks, ensure_ascii=False)}.
The angular axis labels are: {json.dumps(theta_ticks, ensure_ascii=False)}.
Their angle positions are: {json.dumps(theta_angles, ensure_ascii=False)} degrees.
The chart entities and colors are: {json.dumps(_series_color(dataset), ensure_ascii=False)}.

Task: locate entity "{target.series_name}" on axis "{target.theta_label}" and estimate its radial data value.
{color_hint}
{grid_hint}

Return only JSON in this exact shape:
{{"datapoints":[{{"{target.point_name}":[r_value,null]}}]}}
""".strip()

    return f"""
You are analyzing a rose chart.
The radial tick values are: {json.dumps(r_ticks, ensure_ascii=False)}.
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
        session = await self._ensure_session()
        for attempt in range(1, self.max_retries + 1):
            url = self._next_url()
            print(f"[polar model] {label} -> {url}")
            try:
                async with session.post(url, headers=self.headers, json=self._payload(prompt, used_image)) as resp:
                    text = await asyncio.wait_for(resp.text(), timeout=90)
                    if resp.status != 200:
                        print(f"[polar model] HTTP {resp.status}: {text[:200]}")
                        await asyncio.sleep(2 * attempt)
                        continue
                    return unwrap_openai_content(text)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                print(f"[polar model] attempt {attempt}/{self.max_retries} failed: {exc}")
                await asyncio.sleep(2 * attempt)
        return None


def _use_circular_model_fallback() -> bool:
    raw = os.getenv("POLAR_USE_CIRCULAR_MODEL", "").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if os.getenv("POLAR_CHART_MODEL_NAME") or os.getenv("CHART_MODEL_NAME") or use_legacy_pixtral():
        return False
    return get_model_name().strip().lower() == "gpt-5.3-codex"


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
                    tasks = [
                        _run_target(client, dataset, target, chart_type, repeat_times)
                        for target in targets
                    ]
                    for result in await asyncio.gather(*tasks):
                        records.extend(result)
                predictions = select_predictions(records)
                if not predictions:
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
    for prompt_type, image_type in (("baseline", "no_grid"), ("grid", "grid_with_grid")):
        for run_idx in range(1, repeat_times + 1):
            try:
                used_image = image_path(dataset, image_type)
            except KeyError:
                if image_type != "grid_with_grid":
                    raise
                used_image = image_path(dataset, "with_grid")
            prompt = build_prompt(dataset, target, chart_type, "grid" if prompt_type == "grid" else "baseline")
            print(f"[{chart_type}] Round {run_idx} | Point: {target.point_name} | {prompt_type} - {image_type}")
            pred = await client.predict_value(prompt, used_image, target.point_name)
            records.append(
                {
                    "chart_id": dataset["chart_id"],
                    "point": target.point_name,
                    "series_name": target.series_name,
                    "theta_label": target.theta_label,
                    "prompt_type": prompt_type,
                    "image_type": image_type,
                    "run": run_idx,
                    "image_path": str(used_image),
                    "gt_r": None,
                    "pred_r": pred,
                    "mae": None,
                    "r_re": None,
                }
            )
    return records


async def _run_whole_chart(
    client: PolarModelClient,
    dataset: dict[str, Any],
    chart_type: str,
    repeat_times: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_type, image_type in (("grid", "grid_with_grid"), ("baseline", "no_grid")):
        for run_idx in range(1, repeat_times + 1):
            try:
                used_image = image_path(dataset, image_type)
            except KeyError:
                if image_type != "grid_with_grid":
                    raise
                used_image = image_path(dataset, "with_grid")
            prompt = build_whole_chart_prompt(dataset, chart_type, prompt_type)
            print(f"[{chart_type}] Round {run_idx} | Whole chart fallback | {prompt_type} - {image_type}")
            predictions = await client.predict_all_values(prompt, used_image, str(dataset["chart_id"]))
            if not predictions:
                records.append(
                    {
                        "chart_id": dataset["chart_id"],
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
    merged["image_paths"] = _image_paths(merged, path.parent, chart_type, path.stem.removesuffix("_image").removesuffix("_axes"))
    return merged


def _strip_reference_data(dataset: dict[str, Any]) -> None:
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


def _image_paths(dataset: dict[str, Any], base_dir: Path, chart_type: str, stem: str) -> dict[str, str]:
    raw = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    no_grid = raw.get("no_grid") or dataset.get("image_path")
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
    values = {"no_grid": no_grid, "with_grid": with_grid, "grid_with_grid": with_grid}
    return {key: str(_resolve_path(value, base_dir)) for key, value in values.items() if isinstance(value, str) and value}


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


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


def _choose_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    for prompt_type in ("grid", "baseline"):
        candidates = [record for record in records if record.get("prompt_type") == prompt_type]
        if candidates:
            return candidates[-1]
    return records[-1]


def _split_point_name(value: str) -> tuple[str, str]:
    if "," not in value:
        return "", value.strip()
    series_name, theta_label = value.rsplit(",", 1)
    return series_name.strip(), theta_label.strip()


def _save_records(records: list[dict[str, Any]], result_dir: Path) -> None:
    if not records:
        return
    columns = list(records[0])
    with (result_dir / "experiment_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
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
        "note": "Backend prediction artifacts are based only on system-generated JSON/images; GT-based MAE is intentionally not computed.",
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
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
