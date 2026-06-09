"""Runner for backend-local donut value extraction."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from aiohttp import ClientTimeout
from PIL import Image

from ...common.chart_io import ensure_dir, image_to_data_url
from ...common.paths import RESULTS_ROOT
from ...common.runtime import get_repeat_times

from ..circular_artifacts import save_circular_artifacts
from ..circular_fallback import color_area_predictions, records_from_predictions
from ..circular_model_config import get_chat_completion_url, get_headers, get_model_name
from ..circular_predictions import complete_circular_predictions, system_label_order
from .data import CircularTarget, image_path, iter_targets, load_datasets
from .model import call_llm_once
from .prompts import generate_prompt
from . import visual as donut_visual
from .visual import crop_sector_for_amplifier, draw_angle_feedback


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "with_grid"),
]
PREFERRED_PROMPTS = ["amplifier_pct", "amplifier", "feedback", "grid", "whole_chart", "baseline"]
CHART_TYPE = "donut"


async def _run_target(dataset: dict[str, Any], target: CircularTarget, repeat_times: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_type, image_type in EXPERIMENT_TYPES:
        for run_idx in range(1, repeat_times + 1):
            used_image = image_path(dataset, image_type)
            prompt = generate_prompt(
                target.point_name,
                prompt_type,
                dataset.get("theta_ticks") if prompt_type == "baseline" else None,
            )
            print(f"[donut] Round {run_idx} | Segment: {target.point_name} | {prompt_type} - {image_type}")
            raw_pred = await call_llm_once(prompt=prompt, image_path=str(used_image), item_name=target.point_name)
            prediction = _prediction_from_raw(raw_pred)
            records.append(
                {
                    "chart_id": dataset["chart_id"],
                    "point": target.point_name,
                    "prompt_type": prompt_type,
                    "image_type": image_type,
                    "run": run_idx,
                    "image_path": str(used_image),
                    "pred": prediction.get("value"),
                    "pred_pct": prediction.get("value"),
                    "percentage": prediction.get("percentage"),
                    "start_angle": prediction.get("start_angle"),
                    "end_angle": prediction.get("end_angle"),
                    "raw_prediction": json.dumps(raw_pred, ensure_ascii=False),
                }
            )
    return records


async def _run_refined_target(
    dataset: dict[str, Any],
    target: CircularTarget,
    result_dir: Path,
    initial_prediction: dict[str, Any] | None,
    repeat_times: int,
) -> list[dict[str, Any]]:
    """Run the reference-style donut chain for one system-generated label."""

    records: list[dict[str, Any]] = []
    chart_id = str(dataset["chart_id"])
    if initial_prediction is not None:
        records.append(
            _record_from_prediction(
                dataset,
                target.point_name,
                "whole_chart",
                "with_grid",
                1,
                initial_prediction.get("image_path"),
                initial_prediction,
                initial_prediction,
            )
        )

    try:
        with_grid_image = image_path(dataset, "with_grid")
        no_grid_image = image_path(dataset, "no_grid")
    except Exception as exc:
        print(f"[donut] Missing image path for {target.point_name}: {exc}")
        return records

    center, inner_radius, outer_radius = _donut_geometry(dataset, no_grid_image)
    last_pred = None
    feedback_image = str(with_grid_image)
    feedback_rounds = max(1, repeat_times)

    for round_idx in range(1, feedback_rounds + 1):
        prompt_type = "grid" if last_pred is None else "feedback"
        used_image = str(with_grid_image) if prompt_type == "grid" else feedback_image
        prompt = generate_prompt(
            item_name=target.point_name,
            prompt_type=prompt_type,
            prev_angle=last_pred,
        )
        print(f"[donut] {chart_id} | {target.point_name} | {prompt_type} round {round_idx}")
        raw_pred = await call_llm_once(prompt=prompt, image_path=used_image, item_name=target.point_name)
        prediction = _prediction_from_raw(raw_pred)
        records.append(
            _record_from_prediction(
                dataset,
                target.point_name,
                prompt_type,
                "with_grid" if prompt_type == "grid" else "feedback",
                round_idx,
                used_image,
                prediction,
                raw_pred,
            )
        )

        if _has_angles(prediction):
            last_pred = {
                "start_angle": float(prediction["start_angle"]),
                "end_angle": float(prediction["end_angle"]),
            }
        elif last_pred is None:
            last_pred = _angles_from_prediction(initial_prediction)

        if not _has_angles(last_pred):
            break

        if round_idx < feedback_rounds:
            try:
                feedback_image = draw_angle_feedback(
                    image_path=str(with_grid_image),
                    angle_deg=[float(last_pred["start_angle"]), float(last_pred["end_angle"])],
                    output_path=str(result_dir / f"{_safe_name(target.point_name)}_feedback_round{round_idx}.png"),
                    circle_center=center,
                    inner_radius=outer_radius,
                )
            except Exception as exc:
                print(f"[donut] Feedback image failed for {target.point_name}: {exc}")
                break

    if not _has_angles(last_pred):
        last_pred = _angles_from_prediction(initial_prediction)
    if not _has_angles(last_pred):
        return records

    previous_output_root = donut_visual.AMPLIFIER_OUTPUT_ROOT
    donut_visual.AMPLIFIER_OUTPUT_ROOT = str(result_dir.parent)
    amp_pred = dict(last_pred)
    last_amp_image: str | None = None
    try:
        for amp_round in range(1, 4):
            try:
                amp_image, drawn_angles, angle_order_hint = crop_sector_for_amplifier(
                    image_path=str(no_grid_image),
                    centre=center,
                    inner_r=inner_radius,
                    outer_r=outer_radius,
                    feedback_angles=amp_pred,
                    chart_id=chart_id,
                    point_name=target.point_name,
                    save_suffix=f"_amp{amp_round}",
                    amp_round=amp_round,
                )
            except Exception as exc:
                print(f"[donut] Amplifier crop failed for {target.point_name}, round {amp_round}: {exc}")
                break

            last_amp_image = amp_image
            prompt = generate_prompt(
                item_name=target.point_name,
                prompt_type="amplifier",
                prev_angle=amp_pred,
                drawn_angles=drawn_angles,
                angle_order_hint=angle_order_hint,
            )
            print(f"[donut] {chart_id} | {target.point_name} | amplifier round {amp_round}")
            raw_pred = await call_llm_once(prompt=prompt, image_path=amp_image, item_name=target.point_name)
            prediction = _prediction_from_raw(raw_pred)
            records.append(
                _record_from_prediction(
                    dataset,
                    target.point_name,
                    "amplifier",
                    "no_grid",
                    amp_round,
                    amp_image,
                    prediction,
                    raw_pred,
                )
            )
            if not _has_angles(prediction):
                break
            amp_pred = {
                "start_angle": float(prediction["start_angle"]),
                "end_angle": float(prediction["end_angle"]),
            }
    finally:
        donut_visual.AMPLIFIER_OUTPUT_ROOT = previous_output_root

    if _has_angles(amp_pred):
        final_prediction = _prediction_from_raw(amp_pred)
        records.append(
            _record_from_prediction(
                dataset,
                target.point_name,
                "amplifier_pct",
                "no_grid",
                1,
                last_amp_image or str(no_grid_image),
                final_prediction,
                amp_pred,
            )
        )

    return records


def _prediction_from_raw(raw_pred: Any) -> dict[str, float | None]:
    if isinstance(raw_pred, dict) and "start_angle" in raw_pred and "end_angle" in raw_pred:
        try:
            start = float(raw_pred["start_angle"])
            end = float(raw_pred["end_angle"])
            value = ((end - start + 360.0) % 360.0) / 360.0
            return {
                "value": value,
                "percentage": value * 100.0,
                "start_angle": start,
                "end_angle": end,
            }
        except Exception:
            return {"value": None, "percentage": None, "start_angle": None, "end_angle": None}
    try:
        number = float(raw_pred)
    except Exception:
        return {"value": None, "percentage": None, "start_angle": None, "end_angle": None}
    value = number / 100.0 if abs(number) > 1.0 else number
    return {"value": value, "percentage": value * 100.0, "start_angle": None, "end_angle": None}


def _record_from_prediction(
    dataset: dict[str, Any],
    point: str,
    prompt_type: str,
    image_type: str,
    run: int,
    used_image: Any,
    prediction: dict[str, Any],
    raw_pred: Any,
) -> dict[str, Any]:
    value = prediction.get("value") if isinstance(prediction, dict) else None
    return {
        "chart_id": dataset["chart_id"],
        "point": point,
        "prompt_type": prompt_type,
        "image_type": image_type,
        "run": run,
        "image_path": str(used_image) if used_image else None,
        "pred": value,
        "pred_pct": value,
        "percentage": prediction.get("percentage") if isinstance(prediction, dict) else None,
        "start_angle": prediction.get("start_angle") if isinstance(prediction, dict) else None,
        "end_angle": prediction.get("end_angle") if isinstance(prediction, dict) else None,
        "raw_prediction": json.dumps(raw_pred, ensure_ascii=False),
    }


def _has_angles(prediction: Any) -> bool:
    if not isinstance(prediction, dict):
        return False
    return _number_or_none(prediction.get("start_angle")) is not None and _number_or_none(prediction.get("end_angle")) is not None


def _angles_from_prediction(prediction: dict[str, Any] | None) -> dict[str, float] | None:
    if not _has_angles(prediction):
        return None
    return {
        "start_angle": float(prediction["start_angle"]),
        "end_angle": float(prediction["end_angle"]),
    }


def _prediction_value(record: dict[str, Any]) -> float | None:
    try:
        value = float(record["pred_pct"])
        return value if value == value else None
    except Exception:
        return None


def _select_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if _prediction_value(record) is not None:
            by_point.setdefault(str(record["point"]), []).append(record)

    predictions: list[dict[str, Any]] = []
    for point, point_records in by_point.items():
        chosen = None
        for prompt_type in PREFERRED_PROMPTS:
            candidates = [record for record in point_records if record["prompt_type"] == prompt_type]
            if candidates:
                chosen = candidates[-1]
                break
        if chosen is None:
            chosen = point_records[-1]
        value = _prediction_value(chosen)
        if value is None:
            continue
        predictions.append(
            {
                "id": point,
                "series_name": "",
                "label": point,
                "axis": "theta",
                "value": value,
                "percentage": chosen.get("percentage"),
                "start_angle": chosen.get("start_angle"),
                "end_angle": chosen.get("end_angle"),
                "prompt_type": chosen.get("prompt_type"),
                "image_type": chosen.get("image_type"),
                "image_path": chosen.get("image_path"),
            }
        )
    return predictions


async def _extract_all_segments(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    used_image = image_path(dataset, "with_grid")
    labels = system_label_order(dataset)
    label_text = ", ".join(json.dumps(label, ensure_ascii=False) for label in labels)
    label_rule = (
        f"Use exactly these labels and return one datapoint for each of them, in this order: [{label_text}]."
        if labels
        else "Use visible legend or slice labels when available."
    )
    prompt = """
You are analyzing a donut chart with angular reference lines.
Identify every visible ring sector and estimate its share of the whole chart.

Return only JSON in this exact shape:
{{"datapoints":[{{"name":"A","percentage":20,"start_angle":0,"end_angle":72}}]}}

Rules:
1. {label_rule}
2. If labels are not visible, name sectors Segment 1, Segment 2, ... in clockwise order.
3. percentage is a number from 0 to 100.
4. Angles are optional; include them only when you can estimate them from the grid.
""".format(label_rule=label_rule)
    payload = {
        "model": get_model_name(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(used_image)}},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    timeout = ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(get_chat_completion_url(), headers=get_headers(), json=payload) as response:
            data = await response.json()
    text = _response_text(data)
    parsed = _parse_json_object(text)
    items = parsed.get("datapoints") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []
    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("name") or item.get("label") or f"Segment {index}").strip()
        value = _percentage_to_value(item.get("percentage") if "percentage" in item else item.get(label))
        if value is None:
            continue
        predictions.append(
            {
                "id": label,
                "series_name": "",
                "label": label,
                "axis": "theta",
                "value": value,
                "percentage": value * 100.0,
                "start_angle": _number_or_none(item.get("start_angle")),
                "end_angle": _number_or_none(item.get("end_angle")),
                "prompt_type": "whole_chart",
                "image_type": "with_grid",
                "image_path": str(used_image),
            }
        )
    return predictions


def _response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("choices"), list) and data["choices"]:
        return str(data["choices"][0].get("message", {}).get("content", ""))
    return str(data.get("response", ""))


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```json", "", text.strip())
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _percentage_to_value(raw: Any) -> float | None:
    number = _number_or_none(raw)
    if number is None:
        return None
    return number / 100.0 if abs(number) > 1.0 else number


def _number_or_none(raw: Any) -> float | None:
    try:
        value = float(str(raw).strip().rstrip("%"))
        return value if value == value else None
    except Exception:
        return None


def _donut_geometry(dataset: dict[str, Any], source_image: Path) -> tuple[tuple[int, int], int, int]:
    with Image.open(source_image) as img:
        width, height = img.size

    raw_center = dataset.get("center")
    if isinstance(raw_center, dict):
        center = (int(raw_center.get("x", width // 2)), int(raw_center.get("y", height // 2)))
    elif isinstance(raw_center, (list, tuple)) and len(raw_center) >= 2:
        center = (int(raw_center[0]), int(raw_center[1]))
    else:
        center = (width // 2, height // 2)

    raw_radius = dataset.get("r_pixels") or dataset.get("radius")
    if isinstance(raw_radius, (list, tuple)):
        radius_values = [_number_or_none(value) for value in raw_radius]
        valid_radii = [int(value) for value in radius_values if value is not None and value > 0]
        outer_radius = max(valid_radii) if valid_radii else int(min(width, height) * 0.35)
        inner_radius = min(valid_radii) if len(valid_radii) >= 2 else int(outer_radius * 0.45)
    else:
        outer_radius = int(_number_or_none(raw_radius) or min(width, height) * 0.35)
        inner_radius = int(_number_or_none(dataset.get("inner_radius")) or _number_or_none(dataset.get("hole_radius")) or outer_radius * 0.45)

    inner_radius = max(1, min(inner_radius, outer_radius - 1))
    outer_radius = max(inner_radius + 1, outer_radius)
    return center, inner_radius, outer_radius


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "segment"


def _predictions_by_label(predictions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in predictions:
        key = str(item.get("label") or item.get("id") or "").strip().casefold()
        if key:
            result[key] = item
    return result


def _chart_result_dir(chart_id: str) -> Path:
    return ensure_dir(RESULTS_ROOT / CHART_TYPE / chart_id)


def _save_records(records: list[dict[str, Any]], result_dir: Path) -> None:
    ensure_dir(result_dir)
    if not records:
        return
    pd.DataFrame(records).to_csv(result_dir / "experiment_results.csv", index=False, encoding="utf-8")


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    datasets = load_datasets(chart_ids, config_paths=config_paths)
    if not datasets:
        print("[donut] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times()
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        result_dir = _chart_result_dir(str(dataset["chart_id"]))
        targets = iter_targets(dataset)
        records: list[dict[str, Any]] = []
        if targets:
            initial_predictions = await _extract_all_segments(dataset)
            initial_by_label = _predictions_by_label(initial_predictions)
            for target in targets:
                refined_records = await _run_refined_target(
                    dataset=dataset,
                    target=target,
                    result_dir=result_dir,
                    initial_prediction=initial_by_label.get(target.point_name.strip().casefold()),
                    repeat_times=repeat_times,
                )
                records.extend(refined_records)

            predictions = _select_predictions(records)
            if not predictions:
                predictions = initial_predictions
            if not predictions:
                predictions = color_area_predictions(dataset, CHART_TYPE)
                records.extend(records_from_predictions(dataset, predictions))
            predictions = complete_circular_predictions(dataset, CHART_TYPE, predictions)
        else:
            predictions = await _extract_all_segments(dataset)
            if not predictions:
                predictions = color_area_predictions(dataset, CHART_TYPE)
            predictions = complete_circular_predictions(dataset, CHART_TYPE, predictions)
            records = [
                {
                    "chart_id": dataset["chart_id"],
                    "point": item["label"],
                    "prompt_type": "whole_chart",
                    "image_type": "with_grid",
                    "run": 1,
                    "image_path": item.get("image_path"),
                    "pred": item.get("value"),
                    "pred_pct": item.get("value"),
                    "percentage": item.get("percentage"),
                    "start_angle": item.get("start_angle"),
                    "end_angle": item.get("end_angle"),
                    "raw_prediction": "",
                }
                for item in predictions
            ]
        records = save_circular_artifacts(
            dataset=dataset,
            chart_type=CHART_TYPE,
            result_dir=result_dir,
            records=records,
            predictions=predictions,
        )
        summaries.append(
            {
                "chart_id": dataset["chart_id"],
                "result_dir": str(result_dir),
                "record_count": len(records),
                "object_count": len(predictions),
                "predictions": predictions,
            }
        )
    return summaries
