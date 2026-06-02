"""Runner for backend-local pie value extraction."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from aiohttp import ClientTimeout

from ...common.chart_io import ensure_dir, image_to_data_url
from ...common.paths import RESULTS_ROOT
from ...common.runtime import get_repeat_times

from ..circular_fallback import color_area_predictions, records_from_predictions
from ..circular_model_config import get_chat_completion_url, get_headers, get_model_name
from ..circular_predictions import complete_circular_predictions, system_label_order
from .data import CircularTarget, image_path, iter_targets, load_datasets
from .model import call_llm_once
from .prompts import generate_prompt


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "with_grid"),
]
PREFERRED_PROMPTS = ["grid", "baseline"]
CHART_TYPE = "pie"


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
            print(f"[pie] Round {run_idx} | Segment: {target.point_name} | {prompt_type} - {image_type}")
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
You are analyzing a pie chart with angular reference lines.
Identify every visible sector and estimate its share of the whole chart.

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


def _chart_result_dir(chart_id: str) -> Path:
    return ensure_dir(RESULTS_ROOT / CHART_TYPE / chart_id)


def _save_records(records: list[dict[str, Any]], result_dir: Path) -> None:
    ensure_dir(result_dir)
    if not records:
        return
    pd.DataFrame(records).to_csv(result_dir / "experiment_results.csv", index=False)


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    datasets = load_datasets(chart_ids, config_paths=config_paths)
    if not datasets:
        print("[pie] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times()
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        result_dir = _chart_result_dir(str(dataset["chart_id"]))
        targets = iter_targets(dataset)
        records: list[dict[str, Any]] = []
        if targets:
            predictions = await _extract_all_segments(dataset)
            predicted_labels = {
                str(item.get("label") or item.get("id") or "").strip().casefold()
                for item in predictions
            }
            missing_targets = [
                target
                for target in targets
                if str(target.point_name).strip().casefold() not in predicted_labels
            ]
            if missing_targets:
                tasks = [_run_target(dataset, target, repeat_times) for target in missing_targets]
                for result in await asyncio.gather(*tasks):
                    records.extend(result)
                predictions.extend(_select_predictions(records))
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
        _save_records(records, result_dir)
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
