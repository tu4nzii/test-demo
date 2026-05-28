from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

from .metrics import (
    absolute_error,
    relative_error,
    round_metric,
    safe_mean,
    vector_mae,
    vector_relative_error,
)
from .normalizer import flatten_data_points, get_ground_truth, get_predictions, records_by_id


def evaluate_chart_data(data: Mapping[str, Any]) -> Dict[str, Any]:
    chart_type = str(data.get("chart_type", "unknown"))
    chart_id = str(data.get("chart_id", ""))

    ground_truth = get_ground_truth(data)
    predictions = get_predictions(data)
    gt_records = records_by_id(flatten_data_points(ground_truth))
    prediction_records = records_by_id(flatten_data_points(predictions))

    if prediction_records:
        return _evaluate_predictions(chart_id, chart_type, gt_records, prediction_records, data)

    return _evaluate_available_chart_data(chart_id, chart_type, gt_records, data)


def _evaluate_predictions(
    chart_id: str,
    chart_type: str,
    gt_records: Dict[str, Any],
    prediction_records: Dict[str, Any],
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []

    for key, gt_record in gt_records.items():
        pred_record = prediction_records.get(key)
        record: Dict[str, Any] = {
            "id": key,
            "ground_truth": gt_record.value,
            "predicted": pred_record.value if pred_record else None,
            "matched": pred_record is not None,
        }
        if pred_record is not None:
            if isinstance(gt_record.value, list) and isinstance(pred_record.value, list):
                record["mae"] = round_metric(vector_mae(pred_record.value, gt_record.value))
                record["relative_error"] = round_metric(vector_relative_error(pred_record.value, gt_record.value))
            else:
                record["mae"] = round_metric(absolute_error(pred_record.value, gt_record.value))
                record["relative_error"] = round_metric(relative_error(pred_record.value, gt_record.value))
        records.append(record)

    extra_predictions = sorted(set(prediction_records) - set(gt_records))
    mae_values = [record.get("mae") for record in records]
    relative_values = [record.get("relative_error") for record in records]
    matched_count = sum(1 for record in records if record["matched"])
    total_count = len(records)

    return {
        "success": True,
        "mode": "prediction_evaluation",
        "chart_id": chart_id,
        "chart_type": chart_type,
        "summary": {
            "total_items": total_count,
            "matched_items": matched_count,
            "missing_items": total_count - matched_count,
            "extra_predictions": len(extra_predictions),
            "coverage": round_metric(matched_count / total_count if total_count else None),
            "avg_mae": round_metric(safe_mean(mae_values)),
            "avg_relative_error": round_metric(safe_mean(relative_values)),
        },
        "records": records,
        "extra_prediction_ids": extra_predictions,
        "quality": _evaluate_structure(data, total_count),
    }


def _evaluate_available_chart_data(
    chart_id: str,
    chart_type: str,
    gt_records: Dict[str, Any],
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    total_count = len(gt_records)
    structure = _evaluate_structure(data, total_count)

    return {
        "success": True,
        "mode": "data_readiness",
        "chart_id": chart_id,
        "chart_type": chart_type,
        "summary": {
            "total_items": total_count,
            "matched_items": 0,
            "missing_items": total_count,
            "extra_predictions": 0,
            "coverage": 0 if total_count else None,
            "avg_mae": None,
            "avg_relative_error": None,
        },
        "records": [
            {
                "id": key,
                "ground_truth": record.value,
                "predicted": None,
                "matched": False,
                "mae": None,
                "relative_error": None,
            }
            for key, record in gt_records.items()
        ],
        "quality": structure,
        "note": "No prediction data was found. The result reports evaluation readiness and ground-truth coverage.",
    }


def _evaluate_structure(data: Mapping[str, Any], total_items: int) -> Dict[str, Any]:
    image_paths = data.get("image_paths") if isinstance(data.get("image_paths"), Mapping) else {}
    with_grid_path = (
        data.get("encrypted_grid_path")
        or data.get("with_grid")
        or image_paths.get("with_grid")
        or image_paths.get("grid_with_grid")
    )
    basic_grid_path = data.get("basic_grid_path") or image_paths.get("grid")

    x_ticks = data.get("x_ticks", [])
    y_ticks = data.get("y_ticks", [])
    r_ticks = data.get("r_ticks", [])
    theta_ticks = data.get("theta_ticks", [])

    return {
        "has_ground_truth": total_items > 0,
        "has_basic_grid": _path_exists(basic_grid_path),
        "has_encrypted_grid": _path_exists(with_grid_path),
        "x_ticks_count": len(x_ticks) if isinstance(x_ticks, list) else 0,
        "y_ticks_count": len(y_ticks) if isinstance(y_ticks, list) else 0,
        "r_ticks_count": len(r_ticks) if isinstance(r_ticks, list) else 0,
        "theta_ticks_count": len(theta_ticks) if isinstance(theta_ticks, list) else 0,
        "colors_count": _count_colors(data.get("colors") or data.get("series_color")),
    }


def _count_colors(colors: Any) -> int:
    if isinstance(colors, Mapping):
        return len(colors)
    if isinstance(colors, list):
        return len(colors)
    return 0


def _path_exists(path: Optional[Any]) -> bool:
    return isinstance(path, str) and bool(path) and os.path.exists(path)
