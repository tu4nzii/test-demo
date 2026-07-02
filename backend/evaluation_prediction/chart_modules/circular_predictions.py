"""Prediction normalization helpers for pie/donut extraction."""

from __future__ import annotations

from typing import Any



SERIES_PLACEHOLDERS = {"series 1", "系列1"}


def complete_circular_predictions(
    dataset: dict[str, Any],
    chart_type: str,
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = system_label_order(dataset)
    if not labels:
        return predictions

    completed = list(predictions)
    existing = {_label_key(item.get("label") or item.get("id")) for item in completed}
    missing = [label for label in labels if _label_key(label) not in existing]
    for label in labels:
        if _label_key(label) in existing:
            continue
        completed.append(
            {
                "id": label,
                "series_name": "",
                "label": label,
                "axis": "theta",
                "value": None,
                "percentage": None,
                "start_angle": None,
                "end_angle": None,
                "prompt_type": "system_json_label_unestimated",
                "image_type": None,
                "image_path": None,
                "extraction_source": "system_json_label_only",
            }
        )

    return _sort_by_label_order(completed, labels)


def normalize_circular_prediction_shares(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values: list[float] = []
    for item in predictions:
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return predictions
        if value < 0 or value != value:
            return predictions
        values.append(value)
    if len(values) < 2:
        return predictions
    total = sum(values)
    if total <= 0:
        return predictions
    if not 0.85 <= total <= 1.15:
        return predictions
    normalized: list[dict[str, Any]] = []
    for item, value in zip(predictions, values):
        next_item = dict(item)
        share = value / total
        next_item["value"] = share
        next_item["percentage"] = share * 100.0
        source = str(next_item.get("extraction_source") or next_item.get("prompt_type") or "selected_prediction")
        if "closure_normalized" not in source:
            source = f"{source}+closure_normalized"
        next_item["extraction_source"] = source
        normalized.append(next_item)
    return normalized


def system_label_order(dataset: dict[str, Any]) -> list[str]:
    names: list[str] = []
    colors = dataset.get("colors")
    if isinstance(colors, list):
        for item in colors:
            if isinstance(item, dict):
                _append_name(names, item.get("name"))
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict):
        for name in series_color:
            _append_name(names, name)
    data_points = dataset.get("data_points")
    if isinstance(data_points, dict):
        for name in data_points:
            _append_name(names, name)
    return names


def _sort_by_label_order(predictions: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    order = {_label_key(label): index for index, label in enumerate(labels)}
    indexed = sorted(
        enumerate(predictions),
        key=lambda item: (order.get(_label_key(item[1].get("label") or item[1].get("id")), len(order)), item[0]),
    )
    return [item for _, item in indexed]


def _append_name(names: list[str], raw: Any) -> None:
    name = str(raw or "").strip()
    key = _label_key(name)
    if not name or key in SERIES_PLACEHOLDERS:
        return
    if key not in {_label_key(item) for item in names}:
        names.append(name)


def _label_key(value: Any) -> str:
    return str(value or "").strip().casefold()
