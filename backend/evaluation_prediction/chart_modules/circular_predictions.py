"""Prediction normalization helpers for pie/donut extraction."""

from __future__ import annotations

from typing import Any

from .circular_fallback import color_area_predictions


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
    if missing:
        fallback_by_label = {
            _label_key(item.get("label") or item.get("id")): item
            for item in color_area_predictions(dataset, chart_type)
        }
        for label in list(missing):
            fallback = fallback_by_label.get(_label_key(label))
            if fallback is not None:
                completed.append(fallback)
                existing.add(_label_key(label))

    missing = [label for label in labels if _label_key(label) not in existing]
    inferred = _infer_single_missing(labels, completed, missing)
    if inferred is not None:
        completed.append(inferred)
        existing.add(_label_key(inferred["label"]))

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


def system_label_order(dataset: dict[str, Any]) -> list[str]:
    names: list[str] = []
    colors = dataset.get("colors")
    if isinstance(colors, list):
        for item in colors:
            if isinstance(item, dict):
                _append_name(names, item.get("name"))
    return names


def _infer_single_missing(
    labels: list[str],
    predictions: list[dict[str, Any]],
    missing: list[str],
) -> dict[str, Any] | None:
    if len(missing) != 1:
        return None
    label_keys = {_label_key(label) for label in labels}
    total = 0.0
    for item in predictions:
        if _label_key(item.get("label") or item.get("id")) not in label_keys:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return None
        if value != value:
            return None
        total += value
    if not (0.0 <= total <= 1.0):
        return None
    value = max(0.0, 1.0 - total)
    return {
        "id": missing[0],
        "series_name": "",
        "label": missing[0],
        "axis": "theta",
        "value": value,
        "percentage": value * 100.0,
        "start_angle": None,
        "end_angle": None,
        "prompt_type": "single_missing_remainder",
        "image_type": None,
        "image_path": None,
        "extraction_source": "sum_to_one_inference",
    }


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
