"""Image-based fallback extraction for pie/donut charts.

This fallback uses only backend-generated metadata: uploaded image path, detected
circle center/radius, and system-recognized label/color pairs. It does not read
dataset GT JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def color_area_predictions(dataset: dict[str, Any], chart_type: str) -> list[dict[str, Any]]:
    colors = _color_items(dataset)
    if not colors:
        return []

    image_path = _no_grid_image_path(dataset)
    if image_path is None or not image_path.exists():
        return []

    center = dataset.get("center")
    radius = dataset.get("r_pixels")
    if not _valid_center(center) or not isinstance(radius, (int, float)):
        return []

    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image).astype(np.int32)
    height, width = arr.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    radius = int(radius)

    angle_labels = _classify_angles(arr, colors, cx, cy, radius, chart_type)
    if not angle_labels:
        return []
    counts = [angle_labels.count(index) for index in range(len(colors))]
    total = float(sum(counts))

    predictions: list[dict[str, Any]] = []
    for index, (item, count) in enumerate(zip(colors, counts)):
        if count <= 0:
            continue
        value = float(count / total)
        start_angle, end_angle = _dominant_angle_span(angle_labels, index)
        predictions.append(
            {
                "id": item["name"],
                "series_name": "",
                "label": item["name"],
                "axis": "theta",
                "value": value,
                "percentage": value * 100.0,
                "start_angle": start_angle,
                "end_angle": end_angle,
                "prompt_type": "color_area_fallback",
                "image_type": "no_grid",
                "image_path": str(image_path),
                "extraction_source": "system_json_color_area",
                "chart_type": chart_type,
            }
        )
    return predictions


def _classify_angles(
    arr: np.ndarray,
    colors: list[dict[str, Any]],
    cx: int,
    cy: int,
    radius: int,
    chart_type: str,
) -> list[int | None]:
    palette = np.asarray([item["rgb"] for item in colors], dtype=np.int32)
    if chart_type == "donut":
        radius_factors = (0.55, 0.65, 0.75, 0.85, 0.93)
    else:
        radius_factors = (0.35, 0.5, 0.65, 0.8, 0.92)

    height, width = arr.shape[:2]
    labels: list[int | None] = []
    for angle in range(360):
        theta = np.deg2rad(angle - 90)
        cos_v = float(np.cos(theta))
        sin_v = float(np.sin(theta))
        votes: list[int] = []
        for factor in radius_factors:
            x = int(round(cx + radius * factor * cos_v))
            y = int(round(cy + radius * factor * sin_v))
            if not (0 <= x < width and 0 <= y < height):
                continue
            pixel = arr[y, x]
            if _looks_like_background(pixel):
                continue
            diffs = palette - pixel
            distances = np.sum(diffs * diffs, axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= 150**2:
                votes.append(nearest)
        labels.append(_majority(votes))
    return _fill_small_gaps(labels)


def _looks_like_background(pixel: np.ndarray) -> bool:
    channels = [int(value) for value in pixel]
    spread = max(channels) - min(channels)
    brightness = sum(channels) / 3
    if brightness > 245:
        return True
    if brightness < 25:
        return True
    return spread < 18


def _majority(values: list[int]) -> int | None:
    if not values:
        return None
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _fill_small_gaps(labels: list[int | None], max_gap: int = 3) -> list[int | None]:
    labels = list(labels)
    n = len(labels)
    for start in range(n):
        if labels[start] is not None:
            continue
        end = start
        while end < n and labels[end] is None:
            end += 1
        if end - start <= max_gap:
            left = labels[(start - 1) % n]
            right = labels[end % n] if end < n else labels[0]
            if left is not None and left == right:
                for index in range(start, end):
                    labels[index] = left
    return labels


def _dominant_angle_span(labels: list[int | None], target: int) -> tuple[float | None, float | None]:
    if target not in labels:
        return None, None
    doubled = labels + labels
    best_start = 0
    best_len = 0
    index = 0
    while index < len(doubled):
        if doubled[index] != target:
            index += 1
            continue
        start = index
        while index < len(doubled) and doubled[index] == target:
            index += 1
        length = min(index - start, 360)
        if length > best_len and start < 360:
            best_start = start
            best_len = length
    if best_len <= 0:
        return None, None
    return float(best_start % 360), float((best_start + best_len) % 360)


def records_from_predictions(dataset: dict[str, Any], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chart_id": dataset.get("chart_id"),
            "point": item.get("label"),
            "prompt_type": item.get("prompt_type"),
            "image_type": item.get("image_type"),
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


def _color_items(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    raw = dataset.get("colors")
    if not isinstance(raw, list):
        return []

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        rgb = _parse_hex_color(item.get("color"))
        if not name or rgb is None:
            continue
        lowered = name.lower()
        if lowered in seen or lowered in {"series 1", "系列1", "绯诲垪1"}:
            continue
        result.append({"name": name, "rgb": rgb})
        seen.add(lowered)
    return result


def _parse_hex_color(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None


def _no_grid_image_path(dataset: dict[str, Any]) -> Path | None:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    value = image_paths.get("no_grid") or dataset.get("image_path")
    return Path(value).resolve() if isinstance(value, str) and value else None


def _valid_center(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )
