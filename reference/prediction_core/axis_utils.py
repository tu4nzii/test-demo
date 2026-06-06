"""Shared axis and label helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def build_axis_mapping(tick_values: Iterable[float], tick_pixels: Iterable[int]):
    return lambda value: np.interp(value, tick_values, tick_pixels)


def normalize_label(label: Any) -> str:
    text = str(label).strip()
    if "," in text:
        return text.split(",")[-1].strip()
    return text


def match_label(label: Any, tick_labels: list[Any]) -> Any:
    normalized = normalize_label(label)
    if normalized in tick_labels:
        return normalized

    tick_text = [str(item) for item in tick_labels]
    if normalized in tick_text:
        return tick_labels[tick_text.index(normalized)]

    matches = [
        original
        for original, text in zip(tick_labels, tick_text)
        if normalized in text or text in normalized
    ]
    if matches:
        return matches[0]
    raise ValueError(f"Label {label!r} not found in tick labels.")


def get_category_span(
    label: Any,
    tick_labels: list[Any],
    tick_pixels: list[int],
    img_min: int,
    img_max: int,
    mode: str = "center",
) -> tuple[int, int]:
    matched_label = match_label(label, tick_labels)
    idx = tick_labels.index(matched_label)

    if mode == "center":
        center = tick_pixels[idx]
        start = (tick_pixels[idx - 1] + center) // 2 if idx > 0 else img_min
        end = (center + tick_pixels[idx + 1]) // 2 if idx < len(tick_pixels) - 1 else img_max
    elif mode == "left":
        start = tick_pixels[idx]
        end = tick_pixels[idx + 1] if idx < len(tick_pixels) - 1 else img_max
    else:
        raise ValueError(f"Unsupported mode {mode!r}; use 'center' or 'left'.")
    return tuple(sorted((start, end)))


def marker_span_for_tick(tick_pixels: list[int], nearest_idx: int, image_size: tuple[int, int]) -> int:
    """Estimate a stable marker span even when only one categorical tick exists."""
    if len(tick_pixels) < 2:
        return max(12, min(image_size) // 40)
    if nearest_idx == 0:
        return abs(tick_pixels[1] - tick_pixels[0])
    if nearest_idx == len(tick_pixels) - 1:
        return abs(tick_pixels[-1] - tick_pixels[-2])
    return (
        abs(tick_pixels[nearest_idx] - tick_pixels[nearest_idx - 1])
        + abs(tick_pixels[nearest_idx] - tick_pixels[nearest_idx + 1])
    ) // 2
