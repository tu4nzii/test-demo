"""Geometry helpers for horizontal bar charts."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...common.axis_utils import marker_span_for_tick


def build_numeric_mapper(tick_values: list[float], tick_pixels: list[int]):
    return lambda value: float(np.interp(float(value), tick_values, tick_pixels))


def normalize_label(label: Any) -> str:
    text = str(label).strip()
    return text.split(",")[-1].strip() if "," in text else text


def match_category(label: Any, tick_labels: list[Any]) -> Any:
    normalized = normalize_label(label)
    if normalized in tick_labels:
        return normalized

    tick_text = [str(item) for item in tick_labels]
    if normalized in tick_text:
        return tick_labels[tick_text.index(normalized)]

    for original, text in zip(tick_labels, tick_text):
        if normalized in text or text in normalized:
            return original
    raise ValueError(f"Label {label!r} not found in tick labels.")


def category_pixel(label: Any, tick_labels: list[Any], tick_pixels: list[int]) -> int:
    matched = match_category(label, tick_labels)
    return int(tick_pixels[tick_labels.index(matched)])


def category_span(label: Any, tick_labels: list[Any], tick_pixels: list[int], image_size: tuple[int, int]) -> int:
    y_pixel = category_pixel(label, tick_labels, tick_pixels)
    nearest_idx = int(np.argmin([abs(p - y_pixel) for p in tick_pixels]))
    return marker_span_for_tick(tick_pixels, nearest_idx, image_size)


def numeric_pixel(value: float, x_ticks: list[float], x_pixels: list[int]) -> int:
    return int(round(build_numeric_mapper(x_ticks, x_pixels)(value)))


def visible_ticks_for_range(x_ticks: list[float], min_value: float, max_value: float) -> list[float]:
    ticks = [float(tick) for tick in x_ticks if min_value <= float(tick) <= max_value]
    if ticks:
        return ticks
    return [round(min_value, 2), round(max_value, 2)]


def value_range_from_pixels(
    left_px: int,
    right_px: int,
    x_ticks: list[float],
    x_pixels: list[int],
) -> tuple[float, float]:
    pairs = sorted(zip([float(x) for x in x_ticks], x_pixels), key=lambda item: item[1])
    p_min, p_max = pairs[0][1], pairs[-1][1]
    v_min, v_max = pairs[0][0], pairs[-1][0]
    if p_max == p_min:
        return v_min, v_max

    def inv(px: int) -> float:
        return v_min + (px - p_min) * (v_max - v_min) / (p_max - p_min)

    a, b = inv(left_px), inv(right_px)
    return tuple(sorted((round(a, 4), round(b, 4))))
