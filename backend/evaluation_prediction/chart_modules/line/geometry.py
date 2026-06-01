"""Geometry helpers for line charts."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...common.axis_utils import marker_span_for_tick


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
    x_pixel = category_pixel(label, tick_labels, tick_pixels)
    nearest_idx = int(np.argmin([abs(p - x_pixel) for p in tick_pixels]))
    return marker_span_for_tick(tick_pixels, nearest_idx, image_size)


def numeric_pixel(value: float, y_ticks: list[float], y_pixels: list[int]) -> int:
    return int(round(np.interp(float(value), [float(v) for v in y_ticks], y_pixels)))


def value_range_from_pixels(top_px: int, bottom_px: int, y_ticks: list[float], y_pixels: list[int]) -> tuple[float, float]:
    pairs = sorted(zip([float(y) for y in y_ticks], y_pixels), key=lambda item: item[1])
    p_top, p_bottom = pairs[0][1], pairs[-1][1]
    v_top, v_bottom = pairs[0][0], pairs[-1][0]
    if p_bottom == p_top:
        return tuple(sorted((v_top, v_bottom)))

    def inv(px: int) -> float:
        return v_top + (px - p_top) * (v_bottom - v_top) / (p_bottom - p_top)

    a, b = inv(top_px), inv(bottom_px)
    return tuple(sorted((round(a, 4), round(b, 4))))


def visible_ticks_for_range(y_ticks: list[float], min_value: float, max_value: float) -> list[float]:
    ticks = [float(tick) for tick in y_ticks if min_value <= float(tick) <= max_value]
    if ticks:
        return ticks
    return [round(min_value, 2), round(max_value, 2)]
