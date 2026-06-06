"""Geometry helpers for scatter charts."""

from __future__ import annotations

from typing import Callable

import numpy as np


def numeric_pixel(value: float, ticks: list[float], pixels: list[int | float]) -> int:
    return int(round(float(np.interp(float(value), [float(t) for t in ticks], [float(p) for p in pixels]))))


def data_to_pixel(
    coord: tuple[float, float],
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int | float],
    y_pixels: list[int | float],
) -> tuple[int, int]:
    return numeric_pixel(coord[0], x_ticks, x_pixels), numeric_pixel(coord[1], y_ticks, y_pixels)


def pixel_to_value(pixel: int | float, ticks: list[float], pixels: list[int | float]) -> float:
    pairs = sorted(zip([float(t) for t in ticks], [float(p) for p in pixels]), key=lambda item: item[1])
    pixel_values = [item[1] for item in pairs]
    tick_values = [item[0] for item in pairs]
    return float(np.interp(float(pixel), pixel_values, tick_values))


def value_range_from_pixels(
    start_px: int,
    end_px: int,
    ticks: list[float],
    pixels: list[int | float],
) -> tuple[float, float]:
    a = pixel_to_value(start_px, ticks, pixels)
    b = pixel_to_value(end_px, ticks, pixels)
    return tuple(sorted((round(a, 4), round(b, 4))))


def visible_ticks_for_range(ticks: list[float], min_value: float, max_value: float) -> list[float]:
    visible = [float(tick) for tick in ticks if min_value <= float(tick) <= max_value]
    if visible:
        return visible
    return [round(min_value, 2), round(max_value, 2)]


def build_axis_mapper(ticks: list[float], pixels: list[int | float]) -> Callable[[float], float]:
    return lambda value: float(np.interp(float(value), [float(t) for t in ticks], [float(p) for p in pixels]))


def compute_pixel_relative_error_xy(
    pred: tuple[float, float],
    gt: tuple[float, float],
    *,
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int | float],
    y_pixels: list[int | float],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    x_mapper = build_axis_mapper(x_ticks, x_pixels)
    y_mapper = build_axis_mapper(y_ticks, y_pixels)
    pred_px, pred_py = x_mapper(pred[0]), y_mapper(pred[1])
    gt_px, gt_py = x_mapper(gt[0]), y_mapper(gt[1])
    width, height = image_size
    return round(abs(pred_px - gt_px) / width, 4), round(abs(pred_py - gt_py) / height, 4)
