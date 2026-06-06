"""Image geometry helpers for stacked bar prediction flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageColor

from .backend_dataset import resolve_image_path
from .axis_utils import marker_span_for_tick


@dataclass(frozen=True)
class StackedSegmentPrior:
    start_value: float
    end_value: float
    center_value: float
    segment_value: float
    start_pixel: int
    end_pixel: int
    center_pixel: int


def is_stacked_bar_type(chart_type: Any) -> bool:
    return str(chart_type or "").lower() in {"h_stacked_bar", "v_stacked_bar"}


def stacked_segment_prior(
    dataset: dict[str, Any],
    *,
    series_name: str,
    category_label: str,
    orientation: str,
) -> StackedSegmentPrior | None:
    if not is_stacked_bar_type(dataset.get("chart_type")):
        return None

    color = _series_color(dataset, series_name)
    if color is None:
        return None

    image = _open_source_image(dataset)
    if image is None:
        return None
    arr = np.asarray(image.convert("RGB"))

    if orientation == "h":
        return _horizontal_prior(arr, dataset, category_label, color)
    if orientation == "v":
        return _vertical_prior(arr, dataset, category_label, color)
    return None


def _open_source_image(dataset: dict[str, Any]) -> Image.Image | None:
    for image_type in ("no_grid", "grid_with_grid", "with_grid"):
        try:
            path = resolve_image_path(dataset, image_type)
        except Exception:
            continue
        if path.exists():
            return Image.open(path)
    return None


def _series_color(dataset: dict[str, Any], series_name: str) -> tuple[int, int, int] | None:
    series_color = dataset.get("series_color")
    value = series_color.get(series_name) if isinstance(series_color, dict) else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return tuple(ImageColor.getrgb(value.strip())[:3])
    except Exception:
        return None


def _horizontal_prior(
    arr: np.ndarray,
    dataset: dict[str, Any],
    category_label: str,
    rgb: tuple[int, int, int],
) -> StackedSegmentPrior | None:
    y_ticks = dataset.get("y_ticks")
    y_pixels = dataset.get("y_pixels")
    x_ticks = dataset.get("x_ticks")
    x_pixels = dataset.get("x_pixels")
    if not all(isinstance(value, list) and value for value in (y_ticks, y_pixels, x_ticks, x_pixels)):
        return None

    center_y = _category_pixel(category_label, y_ticks, y_pixels)
    if center_y is None:
        return None

    height, width = arr.shape[:2]
    span = marker_span_for_tick([int(pixel) for pixel in y_pixels], y_pixels.index(center_y), (width, height))
    half_band = max(4, min(18, span // 2 - 2))
    y0 = max(0, center_y - half_band)
    y1 = min(height, center_y + half_band + 1)
    x_min = max(0, min(int(pixel) for pixel in x_pixels) - 2)
    x_max = min(width - 1, max(int(pixel) for pixel in x_pixels) + 2)
    if y1 <= y0 or x_max <= x_min:
        return None

    band = arr[y0:y1, x_min : x_max + 1, :]
    mask = _color_mask(band, rgb)
    counts = mask.sum(axis=0)
    threshold = max(2, int(mask.shape[0] * 0.22))
    run = _best_run(np.flatnonzero(counts >= threshold))
    if run is None:
        return None

    left_px = x_min + run[0]
    right_px = x_min + run[1]
    start_value = _value_from_pixel(left_px, x_ticks, x_pixels)
    end_value = _value_from_pixel(right_px, x_ticks, x_pixels)
    if start_value is None or end_value is None:
        return None
    start_value, end_value = sorted((start_value, end_value))
    center_value = (start_value + end_value) / 2
    return StackedSegmentPrior(
        start_value=start_value,
        end_value=end_value,
        center_value=center_value,
        segment_value=end_value - start_value,
        start_pixel=int(left_px),
        end_pixel=int(right_px),
        center_pixel=int(round((left_px + right_px) / 2)),
    )


def _vertical_prior(
    arr: np.ndarray,
    dataset: dict[str, Any],
    category_label: str,
    rgb: tuple[int, int, int],
) -> StackedSegmentPrior | None:
    x_ticks = dataset.get("x_ticks")
    x_pixels = dataset.get("x_pixels")
    y_ticks = dataset.get("y_ticks")
    y_pixels = dataset.get("y_pixels")
    if not all(isinstance(value, list) and value for value in (x_ticks, x_pixels, y_ticks, y_pixels)):
        return None

    center_x = _category_pixel(category_label, x_ticks, x_pixels)
    if center_x is None:
        return None

    height, width = arr.shape[:2]
    span = marker_span_for_tick([int(pixel) for pixel in x_pixels], x_pixels.index(center_x), (width, height))
    half_band = max(4, min(18, span // 2 - 2))
    x0 = max(0, center_x - half_band)
    x1 = min(width, center_x + half_band + 1)
    y_min = max(0, min(int(pixel) for pixel in y_pixels) - 2)
    y_max = min(height - 1, max(int(pixel) for pixel in y_pixels) + 2)
    if x1 <= x0 or y_max <= y_min:
        return None

    band = arr[y_min : y_max + 1, x0:x1, :]
    mask = _color_mask(band, rgb)
    counts = mask.sum(axis=1)
    threshold = max(2, int(mask.shape[1] * 0.22))
    run = _best_run(np.flatnonzero(counts >= threshold))
    if run is None:
        return None

    top_px = y_min + run[0]
    bottom_px = y_min + run[1]
    value_a = _value_from_pixel(top_px, y_ticks, y_pixels)
    value_b = _value_from_pixel(bottom_px, y_ticks, y_pixels)
    if value_a is None or value_b is None:
        return None
    start_value, end_value = sorted((value_a, value_b))
    center_value = (start_value + end_value) / 2
    return StackedSegmentPrior(
        start_value=start_value,
        end_value=end_value,
        center_value=center_value,
        segment_value=end_value - start_value,
        start_pixel=int(bottom_px),
        end_pixel=int(top_px),
        center_pixel=int(round((top_px + bottom_px) / 2)),
    )


def _color_mask(arr: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    target = np.array(rgb, dtype=np.int32)
    diff = arr.astype(np.int32) - target
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist <= 38


def _best_run(indices: np.ndarray) -> tuple[int, int] | None:
    if len(indices) == 0:
        return None
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current == previous + 1:
            previous = current
            continue
        runs.append((start, previous))
        start = previous = current
    runs.append((start, previous))
    runs = [run for run in runs if run[1] - run[0] + 1 >= 4]
    if not runs:
        return None
    return max(runs, key=lambda run: run[1] - run[0])


def _category_pixel(label: Any, ticks: list[Any], pixels: list[Any]) -> int | None:
    text = str(label).strip()
    tick_text = [str(item).strip() for item in ticks]
    if text in tick_text:
        try:
            return int(pixels[tick_text.index(text)])
        except Exception:
            return None
    for index, tick in enumerate(tick_text):
        if text in tick or tick in text:
            try:
                return int(pixels[index])
            except Exception:
                return None
    return None


def _value_from_pixel(pixel: int, ticks: list[Any], pixels: list[Any]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for tick, tick_pixel in zip(ticks, pixels):
        try:
            pairs.append((float(tick), float(tick_pixel)))
        except Exception:
            continue
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda item: item[1])
    px = [item[1] for item in pairs]
    values = [item[0] for item in pairs]
    return float(np.interp(float(pixel), px, values))
