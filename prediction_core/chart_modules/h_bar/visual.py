"""Visual feedback and crop generation for horizontal bar charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from prediction_core.chart_io import ensure_dir, safe_filename
from prediction_core.specs import PROJECT_ROOT

from .geometry import category_pixel, category_span, numeric_pixel, value_range_from_pixels, visible_ticks_for_range


RESULT_ROOT = PROJECT_ROOT / "prediction_core" / "assets" / "h_bar" / "results_Pixtral"


def chart_result_dir(chart_id: str) -> Path:
    return ensure_dir(RESULT_ROOT / chart_id)


def temp_dir(chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(chart_id) / "tempy")


def draw_prediction_overlay(
    *,
    chart_id: str,
    original_img_path: Path,
    pred_coords: list[tuple[Any, Any]],
    x_ticks: list[float],
    y_ticks: list[Any],
    x_pixels: list[int],
    y_pixels: list[int],
    point_name: str,
    draw_all_preds: bool = False,
) -> Path:
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]
    colors = ["red", "purple", "orange", "green", "blue", "black", "brown", "pink"]

    for idx, coord in enumerate(coords_to_draw):
        try:
            x_pixel = numeric_pixel(float(coord[0]), x_ticks, x_pixels)
            y_label = str(coord[1]).split(",")[-1].strip()
            y_pixel = category_pixel(y_label, y_ticks, y_pixels)
        except Exception as exc:
            print(f"[h_bar visual] Skip overlay coord {coord}: {exc}")
            continue

        half_span = category_span(coord[1], y_ticks, y_pixels, img.size) // 2
        color = colors[idx % len(colors)]
        draw.line((x_pixel - half_span, y_pixel, x_pixel + half_span, y_pixel), fill=color, width=2)
        draw.line((x_pixel, y_pixel - half_span, x_pixel, y_pixel + half_span), fill=color, width=2)

    output = temp_dir(chart_id) / f"final_overlay_{safe_filename(chart_id)}_{safe_filename(point_name)}.png"
    img.save(output)
    return output


def crop_bar_window(
    *,
    chart_id: str,
    image_path: Path,
    point_name: str,
    y_label: str,
    center_value: float,
    x_ticks: list[float],
    x_pixels: list[int],
    y_ticks: list[Any],
    y_pixels: list[int],
    round_index: int = 1,
    pad_x: int = 90,
    pad_y: int | None = None,
) -> tuple[Path, list[float], tuple[float, float]]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    center_x = numeric_pixel(center_value, x_ticks, x_pixels)
    center_y = category_pixel(y_label, y_ticks, y_pixels)
    span_y = category_span(y_label, y_ticks, y_pixels, img.size)
    half_y = max(24, (pad_y if pad_y is not None else span_y // 2 + 24))

    left = max(0, center_x - pad_x)
    right = min(width, center_x + pad_x)
    top = max(0, center_y - half_y)
    bottom = min(height, center_y + half_y)

    crop = img.crop((left, top, right, bottom))
    draw = ImageDraw.Draw(crop)

    min_val, max_val = value_range_from_pixels(left, right, x_ticks, x_pixels)
    visible_ticks = visible_ticks_for_range(x_ticks, min_val, max_val)
    for tick in visible_ticks:
        try:
            tick_x = numeric_pixel(float(tick), x_ticks, x_pixels) - left
        except Exception:
            continue
        draw.line((tick_x, 0, tick_x, crop.height), fill=(120, 120, 120), width=1)
        draw.text((tick_x + 2, 2), str(round(float(tick), 2)), fill=(0, 0, 0))

    local_y = center_y - top
    draw.line((0, local_y, crop.width, local_y), fill=(180, 180, 180), width=1)
    draw.rectangle((0, 0, crop.width - 1, crop.height - 1), outline=(0, 0, 0), width=1)

    output = temp_dir(chart_id) / (
        f"amplifier_crop_{safe_filename(point_name)}_round{round_index}.png"
    )
    crop.save(output)
    return output, visible_ticks, (min_val, max_val)
