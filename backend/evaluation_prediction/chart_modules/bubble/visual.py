"""Visual feedback and crop generation for bubble charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ...common.chart_io import ensure_dir, safe_filename

from .data import PointChartConfig
from .geometry import data_to_pixel, value_range_from_pixels, visible_ticks_for_range


def chart_result_dir(config: PointChartConfig, chart_id: str) -> Path:
    return ensure_dir(config.result_root / chart_id)


def temp_dir(config: PointChartConfig, chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(config, chart_id) / "tempy")


def raw_crop_dir(config: PointChartConfig, chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(config, chart_id) / "raw_crops")


def draw_crosshair(draw: ImageDraw.ImageDraw, x: int, y: int, *, color: str = "red", size: int = 12, width: int = 2) -> None:
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)


def draw_prediction_overlay(
    *,
    config: PointChartConfig,
    chart_id: str,
    original_img_path: Path,
    pred_coords: list[tuple[Any, Any]],
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int],
    y_pixels: list[int],
    point_name: str,
    draw_all_preds: bool = False,
) -> Path:
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]
    colors = ["red", "purple", "orange", "green", "blue", "black"]

    for idx, coord in enumerate(coords_to_draw):
        try:
            x_pixel, y_pixel = data_to_pixel((float(coord[0]), float(coord[1])), x_ticks, y_ticks, x_pixels, y_pixels)
        except Exception as exc:
            print(f"[bubble visual] Skip overlay coord {coord}: {exc}")
            continue
        draw_crosshair(draw, x_pixel, y_pixel, color=colors[idx % len(colors)])

    output = temp_dir(config, chart_id) / f"final_overlay_{safe_filename(chart_id)}_{safe_filename(point_name)}.png"
    img.save(output)
    return output


def crop_point_window(
    *,
    config: PointChartConfig,
    chart_id: str,
    image_path: Path,
    point_name: str,
    center_coord: tuple[float, float],
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int],
    y_pixels: list[int],
    round_index: int = 1,
    crop_size: int = 180,
) -> tuple[Path, list[float], list[float], tuple[float, float], tuple[float, float]]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    center_x, center_y = data_to_pixel(center_coord, x_ticks, y_ticks, x_pixels, y_pixels)
    half = crop_size // 2

    left = max(0, min(width - 1, center_x - half))
    upper = max(0, min(height - 1, center_y - half))
    right = min(width, left + crop_size)
    lower = min(height, upper + crop_size)
    left = max(0, right - crop_size)
    upper = max(0, lower - crop_size)

    crop = img.crop((left, upper, right, lower))
    draw = ImageDraw.Draw(crop)

    x_range = value_range_from_pixels(left, right, x_ticks, x_pixels)
    y_range = value_range_from_pixels(upper, lower, y_ticks, y_pixels)
    visible_x = visible_ticks_for_range(x_ticks, x_range[0], x_range[1])
    visible_y = visible_ticks_for_range(y_ticks, y_range[0], y_range[1])

    for tick in visible_x:
        try:
            x = data_to_pixel((float(tick), center_coord[1]), x_ticks, y_ticks, x_pixels, y_pixels)[0] - left
        except Exception:
            continue
        draw.line((x, 0, x, crop.height), fill=(130, 130, 130), width=1)
        draw.text((max(0, x - 16), crop.height - 14), str(round(float(tick), 2)), fill=(0, 0, 0))

    for tick in visible_y:
        try:
            y = data_to_pixel((center_coord[0], float(tick)), x_ticks, y_ticks, x_pixels, y_pixels)[1] - upper
        except Exception:
            continue
        draw.line((0, y, crop.width, y), fill=(130, 130, 130), width=1)
        draw.text((2, max(0, y - 12)), str(round(float(tick), 2)), fill=(0, 0, 0))

    draw.rectangle((0, 0, crop.width - 1, crop.height - 1), outline=(0, 0, 0), width=1)
    output = raw_crop_dir(config, chart_id) / f"{safe_filename(point_name)}_round{round_index}_adaptive.png"
    crop.save(output)
    return output, visible_x, visible_y, x_range, y_range
