"""Visual feedback and crop generation for scatter charts."""

from __future__ import annotations

from math import hypot
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...common.chart_io import ensure_dir, safe_filename

from .data import PointChartConfig
from .geometry import data_to_pixel


def chart_result_dir(config: PointChartConfig, chart_id: str) -> Path:
    return ensure_dir(config.result_root / chart_id)


def temp_dir(config: PointChartConfig, chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(config, chart_id) / "tempy")


def feedback_dir(config: PointChartConfig, chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(config, chart_id) / "feedback")


def raw_crop_dir(config: PointChartConfig, chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(config, chart_id) / "raw_crops")


def draw_crosshair(draw: ImageDraw.ImageDraw, x: int, y: int, *, color: str = "red", size: int = 12, width: int = 2) -> None:
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)


def _format_tick_value(value: float) -> str:
    rounded = round(float(value), 2)
    return f"{int(round(rounded))}" if abs(rounded - round(rounded)) < 1e-6 else f"{rounded:.2f}"


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str | tuple[int, int, int] = "gray",
    width: int = 1,
    dash_length: int = 1,
    gap_length: int = 4,
) -> None:
    x1, y1 = start
    x2, y2 = end
    total = hypot(x2 - x1, y2 - y1)
    if total <= 0:
        return
    step = dash_length + gap_length
    for index in range(int(total // step) + 1):
        start_frac = (index * step) / total
        end_frac = min((index * step + dash_length) / total, 1)
        sx = x1 + (x2 - x1) * start_frac
        sy = y1 + (y2 - y1) * start_frac
        ex = x1 + (x2 - x1) * end_frac
        ey = y1 + (y2 - y1) * end_frac
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)


def _draw_tick_text(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    position: float,
    text: str,
    *,
    axis: str,
    font: ImageFont.ImageFont,
    color: str = "red",
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if axis == "x":
        txt_img = Image.new("RGBA", (max(1, text_w * 3), max(1, text_h * 3)), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((text_w * 1.5, text_h * 1.5), text, fill=color, font=font, anchor="mm")
        rotated = txt_img.rotate(45, expand=1, resample=Image.BICUBIC)
        crop_box = rotated.getbbox()
        if crop_box:
            rotated = rotated.crop(crop_box)
        image.paste(rotated, (int(position - rotated.width / 2), image.height - rotated.height), rotated)
    else:
        draw.text((4, position - text_h / 2), text, fill=color, font=font)


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
    run_index: int | None = None,
    prompt_type: str = "feedback",
    image_type: str = "grid_with_grid",
    final_overlay: bool = False,
) -> Path:
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]
    colors = ["red", "purple", "orange", "green", "blue", "black"]

    for idx, coord in enumerate(coords_to_draw):
        try:
            x_pixel, y_pixel = data_to_pixel((float(coord[0]), float(coord[1])), x_ticks, y_ticks, x_pixels, y_pixels)
        except Exception as exc:
            print(f"[scatter visual] Skip overlay coord {coord}: {exc}")
            continue
        draw_crosshair(draw, x_pixel, y_pixel, color=colors[idx % len(colors)])

    safe_point_name = safe_filename(point_name)
    if final_overlay:
        round_suffix = f"_run{run_index}" if run_index is not None else ""
        filename = f"final_overlay_{safe_point_name}_{prompt_type}_{image_type}{round_suffix}.png"
    elif run_index is None:
        filename = f"final_overlay_{safe_point_name}_{prompt_type}_{image_type}.png"
    else:
        filename = f"overlay_{safe_point_name}_{prompt_type}_{image_type}_run{run_index}.png"
    output = feedback_dir(config, chart_id) / filename
    img.save(output)
    return output


def crop_draw_ticks_resize(
    *,
    config: PointChartConfig,
    chart_id: str,
    image_path: Path,
    point_name: str,
    pred_coord: tuple[float, float],
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int],
    y_pixels: list[int],
    feedback_round: int = 1,
    window_size: int = 120,
    output_size: tuple[int, int] | None = (224, 224),
    font_size: int = 10,
    x_grid_density: int = 0,
    y_grid_density: int = 0,
) -> tuple[Path, list[float], list[float], list[float], list[float], Callable[[float, float], tuple[float, float]], int, int]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    center_x, center_y = data_to_pixel(pred_coord, x_ticks, y_ticks, x_pixels, y_pixels)
    half = window_size // 2

    left = max(0, min(width - 1, center_x - half))
    upper = max(0, min(height - 1, center_y - half))
    right = min(width, left + window_size)
    lower = min(height, upper + window_size)
    left = max(0, right - window_size)
    upper = max(0, lower - window_size)

    crop = img.crop((left, upper, right, lower))
    draw = ImageDraw.Draw(crop)
    font = _load_font(font_size)

    new_x_ticks: list[float] = []
    new_y_ticks: list[float] = []
    new_x_pixels: list[float] = []
    new_y_pixels: list[float] = []

    def add_x_tick(value: float, pixel: float, *, interpolated: bool = False) -> None:
        if left <= pixel <= right:
            rel = pixel - left
            _draw_dashed_line(draw, (rel, 0), (rel, crop.height), fill="lightgray" if interpolated else "gray")
            _draw_tick_text(crop, draw, rel, _format_tick_value(value), axis="x", font=font)
            new_x_ticks.append(round(float(value), 4))
            new_x_pixels.append(float(rel))

    def add_y_tick(value: float, pixel: float, *, interpolated: bool = False) -> None:
        if upper <= pixel <= lower:
            rel = pixel - upper
            _draw_dashed_line(draw, (0, rel), (crop.width, rel), fill="lightgray" if interpolated else "gray")
            _draw_tick_text(crop, draw, rel, _format_tick_value(value), axis="y", font=font)
            new_y_ticks.append(round(float(value), 4))
            new_y_pixels.append(float(rel))

    for value, pixel in zip(x_ticks, x_pixels):
        add_x_tick(float(value), float(pixel))
    for value, pixel in zip(y_ticks, y_pixels):
        add_y_tick(float(value), float(pixel))

    if x_grid_density > 0:
        for index in range(len(x_ticks) - 1):
            value1, value2 = float(x_ticks[index]), float(x_ticks[index + 1])
            pixel1, pixel2 = float(x_pixels[index]), float(x_pixels[index + 1])
            for step in range(1, x_grid_density + 1):
                ratio = step / (x_grid_density + 1)
                add_x_tick(value1 + (value2 - value1) * ratio, pixel1 + (pixel2 - pixel1) * ratio, interpolated=True)

    if y_grid_density > 0:
        for index in range(len(y_ticks) - 1):
            value1, value2 = float(y_ticks[index]), float(y_ticks[index + 1])
            pixel1, pixel2 = float(y_pixels[index]), float(y_pixels[index + 1])
            for step in range(1, y_grid_density + 1):
                ratio = step / (y_grid_density + 1)
                add_y_tick(value1 + (value2 - value1) * ratio, pixel1 + (pixel2 - pixel1) * ratio, interpolated=True)

    draw.rectangle((0, 0, crop.width - 1, crop.height - 1), outline=(0, 0, 0), width=1)

    if output_size:
        scale_x = output_size[0] / crop.width
        scale_y = output_size[1] / crop.height
        crop = crop.resize(output_size, Image.LANCZOS)
        new_x_pixels = [round(pixel * scale_x, 4) for pixel in new_x_pixels]
        new_y_pixels = [round(pixel * scale_y, 4) for pixel in new_y_pixels]
    else:
        scale_x = scale_y = 1.0

    def pixel_to_cropped_coords(x_px: float, y_px: float) -> tuple[float, float]:
        return (x_px - left) * scale_x, (y_px - upper) * scale_y

    output = raw_crop_dir(config, chart_id) / f"{safe_filename(point_name)}_round{feedback_round}_crop.png"
    crop.save(output)
    return output, new_x_ticks, new_y_ticks, new_x_pixels, new_y_pixels, pixel_to_cropped_coords, left, upper


def generate_expanded_crop_with_grid_by_diameter(
    *,
    config: PointChartConfig,
    chart_id: str,
    image_path: Path,
    point_name: str,
    pred_coord: tuple[float, float],
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int],
    y_pixels: list[int],
    diameter: float,
    feedback_round: int = 1,
    base_crop_size: int = 120,
    resize_to: tuple[int, int] = (224, 224),
) -> tuple[Path, list[float], list[float], list[float], list[float], Callable[[float, float], tuple[float, float]], int, int]:
    base_img = Image.open(image_path).convert("RGB")
    pixel_x, pixel_y = data_to_pixel(pred_coord, x_ticks, y_ticks, x_pixels, y_pixels)
    diameter = max(8.0, min(float(diameter), 180.0))
    rounded_diameter = round(diameter / 10) * 10
    grid_span = diameter * 2 if rounded_diameter <= 10 else diameter
    canvas_size = max(base_crop_size, int(grid_span * 6))

    left = int(max(pixel_x - base_crop_size // 2, 0))
    upper = int(max(pixel_y - base_crop_size // 2, 0))
    right = min(base_img.width, left + base_crop_size)
    lower = min(base_img.height, upper + base_crop_size)
    left = max(0, right - base_crop_size)
    upper = max(0, lower - base_crop_size)

    cropped = base_img.crop((left, upper, right, lower))
    paste_x = (canvas_size - cropped.width) // 2
    paste_y = (canvas_size - cropped.height) // 2
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    canvas.paste(cropped, (paste_x, paste_y))

    resized = canvas.resize(resize_to, Image.LANCZOS)
    draw = ImageDraw.Draw(resized)
    font = _load_font(12)
    scale = resize_to[0] / canvas_size

    def closest_tick(coord: float, ticks: list[float], pixels: list[int]) -> tuple[float, float]:
        index = min(range(len(ticks)), key=lambda item: abs(float(ticks[item]) - float(coord)))
        return float(ticks[index]), float(pixels[index])

    tick_x, px_x = closest_tick(pred_coord[0], x_ticks, x_pixels)
    tick_y, px_y = closest_tick(pred_coord[1], y_ticks, y_pixels)
    x_value_span = float(x_ticks[1]) - float(x_ticks[0]) if len(x_ticks) > 1 else 1.0
    y_value_span = float(y_ticks[1]) - float(y_ticks[0]) if len(y_ticks) > 1 else 1.0
    x_pixel_span = float(x_pixels[1]) - float(x_pixels[0]) if len(x_pixels) > 1 else 20.0
    y_pixel_span = float(y_pixels[1]) - float(y_pixels[0]) if len(y_pixels) > 1 else 20.0

    grid_center_x = paste_x + (px_x - left)
    grid_center_y = paste_y + (px_y - upper)
    new_x_ticks: list[float] = []
    new_y_ticks: list[float] = []
    new_x_pixels: list[float] = []
    new_y_pixels: list[float] = []

    def draw_axis_ticks(
        *,
        axis: str,
        center_px: float,
        tick_start: float,
        value_span: float,
        pixel_span: float,
        tick_list: list[float],
        pixel_list: list[float],
    ) -> None:
        for direction in (1, -1):
            index = 0 if direction == 1 else 1
            while True:
                pos = center_px + direction * index * grid_span
                if pos < 0 or pos >= canvas_size:
                    break
                tick_val = tick_start + direction * index * (grid_span / pixel_span) * value_span
                scaled_pos = pos * scale
                if axis == "x":
                    _draw_dashed_line(draw, (scaled_pos, 0), (scaled_pos, resize_to[1]), fill="gray")
                    _draw_tick_text(resized, draw, scaled_pos, _format_tick_value(tick_val), axis="x", font=font)
                else:
                    _draw_dashed_line(draw, (0, scaled_pos), (resize_to[0], scaled_pos), fill="gray")
                    _draw_tick_text(resized, draw, scaled_pos, _format_tick_value(tick_val), axis="y", font=font)
                if direction == 1:
                    tick_list.append(round(tick_val, 4))
                    pixel_list.append(round(scaled_pos, 4))
                else:
                    tick_list.insert(0, round(tick_val, 4))
                    pixel_list.insert(0, round(scaled_pos, 4))
                index += 1

    draw_axis_ticks(
        axis="x",
        center_px=grid_center_x,
        tick_start=tick_x,
        value_span=x_value_span,
        pixel_span=x_pixel_span,
        tick_list=new_x_ticks,
        pixel_list=new_x_pixels,
    )
    draw_axis_ticks(
        axis="y",
        center_px=grid_center_y,
        tick_start=tick_y,
        value_span=y_value_span,
        pixel_span=y_pixel_span,
        tick_list=new_y_ticks,
        pixel_list=new_y_pixels,
    )

    def pixel_to_cropped_coords(x_px: float, y_px: float) -> tuple[float, float]:
        return (x_px - left + paste_x) * scale, (y_px - upper + paste_y) * scale

    output = raw_crop_dir(config, chart_id) / f"{safe_filename(point_name)}_round{feedback_round}_adaptive.png"
    resized.save(output)
    return output, new_x_ticks, new_y_ticks, new_x_pixels, new_y_pixels, pixel_to_cropped_coords, left, upper
