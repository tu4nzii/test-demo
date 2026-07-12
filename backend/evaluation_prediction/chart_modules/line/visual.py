"""Visual feedback and crop generation for line charts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ...common.chart_io import ensure_dir, safe_filename
from ...common.paths import RESULTS_ROOT

from .geometry import category_pixel, category_span, numeric_pixel, value_range_from_pixels, visible_ticks_for_range


RESULT_ROOT = RESULTS_ROOT / "line"


def chart_result_dir(chart_id: str) -> Path:
    return ensure_dir(RESULT_ROOT / chart_id)


def temp_dir(chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(chart_id) / "tempy")


def feedback_dir(chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(chart_id) / "feedback")


def draw_prediction_overlay(
    *,
    chart_id: str,
    original_img_path: Path,
    pred_coords: list[tuple[Any, Any]],
    x_ticks: list[Any],
    y_ticks: list[float],
    x_pixels: list[int],
    y_pixels: list[int],
    point_name: str,
    draw_all_preds: bool = False,
    prompt_type: str = "feedback",
    image_type: str = "grid_with_grid",
    run_index: int | None = None,
    final_overlay: bool = False,
) -> Path:
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]
    colors = ["red", "purple", "orange", "green", "blue", "black"]

    for idx, coord in enumerate(coords_to_draw):
        try:
            x_pixel = category_pixel(coord[0], x_ticks, x_pixels)
            y_pixel = numeric_pixel(float(coord[1]), y_ticks, y_pixels)
        except Exception as exc:
            print(f"[line visual] Skip overlay coord {coord}: {exc}")
            continue
        size = 10
        color = colors[idx % len(colors)]
        draw.line((x_pixel - size, y_pixel, x_pixel + size, y_pixel), fill=color, width=2)
        draw.line((x_pixel, y_pixel - size, x_pixel, y_pixel + size), fill=color, width=2)

    safe_point_name = safe_filename(point_name)
    if final_overlay:
        round_suffix = f"_run{run_index}" if run_index is not None else ""
        filename = f"final_overlay_{safe_point_name}_{prompt_type}_{image_type}{round_suffix}.png"
    else:
        round_no = int(run_index) if run_index is not None else 1
        filename = f"overlay_{safe_point_name}_{prompt_type}_{image_type}_run{round_no}.png"
    output = feedback_dir(chart_id) / filename
    img.save(output)
    return output


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _numeric_pairs(ticks: list[float], pixels: list[int]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for tick, pixel in zip(ticks, pixels):
        try:
            pairs.append((float(tick), float(pixel)))
        except Exception:
            continue
    return sorted(pairs, key=lambda item: item[0])


def _dense_ticks_for_crop(
    pairs: list[tuple[float, float]],
    top_px: int,
    bottom_px: int,
    *,
    round_index: int = 1,
    grid_div: int | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
) -> tuple[list[float], list[float]]:
    lo, hi = sorted((top_px, bottom_px))
    dense_ticks: list[float] = []
    dense_pixels: list[float] = []
    if grid_div is not None:
        values = [value for value, _ in pairs]
        gaps = sorted(abs(right - left) for left, right in zip(values, values[1:]) if right != left)
        if gaps:
            base_step = gaps[len(gaps) // 2]
            tick_step = base_step / (2 ** max(0, int(grid_div))) if int(grid_div) > 0 else base_step
            region_min = min(values) if value_min is None else float(value_min)
            region_max = max(values) if value_max is None else float(value_max)
            start = math.floor(region_min / base_step) * base_step
            end = math.ceil(region_max / base_step) * base_step
            value = start - tick_step
            while value <= end + tick_step + 1e-9:
                dense_ticks.append(round(value, 4))
                dense_pixels.append(float(numeric_pixel(value, [p[0] for p in pairs], [int(p[1]) for p in pairs])))
                value += tick_step
            return dense_ticks, dense_pixels

    preferred = max(2, 2 ** max(1, int(round_index)))
    divisors = [preferred] + [divisor for divisor in (2, 3, 4, 5, 6, 8, 10, 12, 16) if divisor != preferred]
    for divisor in divisors:
        dense_ticks.clear()
        dense_pixels.clear()
        for (v1, p1), (v2, p2) in zip(pairs, pairs[1:]):
            for step in range(divisor + 1):
                alpha = step / divisor
                dense_ticks.append(round(v1 + (v2 - v1) * alpha, 4))
                dense_pixels.append(p1 + (p2 - p1) * alpha)
        if sum(lo <= pixel <= hi for pixel in dense_pixels) >= 6:
            break
    return dense_ticks, dense_pixels


def _format_tick(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def crop_line_point_window(
    *,
    chart_id: str,
    image_path: Path,
    point_name: str,
    x_label: str,
    center_value: float,
    x_ticks: list[Any],
    x_pixels: list[int],
    y_ticks: list[float],
    y_pixels: list[int],
    round_index: int = 1,
    attempt_index: int = 0,
    pad_x: int | None = None,
    pad_y: int = 90,
    half_ratio: float | None = None,
    zoom_factor: int | float | None = None,
    grid_div: int | None = None,
    max_canvas_size: int | None = 768,
) -> tuple[Path, list[float], tuple[float, float]]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    center_x = category_pixel(x_label, x_ticks, x_pixels)
    pairs = _numeric_pairs(y_ticks, y_pixels)
    if len(pairs) < 2:
        raise ValueError("At least two numeric y ticks are required for line amplifier crop.")
    min_value = min(value for value, _ in pairs)
    max_value = max(value for value, _ in pairs)
    clamped_value = min(max(float(center_value), min_value), max_value)
    center_y = numeric_pixel(clamped_value, y_ticks, y_pixels)
    span_x = category_span(x_label, x_ticks, x_pixels, img.size)
    half_x = max(28, (pad_x if pad_x is not None else span_x // 2 + 28))
    if half_ratio is not None:
        p_by_value = sorted(pairs, key=lambda item: item[0])
        value_span = max_value - min_value
        axis_pixel_span = abs(p_by_value[-1][1] - p_by_value[0][1])
        pixels_per_value = axis_pixel_span / value_span if value_span else 1.0
        half_y = int(max(5, abs(value_span * float(half_ratio) * pixels_per_value)))
    else:
        half_y = int(max(18, pad_y / (2 ** max(0, round_index - 1))))

    left = max(0, center_x - half_x)
    right = min(width, center_x + half_x)
    top = max(0, int(center_y - half_y))
    bottom = min(height, int(center_y + half_y))
    if bottom <= top:
        bottom = min(height, top + 12)

    raw_crop = img.crop((left, top, right, bottom))
    crop_w, crop_h = raw_crop.size
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop area: {(left, top, right, bottom)}")

    requested_zoom = float(zoom_factor) if zoom_factor is not None else 2 ** max(0, round_index - 1)
    new_w = max(1, int(round(crop_w * requested_zoom)))
    new_h = max(1, int(round(crop_h * requested_zoom)))
    zoom = requested_zoom
    if max_canvas_size:
        scale_down = min(float(max_canvas_size) / max(new_w, new_h, 1), 1.0)
        if scale_down < 1.0:
            new_w = max(1, int(round(new_w * scale_down)))
            new_h = max(1, int(round(new_h * scale_down)))
            zoom *= scale_down
    crop = raw_crop.resize((new_w, new_h), Image.NEAREST)

    min_val, max_val = value_range_from_pixels(top, bottom, y_ticks, y_pixels)
    dense_ticks, dense_pixels = _dense_ticks_for_crop(
        pairs,
        top,
        bottom,
        round_index=round_index,
        grid_div=grid_div,
        value_min=min_val,
        value_max=max_val,
    )
    font = _font(14)
    dummy = Image.new("RGB", (10, 10))
    ddraw = ImageDraw.Draw(dummy)
    max_text_w = 0
    for tick in dense_ticks or visible_ticks_for_range(y_ticks, min_val, max_val):
        bbox = ddraw.textbbox((0, 0), _format_tick(tick), font=font)
        max_text_w = max(max_text_w, bbox[2] - bbox[0])
    side_pad = max_text_w + 18
    canvas = Image.new("RGB", (new_w + side_pad, new_h), "white")
    canvas.paste(crop, (side_pad, 0))
    draw = ImageDraw.Draw(canvas)

    visible_ticks: list[float] = []
    seen_ticks: set[tuple[float, int]] = set()
    for tick, pixel in zip(dense_ticks, dense_pixels):
        lo, hi = sorted((top, bottom))
        if not (lo <= pixel <= hi):
            continue
        local_y = int(round(((pixel - top) / crop_h) * new_h))
        key = (round(float(tick), 6), local_y)
        if 0 <= local_y <= new_h and key not in seen_ticks:
            seen_ticks.add(key)
            visible_ticks.append(tick)
            draw.line((side_pad, local_y, side_pad + new_w, local_y), fill=(120, 120, 120), width=1)
            text = _format_tick(tick)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_h = bbox[3] - bbox[1]
            draw.text((4, local_y - text_h // 2), text, fill=(0, 0, 0), font=font)

    local_x = side_pad + int(round(((center_x - left) / crop_w) * new_w))
    draw.line((local_x, 0, local_x, canvas.height), fill=(180, 180, 180), width=1)
    draw.rectangle((side_pad, 0, side_pad + new_w - 1, new_h - 1), outline=(0, 0, 0), width=1)
    draw.text((side_pad + 4, 4), f"R{round_index}", font=font, fill="black")

    if not visible_ticks:
        visible_ticks = visible_ticks_for_range(y_ticks, min_val, max_val)

    output = temp_dir(chart_id) / f"amplifier_crop_{safe_filename(point_name)}_round{round_index}_attempt{attempt_index}.png"
    canvas.save(output)
    return output, visible_ticks, (min_val, max_val)
