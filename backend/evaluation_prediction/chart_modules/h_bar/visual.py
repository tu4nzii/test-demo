"""Visual feedback and crop generation for horizontal bar charts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ...common.chart_io import ensure_dir, safe_filename
from ...common.paths import RESULTS_ROOT

from .geometry import category_pixel, category_span, numeric_pixel, value_range_from_pixels, visible_ticks_for_range


def chart_result_dir(chart_id: str, chart_type: str = "h_bar") -> Path:
    return ensure_dir(RESULTS_ROOT / chart_type / chart_id)


def temp_dir(chart_id: str, chart_type: str = "h_bar") -> Path:
    return ensure_dir(chart_result_dir(chart_id, chart_type) / "tempy")


def feedback_dir(chart_id: str, chart_type: str = "h_bar") -> Path:
    return ensure_dir(chart_result_dir(chart_id, chart_type) / "feedback")


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
    prompt_type: str = "feedback",
    image_type: str = "grid_with_grid",
    run_index: int | None = None,
    final_overlay: bool = False,
    chart_type: str = "h_bar",
    stacked_start_value: float | None = None,
) -> Path:
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]
    colors = ["red", "purple", "orange", "green", "blue", "black", "brown", "pink"]

    for idx, coord in enumerate(coords_to_draw):
        try:
            y_label = str(coord[1]).split(",")[-1].strip()
            y_pixel = category_pixel(y_label, y_ticks, y_pixels)
            if str(chart_type or "").lower() == "h_stacked_bar" and stacked_start_value is not None:
                start_x = numeric_pixel(float(stacked_start_value), x_ticks, x_pixels)
                end_x = numeric_pixel(float(stacked_start_value) + float(coord[0]), x_ticks, x_pixels)
                x_pixel = end_x
            else:
                start_x = None
                end_x = None
                x_pixel = numeric_pixel(float(coord[0]), x_ticks, x_pixels)
        except Exception as exc:
            print(f"[h_bar visual] Skip overlay coord {coord}: {exc}")
            continue

        half_span = category_span(coord[1], y_ticks, y_pixels, img.size) // 2
        color = colors[idx % len(colors)]
        if start_x is not None and end_x is not None:
            left_x, right_x = sorted((start_x, end_x))
            draw.line((left_x, y_pixel, right_x, y_pixel), fill=color, width=3)
            draw.line((left_x, y_pixel - half_span, left_x, y_pixel + half_span), fill=color, width=2)
            draw.line((right_x, y_pixel - half_span, right_x, y_pixel + half_span), fill=color, width=2)
        else:
            draw.line((x_pixel - half_span, y_pixel, x_pixel + half_span, y_pixel), fill=color, width=2)
            draw.line((x_pixel, y_pixel - half_span, x_pixel, y_pixel + half_span), fill=color, width=2)

    safe_point_name = safe_filename(point_name)
    if final_overlay:
        round_suffix = f"_run{run_index}" if run_index is not None else ""
        filename = f"final_overlay_{safe_point_name}_{prompt_type}_{image_type}{round_suffix}.png"
    else:
        round_no = int(run_index) if run_index is not None else 1
        filename = f"overlay_{safe_point_name}_{prompt_type}_{image_type}_run{round_no}.png"
    output = feedback_dir(chart_id, chart_type) / filename
    img.save(output)
    return output


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_rotated_text(
    image: Image.Image,
    text: str,
    center: tuple[int, int],
    angle: float,
    font: ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    scratch = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    pad = 8
    text_img = Image.new("RGBA", (width + pad * 2, height + pad * 2), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((pad, pad), text, font=font, fill=fill)
    rotated = text_img.rotate(angle, expand=True)
    x = int(center[0] - rotated.width / 2)
    y = int(center[1] - rotated.height / 2)
    image.paste(rotated, (x, y), rotated)


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
    left_px: int,
    right_px: int,
    round_index: int = 1,
    grid_div: int | None = None,
    value_min: float | None = None,
    value_max: float | None = None,
) -> tuple[list[float], list[float]]:
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
            ticks = [p[0] for p in pairs]
            pixels = [int(p[1]) for p in pairs]
            value = start - tick_step
            while value <= end + tick_step + 1e-9:
                dense_ticks.append(round(value, 4))
                dense_pixels.append(float(numeric_pixel(value, ticks, pixels)))
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
        if sum(left_px <= pixel <= right_px for pixel in dense_pixels) >= 6:
            break
    return dense_ticks, dense_pixels


def _format_tick(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


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
    attempt_index: int = 0,
    pad_x: int | None = None,
    pad_y: int | None = None,
    half_ratio: float | None = None,
    zoom_factor: int | float | None = None,
    grid_div: int | None = None,
    max_canvas_size: int | None = 768,
    chart_type: str = "h_bar",
) -> tuple[Path, list[float], tuple[float, float]]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pairs = _numeric_pairs(x_ticks, x_pixels)
    if len(pairs) < 2:
        raise ValueError("At least two numeric x ticks are required for h_bar amplifier crop.")
    min_value = min(value for value, _ in pairs)
    max_value = max(value for value, _ in pairs)
    clamped_value = min(max(float(center_value), min_value), max_value)
    value_pixels = sorted(pairs, key=lambda item: item[0])
    v_min, p_min = value_pixels[0]
    v_max, p_max = value_pixels[-1]
    scale = (p_max - p_min) / (v_max - v_min) if v_max != v_min else 1.0
    center_x = p_min + (clamped_value - v_min) * scale
    center_y = category_pixel(y_label, y_ticks, y_pixels)
    span_y = category_span(y_label, y_ticks, y_pixels, img.size)
    half_y = max(18, (pad_y if pad_y is not None else span_y // 2))

    pixel_gaps = [abs(right - left) for (_, left), (_, right) in zip(pairs, pairs[1:])]
    base_span = sorted(pixel_gaps)[len(pixel_gaps) // 2] if pixel_gaps else 90
    shrink = 2 ** max(0, round_index - 1)
    if half_ratio is not None:
        p_by_value = sorted(pairs, key=lambda item: item[0])
        value_span = max_value - min_value
        axis_pixel_span = abs(p_by_value[-1][1] - p_by_value[0][1])
        pixels_per_value = axis_pixel_span / value_span if value_span else 1.0
        half_x = max(5.0, abs(value_span * float(half_ratio) * pixels_per_value))
    else:
        half_x = int(max(18, (pad_x if pad_x is not None else base_span) / shrink))

    left = max(0, int(center_x - half_x))
    right = min(width, int(center_x + half_x))
    if right <= left:
        right = min(width, left + 12)
    top = max(0, center_y - half_y)
    bottom = min(height, center_y + half_y)
    min_val, max_val = value_range_from_pixels(left, right, x_ticks, x_pixels)

    raw_crop = img.crop((left, top, right, bottom))
    crop_w, crop_h = raw_crop.size
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop area: {(left, top, right, bottom)}")

    requested_zoom = float(zoom_factor) if zoom_factor is not None else 2 ** max(0, round_index - 1)
    new_w = max(1, int(round(crop_w * requested_zoom)))
    new_h = max(1, int(round(crop_h * requested_zoom)))
    zoom = requested_zoom
    label_pad = 70
    if max_canvas_size:
        scale_down = min(
            float(max_canvas_size) / max(new_w, 1),
            float(max_canvas_size) / max(new_h + label_pad * 2, 1),
            1.0,
        )
        if scale_down < 1.0:
            new_w = max(1, int(round(new_w * scale_down)))
            new_h = max(1, int(round(new_h * scale_down)))
            zoom *= scale_down
    resized = raw_crop.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("RGB", (new_w, new_h + label_pad * 2), "white")
    offset_x = 0
    offset_y = label_pad
    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)

    crop_top = offset_y
    crop_bottom = offset_y + new_h
    draw.text((offset_x + 4, crop_top + 4), f"R{round_index}", font=font, fill="black")

    dense_ticks, dense_pixels = _dense_ticks_for_crop(
        pairs,
        left,
        right,
        round_index=round_index,
        grid_div=grid_div,
        value_min=min_val,
        value_max=max_val,
    )
    visible_ticks: list[float] = []
    mapped_ticks: list[tuple[float, int]] = []
    seen_ticks: set[tuple[float, int]] = set()
    for tick, pixel in zip(dense_ticks, dense_pixels):
        if not (left <= pixel <= right):
            continue
        local_x = offset_x + int(round(((pixel - left) / crop_w) * new_w))
        key = (round(float(tick), 6), local_x)
        if offset_x <= local_x <= offset_x + new_w and key not in seen_ticks:
            seen_ticks.add(key)
            mapped_ticks.append((tick, local_x))
            visible_ticks.append(tick)

    dash_region = 5
    dash_len = 10
    dash_gap = 4
    tick_len = 6
    for tick, tick_x in mapped_ticks:
        y = crop_top
        while y < crop_bottom:
            y_end = min(y + dash_len, crop_bottom)
            draw.line((tick_x, y, tick_x, y_end), fill="gray", width=1)
            y += dash_len + dash_gap
        draw.line(
            (tick_x, crop_top - dash_region - tick_len, tick_x, crop_top - dash_region),
            fill="black",
            width=1,
        )
        draw.line(
            (tick_x, crop_bottom + dash_region, tick_x, crop_bottom + dash_region + tick_len),
            fill="black",
            width=1,
        )
        _draw_rotated_text(
            canvas,
            _format_tick(tick),
            (tick_x, crop_top - dash_region - tick_len - 25),
            45,
            font,
        )
        _draw_rotated_text(
            canvas,
            _format_tick(tick),
            (tick_x, crop_bottom + dash_region + tick_len + 25),
            -90,
            font,
        )

    draw.line((offset_x, crop_top, offset_x + new_w, crop_top), fill="black", width=2)
    draw.line((offset_x, crop_bottom, offset_x + new_w, crop_bottom), fill="black", width=2)

    if not visible_ticks:
        visible_ticks = visible_ticks_for_range(x_ticks, min_val, max_val)

    output = temp_dir(chart_id, chart_type) / (
        f"amplifier_crop_{safe_filename(point_name)}_round{round_index}_attempt{attempt_index}.png"
    )
    canvas.save(output)
    return output, visible_ticks, (min_val, max_val)
