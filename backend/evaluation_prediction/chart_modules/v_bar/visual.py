"""Visual feedback and crop generation for vertical bar charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ...common.chart_io import ensure_dir, safe_filename
from ...common.paths import RESULTS_ROOT

from .geometry import category_pixel, category_span, numeric_pixel, value_range_from_pixels, visible_ticks_for_range


def chart_result_dir(chart_id: str, chart_type: str = "v_bar") -> Path:
    return ensure_dir(RESULTS_ROOT / chart_type / chart_id)


def temp_dir(chart_id: str, chart_type: str = "v_bar") -> Path:
    return ensure_dir(chart_result_dir(chart_id, chart_type) / "tempy")


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
    chart_type: str = "v_bar",
    stacked_start_value: float | None = None,
) -> Path:
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]
    colors = ["red", "purple", "orange", "green", "blue", "black", "brown", "pink"]

    for idx, coord in enumerate(coords_to_draw):
        try:
            x_pixel = category_pixel(coord[0], x_ticks, x_pixels)
            if str(chart_type or "").lower() == "v_stacked_bar" and stacked_start_value is not None:
                start_y = numeric_pixel(float(stacked_start_value), y_ticks, y_pixels)
                end_y = numeric_pixel(float(stacked_start_value) + float(coord[1]), y_ticks, y_pixels)
                y_pixel = end_y
            else:
                start_y = None
                end_y = None
                y_pixel = numeric_pixel(float(coord[1]), y_ticks, y_pixels)
        except Exception as exc:
            print(f"[v_bar visual] Skip overlay coord {coord}: {exc}")
            continue

        half_span = category_span(coord[0], x_ticks, x_pixels, img.size) // 2
        color = colors[idx % len(colors)]
        if start_y is not None and end_y is not None:
            top_y, bottom_y = sorted((start_y, end_y))
            draw.line((x_pixel, top_y, x_pixel, bottom_y), fill=color, width=3)
            draw.line((x_pixel - half_span, top_y, x_pixel + half_span, top_y), fill=color, width=2)
            draw.line((x_pixel - half_span, bottom_y, x_pixel + half_span, bottom_y), fill=color, width=2)
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
    output = temp_dir(chart_id, chart_type) / filename
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
    round_index: int = 1,
) -> tuple[list[float], list[float]]:
    lo, hi = sorted((top_px, bottom_px))
    dense_ticks: list[float] = []
    dense_pixels: list[float] = []
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


def crop_bar_window(
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
    pad_y: int | None = None,
    chart_type: str = "v_bar",
) -> tuple[Path, list[float], tuple[float, float]]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    center_x = category_pixel(x_label, x_ticks, x_pixels)
    pairs = _numeric_pairs(y_ticks, y_pixels)
    if len(pairs) < 2:
        raise ValueError("At least two numeric y ticks are required for v_bar amplifier crop.")
    min_value = min(value for value, _ in pairs)
    max_value = max(value for value, _ in pairs)
    clamped_value = min(max(float(center_value), min_value), max_value)
    center_y = numeric_pixel(clamped_value, y_ticks, y_pixels)
    span_x = category_span(x_label, x_ticks, x_pixels, img.size)
    half_x = max(18, (pad_x if pad_x is not None else span_x // 2 + 8))

    pixel_gaps = [abs(right - left) for (_, left), (_, right) in zip(pairs, pairs[1:])]
    base_span = sorted(pixel_gaps)[len(pixel_gaps) // 2] if pixel_gaps else 90
    shrink = 2 ** max(0, round_index - 1)
    half_y = int(max(18, (pad_y if pad_y is not None else base_span) / shrink))

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

    out_size = 240
    label_pad = 76
    scale = min(out_size / max(1, crop_w), out_size / max(1, crop_h))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    resized = raw_crop.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("RGB", (out_size + label_pad * 2, out_size), "white")
    offset_x = label_pad + (out_size - new_w) // 2
    offset_y = (out_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)

    crop_left = offset_x
    crop_right = offset_x + new_w
    draw.text((crop_left + 4, offset_y + 4), f"R{round_index}", font=font, fill="black")

    dense_ticks, dense_pixels = _dense_ticks_for_crop(pairs, top, bottom, round_index=round_index)
    visible_ticks: list[float] = []
    mapped_ticks: list[tuple[float, int]] = []
    seen_ticks: set[tuple[float, int]] = set()
    for tick, pixel in zip(dense_ticks, dense_pixels):
        lo, hi = sorted((top, bottom))
        if not (lo <= pixel <= hi):
            continue
        local_y = offset_y + int(round(((pixel - top) / crop_h) * new_h))
        key = (round(float(tick), 6), local_y)
        if offset_y <= local_y <= offset_y + new_h and key not in seen_ticks:
            seen_ticks.add(key)
            mapped_ticks.append((tick, local_y))
            visible_ticks.append(tick)

    dash_region = 5
    dash_len = 10
    dash_gap = 4
    tick_len = 6
    for tick, tick_y in mapped_ticks:
        x = crop_left
        while x < crop_right:
            x_end = min(x + dash_len, crop_right)
            draw.line((x, tick_y, x_end, tick_y), fill="gray", width=1)
            x += dash_len + dash_gap
        text = _format_tick(tick)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tick_x0 = crop_left - dash_region - tick_len
        tick_x1 = crop_left - dash_region
        draw.line((tick_x0, tick_y, tick_x1, tick_y), fill="black", width=1)
        draw.text((tick_x0 - 4 - tw, tick_y - th // 2), text, font=font, fill="black")
        tick_x0 = crop_right + dash_region
        tick_x1 = tick_x0 + tick_len
        draw.line((tick_x0, tick_y, tick_x1, tick_y), fill="black", width=1)

    draw.line((crop_left, offset_y, crop_left, offset_y + new_h), fill="black", width=2)
    draw.line((crop_right, offset_y, crop_right, offset_y + new_h), fill="black", width=2)

    min_val, max_val = value_range_from_pixels(top, bottom, y_ticks, y_pixels)
    if not visible_ticks:
        visible_ticks = visible_ticks_for_range(y_ticks, min_val, max_val)

    output = temp_dir(chart_id, chart_type) / (
        f"amplifier_crop_{safe_filename(point_name)}_round{round_index}_attempt{attempt_index}.png"
    )
    canvas.save(output)
    return output, visible_ticks, (min_val, max_val)
