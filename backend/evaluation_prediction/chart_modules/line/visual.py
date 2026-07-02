"""Visual feedback and crop generation for line charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ...common.amplifier_style import (
    AMPLIFIER_GRID_COLOR,
    amplifier_half_window,
    amplifier_label_pad,
    amplifier_output_side,
    amplifier_tick_divisor,
    draw_amplifier_dashed_line,
    draw_centered_label_box,
)
from ...common.chart_io import ensure_dir, safe_filename
from ...common.paths import RESULTS_ROOT

from .geometry import category_pixel, category_span, numeric_pixel, value_range_from_pixels, visible_ticks_for_range


RESULT_ROOT = RESULTS_ROOT / "line"


def chart_result_dir(chart_id: str) -> Path:
    return ensure_dir(RESULT_ROOT / chart_id)


def temp_dir(chart_id: str) -> Path:
    return ensure_dir(chart_result_dir(chart_id) / "tempy")


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

    output = temp_dir(chart_id) / f"final_overlay_{safe_filename(chart_id)}_{safe_filename(point_name)}.png"
    img.save(output)
    return output


def _font(size: int = 16) -> ImageFont.ImageFont:
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
    round_index: int,
) -> tuple[list[float], list[float]]:
    lo, hi = sorted((top_px, bottom_px))
    dense_ticks: list[float] = []
    dense_pixels: list[float] = []
    preferred = min(4, amplifier_tick_divisor(round_index))
    divisors = [preferred] + [divisor for divisor in (2, 3, 4, 5, 6) if divisor != preferred]
    for divisor in divisors:
        dense_ticks.clear()
        dense_pixels.clear()
        for (v1, p1), (v2, p2) in zip(pairs, pairs[1:]):
            for step in range(divisor + 1):
                alpha = step / divisor
                dense_ticks.append(round(v1 + (v2 - v1) * alpha, 4))
                dense_pixels.append(p1 + (p2 - p1) * alpha)
        if sum(lo <= pixel <= hi for pixel in dense_pixels) >= 5:
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
    zoom_round_index: int | None = None,
    roi_scale: float = 1.0,
    pad_x: int | None = None,
    pad_y: int = 90,
) -> tuple[Path, list[float], tuple[float, float]]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    center_x = category_pixel(x_label, x_ticks, x_pixels)
    center_y = numeric_pixel(center_value, y_ticks, y_pixels)
    span_x = category_span(x_label, x_ticks, x_pixels, img.size)
    half_x = max(28, (pad_x if pad_x is not None else span_x // 2 + 28))

    plot_left = int(max(0, min(x_pixels)))
    plot_right = int(min(width, max(x_pixels)))
    left = max(plot_left, center_x - half_x)
    right = min(plot_right, center_x + half_x)
    pairs = _numeric_pairs(y_ticks, y_pixels)
    pixel_gaps = [abs(right - left) for (_, left), (_, right) in zip(pairs, pairs[1:])]
    base_y = pad_y if pad_y is not None else (sorted(pixel_gaps)[len(pixel_gaps) // 2] if pixel_gaps else 90)
    zoom_round = zoom_round_index if zoom_round_index is not None else round_index
    plot_top = int(max(0, min(y_pixels)))
    plot_bottom = int(min(height, max(y_pixels)))
    half_y = amplifier_half_window(
        base_y,
        zoom_round,
        roi_scale=roi_scale,
        min_px=40,
        plot_span_px=max(1, plot_bottom - plot_top),
    )
    top = max(plot_top, center_y - half_y)
    bottom = min(plot_bottom, center_y + half_y)

    raw_crop = img.crop((left, top, right, bottom))
    crop_w, crop_h = raw_crop.size
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop area: {(left, top, right, bottom)}")

    out_size = amplifier_output_side(crop_w, crop_h)
    label_pad = amplifier_label_pad()
    scale = min(out_size / max(1, crop_w), out_size / max(1, crop_h))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    resized = raw_crop.resize((new_w, new_h), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (out_size + label_pad, out_size), "white")
    offset_x = label_pad + (out_size - new_w) // 2
    offset_y = (out_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    font = _font(18)

    min_val, max_val = value_range_from_pixels(top, bottom, y_ticks, y_pixels)
    dense_ticks, dense_pixels = _dense_ticks_for_crop(pairs, top, bottom, round_index)
    visible_ticks: list[float] = []
    mapped_ticks: list[tuple[float, int]] = []
    crop_left = offset_x
    crop_right = offset_x + new_w
    crop_top = offset_y
    crop_bottom = offset_y + new_h
    for tick, pixel in zip(dense_ticks, dense_pixels):
        lo, hi = sorted((top, bottom))
        if not (lo <= pixel <= hi):
            continue
        tick_y = offset_y + int(round(((pixel - top) / crop_h) * new_h))
        mapped_ticks.append((tick, tick_y))
        draw_amplifier_dashed_line(draw, (crop_left, tick_y), (crop_right, tick_y))

    last_label_y: int | None = None
    for tick, tick_y in sorted(mapped_ticks, key=lambda item: item[1]):
        if last_label_y is not None and abs(tick_y - last_label_y) < 34:
            continue
        text = _format_tick(tick)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if tick_y < text_h / 2 + 4 or tick_y > canvas.height - text_h / 2 - 4:
            continue
        label_center_x = max(6 + text_w / 2, crop_left - 10 - text_w / 2)
        draw_amplifier_dashed_line(
            draw,
            (label_center_x, tick_y),
            (crop_right, tick_y),
            fill=AMPLIFIER_GRID_COLOR,
        )
        draw_centered_label_box(
            draw,
            text,
            (label_center_x, tick_y),
            font=font,
            fill=(0, 0, 0),
        )
        visible_ticks.append(tick)
        last_label_y = tick_y

    if not visible_ticks:
        visible_ticks = visible_ticks_for_range(y_ticks, min_val, max_val)

    local_x = offset_x + int(round(((center_x - left) / crop_w) * new_w))
    draw_amplifier_dashed_line(draw, (local_x, crop_top), (local_x, crop_bottom), fill=AMPLIFIER_GRID_COLOR)
    draw.line((crop_left, crop_top, crop_left, crop_bottom), fill="black", width=2)
    draw.line((crop_right, crop_top, crop_right, crop_bottom), fill="black", width=2)
    draw.rectangle((crop_left, crop_top, crop_right - 1, crop_bottom - 1), outline=(0, 0, 0), width=1)
    draw.text((crop_left + 4, crop_top + 4), f"R{round_index}", fill=(0, 0, 0), font=font)

    output = temp_dir(chart_id) / f"amplifier_crop_{safe_filename(point_name)}_round{round_index}.png"
    canvas.save(output)
    return output, visible_ticks, (min_val, max_val)
