"""Visual feedback and crop generation for horizontal bar charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ...common.amplifier_style import (
    AMPLIFIER_BAR_DASH_GAP,
    AMPLIFIER_BAR_DASH_LENGTH,
    AMPLIFIER_GRID_COLOR,
    amplifier_half_window,
    amplifier_label_pad,
    amplifier_output_side,
    amplifier_tick_divisor,
    draw_amplifier_dashed_line,
    draw_rotated_centered_label_box,
)
from ...common.chart_io import ensure_dir, safe_filename
from ...common.paths import RESULTS_ROOT

from .geometry import category_pixel, category_span, numeric_pixel, value_range_from_pixels, visible_ticks_for_range


def chart_result_dir(chart_id: str, chart_type: str = "h_bar") -> Path:
    return ensure_dir(RESULTS_ROOT / chart_type / chart_id)


def temp_dir(chart_id: str, chart_type: str = "h_bar") -> Path:
    return ensure_dir(chart_result_dir(chart_id, chart_type) / "tempy")


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
        filename = f"final_overlay_{safe_point_name}_{prompt_type}_{image_type}.png"
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


def _series_adjusted_category_window(
    *,
    y_label: str,
    y_ticks: list[Any],
    y_pixels: list[int],
    image_size: tuple[int, int],
    series_name: str | None = None,
    series_order: list[str] | None = None,
) -> tuple[int, int]:
    center_y = category_pixel(y_label, y_ticks, y_pixels)
    span_y = category_span(y_label, y_ticks, y_pixels, image_size)
    order = [str(item) for item in (series_order or []) if str(item).strip()]
    if series_name and len(order) > 1 and str(series_name) in order:
        slot = max(8.0, float(span_y) / len(order))
        index = order.index(str(series_name))
        center_y = int(round(center_y + ((len(order) - 1) / 2.0 - index) * slot))
        half_y = max(12, int(round(slot / 2.0 + 7)))
        return center_y, half_y
    return center_y, max(18, span_y // 2 + 8)


def _extend_pairs_to_image_bounds(pairs: list[tuple[float, float]], width: int) -> list[tuple[float, float]]:
    if len(pairs) < 2:
        return pairs
    extended = list(pairs)
    v0, p0 = extended[0]
    v1, p1 = extended[1]
    if v1 != v0 and p1 != p0:
        step_v = v1 - v0
        step_p = p1 - p0
        while extended[0][1] - step_p >= 0:
            first_v, first_p = extended[0]
            extended.insert(0, (first_v - step_v, first_p - step_p))
    v0, p0 = extended[-2]
    v1, p1 = extended[-1]
    if v1 != v0 and p1 != p0:
        step_v = v1 - v0
        step_p = p1 - p0
        while extended[-1][1] + step_p <= width:
            last_v, last_p = extended[-1]
            extended.append((last_v + step_v, last_p + step_p))
    return extended


def _dense_ticks_for_crop(
    pairs: list[tuple[float, float]],
    left_px: int,
    right_px: int,
    round_index: int = 1,
) -> tuple[list[float], list[float]]:
    dense_ticks: list[float] = []
    dense_pixels: list[float] = []
    preferred = amplifier_tick_divisor(round_index)
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
    zoom_round_index: int | None = None,
    roi_scale: float = 1.0,
    attempt_index: int = 0,
    pad_x: int | None = None,
    pad_y: int | None = None,
    chart_type: str = "h_bar",
    series_name: str | None = None,
    series_order: list[str] | None = None,
) -> tuple[Path, list[float], tuple[float, float]]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    pairs = _extend_pairs_to_image_bounds(_numeric_pairs(x_ticks, x_pixels), width)
    if len(pairs) < 2:
        raise ValueError("At least two numeric x ticks are required for h_bar amplifier crop.")
    min_value = min(value for value, _ in pairs)
    max_value = max(value for value, _ in pairs)
    clamped_value = min(max(float(center_value), min_value), max_value)
    center_x = numeric_pixel(clamped_value, x_ticks, x_pixels)
    lane_center_y, default_half_y = _series_adjusted_category_window(
        y_label=y_label,
        y_ticks=y_ticks,
        y_pixels=y_pixels,
        image_size=img.size,
        series_name=series_name,
        series_order=series_order,
    )
    group_center_y = category_pixel(y_label, y_ticks, y_pixels)
    group_half_y = max(18, category_span(y_label, y_ticks, y_pixels, img.size) // 2 + 8)
    center_y = group_center_y if series_name and series_order and len(series_order) > 1 else lane_center_y
    half_y = max(12, (pad_y if pad_y is not None else default_half_y))
    if series_name and series_order and len(series_order) > 1 and pad_y is None:
        half_y = group_half_y

    pixel_gaps = [abs(right - left) for (_, left), (_, right) in zip(pairs, pairs[1:])]
    base_span = sorted(pixel_gaps)[len(pixel_gaps) // 2] if pixel_gaps else 90
    zoom_round = zoom_round_index if zoom_round_index is not None else round_index
    base_window = pad_x if pad_x is not None else base_span

    pair_pixels = [pixel for _, pixel in pairs]
    plot_left = int(max(0, min(pair_pixels)))
    plot_right = int(min(width, max(pair_pixels)))
    half_x = amplifier_half_window(
        base_window,
        zoom_round,
        roi_scale=roi_scale,
        min_px=36,
        plot_span_px=max(1, plot_right - plot_left),
    )
    left = max(plot_left, int(center_x - half_x))
    right = min(plot_right, int(center_x + half_x))
    if right <= left:
        right = min(plot_right, left + 12)
    top = max(0, center_y - half_y)
    bottom = min(height, center_y + half_y)

    raw_crop = img.crop((left, top, right, bottom))
    crop_w, crop_h = raw_crop.size
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Invalid crop area: {(left, top, right, bottom)}")

    out_size = amplifier_output_side(crop_w, crop_h)
    label_pad = amplifier_label_pad()
    scale = min(out_size / max(1, crop_w), out_size / max(1, crop_h))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    resized = raw_crop.resize((new_w, new_h), Image.NEAREST)

    canvas = Image.new("RGB", (out_size, out_size + label_pad * 2), "white")
    offset_x = (out_size - new_w) // 2
    offset_y = label_pad + (out_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    font = _font(18)

    crop_top = offset_y
    crop_bottom = offset_y + new_h
    draw.text((offset_x + 4, crop_top + 4), f"R{round_index}", font=font, fill="black")
    if series_name and series_order and len(series_order) > 1:
        lane_y = offset_y + int(round(((lane_center_y - top) / crop_h) * new_h))
        lane_half = max(5, int(round((default_half_y / max(1, crop_h)) * new_h)))
        lane_top = max(crop_top, lane_y - lane_half)
        lane_bottom = min(crop_bottom, lane_y + lane_half)
        draw.line((offset_x, lane_y, offset_x + new_w, lane_y), fill=(255, 0, 0), width=2)
        draw.rectangle((offset_x, lane_top, offset_x + new_w, lane_bottom), outline=(255, 0, 0), width=2)
        target_label = f"target lane: {series_name}"
        label_bbox = draw.textbbox((0, 0), target_label, font=font)
        label_w = label_bbox[2] - label_bbox[0] + 8
        label_h = label_bbox[3] - label_bbox[1] + 6
        lx = min(max(offset_x + 4, 0), max(0, canvas.width - label_w - 2))
        ly = max(2, crop_top - label_h - 4)
        draw.rectangle((lx, ly, lx + label_w, ly + label_h), fill="white", outline=(255, 0, 0), width=1)
        draw.text((lx + 4, ly + 3), target_label, font=font, fill=(255, 0, 0))

    dense_ticks, dense_pixels = _dense_ticks_for_crop(pairs, left, right, round_index=round_index)
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
    dash_len = AMPLIFIER_BAR_DASH_LENGTH
    dash_gap = AMPLIFIER_BAR_DASH_GAP
    tick_len = 6
    drawn_ticks: list[tuple[float, int]] = []
    min_grid_gap = 18
    for tick, tick_x in mapped_ticks:
        if drawn_ticks and abs(tick_x - drawn_ticks[-1][1]) < min_grid_gap:
            continue
        drawn_ticks.append((tick, tick_x))

    for tick, tick_x in drawn_ticks:
        y = crop_top
        while y < crop_bottom:
            y_end = min(y + dash_len, crop_bottom)
            draw.line((tick_x, y, tick_x, y_end), fill=AMPLIFIER_GRID_COLOR, width=1)
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

    labeled_ticks: list[tuple[float, int]] = []
    min_label_gap = max(28, int(font.size * 1.65))
    for tick, tick_x in drawn_ticks:
        if not labeled_ticks or abs(tick_x - labeled_ticks[-1][1]) >= min_label_gap:
            labeled_ticks.append((tick, tick_x))

    for tick, tick_x in labeled_ticks:
        label_center_y = crop_bottom + dash_region + tick_len + 25
        draw_amplifier_dashed_line(
            draw,
            (tick_x, crop_bottom),
            (tick_x, label_center_y + 24),
            fill=AMPLIFIER_GRID_COLOR,
            width=1,
            dash_length=dash_len,
            gap_length=dash_gap,
        )
        draw_rotated_centered_label_box(
            canvas,
            _format_tick(tick),
            (tick_x, label_center_y),
            -90,
            font=font,
            fill=(0, 0, 0, 255),
        )

    draw.line((offset_x, crop_top, offset_x + new_w, crop_top), fill="black", width=2)
    draw.line((offset_x, crop_bottom, offset_x + new_w, crop_bottom), fill="black", width=2)

    min_val, max_val = value_range_from_pixels(left, right, x_ticks, x_pixels)
    visible_ticks = [tick for tick, _ in labeled_ticks] or [tick for tick, _ in drawn_ticks] or visible_ticks
    if not visible_ticks:
        visible_ticks = visible_ticks_for_range(x_ticks, min_val, max_val)

    output = temp_dir(chart_id, chart_type) / (
        f"amplifier_crop_{safe_filename(point_name)}_round{round_index}_attempt{attempt_index}.png"
    )
    canvas.save(output)
    return output, visible_ticks, (min_val, max_val)
