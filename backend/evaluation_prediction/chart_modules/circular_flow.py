"""Shared circular-chart flow calibration for pie and donut experiments."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..common.amplifier_style import amplifier_max_side, draw_centered_label_box
from ..common.chart_io import ensure_dir


ANGLE_STOP_TOLERANCE_DEG = 2.0


def amplifier_round_config(round_index: int) -> dict[str, float | int]:
    if int(round_index) <= 1:
        return {"padding_deg": 15.0, "grid_interval_deg": 5, "zoom_scale": 2.0}
    return {"padding_deg": 8.0, "grid_interval_deg": 2, "zoom_scale": 2.0}


def angle_prediction_stable(
    current: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    *,
    tolerance_deg: float = ANGLE_STOP_TOLERANCE_DEG,
) -> bool:
    if not isinstance(current, dict) or not isinstance(reference, dict):
        return False
    current_start = _number_or_none(current.get("start_angle"))
    current_end = _number_or_none(current.get("end_angle"))
    reference_start = _number_or_none(reference.get("start_angle"))
    reference_end = _number_or_none(reference.get("end_angle"))
    if None in {current_start, current_end, reference_start, reference_end}:
        return False
    return (
        circular_angle_distance(float(current_start), float(reference_start)) <= tolerance_deg
        and circular_angle_distance(float(current_end), float(reference_end)) <= tolerance_deg
    )


def circular_angle_distance(a: float, b: float) -> float:
    diff = abs((float(a) - float(b)) % 360.0)
    return min(diff, 360.0 - diff)


def draw_circular_grid_image(
    *,
    source: Path,
    output_path: Path,
    center: tuple[int, int],
    outer_radius: int,
    inner_radius: int = 0,
    interval_deg: int = 15,
) -> Path:
    ensure_dir(output_path.parent)
    with Image.open(source).convert("RGBA") as base:
        img = base.copy()
    draw = ImageDraw.Draw(img)
    font = _font(14 if max(img.size) < 1000 else 18)
    cx, cy = center
    r0 = max(0, int(inner_radius))
    r1 = max(1, int(outer_radius))
    label_radius = r1 + max(12, int(r1 * 0.08))

    for angle in range(0, 360, int(interval_deg)):
        theta = math.radians(angle - 90.0)
        x0 = cx + r0 * math.cos(theta)
        y0 = cy + r0 * math.sin(theta)
        x1 = cx + label_radius * math.cos(theta)
        y1 = cy + label_radius * math.sin(theta)
        color = (0, 0, 0, 190) if angle else (255, 0, 0, 230)
        width = 1 if angle else 2
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)

        label = f"{angle}{chr(176)}"
        tx = cx + label_radius * math.cos(theta)
        ty = cy + label_radius * math.sin(theta)
        draw_centered_label_box(
            draw,
            label,
            (tx, ty),
            font=font,
            fill=(0, 0, 0, 255),
            background=(255, 255, 255, 210),
        )

    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0, 255))
    img.convert("RGB").save(output_path)
    return output_path


def crop_circular_amplifier_image(
    *,
    source: Path,
    output_path: Path,
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
    start_angle: float,
    end_angle: float,
    round_index: int,
) -> tuple[Path, list[int]]:
    cfg = amplifier_round_config(round_index)
    padding_deg = float(cfg["padding_deg"])
    interval_deg = int(cfg["grid_interval_deg"])
    zoom_scale = float(cfg["zoom_scale"])
    ensure_dir(output_path.parent)

    with Image.open(source).convert("RGBA") as base:
        img = base.copy()
    width, height = img.size
    cx, cy = center
    start_ext = (float(start_angle) - padding_deg) % 360.0
    end_ext = (float(end_angle) + padding_deg) % 360.0
    span = (end_ext - start_ext) % 360.0
    raw_span = (float(end_angle) - float(start_angle)) % 360.0
    if raw_span >= 330.0 or span >= 359.5:
        start_ext = 0.0
        span = 360.0

    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if span >= 359.5:
        mask_draw.ellipse((cx - outer_radius, cy - outer_radius, cx + outer_radius, cy + outer_radius), fill=255)
    else:
        points = [(cx, cy)]
        steps = max(16, int(math.ceil(span)))
        for index in range(steps + 1):
            angle = (start_ext + span * index / steps) % 360.0
            theta = math.radians(angle - 90.0)
            points.append((cx + outer_radius * math.cos(theta), cy + outer_radius * math.sin(theta)))
        mask_draw.polygon(points, fill=255)

    if inner_radius > 0:
        mask_draw.ellipse((cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius), fill=0)

    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Empty circular amplifier crop mask.")
    extra = max(8, int(outer_radius * 0.12))
    left = max(0, bbox[0] - extra)
    top = max(0, bbox[1] - extra)
    right = min(width, bbox[2] + extra)
    bottom = min(height, bbox[3] + extra)

    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    white.paste(img, (0, 0), mask)
    crop = white.crop((left, top, right, bottom))
    crop = crop.resize(
        (max(1, int(round(crop.width * zoom_scale))), max(1, int(round(crop.height * zoom_scale)))),
        Image.Resampling.BICUBIC,
    )
    local_center = (int(round((cx - left) * zoom_scale)), int(round((cy - top) * zoom_scale)))
    local_outer = int(round(outer_radius * zoom_scale))
    local_inner = int(round(inner_radius * zoom_scale))

    max_side = amplifier_max_side()
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize(
            (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
            Image.Resampling.BICUBIC,
        )
        local_center = (int(round(local_center[0] * scale)), int(round(local_center[1] * scale)))
        local_outer = int(round(local_outer * scale))
        local_inner = int(round(local_inner * scale))

    drawn_angles = _draw_local_angle_grid(
        crop,
        center=local_center,
        inner_radius=local_inner,
        outer_radius=local_outer,
        start_angle=start_ext,
        span=span,
        interval_deg=interval_deg,
    )
    crop.convert("RGB").save(output_path)
    return output_path, drawn_angles


def _draw_local_angle_grid(
    image: Image.Image,
    *,
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
    start_angle: float,
    span: float,
    interval_deg: int,
) -> list[int]:
    draw = ImageDraw.Draw(image)
    font = _font(12 if max(image.size) < 700 else 15)
    cx, cy = center
    drawn: list[int] = []
    start = float(start_angle)
    end = start + float(span)
    for base in range(0, 720, int(interval_deg)):
        angle = base % 360
        logical = float(base)
        if logical < start:
            logical += 360.0
        if not (start <= logical <= end):
            continue
        drawn.append(angle)
        theta = math.radians(angle - 90.0)
        r0 = max(0, inner_radius)
        r1 = max(1, outer_radius + max(8, int(outer_radius * 0.08)))
        label_radius = r1 + 8
        x0 = cx + r0 * math.cos(theta)
        y0 = cy + r0 * math.sin(theta)
        x1 = cx + label_radius * math.cos(theta)
        y1 = cy + label_radius * math.sin(theta)
        color = (255, 0, 0, 255) if angle == 0 else (0, 0, 0, 255)
        width = 2 if angle == 0 else 1
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
        label = f"{angle}{chr(176)}"
        tx = cx + label_radius * math.cos(theta)
        ty = cy + label_radius * math.sin(theta)
        draw_centered_label_box(
            draw,
            label,
            (tx, ty),
            font=font,
            fill=(0, 0, 0, 255),
        )
    return drawn


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
