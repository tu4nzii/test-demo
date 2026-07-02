"""Visual feedback and crop helpers for radar/rose GT experiments."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..common.amplifier_style import (
    AMPLIFIER_GRID_COLOR_RGBA,
    AMPLIFIER_GUIDE_COLOR_RGBA,
    draw_centered_label_box,
)
from ..common.chart_io import ensure_dir


def draw_polar_feedback(
    *,
    dataset: dict[str, Any],
    chart_type: str,
    source_image: Path,
    result_dir: Path,
    point_name: str,
    theta_label: str,
    pred_r: float,
    round_index: int,
) -> Path:
    center, radius = polar_geometry(dataset, source_image)
    angle = rose_theta_angle(dataset, theta_label) if str(chart_type).lower() == "rose" else theta_angle(dataset, theta_label)
    if str(chart_type).lower() == "rose":
        angle = (angle + rose_sector_center_offset(dataset)) % 360.0
    px, py = polar_point(center, radius, dataset.get("r_ticks"), pred_r, angle, dataset.get("r_pixels"))

    out_dir = ensure_dir(result_dir / "feedback_img")
    out_path = out_dir / f"{safe_name(point_name)}_feedback_round{round_index}.png"
    with Image.open(source_image).convert("RGBA") as base:
        img = base.copy()
    draw = ImageDraw.Draw(img)
    draw.line([center, (px, py)], fill=AMPLIFIER_GUIDE_COLOR_RGBA, width=3)
    marker = max(5, int(radius * 0.025))
    draw.ellipse((px - marker, py - marker, px + marker, py + marker), outline=AMPLIFIER_GUIDE_COLOR_RGBA, width=3)
    draw.line((px - marker * 2, py, px + marker * 2, py), fill=AMPLIFIER_GUIDE_COLOR_RGBA, width=2)
    draw.line((px, py - marker * 2, px, py + marker * 2), fill=AMPLIFIER_GUIDE_COLOR_RGBA, width=2)
    label = f"pred r={pred_r:g}"
    try:
        font = ImageFont.load_default()
        draw.text((px + marker + 4, py + marker + 4), label, fill=AMPLIFIER_GUIDE_COLOR_RGBA, font=font)
    except Exception:
        pass
    img.save(out_path)
    return out_path


def crop_polar_amplifier(
    *,
    dataset: dict[str, Any],
    chart_type: str,
    source_image: Path,
    result_dir: Path,
    point_name: str,
    theta_label: str,
    pred_r: float,
    round_index: int,
    output_size: int | None = None,
) -> tuple[Path, list[float]]:
    center, radius = polar_geometry(dataset, source_image)
    angle = rose_theta_angle(dataset, theta_label) if str(chart_type).lower() == "rose" else theta_angle(dataset, theta_label)
    if str(chart_type).lower() == "rose":
        angle = (angle + rose_sector_center_offset(dataset)) % 360.0
    px, py = polar_point(center, radius, dataset.get("r_ticks"), pred_r, angle, dataset.get("r_pixels"))

    out_dir = ensure_dir(result_dir / "amplifier_img")
    out_path = out_dir / f"{safe_name(point_name)}_amp{round_index}.png"
    guide_ticks = local_visible_ticks(dataset.get("r_ticks"), pred_r, round_index)

    with Image.open(source_image).convert("RGBA") as base:
        width, height = base.size
        left, top, right, bottom = _polar_roi_bounds(
            dataset=dataset,
            chart_type=chart_type,
            center=center,
            radius=radius,
            angle=angle,
            pred_r=pred_r,
            width=width,
            height=height,
        )
        crop_w = max(1, right - left)
        crop_h = max(1, bottom - top)
        if output_size is None:
            output_w = max(1, crop_w * 3)
            output_h = max(1, crop_h * 3)
        else:
            output_w = output_h = int(output_size)
        crop = base.crop((left, top, right, bottom)).resize((output_w, output_h), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(crop)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    sx = output_w / max(1, right - left)
    sy = output_h / max(1, bottom - top)
    local_center = ((center[0] - left) * sx, (center[1] - top) * sy)
    local_point = ((px - left) * sx, (py - top) * sy)
    local_scale = (sx + sy) / 2
    local_radius = radius * local_scale

    if str(chart_type).lower() == "rose":
        half_step = rose_sector_center_offset(dataset)
        _fade_outside_rose_sector(crop, center=local_center, angle=angle, half_step=half_step)

    for tick in guide_ticks:
        source_tick_radius = radius_for_value(dataset.get("r_ticks"), tick, radius, dataset.get("r_pixels"))
        tick_radius = source_tick_radius * local_scale
        bbox = (
            local_center[0] - tick_radius,
            local_center[1] - tick_radius,
            local_center[0] + tick_radius,
            local_center[1] + tick_radius,
        )
        draw.ellipse(bbox, outline=AMPLIFIER_GRID_COLOR_RGBA[:3] + (160,), width=1)
        label_point = _visible_ring_label_point(
            center=local_center,
            radius=tick_radius,
            preferred_angle=angle,
            width=output_w,
            height=output_h,
            avoid=local_point,
        )
        if label_point is not None:
            draw_centered_label_box(
                draw,
                _format_tick(tick),
                label_point,
                font=font,
                fill=AMPLIFIER_GRID_COLOR_RGBA,
            )
    end = _polar_offset(local_center, local_radius, angle)
    draw.line([local_center, end], fill=AMPLIFIER_GRID_COLOR_RGBA[:3] + (180,), width=2)
    if str(chart_type).lower() == "rose":
        half_step = rose_sector_center_offset(dataset)
        for boundary_angle in (angle - half_step, angle + half_step):
            boundary_end = _polar_offset(local_center, local_radius, boundary_angle)
            draw.line([local_center, boundary_end], fill=AMPLIFIER_GUIDE_COLOR_RGBA[:3] + (150,), width=2)
        label_text = f"target sector: {theta_label}"
        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        label_w = label_bbox[2] - label_bbox[0] + 10
        label_h = label_bbox[3] - label_bbox[1] + 8
        draw.rectangle((6, 6, 6 + label_w, 6 + label_h), fill="white", outline=AMPLIFIER_GUIDE_COLOR_RGBA, width=1)
        draw.text((11, 10), label_text, fill=AMPLIFIER_GUIDE_COLOR_RGBA, font=font)
    marker = 7
    draw.ellipse(
        (local_point[0] - marker, local_point[1] - marker, local_point[0] + marker, local_point[1] + marker),
        outline=AMPLIFIER_GRID_COLOR_RGBA,
        width=3,
    )
    crop.save(out_path)
    return out_path, guide_ticks


def add_target_color_swatch(
    *,
    image_path: Path,
    output_path: Path | None = None,
    point_name: str,
    color: str | None,
) -> Path:
    if not color:
        return image_path
    out_path = output_path or image_path
    ensure_dir(out_path.parent)
    try:
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            box_w = min(170, max(110, width // 2))
            box_h = 30
            x0 = 6
            y0 = max(6, height - box_h - 6)
            draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill="white", outline="black", width=1)
            draw.rectangle((x0 + 5, y0 + 5, x0 + 29, y0 + 25), fill=str(color), outline="black", width=1)
            draw.text((x0 + 36, y0 + 8), f"target: {point_name}", fill="black")
            image.save(out_path)
        return out_path
    except Exception as exc:
        print(f"[polar visual] Target color swatch skipped for {point_name}: {exc}")
        return image_path


def theta_angle_rad(angle: float) -> float:
    return math.radians(angle)


def _format_tick(value: float) -> str:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def polar_geometry(dataset: dict[str, Any], source_image: Path) -> tuple[tuple[int, int], int]:
    with Image.open(source_image) as img:
        width, height = img.size
    raw_center = dataset.get("center")
    if isinstance(raw_center, dict):
        center = (int(raw_center.get("x", width // 2)), int(raw_center.get("y", height // 2)))
    elif isinstance(raw_center, (list, tuple)) and len(raw_center) >= 2:
        center = (int(raw_center[0]), int(raw_center[1]))
    else:
        center = (width // 2, height // 2)

    raw_radius = dataset.get("r_pixels") or dataset.get("radius")
    if isinstance(raw_radius, (list, tuple)):
        values = [float(value) for value in raw_radius if _number_or_none(value) is not None]
        radius = int(max(values)) if values else int(min(width, height) * 0.36)
    else:
        radius = int(_number_or_none(raw_radius) or min(width, height) * 0.36)
    return center, max(1, radius)


def theta_angle(dataset: dict[str, Any], theta_label: str) -> float:
    labels = [str(item) for item in dataset.get("theta_ticks", [])] if isinstance(dataset.get("theta_ticks"), list) else []
    raw_angles = dataset.get("theta_angles") or dataset.get("axes_angles")
    if isinstance(raw_angles, list) and labels:
        for index, label in enumerate(labels):
            if _same(label, theta_label) and index < len(raw_angles):
                number = _number_or_none(raw_angles[index])
                if number is not None:
                    return number
    axis_labels = dataset.get("axis_labels")
    if isinstance(axis_labels, dict):
        for angle, label in axis_labels.items():
            if _same(label, theta_label):
                number = _number_or_none(angle)
                if number is not None:
                    return number
    if labels:
        for index, label in enumerate(labels):
            if _same(label, theta_label):
                return 360.0 * index / len(labels)
    return 0.0


def rose_theta_angle(dataset: dict[str, Any], theta_label: str) -> float:
    labels = [str(item) for item in dataset.get("theta_ticks", [])] if isinstance(dataset.get("theta_ticks"), list) else []
    raw_angles = dataset.get("theta_angles") or dataset.get("axes_angles")
    if isinstance(raw_angles, list) and labels:
        descending = len(raw_angles) >= 2 and all(
            (_number_or_none(raw_angles[i]) or 0.0) > (_number_or_none(raw_angles[i + 1]) or -1.0)
            for i in range(min(len(raw_angles) - 1, 3))
        )
        for index, label in enumerate(labels):
            if _same(label, theta_label) and index < len(raw_angles):
                number = _number_or_none(raw_angles[index])
                if number is None:
                    break
                return (90.0 - number) % 360.0 if descending else number
    return theta_angle(dataset, theta_label)


def rose_sector_center_offset(dataset: dict[str, Any]) -> float:
    raw_angles = dataset.get("theta_angles") or dataset.get("axes_angles")
    angles = sorted(
        value % 360.0
        for value in (_number_or_none(item) for item in raw_angles or [])
        if value is not None
    )
    if len(angles) >= 2:
        gaps = []
        for index, value in enumerate(angles):
            nxt = angles[(index + 1) % len(angles)]
            gaps.append((nxt - value) % 360.0 or 360.0)
        return min(gaps) / 2.0
    labels = dataset.get("theta_ticks")
    count = len(labels) if isinstance(labels, list) and labels else 8
    return 180.0 / max(1, count)


def polar_point(
    center: tuple[int, int],
    radius: int,
    ticks: Any,
    value: float,
    angle: float,
    r_pixels: Any = None,
) -> tuple[float, float]:
    r = radius_for_value(ticks, value, radius, r_pixels)
    return _polar_offset(center, r, angle)


def radius_for_value(ticks: Any, value: float, radius: float, r_pixels: Any = None) -> float:
    tick_values = [_number_or_none(item) for item in ticks] if isinstance(ticks, list) else []
    tick_values = [item for item in tick_values if item is not None]
    pixel_values = [_number_or_none(item) for item in r_pixels] if isinstance(r_pixels, list) else []
    pixel_values = [item for item in pixel_values if item is not None]
    if len(tick_values) >= 2 and len(pixel_values) >= len(tick_values):
        pairs = sorted(zip(tick_values, pixel_values[: len(tick_values)]), key=lambda item: item[0])
        return _interpolate_pairs(pairs, float(value))
    if len(tick_values) >= 2:
        low, high = min(tick_values), max(tick_values)
    else:
        low, high = 0.0, max(1.0, abs(value))
    ratio = (float(value) - low) / max(high - low, 1e-9)
    ratio = max(0.0, min(1.15, ratio))
    return ratio * radius


def local_visible_ticks(ticks: Any, center_value: float, round_index: int) -> list[float]:
    values = [_number_or_none(item) for item in ticks] if isinstance(ticks, list) else []
    values = sorted(item for item in values if item is not None)
    if not values:
        return []
    dense: set[float] = set(values)
    for left, right in zip(values, values[1:]):
        dense.add(round((left + right) / 2.0, 4))
    ordered = sorted(dense, key=lambda item: abs(item - center_value))
    keep = 9
    return sorted(ordered[:keep])


def _visible_ring_label_point(
    *,
    center: tuple[float, float],
    radius: float,
    preferred_angle: float,
    width: int,
    height: int,
    avoid: tuple[float, float],
) -> tuple[float, float] | None:
    """Place a tick label on a visible part of its ring after crop resizing."""
    margin = 36.0
    candidates: list[tuple[float, float]] = []
    offsets = (0, 12, -12, 24, -24, 36, -36, 60, -60, 90, -90, 120, -120, 180)
    for offset in offsets:
        point = _polar_offset(center, radius, preferred_angle + offset)
        x, y = point
        if margin <= x <= width - margin and margin <= y <= height - margin:
            candidates.append((x, y))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0] - avoid[0]) ** 2 + (item[1] - avoid[1]) ** 2)


def _polar_roi_bounds(
    *,
    dataset: dict[str, Any],
    chart_type: str,
    center: tuple[int, int],
    radius: int,
    angle: float,
    pred_r: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    ticks = sorted(value for value in (_number_or_none(item) for item in dataset.get("r_ticks", [])) if value is not None)
    r_pixels = dataset.get("r_pixels")
    seed_radius = radius_for_value(ticks, pred_r, radius, r_pixels)
    nearby_tick_radii = [radius_for_value(ticks, tick, radius, r_pixels) for tick in ticks] if ticks else []
    if nearby_tick_radii:
        spacing = min(
            (abs(right - left) for left, right in zip(nearby_tick_radii, nearby_tick_radii[1:]) if abs(right - left) > 0),
            default=max(radius * 0.12, 24),
        )
    else:
        spacing = max(radius * 0.12, 24)
    radial_pad = max(24.0, spacing * 0.9)
    angle_pad = _angle_padding(dataset, chart_type)

    if str(chart_type).lower() in {"radar", "rose"}:
        min_r = 0.0
        max_r = min(radius * 1.08, max(radius * 1.02, seed_radius + radial_pad))
    else:
        min_r = max(0.0, seed_radius - radial_pad)
        max_r = min(radius * 1.08, seed_radius + radial_pad)
    points = [_polar_offset(center, r, theta) for r in (min_r, seed_radius, max_r) for theta in (angle - angle_pad, angle, angle + angle_pad)]
    points.append(_polar_offset(center, seed_radius, angle))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    margin = max(18.0, spacing * 0.35)
    left = max(0, int(math.floor(min(xs) - margin)))
    top = max(0, int(math.floor(min(ys) - margin)))
    right = min(width, int(math.ceil(max(xs) + margin)))
    bottom = min(height, int(math.ceil(max(ys) + margin)))
    min_side = int(max(150, min(360, spacing * 2.2)))
    if right - left < min_side:
        cx = (left + right) / 2
        left = max(0, int(cx - min_side / 2))
        right = min(width, left + min_side)
        left = max(0, right - min_side)
    if bottom - top < min_side:
        cy = (top + bottom) / 2
        top = max(0, int(cy - min_side / 2))
        bottom = min(height, top + min_side)
        top = max(0, bottom - min_side)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _fade_outside_rose_sector(
    image: Image.Image,
    *,
    center: tuple[float, float],
    angle: float,
    half_step: float,
) -> None:
    pixels = image.load()
    width, height = image.size
    cx, cy = center
    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            if dx == 0 and dy == 0:
                continue
            local_angle = (math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0
            if _angle_distance(local_angle, angle) <= half_step:
                continue
            r, g, b, a = pixels[x, y]
            pixels[x, y] = (
                int(r * 0.35 + 255 * 0.65),
                int(g * 0.35 + 255 * 0.65),
                int(b * 0.35 + 255 * 0.65),
                a,
            )


def _angle_distance(left: float, right: float) -> float:
    diff = abs((float(left) - float(right)) % 360.0)
    return min(diff, 360.0 - diff)


def _angle_padding(dataset: dict[str, Any], chart_type: str) -> float:
    raw_angles = [_number_or_none(item) for item in (dataset.get("theta_angles") or dataset.get("axes_angles") or [])]
    angles = sorted(item % 360.0 for item in raw_angles if item is not None)
    if len(angles) >= 2:
        gaps = []
        for index, value in enumerate(angles):
            nxt = angles[(index + 1) % len(angles)]
            gaps.append((nxt - value) % 360.0 or 360.0)
        step = min(gaps)
    else:
        labels = dataset.get("theta_ticks")
        step = 360.0 / max(1, len(labels) if isinstance(labels, list) else 8)
    if chart_type == "rose":
        return min(28.0, max(10.0, step * 0.45))
    return min(16.0, max(6.0, step * 0.22))


def _polar_offset(center: tuple[float, float], radius: float, angle: float) -> tuple[float, float]:
    theta = math.radians(float(angle) - 90.0)
    return center[0] + radius * math.cos(theta), center[1] + radius * math.sin(theta)


def _interpolate_pairs(pairs: list[tuple[float, float]], value: float) -> float:
    if len(pairs) < 2:
        return pairs[0][1] if pairs else 0.0
    if value <= pairs[0][0]:
        left, right = pairs[0], pairs[1]
    elif value >= pairs[-1][0]:
        left, right = pairs[-2], pairs[-1]
    else:
        for left, right in zip(pairs, pairs[1:]):
            if left[0] <= value <= right[0]:
                break
        else:
            left, right = pairs[0], pairs[-1]
    denom = right[0] - left[0]
    if abs(denom) < 1e-9:
        return float(left[1])
    alpha = (value - left[0]) / denom
    return float(left[1] + (right[1] - left[1]) * alpha)


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "polar_object"


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number else None
    except Exception:
        return None


def _same(left: Any, right: Any) -> bool:
    return str(left or "").strip().casefold() == str(right or "").strip().casefold()
