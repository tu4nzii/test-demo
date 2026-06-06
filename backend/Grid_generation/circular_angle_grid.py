"""Angle-grid generation for pie and donut charts in the backend.

This module is adapted from the original pie/donut angle-grid helper and is
kept independent for backend use.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from function_calling.color.extract_chart_colors import extract_chart_series_color
except Exception:  # pragma: no cover - keeps angle-grid generation usable in isolation.
    try:
        from Grid_generation.function_calling.color.extract_chart_colors import extract_chart_series_color
    except Exception:
        extract_chart_series_color = None


ANGLE_STEP_DEGREES = 15
ANGLE_TICKS = list(range(0, 360, ANGLE_STEP_DEGREES))


def _load_font(img_width: int, img_height: int) -> ImageFont.ImageFont:
    font_size = 20 if img_width >= 1000 or img_height >= 1000 else 12
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _clamp(value: float, lower: float, upper: float) -> float:
    if upper < lower:
        return lower
    return max(lower, min(upper, value))


def draw_angle_grid_30deg(
    cfg: dict,
    img_type: str,
    output_suffix: str,
    inner_radius: int,
    line_color: tuple = (0, 0, 0, 255),
    line_width: int = 1,
    font_size: int = 8,
    text_offset: int = 15,
    grid_line_ratio: float = 0.1,
    text_offset_ratio: float = 0.1,
    output_path: Optional[str] = None,
) -> str:
    """Generate the clockwise angle-grid image used as the with_grid input."""
    del font_size, text_offset

    src_path = cfg["image_paths"][img_type]
    img = Image.open(src_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    cx, cy = cfg["center"]
    img_width, img_height = img.size
    font = _load_font(img_width, img_height)

    grid_line_length = max(8, int(inner_radius * grid_line_ratio))
    radius_text = inner_radius + grid_line_length + int(inner_radius * text_offset_ratio)

    labels = []
    for angle_deg in ANGLE_TICKS:
        theta = math.radians(angle_deg - 90)
        cos_v = math.cos(theta)
        sin_v = math.sin(theta)

        x_start = cx + inner_radius * cos_v
        y_start = cy + inner_radius * sin_v
        x_end = cx + (inner_radius + grid_line_length) * cos_v
        y_end = cy + (inner_radius + grid_line_length) * sin_v
        draw.line([(x_start, y_start), (x_end, y_end)], fill=line_color, width=line_width)

        label = f"{angle_deg}{chr(176)}"
        text_w, text_h = _text_size(font, label)
        tx = cx + radius_text * cos_v
        ty = cy + radius_text * sin_v
        pad = 4
        tx = _clamp(tx, text_w / 2 + pad, img_width - text_w / 2 - pad)
        ty = _clamp(ty, text_h / 2 + pad, img_height - text_h / 2 - pad)
        labels.append((tx, ty, label, text_w, text_h))

    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0, 255))

    for tx, ty, label, text_w, text_h in labels:
        draw.text(
            (tx - text_w / 2, ty - text_h / 2),
            label,
            fill=(0, 0, 0, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )

    if output_path is None:
        base, ext = os.path.splitext(src_path)
        output_path = f"{base}{output_suffix}{ext}"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output_path)
    return output_path


def detect_circular_plot_area(image_path: str) -> Dict[str, Any]:
    """Estimate the chart center and outer radius from colored pie/donut marks."""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = np.where((saturation > 12) & (value > 40), 255, 0).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(64, int(width * height * 0.003))
    candidates = []
    for index in range(1, components):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_area:
            candidates.append((index, area))

    if not candidates:
        return _detect_circular_plot_area_hough(image, image_path)

    largest_area = max(area for _, area in candidates)
    point_sets = []
    for index, area in candidates:
        if area < max(min_area, int(largest_area * 0.08)):
            continue
        ys, xs = np.where(labels == index)
        point_sets.append(np.column_stack([xs, ys]).astype("float32"))

    if not point_sets:
        return _detect_circular_plot_area_hough(image, image_path)

    points = np.vstack(point_sets)
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    radius = min(radius, cx, cy, width - cx - 1, height - cy - 1) if radius > min(width, height) else radius

    return {
        "center": [int(round(cx)), int(round(cy))],
        "r_pixels": int(round(radius)),
        "detection_method": "color_mask",
    }


def _detect_circular_plot_area_hough(image: np.ndarray, image_path: str) -> Dict[str, Any]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(width, height) // 3,
        param1=50,
        param2=30,
        minRadius=max(10, min(width, height) // 10),
        maxRadius=max(20, min(width, height) // 2),
    )

    if circles is None:
        raise ValueError(f"Unable to detect circular plot area: {image_path}")

    circle = np.round(circles[0, 0]).astype(int)
    return {
        "center": [int(circle[0]), int(circle[1])],
        "r_pixels": int(circle[2]),
        "detection_method": "hough",
    }


def _angle_labels(angles: Iterable[int]) -> list[str]:
    return [f"{angle}{chr(176)}" for angle in angles]


def process_circular_angle_chart(
    image_path: str,
    output_dir: str,
    chart_type: str,
    chart_id_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a pie/donut with_grid image and sidecar JSON for backend routes."""
    chart_id = chart_id_override or Path(image_path).stem
    output_path = os.path.join(output_dir, f"{chart_id}_with_grid.png")
    plot_area = detect_circular_plot_area(image_path)

    cfg = {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "image_paths": {"no_grid": image_path},
        "center": plot_area["center"],
        "r_pixels": plot_area["r_pixels"],
    }
    encrypted_grid_path = draw_angle_grid_30deg(
        cfg,
        img_type="no_grid",
        output_suffix="_with_grid",
        inner_radius=int(plot_area["r_pixels"]),
        output_path=output_path,
        line_color=(0, 0, 0, 255),
    )
    colors = []
    if extract_chart_series_color is not None:
        try:
            colors = extract_chart_series_color(image_path)
        except Exception:
            colors = []

    result = {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "coordinate_system": "polar",
        "center": plot_area["center"],
        "r_pixels": plot_area["r_pixels"],
        "theta_angles": ANGLE_TICKS,
        "theta_ticks": _angle_labels(ANGLE_TICKS),
        "r_ticks": [],
        "image_path": image_path,
        "basic_grid_path": encrypted_grid_path,
        "encrypted_grid_path": encrypted_grid_path,
        "image_paths": {
            "no_grid": str(Path(image_path).absolute()),
            "with_grid": str(Path(encrypted_grid_path).absolute()),
        },
        "colors": colors,
        "grid_generation": {
            "type": "angle_grid",
            "angle_step_degrees": ANGLE_STEP_DEGREES,
            "plot_area_detection": plot_area.get("detection_method"),
        },
    }

    json_path = os.path.join(output_dir, f"{chart_id}.json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    return result
