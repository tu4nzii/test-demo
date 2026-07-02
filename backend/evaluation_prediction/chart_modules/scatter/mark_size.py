"""Deterministic point/bubble diameter estimation for amplifier crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import data_to_pixel


@dataclass(frozen=True)
class MarkDiameterEstimate:
    diameter_px: float
    source: str
    confidence: float
    center_x_px: int | None = None
    center_y_px: int | None = None
    component_bbox: tuple[int, int, int, int] | None = None
    component_area: int | None = None

    def as_record_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "estimated_mark_diameter_px": round(float(self.diameter_px), 3),
            "mark_size_source": self.source,
            "mark_size_confidence": round(float(self.confidence), 4),
        }
        if self.center_x_px is not None and self.center_y_px is not None:
            fields["mark_size_center_x_px"] = self.center_x_px
            fields["mark_size_center_y_px"] = self.center_y_px
        if self.component_bbox is not None:
            fields["mark_size_component_bbox"] = list(self.component_bbox)
        if self.component_area is not None:
            fields["mark_size_component_area"] = self.component_area
        return fields


def estimate_mark_diameter(
    *,
    image_path: Path,
    pred_coord: tuple[float, float],
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int | float],
    y_pixels: list[int | float],
    mark_name: str,
    default_diameter: float,
    min_diameter: float,
    max_diameter: float,
) -> MarkDiameterEstimate:
    """Estimate marker diameter near the model prediction without using GT."""

    try:
        image = Image.open(image_path).convert("RGB")
        center_x, center_y = data_to_pixel(pred_coord, x_ticks, y_ticks, x_pixels, y_pixels)
    except Exception:
        return _fallback(default_diameter, min_diameter, max_diameter, "default_unavailable")

    search_radius = int(min(max(72.0, default_diameter * 5.0, max_diameter * 0.75), 180.0))
    left = max(0, center_x - search_radius)
    top = max(0, center_y - search_radius)
    right = min(image.width, center_x + search_radius + 1)
    bottom = min(image.height, center_y + search_radius + 1)
    if right <= left or bottom <= top:
        return _fallback(default_diameter, min_diameter, max_diameter, "default_empty_crop")

    crop = np.asarray(image.crop((left, top, right, bottom)), dtype=np.int16)
    local_center = (center_x - left, center_y - top)

    component = _best_component(crop, local_center, chromatic_only=True, mark_name=mark_name)
    source = "image_chromatic_component"
    if component is None:
        component = _best_component(crop, local_center, chromatic_only=False, mark_name=mark_name)
        source = "image_dark_component"
    if component is None:
        radial = _radial_diameter(crop, local_center, chromatic_only=True)
        source = "image_chromatic_radial"
        if radial is None:
            radial = _radial_diameter(crop, local_center, chromatic_only=False)
            source = "image_dark_radial"
        if radial is not None:
            diameter, bbox, area = radial
            confidence = min(0.85, max(0.15, area / 96.0))
            return MarkDiameterEstimate(
                diameter_px=_clamp(diameter, min_diameter, max_diameter),
                source=source,
                confidence=confidence,
                center_x_px=center_x,
                center_y_px=center_y,
                component_bbox=(left + bbox[0], top + bbox[1], left + bbox[2], top + bbox[3]),
                component_area=area,
            )
        return _fallback(default_diameter, min_diameter, max_diameter, "default_no_component")

    min_x, min_y, max_x, max_y, area, distance = component
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    bbox_diameter = (float(width) + float(height)) / 2.0
    area_diameter = float(np.sqrt(max(area, 1) * 4.0 / np.pi))
    diameter = max(bbox_diameter, area_diameter)
    diameter = _clamp(diameter, min_diameter, max_diameter)
    confidence = max(0.0, min(1.0, 1.0 - distance / max(1.0, search_radius)))
    return MarkDiameterEstimate(
        diameter_px=diameter,
        source=source,
        confidence=confidence,
        center_x_px=center_x,
        center_y_px=center_y,
        component_bbox=(left + min_x, top + min_y, left + max_x, top + max_y),
        component_area=area,
    )


def crop_size_from_mark_diameter(
    *,
    base_crop_size: int,
    estimate: MarkDiameterEstimate,
    mark_name: str,
    expand: bool,
) -> int:
    multiplier = 2.6 if mark_name == "bubble" else 3.4
    size = max(float(base_crop_size), estimate.diameter_px * multiplier, 96.0)
    if expand:
        size *= 2.0
    return int(min(max(round(size), 60), 720))


def _fallback(
    default_diameter: float,
    min_diameter: float,
    max_diameter: float,
    source: str,
) -> MarkDiameterEstimate:
    return MarkDiameterEstimate(
        diameter_px=_clamp(default_diameter, min_diameter, max_diameter),
        source=source,
        confidence=0.0,
    )


def _best_component(
    crop: np.ndarray,
    local_center: tuple[int, int],
    *,
    chromatic_only: bool,
    mark_name: str,
) -> tuple[int, int, int, int, int, float] | None:
    mask = _mark_like_mask(crop, chromatic_only=chromatic_only)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: tuple[float, tuple[int, int, int, int, int, float]] | None = None
    min_area = 12 if mark_name == "bubble" else 5

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not mask[y, x]:
                continue
            component = _collect_component(mask, visited, x, y)
            min_x, min_y, max_x, max_y, area = component
            if area < min_area:
                continue
            box_w = max_x - min_x + 1
            box_h = max_y - min_y + 1
            if box_w > width * 0.9 or box_h > height * 0.9:
                continue
            if max(box_w, box_h) > 260:
                continue
            distance = _distance_to_box(local_center, min_x, min_y, max_x, max_y)
            if distance > max(36.0, max(box_w, box_h) * 1.8):
                continue
            elongation = max(box_w, box_h) / max(1.0, min(box_w, box_h))
            if mark_name != "bubble" and elongation > 6.0 and area > 24:
                continue
            if mark_name == "bubble" and elongation > 4.5:
                continue
            score = distance + max(0.0, elongation - 1.0) * 5.0 - min(area, 900) * 0.002
            result = (min_x, min_y, max_x, max_y, area, distance)
            if best is None or score < best[0]:
                best = (score, result)
    return best[1] if best else None


def _mark_like_mask(crop: np.ndarray, *, chromatic_only: bool) -> np.ndarray:
    max_channel = crop.max(axis=2)
    min_channel = crop.min(axis=2)
    chroma = max_channel - min_channel
    not_light = max_channel < 245
    chromatic = (chroma >= 22) & not_light
    if chromatic_only:
        return chromatic
    dark = max_channel < 90
    mid_gray = (max_channel < 185) & (chroma < 16)
    return chromatic | dark | mid_gray


def _radial_diameter(
    crop: np.ndarray,
    local_center: tuple[int, int],
    *,
    chromatic_only: bool,
) -> tuple[float, tuple[int, int, int, int], int] | None:
    mask = _mark_like_mask(crop, chromatic_only=chromatic_only)
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    cx, cy = local_center
    distances = np.hypot(xs.astype(float) - float(cx), ys.astype(float) - float(cy))
    close = distances <= 120.0
    xs = xs[close]
    ys = ys[close]
    distances = distances[close]
    if len(xs) < 12:
        return None
    outer = float(np.percentile(distances, 90))
    inner = float(np.percentile(distances, 10))
    if outer < 3.0 or outer - inner < 1.0:
        return None
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_diameter = (float(max_x - min_x + 1) + float(max_y - min_y + 1)) / 2.0
    radial_diameter = outer * 2.0
    return max(radial_diameter, bbox_diameter), (min_x, min_y, max_x, max_y), int(len(xs))


def _collect_component(
    mask: np.ndarray,
    visited: np.ndarray,
    start_x: int,
    start_y: int,
) -> tuple[int, int, int, int, int]:
    height, width = mask.shape
    stack = [(start_x, start_y)]
    visited[start_y, start_x] = True
    min_x = max_x = start_x
    min_y = max_y = start_y
    area = 0
    while stack:
        x, y = stack.pop()
        area += 1
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        for ny in range(max(0, y - 1), min(height, y + 2)):
            for nx in range(max(0, x - 1), min(width, x + 2)):
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((nx, ny))
    return min_x, min_y, max_x, max_y, area


def _distance_to_box(
    center: tuple[int, int],
    min_x: int,
    min_y: int,
    max_x: int,
    max_y: int,
) -> float:
    x, y = center
    dx = max(min_x - x, 0, x - max_x)
    dy = max(min_y - y, 0, y - max_y)
    return float(np.hypot(dx, dy))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(float(minimum), min(float(value), float(maximum)))
