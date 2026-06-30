"""Rose chart grid encryption and circle detection helpers.

Rose charts share the final encrypted-grid drawing code with radar charts, but
their radius detection is different: colored wedge tops create strong circular
edges that can be confused with true grid rings.  This module keeps the radar
encoder drawing pipeline while replacing the rose circle detector with a
low-saturation grid-edge radius profile.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.polar.radar.encrypt_radar import RadarChartEncoder  # noqa: E402


def _circle_edge_support(
    edges: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    samples: int = 240,
    radius_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> float:
    h, w = edges.shape[:2]
    supported = 0
    for angle in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        found = False
        for dr in radius_offsets:
            x = int(round(cx + (radius + dr) * math.cos(angle)))
            y = int(round(cy + (radius + dr) * math.sin(angle)))
            if 0 <= x < w and 0 <= y < h and edges[y, x] > 0:
                found = True
                break
        supported += int(found)
    return supported / samples


def rose_grid_edges(image: np.ndarray) -> np.ndarray:
    """Return edges for gray/black grid structures while suppressing wedges."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove saturated colored wedges before Canny.  Text, gray dashed grid
    # rings, black outer rings and gray axes remain visible.
    cleaned = gray.copy()
    saturated = hsv[:, :, 1] > 70
    cleaned[saturated] = 255
    cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0.7)
    return cv2.Canny(cleaned, 35, 120)


def _find_rose_center(image: np.ndarray, debug: dict[str, Any]) -> tuple[int, int, int, float, str]:
    h, w = image.shape[:2]
    short_side = min(h, w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = rose_grid_edges(image)

    min_radius = max(10, int(short_side * 0.06))
    max_radius = int(short_side * 0.56)
    best: tuple[int, int, int] | None = None
    best_support = 0.0
    source = "failed"

    for param2 in (35, 30, 25, 20, 15, 10):
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(30, short_side // 10),
            param1=40,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue
        for cx, cy, radius in np.around(circles[0]).astype(int):
            if radius <= 0 or not (0 <= cx < w and 0 <= cy < h):
                continue
            support = _circle_edge_support(edges, cx, cy, radius)
            # Prefer a strong grid-like ring, then a circle close to the image
            # center.  This avoids wedge top arcs becoming the center estimate.
            center_penalty = math.hypot(cx - w / 2, cy - h / 2) / max(short_side, 1)
            score = support - 0.12 * center_penalty
            if best is None or score > best_support - 0.12 * math.hypot(best[0] - w / 2, best[1] - h / 2) / max(short_side, 1):
                best = (int(cx), int(cy), int(radius))
                best_support = float(support)
                source = f"rose_center_p2={param2}"
        if best is not None and best_support >= 0.35:
            break

    if best is None:
        return 0, 0, 0, 0.0, source

    debug["center_circle"] = {
        "cx": best[0],
        "cy": best[1],
        "r": best[2],
        "edge_support": round(best_support, 4),
    }
    return best[0], best[1], best[2], round(best_support, 4), source


def rose_radius_peaks(
    image: np.ndarray,
    center: tuple[int, int],
    max_radius: int | None = None,
) -> list[dict[str, float]]:
    """Find likely true grid-ring radii from a low-saturation edge profile."""
    h, w = image.shape[:2]
    short_side = min(h, w)
    cx, cy = center
    edges = rose_grid_edges(image)

    min_radius = max(18, int(short_side * 0.045))
    if max_radius is None:
        max_radius = int(short_side * 0.56)
    max_radius = min(max_radius, int(min(cx, cy, w - cx, h - cy) + short_side * 0.08))
    if max_radius <= min_radius:
        return []

    profile: list[tuple[float, int]] = []
    for radius in range(min_radius, max_radius + 1):
        support = _circle_edge_support(edges, cx, cy, radius)
        profile.append((support, radius))

    raw_peaks: list[tuple[float, int]] = []
    for idx in range(2, len(profile) - 2):
        support, radius = profile[idx]
        if support < 0.10:
            continue
        if support >= profile[idx - 1][0] and support >= profile[idx + 1][0]:
            raw_peaks.append((support, radius))

    if not raw_peaks:
        return []

    # Merge adjacent maxima generated by one thick or anti-aliased ring.
    merge_gap = max(4, int(short_side * 0.006))
    clusters: list[list[tuple[float, int]]] = []
    for support, radius in sorted(raw_peaks, key=lambda item: item[1]):
        if clusters and radius - clusters[-1][-1][1] <= merge_gap:
            clusters[-1].append((support, radius))
        else:
            clusters.append([(support, radius)])

    peaks = [max(cluster, key=lambda item: item[0]) for cluster in clusters]
    best_support = max(support for support, _ in peaks)
    keep_threshold = max(0.18, best_support * 0.42)
    min_spacing = max(10, int(short_side * 0.035))

    selected: list[tuple[float, int]] = []
    for support, radius in sorted(peaks, reverse=True):
        if support < keep_threshold:
            continue
        if any(abs(radius - kept_radius) < min_spacing for _, kept_radius in selected):
            continue
        selected.append((support, radius))

    selected = _regularize_radius_peaks(selected, short_side)
    selected.sort(key=lambda item: item[1])
    return [{"radius": float(radius), "support": float(support)} for support, radius in selected]


def _regularize_radius_peaks(peaks: list[tuple[float, int]], short_side: int) -> list[tuple[float, int]]:
    """Drop isolated wedge/text peaks that do not fit a radial tick grid."""
    if len(peaks) <= 2:
        return peaks

    min_base = max(12.0, short_side * 0.045)
    max_k = 10
    candidates: list[tuple[float, list[tuple[float, int]]]] = []

    for support, radius in peaks:
        for k in range(1, max_k + 1):
            base = radius / k
            if base < min_base:
                continue
            buckets: dict[int, tuple[float, int, float]] = {}
            for peak_support, peak_radius in peaks:
                peak_k = int(round(peak_radius / base))
                if peak_k <= 0 or peak_k > max_k:
                    continue
                expected = peak_k * base
                distance = abs(peak_radius - expected)
                tolerance = max(5.0, base * 0.10)
                if distance > tolerance:
                    continue
                current = buckets.get(peak_k)
                if current is None or peak_support > current[0]:
                    buckets[peak_k] = (peak_support, peak_radius, distance)

            chosen = [(value[0], value[1]) for _, value in sorted(buckets.items())]
            if len(chosen) < 2:
                continue
            mean_distance = float(np.mean([value[2] for value in buckets.values()]))
            ks = sorted(buckets)
            missing_steps = (ks[-1] - ks[0] + 1) - len(ks)
            outer_bonus = 0.18 if max(r for _, r in chosen) >= max(r for _, r in peaks) - max(8, int(short_side * 0.012)) else 0.0
            score = (
                sum(s for s, _ in chosen)
                + 0.16 * len(chosen)
                + outer_bonus
                - 0.025 * mean_distance
                - 0.20 * missing_steps
            )
            candidates.append((score, chosen))

    if not candidates:
        return peaks

    _, best = max(candidates, key=lambda item: item[0])
    # Keep the original list if the regularized subset is too weak; otherwise
    # use the cleaner grid-consistent subset.
    if len(best) < len(peaks) and sum(s for s, _ in best) >= 0.55 * sum(s for s, _ in peaks):
        return best
    return peaks


def detect_rose_circles(image: np.ndarray, debug: dict[str, Any] | None = None) -> tuple[int, int, list[int], str, float]:
    """Detect rose chart center and true grid ring radii.

    Returns ``(cx, cy, radii, detection_source, edge_support)``.  ``radii`` is
    sorted from inner to outer and excludes the origin.
    """
    if debug is None:
        debug = {}
    cx, cy, hough_radius, edge_support, source = _find_rose_center(image, debug)
    if hough_radius <= 0:
        return 0, 0, [], source, edge_support

    peaks = rose_radius_peaks(image, (cx, cy))
    debug["radius_peaks"] = [
        {"radius": int(round(item["radius"])), "support": round(item["support"], 4)}
        for item in peaks
    ]

    radii = [int(round(item["radius"])) for item in peaks]
    if not radii:
        radii = [int(hough_radius)]
        source = f"{source}+hough_only"
    else:
        source = f"{source}+gray_profile"

    best_radius_support = max((item["support"] for item in peaks), default=edge_support)
    return int(cx), int(cy), radii, source, round(float(best_radius_support), 4)


class RoseChartEncoder(RadarChartEncoder):
    """Radar encoder with rose-specific center/radius detection."""

    def visualize_ring_mask(self, image_path, ring_width=5):
        image = cv2.imread(str(image_path))
        if image is None:
            return image

        debug: dict[str, Any] = {}
        cx, cy, radii, source, support = detect_rose_circles(image, debug)
        self.coords = [cx, cy]
        self.first_r = radii[0] if radii else 0
        self.second_r = radii[1] if len(radii) > 1 else 0
        self.detection_source = source
        self.last_edge_support = support
        self.last_rose_debug = debug

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked = image.copy()
        for radius in radii[:2]:
            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), int(radius + ring_width), 255, -1)
            cv2.circle(mask, (cx, cy), max(0, int(radius - ring_width)), 0, -1)
            masked[mask == 255] = 255
        return masked

    def second_circle_find(self, image):
        return self.second_r


def encrypt_rose(image_path, output_dir=None):
    """Encrypt a rose chart with the rose-specific grid detector."""
    encoder = RoseChartEncoder()
    return encoder.process_single_image(image_path, output_dir)
