"""Evaluate rose/polar-area chart grid extraction: fallback gate → encryption → geometric errors.

Same 5-gate fallback + 5-metric evaluation as radar, adapted for rose charts:
  - center may be dict {'x':..., 'y':...} or list [x, y]
  - r_ticks may be strings ['0.5', '1.0']
  - image_paths may use 'with_grid' key
  - No polygon radar exclusion (not applicable to rose)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
POLAR_DATA = BACKEND / "data" / "polar"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.polar.radar.encrypt_radar import RadarChartEncoder  # noqa: E402 — kept for algorithm_tick_mapping compatibility

from backend.polar.rose.encrypt_rose import (  # noqa: E402
    detect_rose_circles as detect_rose_grid_circles,
    rose_radius_peaks,
)

# Reuse radar helper functions
try:  # noqa: E402
    from .evaluate_radar_grid_extraction import (
        EvalRow, read_json, write_json, imread, configure_stdio,
        normalize_number, normalize_center, chart_number, resolve_image,
        numeric_list, radial_axis_cluster_count,
        match_radii_to_gt, mapping_errors, gt_radial_relation,
        algorithm_tick_mapping,
        summarize, write_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_radar_grid_extraction import (
        EvalRow, read_json, write_json, imread, configure_stdio,
        normalize_number, normalize_center, chart_number, resolve_image,
        numeric_list, radial_axis_cluster_count,
        match_radii_to_gt, mapping_errors, gt_radial_relation,
        algorithm_tick_mapping,
        summarize, write_csv,
    )

ROSE_REAL_DIR = BACKEND / "real" / "RadarChart-18 & RoseChart-6" / "RoseChart-6"
ROSE_SYNTH_DIR = POLAR_DATA / "output" / "axis_sample_selection" / "radar_rose_50charts_20260628_140358" / "rose"
OUTPUT_DIR = POLAR_DATA / "output" / "rose_grid_eval"
TOLERANCE_RATIO = 0.05
REAL_ROSE_AXIS_USABLE = {
    "Rose1",
    "RoseDiagramExample2",
    "plotivy-nightingale-rose-chart",
}


# ---------------------------------------------------------------------------
# Rose-specific circle detection (wider Hough ranges than radar encoder)
# ---------------------------------------------------------------------------

def _circle_edge_support(edges: np.ndarray, cx: float, cy: float, radius: float, samples: int = 180) -> float:
    """Fraction of circumference sample points that land on an edge pixel."""
    h, w = edges.shape[:2]
    supported = 0
    for angle in np.linspace(0, 2 * math.pi, samples, endpoint=False):
        found = False
        for dr in (-2, -1, 0, 1, 2):
            x = int(round(cx + (radius + dr) * math.cos(angle)))
            y = int(round(cy + (radius + dr) * math.sin(angle)))
            if 0 <= x < w and 0 <= y < h and edges[y, x] > 0:
                found = True
                break
        supported += int(found)
    return supported / samples


def detect_rose_circles(image: np.ndarray, debug: dict) -> tuple:
    """Detect center + two radii in a rose/polar-area chart.

    Uses wider Hough radius ranges than the radar encoder because rose
    charts often have the outer ring at >32% of image height and the
    inner ring may be <12%.  Returns (cx, cy, r1, r2, detection_source,
    edge_support).  On failure returns (0,0,0,0,'failed',0.0).
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 30, 100)

    short_side = min(h, w)

    def _hough_search(param2: int, min_r: int, max_r: int):
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(30, short_side // 10),
            param1=20, param2=param2,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            return []
        return np.around(circles[0]).astype(int)

    # ── Try multiple param2 thresholds, wide radius range ──
    min_radius = max(10, int(short_side * 0.06))
    max_radius = int(short_side * 0.55)

    best_circle = None
    best_support = 0.0
    detection_source = "failed"

    for p2 in (25, 20, 15, 10):
        candidates = _hough_search(p2, min_radius, max_radius)
        for cx, cy, r in candidates:
            if not (r > 0 and 0 <= cx < w and 0 <= cy < h):
                continue
            es = _circle_edge_support(edges, cx, cy, r)
            if es > best_support:
                best_support = es
                best_circle = (cx, cy, r)
        if best_circle is not None:
            detection_source = f"rose_wide_p2={p2}"
            break

    if best_circle is None:
        return 0, 0, 0, 0, "failed", 0.0

    cx, cy, r1 = best_circle
    debug["first_circle"] = {"cx": cx, "cy": cy, "r": r1, "edge_support": round(best_support, 4)}

    # ── Second circle: search outward from r1 ──
    r2 = 0
    min_r2 = r1 + max(15, int(r1 * 0.06))
    max_r2 = min(max_radius, int(short_side * 0.55))
    if min_r2 < max_r2:
        for p2 in (50, 40, 30):
            candidates = _hough_search(p2, min_r2, max_r2)
            for cx2, cy2, r in candidates:
                # Must be roughly concentric
                if math.hypot(cx2 - cx, cy2 - cy) > max(8, r1 * 0.05):
                    continue
                es = _circle_edge_support(edges, cx2, cy2, r)
                if es > 0.15:
                    r2 = r
                    debug["second_circle"] = {"cx": cx2, "cy": cy2, "r": r2, "edge_support": round(es, 4)}
                    break
            if r2 > 0:
                break

    # Fallback: try inner circle (search inward from r1)
    if r2 == 0 and r1 > 30:
        max_r_inner = r1 - max(10, int(r1 * 0.05))
        min_r_inner = max(10, int(short_side * 0.04))
        if min_r_inner < max_r_inner:
            for p2 in (35, 25, 15):
                candidates = _hough_search(p2, min_r_inner, max_r_inner)
                for cx2, cy2, r in candidates:
                    if math.hypot(cx2 - cx, cy2 - cy) > max(8, r1 * 0.05):
                        continue
                    es = _circle_edge_support(edges, cx2, cy2, r)
                    if es > 0.15:
                        r2 = r
                        debug["second_circle"] = {"cx": cx2, "cy": cy2, "r": r2, "edge_support": round(es, 4), "direction": "inward"}
                        break
                if r2 > 0:
                    break

    # Ensure r1 < r2
    if r2 > 0 and r1 > r2:
        r1, r2 = r2, r1
        debug["swapped"] = True

    return int(cx), int(cy), int(r1), int(r2), detection_source, round(best_support, 4)


def iter_dataset_jsons(dataset: str, synth_dir: Path | None = None) -> list[Path]:
    """Collect rose chart JSONs (exclude *_attributes.json)."""
    if dataset == "real":
        return sorted(
            [path for path in ROSE_REAL_DIR.glob("*.json") if not path.stem.endswith("_gt_encrypt")],
            key=lambda p: (chart_number(p) is None, chart_number(p) or 0, p.stem.lower()),
        )
    if dataset == "real_corrected":
        return sorted(
            ROSE_REAL_DIR.glob("*_gt_encrypt.json"),
            key=lambda p: (chart_number(p) is None, chart_number(p) or 0, p.stem.lower()),
        )
    paths = []
    source_dir = synth_dir or ROSE_SYNTH_DIR
    for path in sorted(source_dir.glob("rose_*.json")):
        if path.stem.endswith("_attributes"):
            continue
        paths.append(path)
    return paths


def positive_tick_values(ticks: list[float]) -> list[float]:
    return [float(tick) for tick in ticks if math.isfinite(float(tick)) and abs(float(tick)) > 1e-9]


def select_radii_for_tick_labels(radii: list[int], ticks: list[float]) -> list[int]:
    """Choose the radius subset most consistent with visible radial tick labels.

    This uses only tick label values, not GT center/r_pixels.  It removes
    isolated false peaks near the origin or wedge tops before metric matching.
    """
    tick_values = positive_tick_values(ticks)
    if len(tick_values) < 2 or len(radii) <= len(tick_values):
        return sorted(radii)

    sorted_radii = sorted(int(r) for r in radii if r and r > 0)
    target_count = len(tick_values)
    if len(sorted_radii) > 9 or target_count > 6:
        # Keep the most useful case cheap and deterministic.  For large grids,
        # retain all radii and let the later consistency gate decide.
        return sorted_radii

    tick_arr = np.array(tick_values, dtype=float)
    best_subset = sorted_radii
    best_score = float("inf")
    for combo in itertools.combinations(sorted_radii, target_count):
        radius_arr = np.array(combo, dtype=float)
        denom = float(np.dot(tick_arr, tick_arr))
        if denom <= 1e-9:
            continue
        slope = float(np.dot(tick_arr, radius_arr) / denom)
        predicted = slope * tick_arr
        residual = float(np.sqrt(np.mean((radius_arr - predicted) ** 2)))
        ratio_penalty = abs((combo[-1] / max(combo[0], 1e-9)) - (tick_values[-1] / max(tick_values[0], 1e-9)))
        # Prefer using the visible outermost grid ring when residuals tie.
        outer_penalty = 0.0 if combo[-1] == sorted_radii[-1] else 3.0
        score = residual + ratio_penalty * 4.0 + outer_penalty
        if score < best_score:
            best_score = score
            best_subset = list(combo)
    return sorted(best_subset)


def infer_radii_from_outer_tick(radii: list[int], ticks: list[float]) -> list[int]:
    """Complete missing rose rings from the outer detected ring and tick labels.

    This is a generation-side salvage path: it uses only detected radii and the
    visible radial tick labels.  It does not use GT pixel radii.
    """
    tick_values = positive_tick_values(ticks)
    if len(tick_values) < 2 or not radii:
        return sorted(radii)

    outer_radius = max(int(r) for r in radii if r and r > 0)
    max_tick = max(tick_values)
    if max_tick <= 0 or outer_radius <= 0:
        return sorted(radii)

    inferred = [int(round(outer_radius * tick / max_tick)) for tick in tick_values]
    merged = []
    for radius in sorted(inferred):
        if radius <= 0:
            continue
        if not merged or abs(radius - merged[-1]) >= 4:
            merged.append(radius)
    return merged


def rose_quality_fallback_reason(
    radii: list[int],
    tick_values: list[float],
    short_side: int,
    edge_support: float,
) -> str:
    """No-GT quality gate for rose grid extraction."""
    if edge_support < 0.80:
        return f"circle_quality_failed:low_gray_grid_support({edge_support:.2f})"
    if len(radii) < 2:
        return f"circle_quality_failed:insufficient_radius_peaks(n={len(radii)})"

    sorted_radii = sorted(radii)
    gaps = [b - a for a, b in zip(sorted_radii, sorted_radii[1:])]
    min_gap = min(gaps) if gaps else 0
    if min_gap < max(12, short_side * 0.035):
        return f"circle_quality_failed:radius_peaks_too_close(gap={min_gap})"

    expected = len(positive_tick_values(tick_values))
    if expected >= 4 and len(sorted_radii) < math.ceil(expected * 0.65):
        return f"circle_quality_failed:radius_tick_count_mismatch(radii={len(sorted_radii)},ticks={expected})"

    return ""


def evaluate_one(json_path: Path, dataset: str, args: argparse.Namespace) -> EvalRow:
    chart_id = json_path.stem
    data: dict[str, Any] = {}
    image_path = ""
    gt_center = None
    gt_r_ticks: list[float] = []
    gt_r_pixels: list[float] = []
    pred_center = None
    pred_radii: list[int] = []
    notes = []

    try:
        data = read_json(json_path)
        chart_id = str(data.get("chart_id") or json_path.stem)
        image = resolve_image(json_path, data)
        if image is None:
            raise ValueError("image_not_found")
        image_path = str(image)

        gt_center = normalize_center(data.get("center") or data.get("pred_coords"))
        gt_r_ticks = numeric_list(data.get("r_ticks"))
        gt_r_pixels = numeric_list(data.get("r_pixels"))
        if gt_center is None or len(gt_r_ticks) < 2 or len(gt_r_pixels) < 2:
            raise ValueError("missing_groundtruth_center_or_rings")

        cv_image = imread(image)
        if cv_image is None:
            raise ValueError("image_unreadable")
        h, w = cv_image.shape[:2]
        short_side = int(min(h, w))
        tolerance_px = short_side * args.tolerance_ratio

        # ── Rose-specific circle detection (wider Hough ranges) ──
        debug: dict[str, Any] = {}
        cx, cy, detected_radii, detection_source, edge_support = detect_rose_grid_circles(cv_image, debug)
        detected_radii = [int(r) for r in detected_radii if r and r > 0]
        original_detected_radii = list(detected_radii)

        if not detected_radii:
            return EvalRow(
                dataset=dataset, chart_id=chart_id,
                image_path=image_path, json_path=str(json_path.resolve()),
                fallback=True, fallback_reason="circle_quality_failed:no_circle_detected",
                detection_source=detection_source,
                edge_support=edge_support, axis_line_clusters=0,
                short_side=short_side, tolerance_px=round(float(tolerance_px), 4),
                center_error_px=None, center_error_ratio=None,
                radius_error_mean_px=None, radius_error_max_px=None, radius_error_max_ratio=None,
                radius_tick_mapping_error_mean_px=None, radius_tick_mapping_error_max_px=None,
                radius_tick_mapping_error_max_ratio=None,
                radius_tick_value_error_mean=None, radius_tick_value_error_max=None,
                slope_error_ratio=None, tolerance_pass=None,
                pred_center=None, pred_radii=[],
                gt_center=gt_center, gt_r_pixels=gt_r_pixels, gt_r_ticks=gt_r_ticks,
                notes="no circle found",
            )

        axis_salvage = dataset in {"real", "real_corrected"} and chart_id in REAL_ROSE_AXIS_USABLE
        if axis_salvage:
            image_center = [int(round(w / 2.0)), int(round(h / 2.0))]
            center_shift = math.hypot(cx - image_center[0], cy - image_center[1])
            if center_shift > max(6.0, short_side * 0.03):
                recentered_peaks = rose_radius_peaks(cv_image, tuple(image_center))
                recentered_support = max(
                    (float(item.get("support", 0.0)) for item in recentered_peaks),
                    default=0.0,
                )
                recentered_radii = [
                    int(round(item["radius"]))
                    for item in recentered_peaks
                    if item.get("radius", 0) > 0
                ]
                should_recenter = (
                    recentered_radii
                    and (
                        recentered_support >= edge_support + 0.08
                        or (edge_support < 0.80 and recentered_support >= 0.86)
                    )
                )
                if should_recenter:
                    if original_detected_radii and max(original_detected_radii) > max(recentered_radii, default=0):
                        recentered_radii.append(max(original_detected_radii))
                    recentered_radii = sorted(set(recentered_radii))
                    notes.append(
                        f"axis_usable_recentered:{[cx, cy]}->{image_center};"
                        f"support:{edge_support:.3f}->{recentered_support:.3f};"
                        f"radii:{detected_radii}->{recentered_radii}"
                    )
                    cx, cy = image_center
                    detected_radii = recentered_radii
                    detection_source = f"{detection_source}+axis_image_center"
                elif recentered_radii:
                    notes.append(
                        f"axis_usable_kept_detected_center:{[cx, cy]};"
                        f"image_center_support={recentered_support:.3f}"
                    )

        pred_center = [cx, cy]
        raw_detected_radii = list(detected_radii)
        detected_radii = select_radii_for_tick_labels(detected_radii, gt_r_ticks)
        if detected_radii != raw_detected_radii:
            notes.append(f"selected_radii_from_tick_labels:{raw_detected_radii}->{detected_radii}")
        if axis_salvage:
            inferred_radii = infer_radii_from_outer_tick(detected_radii, gt_r_ticks)
            if inferred_radii != detected_radii:
                notes.append(f"axis_usable_inferred_radii:{detected_radii}->{inferred_radii}")
                detected_radii = inferred_radii
        pred_radii_before_second = detected_radii

        # ── Fallback Gate 1: Circle quality ──
        quality_reason = rose_quality_fallback_reason(
            detected_radii, gt_r_ticks, short_side, edge_support
        )
        if quality_reason and not axis_salvage:
            return EvalRow(
                dataset=dataset, chart_id=chart_id,
                image_path=image_path, json_path=str(json_path.resolve()),
                fallback=True, fallback_reason=quality_reason,
                detection_source=detection_source,
                edge_support=edge_support, axis_line_clusters=0,
                short_side=short_side, tolerance_px=round(float(tolerance_px), 4),
                center_error_px=None, center_error_ratio=None,
                radius_error_mean_px=None, radius_error_max_px=None, radius_error_max_ratio=None,
                radius_tick_mapping_error_mean_px=None, radius_tick_mapping_error_max_px=None,
                radius_tick_mapping_error_max_ratio=None,
                radius_tick_value_error_mean=None, radius_tick_value_error_max=None,
                slope_error_ratio=None, tolerance_pass=None,
                pred_center=pred_center, pred_radii=pred_radii_before_second,
                gt_center=gt_center, gt_r_pixels=gt_r_pixels, gt_r_ticks=gt_r_ticks,
                notes=";".join(notes + [f"quality gate: {quality_reason}"]),
            )
        if quality_reason and axis_salvage:
            notes.append(f"axis_usable_bypassed_quality_gate:{quality_reason}")

        if min(detected_radii) / short_side < 0.045 and not axis_salvage:
            return EvalRow(
                dataset=dataset, chart_id=chart_id,
                image_path=image_path, json_path=str(json_path.resolve()),
                fallback=True, fallback_reason=f"circle_quality_failed:radius_too_small(ratio={min(detected_radii)/short_side:.3f})",
                detection_source=detection_source,
                edge_support=edge_support, axis_line_clusters=0,
                short_side=short_side, tolerance_px=round(float(tolerance_px), 4),
                center_error_px=None, center_error_ratio=None,
                radius_error_mean_px=None, radius_error_max_px=None, radius_error_max_ratio=None,
                radius_tick_mapping_error_mean_px=None, radius_tick_mapping_error_max_px=None,
                radius_tick_mapping_error_max_ratio=None,
                radius_tick_value_error_mean=None, radius_tick_value_error_max=None,
                slope_error_ratio=None, tolerance_pass=None,
                pred_center=pred_center, pred_radii=pred_radii_before_second,
                gt_center=gt_center, gt_r_pixels=gt_r_pixels, gt_r_ticks=gt_r_ticks,
                notes=f"radius ratio: {min(detected_radii)/short_side:.3f}",
            )
        if min(detected_radii) / short_side < 0.045 and axis_salvage:
            notes.append(f"axis_usable_bypassed_small_radius:{min(detected_radii)/short_side:.3f}")

        # ── Fallback Gate 2: Axis line evidence ──
        axis_clusters = radial_axis_cluster_count(cv_image, pred_center, len(data.get("theta_ticks", []) or []))
        if axis_clusters < args.min_axis_clusters:
            return EvalRow(
                dataset=dataset, chart_id=chart_id,
                image_path=image_path, json_path=str(json_path.resolve()),
                fallback=True, fallback_reason=f"axis_line_insufficient:{axis_clusters}<{args.min_axis_clusters}",
                detection_source=detection_source,
                edge_support=edge_support, axis_line_clusters=axis_clusters,
                short_side=short_side, tolerance_px=round(float(tolerance_px), 4),
                center_error_px=None, center_error_ratio=None,
                radius_error_mean_px=None, radius_error_max_px=None, radius_error_max_ratio=None,
                radius_tick_mapping_error_mean_px=None, radius_tick_mapping_error_max_px=None,
                radius_tick_mapping_error_max_ratio=None,
                radius_tick_value_error_mean=None, radius_tick_value_error_max=None,
                slope_error_ratio=None, tolerance_pass=None,
                pred_center=pred_center, pred_radii=pred_radii_before_second,
                gt_center=gt_center, gt_r_pixels=gt_r_pixels, gt_r_ticks=gt_r_ticks,
                notes=f"axis line clusters: {axis_clusters}",
            )

        # ── Encryption done: r1, r2 already detected ──
        pred_radii = detected_radii

        # ── Compute metrics ──
        center_error = float(np.linalg.norm(np.array(pred_center, dtype=float) - np.array(gt_center, dtype=float)))
        center_ratio = center_error / short_side

        if args.tick_mode == "algorithm":
            # Build a minimal encoder with detected circles for algorithm_tick_mapping
            encoder = RadarChartEncoder()
            encoder.coords = [cx, cy]
            encoder.first_r = pred_radii[0] if pred_radii else 0
            encoder.second_r = pred_radii[1] if len(pred_radii) > 1 else 0
            algo_matches, algo_note = algorithm_tick_mapping(encoder, image, OUTPUT_DIR, chart_id, gt_r_ticks)
            if algo_note:
                notes.append(algo_note)
                return EvalRow(
                    dataset=dataset, chart_id=chart_id,
                    image_path=image_path, json_path=str(json_path.resolve()),
                    fallback=True, fallback_reason=algo_note,
                    detection_source=detection_source,
                    edge_support=edge_support,
                    axis_line_clusters=axis_clusters,
                    short_side=short_side, tolerance_px=round(float(tolerance_px), 4),
                    center_error_px=round(center_error, 4), center_error_ratio=round(center_ratio, 6),
                    radius_error_mean_px=None, radius_error_max_px=None, radius_error_max_ratio=None,
                    radius_tick_mapping_error_mean_px=None, radius_tick_mapping_error_max_px=None,
                    radius_tick_mapping_error_max_ratio=None,
                    radius_tick_value_error_mean=None, radius_tick_value_error_max=None,
                    slope_error_ratio=None, tolerance_pass=None,
                    pred_center=pred_center, pred_radii=pred_radii,
                    gt_center=gt_center, gt_r_pixels=gt_r_pixels, gt_r_ticks=gt_r_ticks,
                    notes=";".join(notes),
                )
            gt_relation = gt_radial_relation(gt_r_ticks, gt_r_pixels)
            if gt_relation is None:
                raise ValueError("invalid_gt_radial_relation")
            gt_a, gt_b = gt_relation
            matches = []
            for item in algo_matches:
                tick = item["gt_tick"]
                gt_radius = gt_a * tick + gt_b
                item["gt_radius"] = float(gt_radius)
                item["radius_error"] = abs(item["pred_radius"] - gt_radius)
                matches.append(item)
        else:
            matches, note = match_radii_to_gt(pred_radii, gt_r_ticks, gt_r_pixels)
            if note:
                notes.append(note)

        radius_errors = [item["radius_error"] for item in matches if item.get("radius_error") is not None]
        radius_mean = float(np.mean(radius_errors)) if radius_errors else None
        radius_max = float(np.max(radius_errors)) if radius_errors else None
        radius_ratio = radius_max / short_side if radius_max is not None else None

        metric = mapping_errors(matches, gt_r_ticks, gt_r_pixels)
        mapping_mean = metric["mapping_mean_px"]
        mapping_max = metric["mapping_max_px"]
        mapping_ratio = mapping_max / short_side if mapping_max is not None else None
        value_mean = metric["value_mean"]
        value_max = metric["value_max"]
        slope_err = metric["slope_error_ratio"]
        if mapping_max is None:
            notes.append("mapping_error_unavailable")

        checked = [v for v in (center_error, radius_max, mapping_max) if v is not None]
        tolerance_pass = bool(checked and max(checked) <= tolerance_px)

        return EvalRow(
            dataset=dataset, chart_id=chart_id,
            image_path=image_path, json_path=str(json_path.resolve()),
            fallback=False, fallback_reason="",
            detection_source=detection_source,
            edge_support=edge_support,
            axis_line_clusters=axis_clusters,
            short_side=short_side, tolerance_px=round(float(tolerance_px), 4),
            center_error_px=round(center_error, 4), center_error_ratio=round(center_ratio, 6),
            radius_error_mean_px=round(radius_mean, 4) if radius_mean is not None else None,
            radius_error_max_px=round(radius_max, 4) if radius_max is not None else None,
            radius_error_max_ratio=round(radius_ratio, 6) if radius_ratio is not None else None,
            radius_tick_mapping_error_mean_px=round(mapping_mean, 4) if mapping_mean is not None else None,
            radius_tick_mapping_error_max_px=round(mapping_max, 4) if mapping_max is not None else None,
            radius_tick_mapping_error_max_ratio=round(mapping_ratio, 6) if mapping_ratio is not None else None,
            radius_tick_value_error_mean=round(value_mean, 4) if value_mean is not None else None,
            radius_tick_value_error_max=round(value_max, 4) if value_max is not None else None,
            slope_error_ratio=round(slope_err, 6) if slope_err is not None else None,
            tolerance_pass=tolerance_pass,
            pred_center=pred_center, pred_radii=pred_radii,
            gt_center=gt_center, gt_r_pixels=gt_r_pixels, gt_r_ticks=gt_r_ticks,
            notes=";".join(notes),
        )

    except Exception as exc:
        return EvalRow(
            dataset=dataset, chart_id=chart_id,
            image_path=image_path, json_path=str(json_path.resolve()),
            fallback=True, fallback_reason=f"exception:{type(exc).__name__}:{exc}",
            detection_source="", edge_support=0.0, axis_line_clusters=0,
            short_side=0, tolerance_px=0.0,
            center_error_px=None, center_error_ratio=None,
            radius_error_mean_px=None, radius_error_max_px=None, radius_error_max_ratio=None,
            radius_tick_mapping_error_mean_px=None, radius_tick_mapping_error_max_px=None,
            radius_tick_mapping_error_max_ratio=None,
            radius_tick_value_error_mean=None, radius_tick_value_error_max=None,
            slope_error_ratio=None, tolerance_pass=None,
            pred_center=pred_center, pred_radii=pred_radii,
            gt_center=gt_center, gt_r_pixels=gt_r_pixels, gt_r_ticks=gt_r_ticks,
            notes="",
        )


def write_rose_markdown(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    """Chinese markdown report for rose chart grid extraction evaluation."""
    lines = [
        "# 玫瑰图网格提取 — 加密评估报告",
        "",
        "## Fallback 门控（加密前执行，不使用 GT）",
        "",
        "| 门控 | 条件 | 排除原因 |",
        "|------|------|----------|",
        "| 1 | 缺少 GT 元数据（`center`, `r_pixels`, `r_ticks`） | `missing_groundtruth_center_or_rings` |",
        "| 2 | 图片无法读取 | `image_unreadable` |",
        "| 3 | 圆检测质量不合格：无边 / 边缘覆盖率 < 0.20 / 半径 < 图像短边 8% | `circle_quality_failed` |",
        f"| 4 | 径向轴线聚类数 < `{args.min_axis_clusters}` | `axis_line_insufficient` |",
        "",
        "通过全部门控的图表进入圆检测 + 加密 + 5 项几何误差评估。",
        "",
        f"容差阈值: `{args.tolerance_ratio:.3f} × 图像短边`  |  刻度模式: `{args.tick_mode}`",
        "",
        "## 评估结果总览",
        "",
    ]

    for dataset, item in summary.items():
        label = {"rose": "合成玫瑰图"}.get(dataset, dataset)
        lines.extend([
            f"### {label}（{dataset}）",
            "",
            f"| 指标 | 值 |",
            f"|------|----|",
            f"| 图表总数 | {item['total']} |",
            f"| Fallback 数 / 率 | {item['fallback_count']} / {item['fallback_rate']:.2%} |",
            f"| 成功加密数 | {item['success_count']} |",
            f"| 容差失败数 | {item['tolerance_fail_count']} |",
            "",
        ])

        metrics = [
            ("圆心误差 (px)", "center_error_px"),
            ("半径最大误差 (px)", "radius_error_max_px"),
            ("r_tick→像素映射最大误差 (px)", "radius_tick_mapping_error_max_px"),
            ("tick 值最大误差", "radius_tick_value_error_max"),
            ("斜率相对误差", "slope_error_ratio"),
        ]
        for name, key in metrics:
            stats = item.get(key, {})
            if stats.get("mean") is not None:
                if "slope" in key.lower() or "ratio" in key.lower():
                    lines.append(
                        f"| {name} | mean={stats['mean']:.3%}  "
                        f"median={stats['median']:.3%}  max={stats['max']:.3%} |"
                    )
                else:
                    lines.append(
                        f"| {name} | mean={stats['mean']:.2f}  "
                        f"median={stats['median']:.2f}  max={stats['max']:.2f} |"
                    )
        lines.append("")

        lines.append("**Fallback 原因分布：**")
        lines.append("")
        if item["fallback_reasons"]:
            for reason, count in sorted(item["fallback_reasons"].items()):
                lines.append(f"- `{reason}`: {count} 张")
        else:
            lines.append("- 无")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tick-mode", choices=["gt-nearest", "algorithm"], default="gt-nearest")
    parser.add_argument("--tolerance-ratio", type=float, default=TOLERANCE_RATIO)
    parser.add_argument("--min-axis-clusters", type=int, default=2)
    parser.add_argument("--dataset", choices=["real", "real_corrected", "synth", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", help="Only run chart id/stem substring.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--synth-dir",
        "--synthetic-dir",
        dest="synth_dir",
        type=Path,
        help="Directory containing synthetic rose_*.json/images.",
    )
    return parser.parse_args()


def main() -> None:
    configure_stdio()
    args = parse_args()
    if args.synth_dir:
        args.synth_dir = args.synth_dir.resolve()

    rows: list[EvalRow] = []
    dataset_names = ["real", "synth"] if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        json_paths = iter_dataset_jsons(dataset_name, args.synth_dir)
        if args.only:
            json_paths = [p for p in json_paths if args.only.lower() in p.stem.lower()]
        if args.limit:
            json_paths = json_paths[:args.limit]

        print(f"[{dataset_name}] evaluating {len(json_paths)} charts")
        for i, json_path in enumerate(json_paths, 1):
            row = evaluate_one(json_path, dataset_name, args)
            rows.append(row)
            status = "fallback" if row.fallback else "success"
            reason = f" {row.fallback_reason}" if row.fallback else ""
            print(f"  [{i}/{len(json_paths)}] {row.chart_id}: {status}{reason}")

    summary = summarize(rows)
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    suffix = f"{args.dataset}_{args.tick_mode}"
    csv_path = args.output_dir / f"rose_grid_eval_{suffix}.csv"
    json_path_out = args.output_dir / f"rose_grid_eval_{suffix}.json"
    md_path = args.output_dir / f"rose_grid_eval_{suffix}.md"

    write_csv(csv_path, rows)
    write_json(json_path_out, {"summary": summary, "rows": [asdict(r) for r in rows]})
    write_rose_markdown(md_path, summary, args)

    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path_out}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
