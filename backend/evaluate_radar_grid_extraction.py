"""Evaluate radar grid extraction fallback rate and geometric errors.

This script evaluates the radar preprocessing stage used before encrypted-grid
generation. It reports, separately for real and synthetic charts:

1. fallback activation rate
2. center error
3. detected radius error
4. radius-to-tick mapping error

The default tick mode is ``gt-nearest``: detected radii are paired with the
nearest ground-truth tick rings to measure grid geometry. Use
``--tick-mode algorithm`` in an OCR/LLM-ready environment to evaluate the
full tick-recognition branch.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
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


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from demo_radar.demo_radar_circle_find_1 import RadarChartEncoder  # noqa: E402


REAL_RADAR_DIR = BACKEND / "real" / "RadarChart-18 & RoseChart-6" / "RadarChart-18-final"
SYNTH_RADAR_DIR = BACKEND / "real" / "radar"
OUTPUT_DIR = ROOT / "data" / "output" / "radar_grid_eval"
POLYGON_REAL_NUMBERS = {1, 5, 6, 8, 16, 17, 18, 23}
TOLERANCE_RATIO = 0.05


@dataclass
class EvalRow:
    dataset: str
    chart_id: str
    image_path: str
    json_path: str
    fallback: bool
    fallback_reason: str
    detection_source: str
    edge_support: float
    axis_line_clusters: int
    short_side: int
    tolerance_px: float
    center_error_px: float | None
    center_error_ratio: float | None
    radius_error_mean_px: float | None
    radius_error_max_px: float | None
    radius_error_max_ratio: float | None
    radius_tick_mapping_error_mean_px: float | None
    radius_tick_mapping_error_max_px: float | None
    radius_tick_mapping_error_max_ratio: float | None
    radius_tick_value_error_mean: float | None
    radius_tick_value_error_max: float | None
    slope_error_ratio: float | None
    tolerance_pass: bool | None
    pred_center: list[int] | None
    pred_radii: list[int]
    gt_center: list[float] | None
    gt_r_pixels: list[float]
    gt_r_ticks: list[float]
    notes: str


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is {type(value).__name__}, expected object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def imread(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def normalize_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def normalize_center(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        x = normalize_number(value.get("x"))
        y = normalize_number(value.get("y"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x = normalize_number(value[0])
        y = normalize_number(value[1])
    else:
        return None
    if x is None or y is None:
        return None
    return [x, y]


def chart_number(path: Path, data: dict[str, Any] | None = None) -> int | None:
    candidates = [path.stem]
    if data and data.get("chart_id") is not None:
        candidates.append(str(data["chart_id"]))
    for candidate in candidates:
        digits = "".join(ch for ch in candidate if ch.isdigit())
        if digits:
            return int(digits)
    return None


def resolve_image(json_path: Path, data: dict[str, Any]) -> Path | None:
    image_paths = data.get("image_paths") if isinstance(data.get("image_paths"), dict) else {}
    candidates: list[Path] = []
    for key in ("no_grid", "image", "with_grid"):
        value = image_paths.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            candidates.append(path if path.is_absolute() else json_path.parent / path)
            candidates.append(ROOT / value)
            candidates.append(BACKEND / value)
    for suffix in (".png", ".jpg", ".jpeg"):
        candidates.append(json_path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def iter_dataset_jsons(dataset: str) -> list[Path]:
    if dataset == "real":
        return sorted(REAL_RADAR_DIR.glob("RadarChart*.json"), key=lambda p: chart_number(p) or 0)
    paths = []
    for path in sorted(SYNTH_RADAR_DIR.glob("radar_*.json")):
        if path.stem.endswith("_attributes"):
            continue
        paths.append(path)
    return paths


def numeric_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    numbers = []
    for value in values:
        number = normalize_number(value)
        if number is not None:
            numbers.append(number)
    return numbers


def line_distance_to_point(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    qx = x1 + t * dx
    qy = y1 + t * dy
    return math.hypot(px - qx, py - qy)


def cluster_angles(angles: list[float], tolerance_deg: float = 8.0) -> list[list[float]]:
    clusters: list[list[float]] = []
    for angle in sorted(angles):
        placed = False
        for cluster in clusters:
            center = float(np.mean(cluster))
            if abs(angle - center) <= tolerance_deg:
                cluster.append(angle)
                placed = True
                break
        if not placed:
            clusters.append([angle])
    if len(clusters) >= 2 and abs((clusters[0][0] + 180) - clusters[-1][-1]) <= tolerance_deg:
        clusters[0].extend([value - 180 for value in clusters.pop()])
    return clusters


def radial_axis_cluster_count(image: np.ndarray, center: list[int], expected_axes: int | None) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    h, w = gray.shape[:2]
    short = min(h, w)
    min_len = max(18, int(short * 0.045))
    threshold = max(10, int(short * 0.022))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=max(12, int(short * 0.05)),
    )
    if lines is None:
        return 0

    cx, cy = center
    max_center_distance = max(18, short * 0.065)
    angles = []
    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, item)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        if line_distance_to_point(x1, y1, x2, y2, cx, cy) > max_center_distance:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        angles.append(angle)
    clusters = cluster_angles(angles)
    return len(clusters)


def gt_radial_relation(ticks: list[float], pixels: list[float]) -> tuple[float, float] | None:
    pairs = [(tick, pixel) for tick, pixel in zip(ticks, pixels)]
    if len(pairs) < 2:
        return None
    tick_values = np.array([pair[0] for pair in pairs], dtype=float)
    pixel_values = np.array([pair[1] for pair in pairs], dtype=float)
    a, b = np.polyfit(tick_values, pixel_values, deg=1)
    if not math.isfinite(a) or a == 0:
        return None
    return float(a), float(b)


def match_radii_to_gt(pred_radii: list[int], gt_ticks: list[float], gt_pixels: list[float]) -> tuple[list[dict], str]:
    matches = []
    used = set()
    for radius in pred_radii:
        if radius <= 0 or not gt_pixels:
            continue
        distances = [(abs(radius - pixel), index) for index, pixel in enumerate(gt_pixels)]
        distances.sort()
        _, index = distances[0]
        matches.append(
            {
                "pred_radius": float(radius),
                "gt_radius": float(gt_pixels[index]),
                "gt_tick": float(gt_ticks[index]) if index < len(gt_ticks) else None,
                "radius_error": abs(float(radius) - float(gt_pixels[index])),
                "duplicate": index in used,
            }
        )
        used.add(index)
    duplicate_note = "duplicate_radius_tick_match" if any(item["duplicate"] for item in matches) else ""
    return matches, duplicate_note


def mapping_errors(matches: list[dict], gt_ticks: list[float], gt_pixels: list[float]) -> dict[str, float | None]:
    unique = []
    seen_ticks = set()
    for match in matches:
        tick = match.get("gt_tick")
        if tick is None or tick in seen_ticks:
            continue
        seen_ticks.add(tick)
        unique.append(match)
    if len(unique) < 2:
        return {
            "mapping_mean_px": None,
            "mapping_max_px": None,
            "value_mean": None,
            "value_max": None,
            "slope_error_ratio": None,
        }

    tick_values = np.array([item["gt_tick"] for item in unique], dtype=float)
    pred_pixels = np.array([item["pred_radius"] for item in unique], dtype=float)
    pred_a, pred_b = np.polyfit(tick_values, pred_pixels, deg=1)
    if not math.isfinite(pred_a) or abs(pred_a) <= 1e-9:
        return {
            "mapping_mean_px": None,
            "mapping_max_px": None,
            "value_mean": None,
            "value_max": None,
            "slope_error_ratio": None,
        }

    gt_tick_values = np.array(gt_ticks, dtype=float)
    gt_pixel_values = np.array(gt_pixels, dtype=float)
    pred_pixel_values = pred_a * gt_tick_values + pred_b
    pixel_errors = np.abs(pred_pixel_values - gt_pixel_values)

    pred_tick_values = (gt_pixel_values - pred_b) / pred_a
    value_errors = np.abs(pred_tick_values - gt_tick_values)

    # ── 斜率相对误差: |pred_a - gt_a| / |gt_a| ──
    if len(gt_ticks) >= 2 and len(gt_pixels) >= 2:
        gt_a = (gt_pixels[-1] - gt_pixels[0]) / (gt_ticks[-1] - gt_ticks[0])
        slope_error_ratio = abs(pred_a - gt_a) / abs(gt_a) if abs(gt_a) > 1e-9 else None
    else:
        slope_error_ratio = None

    return {
        "mapping_mean_px": float(np.mean(pixel_errors)),
        "mapping_max_px": float(np.max(pixel_errors)),
        "value_mean": float(np.mean(value_errors)),
        "value_max": float(np.max(value_errors)),
        "slope_error_ratio": float(slope_error_ratio) if slope_error_ratio is not None else None,
    }


def algorithm_tick_mapping(
    encoder: RadarChartEncoder,
    image_path: Path,
    output_dir: Path,
    chart_id: str,
    gt_ticks: list[float],
) -> tuple[list[dict], str]:
    """Run OCR/LLM tick branch and return detected radius/tick pairs."""
    image = imread(image_path)
    if image is None:
        return [], "image_unreadable_for_tick"
    temp_path = output_dir / "tick_debug" / f"{chart_id}_marked.png"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    marked = image.copy()
    cv2.circle(marked, tuple(map(int, encoder.coords)), int(encoder.first_r), (0, 255, 0), 1)
    if encoder.second_r > 0:
        cv2.circle(marked, tuple(map(int, encoder.coords)), int(encoder.second_r), (0, 255, 0), 1)
    cv2.imencode(".png", marked)[1].tofile(str(temp_path))

    matches = []
    tick1, source1, _ = encoder.ocr_find_tick(encoder.first_r, str(temp_path))
    if tick1 is not None:
        matches.append({"pred_radius": float(encoder.first_r), "gt_tick": float(tick1), "tick_source": source1})
    if encoder.second_r > 0:
        tick2, source2, _ = encoder.ocr_find_tick(encoder.second_r, str(temp_path))
        if tick2 is not None:
            matches.append({"pred_radius": float(encoder.second_r), "gt_tick": float(tick2), "tick_source": source2})

    if len({item["gt_tick"] for item in matches}) < 2:
        return matches, "algorithm_tick_mapping_insufficient"
    return matches, ""


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

        if dataset == "real" and chart_number(json_path, data) in POLYGON_REAL_NUMBERS:
            return EvalRow(
                dataset=dataset,
                chart_id=chart_id,
                image_path=image_path,
                json_path=str(json_path.resolve()),
                fallback=True,
                fallback_reason="polygon_radar_excluded",
                detection_source="",
                edge_support=0.0,
                axis_line_clusters=0,
                short_side=short_side,
                tolerance_px=tolerance_px,
                center_error_px=None,
                center_error_ratio=None,
                radius_error_mean_px=None,
                radius_error_max_px=None,
                radius_error_max_ratio=None,
                radius_tick_mapping_error_mean_px=None,
                radius_tick_mapping_error_max_px=None,
                radius_tick_mapping_error_max_ratio=None,
                radius_tick_value_error_mean=None,
                radius_tick_value_error_max=None,
                slope_error_ratio=None,
                tolerance_pass=None,
                pred_center=None,
                pred_radii=[],
                gt_center=gt_center,
                gt_r_pixels=gt_r_pixels,
                gt_r_ticks=gt_r_ticks,
                notes="known polygon radar; routed to fallback",
            )

        encoder = RadarChartEncoder()
        with contextlib.redirect_stdout(io.StringIO()):
            ring_mask = encoder.visualize_ring_mask(str(image))
            encoder.second_circle_find(ring_mask)
        pred_center = [int(encoder.coords[0]), int(encoder.coords[1])] if encoder.first_r > 0 else None
        pred_radii = [int(r) for r in [encoder.first_r, encoder.second_r] if r and r > 0]

        pass_quality, reason = encoder.check_circle_quality(cv_image.shape)
        axis_clusters = radial_axis_cluster_count(cv_image, pred_center or [0, 0], len(data.get("theta_ticks", []) or []))
        min_axis_clusters = args.min_axis_clusters

        if not pass_quality:
            fallback_reason = f"circle_quality_failed:{reason}"
        elif axis_clusters < min_axis_clusters:
            fallback_reason = f"axis_line_insufficient:{axis_clusters}<{min_axis_clusters}"
        else:
            fallback_reason = ""

        center_error = None
        center_ratio = None
        radius_mean = None
        radius_max = None
        radius_ratio = None
        mapping_mean = None
        mapping_max = None
        mapping_ratio = None
        value_mean = None
        value_max = None
        slope_err = None
        tolerance_pass = None

        if not fallback_reason and pred_center:
            center_error = float(np.linalg.norm(np.array(pred_center, dtype=float) - np.array(gt_center, dtype=float)))
            center_ratio = center_error / short_side

            if args.tick_mode == "algorithm":
                algo_matches, algo_note = algorithm_tick_mapping(encoder, image, OUTPUT_DIR, chart_id, gt_r_ticks)
                if algo_note:
                    fallback_reason = algo_note
                    notes.append(algo_note)
                    matches = []
                else:
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

            if not fallback_reason:
                radius_errors = [item["radius_error"] for item in matches if item.get("radius_error") is not None]
                if radius_errors:
                    radius_mean = float(np.mean(radius_errors))
                    radius_max = float(np.max(radius_errors))
                    radius_ratio = radius_max / short_side

                metric = mapping_errors(matches, gt_r_ticks, gt_r_pixels)
                mapping_mean = metric["mapping_mean_px"]
                mapping_max = metric["mapping_max_px"]
                mapping_ratio = mapping_max / short_side if mapping_max is not None else None
                value_mean = metric["value_mean"]
                value_max = metric["value_max"]
                slope_err = metric["slope_error_ratio"]
                if mapping_max is None:
                    notes.append("mapping_error_unavailable")

                checked = [
                    value
                    for value in (center_error, radius_max, mapping_max)
                    if value is not None
                ]
                tolerance_pass = bool(checked and max(checked) <= tolerance_px)

        return EvalRow(
            dataset=dataset,
            chart_id=chart_id,
            image_path=image_path,
            json_path=str(json_path.resolve()),
            fallback=bool(fallback_reason),
            fallback_reason=fallback_reason,
            detection_source=str(encoder.detection_source or ""),
            edge_support=round(float(encoder.last_edge_support), 4),
            axis_line_clusters=axis_clusters,
            short_side=short_side,
            tolerance_px=round(float(tolerance_px), 4),
            center_error_px=round(center_error, 4) if center_error is not None else None,
            center_error_ratio=round(center_ratio, 6) if center_ratio is not None else None,
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
            pred_center=pred_center,
            pred_radii=pred_radii,
            gt_center=gt_center,
            gt_r_pixels=gt_r_pixels,
            gt_r_ticks=gt_r_ticks,
            notes=";".join(notes),
        )
    except Exception as exc:
        return EvalRow(
            dataset=dataset,
            chart_id=chart_id,
            image_path=image_path,
            json_path=str(json_path.resolve()),
            fallback=True,
            fallback_reason=f"exception:{type(exc).__name__}:{exc}",
            detection_source="",
            edge_support=0.0,
            axis_line_clusters=0,
            short_side=0,
            tolerance_px=0.0,
            center_error_px=None,
            center_error_ratio=None,
            radius_error_mean_px=None,
            radius_error_max_px=None,
            radius_error_max_ratio=None,
            radius_tick_mapping_error_mean_px=None,
            radius_tick_mapping_error_max_px=None,
            radius_tick_mapping_error_max_ratio=None,
            radius_tick_value_error_mean=None,
            radius_tick_value_error_max=None,
            slope_error_ratio=None,
            tolerance_pass=None,
            pred_center=pred_center,
            pred_radii=pred_radii,
            gt_center=gt_center,
            gt_r_pixels=gt_r_pixels,
            gt_r_ticks=gt_r_ticks,
            notes="",
        )


def summarize(rows: list[EvalRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in sorted({row.dataset for row in rows}):
        group = [row for row in rows if row.dataset == dataset]
        fallback = [row for row in group if row.fallback]
        success = [row for row in group if not row.fallback]
        tolerance_fail = [row for row in success if row.tolerance_pass is False]

        def values(field: str) -> list[float]:
            out = []
            for row in success:
                value = getattr(row, field)
                if isinstance(value, (int, float)):
                    out.append(float(value))
            return out

        def stats(field: str) -> dict[str, float | None]:
            vals = values(field)
            if not vals:
                return {"mean": None, "median": None, "max": None}
            return {
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        reasons: dict[str, int] = {}
        for row in fallback:
            reasons[row.fallback_reason] = reasons.get(row.fallback_reason, 0) + 1

        summary[dataset] = {
            "total": len(group),
            "fallback_count": len(fallback),
            "fallback_rate": round(len(fallback) / len(group), 4) if group else None,
            "success_count": len(success),
            "tolerance_fail_count": len(tolerance_fail),
            "fallback_reasons": reasons,
            "center_error_px": stats("center_error_px"),
            "radius_error_max_px": stats("radius_error_max_px"),
            "radius_tick_mapping_error_max_px": stats("radius_tick_mapping_error_max_px"),
            "radius_tick_value_error_max": stats("radius_tick_value_error_max"),
            "slope_error_ratio": stats("slope_error_ratio"),
            "max_tolerance_px": round(max((row.tolerance_px for row in success), default=0.0), 4),
        }
    return summary


def write_csv(path: Path, rows: list[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EvalRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Radar Grid Extraction Evaluation",
        "",
        "## Fallback Mechanism",
        "",
        "A chart is routed to fallback and excluded from the CV grid-extraction pipeline when any of the following holds:",
        "",
        "1. It is a known polygon radar chart in the real set: `1, 5, 6, 8, 16, 17, 18, 23`.",
        "2. Required image or ground-truth metadata (`center`, `r_pixels`, `r_ticks`) is missing.",
        "3. Hough/circle detection fails quality checks: no circle, `detection_source == failed`, edge support `< 0.20`, or radius shorter than `8%` of the image short side.",
        f"4. Fewer than `{args.min_axis_clusters}` radial axis-line clusters are detected by Hough line support near the detected center.",
        "5. In `--tick-mode algorithm`, OCR/LLM cannot produce at least two usable radius-tick pairs.",
        "",
        f"Tolerance for reported geometry errors: `{args.tolerance_ratio:.3f} * image_short_side`.",
        f"Tick mode used in this run: `{args.tick_mode}`.",
        "",
        "## Summary",
        "",
    ]
    for dataset, item in summary.items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"- Total charts: {item['total']}",
                f"- Fallback count/rate: {item['fallback_count']} / {item['fallback_rate']:.2%}",
                f"- Successful CV charts: {item['success_count']}",
                f"- Tolerance failures among successful charts: {item['tolerance_fail_count']}",
                f"- Center error px: {item['center_error_px']}",
                f"- Max radius error px: {item['radius_error_max_px']}",
                f"- Max radius-tick mapping error px: {item['radius_tick_mapping_error_max_px']}",
                f"- Max radius-tick value error: {item['radius_tick_value_error_max']}",
                f"- Slope error ratio (|a_pred - a_gt| / |a_gt|): {item['slope_error_ratio']}",
                "",
                "Fallback reasons:",
            ]
        )
        if item["fallback_reasons"]:
            for reason, count in sorted(item["fallback_reasons"].items()):
                lines.append(f"- `{reason}`: {count}")
        else:
            lines.append("- None")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["real", "synthetic", "all"], default="all")
    parser.add_argument("--tick-mode", choices=["gt-nearest", "algorithm"], default="gt-nearest")
    parser.add_argument("--tolerance-ratio", type=float, default=TOLERANCE_RATIO)
    parser.add_argument("--min-axis-clusters", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", help="Only run chart id/stem substring.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    configure_stdio()
    args = parse_args()
    selected = ["real", "synthetic"] if args.dataset == "all" else [args.dataset]
    rows: list[EvalRow] = []
    for dataset in selected:
        paths = iter_dataset_jsons(dataset)
        if args.only:
            needle = args.only.lower()
            paths = [path for path in paths if needle in path.stem.lower()]
        if args.limit:
            paths = paths[: args.limit]
        print(f"[{dataset}] evaluating {len(paths)} charts")
        for index, path in enumerate(paths, start=1):
            row = evaluate_one(path, dataset, args)
            rows.append(row)
            status = "fallback" if row.fallback else "success"
            print(f"  [{index}/{len(paths)}] {row.chart_id}: {status} {row.fallback_reason}")

    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name_part = args.dataset
    csv_path = args.output_dir / f"radar_grid_eval_{name_part}_{args.tick_mode}.csv"
    json_path = args.output_dir / f"radar_grid_eval_{name_part}_{args.tick_mode}.json"
    md_path = args.output_dir / f"radar_grid_eval_{name_part}_{args.tick_mode}.md"
    write_csv(csv_path, rows)
    write_json(json_path, {"summary": summary, "rows": [asdict(row) for row in rows]})
    write_markdown(md_path, summary, args)

    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
