from __future__ import annotations

import argparse
import base64
import itertools
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

import cv2
import numpy as np

from grid_math import parse_numeric_label, regularity_score, is_valid_tick_series, circular_residual

def extract_line_segments(mask: np.ndarray, orientation: str) -> list[dict[str, float]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segments: list[dict[str, float]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if orientation == "horizontal":
            length = width
            thickness = height
            position = y + (height - 1) / 2
            start = x
            end = x + width - 1
        else:
            length = height
            thickness = width
            position = x + (width - 1) / 2
            start = y
            end = y + height - 1
        if length <= 0:
            continue
        segments.append(
            {
                "position": float(position),
                "start": float(start),
                "end": float(end),
                "length": float(length),
                "thickness": float(thickness),
                "area": float(area),
            }
        )
    return segments

def cluster_line_segments(segments: list[dict[str, float]], tolerance: int) -> list[dict[str, float]]:
    if not segments:
        return []

    clusters: list[list[dict[str, float]]] = []
    for segment in sorted(segments, key=lambda item: item["position"]):
        if not clusters:
            clusters.append([segment])
            continue
        positions = [item["position"] for item in clusters[-1]]
        if abs(segment["position"] - float(np.median(positions))) <= tolerance:
            clusters[-1].append(segment)
        else:
            clusters.append([segment])

    result: list[dict[str, float]] = []
    for cluster in clusters:
        weights = np.array([max(1.0, item["length"]) for item in cluster], dtype=np.float32)
        positions = np.array([item["position"] for item in cluster], dtype=np.float32)
        start = min(item["start"] for item in cluster)
        end = max(item["end"] for item in cluster)
        result.append(
            {
                "position": float(np.average(positions, weights=weights)),
                "start": float(start),
                "end": float(end),
                "span": float(end - start + 1),
                "weight": float(weights.sum()),
                "pieces": float(len(cluster)),
            }
        )
    return result

def keep_long_grid_candidates(
    clusters: list[dict[str, float]],
    image_extent: int,
    *,
    min_extent_frac: float,
) -> list[dict[str, float]]:
    if not clusters:
        return []
    spans = np.array([item["span"] for item in clusters], dtype=np.float32)
    max_span = float(spans.max())
    cutoff = max(image_extent * min_extent_frac, max_span * 0.55)
    kept = [item for item in clusters if item["span"] >= cutoff]
    return sorted(kept, key=lambda item: item["position"])

def keep_dominant_span_group(
    lines: list[dict[str, float]],
    image_extent: int,
) -> list[dict[str, float]]:
    """Keep the plot-span-consistent line group before regularizing positions."""
    if len(lines) < 4:
        return lines

    tolerance = max(12.0, image_extent * 0.04)
    best_group: list[dict[str, float]] = []
    best_score: tuple[int, float] | None = None
    for seed in lines:
        seed_start = float(seed["start"])
        seed_end = float(seed["end"])
        group = [
            item
            for item in lines
            if abs(float(item["start"]) - seed_start) <= tolerance
            and abs(float(item["end"]) - seed_end) <= tolerance
        ]
        score = (len(group), float(sum(float(item["span"]) for item in group)))
        if best_score is None or score > best_score:
            best_score = score
            best_group = group

    min_group = max(3, int(np.ceil(len(lines) * 0.5)))
    if len(best_group) < min_group or len(best_group) == len(lines):
        return lines
    best_starts = np.array([float(item["start"]) for item in best_group], dtype=np.float64)
    best_ends = np.array([float(item["end"]) for item in best_group], dtype=np.float64)
    best_spans = np.array([float(item["span"]) for item in best_group], dtype=np.float64)
    median_start = float(np.median(best_starts))
    median_end = float(np.median(best_ends))
    median_span = float(np.median(best_spans))
    recovered = list(best_group)
    for item in lines:
        if item in best_group:
            continue
        start = float(item["start"])
        end = float(item["end"])
        span = float(item["span"])
        same_plot_end = abs(end - median_end) <= tolerance
        contains_group_span = start <= median_start + tolerance and end >= median_end - tolerance
        if same_plot_end and contains_group_span and span >= median_span * 0.95:
            recovered.append(item)
    return sorted(recovered, key=lambda item: item["position"])

def infer_bounds(
    lines: list[dict[str, float]],
    positions: list[float],
    image_extent: int,
) -> tuple[int, int] | None:
    if lines:
        starts = np.array([item["start"] for item in lines], dtype=np.float32)
        ends = np.array([item["end"] for item in lines], dtype=np.float32)
        start = int(round(float(np.median(starts))))
        end = int(round(float(np.median(ends))))
    elif len(positions) >= 2:
        start = int(round(min(positions)))
        end = int(round(max(positions)))
    else:
        return None

    start = max(0, min(image_extent - 1, start))
    end = max(0, min(image_extent - 1, end))
    if end <= start:
        return None
    return start, end

def regularize_positions(
    lines: list[dict[str, float]],
    bounds: tuple[int, int] | None,
    image_extent: int,
) -> list[float]:
    positions = np.array([item["position"] for item in lines], dtype=np.float32)
    weights = np.array([max(1.0, item["span"]) for item in lines], dtype=np.float32)
    if len(positions) < 3:
        return sorted(float(item["position"]) for item in lines)

    min_step = max(6.0, image_extent * 0.012)
    max_step = max(min_step + 1.0, image_extent * 0.45)
    sorted_positions = np.sort(positions)
    candidate_steps: list[float] = []
    for i in range(len(sorted_positions)):
        for j in range(i + 1, len(sorted_positions)):
            diff = float(sorted_positions[j] - sorted_positions[i])
            if diff < min_step:
                continue
            max_divisor = min(8, int(diff // min_step))
            for divisor in range(1, max_divisor + 1):
                step = diff / divisor
                if min_step <= step <= max_step:
                    candidate_steps.append(step)

    if not candidate_steps:
        return sorted(float(item["position"]) for item in lines)

    unique_steps = sorted({round(step * 2) / 2 for step in candidate_steps})
    best: tuple[float, float, float, int] | None = None
    if bounds is None:
        score_lower = float(positions.min())
        score_upper = float(positions.max())
    else:
        score_lower, score_upper = map(float, bounds)

    for step in unique_steps:
        tolerance = max(3.0, min(8.0, step * 0.18))
        for offset in positions:
            residuals = circular_residual(positions, float(offset), step)
            matched = residuals <= tolerance
            matched_count = int(np.count_nonzero(matched))
            if matched_count < 3:
                continue

            first_k = int(np.ceil((score_lower - float(offset) - tolerance) / step))
            last_k = int(np.floor((score_upper - float(offset) + tolerance) / step))
            generated_count = max(0, last_k - first_k + 1)
            allowed_count = matched_count + max(2, int(np.ceil(matched_count * 0.35)))
            if generated_count > allowed_count:
                continue

            score = float(
                weights[matched].sum()
                + matched_count * image_extent
                - residuals[matched].sum() * 5
                - generated_count * image_extent * 0.03
            )
            if best is None or score > best[0]:
                best = (score, float(offset), float(step), matched_count)

    if best is None:
        return sorted(float(item["position"]) for item in lines)

    _, offset, step, matched_count = best
    if matched_count < max(3, int(round(len(positions) * 0.55))):
        return sorted(float(item["position"]) for item in lines)

    tolerance = max(3.0, min(8.0, step * 0.18))
    if bounds is None:
        lower = float(positions.min())
        upper = float(positions.max())
    else:
        lower, upper = map(float, bounds)

    first_k = int(np.ceil((lower - offset - tolerance) / step))
    last_k = int(np.floor((upper - offset + tolerance) / step))
    generated = [offset + k * step for k in range(first_k, last_k + 1)]
    generated = [value for value in generated if lower - tolerance <= value <= upper + tolerance]
    if len(generated) > 60:
        return sorted(float(item["position"]) for item in lines)
    return sorted(float(value) for value in generated)

def prune_irregular_edges(lines: list[dict[str, float]], max_rounds: int = 2) -> list[dict[str, float]]:
    pruned = list(lines)
    for _ in range(max_rounds):
        if len(pruned) < 5:
            break
        current = [item["position"] for item in pruned]
        best_score = regularity_score(current)
        without_first = pruned[1:]
        without_last = pruned[:-1]
        first_score = regularity_score([item["position"] for item in without_first])
        last_score = regularity_score([item["position"] for item in without_last])
        if first_score + 0.02 < best_score and first_score <= last_score:
            pruned = without_first
        elif last_score + 0.02 < best_score:
            pruned = without_last
        else:
            break
    return pruned

def reconstruct_grid(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    *,
    min_grid_span_frac: float,
    cluster_tolerance: int,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = horizontal.shape[:2]
    h_segments = extract_line_segments(horizontal, "horizontal")
    v_segments = extract_line_segments(vertical, "vertical")
    h_clusters = cluster_line_segments(h_segments, cluster_tolerance)
    v_clusters = cluster_line_segments(v_segments, cluster_tolerance)

    h_lines = keep_long_grid_candidates(h_clusters, w, min_extent_frac=min_grid_span_frac)
    v_lines = keep_long_grid_candidates(v_clusters, h, min_extent_frac=min_grid_span_frac)
    h_lines = keep_dominant_span_group(h_lines, w)
    v_lines = keep_dominant_span_group(v_lines, h)
    h_lines = prune_irregular_edges(h_lines)
    v_lines = prune_irregular_edges(v_lines)

    h_raw_positions = [float(item["position"]) for item in h_lines]
    v_raw_positions = [float(item["position"]) for item in v_lines]
    h_positions = list(h_raw_positions)
    v_positions = list(v_raw_positions)
    x_bounds = infer_bounds(h_lines, v_positions, w)
    y_bounds = infer_bounds(v_lines, h_positions, h)
    h_positions = regularize_positions(h_lines, y_bounds, h) if len(h_lines) >= 2 else []
    v_positions = regularize_positions(v_lines, x_bounds, w) if len(v_lines) >= 2 else []
    h_positions, _ = snap_positions_to_reference(
        h_positions,
        h_raw_positions,
        tolerance=max(3.0, h * 0.012),
    )
    v_positions, _ = snap_positions_to_reference(
        v_positions,
        v_raw_positions,
        tolerance=max(3.0, w * 0.012),
    )

    h_grid = np.zeros_like(horizontal)
    v_grid = np.zeros_like(vertical)
    if x_bounds is not None:
        x0, x1 = x_bounds
        for y in h_positions:
            y_i = int(round(y))
            cv2.line(h_grid, (x0, y_i), (x1, y_i), 255, thickness, cv2.LINE_AA)
    if y_bounds is not None:
        y0, y1 = y_bounds
        for x in v_positions:
            x_i = int(round(x))
            cv2.line(v_grid, (x_i, y0), (x_i, y1), 255, thickness, cv2.LINE_AA)

    return cv2.bitwise_or(h_grid, v_grid), h_grid, v_grid

def ocr_item_edges(item: dict[str, object]) -> tuple[float, float, float, float] | None:
    box = item.get("box")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            points = np.array([[float(point[0]), float(point[1])] for point in box], dtype=np.float32)
        except (TypeError, ValueError, IndexError):
            points = np.empty((0, 2), dtype=np.float32)
        if points.size:
            return (
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            )

    center = item.get("center")
    size = item.get("size")
    if (
        isinstance(center, (list, tuple))
        and len(center) >= 2
        and isinstance(size, (list, tuple))
        and len(size) >= 2
    ):
        try:
            cx = float(center[0])
            cy = float(center[1])
            width = float(size[0])
            height = float(size[1])
        except (TypeError, ValueError):
            return None
        return cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0
    return None

def semantic_axis_tick_positions(
    ocr_items: list[dict[str, object]],
    axis_key: str,
    image_shape: tuple[int, int],
) -> list[float]:
    h, w = image_shape
    positions: list[float] = []
    for item in ocr_items:
        if item.get("role") != axis_key:
            continue
        text = str(item.get("text", "") or "").strip()
        if parse_numeric_label(text) is None:
            continue
        center = item.get("center")
        raw_role = str(item.get("raw_role", "") or "")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        try:
            cx = float(center[0])
            cy = float(center[1])
        except (TypeError, ValueError):
            continue
        if axis_key == "y_axis":
            if raw_role != "y_axis" and cx > w * 0.30:
                continue
            positions.append(cy)
        else:
            if raw_role != "x_axis" and not (cy >= h * 0.62 or cy <= h * 0.22):
                continue
            positions.append(cx)
    return sorted(positions)

def semantic_plot_interval(positions: list[float], extent: int) -> tuple[float, float] | None:
    ordered = sorted(float(value) for value in positions)
    if len(ordered) < 3:
        return None
    diffs = np.diff(np.array(ordered, dtype=np.float64))
    positive = [float(diff) for diff in diffs if float(diff) > 1.0]
    if not positive:
        return None
    gap = float(np.median(np.array(positive, dtype=np.float64)))
    pad = max(8.0, min(60.0, gap * 0.72))
    return max(0.0, ordered[0] - pad), min(float(extent - 1), ordered[-1] + pad)

def ocr_text_suppression_reason(item: dict[str, object]) -> str | None:
    """Return why an OCR item should be isolated from combined-mask line evidence."""
    text = str(item.get("text", "") or "").strip()
    if not text:
        return None
    label_kind = str(item.get("label_kind", "") or "")
    role_reason = str(item.get("role_reason", "") or "")
    text_source = str(item.get("text_source", "") or "")
    if label_kind == "tick_label" or role_reason == "tick_text":
        return None
    if label_kind == "axis_label" or text_source == "mllm_axis_title":
        return "axis_label"
    role = str(item.get("role", "other") or "other")
    if role == "other":
        return "other_text"
    numeric_only_text = parse_numeric_label(text) is not None and not re.search(r"[A-Za-z\u4e00-\u9fff]", text)
    if numeric_only_text:
        return None
    if re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return "axis_or_non_tick_text"
    return None

def snap_positions_to_reference(
    positions: list[float],
    reference_positions: list[float],
    *,
    tolerance: float,
) -> tuple[list[float], list[dict[str, float]]]:
    if not positions or not reference_positions:
        return positions, []
    snapped: list[float] = []
    details: list[dict[str, float]] = []
    used: set[int] = set()
    ordered_refs = [float(value) for value in reference_positions]
    for position in positions:
        best_index = -1
        best_distance = float("inf")
        for index, reference in enumerate(ordered_refs):
            if index in used:
                continue
            distance = abs(float(position) - reference)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index >= 0 and best_distance <= tolerance:
            used.add(best_index)
            snapped_value = ordered_refs[best_index]
            snapped.append(snapped_value)
            if abs(snapped_value - float(position)) > 1e-6:
                details.append(
                    {
                        "from": round(float(position), 3),
                        "to": round(snapped_value, 3),
                        "distance": round(best_distance, 3),
                    }
                )
        else:
            snapped.append(float(position))
    return snapped, details

def redraw_line_mask(
    reference_mask: np.ndarray,
    orientation: str,
    positions: list[float],
    *,
    thickness: int,
) -> np.ndarray:
    output = np.zeros_like(reference_mask)
    _, bounds = grid_positions_and_bounds(reference_mask, orientation)
    if bounds is None:
        return output
    start, end = bounds
    for position in positions:
        value = int(round(float(position)))
        if orientation == "horizontal":
            cv2.line(output, (start, value), (end, value), 255, thickness, cv2.LINE_AA)
        else:
            cv2.line(output, (value, start), (value, end), 255, thickness, cv2.LINE_AA)
    return output

def suppress_direct_grid_text_regions(
    direct_horizontal: np.ndarray,
    direct_vertical: np.ndarray,
    ocr_items: list[dict[str, object]],
    *,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Remove direct-grid lines that are better explained by OCR text outside the plot."""
    h, w = direct_horizontal.shape[:2]
    y_tick_positions = semantic_axis_tick_positions(ocr_items, "y_axis", (h, w))
    x_tick_positions = semantic_axis_tick_positions(ocr_items, "x_axis", (h, w))
    y_plot = semantic_plot_interval(y_tick_positions, h)
    x_plot = semantic_plot_interval(x_tick_positions, w)
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_applied",
        "horizontal_removed": [],
        "vertical_removed": [],
        "y_plot_interval": [round(y_plot[0], 3), round(y_plot[1], 3)] if y_plot else None,
        "x_plot_interval": [round(x_plot[0], 3), round(x_plot[1], 3)] if x_plot else None,
    }
    if not ocr_items or (y_plot is None and x_plot is None):
        details["reason"] = "insufficient_axis_tick_bounds"
        return direct_horizontal, direct_vertical, details

    text_regions: list[dict[str, object]] = []
    for item in ocr_items:
        suppression_reason = ocr_text_suppression_reason(item)
        if suppression_reason is None:
            continue
        text = str(item.get("text", "") or "").strip()
        edges = ocr_item_edges(item)
        if edges is None:
            continue
        x0, y0, x1, y1 = edges
        if x1 <= x0 or y1 <= y0:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        outside_y_plot = bool(y_plot is not None and (cy < y_plot[0] or cy > y_plot[1]))
        outside_x_plot = bool(x_plot is not None and (cx < x_plot[0] or cx > x_plot[1]))
        if not outside_y_plot and not outside_x_plot:
            continue
        text_regions.append(
            {
                "text": text[:80],
                "edges": [x0, y0, x1, y1],
                "suppression_reason": suppression_reason,
                "outside_y_plot": outside_y_plot,
                "outside_x_plot": outside_x_plot,
            }
        )

    if not text_regions:
        details["reason"] = "no_non_axis_text_regions"
        return direct_horizontal, direct_vertical, details

    h_positions, _ = grid_positions_and_bounds(direct_horizontal, "horizontal")
    v_positions, _ = grid_positions_and_bounds(direct_vertical, "vertical")
    kept_h: list[float] = []
    kept_v: list[float] = []
    removed_h: list[dict[str, object]] = []
    removed_v: list[dict[str, object]] = []
    pad = 2.0

    for position in h_positions:
        if y_plot is not None and y_plot[0] - pad <= float(position) <= y_plot[1] + pad:
            kept_h.append(float(position))
            continue
        suppressor = None
        for region in text_regions:
            if not region["outside_y_plot"]:
                continue
            x0, y0, x1, y1 = [float(value) for value in region["edges"]]
            if y0 - pad <= position <= y1 + pad:
                suppressor = region
                break
        if suppressor is None:
            kept_h.append(float(position))
        else:
            removed_h.append({"position": round(float(position), 3), "text": suppressor["text"]})

    for position in v_positions:
        if x_plot is not None and x_plot[0] - pad <= float(position) <= x_plot[1] + pad:
            kept_v.append(float(position))
            continue
        suppressor = None
        for region in text_regions:
            if not region["outside_x_plot"]:
                continue
            x0, y0, x1, y1 = [float(value) for value in region["edges"]]
            if x0 - pad <= position <= x1 + pad:
                suppressor = region
                break
        if suppressor is None:
            kept_v.append(float(position))
        else:
            removed_v.append({"position": round(float(position), 3), "text": suppressor["text"]})

    if not removed_h and not removed_v:
        details["reason"] = "no_grid_lines_inside_text_regions"
        return direct_horizontal, direct_vertical, details

    pruned_horizontal = redraw_line_mask(direct_horizontal, "horizontal", kept_h, thickness=thickness)
    pruned_vertical = redraw_line_mask(direct_vertical, "vertical", kept_v, thickness=thickness)
    details.update(
        {
            "enabled": True,
            "reason": "ocr_text_region",
            "horizontal_removed": removed_h,
            "vertical_removed": removed_v,
            "text_region_count": len(text_regions),
        }
    )
    return pruned_horizontal, pruned_vertical, details

def suppress_line_mask_text_regions(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    ocr_items: list[dict[str, object]],
    *,
    padding: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Remove raw line-candidate pixels inside OCR text regions that are not tick labels."""
    h, w = horizontal.shape[:2]
    y_tick_positions = semantic_axis_tick_positions(ocr_items, "y_axis", (h, w))
    x_tick_positions = semantic_axis_tick_positions(ocr_items, "x_axis", (h, w))
    y_plot = semantic_plot_interval(y_tick_positions, h)
    x_plot = semantic_plot_interval(x_tick_positions, w)
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_applied",
        "regions": [],
        "horizontal_pixels_removed": 0,
        "vertical_pixels_removed": 0,
        "y_plot_interval": [round(y_plot[0], 3), round(y_plot[1], 3)] if y_plot else None,
        "x_plot_interval": [round(x_plot[0], 3), round(x_plot[1], 3)] if x_plot else None,
    }
    if not ocr_items or (y_plot is None and x_plot is None):
        details["reason"] = "insufficient_axis_tick_bounds"
        return horizontal, vertical, details

    cleaned_horizontal = horizontal.copy()
    cleaned_vertical = vertical.copy()
    total_h_removed = 0
    total_v_removed = 0
    regions: list[dict[str, object]] = []

    for item in ocr_items:
        suppression_reason = ocr_text_suppression_reason(item)
        if suppression_reason is None:
            continue
        text = str(item.get("text", "") or "").strip()
        edges = ocr_item_edges(item)
        if edges is None:
            continue
        x0, y0, x1, y1 = edges
        if x1 <= x0 or y1 <= y0:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        outside_y_plot = bool(y_plot is not None and (cy < y_plot[0] or cy > y_plot[1]))
        outside_x_plot = bool(x_plot is not None and (cx < x_plot[0] or cx > x_plot[1]))

        left = max(0, int(np.floor(x0)) - padding)
        top = max(0, int(np.floor(y0)) - padding)
        right = min(w, int(np.ceil(x1)) + padding + 1)
        bottom = min(h, int(np.ceil(y1)) + padding + 1)
        if right <= left or bottom <= top:
            continue

        h_removed = 0
        v_removed = 0
        before = int(np.count_nonzero(cleaned_horizontal[top:bottom, left:right]))
        cleaned_horizontal[top:bottom, left:right] = 0
        after = int(np.count_nonzero(cleaned_horizontal[top:bottom, left:right]))
        h_removed = before - after
        total_h_removed += h_removed
        before = int(np.count_nonzero(cleaned_vertical[top:bottom, left:right]))
        cleaned_vertical[top:bottom, left:right] = 0
        after = int(np.count_nonzero(cleaned_vertical[top:bottom, left:right]))
        v_removed = before - after
        total_v_removed += v_removed

        if h_removed or v_removed:
            regions.append(
                {
                    "text": text[:80],
                    "box": [left, top, right - 1, bottom - 1],
                    "suppression_reason": suppression_reason,
                    "outside_y_plot": outside_y_plot,
                    "outside_x_plot": outside_x_plot,
                    "horizontal_pixels_removed": h_removed,
                    "vertical_pixels_removed": v_removed,
                }
            )

    if not regions:
        details["reason"] = "no_candidate_pixels_inside_text_regions"
        return horizontal, vertical, details

    details.update(
        {
            "enabled": True,
            "reason": "ocr_non_tick_text_region_mask",
            "regions": regions,
            "horizontal_pixels_removed": total_h_removed,
            "vertical_pixels_removed": total_v_removed,
        }
    )
    return cleaned_horizontal, cleaned_vertical, details

def grid_line_count(mask: np.ndarray, orientation: str, tolerance: int = 2) -> int:
    segments = extract_line_segments(mask, orientation)
    clusters = cluster_line_segments(segments, tolerance)
    return len(clusters)

def axis_supports_tick_completion(
    axis_key: str,
    direction: str,
    tick_count: int,
    ocr_axis_evidence: dict[str, object] | None,
    mllm_result: dict[str, object] | None,
    *,
    min_lines: int,
    semantic_guard: bool,
) -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {"axis": axis_key, "semantic_guard": semantic_guard}
    if not semantic_guard:
        details["decision"] = "visual_only"
        return True, details

    ocr_axis = {}
    if isinstance(ocr_axis_evidence, dict):
        value = ocr_axis_evidence.get(axis_key)
        if isinstance(value, dict):
            ocr_axis = value
    ocr_count = int(ocr_axis.get("count", 0) or 0)
    ocr_confidence = float(ocr_axis.get("confidence", 0.0) or 0.0)
    ocr_type = str(ocr_axis.get("type", "unknown"))

    mllm_axis = {}
    recommended = {}
    mllm_enabled = False
    mllm_error = None
    if isinstance(mllm_result, dict):
        mllm_enabled = bool(mllm_result.get("enabled", False))
        mllm_error = mllm_result.get("error")
        value = mllm_result.get(axis_key)
        if isinstance(value, dict):
            mllm_axis = value
        value = mllm_result.get("recommended_grid")
        if isinstance(value, dict):
            recommended = value

    rec = str(recommended.get(direction, "unknown"))
    mllm_confidence = float(mllm_axis.get("confidence", 0.0) or 0.0)
    mllm_type = str(mllm_axis.get("type") or mllm_axis.get("kind") or "unknown")

    ocr_support = ocr_count >= min_lines and ocr_confidence >= 0.20
    mllm_support = rec in {"existing", "reconstruct"} or mllm_confidence >= 0.55
    strong_mllm_avoid = rec == "avoid" and mllm_confidence >= 0.70
    visual_fallback = tick_count >= max(min_lines + 2, 4)
    allow = (ocr_support or mllm_support or visual_fallback) and not (strong_mllm_avoid and not ocr_support)

    details.update(
        {
            "decision": "allow" if allow else "reject",
            "ocr_count": ocr_count,
            "ocr_confidence": round(ocr_confidence, 3),
            "ocr_type": ocr_type,
            "mllm_enabled": mllm_enabled,
            "mllm_error": mllm_error,
            "mllm_confidence": round(mllm_confidence, 3),
            "mllm_type": mllm_type,
            "mllm_recommendation": rec,
            "visual_fallback": visual_fallback,
        }
    )
    return allow, details

def should_replace_weak_direct_with_tick(
    direct_count: int,
    tick_count: int,
    semantic: dict[str, object],
    *,
    min_lines: int,
) -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {
        "direct_count": direct_count,
        "tick_count": tick_count,
        "reason": "not_weak_direct",
    }
    if direct_count < min_lines:
        details["reason"] = "no_direct"
        return False, details
    if direct_count > min_lines and tick_count < max(direct_count + 4, int(np.ceil(direct_count * 1.8))):
        return False, details
    if tick_count < max(4, direct_count + 2):
        details["reason"] = "insufficient_tick_advantage"
        return False, details

    rec = str(semantic.get("mllm_recommendation", "unknown"))
    ocr_count = int(semantic.get("ocr_count", 0) or 0)
    ocr_confidence = float(semantic.get("ocr_confidence", 0.0) or 0.0)
    ocr_type = str(semantic.get("ocr_type", "unknown"))
    visual_fallback = bool(semantic.get("visual_fallback", False))
    ocr_numeric_support = (
        ocr_count >= tick_count
        and ocr_confidence >= 0.55
        and ocr_type in {"numeric", "time", "mixed"}
    )
    replace = visual_fallback and (rec != "avoid" or ocr_numeric_support)
    details.update(
        {
            "reason": "tick_more_complete" if replace else "semantic_not_strong_enough",
            "mllm_recommendation": rec,
            "ocr_count": ocr_count,
            "ocr_confidence": round(ocr_confidence, 3),
            "ocr_type": ocr_type,
            "visual_fallback": visual_fallback,
        }
    )
    return replace, details

def mllm_numeric_tick_target(
    mllm_result: dict[str, object] | None,
    axis_key: str,
) -> int | None:
    if not isinstance(mllm_result, dict) or mllm_result.get("error") is not None:
        return None
    axis = mllm_result.get(axis_key, {})
    if not isinstance(axis, dict):
        return None
    ticks = axis.get("tick_labels", [])
    if not isinstance(ticks, list):
        return None
    values: list[float] = []
    for tick in ticks:
        text = str(tick.get("text", "") if isinstance(tick, dict) else tick).strip()
        numeric = parse_numeric_label(text)
        if numeric is None:
            return None
        values.append(float(numeric))
    if len(values) < 4:
        return None
    diffs = np.diff(np.array(values, dtype=np.float64))
    nonzero = [float(diff) for diff in diffs if abs(float(diff)) > 1e-9]
    if not nonzero:
        return None
    step = float(np.median(nonzero))
    residual = float(np.median(np.abs(np.array(nonzero, dtype=np.float64) - step)) / max(1.0, abs(step)))
    if residual > 0.08:
        return None
    max_residual = float(np.max(np.abs(np.array(nonzero, dtype=np.float64) - step)) / max(1.0, abs(step)))
    if max_residual > 0.35:
        return None
    return len(values)

def mllm_tick_target(
    mllm_result: dict[str, object] | None,
    axis_key: str,
) -> int | None:
    numeric_target = mllm_numeric_tick_target(mllm_result, axis_key)
    if numeric_target is not None:
        return numeric_target
    if not isinstance(mllm_result, dict) or mllm_result.get("error") is not None:
        return None
    axis = mllm_result.get(axis_key, {})
    if not isinstance(axis, dict):
        return None
    ticks = axis.get("tick_labels", [])
    if not isinstance(ticks, list):
        return None
    labels = [
        str(tick.get("text", "") if isinstance(tick, dict) else tick).strip()
        for tick in ticks
    ]
    labels = [label for label in labels if label and label.lower() not in {"none", "unknown", "null"}]
    if len(labels) < 4:
        return None
    try:
        confidence = float(axis.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.75:
        return None
    return len(labels)

def is_regular_position_sequence(positions: list[float]) -> bool:
    if len(positions) < 3:
        return True
    diffs = np.diff(np.array(sorted(positions), dtype=np.float64))
    positive = [float(diff) for diff in diffs if float(diff) > 1e-6]
    if len(positive) != len(diffs):
        return False
    median = float(np.median(np.array(positive, dtype=np.float64)))
    if median <= 1.0:
        return False
    residual = np.abs(np.array(positive, dtype=np.float64) - median)
    min_gap = float(min(positive))
    mean_residual = float(np.mean(residual) / max(1.0, median))
    max_residual = float(np.max(residual) / max(1.0, median))
    return min_gap >= median * 0.45 and mean_residual <= 0.20 and max_residual <= 0.55


def regular_subset(
    positions: list[float],
    target_count: int,
) -> list[float] | None:
    ordered = sorted(float(value) for value in positions)
    if target_count <= 0 or len(ordered) < target_count:
        return None
    if len(ordered) == target_count:
        return ordered if is_regular_position_sequence(ordered) else None
    if len(ordered) <= 18 and target_count <= 14:
        candidates = itertools.combinations(ordered, target_count)
    else:
        candidates = (ordered[index : index + target_count] for index in range(len(ordered) - target_count + 1))

    def score(subset_tuple: tuple[float, ...] | list[float]) -> tuple[float, float]:
        subset = list(subset_tuple)
        diffs = np.diff(np.array(subset, dtype=np.float64))
        median = float(np.median(diffs)) if len(diffs) else 0.0
        if median <= 0:
            return (1e9, 1e9)
        abs_residual = np.abs(diffs - median)
        residual = float(np.mean(abs_residual) / max(1.0, abs(median)))
        max_residual = float(np.max(abs_residual) / max(1.0, abs(median)))
        span_penalty = -float(subset[-1] - subset[0]) / 10000.0
        return (residual + max_residual * 0.2, span_penalty)

    best = min(candidates, key=score, default=None)
    if best is None:
        return None
    selected = list(best)
    return selected if is_regular_position_sequence(selected) else None

def prune_grid_to_mllm_regular_ticks(
    candidate_mask: np.ndarray,
    orientation: str,
    target_count: int | None,
    *,
    thickness: int,
    guide_positions: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_needed",
        "target_count": target_count,
        "input_count": 0,
        "positions": [],
    }
    if target_count is None:
        details["reason"] = "no_mllm_numeric_target"
        return candidate_mask, details
    positions, bounds = grid_positions_and_bounds(candidate_mask, orientation)
    details["input_count"] = len(positions)
    if bounds is None or len(positions) <= target_count:
        details["reason"] = "not_overcomplete"
        return candidate_mask, details
    selected = regular_subset(positions, target_count)
    if guide_positions and len(guide_positions) >= max(3, int(target_count * 0.5)):
        ordered = sorted(float(value) for value in positions)
        if len(ordered) <= 18 and target_count <= 14:
            subsets = itertools.combinations(ordered, target_count)
        else:
            subsets = (ordered[index : index + target_count] for index in range(len(ordered) - target_count + 1))

        guides = [float(value) for value in guide_positions]

        def anchored_score(subset_tuple: tuple[float, ...] | list[float]) -> tuple[float, float]:
            subset = list(subset_tuple)
            diffs = np.diff(np.array(subset, dtype=np.float64))
            median = float(np.median(diffs)) if len(diffs) else 0.0
            if median <= 0:
                return (1e9, 1e9)
            abs_residual = np.abs(diffs - median)
            residual = float(np.mean(abs_residual) / max(1.0, abs(median)))
            max_residual = float(np.max(abs_residual) / max(1.0, abs(median)))
            if len(guides) == len(subset):
                guide_error = float(np.mean(np.abs(np.array(sorted(guides), dtype=np.float64) - np.array(subset, dtype=np.float64))) / max(1.0, abs(median)))
            else:
                guide_error = float(np.mean([min(abs(guide - pos) for pos in subset) for guide in guides]) / max(1.0, abs(median)))
            span_penalty = -float(subset[-1] - subset[0]) / 10000.0
            return (guide_error * 2.5 + residual + max_residual * 0.2, span_penalty)

        anchored = min(subsets, key=anchored_score, default=None)
        if anchored is not None:
            anchored_selected = list(anchored)
            if is_regular_position_sequence(anchored_selected):
                selected = anchored_selected
    if selected is None or len(selected) != target_count:
        details["reason"] = "no_regular_subset"
        return candidate_mask, details
    if set(round(value, 3) for value in selected) == set(round(value, 3) for value in positions):
        details["reason"] = "subset_unchanged"
        return candidate_mask, details
    pruned = np.zeros_like(candidate_mask)
    start, end = bounds
    for position in selected:
        value = int(round(position))
        if orientation == "horizontal":
            cv2.line(pruned, (start, value), (end, value), 255, thickness, cv2.LINE_AA)
        else:
            cv2.line(pruned, (value, start), (value, end), 255, thickness, cv2.LINE_AA)
    details.update(
        {
            "enabled": True,
            "reason": "mllm_regular_numeric_subset",
            "positions": [round(value, 3) for value in selected],
        }
    )
    return pruned, details

def complete_regular_grid_to_target(
    candidate_mask: np.ndarray,
    orientation: str,
    target_count: int | None,
    *,
    thickness: int,
    guide_positions: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_needed",
        "target_count": target_count,
        "input_count": 0,
        "positions": [],
    }
    if target_count is None:
        details["reason"] = "no_mllm_target"
        return candidate_mask, details
    positions, bounds = grid_positions_and_bounds(candidate_mask, orientation)
    details["input_count"] = len(positions)
    if bounds is None or len(positions) >= target_count or len(positions) < 2:
        details["reason"] = "not_undercomplete"
        return candidate_mask, details

    ordered = sorted(float(value) for value in positions)
    diffs = np.diff(np.array(ordered, dtype=np.float64))
    positive = sorted(float(diff) for diff in diffs if float(diff) > 1.0)
    if not positive:
        details["reason"] = "no_positive_gap"
        return candidate_mask, details
    small = positive[: max(1, int(np.ceil(len(positive) * 0.65)))]
    step = float(np.median(np.array(small, dtype=np.float64)))
    if step <= 1.0:
        details["reason"] = "invalid_step"
        return candidate_mask, details

    missing = target_count - len(ordered)
    if missing > 20:
        details["reason"] = "too_many_missing"
        return candidate_mask, details

    axis_extent = candidate_mask.shape[1] if orientation == "vertical" else candidate_mask.shape[0]
    guides = [float(value) for value in (guide_positions or [])]
    guide_selected: list[float] | None = None
    if len(guides) >= target_count:
        if len(guides) == target_count and is_regular_position_sequence(guides):
            guide_selected = sorted(guides)
        else:
            guide_selected = regular_subset(guides, target_count)
    if guide_selected is not None and len(guide_selected) == target_count:
        guide_diffs = np.diff(np.array(sorted(guide_selected), dtype=np.float64))
        guide_step = float(np.median(guide_diffs)) if len(guide_diffs) else step
        preserve_tolerance = max(3.0, min(8.0, guide_step * 0.12))
        coverage_tolerance = max(5.0, min(14.0, guide_step * 0.22))
        selected: list[float] = []
        guide_anchor_details: list[dict[str, float | str]] = []
        for guide in sorted(guide_selected):
            nearest = min(ordered, key=lambda value: abs(value - guide))
            distance = abs(nearest - guide)
            if distance <= preserve_tolerance:
                selected.append(float(nearest))
                guide_anchor_details.append(
                    {
                        "guide": round(float(guide), 3),
                        "selected": round(float(nearest), 3),
                        "source": "combined_mask",
                    }
                )
            else:
                selected.append(float(guide))
                guide_anchor_details.append(
                    {
                        "guide": round(float(guide), 3),
                        "selected": round(float(guide), 3),
                        "source": "ocr_label_lab",
                    }
                )
        selected = sorted(selected)
        covers_existing = all(
            min(abs(existing - value) for value in selected) <= coverage_tolerance
            for existing in ordered
        )
        distinct = all(
            abs(selected[index + 1] - selected[index]) > max(2.0, guide_step * 0.2)
            for index in range(len(selected) - 1)
        )
        in_bounds = all(-2 <= value <= axis_extent + 1 for value in selected)
        if covers_existing and distinct and in_bounds and is_regular_position_sequence(selected):
            completed = candidate_mask.copy()
            start_bound, end_bound = bounds
            kept_positions: list[float] = []
            added_positions: list[float] = []
            for position in selected:
                value = int(round(position))
                if value < 0 or value >= axis_extent:
                    continue
                kept_positions.append(float(position))
                if min(abs(float(position) - existing) for existing in ordered) <= preserve_tolerance:
                    continue
                added_positions.append(float(position))
                if orientation == "horizontal":
                    cv2.line(completed, (start_bound, value), (end_bound, value), 255, thickness, cv2.LINE_AA)
                else:
                    cv2.line(completed, (value, start_bound), (value, end_bound), 255, thickness, cv2.LINE_AA)
            completed_positions, _ = grid_positions_and_bounds(completed, orientation)
            if len(completed_positions) >= target_count:
                details.update(
                    {
                        "enabled": True,
                        "reason": "ocr_label_anchored_tick_completion_preserve_native",
                        "step": round(guide_step, 3),
                        "positions": [round(value, 3) for value in completed_positions],
                        "added_positions": [round(value, 3) for value in added_positions],
                        "guide_positions": [round(value, 3) for value in sorted(guide_selected)],
                        "preserve_tolerance": round(preserve_tolerance, 3),
                        "anchors": guide_anchor_details,
                    }
                )
                return completed, details

    sequences: list[list[float]] = []
    for prepend in range(missing + 1):
        start = ordered[0] - prepend * step
        sequences.append([start + index * step for index in range(target_count)])

    def score(seq: list[float]) -> tuple[float, float]:
        outside = sum(1 for value in seq if value < -2 or value > axis_extent + 1)
        tolerance = max(4.0, step * 0.18)
        coverage = sum(min(abs(pos - value) for value in seq) <= tolerance for pos in ordered)
        guide_error = 0.0
        if guides:
            guide_error = float(np.mean([min(abs(guide - value) for value in seq) for guide in guides]) / max(1.0, step))
        span_penalty = -float(seq[-1] - seq[0]) / 10000.0
        return (outside * 100.0 + (len(ordered) - coverage) * 10.0 + guide_error, span_penalty)

    selected = min(sequences, key=score)
    if score(selected)[0] >= 50.0:
        details["reason"] = "completion_score_too_weak"
        details["score"] = round(score(selected)[0], 3)
        return candidate_mask, details

    completion_tolerance = max(4.0, step * 0.18)
    completed = candidate_mask.copy()
    start_bound, end_bound = bounds
    added_positions: list[float] = []
    for position in selected:
        if min(abs(float(position) - existing) for existing in ordered) <= completion_tolerance:
            continue
        value = int(round(position))
        if value < 0 or value >= axis_extent:
            continue
        added_positions.append(float(position))
        if orientation == "horizontal":
            cv2.line(completed, (start_bound, value), (end_bound, value), 255, thickness, cv2.LINE_AA)
        else:
            cv2.line(completed, (value, start_bound), (value, end_bound), 255, thickness, cv2.LINE_AA)
    completed_positions, _ = grid_positions_and_bounds(completed, orientation)
    if len(completed_positions) < target_count:
        details["reason"] = "completion_out_of_bounds"
        return candidate_mask, details
    details.update(
        {
            "enabled": True,
            "reason": "mllm_regular_tick_completion_preserve_native",
            "step": round(step, 3),
            "positions": [round(value, 3) for value in completed_positions],
            "added_positions": [round(value, 3) for value in added_positions],
            "selected_sequence": [round(value, 3) for value in selected],
            "preserve_tolerance": round(completion_tolerance, 3),
        }
    )
    return completed, details

def merge_grid_by_hierarchy(
    direct_horizontal: np.ndarray,
    direct_vertical: np.ndarray,
    tick_horizontal: np.ndarray,
    tick_vertical: np.ndarray,
    *,
    min_lines: int,
    ocr_axis_evidence: dict[str, object] | None = None,
    mllm_result: dict[str, object] | None = None,
    semantic_guard: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    direct_h_count = grid_line_count(direct_horizontal, "horizontal")
    direct_v_count = grid_line_count(direct_vertical, "vertical")
    tick_h_count = grid_line_count(tick_horizontal, "horizontal")
    tick_v_count = grid_line_count(tick_vertical, "vertical")
    h_target = mllm_tick_target(mllm_result, "y_axis")
    v_target = mllm_tick_target(mllm_result, "x_axis")

    allow_tick_h, h_semantic = axis_supports_tick_completion(
        "y_axis",
        "horizontal",
        tick_h_count,
        ocr_axis_evidence,
        mllm_result,
        min_lines=min_lines,
        semantic_guard=semantic_guard,
    )
    allow_tick_v, v_semantic = axis_supports_tick_completion(
        "x_axis",
        "vertical",
        tick_v_count,
        ocr_axis_evidence,
        mllm_result,
        min_lines=min_lines,
        semantic_guard=semantic_guard,
    )
    replace_direct_h, h_direct_override = should_replace_weak_direct_with_tick(
        direct_h_count,
        tick_h_count,
        h_semantic,
        min_lines=min_lines,
    )
    replace_direct_v, v_direct_override = should_replace_weak_direct_with_tick(
        direct_v_count,
        tick_v_count,
        v_semantic,
        min_lines=min_lines,
    )
    if h_target is not None and direct_h_count >= min_lines and direct_h_count == h_target:
        replace_direct_h = False
        h_direct_override = {
            "direct_count": direct_h_count,
            "tick_count": tick_h_count,
            "reason": "direct_matches_mllm_target",
            "mllm_target": h_target,
        }
    if v_target is not None and direct_v_count >= min_lines and direct_v_count == v_target:
        replace_direct_v = False
        v_direct_override = {
            "direct_count": direct_v_count,
            "tick_count": tick_v_count,
            "reason": "direct_matches_mllm_target",
            "mllm_target": v_target,
        }
    if (
        h_target is not None
        and direct_h_count >= min_lines
        and direct_h_count >= max(2, h_target - 1)
        and tick_h_count > max(h_target + 3, int(np.ceil(h_target * 1.7)))
    ):
        replace_direct_h = False
        h_direct_override = {
            "direct_count": direct_h_count,
            "tick_count": tick_h_count,
            "reason": "tick_overcomplete_vs_mllm_target",
            "mllm_target": h_target,
        }
    if (
        v_target is not None
        and direct_v_count >= min_lines
        and direct_v_count >= max(2, v_target - 1)
        and tick_v_count > max(v_target + 3, int(np.ceil(v_target * 1.7)))
    ):
        replace_direct_v = False
        v_direct_override = {
            "direct_count": direct_v_count,
            "tick_count": tick_v_count,
            "reason": "tick_overcomplete_vs_mllm_target",
            "mllm_target": v_target,
        }
    replace_direct_h = replace_direct_h and allow_tick_h
    replace_direct_v = replace_direct_v and allow_tick_v

    def ocr_numeric_tick_positions(axis_key: str, position_key: str) -> list[float]:
        if not isinstance(ocr_axis_evidence, dict):
            return []
        axis = ocr_axis_evidence.get(axis_key, {})
        if not isinstance(axis, dict):
            return []
        ticks = axis.get("ticks", [])
        if not isinstance(ticks, list):
            return []
        values: list[float] = []
        for tick in ticks:
            if not isinstance(tick, dict) or tick.get("numeric") is None:
                continue
            try:
                values.append(float(tick[position_key]))
            except (KeyError, TypeError, ValueError):
                continue
        return values

    use_direct_h = direct_h_count >= min_lines and not replace_direct_h
    use_direct_v = direct_v_count >= min_lines and not replace_direct_v
    use_tick_h = not use_direct_h and tick_h_count >= min_lines and allow_tick_h
    use_tick_v = not use_direct_v and tick_v_count >= min_lines and allow_tick_v

    final_horizontal = direct_horizontal if use_direct_h else np.zeros_like(direct_horizontal)
    final_vertical = direct_vertical if use_direct_v else np.zeros_like(direct_vertical)
    if use_tick_h:
        final_horizontal = cv2.bitwise_or(final_horizontal, tick_horizontal)
    if use_tick_v:
        final_vertical = cv2.bitwise_or(final_vertical, tick_vertical)

    h_prune_details: dict[str, object] = {"enabled": False, "reason": "not_applied"}
    v_prune_details: dict[str, object] = {"enabled": False, "reason": "not_applied"}
    h_completion_details: dict[str, object] = {"enabled": False, "reason": "not_applied"}
    v_completion_details: dict[str, object] = {"enabled": False, "reason": "not_applied"}
    if use_tick_h or replace_direct_h:
        h_candidates = cv2.bitwise_or(final_horizontal, direct_horizontal)
        final_horizontal, h_prune_details = prune_grid_to_mllm_regular_ticks(
            h_candidates,
            "horizontal",
            h_target,
            thickness=1,
            guide_positions=ocr_numeric_tick_positions("y_axis", "y"),
        )
        final_horizontal, h_completion_details = complete_regular_grid_to_target(
            final_horizontal,
            "horizontal",
            h_target,
            thickness=1,
            guide_positions=ocr_numeric_tick_positions("y_axis", "y"),
        )
    if use_tick_v or replace_direct_v:
        v_candidates = cv2.bitwise_or(final_vertical, direct_vertical)
        final_vertical, v_prune_details = prune_grid_to_mllm_regular_ticks(
            v_candidates,
            "vertical",
            v_target,
            thickness=1,
            guide_positions=ocr_numeric_tick_positions("x_axis", "x"),
        )
        final_vertical, v_completion_details = complete_regular_grid_to_target(
            final_vertical,
            "vertical",
            v_target,
            thickness=1,
            guide_positions=ocr_numeric_tick_positions("x_axis", "x"),
        )
    if (
        use_direct_h
        and h_target is not None
        and direct_h_count > max(h_target + 3, int(np.ceil(h_target * 1.7)))
        and h_semantic.get("decision") == "allow"
    ):
        pruned_horizontal, h_direct_prune = prune_grid_to_mllm_regular_ticks(
            final_horizontal,
            "horizontal",
            h_target,
            thickness=1,
            guide_positions=ocr_numeric_tick_positions("y_axis", "y"),
        )
        if h_direct_prune.get("enabled"):
            final_horizontal = pruned_horizontal
            h_prune_details = h_direct_prune
    if (
        use_direct_v
        and v_target is not None
        and direct_v_count > max(v_target + 3, int(np.ceil(v_target * 1.7)))
        and v_semantic.get("decision") == "allow"
    ):
        pruned_vertical, v_direct_prune = prune_grid_to_mllm_regular_ticks(
            final_vertical,
            "vertical",
            v_target,
            thickness=1,
            guide_positions=ocr_numeric_tick_positions("x_axis", "x"),
        )
        if v_direct_prune.get("enabled"):
            final_vertical = pruned_vertical
            v_prune_details = v_direct_prune

    metadata: dict[str, object] = {
        "direct_horizontal_count": direct_h_count,
        "direct_vertical_count": direct_v_count,
        "tick_horizontal_count": tick_h_count,
        "tick_vertical_count": tick_v_count,
        "horizontal_source": "combined_mask" if use_direct_h else ("tick" if use_tick_h else "none"),
        "vertical_source": "combined_mask" if use_direct_v else ("tick" if use_tick_v else "none"),
        "horizontal_semantic": h_semantic,
        "vertical_semantic": v_semantic,
        "horizontal_direct_override": h_direct_override,
        "vertical_direct_override": v_direct_override,
        "horizontal_mllm_prune": h_prune_details,
        "vertical_mllm_prune": v_prune_details,
        "horizontal_mllm_completion": h_completion_details,
        "vertical_mllm_completion": v_completion_details,
    }
    return cv2.bitwise_or(final_horizontal, final_vertical), final_horizontal, final_vertical, metadata

def median_gap(positions: list[float]) -> float | None:
    if len(positions) < 2:
        return None
    diffs = np.diff(np.array(sorted(positions), dtype=np.float32))
    positive = [float(diff) for diff in diffs if float(diff) > 1e-6]
    if not positive:
        return None
    return float(np.median(positive))

def ocr_numeric_value(item: dict[str, object]) -> float | None:
    text = str(item.get("text", "") or "").strip()
    return parse_numeric_label(text) if text else None

def ocr_center(item: dict[str, object]) -> tuple[float, float] | None:
    center = item.get("center")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    try:
        return float(center[0]), float(center[1])
    except (TypeError, ValueError):
        return None

def ocr_box_edges(item: dict[str, object]) -> tuple[float, float, float, float] | None:
    box = item.get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        center = ocr_center(item)
        size = item.get("size")
        if center is None or not isinstance(size, (list, tuple)) or len(size) < 2:
            return None
        try:
            cx, cy = center
            width = float(size[0])
            height = float(size[1])
        except (TypeError, ValueError):
            return None
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
    try:
        points = np.array(box, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[1] < 2:
        return None
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )

def ocr_axis_anchor_x(item: dict[str, object], side: str) -> float | None:
    box = item.get("box")
    center = ocr_center(item)
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return center[0] if center is not None else None
    try:
        points = np.array(box, dtype=np.float32)
    except (TypeError, ValueError):
        return center[0] if center is not None else None
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
        return center[0] if center is not None else None
    order = np.argsort(points[:, 1])
    edge = points[order[:2]] if side == "bottom" else points[order[-2:]]
    if edge.shape[0] < 2:
        return center[0] if center is not None else None
    return float(np.mean(edge[:, 0]))


def bottom_numeric_ocr_row(
    ocr_items: list[dict[str, object]] | None,
    image_shape: tuple[int, int],
    *,
    min_count: int,
) -> list[dict[str, object]]:
    if not ocr_items:
        return []
    h, w = image_shape
    candidates: list[dict[str, object]] = []
    for item in ocr_items:
        center = ocr_center(item)
        edges = ocr_box_edges(item)
        numeric = ocr_numeric_value(item)
        text = str(item.get("text", "") or "").strip()
        if center is None or edges is None or numeric is None or not text:
            continue
        cx, cy = center
        if cy < h * 0.68:
            continue
        candidates.append(
            {
                "text": text,
                "numeric": numeric,
                "x": cx,
                "y": cy,
                "x0": edges[0],
                "y0": edges[1],
                "x1": edges[2],
                "y1": edges[3],
            }
        )
    row = choose_bottom_ocr_label_row(candidates, h)
    if len(row) < min_count:
        return []
    row = choose_regular_numeric_window(row, min_count=min_count)
    if len(row) < min_count:
        return []
    xs = [float(item["x"]) for item in row]
    if max(xs) - min(xs) < w * 0.42:
        return []
    numeric_values = [float(item["numeric"]) for item in row]
    diffs = np.diff(np.array(numeric_values, dtype=np.float64))
    if len(diffs) and not (np.all(diffs >= 0) or np.all(diffs <= 0)):
        return []
    if len(xs) >= 4 and regularity_score(xs) > 0.45:
        return []
    return row

def x_axis_numeric_ocr_row(
    ocr_items: list[dict[str, object]] | None,
    image_shape: tuple[int, int],
    *,
    min_count: int,
) -> tuple[list[dict[str, object]], str]:
    if not ocr_items:
        return [], "none"
    h, w = image_shape
    candidates_by_side: dict[str, list[dict[str, object]]] = {"top": [], "bottom": []}
    for item in ocr_items:
        center = ocr_center(item)
        edges = ocr_box_edges(item)
        numeric = ocr_numeric_value(item)
        text = str(item.get("text", "") or "").strip()
        role = str(item.get("role", "other"))
        if center is None or edges is None or numeric is None or not text:
            continue
        cx, cy = center
        if role != "x_axis" and not (cy <= h * 0.22 or cy >= h * 0.68):
            continue
        payload = {
            "text": text,
            "numeric": numeric,
            "x": cx,
            "y": cy,
            "x0": edges[0],
            "y0": edges[1],
            "x1": edges[2],
            "y1": edges[3],
        }
        if cy <= h * 0.28:
            candidates_by_side["top"].append(payload)
        if cy >= h * 0.68:
            candidates_by_side["bottom"].append(payload)

    best_side = "none"
    best_row: list[dict[str, object]] = []
    best_score = -1.0
    for side, candidates in candidates_by_side.items():
        if len(candidates) < min_count:
            continue
        row = choose_ocr_label_row(candidates, h, preference=side)
        if len(row) < min_count:
            continue
        xs = [float(item["x"]) for item in row]
        if max(xs) - min(xs) < w * 0.42:
            continue
        numeric_values = [float(item["numeric"]) for item in row]
        diffs = np.diff(np.array(numeric_values, dtype=np.float64))
        if len(diffs) and not (np.all(diffs >= 0) or np.all(diffs <= 0)):
            continue
        if len(xs) >= 4 and regularity_score(xs) > 0.45:
            continue
        score = len(row) * 1000.0 + (max(xs) - min(xs))
        if side == "bottom":
            score += 5.0
        if score > best_score:
            best_score = score
            best_side = side
            best_row = sorted(row, key=lambda value: float(value["x"]))
    return best_row, best_side

def x_axis_category_ocr_row(
    ocr_items: list[dict[str, object]] | None,
    image_shape: tuple[int, int],
    *,
    min_count: int,
) -> tuple[list[dict[str, object]], str]:
    if not ocr_items:
        return [], "none"
    h, w = image_shape
    candidates_by_side: dict[str, list[dict[str, object]]] = {"top": [], "bottom": []}
    for item in ocr_items:
        center = ocr_center(item)
        edges = ocr_box_edges(item)
        text = str(item.get("text", "") or "").strip()
        role = str(item.get("role", "other"))
        label_kind = str(item.get("label_kind", "tick_label") or "tick_label")
        if center is None or edges is None or not text or role != "x_axis" or label_kind == "axis_label":
            continue
        cx, cy = center
        if cy <= h * 0.34:
            side = "top"
        elif cy >= h * 0.60:
            side = "bottom"
        else:
            continue
        anchor = ocr_axis_anchor_x(item, side)
        payload = {
            "text": text,
            "numeric": None,
            "x": cx,
            "anchor_x": anchor if anchor is not None else cx,
            "y": cy,
            "x0": edges[0],
            "y0": edges[1],
            "x1": edges[2],
            "y1": edges[3],
            "canonical_index": item.get("canonical_index"),
        }
        candidates_by_side[side].append(payload)

    best_side = "none"
    best_row: list[dict[str, object]] = []
    best_score = -1.0
    for side, candidates in candidates_by_side.items():
        if len(candidates) < min_count:
            continue
        row = choose_ocr_label_row(candidates, h, preference=side)
        if len(row) < min_count:
            continue
        anchors = [float(item.get("anchor_x", item["x"])) for item in row]
        if max(anchors) - min(anchors) < w * 0.35:
            continue
        if len(anchors) >= 4 and regularity_score(anchors) > 0.55:
            continue

        def order_key(item: dict[str, object]) -> tuple[int, float]:
            try:
                return 0, float(item.get("canonical_index"))
            except (TypeError, ValueError):
                return 1, float(item.get("anchor_x", item["x"]))

        ordered = sorted(row, key=order_key)
        score = len(ordered) * 1000.0 + (max(anchors) - min(anchors))
        if side == "bottom":
            score += 5.0
        if score > best_score:
            best_score = score
            best_side = side
            best_row = ordered
    return best_row, best_side


def choose_regular_numeric_window(
    row: list[dict[str, object]],
    *,
    min_count: int,
    order_key: str = "x",
) -> list[dict[str, object]]:
    ordered = sorted(row, key=lambda value: float(value[order_key]))
    best: tuple[float, list[dict[str, object]]] | None = None
    for start in range(len(ordered)):
        for end in range(start + min_count, len(ordered) + 1):
            window = ordered[start:end]
            values = [float(item["numeric"]) for item in window if item.get("numeric") is not None]
            if len(values) != len(window):
                continue
            diffs = np.diff(np.array(values, dtype=np.float64))
            if len(diffs) == 0:
                continue
            if not (np.all(diffs >= 0) or np.all(diffs <= 0)):
                continue
            nonzero = [abs(float(diff)) for diff in diffs if abs(float(diff)) > 1e-9]
            if not nonzero:
                continue
            step = float(np.median(nonzero))
            residual = float(np.max(np.abs(np.array(nonzero, dtype=np.float64) - step)))
            if residual > max(1e-6, step * 0.12):
                continue
            positions = [float(item[order_key]) for item in window]
            if len(positions) >= 4 and regularity_score(positions) > 0.45:
                continue
            score = len(window) * 1000.0 + (max(positions) - min(positions))
            if best is None or score > best[0]:
                best = (score, window)
    return best[1] if best is not None else []


def left_numeric_ocr_column(
    ocr_items: list[dict[str, object]] | None,
    image_shape: tuple[int, int],
    *,
    min_count: int,
) -> list[dict[str, object]]:
    if not ocr_items:
        return []
    h, w = image_shape
    candidates: list[dict[str, object]] = []
    for item in ocr_items:
        center = ocr_center(item)
        edges = ocr_box_edges(item)
        numeric = ocr_numeric_value(item)
        text = str(item.get("text", "") or "").strip()
        if center is None or edges is None or numeric is None or not text:
            continue
        cx, cy = center
        if cx > w * 0.28:
            continue
        candidates.append(
            {
                "text": text,
                "numeric": numeric,
                "x": cx,
                "y": cy,
                "x0": edges[0],
                "y0": edges[1],
                "x1": edges[2],
                "y1": edges[3],
            }
        )
    if len(candidates) < min_count:
        return []
    tolerance = max(12.0, w * 0.025)
    columns: list[list[dict[str, object]]] = []
    for item in sorted(candidates, key=lambda value: float(value["x"])):
        if not columns:
            columns.append([item])
            continue
        median_x = float(np.median([float(value["x"]) for value in columns[-1]]))
        if abs(float(item["x"]) - median_x) <= tolerance:
            columns[-1].append(item)
        else:
            columns.append([item])

    def column_score(column: list[dict[str, object]]) -> float:
        ys = [float(item["y"]) for item in column]
        span = max(ys) - min(ys) if len(ys) >= 2 else 0.0
        return len(column) * 1000.0 + span - float(np.median([item["x"] for item in column])) * 0.1

    column = sorted(max(columns, key=column_score), key=lambda value: float(value["y"]))
    if len(column) < min_count:
        return []
    column = choose_regular_numeric_window(column, min_count=min_count, order_key="y")
    if len(column) < min_count:
        return []
    ys = [float(item["y"]) for item in column]
    if max(ys) - min(ys) < h * 0.25:
        return []
    numeric_values = [float(item["numeric"]) for item in column]
    diffs = np.diff(np.array(numeric_values, dtype=np.float64))
    if len(diffs) and not (np.all(diffs >= 0) or np.all(diffs <= 0)):
        return []
    if len(ys) >= 4 and regularity_score(ys) > 0.45:
        return []
    return column


def isolate_tick_label_boxes_from_mask(
    tick_mask: np.ndarray,
    labels: list[dict[str, object]],
    *,
    padding: int = 1,
) -> np.ndarray:
    """Remove the OCR tick-label glyph boxes before local tick scanning."""
    if not labels:
        return tick_mask
    h, w = tick_mask.shape[:2]
    cleaned = tick_mask.copy()
    for label in labels:
        try:
            x0 = float(label["x0"])
            y0 = float(label["y0"])
            x1 = float(label["x1"])
            y1 = float(label["y1"])
        except (KeyError, TypeError, ValueError):
            continue
        left = max(0, int(np.floor(min(x0, x1))) - padding)
        top = max(0, int(np.floor(min(y0, y1))) - padding)
        right = min(w, int(np.ceil(max(x0, x1))) + padding + 1)
        bottom = min(h, int(np.ceil(max(y0, y1))) + padding + 1)
        if right <= left or bottom <= top:
            continue
        cleaned[top:bottom, left:right] = 0
    return cleaned


def isolate_axis_tick_label_boxes_for_scan(
    tick_mask: np.ndarray,
    image_shape: tuple[int, int],
    ocr_items: list[dict[str, object]] | None,
    *,
    min_count: int = 4,
) -> np.ndarray:
    if not ocr_items:
        return tick_mask
    x_labels, _ = x_axis_numeric_ocr_row(ocr_items, image_shape, min_count=min_count)
    if not x_labels:
        x_labels, _ = x_axis_category_ocr_row(ocr_items, image_shape, min_count=min_count)
    y_labels = left_numeric_ocr_column(ocr_items, image_shape, min_count=min_count)
    labels = [*x_labels, *y_labels]
    if not labels:
        return tick_mask
    return isolate_tick_label_boxes_from_mask(tick_mask, labels, padding=1)


def search_horizontal_ticks_near_ocr(
    tick_mask: np.ndarray,
    labels: list[dict[str, object]],
    *,
    max_search_x: float,
    axis_x_hint: float | None = None,
) -> tuple[list[float], float | None]:
    h, w = tick_mask.shape[:2]
    tick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    raw_horizontal = cv2.morphologyEx(tick_mask, cv2.MORPH_OPEN, tick_kernel)
    raw_segments = extract_line_segments(raw_horizontal, "horizontal")

    def raw_candidates_for_label(label: dict[str, object]) -> list[dict[str, float]]:
        y = float(label["y"])
        label_left = float(label["x0"])
        label_right = float(label["x1"])
        label_width = max(1.0, label_right - label_left)
        right_band_left = max(label_left, label_right - max(14.0, label_width * 0.42))
        candidates: list[dict[str, float]] = []
        for segment in raw_segments:
            position = float(segment["position"])
            if abs(position - y) > max(5.0, h * 0.006):
                continue
            length = float(segment["length"])
            if not (2 <= length <= max(28.0, w * 0.045)):
                continue
            if float(segment["thickness"]) > 4:
                continue
            start = float(segment["start"])
            end = float(segment["end"])
            center = (start + end) * 0.5
            if end < right_band_left or start > min(max_search_x, label_right + max(18.0, w * 0.025)):
                continue
            if center < label_right - max(12.0, label_width * 0.35):
                continue
            if axis_x_hint is not None:
                axis_x = float(axis_x_hint)
                axis_tolerance = max(4.0, w * 0.01)
                touches_axis = (
                    start - 2 <= axis_x <= end + 2
                    or abs(start - axis_x) <= axis_tolerance
                    or abs(end - axis_x) <= axis_tolerance
                    or abs(center - axis_x) <= axis_tolerance
                )
                if not touches_axis and start > axis_x + axis_tolerance:
                    continue
            copy = dict(segment)
            copy["label_y"] = y
            copy["label_right"] = label_right
            copy["edge"] = end
            copy["edge_distance"] = min(abs(end - label_right), abs(start - label_right), abs(center - label_right))
            candidates.append(copy)
        return candidates

    candidates_by_label = [raw_candidates_for_label(label) for label in labels]
    coherent: list[dict[str, object]] = []
    for label_index, candidates in enumerate(candidates_by_label):
        for candidate in candidates:
            placed = False
            for group in coherent:
                group_edges = [float(item["edge"]) for item in group["items"]]  # type: ignore[index]
                group_lengths = [float(item["length"]) for item in group["items"]]  # type: ignore[index]
                if (
                    abs(float(candidate["edge"]) - float(np.median(group_edges))) <= max(3.0, w * 0.006)
                    and abs(float(candidate["length"]) - float(np.median(group_lengths))) <= 6.0
                ):
                    group["items"].append(candidate)  # type: ignore[index]
                    group["label_indexes"].add(label_index)  # type: ignore[index]
                    placed = True
                    break
            if not placed:
                coherent.append({"items": [candidate], "label_indexes": {label_index}})

    viable_groups: list[tuple[float, dict[str, object]]] = []
    required = max(4, int(len(labels) * 0.55))
    for group in coherent:
        label_indexes = group["label_indexes"]  # type: ignore[index]
        items = group["items"]  # type: ignore[index]
        if len(label_indexes) < required:
            continue
        edges = [float(item["edge"]) for item in items]
        positions = [float(item["position"]) for item in items]
        edge_std = float(np.std(np.array(edges, dtype=np.float64))) if len(edges) >= 2 else 0.0
        regularity = regularity_score(merge_close_values(positions, tolerance=3))
        if edge_std > max(3.0, w * 0.006):
            continue
        if len(positions) >= 4 and regularity > 0.35:
            continue
        median_edge = float(np.median(np.array(edges, dtype=np.float64)))
        median_distance = float(np.median(np.array([float(item["edge_distance"]) for item in items], dtype=np.float64)))
        score = len(label_indexes) * 1000.0 + median_edge - edge_std * 25.0 - median_distance * 2.0
        viable_groups.append((score, group))

    if viable_groups:
        _, group = max(viable_groups, key=lambda item: item[0])
        selected: list[dict[str, float]] = []
        items = group["items"]  # type: ignore[index]
        label_indexes = group["label_indexes"]  # type: ignore[index]
        median_edge = float(np.median(np.array([float(item["edge"]) for item in items], dtype=np.float64)))
        for label_index in sorted(label_indexes):
            candidates = [
                item
                for item in candidates_by_label[label_index]
                if abs(float(item["edge"]) - median_edge) <= max(3.0, w * 0.006)
            ]
            if not candidates:
                continue
            selected.append(
                min(
                    candidates,
                    key=lambda item: (
                        abs(float(item["edge"]) - median_edge),
                        abs(float(item["position"]) - float(item["label_y"])),
                        float(item["edge_distance"]),
                    ),
                )
            )
        values = merge_close_values([float(item["position"]) for item in selected], tolerance=3)
        if len(values) >= required:
            axis_x = float(np.median(np.array([float(item["edge"]) for item in selected], dtype=np.float64)))
            return values, axis_x

    values: list[float] = []
    axis_x_values: list[float] = []
    for label in labels:
        if axis_x_hint is None:
            continue
        y = float(label["y"])
        axis_x = float(axis_x_hint)
        axis_tolerance = max(4.0, w * 0.01)
        local = []
        for segment in raw_segments:
            if abs(float(segment["position"]) - y) > max(5.0, h * 0.006):
                continue
            if not (2 <= float(segment["length"]) <= max(28.0, w * 0.045)):
                continue
            if float(segment["thickness"]) > 4:
                continue
            start = float(segment["start"])
            end = float(segment["end"])
            center = (start + end) * 0.5
            touches_axis = (
                start - 2 <= axis_x <= end + 2
                or abs(start - axis_x) <= axis_tolerance
                or abs(end - axis_x) <= axis_tolerance
                or abs(center - axis_x) <= axis_tolerance
            )
            if not touches_axis:
                continue
            copy = dict(segment)
            copy["axis_distance"] = min(abs(start - axis_x), abs(end - axis_x), abs(center - axis_x))
            local.append(copy)
        if local:
            best = min(
                local,
                key=lambda segment: (
                    abs(float(segment["position"]) - y),
                    float(segment.get("axis_distance", 0.0)),
                    abs(float(segment["end"]) - axis_x),
                ),
            )
            values.append(float(best["position"]))
            axis_x_values.append(axis_x)
    if len(values) >= max(4, int(len(labels) * 0.55)):
        return merge_close_values(values, tolerance=3), float(np.median(axis_x_values)) if axis_x_values else None
    return [], None


def search_vertical_ticks_near_ocr(
    tick_mask: np.ndarray,
    labels: list[dict[str, object]],
    *,
    min_search_y: float,
    side: str = "bottom",
) -> tuple[list[float], float | None]:
    h, w = tick_mask.shape[:2]
    scan_mask = isolate_tick_label_boxes_from_mask(tick_mask, labels, padding=1)
    raw_scan_mask = tick_mask
    tick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    vertical = cv2.morphologyEx(scan_mask, cv2.MORPH_OPEN, tick_kernel)
    segments = extract_line_segments(vertical, "vertical")
    raw_vertical = cv2.morphologyEx(raw_scan_mask, cv2.MORPH_OPEN, tick_kernel)
    raw_segments = extract_line_segments(raw_vertical, "vertical")

    def bottom_tick_row_from_raw_segments() -> tuple[list[float], float | None]:
        if side != "bottom" or len(labels) < 4:
            return [], None
        label_tops = [float(label["y0"]) for label in labels]
        label_left = min(float(label["x0"]) for label in labels)
        label_right = max(float(label["x1"]) for label in labels)
        median_top = float(np.median(np.array(label_tops, dtype=np.float64)))
        min_top = min(label_tops)
        max_top = max(label_tops)
        x_padding = max(18.0, w * 0.025)
        candidates: list[dict[str, float]] = []
        for segment in raw_segments:
            length = float(segment["length"])
            thickness = float(segment["thickness"])
            position = float(segment["position"])
            start_y = float(segment["start"])
            end_y = float(segment["end"])
            if not (3 <= length <= max(14.0, h * 0.025)):
                continue
            if thickness > 4:
                continue
            if not (label_left - x_padding <= position <= label_right + x_padding):
                continue
            if end_y < median_top - max(12.0, h * 0.02):
                continue
            if end_y > median_top + max(5.0, h * 0.01):
                continue
            if start_y > max_top + 2:
                continue
            if end_y > max_top + 2:
                continue
            # A real x-axis tick row is a compact band just above the label text.
            # Text glyph strokes tend to end much lower inside the label boxes.
            if start_y > min_top + max(5.0, h * 0.012):
                continue
            candidates.append(
                {
                    "position": position,
                    "start": start_y,
                    "end": end_y,
                    "length": length,
                }
            )
        if not candidates:
            return [], None

        row_groups: list[list[dict[str, float]]] = []
        for candidate in sorted(candidates, key=lambda item: (item["end"], item["start"])):
            if not row_groups:
                row_groups.append([candidate])
                continue
            group_end = float(np.median([item["end"] for item in row_groups[-1]]))
            group_start = float(np.median([item["start"] for item in row_groups[-1]]))
            if abs(candidate["end"] - group_end) <= 2.5 and abs(candidate["start"] - group_start) <= 3.5:
                row_groups[-1].append(candidate)
            else:
                row_groups.append([candidate])

        target = len(labels)
        viable: list[tuple[float, list[dict[str, float]]]] = []
        for group in row_groups:
            positions = merge_close_values([item["position"] for item in group], tolerance=3)
            if len(positions) < max(4, int(target * 0.70)):
                continue
            if len(positions) > max(target + 2, int(target * 1.35)):
                continue
            span = max(positions) - min(positions) if len(positions) >= 2 else 0.0
            if span < w * 0.35:
                continue
            end_y = float(np.median([item["end"] for item in group]))
            count_penalty = abs(len(positions) - target) * 100.0
            row_distance = abs(end_y - median_top)
            score = count_penalty + row_distance - span / max(1.0, float(w))
            viable.append((score, group))
        if not viable:
            return [], None
        _, chosen = min(viable, key=lambda item: item[0])
        positions = merge_close_values([item["position"] for item in chosen], tolerance=3)
        tick_y = float(np.median([item["end"] for item in chosen]))
        return positions, tick_y

    def local_vertical_run(
        mask: np.ndarray,
        anchor: float,
        label_top: float,
        label_bottom: float,
    ) -> tuple[float, float] | None:
        radius = max(4, int(round(w * 0.012)))
        best: tuple[float, float, float] | None = None
        for xi in range(max(0, int(round(anchor)) - radius), min(w - 1, int(round(anchor)) + radius) + 1):
            column = mask[:, xi] > 0
            ys = np.where(column)[0]
            if len(ys) == 0:
                continue
            runs: list[tuple[int, int]] = []
            start = int(ys[0])
            previous = int(ys[0])
            for raw_y in ys[1:]:
                y = int(raw_y)
                if y == previous + 1:
                    previous = y
                    continue
                runs.append((start, previous))
                start = y
                previous = y
            runs.append((start, previous))
            for start_y, end_y in runs:
                length = end_y - start_y + 1
                min_run_length = 2 if side == "bottom" else 3
                if length < min_run_length:
                    continue
                if side == "top":
                    if start_y < label_bottom - 2 or start_y > label_bottom + max(48.0, h * 0.08):
                        continue
                    edge_y = float(end_y)
                    edge_distance = abs(float(start_y) - label_bottom)
                else:
                    if start_y >= label_top or end_y > label_top + 3:
                        continue
                    if end_y > label_top + 1:
                        continue
                    if end_y < label_top - max(36.0, h * 0.16):
                        continue
                    edge_y = float(end_y)
                    edge_distance = abs(float(end_y) - label_top)
                score = abs(float(xi) - anchor) + edge_distance * 0.12
                if best is None or score < best[0]:
                    best = (score, float(xi), edge_y)
        if best is None:
            return None
        return best[1], best[2]

    values: list[float] = []
    tick_y_values: list[float] = []
    row_values, row_tick_y = bottom_tick_row_from_raw_segments()
    if len(row_values) >= max(4, int(len(labels) * 0.70)):
        return row_values, row_tick_y
    for label in labels:
        label_width = max(1.0, float(label["x1"]) - float(label["x0"]))
        center_x = float(label["x"])
        anchor_x = float(label.get("anchor_x", center_x))
        if side == "top":
            raw_anchors = [
                anchor_x,
                float(label["x0"]) + label_width * 0.12,
                float(label["x0"]) + label_width * 0.18,
                float(label["x0"]) + label_width * 0.28,
                center_x,
            ]
            anchors = [max(0.0, min(w - 1.0, value)) for value in raw_anchors]
        else:
            raw_anchors = [anchor_x]
            anchors = [max(0.0, min(w - 1.0, value)) for value in raw_anchors]
        label_top = float(label["y0"])
        label_bottom = float(label["y1"])
        local = []
        for segment in segments:
            position = float(segment["position"])
            anchor_distance = min(abs(position - anchor) for anchor in anchors)
            if anchor_distance > max(8.0, w * 0.018):
                continue
            if not (3 <= float(segment["length"]) <= max(24.0, h * 0.045)):
                continue
            if side == "top":
                if float(segment["start"]) < label_bottom - 2:
                    continue
                if float(segment["start"]) > label_bottom + max(48.0, h * 0.08):
                    continue
                if float(segment["end"]) > h * 0.26:
                    continue
            else:
                if float(segment["end"]) < min_search_y:
                    continue
                if float(segment["start"]) >= label_top:
                    continue
                if float(segment["end"]) > label_top + 1:
                    continue
                if float(segment["end"]) < label_top - max(36.0, h * 0.035):
                    continue
            copy = dict(segment)
            copy["anchor_distance"] = anchor_distance
            local.append(copy)
        if local:
            if side == "top":
                best = min(
                    local,
                    key=lambda segment: (
                        float(segment.get("anchor_distance", 0.0)),
                        abs(float(segment["start"]) - label_bottom),
                    ),
                )
                tick_y_values.append(float(best["end"]))
            else:
                best = min(
                    local,
                    key=lambda segment: (
                        float(segment.get("anchor_distance", 0.0)),
                        abs(float(segment["end"]) - label_top),
                    ),
                )
                tick_y_values.append(float(best["end"]))
            values.append(float(best["position"]))
            continue
        run = min(
            (local_vertical_run(scan_mask, anchor, label_top, label_bottom) for anchor in anchors),
            key=lambda value: abs(float(value[0]) - anchor_x) if value is not None else float("inf"),
            default=None,
        )
        if run is None:
            run = min(
                (local_vertical_run(raw_scan_mask, anchor, label_top, label_bottom) for anchor in anchors),
                key=lambda value: abs(float(value[0]) - anchor_x) if value is not None else float("inf"),
                default=None,
            )
        if run is not None:
            x_value, y_value = run
            values.append(float(x_value))
            tick_y_values.append(float(y_value))
    if len(values) >= max(4, int(len(labels) * 0.55)):
        return merge_close_values(values, tolerance=3), float(np.median(tick_y_values)) if tick_y_values else None
    return [], None

def infer_bottom_regular_ticks_from_content_span(
    image: np.ndarray,
    image_shape: tuple[int, int],
    labels: list[dict[str, object]],
    side: str,
) -> list[float]:
    if side != "bottom" or len(labels) < 4:
        return []
    h, w = image_shape
    numeric_values: list[float] = []
    for label in labels:
        try:
            raw_numeric = label.get("numeric")
            if raw_numeric is None:
                raw_numeric = parse_numeric_label(str(label.get("text", "") or ""))
            numeric_values.append(float(raw_numeric))
        except (KeyError, TypeError, ValueError):
            return []
    diffs = np.diff(np.array(numeric_values, dtype=np.float64))
    nonzero = [abs(float(diff)) for diff in diffs if abs(float(diff)) > 1e-9]
    if not nonzero:
        return []
    step_value = float(np.median(np.array(nonzero, dtype=np.float64)))
    if any(abs(float(diff)) > 1e-9 and abs(abs(float(diff)) - step_value) > max(1e-6, step_value * 0.12) for diff in diffs):
        return []

    label_tops = [float(label["y0"]) for label in labels]
    label_widths = [float(label["x1"]) - float(label["x0"]) for label in labels]
    label_left = min(float(label["x0"]) for label in labels)
    label_right = max(float(label["x1"]) for label in labels)
    median_top = float(np.median(np.array(label_tops, dtype=np.float64)))
    y0 = max(0, int(round(median_top - max(80.0, h * 0.66))))
    y1 = min(h - 1, int(round(median_top - 2)))
    if y1 <= y0:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    x_margin = max(20.0, w * 0.04)
    search_left = max(0, int(np.floor(label_left - x_margin)))
    search_right = min(w - 1, int(np.ceil(label_right + x_margin)))
    colored = ((sat[y0 : y1 + 1, search_left : search_right + 1] > 45) & (gray[y0 : y1 + 1, search_left : search_right + 1] < 250))
    column_counts = colored.sum(axis=0)
    threshold = max(8, int((y1 - y0 + 1) * 0.08))
    active = np.where(column_counts >= threshold)[0]
    if len(active) < 2:
        return []

    runs: list[tuple[int, int]] = []
    start = int(active[0])
    previous = int(active[0])
    for raw in active[1:]:
        value = int(raw)
        if value <= previous + max(2, int(round(w * 0.006))):
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))

    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if not merged:
            merged.append((start, end))
            continue
        gap = start - merged[-1][1] - 1
        if gap <= max(8, int(round(w * 0.018))):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    label_mid = (label_left + label_right) * 0.5 - search_left
    viable: list[tuple[float, int, int]] = []
    for start, end in merged:
        span = end - start + 1
        if span < max(w * 0.28, (label_right - label_left) * 0.55):
            continue
        center = (start + end) * 0.5
        overlap = max(0.0, min(float(end), label_right - search_left) - max(float(start), label_left - search_left))
        score = span + overlap * 0.4 - abs(center - label_mid) * 0.05
        viable.append((score, start, end))
    if not viable:
        return []
    _, start, end = max(viable, key=lambda item: item[0])
    x0 = float(search_left + start)
    x1 = float(search_left + end)
    if x1 - x0 < w * 0.35:
        return []

    count = len(labels)
    generated = [float(value) for value in np.linspace(x0, x1, count)]
    tick_step = (x1 - x0) / max(1, count - 1)
    median_label_width = float(np.median(np.array(label_widths, dtype=np.float64))) if label_widths else 0.0
    if median_label_width < tick_step * 0.62:
        return []
    first = labels[0]
    last = labels[-1]
    if not (float(first["x0"]) - tick_step * 0.35 <= x0 <= float(first["x1"]) + tick_step * 0.35):
        return []
    if not (float(last["x0"]) - tick_step * 0.35 <= x1 <= float(last["x1"]) + tick_step * 0.35):
        return []
    return generated


def infer_ocr_tick_supplement(
    tick_mask: np.ndarray,
    image: np.ndarray,
    image_shape: tuple[int, int],
    ocr_items: list[dict[str, object]] | None,
    *,
    min_count: int = 4,
    y_axis_x_hint: float | None = None,
) -> dict[str, object]:
    h, w = image_shape
    x_labels, x_side = x_axis_numeric_ocr_row(ocr_items, image_shape, min_count=min_count)
    if not x_labels:
        x_labels, x_side = x_axis_category_ocr_row(ocr_items, image_shape, min_count=min_count)
    y_labels = left_numeric_ocr_column(ocr_items, image_shape, min_count=min_count)
    x_values: list[float] = []
    y_values: list[float] = []
    x_tick_y: float | None = None
    y_axis_x: float | None = None
    x_source = "none"
    y_source = "none"

    if x_labels:
        x_values, x_tick_y = search_vertical_ticks_near_ocr(
            tick_mask,
            x_labels,
            min_search_y=h * 0.45,
            side=x_side,
        )
        if len(x_values) >= min_count:
            x_source = f"physical_tick_near_ocr_{x_side}"
        else:
            x_values = []
            x_tick_y = None
        content_span_values = infer_bottom_regular_ticks_from_content_span(image, image_shape, x_labels, x_side)
        if content_span_values:
            if not x_values:
                x_values = content_span_values
                x_source = "physical_tick_content_span_top_edge"
            else:
                current = sorted(float(value) for value in x_values)
                candidate = sorted(float(value) for value in content_span_values)
                if len(current) == len(candidate):
                    step = float(np.median(np.diff(np.array(candidate, dtype=np.float64)))) if len(candidate) >= 2 else 0.0
                    distances = np.abs(np.array(current, dtype=np.float64) - np.array(candidate, dtype=np.float64))
                    label_widths = [float(label["x1"]) - float(label["x0"]) for label in x_labels]
                    median_label_width = float(np.median(np.array(label_widths, dtype=np.float64))) if label_widths else 0.0
                    broad_label_boxes = step > 0 and median_label_width >= step * 0.62
                    if broad_label_boxes and float(np.max(distances)) >= max(4.0, step * 0.08):
                        x_values = content_span_values
                        x_source = "physical_tick_content_span_top_edge"

    if y_labels:
        y_values, y_axis_x = search_horizontal_ticks_near_ocr(
            tick_mask,
            y_labels,
            max_search_x=w * 0.35,
            axis_x_hint=y_axis_x_hint,
        )
        if len(y_values) >= min_count:
            y_source = "physical_tick_near_ocr"
        else:
            y_values = []
            y_axis_x = None

    return {
        "x_values": merge_close_values(x_values, tolerance=3),
        "y_values": merge_close_values(y_values, tolerance=3),
        "x_tick_y": x_tick_y,
        "y_axis_x": y_axis_x,
        "x_source": x_source,
        "y_source": y_source,
        "x_side": x_side,
        "x_label_count": len(x_labels),
        "y_label_count": len(y_labels),
    }


def build_horizontal_ocr_endpoint_guide(
    shape: tuple[int, int],
    base_horizontal: np.ndarray,
    ocr_items: list[dict[str, object]],
    *,
    thickness: int,
    max_extra_lines: int = 2,
    target_count: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    h, w = shape
    positions, bounds = grid_positions_and_bounds(base_horizontal, "horizontal")
    guide = np.zeros_like(base_horizontal)
    details: dict[str, object] = {
        "enabled": False,
        "reason": "insufficient_horizontal_grid",
        "candidate_count": 0,
        "added_positions": [],
    }
    if len(positions) < 4 or bounds is None:
        return guide, details
    if target_count is not None and len(positions) >= target_count:
        details["reason"] = "mllm_target_already_satisfied"
        details["target_count"] = target_count
        return guide, details

    gap = median_gap(positions)
    if gap is None or gap < max(6.0, h * 0.01):
        details["reason"] = "invalid_gap"
        return guide, details
    if regularity_score(positions) > 0.12:
        details["reason"] = "irregular_horizontal_grid"
        details["regularity"] = round(regularity_score(positions), 3)
        return guide, details

    x0, x1 = bounds
    left_limit = max(w * 0.08, x0 + max(8.0, w * 0.012))
    endpoint_window = max(gap * (max_extra_lines + 2.5), h * 0.10)
    lower = min(positions) - endpoint_window
    upper = max(positions) + endpoint_window
    candidates: list[dict[str, object]] = []
    for item in ocr_items:
        center = ocr_center(item)
        numeric = ocr_numeric_value(item)
        if center is None or numeric is None:
            continue
        cx, cy = center
        if cx > left_limit or not (lower <= cy <= upper):
            continue
        candidates.append({"text": str(item.get("text", "")), "position": cy, "numeric": numeric, "x": cx})
    candidates = sorted(candidates, key=lambda item: float(item["position"]))
    details["candidate_count"] = len(candidates)
    if len(candidates) < 4:
        details["reason"] = "insufficient_left_numeric_ocr"
        return guide, details

    existing_tolerance = max(4.0, min(18.0, gap * 0.28))
    snap_tolerance = max(6.0, min(24.0, gap * 0.35))
    ordered_positions = sorted(float(pos) for pos in positions)
    additions: list[float] = []
    for candidate in candidates:
        cy = float(candidate["position"])
        if min(abs(cy - pos) for pos in ordered_positions) <= existing_tolerance:
            continue
        if cy > ordered_positions[-1]:
            steps = max(1, round((cy - ordered_positions[-1]) / gap))
            snapped = ordered_positions[-1] + steps * gap
        elif cy < ordered_positions[0]:
            steps = max(1, round((ordered_positions[0] - cy) / gap))
            snapped = ordered_positions[0] - steps * gap
        else:
            nearest = min(ordered_positions, key=lambda pos: abs(pos - cy))
            snapped = nearest
        if not (0 <= snapped <= h - 1):
            continue
        if abs(snapped - cy) > snap_tolerance:
            continue
        if min(abs(snapped - pos) for pos in ordered_positions) <= existing_tolerance:
            continue
        if any(abs(snapped - pos) <= existing_tolerance for pos in additions):
            continue
        additions.append(float(snapped))

    if not additions:
        details["reason"] = "no_missing_endpoint"
        return guide, details
    additions = sorted(additions)[:max_extra_lines]
    for y in additions:
        cv2.line(guide, (x0, int(round(y))), (x1, int(round(y))), 255, thickness, cv2.LINE_AA)
    details.update(
        {
            "enabled": True,
            "reason": "ocr_endpoint_extrapolation",
            "gap": round(gap, 3),
            "bounds": [int(x0), int(x1)],
            "added_positions": [round(value, 3) for value in additions],
        }
    )
    return guide, details

def build_horizontal_ocr_label_guide(
    shape: tuple[int, int],
    base_horizontal: np.ndarray,
    base_vertical: np.ndarray,
    ocr_items: list[dict[str, object]],
    *,
    min_lines: int,
    thickness: int,
    mllm_result: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    h, w = shape
    guide = np.zeros_like(base_horizontal)
    positions, h_bounds = grid_positions_and_bounds(base_horizontal, "horizontal")
    v_positions, _ = grid_positions_and_bounds(base_vertical, "vertical")
    mllm_ticks = mllm_axis_tick_texts(mllm_result, "y_axis")
    mllm_target = len(mllm_ticks) if len(mllm_ticks) >= 4 else None
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_needed",
        "candidate_count": 0,
        "added_positions": [],
    }
    if mllm_target is None:
        details["reason"] = "no_mllm_y_tick_sequence"
        return guide, details
    if len(positions) >= mllm_target and len(positions) <= mllm_target + 2 and regularity_score(positions) <= 0.25:
        details["reason"] = "mllm_target_already_satisfied"
        details["target_count"] = mllm_target
        return guide, details
    labels = left_numeric_ocr_column(ocr_items, shape, min_count=max(4, min_lines))
    if not labels:
        candidates: list[dict[str, object]] = []
        for item in ocr_items:
            text = str(item.get("text", "") or "").strip()
            center = ocr_center(item)
            edges = ocr_box_edges(item)
            if not text or center is None or edges is None:
                continue
            cx, cy = center
            if cx > w * 0.36:
                continue
            if not any(label_matches_mllm_tick(text, tick) for tick in mllm_ticks):
                continue
            candidates.append(
                {
                    "text": text,
                    "numeric": parse_numeric_label(text),
                    "x": cx,
                    "y": cy,
                    "x0": edges[0],
                    "y0": edges[1],
                    "x1": edges[2],
                    "y1": edges[3],
                }
            )
        if len(candidates) >= max(4, min_lines):
            tolerance = max(14.0, w * 0.035)
            columns: list[list[dict[str, object]]] = []
            for item in sorted(candidates, key=lambda value: float(value["x"])):
                if not columns:
                    columns.append([item])
                    continue
                median_x = float(np.median([float(value["x"]) for value in columns[-1]]))
                if abs(float(item["x"]) - median_x) <= tolerance:
                    columns[-1].append(item)
                else:
                    columns.append([item])
            labels = sorted(
                max(columns, key=lambda column: (len(column), max(float(value["y"]) for value in column) - min(float(value["y"]) for value in column))),
                key=lambda value: float(value["y"]),
            )
    details["candidate_count"] = len(labels)
    if len(labels) < max(4, min_lines):
        details["reason"] = "insufficient_y_axis_ocr_labels"
        return guide, details

    matched: list[tuple[int, float]] = []
    used: set[int] = set()
    for item in labels:
        text = str(item.get("text", "") or "")
        for index, tick in enumerate(mllm_ticks):
            if index in used:
                continue
            if label_matches_mllm_tick(text, tick):
                used.add(index)
                matched.append((index, float(item["y"])))
                break
    if len(matched) < max(3, int(np.ceil(mllm_target * 0.45))):
        details["reason"] = "y_ocr_labels_do_not_match_mllm_sequence"
        details["matched_count"] = len(matched)
        return guide, details
    indices = np.array([item[0] for item in matched], dtype=np.float64)
    ys = np.array([item[1] for item in matched], dtype=np.float64)
    if len(set(int(value) for value in indices)) < 2:
        details["reason"] = "insufficient_distinct_y_matches"
        return guide, details
    slope, intercept = np.polyfit(indices, ys, 1)
    inferred = [float(intercept + slope * index) for index in range(mllm_target)]
    if regularity_score(inferred) > 0.12:
        details["reason"] = "inferred_y_positions_irregular"
        return guide, details
    if min(inferred) < -2 or max(inferred) > h + 2:
        details["reason"] = "inferred_y_positions_out_of_bounds"
        return guide, details
    snap_details: list[dict[str, float]] = []
    if len(positions) >= max(2, mllm_target - 2):
        if len(positions) >= 2:
            gaps = np.diff(np.array(sorted(positions), dtype=np.float64))
            positive_gaps = [float(value) for value in gaps if float(value) > 1.0]
            reference_gap = float(np.median(np.array(positive_gaps, dtype=np.float64))) if positive_gaps else 0.0
        else:
            reference_gap = 0.0
        snap_tolerance = max(4.0, min(10.0, reference_gap * 0.35 if reference_gap > 0 else h * 0.02))
        snapped, snap_details = snap_positions_to_reference(
            inferred,
            positions,
            tolerance=snap_tolerance,
        )
        if len(snapped) == len(inferred) and regularity_score(snapped) <= 0.18:
            inferred = snapped

    if h_bounds is not None:
        x0, x1 = h_bounds
    elif len(v_positions) >= 2:
        x0, x1 = int(round(min(v_positions))), int(round(max(v_positions)))
    else:
        label_right = max(float(item.get("x1", item["x"])) for item in labels)
        x0 = int(round(label_right + max(8.0, w * 0.015)))
        x1 = w - 1
    if x1 <= x0:
        details["reason"] = "invalid_horizontal_bounds"
        return guide, details
    for y in inferred:
        y_i = int(round(max(0, min(h - 1, y))))
        cv2.line(guide, (int(x0), y_i), (int(x1), y_i), 255, thickness, cv2.LINE_AA)
    details.update(
        {
            "enabled": True,
            "reason": "left_mllm_label_sequence",
            "bounds": [int(x0), int(x1)],
            "added_positions": [round(value, 3) for value in inferred],
            "mllm_target": mllm_target,
            "matched_count": len(matched),
        }
    )
    if snap_details:
        details["snapped_to_direct"] = snap_details
    return guide, details

def choose_ocr_label_row(
    candidates: list[dict[str, object]],
    image_height: int,
    *,
    preference: str,
) -> list[dict[str, object]]:
    if not candidates:
        return []
    tolerance = max(14.0, image_height * 0.025)
    rows: list[list[dict[str, object]]] = []
    for item in sorted(candidates, key=lambda value: float(value["y"])):
        if not rows:
            rows.append([item])
            continue
        median_y = float(np.median([float(value["y"]) for value in rows[-1]]))
        if abs(float(item["y"]) - median_y) <= tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])

    def row_score(row: list[dict[str, object]]) -> float:
        xs = [float(item["x"]) for item in row]
        span = max(xs) - min(xs) if len(xs) >= 2 else 0.0
        regular_bonus = 0.0
        if len(xs) >= 3:
            reg = regularity_score(xs)
            regular_bonus = max(0.0, 1.0 - min(1.0, reg))
        y_bias = float(np.median([item["y"] for item in row])) * 0.01
        if preference == "top":
            y_bias = -y_bias
        return len(row) * 1000.0 + span + regular_bonus * 100.0 + y_bias

    return sorted(max(rows, key=row_score), key=lambda value: float(value["x"]))

def choose_bottom_ocr_label_row(
    candidates: list[dict[str, object]],
    image_height: int,
) -> list[dict[str, object]]:
    return choose_ocr_label_row(candidates, image_height, preference="bottom")

def choose_regular_x_axis_ocr_row(
    candidates: list[dict[str, object]],
    image_height: int,
    *,
    preference: str,
) -> list[dict[str, object]]:
    row = choose_ocr_label_row(candidates, image_height, preference=preference)
    if len(row) < 4:
        return []
    regular = choose_regular_numeric_window(row, min_count=4, order_key="x")
    return regular if len(regular) >= 4 else row

def mllm_axis_tick_texts(mllm_result: dict[str, object] | None, axis_key: str) -> list[str]:
    if not isinstance(mllm_result, dict) or mllm_result.get("error") is not None:
        return []
    axis = mllm_result.get(axis_key, {})
    if not isinstance(axis, dict):
        return []
    ticks = axis.get("tick_labels", [])
    if not isinstance(ticks, list):
        return []
    labels: list[str] = []
    for tick in ticks:
        text = str(tick.get("text", "") if isinstance(tick, dict) else tick).strip()
        if text and text.casefold() not in {"none", "unknown", "null", "n/a"}:
            labels.append(text)
    return labels

def normalized_axis_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())

def numeric_parse_text(text: str) -> str:
    value = str(text or "")
    value = value.replace("'", "").replace("’", "").replace("`", "")
    value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)
    return value

def label_matches_mllm_tick(label: str, tick: str) -> bool:
    left = normalized_axis_text(label)
    right = normalized_axis_text(tick)
    if not left or not right:
        return False
    if left == right:
        return True
    left_num = parse_numeric_label(numeric_parse_text(left))
    right_num = parse_numeric_label(numeric_parse_text(right))
    if left_num is None or right_num is None:
        return False
    scale = abs(float(right_num))
    tolerance = max(1e-6, scale * 0.02)
    if scale < 100.0:
        tolerance = min(tolerance, 0.01)
    return abs(float(left_num) - float(right_num)) <= tolerance

def fit_positions_from_mllm_matches(
    row: list[dict[str, object]],
    mllm_ticks: list[str],
) -> list[float]:
    if len(row) < 2 or len(mllm_ticks) < 4:
        return []
    matched: list[tuple[int, float]] = []
    used: set[int] = set()
    for item in row:
        text = str(item.get("text", "") or "")
        for index, tick in enumerate(mllm_ticks):
            if index in used:
                continue
            if label_matches_mllm_tick(text, tick):
                used.add(index)
                matched.append((index, float(item["x"])))
                break
    if len(matched) < 2:
        return []
    indices = np.array([item[0] for item in matched], dtype=np.float64)
    xs = np.array([item[1] for item in matched], dtype=np.float64)
    if len(set(int(value) for value in indices)) < 2:
        return []
    slope, intercept = np.polyfit(indices, xs, 1)
    if abs(float(slope)) <= 1.0:
        return []
    fitted = [float(intercept + slope * index) for index in range(len(mllm_ticks))]
    if regularity_score(fitted) > 0.10:
        return []
    return fitted

def merged_x_label_box_positions(
    ocr_items: list[dict[str, object]],
    image_shape: tuple[int, int],
    mllm_ticks: list[str],
    *,
    plot_x0: float,
    plot_x1: float,
    top_limit: float,
    bottom_limit: float,
) -> list[float]:
    if len(mllm_ticks) < 4:
        return []
    h, _ = image_shape
    best: tuple[float, tuple[float, float]] | None = None
    for item in ocr_items:
        text = str(item.get("text", "") or "").strip()
        center = ocr_center(item)
        edges = ocr_box_edges(item)
        if not text or center is None or edges is None:
            continue
        cx, cy = center
        if not (cy <= top_limit or cy >= bottom_limit):
            continue
        x0, _, x1, _ = edges
        width = x1 - x0
        plot_width = max(1.0, plot_x1 - plot_x0)
        if width < plot_width * 0.45:
            continue
        match_count = sum(1 for tick in mllm_ticks if normalized_axis_text(tick) in normalized_axis_text(text))
        numeric_matches = 0
        text_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        tick_numbers = [parse_numeric_label(tick) for tick in mllm_ticks]
        for number in text_numbers:
            parsed = parse_numeric_label(number)
            if parsed is not None and any(value is not None and abs(float(parsed) - float(value)) <= max(1e-6, abs(float(value)) * 0.02) for value in tick_numbers):
                numeric_matches += 1
        evidence = max(match_count, numeric_matches)
        if evidence < max(3, int(np.ceil(len(mllm_ticks) * 0.35))):
            continue
        score = evidence * 1000.0 + min(width, plot_width) - abs(cx - (plot_x0 + plot_x1) / 2) * 0.1
        if best is None or score > best[0]:
            best = (score, (max(plot_x0, min(x0, plot_x0)), min(plot_x1, max(x1, plot_x1))))
    if best is None:
        return []
    return [float(value) for value in np.linspace(float(plot_x0), float(plot_x1), len(mllm_ticks))]

def build_vertical_ocr_label_guide(
    shape: tuple[int, int],
    base_horizontal: np.ndarray,
    base_vertical: np.ndarray,
    ocr_items: list[dict[str, object]],
    *,
    min_lines: int,
    thickness: int,
    mllm_result: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    h, w = shape
    guide = np.zeros_like(base_vertical)
    h_positions, h_bounds = grid_positions_and_bounds(base_horizontal, "horizontal")
    v_positions, _ = grid_positions_and_bounds(base_vertical, "vertical")
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_needed",
        "candidate_count": 0,
        "added_positions": [],
    }
    mllm_ticks = mllm_axis_tick_texts(mllm_result, "x_axis")
    mllm_target = len(mllm_ticks) if len(mllm_ticks) >= 4 else None
    if len(v_positions) >= min_lines and (mllm_target is None or len(v_positions) >= mllm_target):
        return guide, details
    if len(h_positions) < max(4, min_lines) or h_bounds is None:
        details["reason"] = "missing_horizontal_bounds"
        return guide, details

    x0, x1 = h_bounds
    plot_width = max(1, x1 - x0)
    min_x = x0 + max(8.0, plot_width * 0.015)
    max_x = min(w - 1.0, x1 + max(8.0, plot_width * 0.03))
    top_limit = min(h * 0.34, min(h_positions) + max(80.0, h * 0.14))
    bottom_limit = max(h * 0.66, max(h_positions) - max(80.0, h * 0.14))
    candidates_by_side: dict[str, list[dict[str, object]]] = {"top": [], "bottom": []}
    for item in ocr_items:
        center = ocr_center(item)
        text = str(item.get("text", "") or "").strip()
        role = str(item.get("role", "other"))
        if center is None or not text:
            continue
        cx, cy = center
        numeric = parse_numeric_label(text)
        top_numeric_other = role == "other" and numeric is not None and cy <= top_limit
        bottom_numeric_other = role == "other" and numeric is not None and cy >= bottom_limit
        if role != "x_axis" and not top_numeric_other and not bottom_numeric_other:
            continue
        if not (min_x <= cx <= max_x):
            continue
        if cy <= top_limit:
            anchor = ocr_axis_anchor_x(item, "top")
            candidates_by_side["top"].append({"text": text, "x": anchor if anchor is not None else cx, "y": cy, "numeric": numeric})
        if cy >= bottom_limit:
            anchor = ocr_axis_anchor_x(item, "bottom")
            candidates_by_side["bottom"].append({"text": text, "x": anchor if anchor is not None else cx, "y": cy, "numeric": numeric})

    row_candidates = [
        ("bottom", choose_regular_x_axis_ocr_row(candidates_by_side["bottom"], h, preference="bottom")),
        ("top", choose_regular_x_axis_ocr_row(candidates_by_side["top"], h, preference="top")),
    ]
    row_side, row = max(
        row_candidates,
        key=lambda item: (
            len(item[1]),
            (max([float(value["x"]) for value in item[1]]) - min([float(value["x"]) for value in item[1]]))
            if len(item[1]) >= 2
            else 0.0,
        ),
    )
    details["candidate_count"] = len(row)
    inferred_from_mllm = fit_positions_from_mllm_matches(row, mllm_ticks)
    if not inferred_from_mllm:
        inferred_from_mllm = merged_x_label_box_positions(
            ocr_items,
            shape,
            mllm_ticks,
            plot_x0=float(x0),
            plot_x1=float(x1),
            top_limit=top_limit,
            bottom_limit=bottom_limit,
        )
        if inferred_from_mllm:
            row_side = "merged"
    if len(row) < max(4, min_lines) and not inferred_from_mllm:
        details["reason"] = "insufficient_x_axis_ocr_labels"
        return guide, details
    xs = inferred_from_mllm if inferred_from_mllm else [float(item["x"]) for item in row]
    if max(xs) - min(xs) < plot_width * 0.45:
        details["reason"] = f"{row_side}_ocr_span_too_small"
        return guide, details

    numeric_values = [item["numeric"] for item in row if item.get("numeric") is not None]
    mllm_axis = mllm_result.get("x_axis", {}) if isinstance(mllm_result, dict) else {}
    mllm_axis_type = str(mllm_axis.get("type", "unknown")) if isinstance(mllm_axis, dict) else "unknown"
    allow_text_axis = bool(inferred_from_mllm) or mllm_axis_type in {"category", "time", "mixed"}
    if len(numeric_values) < max(3, int(np.ceil(len(row) * 0.8))) and not allow_text_axis:
        details["reason"] = f"{row_side}_ocr_not_continuous_numeric"
        return guide, details
    numeric_by_x = [float(item["numeric"]) for item in row if item.get("numeric") is not None]
    diffs = np.diff(np.array(numeric_by_x, dtype=np.float64))
    if len(diffs) and not (np.all(diffs >= 0) or np.all(diffs <= 0)) and not allow_text_axis:
        details["reason"] = f"{row_side}_ocr_numeric_not_monotonic"
        return guide, details

    y0 = int(round(min(h_positions)))
    y1 = int(round(max(h_positions)))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if y1 <= y0:
        details["reason"] = "invalid_vertical_span"
        return guide, details
    for x in xs:
        x_i = int(round(max(0, min(w - 1, x))))
        cv2.line(guide, (x_i, y0), (x_i, y1), 255, thickness, cv2.LINE_AA)
    details.update(
        {
            "enabled": True,
            "reason": f"{row_side}_{'mllm_label_sequence' if inferred_from_mllm else 'ocr_label_centers'}",
            "bounds": [y0, y1],
            "added_positions": [round(value, 3) for value in xs],
            "row_y": round(float(np.median([item["y"] for item in row])), 3) if row else None,
            "side": row_side,
            "mllm_target": mllm_target,
        }
    )
    return guide, details

def build_plot_border_guide(
    shape: tuple[int, int],
    base_horizontal: np.ndarray,
    base_vertical: np.ndarray,
    ocr_items: list[dict[str, object]],
    *,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    h, w = shape
    h_positions, h_bounds = grid_positions_and_bounds(base_horizontal, "horizontal")
    v_positions, v_bounds = grid_positions_and_bounds(base_vertical, "vertical")
    h_guide = np.zeros_like(base_horizontal)
    v_guide = np.zeros_like(base_vertical)
    details: dict[str, object] = {
        "enabled": False,
        "reason": "insufficient_base_grid",
        "horizontal_added_positions": [],
        "vertical_added_positions": [],
    }
    if len(h_positions) < 3 or h_bounds is None:
        return h_guide, v_guide, details

    x0, x1 = h_bounds
    y0 = int(round(min(h_positions)))
    y1 = int(round(max(h_positions)))
    if y1 <= y0 or x1 <= x0:
        details["reason"] = "invalid_bounds"
        return h_guide, v_guide, details

    y_axis_items = []
    x_axis_items = []
    for item in ocr_items:
        center = ocr_center(item)
        if center is None:
            continue
        role = str(item.get("role", "other"))
        text = str(item.get("text", "") or "")
        numeric = parse_numeric_label(text)
        if role == "y_axis" and numeric is not None:
            y_axis_items.append((float(center[0]), float(center[1]), numeric))
        elif role == "x_axis":
            x_axis_items.append((float(center[0]), float(center[1]), numeric))

    y_axis_left_support = sum(
        1
        for cx, cy, _ in y_axis_items
        if cx <= x0 + max(36.0, (x1 - x0) * 0.08) and y0 - 8 <= cy <= y1 + 8
    )
    x_axis_support = sum(
        1
        for cx, cy, _ in x_axis_items
        if x0 - 8 <= cx <= x1 + 8 and (cy <= y0 + max(80.0, h * 0.14) or cy >= y1 - max(80.0, h * 0.14))
    )
    x_axis_numeric_support = sum(
        1
        for cx, cy, numeric in x_axis_items
        if numeric is not None
        and x0 - 8 <= cx <= x1 + 8
        and (cy <= y0 + max(80.0, h * 0.14) or cy >= y1 - max(80.0, h * 0.14))
    )

    existing_v = [float(pos) for pos in v_positions]
    existing_h = [float(pos) for pos in h_positions]
    tolerance = max(6.0, min(14.0, (x1 - x0) * 0.015))
    vertical_added: list[float] = []
    horizontal_added: list[float] = []

    covered_x_ticks = x_axis_numeric_support >= 3 and len(existing_v) >= x_axis_numeric_support
    if (
        y_axis_left_support >= 3
        and not covered_x_ticks
        and not any(abs(float(x0) - pos) <= tolerance for pos in existing_v)
    ):
        cv2.line(v_guide, (int(x0), y0), (int(x0), y1), 255, thickness, cv2.LINE_AA)
        vertical_added.append(float(x0))
    if x_axis_support >= 3:
        if not any(abs(float(y1) - pos) <= tolerance for pos in existing_h):
            cv2.line(h_guide, (int(x0), y1), (int(x1), y1), 255, thickness, cv2.LINE_AA)
            horizontal_added.append(float(y1))
        if not any(abs(float(y0) - pos) <= tolerance for pos in existing_h):
            cv2.line(h_guide, (int(x0), y0), (int(x1), y0), 255, thickness, cv2.LINE_AA)
            horizontal_added.append(float(y0))

    if vertical_added or horizontal_added:
        details.update(
            {
                "enabled": True,
                "reason": "plot_axis_border_inclusion",
                "bounds": [int(x0), int(y0), int(x1), int(y1)],
                "y_axis_left_support": y_axis_left_support,
                "x_axis_support": x_axis_support,
                "x_axis_numeric_support": x_axis_numeric_support,
                "x_ticks_already_covered": covered_x_ticks,
                "horizontal_added_positions": [round(value, 3) for value in horizontal_added],
                "vertical_added_positions": [round(value, 3) for value in vertical_added],
            }
        )
    else:
        details.update(
            {
                "reason": "no_missing_border",
                "bounds": [int(x0), int(y0), int(x1), int(y1)],
                "y_axis_left_support": y_axis_left_support,
                "x_axis_support": x_axis_support,
                "x_axis_numeric_support": x_axis_numeric_support,
                "x_ticks_already_covered": covered_x_ticks,
            }
        )
    return h_guide, v_guide, details

def build_semantic_guide_grid(
    image_shape: tuple[int, int],
    grid_horizontal: np.ndarray,
    grid_vertical: np.ndarray,
    ocr_items: list[dict[str, object]],
    *,
    min_lines: int,
    thickness: int,
    mllm_result: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    h_label_guide, h_label_details = build_horizontal_ocr_label_guide(
        image_shape,
        grid_horizontal,
        grid_vertical,
        ocr_items,
        min_lines=min_lines,
        thickness=thickness,
        mllm_result=mllm_result,
    )
    h_guide, h_details = build_horizontal_ocr_endpoint_guide(
        image_shape,
        cv2.bitwise_or(grid_horizontal, h_label_guide),
        ocr_items,
        thickness=thickness,
        target_count=mllm_tick_target(mllm_result, "y_axis"),
    )
    horizontal_with_guide = cv2.bitwise_or(grid_horizontal, cv2.bitwise_or(h_label_guide, h_guide))
    v_guide, v_details = build_vertical_ocr_label_guide(
        image_shape,
        horizontal_with_guide,
        grid_vertical,
        ocr_items,
        min_lines=min_lines,
        thickness=thickness,
        mllm_result=mllm_result,
    )
    border_h_guide, border_v_guide, border_details = build_plot_border_guide(
        image_shape,
        cv2.bitwise_or(horizontal_with_guide, h_guide),
        cv2.bitwise_or(grid_vertical, v_guide),
        ocr_items,
        thickness=thickness,
    )
    if (
        v_details.get("enabled")
        and "mllm_label_sequence" in str(v_details.get("reason", ""))
        and int(v_details.get("mllm_target", 0) or 0) > 0
        and grid_line_count(v_guide, "vertical") >= int(v_details.get("mllm_target", 0) or 0)
    ):
        border_v_guide = np.zeros_like(border_v_guide)
        if border_details.get("vertical_added_positions"):
            border_details = dict(border_details)
            border_details["vertical_suppressed_by_mllm_sequence"] = border_details.get("vertical_added_positions", [])
            border_details["vertical_added_positions"] = []
    h_guide = cv2.bitwise_or(cv2.bitwise_or(h_label_guide, h_guide), border_h_guide)
    v_guide = cv2.bitwise_or(v_guide, border_v_guide)
    horizontal_details = h_label_details if h_label_details.get("enabled") else h_details
    if h_label_details.get("enabled"):
        horizontal_details = dict(horizontal_details)
        horizontal_details["endpoint_guide"] = h_details
    guide = cv2.bitwise_or(h_guide, v_guide)
    metadata = {
        "horizontal": horizontal_details,
        "vertical": v_details,
        "plot_border": border_details,
        "horizontal_count": grid_line_count(h_guide, "horizontal"),
        "vertical_count": grid_line_count(v_guide, "vertical"),
    }
    return guide, h_guide, v_guide, metadata

def grid_positions_and_bounds(
    mask: np.ndarray,
    orientation: str,
    *,
    tolerance: int = 2,
) -> tuple[list[float], tuple[int, int] | None]:
    clusters = cluster_line_segments(extract_line_segments(mask, orientation), tolerance)
    if not clusters:
        return [], None
    positions = sorted(float(item["position"]) for item in clusters)
    starts = np.array([item["start"] for item in clusters], dtype=np.float32)
    ends = np.array([item["end"] for item in clusters], dtype=np.float32)
    return positions, (int(round(float(np.median(starts)))), int(round(float(np.median(ends)))))

def merge_close_values(values: list[float], tolerance: int) -> list[float]:
    if not values:
        return []
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or abs(value - float(np.median(groups[-1]))) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.median(group)) for group in groups]

def find_axis_lines(dark_mask: np.ndarray) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    h, w = dark_mask.shape[:2]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * 0.18))))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.18)), 1))
    vertical_source = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(h * 0.035)))),
    )
    horizontal_source = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(w * 0.015)), 1)),
    )
    vertical = cv2.morphologyEx(vertical_source, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(horizontal_source, cv2.MORPH_OPEN, horizontal_kernel)

    y_axis: dict[str, float] | None = None
    count, _, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
    candidates: list[dict[str, float]] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if height < h * 0.35 or width > max(5, w * 0.01):
            continue
        candidates.append(
            {
                "x": float(x + (width - 1) / 2),
                "y0": float(y),
                "y1": float(y + height - 1),
                "length": float(height),
                "area": float(area),
            }
        )
    if candidates:
        y_axis = sorted(candidates, key=lambda item: (item["x"] > w * 0.45, item["x"], -item["length"]))[0]

    x_axis: dict[str, float] | None = None
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if width < w * 0.25 or height > max(5, h * 0.01):
            continue
        if y + (height - 1) / 2 < h * 0.32:
            continue
        candidates.append(
            {
                "y": float(y + (height - 1) / 2),
                "x0": float(x),
                "x1": float(x + width - 1),
                "length": float(width),
                "area": float(area),
            }
        )
    if candidates:
        x_axis = sorted(candidates, key=lambda item: (item["y"] < h * 0.35, -item["y"], -item["length"]))[0]

    return y_axis, x_axis

def find_y_tick_positions(dark_mask: np.ndarray, y_axis: dict[str, float] | None) -> list[float]:
    if y_axis is None:
        return []
    h, w = dark_mask.shape[:2]
    axis_x = y_axis["x"]
    tick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    horizontal = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, tick_kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)

    values: list[float] = []
    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        if not (3 <= width <= max(24, w * 0.035) and height <= 4):
            continue
        x0 = x
        x1 = x + width - 1
        y_center = y + (height - 1) / 2
        near_axis = x0 - 2 <= axis_x <= x1 + 2 or abs(x1 - axis_x) <= 3 or abs(x0 - axis_x) <= 3
        inside_axis_span = y_axis["y0"] - 6 <= y_center <= y_axis["y1"] + 24
        if near_axis and inside_axis_span:
            values.append(float(y_center))
    return merge_close_values(values, tolerance=3)

def find_x_tick_positions(
    dark_mask: np.ndarray,
    y_axis: dict[str, float] | None,
    x_axis: dict[str, float] | None,
) -> tuple[list[float], float | None]:
    h, w = dark_mask.shape[:2]
    axis_x = y_axis["x"] if y_axis is not None else 0.0
    if x_axis is None and (y_axis is None or y_axis["length"] < h * 0.55):
        return [], None
    tick_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
    vertical = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, tick_kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)

    candidates: list[tuple[float, float]] = []
    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        if not (width <= 4 and 3 <= height <= max(20, h * 0.05)):
            continue
        x_center = x + (width - 1) / 2
        y_center = y + (height - 1) / 2
        if x_center <= axis_x + 10:
            continue
        if x_axis is not None:
            if abs(y_center - x_axis["y"]) > max(10, h * 0.04):
                continue
        elif y_center < h * 0.58:
            continue
        candidates.append((float(x_center), float(y_center)))

    if not candidates:
        return [], None

    y_groups: list[list[tuple[float, float]]] = []
    for candidate in sorted(candidates, key=lambda item: item[1]):
        if not y_groups or abs(candidate[1] - float(np.median([item[1] for item in y_groups[-1]]))) > 4:
            y_groups.append([candidate])
        else:
            y_groups[-1].append(candidate)

    groups_with_ticks = [group for group in y_groups if len(merge_close_values([item[0] for item in group], 3)) >= 2]
    if not groups_with_ticks:
        return [], None
    if x_axis is not None:
        chosen = min(groups_with_ticks, key=lambda group: abs(float(np.median([item[1] for item in group])) - x_axis["y"]))
    else:
        chosen = sorted(groups_with_ticks, key=lambda group: (float(np.median([item[1] for item in group])), -len(group)))[0]

    x_positions = merge_close_values([item[0] for item in chosen], tolerance=3)
    tick_y = float(np.median([item[1] for item in chosen]))
    return x_positions, tick_y

def infer_plot_x1_from_content(
    image: np.ndarray,
    *,
    x0: int,
    y0: int,
    y1: int,
    x_ticks: list[float],
) -> int:
    h, w = image.shape[:2]
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if y1 < y0:
        y0, y1 = y1, y0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    roi = (sat[y0 : y1 + 1, :] > 35) & (gray[y0 : y1 + 1, :] < 248)
    columns = np.where(roi[:, max(0, x0 + 1) :].any(axis=0))[0]

    x1 = max(x_ticks) if x_ticks else float(x0)
    if len(x_ticks) >= 2:
        step = float(np.median(np.diff(np.array(sorted(x_ticks), dtype=np.float32))))
        x1 = max(x1, max(x_ticks) + step * 0.35)
    if len(columns) > 0:
        x1 = max(x1, float(columns.max() + max(0, x0 + 1)))
    return int(max(x0 + 1, min(w - 1, round(x1))))


def detect_right_plot_border_x(
    image: np.ndarray,
    *,
    x0: int,
    y0: int,
    y1: int,
    expected_x: float,
) -> float | None:
    h, w = image.shape[:2]
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h - 1, int(y1)))
    if y1 < y0:
        y0, y1 = y1, y0
    if y1 - y0 < max(20, h * 0.12):
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    left = max(int(x0) + 1, int(round(expected_x)) - max(12, int(w * 0.02)))
    right = min(w - 1, int(round(expected_x)) + max(12, int(w * 0.02)))
    if right <= left:
        return None
    roi = gray[y0 : y1 + 1, left : right + 1] < 90
    counts = roi.sum(axis=0)
    threshold = max(20, int((y1 - y0 + 1) * 0.45))
    indexes = np.where(counts >= threshold)[0]
    if indexes.size == 0:
        return None
    best = int(indexes[np.argmax(counts[indexes])])
    return float(left + best)


def add_plot_border_endpoint_ticks(
    image: np.ndarray,
    x_values: list[float],
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> list[float]:
    if len(x_values) < 4:
        return x_values
    ordered = sorted(float(value) for value in x_values)
    step = median_gap(ordered)
    if step is None or step <= 2.0:
        return ordered
    tolerance = max(6.0, step * 0.24)
    augmented = list(ordered)
    expected_last = ordered[-2] + step if len(ordered) >= 2 else ordered[-1] + step
    last_gap = ordered[-1] - ordered[-2] if len(ordered) >= 2 else step
    if abs(ordered[-1] - expected_last) <= max(2.5, step * 0.08) and last_gap >= step * 0.85:
        return ordered
    right_border = detect_right_plot_border_x(
        image,
        x0=x0,
        y0=y0,
        y1=y1,
        expected_x=expected_last,
    )
    if right_border is not None and abs(right_border - expected_last) <= tolerance:
        if abs(ordered[-1] - right_border) > 3.0 or last_gap < step * 0.85:
            augmented[-1] = float(right_border)
        else:
            augmented.append(float(right_border))
    elif abs(float(x1) - expected_last) <= tolerance:
        augmented.append(float(x1))
    return merge_close_values(augmented, tolerance=3)


def add_axis_aligned_endpoint_tick(values: list[float], endpoint: float | None) -> list[float]:
    if endpoint is None or len(values) < 3:
        return values
    ordered = sorted(float(value) for value in values)
    step = median_gap(ordered)
    if step is None or step <= 2.0:
        return ordered
    tolerance = max(5.0, step * 0.16)
    endpoint = float(endpoint)
    augmented = list(ordered)
    if abs(endpoint - (ordered[-1] + step)) <= tolerance:
        augmented.append(endpoint)
    elif abs(endpoint - (ordered[0] - step)) <= tolerance:
        augmented.append(endpoint)
    return merge_close_values(augmented, tolerance=3)


def infer_y_axis_from_horizontal_grid_span(
    tick_mask: np.ndarray,
    image_shape: tuple[int, int],
    x_ticks: list[float],
) -> dict[str, float] | None:
    if len(x_ticks) < 2:
        return None
    h, w = image_shape
    horizontal = cv2.morphologyEx(
        tick_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * 0.03)), 1)),
    )
    clusters = cluster_line_segments(extract_line_segments(horizontal, "horizontal"), tolerance=2)
    lines = keep_long_grid_candidates(clusters, w, min_extent_frac=0.20)
    if len(lines) < 2:
        return None

    min_x_tick = min(float(value) for value in x_ticks)
    max_x_tick = max(float(value) for value in x_ticks)
    tick_span = max_x_tick - min_x_tick
    left_candidates = [
        float(item["start"])
        for item in lines
        if float(item["start"]) <= min_x_tick + max(20.0, w * 0.08)
    ]
    if not left_candidates:
        return None
    left = float(np.median(np.array(left_candidates, dtype=np.float64)))
    start_tolerance = max(10.0, w * 0.03)
    min_span = max(w * 0.35, tick_span * 0.65)
    plot_lines = [
        item
        for item in lines
        if abs(float(item["start"]) - left) <= start_tolerance
        and float(item["span"]) >= min_span
        and float(item["position"]) <= h * 0.88
    ]
    if len(plot_lines) < 2:
        return None
    y_positions = [float(item["position"]) for item in plot_lines]
    y0 = min(y_positions)
    y1 = max(y_positions)
    if y1 - y0 < h * 0.22:
        return None
    return {
        "x": float(left),
        "y0": float(y0),
        "y1": float(y1),
        "length": float(y1 - y0),
        "area": 0.0,
        "source": "horizontal_grid_span",
    }


def reconstruct_grid_from_ticks(
    image: np.ndarray,
    *,
    dark_cutoff: int,
    thickness: int,
    ocr_items: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    dark_mask = np.where((gray <= dark_cutoff) & (sat <= 120), 255, 0).astype(np.uint8)
    neutral_mask = np.where((sat <= 80) & (gray <= 245), 255, 0).astype(np.uint8)
    combined_mask = cv2.bitwise_or(dark_mask, neutral_mask)

    solutions = []
    for tick_mask in (dark_mask, combined_mask):
        y_axis, x_axis = find_axis_lines(tick_mask)
        tick_scan_mask = isolate_axis_tick_label_boxes_for_scan(tick_mask, image.shape[:2], ocr_items)
        y_ticks = find_y_tick_positions(tick_scan_mask, y_axis)
        x_ticks, x_tick_y = find_x_tick_positions(tick_scan_mask, y_axis, x_axis)
        x_span = max(x_ticks) - min(x_ticks) if len(x_ticks) >= 2 else 0.0
        y_span = max(y_ticks) - min(y_ticks) if len(y_ticks) >= 2 else 0.0
        bottom_bonus = 0.0
        if x_tick_y is not None:
            bottom_bonus = 2.0 if x_tick_y >= h * 0.55 else -4.0
        score = (
            len(y_ticks) * 1.0
            + len(x_ticks) * 2.0
            + (x_span / max(1, w)) * 10.0
            + (y_span / max(1, h)) * 4.0
            + bottom_bonus
        )
        solutions.append((score, y_axis, x_axis, y_ticks, x_ticks, x_tick_y))

    _, y_axis, x_axis, y_ticks, x_ticks, x_tick_y = max(solutions, key=lambda item: item[0])
    assist_candidates = [
        infer_ocr_tick_supplement(
            dark_mask,
            image,
            image.shape[:2],
            ocr_items,
            min_count=4,
            y_axis_x_hint=float(y_axis["x"]) if y_axis is not None else None,
        ),
        infer_ocr_tick_supplement(
            combined_mask,
            image,
            image.shape[:2],
            ocr_items,
            min_count=4,
            y_axis_x_hint=float(y_axis["x"]) if y_axis is not None else None,
        ),
    ]

    def assist_score(candidate: dict[str, object], axis: str) -> tuple[int, float, float]:
        key = "x_values" if axis == "x" else "y_values"
        values = [float(value) for value in candidate.get(key, [])]
        span = max(values) - min(values) if len(values) >= 2 else 0.0
        residual = regularity_score(values) if len(values) >= 3 else 1.0
        return len(values), -residual, span

    ocr_assist = dict(max(assist_candidates, key=lambda candidate: assist_score(candidate, "x")))
    y_assist = max(assist_candidates, key=lambda candidate: assist_score(candidate, "y"))
    for key in ("y_values", "y_axis_x", "y_source", "y_label_count"):
        ocr_assist[key] = y_assist.get(key)
    if (
        not is_valid_tick_series(sorted(y_ticks), h, min_span_frac=0.22)
        and len(ocr_assist["y_values"]) >= 4
    ):
        y_ticks = list(ocr_assist["y_values"])
    elif (
        len(ocr_assist["y_values"]) > len(y_ticks)
        and is_valid_tick_series(sorted([float(value) for value in ocr_assist["y_values"]]), h, min_span_frac=0.22)
    ):
        y_ticks = list(ocr_assist["y_values"])
    if (
        not is_valid_tick_series(sorted(x_ticks), w, min_span_frac=0.28)
        and len(ocr_assist["x_values"]) >= 4
    ):
        x_ticks = list(ocr_assist["x_values"])
        x_tick_y = ocr_assist["x_tick_y"] if ocr_assist["x_tick_y"] is not None else x_tick_y
    elif (
        len(ocr_assist["x_values"]) > len(x_ticks)
        and is_valid_tick_series(sorted([float(value) for value in ocr_assist["x_values"]]), w, min_span_frac=0.28)
    ):
        x_ticks = list(ocr_assist["x_values"])
        x_tick_y = ocr_assist["x_tick_y"] if ocr_assist["x_tick_y"] is not None else x_tick_y
    elif len(ocr_assist["x_values"]) >= 4 and len(x_ticks) > max(len(ocr_assist["x_values"]) + 2, int(len(ocr_assist["x_values"]) * 1.8)):
        x_ticks = list(ocr_assist["x_values"])
        x_tick_y = ocr_assist["x_tick_y"] if ocr_assist["x_tick_y"] is not None else x_tick_y
    if y_axis is None and ocr_assist["y_axis_x"] is not None and len(y_ticks) >= 2:
        y_axis = {
            "x": float(ocr_assist["y_axis_x"]),
            "y0": float(min(y_ticks)),
            "y1": float(max(y_ticks)),
            "length": float(max(y_ticks) - min(y_ticks)),
            "area": 0.0,
            "source": str(ocr_assist["y_source"]),
        }
    if (
        y_axis is None
        and len(x_ticks) >= 4
        and is_valid_tick_series(sorted(x_ticks), w, min_span_frac=0.28)
    ):
        y_axis = infer_y_axis_from_horizontal_grid_span(combined_mask, image.shape[:2], x_ticks)
    if x_axis is None and x_tick_y is not None and len(x_ticks) >= 2:
        x_axis = {
            "y": float(x_tick_y),
            "x0": float(min(x_ticks)),
            "x1": float(max(x_ticks)),
            "length": float(max(x_ticks) - min(x_ticks)),
            "area": 0.0,
            "source": str(ocr_assist["x_source"]),
        }
    if x_axis is not None and str(x_axis.get("source", "")) != "physical_tick_near_ocr_top":
        y_ticks = [y for y in y_ticks if y <= x_axis["y"] + 3]

    h_grid = np.zeros((h, w), dtype=np.uint8)
    v_grid = np.zeros((h, w), dtype=np.uint8)
    if y_axis is None and x_axis is not None and len(x_ticks) >= 2:
        y_label_column = left_numeric_ocr_column(ocr_items, image.shape[:2], min_count=4)
        y_label_positions = [float(item["y"]) for item in y_label_column]
        if y_label_positions and max(y_label_positions) - min(y_label_positions) >= h * 0.22:
            y0 = float(min(y_label_positions))
            y1 = float(max(max(y_label_positions), float(x_axis["y"])))
            source = "x_tick_scan_with_y_label_span"
        else:
            x0 = max(0, int(round(min(x_ticks))) - 4)
            x1 = min(w - 1, int(round(max(x_ticks))) + 4)
            y_limit = max(0, int(round(float(x_axis["y"]))) - 2)
            gray_roi = gray[:y_limit, x0 : x1 + 1]
            hsv_roi = hsv[:y_limit, x0 : x1 + 1]
            content = np.where((gray_roi <= 245) & (hsv_roi[:, :, 1] <= 150), 255, 0).astype(np.uint8)
            ys = np.where(content > 0)[0]
            if len(ys):
                y0 = float(max(0, int(np.percentile(ys, 2)) - 2))
                y1 = float(x_axis["y"])
                source = "x_tick_scan_with_content_span"
            else:
                y0 = y1 = 0.0
                source = "none"
        # P2 is anchored by physical tick marks near OCR label boxes. A detected
        # axis line can refine the span, but must not gate whether tick lines exist.
        if source != "none" and y1 - y0 >= h * 0.12:
            y_axis = {
                "x": float(min(x_ticks)),
                "y0": y0,
                "y1": y1,
                "length": float(y1 - y0),
                "area": 0.0,
                "source": source,
            }
    if y_axis is None:
        return cv2.bitwise_or(h_grid, v_grid), h_grid, v_grid

    y_values = sorted(y_ticks)
    x_values = sorted(x_ticks)
    if x_axis is not None:
        y_values = add_axis_aligned_endpoint_tick(y_values, float(x_axis["y"]))
    if not is_valid_tick_series(y_values, h, min_span_frac=0.22):
        y_values = []
    if not is_valid_tick_series(x_values, w, min_span_frac=0.28):
        x_values = []
    if len(y_values) < 2 and len(x_values) < 2:
        return cv2.bitwise_or(h_grid, v_grid), h_grid, v_grid

    if len(y_values) >= 2:
        # The reconstructed vertical grid should span the plot area, not the
        # x-axis label or legend band. Stable y tick lines define that area.
        y0_candidates = [*y_values]
        y1_candidates = [*y_values]
    elif y_axis is not None and str(y_axis.get("source", "")) == "horizontal_grid_span":
        y0_candidates = [y_axis["y0"]]
        y1_candidates = [y_axis["y1"]]
    else:
        y0_candidates = [y_axis["y0"], *y_values]
        y1_candidates = [y_axis["y1"], *y_values]
        if x_tick_y is not None:
            y1_candidates.append(x_tick_y)
        if x_axis is not None:
            y1_candidates.append(x_axis["y"])
    y0 = int(round(min(y0_candidates)))
    y1 = int(round(max(y1_candidates)))
    if (
        x_tick_y is not None
        and x_axis is not None
        and str(x_axis.get("source", "")) != "physical_tick_near_ocr_top"
    ):
        y1 = min(y1, int(round(float(x_tick_y))))
    x0 = int(round(y_axis["x"]))
    x1 = infer_plot_x1_from_content(image, x0=x0, y0=y0, y1=y1, x_ticks=x_values)
    x_values = add_plot_border_endpoint_ticks(
        image,
        x_values,
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
    )

    if len(y_values) >= 2:
        for y in y_values:
            cv2.line(h_grid, (x0, int(round(y))), (x1, int(round(y))), 255, thickness, cv2.LINE_AA)
    if len(x_values) >= 2:
        for x in x_values:
            cv2.line(v_grid, (int(round(x)), y0), (int(round(x)), y1), 255, thickness, cv2.LINE_AA)
    return cv2.bitwise_or(h_grid, v_grid), h_grid, v_grid

def build_grid_geometry_evidence(
    direct_horizontal: np.ndarray,
    direct_vertical: np.ndarray,
    tick_horizontal: np.ndarray,
    tick_vertical: np.ndarray,
) -> dict[str, object]:
    return {
        "direct_grid": {
            "horizontal_count": grid_line_count(direct_horizontal, "horizontal"),
            "vertical_count": grid_line_count(direct_vertical, "vertical"),
        },
        "tick_supplement": {
            "horizontal_count": grid_line_count(tick_horizontal, "horizontal"),
            "vertical_count": grid_line_count(tick_vertical, "vertical"),
        },
        "interpretation": {
            "direct_grid": "lines reconstructed from combined mask; strongest pixel evidence",
            "tick_supplement": "lines inferred from physical axis/tick marks, including OCR-assisted local search for real short ticks; use when corresponding direct direction is missing",
            "semantic_guide": "third-priority guide lines inferred from OCR/MLLM semantics after direct grid and physical tick evidence are insufficient",
        },
    }
