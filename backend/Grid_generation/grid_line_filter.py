from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from grid_io import read_image, write_image, write_json_file, discover_images, evenly_sample
from grid_masks import make_line_masks
from grid_visual import (
    mask_panel,
    overlay_masks,
    overlay_grid,
    draw_grid_label_overlay,
    draw_ocr_overlay,
    make_ocr_summary_panel,
    make_ocr_label_lab_preview,
    make_preview,
)
from grid_geometry import (
    reconstruct_grid,
    reconstruct_grid_from_ticks,
    build_grid_geometry_evidence,
    merge_grid_by_hierarchy,
    build_semantic_guide_grid,
    grid_line_count,
    grid_positions_and_bounds,
    prune_grid_to_mllm_regular_ticks,
    complete_regular_grid_to_target,
    mllm_tick_target,
    suppress_line_mask_text_regions,
    suppress_direct_grid_text_regions,
)
from grid_ocr import (
    run_paddle_ocr,
    build_ocr_axis_evidence,
    refine_ocr_roles_with_mllm,
    split_merged_ocr_items_with_mllm,
    merge_split_ocr_items_with_mllm,
    split_numeric_gap_ocr_items,
    refine_mllm_split_boxes_by_projection,
    regularize_mllm_split_sequence_geometry,
    regularize_canonical_numeric_axis_geometry,
    restore_numeric_gap_split_roles,
    add_mllm_missing_label_boxes,
    canonicalize_items_with_mllm_text,
)
from grid_mllm import run_mllm_axis_extraction, mllm_prompt
from grid_bindings import build_fused_axis_evidence, build_grid_label_bindings
from grid_adjudication import arbitrate_priority_grids
from grid_math import parse_numeric_label

DEFAULT_MLLM_ENDPOINT = "https://api.vveai.com/v1/chat/completions"

def load_local_env_files() -> None:
    for env_path in (Path(".env"), Path(".env.local")):
        if not env_path.exists():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

def safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)

def load_cached_mllm_with_source(path: Path, cache_root: Path | None) -> tuple[dict[str, object] | None, Path | None]:
    if cache_root is None:
        return None, None
    root = cache_root.resolve()
    if not root.exists():
        return None, None
    matches = list(root.rglob(f"{path.stem}_mllm_axis.json"))
    if not matches:
        return None, None
    preferred = sorted(
        matches,
        key=lambda candidate: (
            0 if path.parent.name in str(candidate.parent) else 1,
            len(str(candidate)),
        ),
    )
    cache_path = preferred[0]
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return (payload if isinstance(payload, dict) else None), cache_path

def semantic_missing_only(
    base_mask,
    guide_mask,
    orientation: str,
    *,
    thickness: int,
    target_count: int,
) -> tuple[object, dict[str, object]]:
    base_positions, base_bounds = grid_positions_and_bounds(base_mask, orientation)
    guide_positions, guide_bounds = grid_positions_and_bounds(guide_mask, orientation)
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_applicable",
        "base_count": len(base_positions),
        "guide_count": len(guide_positions),
        "target_count": target_count,
        "added_positions": [],
    }
    if target_count <= 0 or not base_positions or not guide_positions:
        return base_mask, details
    if len(base_positions) >= target_count:
        details["reason"] = "base_already_satisfies_target"
        return base_mask, details
    if len(base_positions) < max(3, target_count - 2):
        details["reason"] = "base_too_incomplete_for_missing_only"
        return base_mask, details

    ordered_base = sorted(float(value) for value in base_positions)
    if len(ordered_base) >= 2:
        gaps = [b - a for a, b in zip(ordered_base, ordered_base[1:]) if b - a > 1.0]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.0
    else:
        median_gap = 0.0
    tolerance = max(4.0, min(12.0, median_gap * 0.28 if median_gap > 0 else 6.0))
    missing = [
        float(value)
        for value in sorted(float(value) for value in guide_positions)
        if min(abs(float(value) - base) for base in ordered_base) > tolerance
    ]
    max_missing = target_count - len(base_positions)
    if not missing:
        details["reason"] = "no_missing_guide_positions"
        return base_mask, details
    if len(missing) > max_missing:
        details["reason"] = "guide_sequence_not_aligned_with_base"
        details["candidate_missing_positions"] = [round(value, 3) for value in missing]
        details["tolerance"] = round(tolerance, 3)
        return base_mask, details

    output = base_mask.copy()
    bounds = base_bounds or guide_bounds
    if bounds is None:
        details["reason"] = "missing_bounds"
        return base_mask, details
    start, end = bounds
    for position in missing:
        value = int(round(position))
        if orientation == "horizontal":
            cv2.line(output, (start, value), (end, value), 255, thickness, cv2.LINE_AA)
        else:
            cv2.line(output, (value, start), (value, end), 255, thickness, cv2.LINE_AA)
    details.update(
        {
            "enabled": True,
            "reason": "semantic_missing_only",
            "added_positions": [round(value, 3) for value in missing],
            "tolerance": round(tolerance, 3),
        }
    )
    return output, details

def _axis_tick_positions(
    ocr_axis_evidence: dict[str, object],
    axis_key: str,
    position_key: str,
    *,
    tolerance: float = 3.0,
) -> list[float]:
    axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
    ticks = axis.get("ticks", []) if isinstance(axis, dict) else []
    if not isinstance(ticks, list):
        return []
    values: list[float] = []
    numeric_records: list[tuple[float, float]] = []
    for item in ticks:
        if not isinstance(item, dict):
            continue
        label_kind = str(item.get("label_kind", "tick_label") or "tick_label")
        if label_kind != "tick_label":
            continue
        if item.get("canonical_axis") not in {None, axis_key}:
            continue
        value = item.get(position_key)
        if value is None:
            center = item.get("center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                value = center[0 if position_key == "x" else 1]
        try:
            numeric_value = item.get("numeric")
            if numeric_value is None:
                numeric_value = parse_numeric_label(str(item.get("text", "") or ""))
            position = float(value)
            values.append(position)
            if numeric_value is not None:
                numeric_records.append((position, float(numeric_value)))
        except (TypeError, ValueError):
            continue
    values = sorted(values)
    merged: list[list[float]] = []
    for value in values:
        if not merged or abs(value - float(np.median(merged[-1]))) > tolerance:
            merged.append([value])
        else:
            merged[-1].append(value)
    positions = [float(np.median(group)) for group in merged]
    return adjust_regular_numeric_endpoint_guides(positions, numeric_records)

def adjust_regular_numeric_endpoint_guides(
    positions: list[float],
    numeric_records: list[tuple[float, float]],
) -> list[float]:
    if len(positions) < 5 or len(numeric_records) != len(positions):
        return positions
    records = sorted(numeric_records, key=lambda item: item[0])
    numeric_values = [float(value) for _, value in records]
    numeric_diffs = np.diff(np.array(numeric_values, dtype=np.float64))
    numeric_steps = [abs(float(value)) for value in numeric_diffs if abs(float(value)) > 1e-9]
    if len(numeric_steps) != len(numeric_diffs):
        return positions
    numeric_step = float(np.median(np.array(numeric_steps, dtype=np.float64)))
    if numeric_step <= 0:
        return positions
    numeric_residual = float(np.median(np.abs(np.array(numeric_steps, dtype=np.float64) - numeric_step)) / max(1.0, numeric_step))
    if numeric_residual > 0.05:
        return positions

    adjusted = sorted(float(value) for value in positions)
    diffs = np.diff(np.array(adjusted, dtype=np.float64))
    positive = [float(value) for value in diffs if float(value) > 1.0]
    if len(positive) != len(diffs):
        return positions
    step_pool = positive[1:-1] if len(positive) >= 4 else positive
    step = float(np.median(np.array(step_pool, dtype=np.float64)))
    if step <= 1.0:
        return positions
    compressed_threshold = max(4.0, step * 0.12)
    max_endpoint_shift = max(6.0, step * 0.35)

    first_gap = adjusted[1] - adjusted[0]
    if step - first_gap > compressed_threshold:
        candidate = adjusted[1] - step
        if abs(candidate - adjusted[0]) <= max_endpoint_shift:
            adjusted[0] = candidate

    last_gap = adjusted[-1] - adjusted[-2]
    if step - last_gap > compressed_threshold:
        candidate = adjusted[-2] + step
        if abs(candidate - adjusted[-1]) <= max_endpoint_shift:
            adjusted[-1] = candidate
    return adjusted

def snap_endpoint_guides_to_reference(
    positions: list[float],
    reference_mask: np.ndarray,
    orientation: str,
) -> tuple[list[float], dict[str, object]]:
    details: dict[str, object] = {"enabled": False, "reason": "not_needed", "snaps": []}
    if len(positions) < 5:
        details["reason"] = "too_few_positions"
        return positions, details
    reference_positions, _ = grid_positions_and_bounds(reference_mask, orientation)
    if not reference_positions:
        details["reason"] = "no_reference_positions"
        return positions, details
    adjusted = sorted(float(value) for value in positions)
    diffs = np.diff(np.array(adjusted, dtype=np.float64))
    positive = [float(value) for value in diffs if float(value) > 1.0]
    if not positive:
        details["reason"] = "no_positive_step"
        return positions, details
    step = float(np.median(np.array(positive, dtype=np.float64)))
    threshold = max(3.0, step * 0.08)
    snaps: list[dict[str, float | int]] = []
    for index in (0, len(adjusted) - 1):
        nearest = min((float(value) for value in reference_positions), key=lambda value: abs(value - adjusted[index]))
        distance = abs(nearest - adjusted[index])
        if 0.5 < distance <= threshold:
            snaps.append({"index": index, "from": round(adjusted[index], 3), "to": round(nearest, 3), "distance": round(distance, 3)})
            adjusted[index] = nearest
    if snaps:
        details.update({"enabled": True, "reason": "regular_numeric_endpoint_snapped_to_physical_tick", "snaps": snaps})
        return adjusted, details
    details["reason"] = "no_endpoint_snap_within_threshold"
    return positions, details

def _span_from_reference(
    line_mask,
    orientation: str,
    fallback_positions: list[float],
    extent: int,
) -> tuple[int, int]:
    positions, bounds = grid_positions_and_bounds(line_mask, orientation)
    if bounds is not None and bounds[1] > bounds[0]:
        return bounds
    if len(positions) >= 2:
        return int(round(min(positions))), int(round(max(positions)))
    if len(fallback_positions) >= 2:
        return int(round(min(fallback_positions))), int(round(max(fallback_positions)))
    return int(round(extent * 0.08)), int(round(extent * 0.92))

def _axis_type(ocr_axis_evidence: dict[str, object], axis_key: str) -> str:
    axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
    if not isinstance(axis, dict):
        return "unknown"
    return str(axis.get("type", "unknown") or "unknown")

def _span_from_tick_centers(
    positions: list[float],
    extent: int,
    *,
    include_outer_half_step: bool,
) -> tuple[int, int] | None:
    clean = sorted(float(value) for value in positions if np.isfinite(float(value)))
    if len(clean) < 2:
        return None
    start = clean[0]
    end = clean[-1]
    if include_outer_half_step and len(clean) >= 3:
        diffs = np.diff(np.array(clean, dtype=np.float64))
        positive = [float(value) for value in diffs if float(value) > 0]
        if positive:
            half_step = float(np.median(positive)) * 0.5
            start -= half_step
            end += half_step
    start = max(0, min(extent - 1, int(round(start))))
    end = max(0, min(extent - 1, int(round(end))))
    if end <= start:
        return None
    return start, end

def build_semantic_guide_candidate_grid(
    image_shape: tuple[int, int],
    reference_horizontal,
    reference_vertical,
    ocr_axis_evidence: dict[str, object],
    *,
    thickness: int,
) -> tuple[object, object, object, dict[str, object]]:
    h, w = image_shape[:2]
    x_tick_positions = _axis_tick_positions(ocr_axis_evidence, "x_axis", "x")
    y_tick_positions = _axis_tick_positions(ocr_axis_evidence, "y_axis", "y")
    x_tick_positions, x_endpoint_snap = snap_endpoint_guides_to_reference(x_tick_positions, reference_vertical, "vertical")
    y_tick_positions, y_endpoint_snap = snap_endpoint_guides_to_reference(y_tick_positions, reference_horizontal, "horizontal")
    horizontal = np.zeros((h, w), dtype=np.uint8)
    vertical = np.zeros((h, w), dtype=np.uint8)

    x0, x1 = _span_from_reference(reference_horizontal, "horizontal", x_tick_positions, w)
    y0, y1 = _span_from_reference(reference_vertical, "vertical", y_tick_positions, h)
    y_axis_type = _axis_type(ocr_axis_evidence, "y_axis")
    if y_axis_type == "category" and len(x_tick_positions) >= 2:
        category_horizontal_span = _span_from_tick_centers(
            x_tick_positions,
            w,
            include_outer_half_step=False,
        )
        if category_horizontal_span is not None:
            x0, x1 = category_horizontal_span
    if y_axis_type == "category" and len(y_tick_positions) >= 2:
        category_vertical_span = _span_from_tick_centers(
            y_tick_positions,
            h,
            include_outer_half_step=True,
        )
        if category_vertical_span is not None:
            y0, y1 = category_vertical_span
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    for y in y_tick_positions:
        value = int(round(y))
        if 0 <= value < h:
            cv2.line(horizontal, (x0, value), (x1, value), 255, thickness, cv2.LINE_AA)
    for x in x_tick_positions:
        value = int(round(x))
        if 0 <= value < w:
            cv2.line(vertical, (value, y0), (value, y1), 255, thickness, cv2.LINE_AA)

    metadata = {
        "source": "ocr_label_lab_box_midline",
        "x_tick_positions": [round(value, 3) for value in x_tick_positions],
        "y_tick_positions": [round(value, 3) for value in y_tick_positions],
        "horizontal_span": [int(x0), int(x1)],
        "vertical_span": [int(y0), int(y1)],
        "span_policy": {
            "y_axis_type": y_axis_type,
            "category_y_axis_uses_x_tick_span": bool(y_axis_type == "category" and len(x_tick_positions) >= 2),
            "category_y_axis_uses_y_label_outer_span": bool(y_axis_type == "category" and len(y_tick_positions) >= 2),
        },
        "endpoint_reference_snap": {
            "x_axis": x_endpoint_snap,
            "y_axis": y_endpoint_snap,
        },
        "horizontal_count": grid_line_count(horizontal, "horizontal"),
        "vertical_count": grid_line_count(vertical, "vertical"),
    }
    return cv2.bitwise_or(horizontal, vertical), horizontal, vertical, metadata

def ocr_numeric_tick_positions_for_axis(
    ocr_axis_evidence: dict[str, object],
    axis_key: str,
    position_key: str,
) -> list[float]:
    axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
    ticks = axis.get("ticks", []) if isinstance(axis, dict) else []
    values: list[float] = []
    for tick in ticks if isinstance(ticks, list) else []:
        if not isinstance(tick, dict) or tick.get("numeric") is None:
            continue
        try:
            values.append(float(tick[position_key]))
        except (KeyError, TypeError, ValueError):
            continue
    return values

def ocr_tick_positions_for_axis(
    ocr_axis_evidence: dict[str, object],
    axis_key: str,
    position_key: str,
) -> list[float]:
    axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
    ticks = axis.get("ticks", []) if isinstance(axis, dict) else []
    values: list[float] = []
    numeric_records: list[tuple[float, float]] = []
    for tick in ticks if isinstance(ticks, list) else []:
        if not isinstance(tick, dict):
            continue
        try:
            value = float(tick[position_key])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        values.append(value)
        numeric_value = tick.get("numeric")
        if numeric_value is None:
            numeric_value = parse_numeric_label(str(tick.get("text", "") or ""))
        try:
            if numeric_value is not None:
                numeric_records.append((value, float(numeric_value)))
        except (TypeError, ValueError):
            pass
    return adjust_regular_numeric_endpoint_guides(sorted(values), numeric_records)

def build_ocr_bound_native_grid(
    direct_horizontal: np.ndarray,
    direct_vertical: np.ndarray,
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    *,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    h_target = mllm_tick_target(mllm_result, "y_axis")
    v_target = mllm_tick_target(mllm_result, "x_axis")
    h_guides = ocr_tick_positions_for_axis(ocr_axis_evidence, "y_axis", "y")
    v_guides = ocr_tick_positions_for_axis(ocr_axis_evidence, "x_axis", "x")
    horizontal, h_prune = prune_grid_to_mllm_regular_ticks(
        direct_horizontal,
        "horizontal",
        h_target,
        thickness=thickness,
        guide_positions=h_guides,
    )
    horizontal, h_completion = complete_regular_grid_to_target(
        horizontal,
        "horizontal",
        h_target,
        thickness=thickness,
        guide_positions=h_guides,
    )
    vertical, v_prune = prune_grid_to_mllm_regular_ticks(
        direct_vertical,
        "vertical",
        v_target,
        thickness=thickness,
        guide_positions=v_guides,
    )
    vertical, v_completion = complete_regular_grid_to_target(
        vertical,
        "vertical",
        v_target,
        thickness=thickness,
        guide_positions=v_guides,
    )
    grid = cv2.bitwise_or(horizontal, vertical)
    metadata = {
        "source": "combined_mask_ocr_bound",
        "horizontal_count_before": grid_line_count(direct_horizontal, "horizontal"),
        "vertical_count_before": grid_line_count(direct_vertical, "vertical"),
        "horizontal_count": grid_line_count(horizontal, "horizontal"),
        "vertical_count": grid_line_count(vertical, "vertical"),
        "horizontal_target": h_target,
        "vertical_target": v_target,
        "horizontal_guide_positions": [round(value, 3) for value in h_guides],
        "vertical_guide_positions": [round(value, 3) for value in v_guides],
        "horizontal_prune": h_prune,
        "horizontal_completion": h_completion,
        "vertical_prune": v_prune,
        "vertical_completion": v_completion,
    }
    return grid, horizontal, vertical, metadata

def redraw_tick_grid_from_positions(
    shape: tuple[int, int],
    horizontal_positions: list[float],
    vertical_positions: list[float],
    *,
    horizontal_bounds: tuple[int, int] | None,
    vertical_bounds: tuple[int, int] | None,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    horizontal = np.zeros((h, w), dtype=np.uint8)
    vertical = np.zeros((h, w), dtype=np.uint8)
    if horizontal_bounds is None and len(vertical_positions) >= 2:
        horizontal_bounds = (pixel_round(min(vertical_positions)), pixel_round(max(vertical_positions)))
    if vertical_bounds is None and len(horizontal_positions) >= 2:
        vertical_bounds = (pixel_round(min(horizontal_positions)), pixel_round(max(horizontal_positions)))
    if horizontal_bounds is not None:
        x0, x1 = horizontal_bounds
        x0 = max(0, min(w - 1, pixel_round(x0)))
        x1 = max(0, min(w - 1, pixel_round(x1)))
        if x1 < x0:
            x0, x1 = x1, x0
        for y in horizontal_positions:
            y_i = max(0, min(h - 1, pixel_round(float(y))))
            cv2.line(horizontal, (x0, y_i), (x1, y_i), 255, thickness, cv2.LINE_AA)
    if vertical_bounds is not None:
        y0, y1 = vertical_bounds
        y0 = max(0, min(h - 1, pixel_round(y0)))
        y1 = max(0, min(h - 1, pixel_round(y1)))
        if y1 < y0:
            y0, y1 = y1, y0
        for x in vertical_positions:
            x_i = max(0, min(w - 1, pixel_round(float(x))))
            cv2.line(vertical, (x_i, y0), (x_i, y1), 255, thickness, cv2.LINE_AA)
    return cv2.bitwise_or(horizontal, vertical), horizontal, vertical

def regular_position_guides(values: list[float]) -> bool:
    if len(values) < 3:
        return False
    ordered = sorted(float(value) for value in values)
    diffs = np.diff(np.array(ordered, dtype=np.float64))
    positive = [float(diff) for diff in diffs if float(diff) > 1.0]
    if len(positive) != len(diffs):
        return False
    median = float(np.median(np.array(positive, dtype=np.float64)))
    if median <= 1.0:
        return False
    residual = float(np.median(np.abs(np.array(positive, dtype=np.float64) - median)) / max(1.0, median))
    return residual <= 0.20

def max_step_residual(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    ordered = sorted(float(value) for value in values)
    diffs = np.diff(np.array(ordered, dtype=np.float64))
    positive = [float(diff) for diff in diffs if float(diff) > 1.0]
    if not positive:
        return 1.0
    median = float(np.median(np.array(positive, dtype=np.float64)))
    if median <= 1.0:
        return 1.0
    return float(np.max(np.abs(np.array(positive, dtype=np.float64) - median)) / max(1.0, median))

def regular_sequence_endpoint_adjustment(values: list[float]) -> tuple[list[float], dict[str, object] | None]:
    if len(values) < 6:
        return values, None
    ordered = [float(value) for value in values]
    best: tuple[float, int, float, list[float]] | None = None
    for anchor_start in (0, 1):
        for anchor_end in range(len(ordered), max(anchor_start + 4, len(ordered) - 3), -1):
            indexes = np.arange(anchor_start, anchor_end, dtype=np.float64)
            coords = np.array(ordered[anchor_start:anchor_end], dtype=np.float64)
            if len(coords) < 4 or float(np.ptp(indexes)) <= 0:
                continue
            slope, intercept = np.polyfit(indexes, coords, 1)
            predicted_all = [float(slope * index + intercept) for index in range(len(ordered))]
            residuals = np.abs(coords - (slope * indexes + intercept))
            step = abs(float(slope))
            if step <= 1.0:
                continue
            median_residual = float(np.median(residuals))
            max_residual = float(np.max(residuals))
            if median_residual > max(0.75, step * 0.035) or max_residual > max(1.5, step * 0.055):
                continue
            candidate = list(ordered)
            adjusted_count = 0
            for endpoint in (0, len(ordered) - 1):
                if anchor_start <= endpoint < anchor_end:
                    continue
                delta = predicted_all[endpoint] - ordered[endpoint]
                if abs(delta) >= max(1.5, step * 0.035) and abs(delta) <= max(10.0, step * 0.28):
                    candidate[endpoint] = predicted_all[endpoint]
                    adjusted_count += 1
            if not adjusted_count:
                continue
            score = median_residual + max_residual * 0.5 + (len(ordered) - (anchor_end - anchor_start)) * 0.1
            if best is None or score < best[0]:
                best = (score, adjusted_count, step, candidate)
    if best is None:
        return values, None
    _, adjusted_count, step, candidate = best
    changes = [
        {
            "index": index,
            "from": round(float(before), 3),
            "to": round(float(after), 3),
            "delta": round(float(after - before), 3),
        }
        for index, (before, after) in enumerate(zip(ordered, candidate))
        if abs(float(after) - float(before)) > 1e-6
    ]
    return candidate, {
        "reason": "regular_sequence_endpoint_fit",
        "adjusted_count": adjusted_count,
        "step": round(float(step), 3),
        "changes": changes,
    }

def pixel_round(value: float) -> int:
    return int(np.floor(float(value) + 0.5))

def build_ocr_aligned_tick_grid(
    tick_horizontal: np.ndarray,
    tick_vertical: np.ndarray,
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    *,
    thickness: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    h_positions, h_bounds = grid_positions_and_bounds(tick_horizontal, "horizontal")
    v_positions, v_bounds = grid_positions_and_bounds(tick_vertical, "vertical")
    h_target = mllm_tick_target(mllm_result, "y_axis")
    v_target = mllm_tick_target(mllm_result, "x_axis")
    h_guides = sorted(ocr_numeric_tick_positions_for_axis(ocr_axis_evidence, "y_axis", "y"))
    v_guides = sorted(ocr_numeric_tick_positions_for_axis(ocr_axis_evidence, "x_axis", "x"))
    selected_h = sorted(float(value) for value in h_positions)
    selected_v = sorted(float(value) for value in v_positions)
    details: dict[str, object] = {
        "enabled": False,
        "horizontal": {"enabled": False, "reason": "not_applied"},
        "vertical": {"enabled": False, "reason": "not_applied"},
    }

    if (
        h_target is not None
        and len(selected_h) == h_target
        and len(h_guides) == h_target
        and regular_position_guides(h_guides)
    ):
        errors = np.abs(np.array(selected_h, dtype=np.float64) - np.array(h_guides, dtype=np.float64))
        guide_step = float(np.median(np.diff(np.array(h_guides, dtype=np.float64)))) if len(h_guides) >= 2 else 0.0
        if float(errors.max()) >= max(8.0, guide_step * 0.24) or float(errors.mean()) >= max(4.0, guide_step * 0.12):
            details["horizontal"] = {
                "enabled": False,
                "reason": "preserve_complete_physical_tick_sequence_no_ocr_rewrite",
                "before": [round(float(value), 3) for value in h_positions],
                "ocr_guides": [round(float(value), 3) for value in h_guides],
                "max_error": round(float(errors.max()), 3),
                "mean_error": round(float(errors.mean()), 3),
            }

    if (
        h_target is not None
        and len(selected_h) < h_target
        and len(selected_h) >= max(4, int(h_target * 0.55))
        and len(h_guides) == h_target
        and regular_position_guides(h_guides)
    ):
        guide_step = float(np.median(np.diff(np.array(h_guides, dtype=np.float64)))) if len(h_guides) >= 2 else 0.0
        tolerance = max(5.0, guide_step * 0.22)
        distances = [min(abs(float(position) - float(guide)) for guide in h_guides) for position in selected_h]
        if distances and max(distances) <= tolerance:
            before = list(selected_h)
            details["horizontal"] = {
                "enabled": False,
                "reason": "preserve_partial_physical_tick_sequence_no_ocr_completion",
                "before": [round(float(value), 3) for value in before],
                "ocr_guides": [round(float(value), 3) for value in h_guides],
                "target_count": h_target,
                "max_scan_to_guide_distance": round(float(max(distances)), 3),
                "tolerance": round(float(tolerance), 3),
            }

    if (
        h_target is not None
        and len(selected_h) == h_target
        and len(h_guides) == h_target
        and regular_position_guides(h_guides)
        and len(selected_h) >= 4
    ):
        guide_step = float(np.median(np.diff(np.array(h_guides, dtype=np.float64)))) if len(h_guides) >= 2 else 0.0
        snap_threshold = max(0.45, guide_step * 0.012)
        max_snap_distance = min(4.0, max(2.5, guide_step * 0.10))
        before = list(selected_h)
        physical_residual = max_step_residual(before)
        guide_residual = max_step_residual(h_guides)
        preserve_regular_physical_ticks = (
            regular_position_guides(before)
            and physical_residual <= max(0.025, guide_residual * 0.65)
        )
        if preserve_regular_physical_ticks:
            horizontal_details = details.get("horizontal")
            preserve_details = {
                "enabled": False,
                "reason": "preserve_complete_regular_physical_tick_sequence",
                "physical_residual": round(float(physical_residual), 4),
                "ocr_guide_residual": round(float(guide_residual), 4),
                "snap_threshold": round(float(snap_threshold), 3),
                "max_snap_distance": round(float(max_snap_distance), 3),
            }
            if isinstance(horizontal_details, dict) and horizontal_details.get("enabled"):
                horizontal_details["local_tick_snap_skipped"] = preserve_details
            else:
                details["horizontal"] = preserve_details
        else:
            horizontal_details = details.get("horizontal")
            preserve_details = {
                "enabled": False,
                "reason": "preserve_physical_tick_sequence_no_ocr_snap",
                "before": [round(float(value), 3) for value in before],
                "ocr_guides": [round(float(value), 3) for value in h_guides],
                "max_scan_to_guide_distance": round(
                    float(max(abs(float(before[index]) - float(h_guides[index])) for index in range(len(before)))),
                    3,
                ),
                "snap_threshold": round(float(snap_threshold), 3),
                "max_snap_distance": round(float(max_snap_distance), 3),
            }
            if isinstance(horizontal_details, dict) and horizontal_details.get("enabled"):
                horizontal_details["local_tick_snap_skipped"] = preserve_details
            else:
                details["horizontal"] = preserve_details

    if (
        v_target is not None
        and len(selected_v) > v_target
        and len(selected_v) <= v_target + 3
    ):
        before = list(selected_v)
        candidates = []
        for remove_index in range(len(selected_v)):
            subset = [value for index, value in enumerate(selected_v) if index != remove_index]
            if len(subset) != v_target:
                continue
            guide_error = 0.0
            if len(v_guides) == v_target:
                guide_error = float(
                    np.mean([min(abs(float(guide) - float(position)) for position in subset) for guide in v_guides])
                )
            candidates.append((max_step_residual(subset), guide_error, remove_index, subset))
        if candidates:
            residual, guide_error, remove_index, subset = min(candidates, key=lambda item: (item[0], item[1]))
            if residual <= 0.08:
                selected_v = [float(value) for value in subset]
                details["enabled"] = True
                details["vertical"] = {
                    "enabled": True,
                    "reason": "regular_physical_tick_subset_after_label_box_isolation",
                    "before": [round(float(value), 3) for value in before],
                    "after": [round(float(value), 3) for value in selected_v],
                    "removed_index": remove_index,
                    "removed_position": round(float(before[remove_index]), 3),
                    "max_step_residual": round(float(residual), 3),
                    "guide_error": round(float(guide_error), 3),
                }

    if (
        v_target is not None
        and len(selected_v) == v_target
        and len(v_guides) == v_target
        and regular_position_guides(v_guides)
        and selected_v
    ):
        errors = np.abs(np.array(selected_v, dtype=np.float64) - np.array(v_guides, dtype=np.float64))
        guide_step = float(np.median(np.diff(np.array(v_guides, dtype=np.float64)))) if len(v_guides) >= 2 else 0.0
        if float(errors.max()) >= max(10.0, guide_step * 0.25) or float(errors.mean()) >= max(5.0, guide_step * 0.12):
            before = list(selected_v)
            details["vertical"] = {
                "enabled": False,
                "reason": "preserve_complete_physical_tick_sequence_no_ocr_rewrite",
                "before": [round(float(value), 3) for value in before],
                "ocr_guides": [round(float(value), 3) for value in v_guides],
                "max_error": round(float(errors.max()), 3),
                "mean_error": round(float(errors.mean()), 3),
            }

    if (
        v_target is not None
        and len(selected_v) == v_target
        and len(v_guides) == v_target
        and regular_position_guides(v_guides)
        and h_bounds is not None
        and selected_v
    ):
        x0, _ = h_bounds
        first_error = abs(float(selected_v[0]) - float(x0))
        guide_step = float(np.median(np.diff(np.array(v_guides, dtype=np.float64)))) if len(v_guides) >= 2 else 0.0
        guide_axis_distance = abs(float(v_guides[0]) - float(x0))
        if first_error >= max(8.0, guide_step * 0.18) and guide_axis_distance <= max(4.0, guide_step * 0.08):
            before = list(selected_v)
            selected_v[0] = float(x0)
            details["enabled"] = True
            details["vertical"] = {
                "enabled": True,
                "reason": "origin_tick_snapped_to_y_axis_border",
                "before": [round(float(value), 3) for value in before],
                "after": [round(float(value), 3) for value in selected_v],
                "axis_x": round(float(x0), 3),
                "first_error": round(float(first_error), 3),
                "guide_axis_distance": round(float(guide_axis_distance), 3),
            }

    if (
        v_target is not None
        and len(selected_v) == v_target
        and len(v_guides) == v_target
        and regular_position_guides(v_guides)
        and len(selected_v) >= 4
    ):
        guide_step = float(np.median(np.diff(np.array(v_guides, dtype=np.float64)))) if len(v_guides) >= 2 else 0.0
        snap_threshold = max(0.45, guide_step * 0.012)
        max_snap_distance = min(4.0, max(2.5, guide_step * 0.10))
        before = list(selected_v)
        physical_residual = max_step_residual(before)
        guide_residual = max_step_residual(v_guides)
        preserve_regular_physical_ticks = (
            regular_position_guides(before)
            and physical_residual <= max(0.025, guide_residual * 0.65)
        )
        if preserve_regular_physical_ticks:
            vertical_details = details.get("vertical")
            preserve_details = {
                "enabled": False,
                "reason": "preserve_complete_regular_physical_tick_sequence",
                "physical_residual": round(float(physical_residual), 4),
                "ocr_guide_residual": round(float(guide_residual), 4),
                "snap_threshold": round(float(snap_threshold), 3),
                "max_snap_distance": round(float(max_snap_distance), 3),
            }
            if isinstance(vertical_details, dict) and vertical_details.get("enabled"):
                vertical_details["local_tick_snap_skipped"] = preserve_details
            else:
                details["vertical"] = preserve_details
        else:
            vertical_details = details.get("vertical")
            preserve_details = {
                "enabled": False,
                "reason": "preserve_physical_tick_sequence_no_ocr_snap",
                "before": [round(float(value), 3) for value in before],
                "ocr_guides": [round(float(value), 3) for value in v_guides],
                "max_scan_to_guide_distance": round(
                    float(max(abs(float(before[index]) - float(v_guides[index])) for index in range(len(before)))),
                    3,
                ),
                "snap_threshold": round(float(snap_threshold), 3),
                "max_snap_distance": round(float(max_snap_distance), 3),
            }
            if isinstance(vertical_details, dict) and vertical_details.get("enabled"):
                vertical_details["local_tick_snap_skipped"] = preserve_details
            else:
                details["vertical"] = preserve_details

    if not details["enabled"]:
        return cv2.bitwise_or(tick_horizontal, tick_vertical), tick_horizontal, tick_vertical, details

    vertical_bounds = None
    if selected_h:
        vertical_bounds = (int(round(min(selected_h))), int(round(max(selected_h))))
    grid, horizontal, vertical = redraw_tick_grid_from_positions(
        tick_horizontal.shape[:2],
        selected_h,
        selected_v,
        horizontal_bounds=h_bounds,
        vertical_bounds=vertical_bounds or v_bounds,
        thickness=thickness,
    )
    return grid, horizontal, vertical, details

PRIORITY_GRID_CANDIDATES = {
    "priority1_native_grid": {
        "source": "combined_mask",
        "title": "Priority 1 native grid reconstructed from combined mask with OCR-bound pruning/completion",
    },
    "priority2_tick_scan_grid": {
        "source": "tick_supplement",
        "title": "Priority 2 grid reconstructed from physical ticks near labels",
    },
    "priority3_semantic_guide_grid": {
        "source": "semantic_guide",
        "title": "Priority 3 semantic guide reconstructed from canonical label center lines",
    },
}
PRIORITY_SOURCE_NUMBERS = {
    "combined_mask": 1,
    "tick_supplement": 2,
    "semantic_guide": 3,
}

def final_selection_name(priority_decision: dict[str, object]) -> tuple[str, dict[str, object]]:
    horizontal_source = str(priority_decision.get("y_axis_horizontal_grid_choice", "") or "")
    vertical_source = str(priority_decision.get("x_axis_vertical_grid_choice", "") or "")
    horizontal_priority = PRIORITY_SOURCE_NUMBERS.get(horizontal_source)
    vertical_priority = PRIORITY_SOURCE_NUMBERS.get(vertical_source)
    if horizontal_priority is not None and horizontal_priority == vertical_priority:
        name = f"final{horizontal_priority}"
    elif horizontal_priority is not None and vertical_priority is not None:
        name = f"finalH{horizontal_priority}V{vertical_priority}"
    else:
        name = "final"
    metadata = {
        "name": name,
        "horizontal_source": horizontal_source or None,
        "vertical_source": vertical_source or None,
        "horizontal_priority": horizontal_priority,
        "vertical_priority": vertical_priority,
        "single_priority": bool(horizontal_priority is not None and horizontal_priority == vertical_priority),
        "priority_number_map": {
            "1": "combined_mask",
            "2": "tick_supplement",
            "3": "semantic_guide",
        },
    }
    return name, metadata

def complete_final_grid_spans(
    grid_horizontal: np.ndarray,
    grid_vertical: np.ndarray,
    *,
    thickness: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    h, w = grid_horizontal.shape[:2]
    horizontal_positions, horizontal_bounds = grid_positions_and_bounds(grid_horizontal, "horizontal")
    vertical_positions, vertical_bounds = grid_positions_and_bounds(grid_vertical, "vertical")
    completed_horizontal = grid_horizontal.copy()
    completed_vertical = grid_vertical.copy()
    details: dict[str, object] = {
        "enabled": True,
        "policy": "extend_selected_final_lines_only",
        "horizontal_extended": False,
        "vertical_extended": False,
        "horizontal_bounds_before": list(horizontal_bounds) if horizontal_bounds else None,
        "vertical_bounds_before": list(vertical_bounds) if vertical_bounds else None,
        "horizontal_bounds_after": list(horizontal_bounds) if horizontal_bounds else None,
        "vertical_bounds_after": list(vertical_bounds) if vertical_bounds else None,
    }

    if horizontal_positions and horizontal_bounds is not None and len(vertical_positions) >= 2:
        x0 = max(0, min(horizontal_bounds[0], int(round(min(vertical_positions)))))
        x1 = min(w - 1, max(horizontal_bounds[1], int(round(max(vertical_positions)))))
        if x1 > x0 and (x0, x1) != horizontal_bounds:
            for y in horizontal_positions:
                cv2.line(completed_horizontal, (x0, int(round(y))), (x1, int(round(y))), 255, thickness, cv2.LINE_AA)
            details["horizontal_extended"] = True
            details["horizontal_bounds_after"] = [int(x0), int(x1)]

    if vertical_positions and vertical_bounds is not None and len(horizontal_positions) >= 2:
        y0 = max(0, min(vertical_bounds[0], int(round(min(horizontal_positions)))))
        y1 = min(h - 1, max(vertical_bounds[1], int(round(max(horizontal_positions)))))
        if y1 > y0 and (y0, y1) != vertical_bounds:
            for x in vertical_positions:
                cv2.line(completed_vertical, (int(round(x)), y0), (int(round(x)), y1), 255, thickness, cv2.LINE_AA)
            details["vertical_extended"] = True
            details["vertical_bounds_after"] = [int(y0), int(y1)]

    if not details["horizontal_extended"] and not details["vertical_extended"]:
        details["reason"] = "already_complete_or_missing_cross_axis_reference"
    return completed_horizontal, completed_vertical, details

def clear_previous_final_outputs(out_base: Path) -> None:
    for path in out_base.parent.glob(out_base.name + "_final*"):
        if path.is_file():
            path.unlink()

def build_priority_candidate_outputs(
    candidates: dict[str, tuple[object, object]],
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    fused_axis_evidence: dict[str, object],
) -> dict[str, dict[str, object]]:
    outputs: dict[str, dict[str, object]] = {}
    for output_name, meta in PRIORITY_GRID_CANDIDATES.items():
        source = str(meta["source"])
        horizontal, vertical = candidates.get(source, (None, None))
        if horizontal is None or vertical is None:
            continue
        bindings = build_grid_label_bindings(
            horizontal,
            vertical,
            ocr_axis_evidence,
            mllm_result,
            fused_axis_evidence,
        )
        outputs[output_name] = {
            "name": output_name,
            "source": source,
            "title": meta["title"],
            "horizontal_count": grid_line_count(horizontal, "horizontal"),
            "vertical_count": grid_line_count(vertical, "vertical"),
            "bindings": bindings,
        }
    return outputs

def write_priority_candidate_outputs(
    out_base: Path,
    image,
    candidates: dict[str, tuple[object, object]],
    candidate_outputs: dict[str, dict[str, object]],
) -> None:
    for output_name, payload in candidate_outputs.items():
        source = str(payload.get("source", ""))
        horizontal, vertical = candidates[source]
        grid = cv2.bitwise_or(horizontal, vertical)
        write_image(out_base.with_name(out_base.name + f"_{output_name}.png"), mask_panel(grid))
        write_image(
            out_base.with_name(out_base.name + f"_{output_name}_horizontal.png"),
            mask_panel(horizontal),
        )
        write_image(
            out_base.with_name(out_base.name + f"_{output_name}_vertical.png"),
            mask_panel(vertical),
        )
        write_image(
            out_base.with_name(out_base.name + f"_{output_name}_overlay.png"),
            overlay_grid(image, grid),
        )
        bindings = payload.get("bindings", {})
        write_image(
            out_base.with_name(out_base.name + f"_{output_name}_label_overlay.png"),
            draw_grid_label_overlay(image, horizontal, vertical, bindings if isinstance(bindings, dict) else {}),
        )
        json_payload = {
            "name": payload.get("name"),
            "source": payload.get("source"),
            "title": payload.get("title"),
            "horizontal_count": payload.get("horizontal_count"),
            "vertical_count": payload.get("vertical_count"),
            "bindings": bindings,
        }
        write_json_file(out_base.with_name(out_base.name + f"_{output_name}_bindings.json"), json_payload)

def write_final_selection_outputs(
    out_base: Path,
    image,
    priority_grid_candidates: dict[str, tuple[object, object]],
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    fused_axis_evidence: dict[str, object],
    priority_decision: dict[str, object],
) -> tuple[object, object, object, dict[str, object], dict[str, object]]:
    output_name, metadata = final_selection_name(priority_decision)
    clear_previous_final_outputs(out_base)
    horizontal_source = str(priority_decision.get("y_axis_horizontal_grid_choice", "") or "")
    vertical_source = str(priority_decision.get("x_axis_vertical_grid_choice", "") or "")
    fallback_horizontal, fallback_vertical = next(iter(priority_grid_candidates.values()))
    grid_horizontal = priority_grid_candidates.get(horizontal_source, (fallback_horizontal, fallback_vertical))[0]
    grid_vertical = priority_grid_candidates.get(vertical_source, (fallback_horizontal, fallback_vertical))[1]
    grid_horizontal, grid_vertical, span_completion = complete_final_grid_spans(grid_horizontal, grid_vertical)
    metadata["span_completion"] = span_completion
    grid = cv2.bitwise_or(grid_horizontal, grid_vertical)
    grid_label_bindings = build_grid_label_bindings(
        grid_horizontal,
        grid_vertical,
        ocr_axis_evidence,
        mllm_result,
        fused_axis_evidence,
    )
    write_image(out_base.with_name(out_base.name + f"_{output_name}.png"), mask_panel(grid))
    write_image(out_base.with_name(out_base.name + f"_{output_name}_horizontal.png"), mask_panel(grid_horizontal))
    write_image(out_base.with_name(out_base.name + f"_{output_name}_vertical.png"), mask_panel(grid_vertical))
    write_image(out_base.with_name(out_base.name + f"_{output_name}_overlay.png"), overlay_grid(image, grid))
    write_image(
        out_base.with_name(out_base.name + f"_{output_name}_label_overlay.png"),
        draw_grid_label_overlay(image, grid_horizontal, grid_vertical, grid_label_bindings),
    )
    write_json_file(out_base.with_name(out_base.name + f"_{output_name}_bindings.json"), grid_label_bindings)
    write_json_file(
        out_base.with_name(out_base.name + f"_{output_name}_selection.json"),
        {**metadata, "priority_decision": priority_decision},
    )
    return grid, grid_horizontal, grid_vertical, grid_label_bindings, metadata

def labeled_tick_count(axis_bindings: dict[str, object] | None) -> int:
    if not isinstance(axis_bindings, dict):
        return 0
    bindings = axis_bindings.get("tick_bindings", [])
    if not isinstance(bindings, list):
        return 0
    count = 0
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        label = str(binding.get("label", "") or "").strip()
        source = str(binding.get("source", "") or "")
        if label and source != "none":
            count += 1
    return count

def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default

def priority_axis_quality(
    priority_decision: dict[str, object],
    *,
    axis_name: str,
    choice_key: str,
    reason_key: str,
    scores_key: str,
    axis_type_key: str,
) -> tuple[dict[str, object], list[str]]:
    scores = priority_decision.get(scores_key, {})
    if not isinstance(scores, dict) or not scores:
        return {"axis": axis_name, "available": False}, []
    choice = str(priority_decision.get(choice_key, "") or "")
    selected = scores.get(choice) if choice else None
    valid_scores = [data for data in scores.values() if isinstance(data, dict) and data.get("valid")]
    valid_count = len(valid_scores)
    summary: dict[str, object] = {
        "axis": axis_name,
        "available": True,
        "choice": choice or None,
        "reason": priority_decision.get(reason_key),
        "axis_type": priority_decision.get(axis_type_key),
        "valid_candidate_count": valid_count,
    }
    reasons: list[str] = []
    if not isinstance(selected, dict):
        if choice:
            reasons.append(f"{axis_name}_selected_score_missing")
        return summary, reasons
    target_count = safe_int(selected.get("target_count"))
    line_count = safe_int(selected.get("line_count"))
    labeled_count = safe_int(selected.get("labeled_count"))
    ocr_bound_count = safe_int(selected.get("ocr_bound_count"))
    duplicate_ocr_count = safe_int(selected.get("duplicate_ocr_count"))
    score = safe_float(selected.get("score"))
    invalid_reasons = selected.get("invalid_reasons", [])
    if not isinstance(invalid_reasons, list):
        invalid_reasons = []
    summary.update(
        {
            "score": selected.get("score"),
            "valid": bool(selected.get("valid")),
            "invalid_reasons": invalid_reasons,
            "line_count": line_count,
            "labeled_count": labeled_count,
            "ocr_bound_count": ocr_bound_count,
            "strong_count": selected.get("strong_count"),
            "mllm_only_count": selected.get("mllm_only_count"),
            "unique_ocr_count": selected.get("unique_ocr_count"),
            "duplicate_ocr_count": selected.get("duplicate_ocr_count"),
            "target_count": target_count,
            "mean_ocr_distance": selected.get("mean_ocr_distance"),
            "max_ocr_distance": selected.get("max_ocr_distance"),
        }
    )
    if not selected.get("valid"):
        reasons.append(f"{axis_name}_selected_candidate_invalid")
    if invalid_reasons:
        reasons.append(f"{axis_name}_selected_candidate_has_invalid_reasons")

    if target_count >= 5:
        min_lines = max(2, int(np.floor(target_count * 0.35)))
        min_labels = max(2, int(np.floor(target_count * 0.35)))
        min_ocr_bound = max(1, int(np.floor(target_count * 0.25)))
        if score < 5.0 and (
            line_count <= min_lines
            or labeled_count <= min_labels
            or ocr_bound_count <= min_ocr_bound
        ):
            reasons.append(f"{axis_name}_selected_axis_severely_undercovered")

    axis_type = str(summary.get("axis_type", "") or "").casefold()
    mllm_only_count = safe_int(selected.get("mllm_only_count"))
    if (
        choice == "semantic_guide"
        and axis_type in {"category", "time", "mixed"}
        and target_count >= 12
        and ocr_bound_count / max(1, target_count) < 0.7
        and mllm_only_count >= 5
    ):
        summary["ocr_bound_ratio"] = round(ocr_bound_count / max(1, target_count), 3)
        reasons.append(f"{axis_name}_semantic_only_low_ocr_support")

    unique_ocr_count = safe_int(selected.get("unique_ocr_count"))
    if (
        axis_type in {"numeric", "time"}
        and target_count >= 12
        and labeled_count / max(1, target_count) < 0.65
        and unique_ocr_count / max(1, target_count) < 0.55
    ):
        summary["labeled_ratio"] = round(labeled_count / max(1, target_count), 3)
        summary["unique_ocr_ratio"] = round(unique_ocr_count / max(1, target_count), 3)
        reasons.append(f"{axis_name}_dense_axis_low_unique_ocr_support")

    if (
        choice == "semantic_guide"
        and axis_type == "numeric"
        and target_count >= 4
        and ocr_bound_count / max(1, target_count) < 0.25
        and mllm_only_count >= max(3, int(np.ceil(target_count * 0.7)))
    ):
        summary["ocr_bound_ratio"] = round(ocr_bound_count / max(1, target_count), 3)
        reasons.append(f"{axis_name}_numeric_semantic_only_low_ocr_support")

    if (
        choice == "semantic_guide"
        and axis_type == "time"
        and target_count >= 4
        and (
            line_count > target_count + 2
            or (
                ocr_bound_count / max(1, target_count) < 0.65
                and mllm_only_count >= max(2, int(np.ceil(target_count * 0.35)))
            )
        )
    ):
        max_ocr_distance = safe_float(selected.get("max_ocr_distance"))
        mean_ocr_distance = safe_float(selected.get("mean_ocr_distance"))
        if max_ocr_distance > 6.0 or mean_ocr_distance > 2.5 or ocr_bound_count < max(2, int(np.ceil(target_count * 0.4))):
            summary["ocr_bound_ratio"] = round(ocr_bound_count / max(1, target_count), 3)
            reasons.append(f"{axis_name}_time_semantic_low_ocr_or_extra_lines")

    if (
        axis_type == "numeric"
        and target_count >= 5
        and duplicate_ocr_count >= 1
        and unique_ocr_count < target_count
    ):
        reasons.append(f"{axis_name}_numeric_duplicate_ocr_tick_binding")

    if (
        axis_type == "numeric"
        and target_count >= 6
        and ocr_bound_count / max(1, target_count) < 0.25
        and mllm_only_count >= max(3, int(np.ceil(target_count * 0.65)))
    ):
        summary["ocr_bound_ratio"] = round(ocr_bound_count / max(1, target_count), 3)
        reasons.append(f"{axis_name}_numeric_selected_low_ocr_support")

    max_ocr_distance = safe_float(selected.get("max_ocr_distance"))
    if (
        axis_type == "numeric"
        and target_count >= 8
        and max_ocr_distance > 10.0
        and max_ocr_distance > safe_float(selected.get("mean_ocr_distance")) * 4.0
    ):
        reasons.append(f"{axis_name}_numeric_selected_large_ocr_distance")

    if (
        choice == "semantic_guide"
        and axis_type in {"category", "mixed"}
        and target_count >= 8
    ):
        tick = scores.get("tick_supplement") if isinstance(scores, dict) else None
        if isinstance(tick, dict) and tick.get("valid"):
            tick_line_count = safe_int(tick.get("line_count"))
            tick_labeled_count = safe_int(tick.get("labeled_count"))
            tick_ocr_bound_count = safe_int(tick.get("ocr_bound_count"))
            tick_score = safe_float(tick.get("score"))
            if (
                target_count < tick_line_count <= target_count + 3
                and tick_labeled_count >= target_count
                and tick_ocr_bound_count >= max(3, int(np.ceil(target_count * 0.65)))
                and score - tick_score <= 10.0
                and safe_float(selected.get("max_ocr_distance")) > 2.5
            ):
                summary["tick_supplement_score"] = tick.get("score")
                summary["tick_supplement_line_count"] = tick_line_count
                summary["tick_supplement_labeled_count"] = tick_labeled_count
                summary["tick_supplement_ocr_bound_count"] = tick_ocr_bound_count
                reasons.append(f"{axis_name}_category_semantic_tick_position_ambiguous")

    return summary, reasons

def priority_failure_quality(priority_decision: dict[str, object]) -> dict[str, object]:
    if not isinstance(priority_decision, dict) or not priority_decision.get("enabled"):
        return {"enabled": False, "failure_reasons": []}
    x_summary, x_reasons = priority_axis_quality(
        priority_decision,
        axis_name="x_axis",
        choice_key="x_axis_vertical_grid_choice",
        reason_key="x_axis_reason",
        scores_key="x_scores",
        axis_type_key="x_axis_type",
    )
    y_summary, y_reasons = priority_axis_quality(
        priority_decision,
        axis_name="y_axis",
        choice_key="y_axis_horizontal_grid_choice",
        reason_key="y_axis_reason",
        scores_key="y_scores",
        axis_type_key="y_axis_type",
    )
    reasons = x_reasons + y_reasons
    if priority_decision.get("mllm_fallback_reason"):
        x_choice_missing = not priority_decision.get("x_axis_vertical_grid_choice")
        y_choice_missing = not priority_decision.get("y_axis_horizontal_grid_choice")
        if x_choice_missing or y_choice_missing:
            reasons.append("priority_choice_needed_mllm_but_mllm_unavailable")
    return {
        "enabled": True,
        "failure_reasons": reasons,
        "x_axis": x_summary,
        "y_axis": y_summary,
    }

def normalized_failure_label(value: object) -> str:
    return " ".join(str(value if value is not None else "").strip().casefold().split())

def repeated_semantic_label_failure(
    axis_bindings: dict[str, object] | None,
    axis_quality: dict[str, object],
    axis_name: str,
) -> str | None:
    if not isinstance(axis_bindings, dict) or not isinstance(axis_quality, dict):
        return None
    if axis_quality.get("choice") != "semantic_guide":
        return None
    if safe_int(axis_quality.get("valid_candidate_count")) != 1:
        return None
    axis_type = str(axis_quality.get("axis_type", "") or "").casefold()
    if axis_type not in {"category", "time", "mixed"}:
        return None
    target_count = safe_int(axis_quality.get("target_count"))
    if target_count < 8:
        return None
    bindings = axis_bindings.get("tick_bindings", [])
    labels = [
        normalized_failure_label(item.get("label"))
        for item in bindings
        if isinstance(item, dict) and normalized_failure_label(item.get("label"))
    ] if isinstance(bindings, list) else []
    if len(labels) < 8:
        return None
    unique_count = len(set(labels))
    duplicate_count = len(labels) - unique_count
    unique_ratio = unique_count / len(labels) if labels else 1.0
    if duplicate_count >= max(3, int(np.ceil(len(labels) * 0.25))) and unique_ratio <= 0.75:
        axis_quality["label_unique_count"] = unique_count
        axis_quality["label_duplicate_count"] = duplicate_count
        axis_quality["label_unique_ratio"] = round(unique_ratio, 3)
        return f"{axis_name}_semantic_only_repeated_labels"
    return None

def grid_failure_report(
    grid_horizontal,
    grid_vertical,
    grid_label_bindings: dict[str, object],
    priority_decision: dict[str, object],
    mllm_result: dict[str, object],
) -> dict[str, object]:
    horizontal_count = grid_line_count(grid_horizontal, "horizontal")
    vertical_count = grid_line_count(grid_vertical, "vertical")
    x_labeled = labeled_tick_count(grid_label_bindings.get("x_axis") if isinstance(grid_label_bindings, dict) else None)
    y_labeled = labeled_tick_count(grid_label_bindings.get("y_axis") if isinstance(grid_label_bindings, dict) else None)
    reasons: list[str] = []
    if horizontal_count + vertical_count == 0:
        reasons.append("no_final_grid_lines")
    if max(x_labeled, y_labeled) < 2:
        reasons.append("no_axis_with_two_labeled_ticks")
    if not priority_decision.get("enabled") and mllm_result.get("error"):
        reasons.append("mllm_unavailable_for_priority_arbitration")
    priority_quality = priority_failure_quality(priority_decision)
    reasons.extend(priority_quality.get("failure_reasons", []))
    if isinstance(priority_quality, dict):
        x_repeat_reason = repeated_semantic_label_failure(
            grid_label_bindings.get("x_axis") if isinstance(grid_label_bindings, dict) else None,
            priority_quality.get("x_axis", {}) if isinstance(priority_quality.get("x_axis"), dict) else {},
            "x_axis",
        )
        y_repeat_reason = repeated_semantic_label_failure(
            grid_label_bindings.get("y_axis") if isinstance(grid_label_bindings, dict) else None,
            priority_quality.get("y_axis", {}) if isinstance(priority_quality.get("y_axis"), dict) else {},
            "y_axis",
        )
        for repeat_reason in (x_repeat_reason, y_repeat_reason):
            if repeat_reason:
                reasons.append(repeat_reason)
                failure_reasons = priority_quality.setdefault("failure_reasons", [])
                if isinstance(failure_reasons, list):
                    failure_reasons.append(repeat_reason)
    failed = bool(reasons)
    return {
        "failed": failed,
        "reason": "ok" if not failed else "+".join(reasons),
        "reasons": reasons,
        "policy": "exclude_from_eval_when_failed",
        "final_horizontal_count": horizontal_count,
        "final_vertical_count": vertical_count,
        "x_axis_labeled_tick_count": x_labeled,
        "y_axis_labeled_tick_count": y_labeled,
        "priority_quality": priority_quality,
    }

def process_image(path: Path, root: Path, output_root: Path, args: argparse.Namespace) -> Path:
    image = read_image(path)
    candidate, horizontal, vertical = make_line_masks(
        image,
        sat_max=args.sat_max,
        white_cutoff=args.white_cutoff,
        min_gray=args.min_gray,
        contrast_min=args.contrast_min,
        include_dark=args.include_dark,
        dark_cutoff=args.dark_cutoff,
        min_line_frac=args.min_line_frac,
        gap_frac=args.gap_frac,
        max_thickness_frac=args.max_thickness_frac,
    )
    direct_grid, direct_horizontal, direct_vertical = reconstruct_grid(
        horizontal,
        vertical,
        min_grid_span_frac=args.min_grid_span_frac,
        cluster_tolerance=args.cluster_tolerance,
        thickness=args.grid_thickness,
    )
    ocr_items, ocr_error = run_paddle_ocr(image, args)
    initial_ocr_axis_evidence = build_ocr_axis_evidence(ocr_items, image.shape[:2])
    initial_tick_grid, initial_tick_horizontal, initial_tick_vertical = reconstruct_grid_from_ticks(
        image,
        dark_cutoff=args.tick_dark_cutoff,
        thickness=args.grid_thickness,
        ocr_items=ocr_items,
    )
    initial_grid_geometry_evidence = build_grid_geometry_evidence(
        direct_horizontal,
        direct_vertical,
        initial_tick_horizontal,
        initial_tick_vertical,
    )
    cached_mllm_path: Path | None = None
    mllm_source = "disabled"
    mllm_result: dict[str, object] | None = None
    if args.mllm_cache_root:
        mllm_result, cached_mllm_path = load_cached_mllm_with_source(path, args.mllm_cache_root)
        if mllm_result is not None:
            mllm_source = "cache"
    if mllm_result is None:
        mllm_result = run_mllm_axis_extraction(image, initial_ocr_axis_evidence, initial_grid_geometry_evidence, args)
        mllm_source = "api_call" if args.mllm else "disabled"
    split_ocr_items, mllm_ocr_split_events = split_merged_ocr_items_with_mllm(ocr_items, mllm_result, image.shape[:2])
    split_ocr_items, numeric_gap_split_events = split_numeric_gap_ocr_items(split_ocr_items, image.shape[:2])
    split_ocr_items, split_merge_events = merge_split_ocr_items_with_mllm(split_ocr_items, mllm_result, image.shape[:2])
    ocr_split_events = mllm_ocr_split_events + numeric_gap_split_events
    bbox_refined_ocr_items, bbox_refine_events = refine_mllm_split_boxes_by_projection(image, split_ocr_items, ocr_split_events)
    bbox_refined_ocr_items, bbox_regularize_events = regularize_mllm_split_sequence_geometry(
        bbox_refined_ocr_items,
        mllm_result,
        image.shape[:2],
    )
    role_refined_ocr_items = refine_ocr_roles_with_mllm(bbox_refined_ocr_items, mllm_result, image.shape[:2])
    role_refined_ocr_items = restore_numeric_gap_split_roles(role_refined_ocr_items)
    pseudo_ocr_items, pseudo_events = add_mllm_missing_label_boxes(role_refined_ocr_items, mllm_result, image.shape[:2])
    refined_ocr_items, canonical_events = canonicalize_items_with_mllm_text(pseudo_ocr_items, mllm_result, image.shape[:2])
    refined_ocr_items, canonical_geometry_events = regularize_canonical_numeric_axis_geometry(
        refined_ocr_items,
        mllm_result,
        image.shape[:2],
    )
    ocr_axis_evidence = build_ocr_axis_evidence(refined_ocr_items, image.shape[:2])
    if mllm_ocr_split_events:
        ocr_axis_evidence["mllm_guided_ocr_splits"] = mllm_ocr_split_events
    if numeric_gap_split_events:
        ocr_axis_evidence["ocr_numeric_gap_splits"] = numeric_gap_split_events
    if split_merge_events:
        ocr_axis_evidence["split_ocr_merges"] = split_merge_events
    if bbox_refine_events:
        ocr_axis_evidence["bbox_refinements"] = bbox_refine_events
    if bbox_regularize_events:
        ocr_axis_evidence["bbox_regularizations"] = bbox_regularize_events
    if canonical_geometry_events:
        ocr_axis_evidence["canonical_geometry_regularizations"] = canonical_geometry_events
    if pseudo_events:
        ocr_axis_evidence["mllm_guided_pseudo_boxes"] = pseudo_events
    if canonical_events:
        ocr_axis_evidence["mllm_canonical_text_corrections"] = canonical_events
    clean_horizontal, clean_vertical, line_mask_text_suppression = suppress_line_mask_text_regions(
        horizontal,
        vertical,
        refined_ocr_items,
    )
    direct_grid, direct_horizontal, direct_vertical = reconstruct_grid(
        clean_horizontal,
        clean_vertical,
        min_grid_span_frac=args.min_grid_span_frac,
        cluster_tolerance=args.cluster_tolerance,
        thickness=args.grid_thickness,
    )
    direct_horizontal, direct_vertical, direct_text_suppression = suppress_direct_grid_text_regions(
        direct_horizontal,
        direct_vertical,
        refined_ocr_items,
        thickness=args.grid_thickness,
    )
    direct_grid = cv2.bitwise_or(direct_horizontal, direct_vertical)
    tick_grid, tick_horizontal, tick_vertical = reconstruct_grid_from_ticks(
        image,
        dark_cutoff=args.tick_dark_cutoff,
        thickness=args.grid_thickness,
        ocr_items=refined_ocr_items,
    )
    tick_grid, tick_horizontal, tick_vertical, tick_ocr_alignment = build_ocr_aligned_tick_grid(
        tick_horizontal,
        tick_vertical,
        ocr_axis_evidence,
        mllm_result,
        thickness=args.grid_thickness,
    )
    grid_geometry_evidence = build_grid_geometry_evidence(
        direct_horizontal,
        direct_vertical,
        tick_horizontal,
        tick_vertical,
    )
    grid_geometry_evidence["direct_text_suppression"] = direct_text_suppression
    grid_geometry_evidence["line_mask_text_suppression"] = line_mask_text_suppression
    grid_geometry_evidence["tick_ocr_alignment"] = tick_ocr_alignment
    grid_geometry_evidence["initial_tick_supplement"] = initial_grid_geometry_evidence.get("tick_supplement", {})
    role_refinements = [
        {
            "text": str(item.get("text", "")),
            "raw_role": item.get("raw_role"),
            "role": item.get("role"),
            "source": item.get("role_source"),
            "confidence": item.get("role_confidence"),
            "reason": item.get("role_reason"),
            "center": item.get("center"),
        }
        for item in refined_ocr_items
        if item.get("raw_role") and item.get("raw_role") != item.get("role")
    ]
    if role_refinements:
        ocr_axis_evidence["role_refinements"] = role_refinements
    fused_axis_evidence = build_fused_axis_evidence(ocr_axis_evidence, mllm_result)
    grid, grid_horizontal, grid_vertical, grid_meta = merge_grid_by_hierarchy(
        direct_horizontal,
        direct_vertical,
        tick_horizontal,
        tick_vertical,
        min_lines=args.min_grid_lines,
        ocr_axis_evidence=ocr_axis_evidence,
        mllm_result=mllm_result,
        semantic_guard=not args.no_semantic_guard,
    )
    grid_meta["direct_text_suppression"] = direct_text_suppression
    grid_meta["line_mask_text_suppression"] = line_mask_text_suppression
    semantic_supplement_grid, semantic_supplement_horizontal, semantic_supplement_vertical, semantic_supplement_meta = build_semantic_guide_grid(
        image.shape[:2],
        grid_horizontal,
        grid_vertical,
        refined_ocr_items,
        min_lines=args.min_grid_lines,
        thickness=args.grid_thickness,
        mllm_result=mllm_result,
    )
    if semantic_supplement_meta.get("horizontal_count", 0):
        horizontal_details = semantic_supplement_meta.get("horizontal", {})
        replace_with_semantic = False
        if isinstance(horizontal_details, dict):
            reason = str(horizontal_details.get("reason", ""))
            try:
                mllm_target = int(horizontal_details.get("mllm_target", 0) or 0)
            except (TypeError, ValueError):
                mllm_target = 0
            if "mllm_label_sequence" in reason:
                current_h_count = grid_meta.get("final_horizontal_count")
                if current_h_count is None:
                    current_h_count = grid_line_count(grid_horizontal, "horizontal")
                previous_source = str(grid_meta.get("horizontal_source", "none"))
                current_positions, _ = grid_positions_and_bounds(grid_horizontal, "horizontal")
                semantic_positions, _ = grid_positions_and_bounds(semantic_supplement_horizontal, "horizontal")
                current_span = max(current_positions) - min(current_positions) if len(current_positions) >= 2 else 0.0
                semantic_span = max(semantic_positions) - min(semantic_positions) if len(semantic_positions) >= 2 else 0.0
                semantic_spans_more_plot = (
                    int(current_h_count) > mllm_target + 2
                    and semantic_span > max(current_span * 1.45, current_span + 40.0)
                )
                try:
                    matched_count = int(horizontal_details.get("matched_count", 0) or 0)
                except (TypeError, ValueError):
                    matched_count = 0
                y_axis_ocr = ocr_axis_evidence.get("y_axis", {}) if isinstance(ocr_axis_evidence, dict) else {}
                try:
                    ocr_axis_count = int(y_axis_ocr.get("count", 0) or 0) if isinstance(y_axis_ocr, dict) else 0
                except (TypeError, ValueError):
                    ocr_axis_count = 0
                semantic_major_subset = (
                    int(current_h_count) > mllm_target + 2
                    and matched_count >= max(3, int(mllm_target * 0.6))
                    and ocr_axis_count <= mllm_target + 2
                    and semantic_span >= current_span * 0.55
                )
                replace_with_semantic = bool(
                    mllm_target
                    and (
                        (int(current_h_count) < mllm_target and int(current_h_count) < max(3, mllm_target - 2))
                        or (previous_source == "tick" and int(current_h_count) != mllm_target)
                        or semantic_spans_more_plot
                        or semantic_major_subset
                    )
                )
        skip_sequence_overlay = (
            isinstance(horizontal_details, dict)
            and "mllm_label_sequence" in str(horizontal_details.get("reason", ""))
            and not replace_with_semantic
        )
        missing_only_details = {"enabled": False, "reason": "not_checked"}
        if (
            isinstance(horizontal_details, dict)
            and "mllm_label_sequence" in str(horizontal_details.get("reason", ""))
            and not replace_with_semantic
        ):
            try:
                target_for_missing = int(horizontal_details.get("mllm_target", 0) or 0)
            except (TypeError, ValueError):
                target_for_missing = 0
            grid_horizontal, missing_only_details = semantic_missing_only(
                grid_horizontal,
                semantic_supplement_horizontal,
                "horizontal",
                thickness=args.grid_thickness,
                target_count=target_for_missing,
            )
        else:
            grid_horizontal = (
                semantic_supplement_horizontal
                if replace_with_semantic
                else (grid_horizontal if skip_sequence_overlay else cv2.bitwise_or(grid_horizontal, semantic_supplement_horizontal))
            )
        previous = str(grid_meta.get("horizontal_source", "none"))
        if not skip_sequence_overlay:
            grid_meta["horizontal_source"] = (
                "semantic_supplement" if previous == "none" else f"{previous}+semantic_supplement"
            )
        if replace_with_semantic:
            grid_meta["horizontal_source"] = "semantic_supplement"
            grid_meta["horizontal_semantic_replacement"] = {
                "reason": "complete_mllm_sequence_replaces_undercomplete_direct",
                "previous_source": previous,
            }
        elif skip_sequence_overlay:
            grid_meta["horizontal_semantic_skipped"] = {
                "reason": "complete_mllm_sequence_not_stacked_on_existing_direct",
                "previous_source": previous,
            }
        if missing_only_details.get("enabled") or missing_only_details.get("reason") not in {"not_checked", "not_applicable"}:
            grid_meta["horizontal_semantic_missing_only"] = missing_only_details
    if semantic_supplement_meta.get("vertical_count", 0):
        vertical_details = semantic_supplement_meta.get("vertical", {})
        replace_with_semantic = False
        if isinstance(vertical_details, dict):
            reason = str(vertical_details.get("reason", ""))
            try:
                mllm_target = int(vertical_details.get("mllm_target", 0) or 0)
            except (TypeError, ValueError):
                mllm_target = 0
            if "mllm_label_sequence" in reason:
                current_v_count = grid_meta.get("final_vertical_count")
                if current_v_count is None:
                    current_v_count = grid_line_count(grid_vertical, "vertical")
                mllm_axis = mllm_result.get("x_axis", {}) if isinstance(mllm_result, dict) else {}
                mllm_axis_type = str(mllm_axis.get("type", "unknown")) if isinstance(mllm_axis, dict) else "unknown"
                label_axis_takeover = (
                    mllm_axis_type in {"category", "time", "mixed"}
                    and mllm_target
                    and int(current_v_count) < mllm_target
                )
                replace_with_semantic = bool(
                    mllm_target
                    and (
                        int(current_v_count) < max(3, mllm_target - 2)
                        or label_axis_takeover
                    )
                )
        missing_only_details = {"enabled": False, "reason": "not_checked"}
        if (
            isinstance(vertical_details, dict)
            and "mllm_label_sequence" in str(vertical_details.get("reason", ""))
            and not replace_with_semantic
        ):
            try:
                target_for_missing = int(vertical_details.get("mllm_target", 0) or 0)
            except (TypeError, ValueError):
                target_for_missing = 0
            grid_vertical, missing_only_details = semantic_missing_only(
                grid_vertical,
                semantic_supplement_vertical,
                "vertical",
                thickness=args.grid_thickness,
                target_count=target_for_missing,
            )
        else:
            grid_vertical = semantic_supplement_vertical if replace_with_semantic else cv2.bitwise_or(grid_vertical, semantic_supplement_vertical)
        previous = str(grid_meta.get("vertical_source", "none"))
        grid_meta["vertical_source"] = (
            "semantic_supplement" if previous == "none" else f"{previous}+semantic_supplement"
        )
        if replace_with_semantic:
            grid_meta["vertical_source"] = "semantic_supplement"
            grid_meta["vertical_semantic_replacement"] = {
                "reason": "complete_mllm_sequence_replaces_undercomplete_direct",
                "previous_source": previous,
            }
        if missing_only_details.get("enabled") or missing_only_details.get("reason") not in {"not_checked", "not_applicable"}:
            grid_meta["vertical_semantic_missing_only"] = missing_only_details
    grid = cv2.bitwise_or(grid_horizontal, grid_vertical)
    grid_meta["semantic_supplement"] = semantic_supplement_meta
    semantic_guide_reference_horizontal = cv2.bitwise_or(direct_horizontal, cv2.bitwise_or(tick_horizontal, semantic_supplement_horizontal))
    semantic_guide_reference_vertical = cv2.bitwise_or(direct_vertical, cv2.bitwise_or(tick_vertical, semantic_supplement_vertical))
    semantic_guide_grid, semantic_guide_horizontal, semantic_guide_vertical, semantic_guide_meta = build_semantic_guide_candidate_grid(
        image.shape[:2],
        semantic_guide_reference_horizontal,
        semantic_guide_reference_vertical,
        ocr_axis_evidence,
        thickness=args.grid_thickness,
    )
    grid_meta["semantic_guide"] = semantic_guide_meta
    native_bound_grid, native_bound_horizontal, native_bound_vertical, native_bound_meta = build_ocr_bound_native_grid(
        direct_horizontal,
        direct_vertical,
        ocr_axis_evidence,
        mllm_result,
        thickness=args.grid_thickness,
    )
    grid_meta["priority1_native_grid"] = native_bound_meta
    priority_grid_candidates = {
        "combined_mask": (native_bound_horizontal, native_bound_vertical),
        "tick_supplement": (tick_horizontal, tick_vertical),
        "semantic_guide": (semantic_guide_horizontal, semantic_guide_vertical),
    }
    priority_candidate_outputs = build_priority_candidate_outputs(
        priority_grid_candidates,
        ocr_axis_evidence,
        mllm_result,
        fused_axis_evidence,
    )
    grid_meta["priority_candidate_outputs"] = {
        name: {
            "source": payload.get("source"),
            "title": payload.get("title"),
            "horizontal_count": payload.get("horizontal_count"),
            "vertical_count": payload.get("vertical_count"),
        }
        for name, payload in priority_candidate_outputs.items()
    }
    priority_decision: dict[str, object] = {"enabled": False, "reason": "not_run"}
    priority_review_image = None
    if (
        not args.no_grid_arbitration
        and isinstance(mllm_result, dict)
        and mllm_result.get("enabled")
        and mllm_result.get("error") is None
    ):
        grid_horizontal, grid_vertical, priority_decision, priority_review_image = arbitrate_priority_grids(
            image,
            priority_grid_candidates,
            ocr_axis_evidence,
            mllm_result,
            fused_axis_evidence,
            grid_horizontal,
            grid_vertical,
            args,
        )
        grid = cv2.bitwise_or(grid_horizontal, grid_vertical)
        grid_meta["priority_arbitration"] = priority_decision
        grid_meta["pre_priority_horizontal_source"] = grid_meta.get("horizontal_source", "none")
        grid_meta["pre_priority_vertical_source"] = grid_meta.get("vertical_source", "none")
        grid_meta["horizontal_source"] = priority_decision.get("y_axis_horizontal_grid_choice", "none")
        grid_meta["vertical_source"] = priority_decision.get("x_axis_vertical_grid_choice", "none")
        grid_meta["final_horizontal_count"] = grid_line_count(grid_horizontal, "horizontal")
        grid_meta["final_vertical_count"] = grid_line_count(grid_vertical, "vertical")
    grid_label_bindings = build_grid_label_bindings(
        grid_horizontal,
        grid_vertical,
        ocr_axis_evidence,
        mllm_result,
        fused_axis_evidence,
    )
    rel = safe_relative(path, root)
    out_base = output_root / rel.with_suffix("")
    write_image(out_base.with_name(out_base.name + "_candidate.png"), mask_panel(candidate))
    write_image(out_base.with_name(out_base.name + "_horizontal.png"), mask_panel(clean_horizontal))
    write_image(out_base.with_name(out_base.name + "_vertical.png"), mask_panel(clean_vertical))
    write_image(out_base.with_name(out_base.name + "_overlay.png"), overlay_masks(image, clean_horizontal, clean_vertical))
    write_image(out_base.with_name(out_base.name + "_direct_grid.png"), mask_panel(direct_grid))
    write_image(out_base.with_name(out_base.name + "_direct_grid_horizontal.png"), mask_panel(direct_horizontal))
    write_image(out_base.with_name(out_base.name + "_direct_grid_vertical.png"), mask_panel(direct_vertical))
    write_image(out_base.with_name(out_base.name + "_tick_grid.png"), mask_panel(tick_grid))
    write_image(out_base.with_name(out_base.name + "_tick_grid_horizontal.png"), mask_panel(tick_horizontal))
    write_image(out_base.with_name(out_base.name + "_tick_grid_vertical.png"), mask_panel(tick_vertical))
    write_image(out_base.with_name(out_base.name + "_semantic_guide_grid.png"), mask_panel(semantic_guide_grid))
    write_image(
        out_base.with_name(out_base.name + "_semantic_guide_grid_horizontal.png"),
        mask_panel(semantic_guide_horizontal),
    )
    write_image(
        out_base.with_name(out_base.name + "_semantic_guide_grid_vertical.png"),
        mask_panel(semantic_guide_vertical),
    )
    write_image(out_base.with_name(out_base.name + "_grid.png"), mask_panel(grid))
    write_image(out_base.with_name(out_base.name + "_grid_horizontal.png"), mask_panel(grid_horizontal))
    write_image(out_base.with_name(out_base.name + "_grid_vertical.png"), mask_panel(grid_vertical))
    write_image(out_base.with_name(out_base.name + "_grid_overlay.png"), overlay_grid(image, grid))
    write_image(
        out_base.with_name(out_base.name + "_grid_label_overlay.png"),
        draw_grid_label_overlay(image, grid_horizontal, grid_vertical, grid_label_bindings),
    )
    write_priority_candidate_outputs(
        out_base,
        image,
        priority_grid_candidates,
        priority_candidate_outputs,
    )
    grid, grid_horizontal, grid_vertical, grid_label_bindings, final_selection = write_final_selection_outputs(
        out_base,
        image,
        priority_grid_candidates,
        ocr_axis_evidence,
        mllm_result,
        fused_axis_evidence,
        priority_decision,
    )
    grid_meta["final_selection"] = final_selection
    grid_meta["final_horizontal_count"] = grid_line_count(grid_horizontal, "horizontal")
    grid_meta["final_vertical_count"] = grid_line_count(grid_vertical, "vertical")
    failure_report = grid_failure_report(
        grid_horizontal,
        grid_vertical,
        grid_label_bindings,
        priority_decision,
        mllm_result,
    )
    grid_meta["failure_report"] = failure_report
    final_selection["failure_report"] = failure_report
    final_output_name = str(final_selection.get("name", "final") or "final")
    write_json_file(
        out_base.with_name(out_base.name + f"_{final_output_name}_selection.json"),
        {**final_selection, "priority_decision": priority_decision},
    )
    write_image(out_base.with_name(out_base.name + "_grid.png"), mask_panel(grid))
    write_image(out_base.with_name(out_base.name + "_grid_horizontal.png"), mask_panel(grid_horizontal))
    write_image(out_base.with_name(out_base.name + "_grid_vertical.png"), mask_panel(grid_vertical))
    write_image(out_base.with_name(out_base.name + "_grid_overlay.png"), overlay_grid(image, grid))
    write_image(
        out_base.with_name(out_base.name + "_grid_label_overlay.png"),
        draw_grid_label_overlay(image, grid_horizontal, grid_vertical, grid_label_bindings),
    )
    write_image(out_base.with_name(out_base.name + "_ocr_overlay.png"), draw_ocr_overlay(image, refined_ocr_items, ocr_error))
    ocr_label_lab_metrics = {
        "ocr_error": ocr_error,
        "raw_ocr_count": len(ocr_items),
        "split_item_count": len(split_ocr_items),
        "bbox_refined_item_count": sum(1 for item in bbox_refined_ocr_items if item.get("bbox_refined")),
        "bbox_regularized_item_count": sum(1 for item in bbox_refined_ocr_items if item.get("bbox_regularized")),
        "role_refined_count": sum(
            1
            for item in role_refined_ocr_items
            if item.get("raw_role") and item.get("raw_role") != item.get("role")
        ),
        "split_event_count": len(ocr_split_events),
        "mllm_split_event_count": len(mllm_ocr_split_events),
        "numeric_gap_split_event_count": len(numeric_gap_split_events),
        "split_ocr_merge_event_count": len(split_merge_events),
        "bbox_refine_event_count": len(bbox_refine_events),
        "canonical_geometry_regularize_event_count": len(canonical_geometry_events),
        "pseudo_event_count": len(pseudo_events),
        "pseudo_added_count": sum(1 for event in pseudo_events if event.get("status") == "added"),
        "canonical_text_correction_count": len(canonical_events),
        "other_item_count": sum(1 for item in refined_ocr_items if item.get("role") == "other"),
    }
    write_image(
        out_base.with_name(out_base.name + "_ocr_label_lab_preview.png"),
        make_ocr_label_lab_preview(
            image,
            ocr_items,
            split_ocr_items,
            bbox_refined_ocr_items,
            role_refined_ocr_items,
            pseudo_ocr_items,
            refined_ocr_items,
            mllm_result,
            ocr_label_lab_metrics,
            args.panel_width,
        ),
    )
    write_image(out_base.with_name(out_base.name + "_mllm_input_image.png"), image)
    write_image(
        out_base.with_name(out_base.name + "_ocr_summary.png"),
        make_ocr_summary_panel(image.shape[:2], refined_ocr_items, ocr_error),
    )
    write_json_file(
        out_base.with_name(out_base.name + "_ocr_axis.json"),
        {
            "error": ocr_error,
            "items": refined_ocr_items,
            "initial_items": ocr_items,
            "split_items": split_ocr_items,
            "bbox_refined_items": bbox_refined_ocr_items,
            "role_refined_items": role_refined_ocr_items,
            "pseudo_items": pseudo_ocr_items,
            "mllm_guided_splits": mllm_ocr_split_events,
            "numeric_gap_splits": numeric_gap_split_events,
            "split_ocr_merges": split_merge_events,
            "bbox_refinements": bbox_refine_events,
            "bbox_regularizations": bbox_regularize_events,
            "canonical_geometry_regularizations": canonical_geometry_events,
            "pseudo_events": pseudo_events,
            "canonical_events": canonical_events,
        },
    )
    write_json_file(out_base.with_name(out_base.name + "_ocr_axis_evidence.json"), ocr_axis_evidence)
    write_json_file(out_base.with_name(out_base.name + "_grid_geometry_evidence.json"), grid_geometry_evidence)
    mllm_input_trace = {
        "model": args.mllm_model,
        "endpoint": args.mllm_endpoint,
        "source": mllm_source,
        "cache_path": str(cached_mllm_path) if cached_mllm_path else None,
        "image_path": str(path),
        "image_size": {"height": image.shape[0], "width": image.shape[1]},
        "prompt": mllm_prompt(),
        "ocr_content_included": False,
        "request_payload_shape": {
            "chat_completions": bool(args.mllm_endpoint and "/chat/completions" in args.mllm_endpoint),
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "image": "saved separately as *_mllm_input_image.png",
        },
    }
    write_json_file(out_base.with_name(out_base.name + "_mllm_input.json"), mllm_input_trace)
    write_json_file(out_base.with_name(out_base.name + "_mllm_axis.json"), mllm_result)
    write_json_file(out_base.with_name(out_base.name + "_mllm_output.json"), mllm_result)
    write_json_file(out_base.with_name(out_base.name + "_mllm_io.json"), {"input": mllm_input_trace, "output": mllm_result})
    write_json_file(out_base.with_name(out_base.name + "_axis_fusion.json"), fused_axis_evidence)
    write_json_file(out_base.with_name(out_base.name + "_grid_label_bindings.json"), grid_label_bindings)
    write_json_file(out_base.with_name(out_base.name + "_grid_layers.json"), grid_meta)
    if priority_review_image is not None:
        write_image(out_base.with_name(out_base.name + "_grid_priority_review.png"), priority_review_image)
    write_json_file(out_base.with_name(out_base.name + "_grid_priority_decision.json"), priority_decision)
    write_json_file(out_base.with_name(out_base.name + "_grid_failure.json"), failure_report)

    preview = make_preview(
        image,
        candidate,
        clean_horizontal,
        clean_vertical,
        direct_grid,
        tick_grid,
        semantic_guide_grid,
        grid,
        grid_horizontal,
        grid_vertical,
        grid_meta,
        refined_ocr_items,
        ocr_error,
        ocr_axis_evidence,
        mllm_result,
        fused_axis_evidence,
        grid_label_bindings,
        args.panel_width,
    )
    preview_path = out_base.with_name(out_base.name + "_preview.png")
    write_image(preview_path, preview)
    return preview_path

def parse_args() -> argparse.Namespace:
    load_local_env_files()
    parser = argparse.ArgumentParser(
        description="Extract horizontal and vertical grid-line previews from chart images."
    )
    parser.add_argument("--input", type=Path, default=Path("."), help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=Path("grid_filter_preview"), help="Output directory.")
    parser.add_argument("--sample", type=int, default=None, help="Evenly sample N images from the input set.")
    parser.add_argument("--sat-max", type=int, default=70, help="Maximum HSV saturation for gray/black candidates.")
    parser.add_argument("--white-cutoff", type=int, default=255, help="Maximum gray value kept as a line candidate.")
    parser.add_argument("--min-gray", type=int, default=95, help="Minimum gray value kept; raises this to reject text/axes.")
    parser.add_argument("--contrast-min", type=int, default=7, help="Minimum local contrast against the nearby background.")
    parser.add_argument("--include-dark", action="store_true", help="Also keep dark axes/text candidates.")
    parser.add_argument("--dark-cutoff", type=int, default=80, help="Dark pixel cutoff used only with --include-dark.")
    parser.add_argument("--min-line-frac", type=float, default=0.055, help="Minimum line length as fraction of image size.")
    parser.add_argument("--gap-frac", type=float, default=0.006, help="Gap-closing kernel size as fraction of image size.")
    parser.add_argument("--max-thickness-frac", type=float, default=0.008, help="Maximum line thickness as fraction of image size.")
    parser.add_argument("--min-grid-span-frac", type=float, default=0.18, help="Minimum reconstructed grid-line span.")
    parser.add_argument("--min-grid-lines", type=int, default=2, help="Minimum lines needed before a layer is trusted.")
    parser.add_argument("--cluster-tolerance", type=int, default=3, help="Pixel tolerance for merging line fragments.")
    parser.add_argument("--grid-thickness", type=int, default=1, help="Thickness of reconstructed grid lines.")
    parser.add_argument("--tick-dark-cutoff", type=int, default=150, help="Maximum gray value for axis/tick detection.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable optional PaddleOCR axis text recognition.")
    parser.add_argument("--ocr-lang", default="en", help="PaddleOCR language code.")
    parser.add_argument("--ocr-min-score", type=float, default=0.45, help="Minimum OCR confidence kept in preview.")
    parser.add_argument("--ocr-det-thresh", type=float, default=0.35, help="PaddleOCR text pixel confidence threshold.")
    parser.add_argument("--ocr-det-box-thresh", type=float, default=0.60, help="PaddleOCR text box confidence threshold.")
    parser.add_argument(
        "--ocr-det-unclip-ratio",
        type=float,
        default=1.15,
        help="PaddleOCR detection box expansion ratio; lower values reduce merged long text boxes.",
    )
    parser.add_argument("--ocr-det-limit-side-len", type=int, default=960, help="PaddleOCR detection input side limit.")
    parser.add_argument("--ocr-det-limit-type", default="max", choices=["min", "max"], help="PaddleOCR side limit type.")
    parser.add_argument(
        "--ocr-return-word-box",
        action="store_true",
        help="Ask PaddleOCR to return word/character boxes when supported.",
    )
    parser.add_argument("--mllm", action="store_true", help="Enable optional MLLM axis/grid semantic extraction.")
    parser.add_argument(
        "--mllm-model",
        default=os.environ.get("MLLM_MODEL", "gemini-3.1-flash-lite"),
        help="MLLM model name.",
    )
    parser.add_argument(
        "--mllm-endpoint",
        default=os.environ.get("MLLM_ENDPOINT", DEFAULT_MLLM_ENDPOINT),
        help="Gemini generateContent endpoint or OpenAI-compatible /chat/completions endpoint.",
    )
    parser.add_argument(
        "--mllm-api-key-env",
        default="MLLM_API_KEY",
        help="Environment variable containing the MLLM API key.",
    )
    parser.add_argument(
        "--mllm-cache-root",
        type=Path,
        default=Path("grid_reconstruct_mllm"),
        help="Read cached *_mllm_axis.json before calling MLLM. Use a non-existing path such as none to disable cache.",
    )
    parser.add_argument("--mllm-timeout", type=float, default=30.0, help="MLLM request timeout in seconds.")
    parser.add_argument(
        "--no-semantic-guard",
        action="store_true",
        help="Disable OCR/MLLM semantic gating for tick-based grid completion.",
    )
    parser.add_argument(
        "--no-grid-arbitration",
        action="store_true",
        help="Disable MLLM-assisted priority-grid arbitration.",
    )
    parser.add_argument("--panel-width", type=int, default=360, help="Preview panel width.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output.resolve()

    if input_path.is_file():
        paths = [input_path]
        root = input_path.parent
    else:
        root = input_path
        paths = discover_images(root, output_root)

    paths = evenly_sample(paths, args.sample)
    if not paths:
        raise SystemExit("No images found.")

    print(f"Processing {len(paths)} image(s).")
    for index, path in enumerate(paths, start=1):
        preview_path = process_image(path, root, output_root, args)
        print(f"[{index}/{len(paths)}] {path} -> {preview_path}")


if __name__ == "__main__":
    main()
