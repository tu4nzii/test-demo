from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

import cv2
import numpy as np

from grid_math import parse_numeric_label
from grid_geometry import grid_positions_and_bounds
from grid_ocr import normalize_label_text

def numeric_parse_text(text: str) -> str:
    value = str(text or "")
    value = value.replace("'", "").replace("’", "").replace("`", "")
    value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)
    return value

def normalize_mllm_axis_label(axis: dict[str, object]) -> dict[str, object]:
    raw = axis.get("axis_label", "")
    text = ""
    confidence = 0.0
    if isinstance(raw, dict):
        text = str(raw.get("text", "") or "").strip()
        try:
            confidence = float(raw.get("confidence", axis.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
    elif isinstance(raw, str):
        text = raw.strip()
        try:
            confidence = float(axis.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
    if text.lower() in {"", "none", "null", "unknown", "n/a"}:
        text = ""
    return {"text": text, "confidence": round(max(0.0, min(1.0, confidence)), 3), "source": "mllm"}

def fuse_axis_label(ocr_axis: dict[str, object], mllm_axis: dict[str, object]) -> dict[str, object]:
    ocr_label = ocr_axis.get("axis_label", {}) if isinstance(ocr_axis, dict) else {}
    if not isinstance(ocr_label, dict):
        ocr_label = {"text": str(ocr_label), "confidence": 0.0, "source": "ocr"}
    mllm_label = normalize_mllm_axis_label(mllm_axis if isinstance(mllm_axis, dict) else {})

    choices: list[dict[str, object]] = []
    for label in (ocr_label, mllm_label):
        text = str(label.get("text", "") or "").strip()
        try:
            confidence = float(label.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if text:
            choices.append({"text": text, "confidence": confidence, "source": str(label.get("source", "unknown"))})

    if not choices:
        return {
            "text": "",
            "confidence": 0.0,
            "source": "none",
            "ocr": ocr_label,
            "mllm": mllm_label,
        }

    if len(choices) == 2 and choices[0]["text"].casefold() == choices[1]["text"].casefold():
        return {
            "text": choices[0]["text"],
            "confidence": round(min(1.0, max(float(choices[0]["confidence"]), float(choices[1]["confidence"])) + 0.08), 3),
            "source": "ocr+mllm",
            "ocr": ocr_label,
            "mllm": mllm_label,
        }
    if mllm_label.get("text") and float(mllm_label.get("confidence", 0.0) or 0.0) >= 0.85:
        return {
            "text": mllm_label["text"],
            "confidence": round(float(mllm_label.get("confidence", 0.0) or 0.0), 3),
            "source": "mllm",
            "ocr": ocr_label,
            "mllm": mllm_label,
            "conflict": {
                "reason": "high_confidence_mllm_axis_label_preferred",
                "ocr_text": str(ocr_label.get("text", "") or ""),
            },
        }

    best = max(choices, key=lambda value: (float(value["confidence"]), len(str(value["text"]))))
    return {
        "text": best["text"],
        "confidence": round(float(best["confidence"]), 3),
        "source": best["source"],
        "ocr": ocr_label,
        "mllm": mllm_label,
    }

def build_fused_axis_evidence(
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
) -> dict[str, object]:
    fused: dict[str, object] = {
        "image_size": ocr_axis_evidence.get("image_size", {}),
        "mllm_enabled": bool(mllm_result.get("enabled", False)) if isinstance(mllm_result, dict) else False,
        "mllm_error": mllm_result.get("error") if isinstance(mllm_result, dict) else None,
    }
    for axis_key in ("x_axis", "y_axis"):
        ocr_axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
        mllm_axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
        if not isinstance(ocr_axis, dict):
            ocr_axis = {}
        if not isinstance(mllm_axis, dict):
            mllm_axis = {}
        fused[axis_key] = {
            "type": mllm_axis.get("type") or ocr_axis.get("type", "unknown"),
            "ocr_type": ocr_axis.get("type", "unknown"),
            "mllm_type": mllm_axis.get("type", "unknown"),
            "axis_label": fuse_axis_label(ocr_axis, mllm_axis),
            "ocr_tick_count": ocr_axis.get("count", 0),
            "mllm_tick_labels": mllm_axis.get("tick_labels", []),
        }
    return fused

def normalize_tick_texts(values: object) -> list[dict[str, object]]:
    if not isinstance(values, list):
        return []
    ticks: list[dict[str, object]] = []
    for index, value in enumerate(values):
        text = ""
        confidence = 0.0
        if isinstance(value, dict):
            text = str(value.get("text", "") or "").strip()
            try:
                confidence = float(value.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
        else:
            text = str(value).strip()
            confidence = 0.75
        if not text:
            continue
        ticks.append(
            {
                "text": text,
                "numeric": parse_numeric_label(numeric_parse_text(text)),
                "confidence": round(max(0.0, min(1.0, confidence)), 3),
                "index": index,
            }
        )
    return ticks

def tick_decimal_places(text: str) -> int:
    match = re.search(r"[-+]?\d+(?:\.(\d+))?", text)
    if not match or match.group(1) is None:
        return 0
    return len(match.group(1).rstrip("0"))

def format_tick_numeric(value: float, references: list[dict[str, object]]) -> str:
    decimals = 0
    for item in references:
        decimals = max(decimals, tick_decimal_places(str(item.get("text", ""))))
    decimals = min(decimals, 6)
    if decimals == 0 and abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.{decimals}f}" if decimals else f"{value:.6g}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text

def numeric_like_tick_text(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    value = value.replace("'", "").replace("’", "").replace("`", "")
    # Unit suffixes such as kg/gr/% are fine; month/category words are not.
    return re.fullmatch(r"[$€£¥]?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*(?:%|k|m|b|kg|g|gr|ms|s)?", value, re.IGNORECASE) is not None

def infer_numeric_tick_step(ticks: list[dict[str, object]]) -> float | None:
    if any(not numeric_like_tick_text(str(item.get("text", "") or "")) for item in ticks):
        return None
    values = [float(item["numeric"]) for item in ticks if item.get("numeric") is not None]
    if len(values) < 2 or len(values) != len(ticks):
        return None
    diffs = np.diff(np.array(values, dtype=np.float64))
    nonzero = [float(diff) for diff in diffs if abs(float(diff)) > 1e-9]
    if not nonzero:
        return None
    step = float(np.median(nonzero))
    if abs(step) <= 1e-9:
        return None
    residual = np.array([abs(diff - step) for diff in nonzero], dtype=np.float64)
    tolerance = max(1e-6, abs(step) * 0.08)
    if float(np.median(residual)) > tolerance:
        return None
    if float(np.max(residual)) > max(1e-6, abs(step) * 0.35):
        return None
    return step

def expand_numeric_mllm_ticks(
    ordered_ticks: list[dict[str, object]],
    target_count: int,
    ocr_matches: list[dict[str, object] | None],
) -> list[dict[str, object]]:
    if target_count <= 0 or not ordered_ticks:
        return ordered_ticks
    step = infer_numeric_tick_step(ordered_ticks)
    if step is None:
        return ordered_ticks

    def add_tick(value: float, index: int, confidence: float) -> dict[str, object]:
        return {
            "text": format_tick_numeric(value, ordered_ticks),
            "numeric": value,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "index": index,
            "extrapolated": True,
        }

    base_confidence = min(float(item.get("confidence", 0.0) or 0.0) for item in ordered_ticks)
    expanded: list[dict[str, object]] = []
    inserted_count = 0
    for index, item in enumerate(ordered_ticks):
        expanded.append(dict(item))
        if index >= len(ordered_ticks) - 1:
            continue
        current = item.get("numeric")
        next_value = ordered_ticks[index + 1].get("numeric")
        if current is None or next_value is None:
            continue
        ratio = (float(next_value) - float(current)) / step
        rounded = int(round(ratio))
        if rounded <= 1:
            continue
        if abs(ratio - rounded) > 0.15:
            continue
        for offset in range(1, rounded):
            if len(expanded) >= target_count:
                break
            inserted_count += 1
            value = float(current) + step * offset
            tick = add_tick(value, int(item.get("index", index)) + offset, base_confidence * 0.82)
            tick["interpolated"] = True
            expanded.append(tick)

    if inserted_count:
        ordered_ticks = expanded

    missing = target_count - len(ordered_ticks)
    if missing != 0:
        return ordered_ticks
    return ordered_ticks

def x_axis_plot_side_anchor(item: dict[str, object], image_height: float | None) -> float | None:
    box = item.get("box")
    center = item.get("center")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                return float(center[0])
            except (TypeError, ValueError):
                return None
        return None
    try:
        points = np.array(box, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
        return None
    use_top_edge = True
    if image_height is not None and isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            use_top_edge = float(center[1]) >= image_height * 0.5
        except (TypeError, ValueError):
            use_top_edge = True
    order = np.argsort(points[:, 1])
    edge = points[order[:2]] if use_top_edge else points[order[-2:]]
    if edge.shape[0] < 2:
        return None
    return float(np.mean(edge[:, 0]))

def x_axis_tick_anchor_positions(item: dict[str, object]) -> list[dict[str, object]]:
    positions: list[dict[str, object]] = []
    try:
        center_x = float(item.get("x"))
    except (TypeError, ValueError):
        center = item.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                center_x = float(center[0])
            except (TypeError, ValueError):
                center_x = None
        else:
            center_x = None
    if center_x is not None:
        positions.append({"position": center_x, "source": "center"})

    box = item.get("box")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            points = np.array(box, dtype=np.float32)
        except (TypeError, ValueError):
            points = np.empty((0, 2), dtype=np.float32)
        if points.ndim == 2 and points.shape[0] >= 4 and points.shape[1] >= 2:
            positions.extend(
                [
                    {"position": float(points[:, 0].min()), "source": "left_edge"},
                    {"position": float(points[:, 0].max()), "source": "right_edge"},
                ]
            )
    deduped: list[dict[str, object]] = []
    for item_position in positions:
        value = float(item_position["position"])
        if any(abs(float(existing["position"]) - value) <= 1.0 for existing in deduped):
            continue
        deduped.append(item_position)
    return deduped

def ocr_tick_candidates_for_axis(
    axis: dict[str, object],
    axis_key: str,
    image_size: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    ticks = axis.get("ticks", []) if isinstance(axis, dict) else []
    if not isinstance(ticks, list):
        return []
    axis_type = str(axis.get("type", "unknown"))
    axis_label = axis.get("axis_label", {})
    axis_label_text = ""
    if isinstance(axis_label, dict):
        axis_label_text = str(axis_label.get("text", "") or "").strip()
    numeric_count = int(axis.get("numeric_count", 0) or 0)
    prefer_numeric = axis_type in {"numeric", "time", "mixed"} or (axis_type != "category" and numeric_count >= 2)

    candidates: list[dict[str, object]] = []
    for item in ticks:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text or (axis_label_text and text.casefold() == axis_label_text.casefold()):
            continue
        numeric = item.get("numeric")
        parsed_numeric = parse_numeric_label(numeric_parse_text(text))
        if parsed_numeric is not None:
            numeric = parsed_numeric
        elif numeric is None:
            numeric = parse_numeric_label(numeric_parse_text(text))
        if prefer_numeric and numeric is None:
            continue
        if axis_type == "category" and numeric is not None and numeric_count < 2:
            continue
        position_key = "x" if axis_key == "x_axis" else "y"
        position_source = "center"
        if axis_key == "x_axis" and axis_type in {"category", "time", "mixed"}:
            image_height = None
            if isinstance(image_size, dict):
                try:
                    image_height = float(image_size.get("height"))
                except (TypeError, ValueError):
                    image_height = None
            anchor = x_axis_plot_side_anchor(item, image_height)
            if anchor is not None:
                position = anchor
                position_source = "plot_side_anchor"
            else:
                try:
                    position = float(item[position_key])
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            try:
                position = float(item[position_key])
            except (KeyError, TypeError, ValueError):
                continue
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        candidates.append(
            {
                "text": text,
                "numeric": numeric,
                "position": position,
                "confidence": round(max(0.0, min(1.0, score)), 3),
                "source": "ocr",
                "position_source": position_source,
                "anchor_positions": x_axis_tick_anchor_positions(item) if axis_key == "x_axis" else [],
                "x": item.get("x"),
                "y": item.get("y"),
                "center": item.get("center"),
                "box": item.get("box"),
                "role": item.get("role"),
                "raw_role": item.get("raw_role"),
                "role_source": item.get("role_source"),
                "role_reason": item.get("role_reason"),
                "text_source": item.get("text_source"),
                "label_kind": item.get("label_kind"),
                "canonical_text": item.get("canonical_text"),
                "canonical_axis": item.get("canonical_axis"),
                "canonical_index": item.get("canonical_index"),
                "canonical_match_source": item.get("canonical_match_source"),
                "mllm_pseudo_box": bool(item.get("mllm_pseudo_box")),
                "split_from_merged": bool(item.get("split_from_merged")),
            }
        )
    return sorted(candidates, key=lambda value: float(value["position"]))

def nearest_ocr_tick(
    position: float,
    candidates: list[dict[str, object]],
    tolerance: float,
    expected_tick: dict[str, object] | None = None,
    *,
    require_expected_agreement: bool = False,
) -> dict[str, object] | None:
    if not candidates:
        return None
    def candidate_distance(item: dict[str, object]) -> tuple[float, str]:
        anchors = item.get("anchor_positions")
        best_distance = abs(float(item["position"]) - position)
        best_source = str(item.get("position_source", "center") or "center")
        if isinstance(anchors, list):
            for anchor in anchors:
                if not isinstance(anchor, dict):
                    continue
                try:
                    anchor_position = float(anchor["position"])
                except (KeyError, TypeError, ValueError):
                    continue
                distance = abs(anchor_position - position)
                if distance < best_distance:
                    best_distance = distance
                    best_source = str(anchor.get("source", "anchor") or "anchor")
        return best_distance, best_source

    in_range = [item for item in candidates if candidate_distance(item)[0] <= tolerance]
    if expected_tick is not None:
        compatible = [item for item in in_range if labels_agree(item, expected_tick)]
        if compatible:
            best = min(compatible, key=lambda item: candidate_distance(item)[0])
        elif require_expected_agreement:
            return None
        else:
            best = min(candidates, key=lambda item: candidate_distance(item)[0])
    else:
        best = min(candidates, key=lambda item: candidate_distance(item)[0])
    distance, anchor_source = candidate_distance(best)
    if distance > tolerance:
        return None
    copy = dict(best)
    copy["distance"] = round(distance, 3)
    copy["matched_position_source"] = anchor_source
    return copy

def choose_mllm_tick_order(
    axis_key: str,
    positions: list[float],
    mllm_ticks: list[dict[str, object]],
    ocr_matches: list[dict[str, object] | None],
    mllm_axis: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    if not mllm_ticks:
        return []
    ordered = list(mllm_ticks)
    axis_data = mllm_axis if isinstance(mllm_axis, dict) else {}
    declared_order = str(axis_data.get("tick_order", "") or "").strip().casefold()
    try:
        axis_confidence = float(axis_data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        axis_confidence = 0.0
    numeric_regular = infer_numeric_tick_step(ordered) is not None
    original_numeric_values = [float(item["numeric"]) for item in ordered if item.get("numeric") is not None]
    declared_order_numeric_consistent = True
    if declared_order and len(original_numeric_values) >= 2:
        numeric_delta = original_numeric_values[-1] - original_numeric_values[0]
        if declared_order in {"left_to_right", "bottom_to_top"}:
            declared_order_numeric_consistent = numeric_delta > 0
        elif declared_order in {"right_to_left", "top_to_bottom"}:
            declared_order_numeric_consistent = numeric_delta < 0
    if axis_key == "x_axis" and declared_order == "right_to_left":
        ordered = list(reversed(ordered))
    elif axis_key == "y_axis" and declared_order == "bottom_to_top":
        ordered = list(reversed(ordered))
    elif axis_key == "y_axis" and declared_order not in {"top_to_bottom", "bottom_to_top"}:
        numeric_values = [item["numeric"] for item in ordered if item.get("numeric") is not None]
        if len(numeric_values) >= 2 and float(numeric_values[-1]) > float(numeric_values[0]):
            ordered = list(reversed(ordered))

    def mismatch_score(labels: list[dict[str, object]]) -> tuple[float, int]:
        score = 0.0
        matched = 0
        for index, ocr in enumerate(ocr_matches):
            if not ocr or index >= len(labels):
                continue
            ocr_numeric = ocr.get("numeric")
            label_numeric = labels[index].get("numeric")
            if ocr_numeric is not None and label_numeric is not None:
                denominator = max(1.0, abs(float(ocr_numeric)), abs(float(label_numeric)))
                score += abs(float(ocr_numeric) - float(label_numeric)) / denominator
                matched += 1
            elif str(ocr.get("text", "")).casefold() == str(labels[index].get("text", "")).casefold():
                matched += 1
            else:
                score += 1.0
                matched += 1
        return (score if matched else 0.0), matched

    if positions and len(ordered) > len(positions) and len(ordered) - len(positions) <= 3:
        window_size = len(positions)
        windows = [ordered[offset : offset + window_size] for offset in range(len(ordered) - window_size + 1)]
        if windows:
            ordered = min(windows, key=lambda labels: mismatch_score(labels)[0])

    skip_endpoint_expansion_for_x_window = (
        axis_key == "x_axis"
        and positions
        and len(ordered) >= 4
        and 0 < len(positions) - len(ordered) <= 3
    )
    if positions and len(ordered) < len(positions) and not skip_endpoint_expansion_for_x_window:
        ordered = expand_numeric_mllm_ticks(ordered, len(positions), ocr_matches)

    if len(ordered) == len(positions):
        reversed_order = list(reversed(ordered))
        ordered_score, ordered_matches = mismatch_score(ordered)
        reversed_score, reversed_matches = mismatch_score(reversed_order)
        enough_matches = max(ordered_matches, reversed_matches) >= max(3, int(np.ceil(len(positions) * 0.25)))
        explicit_reliable_order = (
            declared_order in {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}
            and axis_confidence >= 0.85
            and numeric_regular
            and declared_order_numeric_consistent
        )
        if enough_matches and not explicit_reliable_order and reversed_score + 0.05 < ordered_score:
            ordered = reversed_order
    return ordered

def mllm_tick_for_index(
    index: int,
    line_count: int,
    ordered_ticks: list[dict[str, object]],
) -> dict[str, object] | None:
    if not ordered_ticks:
        return None
    if len(ordered_ticks) == line_count:
        return dict(ordered_ticks[index])
    return None

def ordered_mllm_sequence_is_reliable(
    positions: list[float],
    ordered_ticks: list[dict[str, object]],
    ocr_matches: list[dict[str, object] | None],
    mllm_axis: dict[str, object],
) -> dict[str, object]:
    details: dict[str, object] = {
        "enabled": False,
        "reason": "not_checked",
        "agree_count": 0,
        "conflict_count": 0,
        "ocr_match_count": 0,
    }
    if len(positions) < 3 or len(ordered_ticks) != len(positions):
        details["reason"] = "count_mismatch"
        return details
    numeric_regular = infer_numeric_tick_step(ordered_ticks) is not None
    text_sequence = all(str(item.get("text", "") or "").strip() for item in ordered_ticks)
    all_numeric_like_text = all(numeric_like_tick_text(str(item.get("text", "") or "")) for item in ordered_ticks)

    position_diffs = np.diff(np.array(positions, dtype=np.float64))
    if len(position_diffs) >= 2:
        median_gap = float(np.median(position_diffs))
        if median_gap <= 0:
            details["reason"] = "positions_not_increasing"
            return details
        position_residual = float(np.median(np.abs(position_diffs - median_gap)) / max(1.0, abs(median_gap)))
    else:
        position_residual = 0.0
    if position_residual > 0.20:
        details["reason"] = "grid_positions_irregular"
        details["position_residual"] = round(position_residual, 3)
        return details

    agree_count = 0
    conflict_count = 0
    match_count = 0
    for index, ocr in enumerate(ocr_matches):
        if not ocr or index >= len(ordered_ticks):
            continue
        match_count += 1
        if labels_agree(ocr, ordered_ticks[index]):
            agree_count += 1
        else:
            conflict_count += 1

    try:
        axis_confidence = float(mllm_axis.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        axis_confidence = 0.0
    axis_type = str(mllm_axis.get("type", "") or "").strip().casefold()
    if all_numeric_like_text and not numeric_regular and axis_type != "time":
        details["reason"] = "irregular_numeric_mllm_sequence"
        details["axis_confidence"] = round(axis_confidence, 3)
        details["axis_type"] = axis_type or None
        return details
    if not numeric_regular and not (text_sequence and axis_confidence >= 0.85):
        details["reason"] = "mllm_sequence_not_reliable"
        details["axis_confidence"] = round(axis_confidence, 3)
        return details
    enough_ocr_agreement = agree_count >= max(3, int(np.ceil(match_count * 0.55))) if match_count else False
    high_confidence_no_ocr = match_count == 0 and axis_confidence >= 0.85
    high_confidence_low_ocr = 0 < match_count <= 2 and axis_confidence >= 0.85
    high_confidence_some_agreement = axis_confidence >= 0.9 and agree_count >= max(2, int(np.ceil(match_count * 0.35)))
    high_confidence_time_sequence = (
        axis_type == "time"
        and axis_confidence >= 0.9
        and text_sequence
        and position_residual <= 0.08
    )

    details.update(
        {
            "agree_count": agree_count,
            "conflict_count": conflict_count,
            "ocr_match_count": match_count,
            "position_residual": round(position_residual, 3),
            "axis_confidence": round(axis_confidence, 3),
        }
    )
    high_confidence_regular_numeric = numeric_regular and axis_confidence >= 0.85 and position_residual <= 0.08
    if (
        enough_ocr_agreement
        or high_confidence_no_ocr
        or high_confidence_low_ocr
        or high_confidence_some_agreement
        or high_confidence_regular_numeric
        or high_confidence_time_sequence
    ):
        details["enabled"] = True
        if high_confidence_time_sequence and not (enough_ocr_agreement or high_confidence_some_agreement):
            details["reason"] = "confident_mllm_time_sequence"
        else:
            details["reason"] = "regular_mllm_numeric_sequence" if numeric_regular else "confident_mllm_text_sequence"
    else:
        details["reason"] = "insufficient_ocr_agreement"
    return details

def labels_agree(left: dict[str, object], right: dict[str, object]) -> bool:
    left_text = str(left.get("text", "") or "").strip()
    right_text = str(right.get("text", "") or "").strip()
    if left_text and right_text and left_text.casefold() == right_text.casefold():
        return True
    if re.search(r"[A-Za-z]", left_text) or re.search(r"[A-Za-z]", right_text):
        return False
    left_numeric = left.get("numeric")
    right_numeric = right.get("numeric")
    if left_numeric is None or right_numeric is None:
        return False
    left_value = float(left_numeric)
    right_value = float(right_numeric)
    if (
        1000.0 <= abs(left_value) <= 2500.0
        and 1000.0 <= abs(right_value) <= 2500.0
        and float(left_value).is_integer()
        and float(right_value).is_integer()
    ):
        return abs(left_value - right_value) <= 1e-6
    scale = max(abs(float(left_numeric)), abs(float(right_numeric)))
    tolerance = max(1e-6, scale * 0.02)
    if scale < 100.0:
        tolerance = min(tolerance, 0.01)
    return abs(left_value - right_value) <= tolerance

def label_text_quality(text: str) -> tuple[int, int, int]:
    stripped = text.strip()
    parsed = parse_numeric_label(numeric_parse_text(stripped))
    trailing_noise = 0
    if parsed is not None:
        trailing_noise = len(re.sub(r"[-+0-9.,eE%\s]", "", stripped))
        if re.search(r"[-•·|]+$", stripped):
            trailing_noise += 2
    replacement_noise = stripped.count("?") + stripped.count("�")
    return (trailing_noise + replacement_noise * 3, len(stripped), -sum(ch.isalnum() for ch in stripped))

def choose_clean_agreeing_text(ocr_tick: dict[str, object], mllm_tick: dict[str, object]) -> str:
    ocr_text = str(ocr_tick.get("text", "") or "").strip()
    mllm_text = str(mllm_tick.get("text", "") or "").strip()
    if not ocr_text:
        return mllm_text
    if not mllm_text:
        return ocr_text
    if not numeric_like_tick_text(mllm_text) and re.search(r"[A-Za-z]", mllm_text):
        return mllm_text
    ocr_numeric = ocr_tick.get("numeric")
    mllm_numeric = mllm_tick.get("numeric")
    if ocr_numeric is not None and mllm_numeric is not None:
        return min([ocr_text, mllm_text], key=label_text_quality)
    return ocr_text if len(ocr_text) <= len(mllm_text) + 2 else mllm_text

def fuse_tick_label(
    ocr_tick: dict[str, object] | None,
    mllm_tick: dict[str, object] | None,
    *,
    prefer_mllm_on_conflict: bool = False,
) -> dict[str, object]:
    if mllm_tick and mllm_tick.get("extrapolated"):
        if ocr_tick:
            return {
                "text": str(ocr_tick.get("text", "")),
                "numeric": ocr_tick.get("numeric"),
                "source": "ocr",
                "confidence": round(float(ocr_tick.get("confidence", 0.0)), 3),
                "ocr": ocr_tick,
                "mllm": mllm_tick,
                "mllm_inferred_ignored": True,
            }
        return {
            "text": "",
            "numeric": None,
            "source": "none",
            "confidence": 0.0,
            "ocr": None,
            "mllm": mllm_tick,
            "mllm_inferred_ignored": True,
        }
    if ocr_tick and mllm_tick and labels_agree(ocr_tick, mllm_tick):
        confidence = min(1.0, max(float(ocr_tick.get("confidence", 0.0)), float(mllm_tick.get("confidence", 0.0))) + 0.08)
        return {
            "text": choose_clean_agreeing_text(ocr_tick, mllm_tick),
            "numeric": ocr_tick.get("numeric", mllm_tick.get("numeric")),
            "source": "ocr+mllm",
            "confidence": round(confidence, 3),
            "ocr": ocr_tick,
            "mllm": mllm_tick,
        }
    if ocr_tick and mllm_tick and prefer_mllm_on_conflict:
        confidence = float(mllm_tick.get("confidence", 0.0) or 0.0)
        return {
            "text": str(mllm_tick.get("text", "")),
            "numeric": mllm_tick.get("numeric"),
            "source": "mllm",
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "ocr": ocr_tick,
            "mllm": mllm_tick,
            "conflict": {
                "reason": "ocr_mllm_disagree_mllm_pattern_preferred",
                "ocr_text": str(ocr_tick.get("text", "")),
                "mllm_text": str(mllm_tick.get("text", "")),
            },
        }
    if ocr_tick:
        return {
            "text": str(ocr_tick.get("text", "")),
            "numeric": ocr_tick.get("numeric"),
            "source": "ocr",
            "confidence": round(float(ocr_tick.get("confidence", 0.0)), 3),
            "ocr": ocr_tick,
            "mllm": mllm_tick,
        }
    if mllm_tick:
        confidence = float(mllm_tick.get("confidence", 0.0) or 0.0)
        if mllm_tick.get("index_scaled"):
            confidence *= 0.75
        return {
            "text": str(mllm_tick.get("text", "")),
            "numeric": mllm_tick.get("numeric"),
            "source": "mllm",
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "ocr": None,
            "mllm": mllm_tick,
        }
    return {
        "text": "",
        "numeric": None,
        "source": "none",
        "confidence": 0.0,
        "ocr": None,
        "mllm": None,
    }

def bind_axis_ticks_to_grid(
    axis_key: str,
    grid_mask: np.ndarray,
    orientation: str,
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    fused_axis_evidence: dict[str, object],
) -> dict[str, object]:
    positions, bounds = grid_positions_and_bounds(grid_mask, orientation)
    ocr_axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
    mllm_axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
    fused_axis = fused_axis_evidence.get(axis_key, {}) if isinstance(fused_axis_evidence, dict) else {}
    if not isinstance(ocr_axis, dict):
        ocr_axis = {}
    if not isinstance(mllm_axis, dict):
        mllm_axis = {}
    if not isinstance(fused_axis, dict):
        fused_axis = {}

    image_size = ocr_axis_evidence.get("image_size", {}) if isinstance(ocr_axis_evidence, dict) else {}
    ocr_candidates = ocr_tick_candidates_for_axis(ocr_axis, axis_key, image_size if isinstance(image_size, dict) else None)
    mllm_ticks = normalize_tick_texts(mllm_axis.get("tick_labels", []))
    if positions:
        diffs = np.diff(np.array(positions, dtype=np.float32))
        median_gap = float(np.median(diffs)) if len(diffs) else 30.0
    else:
        median_gap = 30.0
    tolerance = max(10.0, min(28.0, median_gap * 0.35))
    ocr_matches = [nearest_ocr_tick(position, ocr_candidates, tolerance) for position in positions]
    ordered_mllm_ticks = choose_mllm_tick_order(axis_key, positions, mllm_ticks, ocr_matches, mllm_axis)
    axis_type = str(mllm_axis.get("type") or fused_axis.get("type") or ocr_axis.get("type") or "unknown")
    binding_positions = list(positions)
    binding_mode = "grid_lines"
    if (
        axis_key == "x_axis"
        and axis_type in {"category", "time", "mixed"}
        and len(positions) >= 3
        and len(ordered_mllm_ticks) == len(positions) - 1
    ):
        interval_positions = [
            (float(positions[index]) + float(positions[index + 1])) / 2.0
            for index in range(len(positions) - 1)
        ]
        if len(interval_positions) >= 3 and float(np.std(np.diff(np.array(interval_positions, dtype=np.float64)))) <= max(3.0, median_gap * 0.12):
            binding_positions = interval_positions
            binding_mode = "interval_centers"
            tolerance = max(10.0, min(28.0, median_gap * 0.45))
            ocr_matches = [nearest_ocr_tick(position, ocr_candidates, tolerance) for position in binding_positions]
    elif (
        axis_key == "x_axis"
        and len(ordered_mllm_ticks) >= 4
        and 0 < len(positions) - len(ordered_mllm_ticks) <= 3
    ):
        window_size = len(ordered_mllm_ticks)
        windows = [positions[offset : offset + window_size] for offset in range(len(positions) - window_size + 1)]

        def window_score(window: list[float]) -> tuple[float, float]:
            local_tolerance = max(10.0, min(28.0, median_gap * 0.45))
            matches = [
                nearest_ocr_tick(pos, ocr_candidates, local_tolerance, ordered_mllm_ticks[index])
                for index, pos in enumerate(window)
            ]
            agree = sum(1 for index, item in enumerate(matches) if item and labels_agree(item, ordered_mllm_ticks[index]))
            conflicts = sum(1 for index, item in enumerate(matches) if item and not labels_agree(item, ordered_mllm_ticks[index]))
            center_penalty = abs((window[0] + window[-1]) / 2.0 - (positions[0] + positions[-1]) / 2.0) / max(1.0, median_gap)
            return (-agree + conflicts * 1.5, center_penalty)

        if windows:
            binding_positions = list(min(windows, key=window_score))
            binding_mode = "grid_line_window"
            tolerance = max(10.0, min(28.0, median_gap * 0.45))
            ocr_matches = [
                nearest_ocr_tick(
                    position,
                    ocr_candidates,
                    tolerance,
                    ordered_mllm_ticks[index] if index < len(ordered_mllm_ticks) else None,
                )
                for index, position in enumerate(binding_positions)
            ]

    mllm_pattern_guard = ordered_mllm_sequence_is_reliable(binding_positions, ordered_mllm_ticks, ocr_matches, mllm_axis)
    prefer_mllm_on_conflict = bool(mllm_pattern_guard.get("enabled", False))
    if prefer_mllm_on_conflict:
        ocr_matches = [
            nearest_ocr_tick(
                position,
                ocr_candidates,
                tolerance,
                mllm_tick_for_index(index, len(binding_positions), ordered_mllm_ticks),
                require_expected_agreement=True,
            )
            for index, position in enumerate(binding_positions)
        ]

    bindings = []
    for index, position in enumerate(binding_positions):
        ocr_tick = ocr_matches[index]
        mllm_tick = mllm_tick_for_index(index, len(binding_positions), ordered_mllm_ticks)
        if not prefer_mllm_on_conflict and ocr_tick is None:
            mllm_tick = None
        fused_label = fuse_tick_label(
            ocr_tick,
            mllm_tick,
            prefer_mllm_on_conflict=prefer_mllm_on_conflict,
        )
        bindings.append(
            {
                "line_index": index,
                "position": round(float(position), 3),
                "orientation": orientation,
                "label": fused_label["text"],
                "numeric": fused_label["numeric"],
                "source": fused_label["source"],
                "confidence": fused_label["confidence"],
                "ocr": fused_label["ocr"],
                "mllm": fused_label["mllm"],
            }
        )

    return {
        "axis": axis_key,
        "grid_orientation": orientation,
        "axis_label": fused_axis.get("axis_label", {}),
        "grid_line_count": len(positions),
        "bounds": bounds,
        "ocr_tick_count": len(ocr_candidates),
        "mllm_tick_count": len(mllm_ticks),
        "binding_tolerance": round(tolerance, 3),
        "binding_mode": binding_mode,
        "mllm_pattern_guard": mllm_pattern_guard,
        "tick_bindings": bindings,
    }

def zero_like_label(binding: dict[str, object]) -> bool:
    text = str(binding.get("label", "") or "").strip()
    if not text:
        return False
    value = parse_numeric_label(numeric_parse_text(text))
    return value is not None and abs(float(value)) <= 1e-9

def axis_tick_label_perpendicular(
    ocr_axis_evidence: dict[str, object] | None,
    axis_key: str,
) -> float | None:
    axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
    ticks = axis.get("ticks", []) if isinstance(axis, dict) else []
    values: list[float] = []
    for tick in ticks if isinstance(ticks, list) else []:
        if not isinstance(tick, dict) or tick.get("mllm_pseudo_box"):
            continue
        key = "y" if axis_key == "x_axis" else "x"
        try:
            values.append(float(tick[key]))
        except (KeyError, TypeError, ValueError):
            center = tick.get("center")
            if isinstance(center, (list, tuple)) and len(center) >= 2:
                try:
                    values.append(float(center[1 if axis_key == "x_axis" else 0]))
                except (TypeError, ValueError):
                    pass
    if not values:
        return None
    return float(np.median(np.array(values, dtype=np.float64)))

def binding_display_point(
    axis: dict[str, object],
    binding: dict[str, object],
    axis_key: str,
    ocr_axis_evidence: dict[str, object] | None = None,
) -> tuple[float, float] | None:
    ocr = binding.get("ocr")
    if isinstance(ocr, dict):
        center = ocr.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            try:
                return float(center[0]), float(center[1])
            except (TypeError, ValueError):
                pass
    try:
        position = float(binding.get("position"))
    except (TypeError, ValueError):
        return None
    tick_perpendicular = axis_tick_label_perpendicular(ocr_axis_evidence, axis_key)
    if tick_perpendicular is not None:
        if axis_key == "x_axis":
            return position, tick_perpendicular
        return tick_perpendicular, position
    bounds = axis.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) < 2:
        return None
    try:
        lower = float(bounds[0])
        upper = float(bounds[1])
    except (TypeError, ValueError):
        return None
    if axis_key == "x_axis":
        return position, max(lower, upper)
    return min(lower, upper), position

def ocr_detection_signature(ocr: dict[str, object] | None) -> tuple[object, ...] | None:
    if not isinstance(ocr, dict) or ocr.get("mllm_pseudo_box"):
        return None
    text = normalize_label_text(str(ocr.get("text", "") or ""))
    box = ocr.get("box")
    if isinstance(box, list) and box:
        coords: list[tuple[int, int]] = []
        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                coords = []
                break
            try:
                coords.append((int(round(float(point[0]))), int(round(float(point[1])))))
            except (TypeError, ValueError):
                coords = []
                break
        if coords:
            return ("box", text, tuple(coords))
    center = ocr.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return ("center", text, int(round(float(center[0]))), int(round(float(center[1]))))
        except (TypeError, ValueError):
            return None
    return None

def duplicate_origin_binding_score(binding: dict[str, object], axis_key: str) -> float:
    score = float(binding.get("confidence", 0.0) or 0.0)
    source = str(binding.get("source", "") or "")
    if source == "ocr+mllm":
        score += 2.0
    elif source == "ocr":
        score += 1.0
    ocr = binding.get("ocr")
    if isinstance(ocr, dict):
        if ocr.get("canonical_axis") == axis_key:
            score += 1.0
        if ocr.get("role") == axis_key:
            score += 0.5
        if ocr.get("raw_role") == axis_key:
            score += 0.25
    return score

def suppress_binding_as_duplicate_origin(
    binding: dict[str, object],
    *,
    kept_axis: str,
) -> None:
    binding["unbound_label"] = binding.get("label")
    binding["unbound_numeric"] = binding.get("numeric")
    binding["unbound_source"] = binding.get("source")
    binding["unbound_confidence"] = binding.get("confidence")
    binding["unbound_ocr"] = binding.get("ocr")
    binding["unbound_mllm"] = binding.get("mllm")
    binding["label"] = ""
    binding["numeric"] = None
    binding["source"] = "none"
    binding["confidence"] = 0.0
    binding["display_suppressed"] = True
    binding["display_suppression_reason"] = "duplicate_origin_ocr_binding"
    binding["display_suppressed_by_axis"] = kept_axis
    binding["unbound_reason"] = "duplicate_origin_ocr_binding"

def suppress_duplicate_origin_ocr_display(
    x_axis: dict[str, object],
    y_axis: dict[str, object],
    ocr_axis_evidence: dict[str, object] | None = None,
) -> None:
    x_bindings = x_axis.get("tick_bindings", [])
    y_bindings = y_axis.get("tick_bindings", [])
    if not isinstance(x_bindings, list) or not isinstance(y_bindings, list):
        return
    suppressed: set[int] = set()
    for x_binding in x_bindings:
        if not isinstance(x_binding, dict) or id(x_binding) in suppressed or not zero_like_label(x_binding):
            continue
        x_signature = ocr_detection_signature(x_binding.get("ocr"))
        if x_signature is None:
            continue
        x_point = binding_display_point(x_axis, x_binding, "x_axis", ocr_axis_evidence)
        if x_point is None:
            continue
        for y_binding in y_bindings:
            if not isinstance(y_binding, dict) or id(y_binding) in suppressed or not zero_like_label(y_binding):
                continue
            if ocr_detection_signature(y_binding.get("ocr")) != x_signature:
                continue
            y_point = binding_display_point(y_axis, y_binding, "y_axis", ocr_axis_evidence)
            if y_point is None:
                continue
            distance = float(np.hypot(x_point[0] - y_point[0], x_point[1] - y_point[1]))
            if distance > 24.0:
                continue
            x_score = duplicate_origin_binding_score(x_binding, "x_axis")
            y_score = duplicate_origin_binding_score(y_binding, "y_axis")
            if y_score > x_score:
                suppress_binding_as_duplicate_origin(x_binding, kept_axis="y_axis")
                suppressed.add(id(x_binding))
            else:
                suppress_binding_as_duplicate_origin(y_binding, kept_axis="x_axis")
                suppressed.add(id(y_binding))
            break

def build_grid_label_bindings(
    grid_horizontal: np.ndarray,
    grid_vertical: np.ndarray,
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    fused_axis_evidence: dict[str, object],
) -> dict[str, object]:
    x_axis = bind_axis_ticks_to_grid(
        "x_axis",
        grid_vertical,
        "vertical",
        ocr_axis_evidence,
        mllm_result,
        fused_axis_evidence,
    )
    y_axis = bind_axis_ticks_to_grid(
        "y_axis",
        grid_horizontal,
        "horizontal",
        ocr_axis_evidence,
        mllm_result,
        fused_axis_evidence,
    )
    suppress_duplicate_origin_ocr_display(x_axis, y_axis, ocr_axis_evidence)
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "summary": {
            "x_bound_labels": sum(1 for item in x_axis["tick_bindings"] if item.get("label")),
            "y_bound_labels": sum(1 for item in y_axis["tick_bindings"] if item.get("label")),
        },
    }
