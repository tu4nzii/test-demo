from __future__ import annotations

import argparse
import cv2
import numpy as np

from grid_bindings import build_grid_label_bindings
from grid_geometry import grid_positions_and_bounds
from grid_mllm import run_mllm_grid_arbitration
from grid_visual import draw_grid_label_overlay

SOURCE_LABELS = {
    "combined_mask": "Priority 1: combined mask",
    "tick_supplement": "Priority 2: tick supplement",
    "semantic_guide": "Priority 3: semantic guide",
}
SOURCE_ORDER = {
    "combined_mask": 0,
    "tick_supplement": 1,
    "semantic_guide": 2,
}
SOURCE_COLORS = {
    "combined_mask": (230, 80, 40),
    "tick_supplement": (0, 150, 255),
    "semantic_guide": (0, 180, 0),
}
SOURCE_COLOR_NAMES = {
    "combined_mask": "blue",
    "tick_supplement": "orange",
    "semantic_guide": "green",
}

def axis_target_count(mllm_result: dict[str, object], axis_key: str) -> int:
    axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
    ticks = axis.get("tick_labels", []) if isinstance(axis, dict) else []
    return len(ticks) if isinstance(ticks, list) else 0

def ocr_binding_identity(binding: dict[str, object]) -> tuple[object, ...] | None:
    ocr = binding.get("ocr")
    if not isinstance(ocr, dict):
        return None
    canonical_axis = ocr.get("canonical_axis")
    canonical_index = ocr.get("canonical_index")
    if canonical_axis is not None and canonical_index is not None:
        return ("canonical", canonical_axis, canonical_index)
    text = str(ocr.get("text", "") or "").strip().casefold()
    center = ocr.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        try:
            return ("center", text, round(float(center[0]), 1), round(float(center[1]), 1))
        except (TypeError, ValueError):
            pass
    return ("text", text) if text else None

def binding_quality(axis_binding: dict[str, object], target_count: int) -> dict[str, object]:
    bindings = axis_binding.get("tick_bindings", []) if isinstance(axis_binding, dict) else []
    if not isinstance(bindings, list):
        bindings = []
    line_count = int(axis_binding.get("grid_line_count", 0) or 0) if isinstance(axis_binding, dict) else 0
    labeled = [item for item in bindings if isinstance(item, dict) and str(item.get("label", "") or "").strip()]
    strong = [item for item in labeled if str(item.get("source", "")) == "ocr+mllm"]
    ocr_bound = [item for item in labeled if str(item.get("source", "")) in {"ocr+mllm", "ocr"}]
    mllm_only = [item for item in labeled if str(item.get("source", "")) == "mllm"]
    identities = [ocr_binding_identity(item) for item in ocr_bound]
    unique_ocr_count = len({identity for identity in identities if identity is not None})
    duplicate_ocr_count = max(0, len([identity for identity in identities if identity is not None]) - unique_ocr_count)
    ocr_distances: list[float] = []
    for item in ocr_bound:
        ocr = item.get("ocr")
        if not isinstance(ocr, dict):
            continue
        try:
            ocr_distances.append(float(ocr.get("distance", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    mean_ocr_distance = float(np.mean(np.array(ocr_distances, dtype=np.float64))) if ocr_distances else None
    max_ocr_distance = float(np.max(np.array(ocr_distances, dtype=np.float64))) if ocr_distances else None
    count_error = abs(line_count - target_count) if target_count >= 2 else 0
    extra_unlabeled = max(0, line_count - len(labeled))
    score = len(labeled) * 1.2 + len(strong) * 2.0 + (len(ocr_bound) - len(strong)) * 1.1 + len(mllm_only) * 0.15
    score -= count_error * 1.25
    score -= extra_unlabeled * 0.35
    score -= duplicate_ocr_count * 1.6
    if mean_ocr_distance is not None:
        score -= mean_ocr_distance * 0.35
    if max_ocr_distance is not None:
        score -= max(0.0, max_ocr_distance - 3.0) * 0.25
    invalid_reasons: list[str] = []
    if line_count <= 0:
        invalid_reasons.append("no_grid_lines")
    if target_count >= 3 and line_count < max(2, int(np.floor(target_count * 0.45))):
        invalid_reasons.append("too_few_lines_for_mllm_ticks")
    if target_count >= 3 and len(labeled) < max(1, int(np.floor(target_count * 0.35))):
        invalid_reasons.append("too_few_bound_labels")
    if target_count >= 4 and len(ocr_bound) >= max(3, int(np.ceil(target_count * 0.6))) and unique_ocr_count < max(3, int(np.ceil(target_count * 0.6))):
        invalid_reasons.append("too_few_unique_ocr_bindings")
    if target_count >= 3 and line_count > max(target_count + 6, int(np.ceil(target_count * 2.2))):
        invalid_reasons.append("too_many_lines_for_mllm_ticks")
    return {
        "score": round(float(score), 3),
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "line_count": line_count,
        "labeled_count": len(labeled),
        "strong_count": len(strong),
        "ocr_bound_count": len(ocr_bound),
        "mllm_only_count": len(mllm_only),
        "unique_ocr_count": unique_ocr_count,
        "duplicate_ocr_count": duplicate_ocr_count,
        "mean_ocr_distance": round(mean_ocr_distance, 3) if mean_ocr_distance is not None else None,
        "max_ocr_distance": round(max_ocr_distance, 3) if max_ocr_distance is not None else None,
        "target_count": target_count,
    }

def apply_native_numeric_axis_bonus(scores: dict[str, dict[str, object]], axis_type: object) -> None:
    if str(axis_type or "").casefold() not in {"numeric", "time"}:
        return
    native = scores.get("combined_mask")
    if not isinstance(native, dict) or not native.get("valid"):
        return
    try:
        line_count = int(native.get("line_count", 0) or 0)
        target_count = int(native.get("target_count", 0) or 0)
        strong_count = int(native.get("strong_count", 0) or 0)
        ocr_bound_count = int(native.get("ocr_bound_count", 0) or 0)
        score = float(native.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return
    if target_count < 3 or line_count != target_count:
        return
    support_count = max(strong_count, ocr_bound_count)
    if support_count < max(3, int(np.ceil(target_count * 0.6))):
        native["score_adjustment"] = {
            "reason": "native_bonus_skipped_without_ocr_strong_binding",
            "required_strong_count": max(3, int(np.ceil(target_count * 0.6))),
            "strong_count": strong_count,
            "ocr_bound_count": ocr_bound_count,
            "base_score": round(score, 3),
        }
        return
    bonus = 30.0
    native["score"] = round(score + bonus, 3)
    native["score_adjustment"] = {
        "reason": "complete_native_grid_on_numeric_or_time_axis",
        "bonus": bonus,
        "base_score": round(score, 3),
    }

def apply_physical_tick_axis_bonus(scores: dict[str, dict[str, object]]) -> None:
    tick = scores.get("tick_supplement")
    if not isinstance(tick, dict) or not tick.get("valid"):
        return
    try:
        line_count = int(tick.get("line_count", 0) or 0)
        target_count = int(tick.get("target_count", 0) or 0)
        labeled_count = int(tick.get("labeled_count", 0) or 0)
        strong_count = int(tick.get("strong_count", 0) or 0)
        ocr_bound_count = int(tick.get("ocr_bound_count", 0) or 0)
        score = float(tick.get("score", 0.0) or 0.0)
        mean_distance_value = tick.get("mean_ocr_distance")
        max_distance_value = tick.get("max_ocr_distance")
        mean_distance = float(mean_distance_value) if mean_distance_value is not None else None
        max_distance = float(max_distance_value) if max_distance_value is not None else None
    except (TypeError, ValueError):
        return
    if target_count < 3 or line_count != target_count:
        return
    if labeled_count < max(3, int(np.ceil(target_count * 0.85))):
        return
    if strong_count < max(3, int(np.ceil(target_count * 0.60))):
        return
    if ocr_bound_count < max(3, int(np.ceil(target_count * 0.60))):
        return
    if mean_distance is not None and mean_distance > 5.0:
        return
    if max_distance is not None and max_distance > 12.0:
        return
    bonus = 5.0
    tick["score"] = round(score + bonus, 3)
    tick["score_adjustment"] = {
        "reason": "complete_physical_tick_grid",
        "bonus": bonus,
        "base_score": round(score, 3),
        "mean_ocr_distance": tick.get("mean_ocr_distance"),
        "max_ocr_distance": tick.get("max_ocr_distance"),
    }

def recommended_grid_axis_value(mllm_result: dict[str, object], orientation: str) -> str:
    if not isinstance(mllm_result, dict):
        return ""
    recommended = mllm_result.get("recommended_grid", {})
    if not isinstance(recommended, dict):
        return ""
    return str(recommended.get(orientation, "") or "").strip().casefold()

def apply_recommended_avoid_penalty(
    scores: dict[str, dict[str, object]],
    mllm_result: dict[str, object],
    orientation: str,
) -> None:
    if recommended_grid_axis_value(mllm_result, orientation) not in {"avoid", "none", "no"}:
        return
    for source, data in scores.items():
        if not isinstance(data, dict) or not data.get("valid"):
            if isinstance(data, dict) and source == "semantic_guide":
                try:
                    line_count = int(data.get("line_count", 0) or 0)
                    ocr_bound_count = int(data.get("ocr_bound_count", 0) or 0)
                    mllm_only_count = int(data.get("mllm_only_count", 0) or 0)
                    score = float(data.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                reasons = data.get("invalid_reasons", [])
                if (
                    line_count > 0
                    and ocr_bound_count > 0
                    and mllm_only_count == 0
                    and isinstance(reasons, list)
                ):
                    remaining = [
                        reason
                        for reason in reasons
                        if reason not in {"too_few_lines_for_mllm_ticks", "too_few_bound_labels"}
                    ]
                    data["invalid_reasons"] = remaining
                    data["valid"] = not remaining
                    if data["valid"]:
                        data["score"] = round(max(score, 0.0) + 2.0, 3)
                        data["score_adjustment"] = {
                            "reason": f"mllm_recommended_{orientation}_avoid_sparse_semantic_grid_allowed",
                            "base_score": round(score, 3),
                            "line_count": line_count,
                            "ocr_bound_count": ocr_bound_count,
                        }
            continue
        try:
            target_count = int(data.get("target_count", 0) or 0)
            line_count = int(data.get("line_count", 0) or 0)
            strong_count = int(data.get("strong_count", 0) or 0)
            ocr_bound_count = int(data.get("ocr_bound_count", 0) or 0)
            mllm_only_count = int(data.get("mllm_only_count", 0) or 0)
            score = float(data.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if target_count < 5 or line_count < max(4, int(np.ceil(target_count * 0.65))):
            continue
        axis_key = "x_axis" if orientation == "vertical" else "y_axis"
        axis_data = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
        axis_type = str(axis_data.get("type", "") or "").strip().casefold() if isinstance(axis_data, dict) else ""
        labeled_count = int(data.get("labeled_count", 0) or 0)
        if (
            axis_type in {"numeric", "time"}
            and line_count == target_count
            and labeled_count >= max(3, int(np.ceil(target_count * 0.8)))
        ):
            data["score_adjustment"] = {
                "reason": f"mllm_recommended_{orientation}_avoid_ignored_for_complete_tick_axis",
                "base_score": round(score, 3),
                "line_count": line_count,
                "target_count": target_count,
                "labeled_count": labeled_count,
            }
            continue
        required_strong = max(3, int(np.ceil(target_count * 0.35)))
        mostly_mllm_only = mllm_only_count > max(ocr_bound_count, int(np.ceil(target_count * 0.45)))
        weak_ocr_support = strong_count < required_strong
        if not (mostly_mllm_only and weak_ocr_support):
            continue
        penalty = max(8.0, min(18.0, mllm_only_count * 0.55 + line_count * 0.20))
        data["score"] = round(score - penalty, 3)
        data["score_adjustment"] = {
            "reason": f"mllm_recommended_{orientation}_avoid_without_ocr_support",
            "penalty": round(float(penalty), 3),
            "base_score": round(score, 3),
            "required_strong_count": required_strong,
            "strong_count": strong_count,
            "mllm_only_count": mllm_only_count,
            "ocr_bound_count": ocr_bound_count,
        }
        if source in {"combined_mask", "tick_supplement"}:
            reasons = data.setdefault("invalid_reasons", [])
            if isinstance(reasons, list) and data["score"] < 20.0:
                reasons.append(f"mllm_recommended_{orientation}_avoid_without_ocr_support")
                data["valid"] = False

def choose_by_score(scores: dict[str, dict[str, object]]) -> tuple[str | None, str]:
    valid = [(source, data) for source, data in scores.items() if data.get("valid")]
    if not valid:
        return None, "no_valid_candidate"
    ordered = sorted(
        valid,
        key=lambda item: (float(item[1].get("score", 0.0)), -SOURCE_ORDER.get(item[0], 99)),
        reverse=True,
    )
    best_source, best = ordered[0]
    if len(ordered) == 1:
        return best_source, "single_valid_candidate"
    second = ordered[1][1]
    if float(best.get("score", 0.0)) > float(second.get("score", 0.0)) + 1e-6:
        return best_source, "highest_score"
    return None, "needs_mllm"

def prefer_physical_tick_over_semantic(
    choice: str | None,
    reason: str,
    scores: dict[str, dict[str, object]],
    axis_type: object,
    *,
    max_semantic_lead: float = 5.5,
) -> tuple[str | None, str]:
    if choice != "semantic_guide":
        return choice, reason
    if str(axis_type or "").casefold() not in {"category", "time", "mixed"}:
        return choice, reason
    semantic = scores.get("semantic_guide")
    tick = scores.get("tick_supplement")
    if not isinstance(semantic, dict) or not isinstance(tick, dict) or not tick.get("valid"):
        return choice, reason
    try:
        semantic_score = float(semantic.get("score", 0.0) or 0.0)
        tick_score = float(tick.get("score", 0.0) or 0.0)
        target_count = int(tick.get("target_count", 0) or 0)
        line_count = int(tick.get("line_count", 0) or 0)
        ocr_bound_count = int(tick.get("ocr_bound_count", 0) or 0)
    except (TypeError, ValueError):
        return choice, reason
    if target_count < 3 or line_count != target_count:
        return choice, reason
    if ocr_bound_count < max(3, int(np.ceil(target_count * 0.8))):
        return choice, reason
    if semantic_score - tick_score <= max_semantic_lead:
        tick["score_adjustment"] = {
            "reason": "physical_tick_preferred_over_semantic_axis_when_complete",
            "semantic_score": round(semantic_score, 3),
            "tick_score": round(tick_score, 3),
            "max_semantic_lead": max_semantic_lead,
        }
        return "tick_supplement", "physical_tick_complete_preferred_over_semantic"
    return choice, reason

def prefer_complete_native_over_semantic(
    choice: str | None,
    reason: str,
    scores: dict[str, dict[str, object]],
    axis_type: object,
) -> tuple[str | None, str]:
    if choice != "semantic_guide":
        return choice, reason
    if str(axis_type or "").casefold() not in {"numeric", "time"}:
        return choice, reason
    native = scores.get("combined_mask")
    if not isinstance(native, dict) or not native.get("valid"):
        return choice, reason
    try:
        target_count = int(native.get("target_count", 0) or 0)
        line_count = int(native.get("line_count", 0) or 0)
        labeled_count = int(native.get("labeled_count", 0) or 0)
        strong_count = int(native.get("strong_count", 0) or 0)
        ocr_bound_count = int(native.get("ocr_bound_count", 0) or 0)
        unique_ocr_count = int(native.get("unique_ocr_count", 0) or 0)
        mllm_only_count = int(native.get("mllm_only_count", 0) or 0)
    except (TypeError, ValueError):
        return choice, reason
    if target_count >= 3 and line_count == target_count and labeled_count >= max(3, int(np.ceil(target_count * 0.8))):
        required_support = max(3, int(np.ceil(target_count * 0.6)))
        support_count = max(strong_count, ocr_bound_count)
        if (
            support_count < required_support
            or unique_ocr_count < required_support
            or mllm_only_count > max(1, int(np.floor(target_count * 0.25)))
        ):
            native["selection_adjustment"] = {
                "reason": "complete_native_grid_not_preferred_without_ocr_support",
                "semantic_score": scores.get("semantic_guide", {}).get("score"),
                "native_score": native.get("score"),
                "target_count": target_count,
                "labeled_count": labeled_count,
                "required_support": required_support,
                "strong_count": strong_count,
                "ocr_bound_count": ocr_bound_count,
                "unique_ocr_count": unique_ocr_count,
                "mllm_only_count": mllm_only_count,
            }
            return choice, reason
        native["selection_adjustment"] = {
            "reason": "complete_native_grid_preferred_over_semantic_axis",
            "semantic_score": scores.get("semantic_guide", {}).get("score"),
            "native_score": native.get("score"),
            "target_count": target_count,
            "labeled_count": labeled_count,
        }
        return "combined_mask", "complete_native_grid_preferred_over_semantic"
    return choice, reason

def fallback_by_score(scores: dict[str, dict[str, object]]) -> str:
    valid_sources = [source for source, data in scores.items() if data.get("valid")]
    pool = valid_sources if valid_sources else list(scores)
    return max(
        pool,
        key=lambda source: (
            float(scores.get(source, {}).get("score", 0.0)),
            -SOURCE_ORDER.get(source, 99),
        ),
    )

def fallback_equivalent_score_tie(scores: dict[str, dict[str, object]], *, tolerance: float = 1.25) -> str:
    top_sources = top_score_sources(scores, tolerance=tolerance)
    if "tick_supplement" in top_sources:
        return "tick_supplement"
    return fallback_by_score(scores)

def effective_base_score(data: dict[str, object]) -> float:
    adjustment = data.get("score_adjustment") if isinstance(data, dict) else None
    if isinstance(adjustment, dict) and adjustment.get("base_score") is not None:
        try:
            return float(adjustment.get("base_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
    try:
        return float(data.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0

def top_score_sources(scores: dict[str, dict[str, object]], *, tolerance: float = 1e-6) -> list[str]:
    valid = [(source, data) for source, data in scores.items() if data.get("valid")]
    if not valid:
        return []
    best = max(float(data.get("score", 0.0)) for _, data in valid)
    return [source for source, data in valid if abs(float(data.get("score", 0.0)) - best) <= tolerance]

def top_base_score_sources(scores: dict[str, dict[str, object]], *, tolerance: float = 1e-6) -> list[str]:
    valid = [(source, data) for source, data in scores.items() if data.get("valid")]
    if not valid:
        return []
    best = max(effective_base_score(data) for _, data in valid)
    return [source for source, data in valid if abs(effective_base_score(data) - best) <= tolerance]

def axis_mask_positions(
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    source: str,
    orientation: str,
) -> list[float]:
    if source not in candidates:
        return []
    horizontal, vertical = candidates[source]
    mask = vertical if orientation == "vertical" else horizontal
    positions, _ = grid_positions_and_bounds(mask, orientation)
    return positions

def paired_position_differences(
    first: list[float],
    second: list[float],
    first_source: str,
    second_source: str,
) -> list[dict[str, float | str]]:
    if len(first) != len(second) or not first:
        return []
    return [
        {
            "index": float(index),
            first_source: float(a),
            second_source: float(b),
            "delta": float(b - a),
            "abs_delta": abs(float(b - a)),
        }
        for index, (a, b) in enumerate(zip(first, second))
    ]

def comparable_position_pair(scores: dict[str, dict[str, object]], *, score_tolerance: float = 1.25) -> tuple[list[str], str] | None:
    current_top = top_score_sources(scores, tolerance=score_tolerance)
    if len(current_top) >= 2:
        top_set = set(current_top)
        reason = "score_near_tie" if score_tolerance > 1e-6 else "score_tie"
    else:
        base_top = top_base_score_sources(scores)
        top_set = set(base_top)
        reason = "base_score_tie_after_adjustment"
    pairs = [
        ("combined_mask", "semantic_guide"),
        ("tick_supplement", "semantic_guide"),
        ("combined_mask", "tick_supplement"),
    ]
    for left, right in pairs:
        if {left, right}.issubset(top_set):
            return [left, right], reason
    return None

def score_tie_position_analysis(
    scores: dict[str, dict[str, object]],
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    orientation: str,
    *,
    equivalent_tolerance: float = 2.5,
) -> dict[str, object] | None:
    pair = comparable_position_pair(scores)
    if pair is None:
        return None
    sources, score_reason = pair
    first_source, second_source = sources
    first_positions = axis_mask_positions(candidates, first_source, orientation)
    second_positions = axis_mask_positions(candidates, second_source, orientation)
    analysis: dict[str, object] = {
        "enabled": True,
        "sources": sources,
        "score_reason": score_reason,
        "orientation": orientation,
        f"{first_source}_positions": [round(value, 3) for value in first_positions],
        f"{second_source}_positions": [round(value, 3) for value in second_positions],
        "equivalent_tolerance": equivalent_tolerance,
    }
    if len(first_positions) != len(second_positions) or not first_positions:
        analysis["reason"] = "position_count_differs"
        analysis["needs_mllm"] = True
        return analysis
    differences = paired_position_differences(first_positions, second_positions, first_source, second_source)
    max_delta = max((item["abs_delta"] for item in differences), default=0.0)
    changed = [item for item in differences if item["abs_delta"] > equivalent_tolerance]
    analysis["position_differences"] = [
        {
            "index": int(item["index"]),
            first_source: round(float(item[first_source]), 3),
            second_source: round(float(item[second_source]), 3),
            "delta": round(item["delta"], 3),
            "abs_delta": round(item["abs_delta"], 3),
        }
        for item in differences
    ]
    analysis["max_abs_delta"] = round(float(max_delta), 3)
    analysis["changed_indices"] = [int(item["index"]) for item in changed]
    endpoint_indexes = {0, len(first_positions) - 1}
    endpoint_only_tick_semantic_diff = (
        set(sources) == {"tick_supplement", "semantic_guide"}
        and bool(changed)
        and all(int(item["index"]) in endpoint_indexes for item in changed)
        and max_delta <= max(equivalent_tolerance * 3.0, 8.0)
    )
    if endpoint_only_tick_semantic_diff:
        analysis["needs_mllm"] = False
        analysis["reason"] = "endpoint_only_difference_physical_tick_preferred"
        analysis["endpoint_only_tick_semantic_diff"] = True
        return analysis
    analysis["needs_mllm"] = bool(changed)
    analysis["reason"] = "position_diff_needs_mllm" if changed else "positions_equivalent"
    return analysis

def labeled_bindings(axis_binding: dict[str, object]) -> list[dict[str, object]]:
    bindings = axis_binding.get("tick_bindings", []) if isinstance(axis_binding, dict) else []
    labeled = [item for item in bindings if isinstance(item, dict) and str(item.get("label", "") or "").strip()]
    return sorted(labeled, key=lambda item: float(item.get("position", 0.0)))

def middle_labeled_position(axis_binding: dict[str, object]) -> float | None:
    labeled = [float(item.get("position", 0.0)) for item in labeled_bindings(axis_binding)]
    if not labeled:
        return None
    return labeled[len(labeled) // 2]

def axis_binding_positions(axis_binding: dict[str, object]) -> list[float]:
    bindings = axis_binding.get("tick_bindings", []) if isinstance(axis_binding, dict) else []
    positions: list[float] = []
    for item in bindings if isinstance(bindings, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            positions.append(float(item.get("position", 0.0)))
        except (TypeError, ValueError):
            continue
    return sorted(positions)

def ocr_axis_ticks(ocr_axis_evidence: dict[str, object], axis_key: str) -> list[dict[str, object]]:
    axis = ocr_axis_evidence.get(axis_key, {}) if isinstance(ocr_axis_evidence, dict) else {}
    ticks = axis.get("ticks", []) if isinstance(axis, dict) else []
    return [item for item in ticks if isinstance(item, dict)] if isinstance(ticks, list) else []

def nearest_ocr_cross_position(
    ocr_axis_evidence: dict[str, object],
    axis_key: str,
    target_position: float,
) -> float | None:
    ticks = ocr_axis_ticks(ocr_axis_evidence, axis_key)
    if not ticks:
        return None
    main_key = "x" if axis_key == "x_axis" else "y"
    cross_key = "y" if axis_key == "x_axis" else "x"
    candidates: list[tuple[float, float]] = []
    for item in ticks:
        try:
            main = float(item.get(main_key, 0.0))
            cross = float(item.get(cross_key, 0.0))
        except (TypeError, ValueError):
            continue
        candidates.append((abs(main - target_position), cross))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]

def local_interval(
    center: float,
    positions: list[float],
    image_len: int,
    *,
    min_frac: float,
    max_frac: float,
    min_px: int,
) -> tuple[int, int]:
    clean = sorted(float(value) for value in positions if np.isfinite(float(value)))
    if len(clean) >= 2:
        gaps = np.diff(np.array(clean, dtype=np.float64))
        median_gap = float(np.median(gaps)) if len(gaps) else image_len * min_frac
        nearest = min(range(len(clean)), key=lambda index: abs(clean[index] - center))
        start_index = max(0, nearest - 1)
        end_index = min(len(clean) - 1, nearest + 1)
        left = clean[start_index] - median_gap * 0.75
        right = clean[end_index] + median_gap * 0.75
    else:
        median_gap = image_len * min_frac
        left = center - median_gap / 2.0
        right = center + median_gap / 2.0
    min_width = max(float(min_px), image_len * min_frac)
    max_width = max(min_width, image_len * max_frac)
    width = min(max(right - left, min_width), max_width)
    mid = (left + right) / 2.0
    left = mid - width / 2.0
    right = mid + width / 2.0
    if left < 0:
        right -= left
        left = 0.0
    if right > image_len - 1:
        left -= right - (image_len - 1)
        right = float(image_len - 1)
    left = max(0.0, left)
    return int(round(left)), int(round(max(left + 1.0, right)))

def choose_reference_binding(candidate_bindings: dict[str, dict[str, object]], axis_key: str) -> dict[str, object]:
    best: dict[str, object] = {}
    best_count = -1
    for binding in candidate_bindings.values():
        axis = binding.get(axis_key, {}) if isinstance(binding, dict) else {}
        ticks = axis.get("tick_bindings", []) if isinstance(axis, dict) else []
        count = sum(1 for item in ticks if isinstance(item, dict) and str(item.get("label", "") or "").strip())
        if count > best_count:
            best = axis if isinstance(axis, dict) else {}
            best_count = count
    return best

def clamp_rect(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = rect
    x0 = max(0, min(w - 2, x0))
    x1 = max(x0 + 1, min(w - 1, x1))
    y0 = max(0, min(h - 2, y0))
    y1 = max(y0 + 1, min(h - 1, y1))
    return x0, y0, x1, y1

def x_axis_crop_rect(
    axis_binding: dict[str, object],
    shape: tuple[int, int],
    ocr_axis_evidence: dict[str, object],
) -> tuple[int, int, int, int]:
    h, w = shape
    pos = middle_labeled_position(axis_binding)
    if pos is None:
        pos = w / 2
    x0, x1 = local_interval(
        pos,
        axis_binding_positions(axis_binding),
        w,
        min_frac=0.30,
        max_frac=0.56,
        min_px=260,
    )
    bounds = axis_binding.get("bounds", []) if isinstance(axis_binding, dict) else []
    if isinstance(bounds, list) and len(bounds) >= 2:
        y0_bound = int(round(float(bounds[0])))
        y1_bound = int(round(float(bounds[1])))
    else:
        y0_bound = int(h * 0.18)
        y1_bound = int(h * 0.82)
    label_y = nearest_ocr_cross_position(ocr_axis_evidence, "x_axis", float(pos))
    if label_y is None:
        label_y = float(y1_bound)
    edge_y = y0_bound if label_y < (y0_bound + y1_bound) / 2.0 else y1_bound
    top = int(min(label_y, edge_y) - max(42, h * 0.06))
    bottom = int(max(label_y, edge_y) + max(72, h * 0.10))
    min_height = max(150, int(h * 0.22))
    if bottom - top < min_height:
        mid = (top + bottom) / 2.0
        top = int(round(mid - min_height / 2.0))
        bottom = int(round(mid + min_height / 2.0))
    return clamp_rect((x0, top, x1, bottom), shape)

def y_axis_crop_rect(
    axis_binding: dict[str, object],
    shape: tuple[int, int],
    ocr_axis_evidence: dict[str, object],
) -> tuple[int, int, int, int]:
    h, w = shape
    pos = middle_labeled_position(axis_binding)
    if pos is None:
        pos = h / 2
    y0, y1 = local_interval(
        pos,
        axis_binding_positions(axis_binding),
        h,
        min_frac=0.30,
        max_frac=0.56,
        min_px=150,
    )
    bounds = axis_binding.get("bounds", []) if isinstance(axis_binding, dict) else []
    if isinstance(bounds, list) and len(bounds) >= 2:
        x0_bound = int(round(float(bounds[0])))
        x1_bound = int(round(float(bounds[1])))
    else:
        x0_bound = int(w * 0.18)
        x1_bound = int(w * 0.86)
    label_x = nearest_ocr_cross_position(ocr_axis_evidence, "y_axis", float(pos))
    if label_x is None:
        label_x = float(x0_bound)
    edge_x = x0_bound if label_x < (x0_bound + x1_bound) / 2.0 else x1_bound
    left = int(min(label_x, edge_x) - max(76, w * 0.08))
    right = int(max(label_x, edge_x) + max(132, w * 0.12))
    min_width = max(250, int(w * 0.30))
    if right - left < min_width:
        mid = (left + right) / 2.0
        left = int(round(mid - min_width / 2.0))
        right = int(round(mid + min_width / 2.0))
    return clamp_rect((left, y0, right, y1), shape)

def titled_panel(image: np.ndarray, title: str, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    out = np.full((height + 26, width, 3), 255, dtype=np.uint8)
    out[26:, :, :] = resized
    cv2.putText(out, title[:48], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    return out

def panel_title(source: str, axis: str, scores: dict[str, dict[str, object]]) -> str:
    score = scores.get(source, {})
    valid = "ok" if score.get("valid") else "bad"
    return (
        f"{SOURCE_LABELS.get(source, source)} {axis} "
        f"s={score.get('score', 0)} n={score.get('line_count', 0)}/{score.get('target_count', 0)} {valid}"
    )

def ordered_sources(candidates: dict[str, object]) -> list[str]:
    return sorted(candidates, key=lambda source: SOURCE_ORDER.get(source, 99))

def crop(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return image[y0 : y1 + 1, x0 : x1 + 1]

def make_priority_review_image(
    image: np.ndarray,
    candidate_overlays: dict[str, np.ndarray],
    candidate_bindings: dict[str, dict[str, object]],
    ocr_axis_evidence: dict[str, object],
    x_scores: dict[str, dict[str, object]],
    y_scores: dict[str, dict[str, object]],
) -> tuple[np.ndarray, dict[str, object]]:
    shape = image.shape[:2]
    ref_x = choose_reference_binding(candidate_bindings, "x_axis")
    ref_y = choose_reference_binding(candidate_bindings, "y_axis")
    x_rect = x_axis_crop_rect(ref_x, shape, ocr_axis_evidence)
    y_rect = y_axis_crop_rect(ref_y, shape, ocr_axis_evidence)
    rows = []
    panel_size = (460, 190)
    original_row = [
        titled_panel(crop(image, x_rect), "Original x-axis crop", panel_size),
        titled_panel(crop(image, y_rect), "Original y-axis crop", panel_size),
    ]
    rows.append(np.hstack(original_row))
    for source in ordered_sources(candidate_overlays):
        overlay = candidate_overlays[source]
        rows.append(
            np.hstack(
                [
                    titled_panel(crop(overlay, x_rect), panel_title(source, "x", x_scores), panel_size),
                    titled_panel(crop(overlay, y_rect), panel_title(source, "y", y_scores), panel_size),
                ]
            )
        )
    crop_meta = {
        "x_axis_crop_rect": list(map(int, x_rect)),
        "y_axis_crop_rect": list(map(int, y_rect)),
        "panel_size": list(panel_size),
        "layout": "rows: original, combined_mask, tick_supplement, semantic_guide; columns: x-axis crop, y-axis crop",
    }
    return np.vstack(rows), crop_meta

def tie_break_crop_rect(
    axis_key: str,
    orientation: str,
    candidate_bindings: dict[str, dict[str, object]],
    position_analysis: dict[str, object],
    shape: tuple[int, int],
    ocr_axis_evidence: dict[str, object],
) -> tuple[int, int, int, int]:
    h, w = shape
    sources = position_analysis.get("sources", [])
    if not isinstance(sources, list) or len(sources) < 2:
        sources = ["tick_supplement", "semantic_guide"]
    first_source = str(sources[0])
    second_source = str(sources[1])
    diffs = position_analysis.get("position_differences", [])
    changed = [
        item
        for item in diffs
        if isinstance(item, dict)
        and float(item.get("abs_delta", 0.0) or 0.0) > float(position_analysis.get("equivalent_tolerance", 2.5) or 2.5)
    ]
    target = max(changed or diffs, key=lambda item: float(item.get("abs_delta", 0.0) or 0.0)) if isinstance(diffs, list) and diffs else {}
    first = float(target.get(first_source, 0.0) or 0.0) if isinstance(target, dict) else 0.0
    second = float(target.get(second_source, first) or first) if isinstance(target, dict) else first
    center = (first + second) / 2.0
    ref_axis = choose_reference_binding(candidate_bindings, axis_key)
    positions = axis_binding_positions(ref_axis)
    bounds = ref_axis.get("bounds", []) if isinstance(ref_axis, dict) else []
    if orientation == "vertical":
        x0, x1 = local_interval(center, positions, w, min_frac=0.12, max_frac=0.26, min_px=170)
        if isinstance(bounds, list) and len(bounds) >= 2:
            y0_bound = int(round(float(bounds[0])))
            y1_bound = int(round(float(bounds[1])))
        else:
            y0_bound = int(h * 0.15)
            y1_bound = int(h * 0.85)
        label_y = nearest_ocr_cross_position(ocr_axis_evidence, axis_key, center)
        if label_y is None:
            label_y = float(y0_bound if axis_key == "x_axis" else y1_bound)
        edge_y = y0_bound if label_y < (y0_bound + y1_bound) / 2.0 else y1_bound
        top = int(min(label_y, edge_y) - max(70, h * 0.05))
        bottom = int(max(label_y, edge_y) + max(170, h * 0.12))
        return clamp_rect((x0, top, x1, bottom), shape)
    y0, y1 = local_interval(center, positions, h, min_frac=0.12, max_frac=0.26, min_px=150)
    if isinstance(bounds, list) and len(bounds) >= 2:
        x0_bound = int(round(float(bounds[0])))
        x1_bound = int(round(float(bounds[1])))
    else:
        x0_bound = int(w * 0.15)
        x1_bound = int(w * 0.85)
    label_x = nearest_ocr_cross_position(ocr_axis_evidence, axis_key, center)
    if label_x is None:
        label_x = float(x0_bound)
    edge_x = x0_bound if label_x < (x0_bound + x1_bound) / 2.0 else x1_bound
    left = int(min(label_x, edge_x) - max(90, w * 0.06))
    right = int(max(label_x, edge_x) + max(180, w * 0.12))
    return clamp_rect((left, y0, right, y1), shape)

def make_position_tie_review_image(
    image: np.ndarray,
    candidate_overlays: dict[str, np.ndarray],
    candidate_bindings: dict[str, dict[str, object]],
    ocr_axis_evidence: dict[str, object],
    axis_key: str,
    orientation: str,
    scores: dict[str, dict[str, object]],
    position_analysis: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    rect = tie_break_crop_rect(axis_key, orientation, candidate_bindings, position_analysis, image.shape[:2], ocr_axis_evidence)
    panel_size = (520, 260)
    axis_title = "x-axis vertical-grid tie-break" if axis_key == "x_axis" else "y-axis horizontal-grid tie-break"
    marked = crop(image, rect).copy()
    x0, y0, _, _ = rect
    sources = position_analysis.get("sources", [])
    if not isinstance(sources, list) or len(sources) < 2:
        sources = ["tick_supplement", "semantic_guide"]
    first_source = str(sources[0])
    second_source = str(sources[1])
    diffs = position_analysis.get("position_differences", [])
    changed = [
        item
        for item in diffs
        if isinstance(item, dict)
        and float(item.get("abs_delta", 0.0) or 0.0) > float(position_analysis.get("equivalent_tolerance", 2.5) or 2.5)
    ] if isinstance(diffs, list) else []
    trusted_positions: list[dict[str, object]] = []
    axis_binding = candidate_bindings.get("semantic_guide", {}).get(axis_key, {})
    bindings = axis_binding.get("tick_bindings", []) if isinstance(axis_binding, dict) else []
    for item in changed:
        try:
            index = int(item.get("index", -1))
            first = float(item.get(first_source, 0.0))
            second = float(item.get(second_source, 0.0))
        except (TypeError, ValueError):
            continue
        trusted = None
        if isinstance(bindings, list) and 0 <= index < len(bindings) and isinstance(bindings[index], dict):
            ocr = bindings[index].get("ocr", {})
            if isinstance(ocr, dict):
                try:
                    trusted = float(ocr.get("position", ocr.get("x" if axis_key == "x_axis" else "y", 0.0)))
                except (TypeError, ValueError):
                    trusted = None
        trusted_positions.append(
            {
                "index": index,
                first_source: round(first, 3),
                second_source: round(second, 3),
                "trusted_ocr_label_position": round(trusted, 3) if trusted is not None else None,
            }
        )
        first_color = SOURCE_COLORS.get(first_source, (0, 150, 255))
        second_color = SOURCE_COLORS.get(second_source, (0, 180, 0))
        if orientation == "vertical":
            first_x = int(round(first)) - x0
            second_x = int(round(second)) - x0
            cv2.line(marked, (first_x, 0), (first_x, marked.shape[0] - 1), first_color, 3, cv2.LINE_AA)
            cv2.line(marked, (second_x, 0), (second_x, marked.shape[0] - 1), second_color, 3, cv2.LINE_AA)
            if trusted is not None:
                tx = int(round(trusted)) - x0
                cv2.line(marked, (tx, 0), (tx, marked.shape[0] - 1), (255, 210, 0), 2, cv2.LINE_AA)
        else:
            first_y = int(round(first)) - y0
            second_y = int(round(second)) - y0
            cv2.line(marked, (0, first_y), (marked.shape[1] - 1, first_y), first_color, 3, cv2.LINE_AA)
            cv2.line(marked, (0, second_y), (marked.shape[1] - 1, second_y), second_color, 3, cv2.LINE_AA)
            if trusted is not None:
                ty = int(round(trusted)) - y0
                cv2.line(marked, (0, ty), (marked.shape[1] - 1, ty), (255, 210, 0), 2, cv2.LINE_AA)
    first_name = SOURCE_COLOR_NAMES.get(first_source, first_source)
    second_name = SOURCE_COLOR_NAMES.get(second_source, second_source)
    cv2.putText(marked, f"{first_source} {first_name}  {second_source} {second_name}  trusted label center cyan", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    rows = [
        titled_panel(marked, f"Original with local tie markers: {axis_title}", panel_size),
    ]
    for source in sources:
        if source in candidate_overlays:
            rows.append(titled_panel(crop(candidate_overlays[source], rect), panel_title(source, axis_key[0], scores), panel_size))
    meta = {
        "tie_break_crop_rect": list(map(int, rect)),
        "tie_break_axis": axis_key,
        "tie_break_orientation": orientation,
        "tie_break_panel_size": list(panel_size),
        "layout": "rows: original, tick_supplement, semantic_guide; focused on differing tied positions",
        "position_analysis": position_analysis,
        "trusted_positions": trusted_positions,
        "legend": {source: SOURCE_COLOR_NAMES.get(str(source), str(source)) for source in sources}
        | {"trusted_ocr_label_position": "cyan"},
    }
    return np.vstack(rows), meta

def apply_choice(
    source: str | None,
    orientation: str,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    fallback: np.ndarray,
) -> np.ndarray:
    if source not in candidates:
        return fallback
    horizontal, vertical = candidates[source]
    return vertical if orientation == "vertical" else horizontal

def arbitrate_priority_grids(
    image: np.ndarray,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    fused_axis_evidence: dict[str, object],
    current_horizontal: np.ndarray,
    current_vertical: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], np.ndarray | None]:
    candidate_bindings = {
        source: build_grid_label_bindings(horizontal, vertical, ocr_axis_evidence, mllm_result, fused_axis_evidence)
        for source, (horizontal, vertical) in candidates.items()
    }
    x_target = axis_target_count(mllm_result, "x_axis")
    y_target = axis_target_count(mllm_result, "y_axis")
    x_scores = {
        source: binding_quality(binding.get("x_axis", {}), x_target)
        for source, binding in candidate_bindings.items()
    }
    y_scores = {
        source: binding_quality(binding.get("y_axis", {}), y_target)
        for source, binding in candidate_bindings.items()
    }
    x_axis_type = (mllm_result.get("x_axis", {}) or {}).get("type") if isinstance(mllm_result, dict) and isinstance(mllm_result.get("x_axis", {}), dict) else None
    y_axis_type = (mllm_result.get("y_axis", {}) or {}).get("type") if isinstance(mllm_result, dict) and isinstance(mllm_result.get("y_axis", {}), dict) else None
    apply_native_numeric_axis_bonus(x_scores, x_axis_type)
    apply_native_numeric_axis_bonus(y_scores, y_axis_type)
    apply_physical_tick_axis_bonus(x_scores)
    apply_physical_tick_axis_bonus(y_scores)
    apply_recommended_avoid_penalty(x_scores, mllm_result, "vertical")
    apply_recommended_avoid_penalty(y_scores, mllm_result, "horizontal")
    x_choice, x_reason = choose_by_score(x_scores)
    y_choice, y_reason = choose_by_score(y_scores)
    x_choice, x_reason = prefer_complete_native_over_semantic(x_choice, x_reason, x_scores, x_axis_type)
    y_choice, y_reason = prefer_complete_native_over_semantic(y_choice, y_reason, y_scores, y_axis_type)
    x_choice, x_reason = prefer_physical_tick_over_semantic(x_choice, x_reason, x_scores, x_axis_type)
    y_choice, y_reason = prefer_physical_tick_over_semantic(y_choice, y_reason, y_scores, y_axis_type)
    x_tie_analysis = score_tie_position_analysis(x_scores, candidates, "vertical")
    y_tie_analysis = score_tie_position_analysis(y_scores, candidates, "horizontal")
    if x_choice is not None and x_tie_analysis and x_tie_analysis.get("needs_mllm"):
        x_choice = None
        x_reason = f"{x_tie_analysis.get('score_reason', 'score_tie')}_position_diff_needs_mllm"
    if y_choice is not None and y_tie_analysis and y_tie_analysis.get("needs_mllm"):
        y_choice = None
        y_reason = f"{y_tie_analysis.get('score_reason', 'score_tie')}_position_diff_needs_mllm"
    if (
        x_choice is not None
        and x_tie_analysis
        and not x_tie_analysis.get("needs_mllm")
        and x_tie_analysis.get("score_reason") == "score_near_tie"
    ):
        x_choice = fallback_equivalent_score_tie(x_scores)
        x_reason = "score_near_tie_positions_equivalent"
    if (
        y_choice is not None
        and y_tie_analysis
        and not y_tie_analysis.get("needs_mllm")
        and y_tie_analysis.get("score_reason") == "score_near_tie"
    ):
        y_choice = fallback_equivalent_score_tie(y_scores)
        y_reason = "score_near_tie_positions_equivalent"
    if x_choice is None and x_tie_analysis and not x_tie_analysis.get("needs_mllm"):
        x_choice = fallback_equivalent_score_tie(x_scores)
        x_reason = "score_tie_positions_equivalent"
    if y_choice is None and y_tie_analysis and not y_tie_analysis.get("needs_mllm"):
        y_choice = fallback_equivalent_score_tie(y_scores)
        y_reason = "score_tie_positions_equivalent"
    candidate_overlays = {
        source: draw_grid_label_overlay(image, horizontal, vertical, candidate_bindings[source])
        for source, (horizontal, vertical) in candidates.items()
    }
    review_image, crop_meta = make_priority_review_image(
        image,
        candidate_overlays,
        candidate_bindings,
        ocr_axis_evidence,
        x_scores,
        y_scores,
    )
    tie_break_meta: dict[str, object] | None = None
    if x_choice is None and x_tie_analysis and x_tie_analysis.get("needs_mllm"):
        review_image, tie_break_meta = make_position_tie_review_image(
            image,
            candidate_overlays,
            candidate_bindings,
            ocr_axis_evidence,
            "x_axis",
            "vertical",
            x_scores,
            x_tie_analysis,
        )
        x_reason = "needs_mllm_position_diff"
    elif y_choice is None and y_tie_analysis and y_tie_analysis.get("needs_mllm"):
        review_image, tie_break_meta = make_position_tie_review_image(
            image,
            candidate_overlays,
            candidate_bindings,
            ocr_axis_evidence,
            "y_axis",
            "horizontal",
            y_scores,
            y_tie_analysis,
        )
        y_reason = "needs_mllm_position_diff"

    decision: dict[str, object] = {
        "enabled": True,
        "x_axis_vertical_grid_choice": x_choice,
        "x_axis_reason": f"score_prefill:{x_reason}",
        "y_axis_horizontal_grid_choice": y_choice,
        "y_axis_reason": f"score_prefill:{y_reason}",
        "chart_type": mllm_result.get("chart_type") if isinstance(mllm_result, dict) else None,
        "x_axis_type": x_axis_type,
        "y_axis_type": y_axis_type,
        "x_scores": x_scores,
        "y_scores": y_scores,
        "mllm_used": False,
        "review_crop": crop_meta,
        "selection_policy": "score_first_mllm_when_needed",
    }
    if x_tie_analysis:
        decision["x_axis_position_tie_analysis"] = x_tie_analysis
    if y_tie_analysis:
        decision["y_axis_position_tie_analysis"] = y_tie_analysis
    if tie_break_meta:
        decision["tie_break_review"] = tie_break_meta
    needs_mllm = x_choice is None or y_choice is None
    if needs_mllm:
        mllm_decision = run_mllm_grid_arbitration(review_image, decision, args)
    else:
        mllm_decision = {"enabled": False, "error": "not_needed_clear_score_winner"}
    decision["mllm"] = mllm_decision
    if needs_mllm and not mllm_decision.get("error"):
        decision["mllm_used"] = True
        proposed = str(mllm_decision.get("x_axis_vertical_grid_choice", ""))
        if x_choice is None and proposed in candidates and x_scores.get(proposed, {}).get("valid"):
            x_choice = proposed
            decision["x_axis_vertical_grid_choice"] = x_choice
            decision["x_axis_reason"] = "mllm_choice"
        elif x_choice is None:
            decision["x_axis_mllm_rejected_choice"] = proposed or None
        proposed = str(mllm_decision.get("y_axis_horizontal_grid_choice", ""))
        if y_choice is None and proposed in candidates and y_scores.get(proposed, {}).get("valid"):
            y_choice = proposed
            decision["y_axis_horizontal_grid_choice"] = y_choice
            decision["y_axis_reason"] = "mllm_choice"
        elif y_choice is None:
            decision["y_axis_mllm_rejected_choice"] = proposed or None
    elif needs_mllm:
        decision["mllm_fallback_reason"] = mllm_decision.get("error")

    if x_choice is None:
        x_choice = fallback_by_score(x_scores)
        decision["x_axis_vertical_grid_choice"] = x_choice
        decision["x_axis_reason"] = "score_fallback"
    if y_choice is None:
        y_choice = fallback_by_score(y_scores)
        decision["y_axis_horizontal_grid_choice"] = y_choice
        decision["y_axis_reason"] = "score_fallback"

    selected_vertical = apply_choice(x_choice, "vertical", candidates, current_vertical)
    selected_horizontal = apply_choice(y_choice, "horizontal", candidates, current_horizontal)
    return selected_horizontal, selected_vertical, decision, review_image
