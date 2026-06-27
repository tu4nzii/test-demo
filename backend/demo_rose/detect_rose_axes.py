"""
Rose / polar-area chart axis label detection.

Wraps the radar detection pipeline (detect_radar_axes.py) with rose-specific
adaptations:
  R1 — Wedge-value filter: pure numbers inside sectors are not axis labels.
  R2 — Sequential letter mode: A,B,C,… labels CCW from 90° (right side).
  R3 — "Label X" prefix detection: when OCR reads "Uabel B" → "Label B".
  R4 — Fragmented-label merge: "After"+"Sales"+"Support" → "After-Sales Support".

Public API:
    detect_rose(image_path, center, outer_radius, use_llm=True) -> (labels, debug)
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.demo_radar.detect_radar_axes import (
    detect as detect_radar,
    detect_axes,
    init_ocr, run_ocr_on_image,
    compact_text, angle_distance, text_has_alpha, text_has_digit,
    in_label_band, in_count_band, candidate_quality,
    llm_refine_labels, crop_axis_region,
)


# ---------------------------------------------------------------------------
# Rose-specific n_axes estimation
# ---------------------------------------------------------------------------

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def _rose_month_labels(n_axes: int) -> Dict[int, str]:
    """Generate month labels CCW from 90° (right side)."""
    if n_axes != 12:
        return {}
    step = 360.0 / 12
    return {
        int(round((90.0 - i * step) % 360)): MONTHS[i]
        for i in range(12)
    }


def _count_label_x_from_dets(detections) -> int:
    """Count unique Label X suffixes with gap inference."""
    suffixes = set()
    for d in detections:
        m = re.match(r"^(Label|Uabel|Jabcl|Labcl)\s+([A-Z])$", d["text"], re.IGNORECASE)
        if m:
            suffixes.add(m.group(2).upper())
    bare = sum(1 for d in detections
               if re.match(r"^(Label|Uabel|Jabcl|Labcl)$", d["text"].strip(), re.IGNORECASE))
    if suffixes and bare:
        return len(suffixes) + bare
    if suffixes:
        letters = sorted(suffixes)
        return max(len(suffixes), ord(letters[-1]) - ord(letters[0]) + 1)
    return 0


# ---------------------------------------------------------------------------
# Rose-specific adaptations
# ---------------------------------------------------------------------------

def _is_wedge_value(text: str) -> bool:
    """Check if text looks like a numeric value inside a rose wedge (not a label)."""
    ctext = compact_text(text)
    return ctext.isdigit() and len(ctext) <= 2 and not text_has_alpha(text)


def _looks_like_label_prefix(text: str) -> bool:
    """Check if text looks like a garbled 'Label X' pattern."""
    ctext = compact_text(text)
    # "Label" → "Uabel", "Jabcl", "Labcl" — first letter may be wrong
    prefixes = ("label", "uabel", "jabcl", "labcl", "labe", "abel")
    return any(ctext.startswith(p) for p in prefixes) or (
        len(ctext) >= 4 and SequenceMatcher(None, ctext[:5], "label").ratio() >= 0.7
    )


def _clean_label_x(text: str) -> str:
    """Clean a garbled 'Label X' text back to standard form."""
    original = text.strip()
    ctext = compact_text(text)
    # Extract the suffix letter/number
    suffix = ""
    for ch in reversed(original):
        if ch.isalpha() and ch.isupper():
            suffix = ch
            break
    if suffix and len(suffix) == 1:
        return f"Label {suffix}"
    # Try to infer from compact text
    if len(ctext) > 5 and ctext[5:].isalpha():
        return f"Label {ctext[5:].upper()}"
    return original


def _rose_sequential_letters(n_axes: int) -> Dict[int, str]:
    """Generate A,B,C,… labels CCW starting from 90° (right side)."""
    if n_axes < 3 or n_axes > 26:
        return {}
    letters = [chr(ord("A") + i) for i in range(n_axes)]
    step = 360.0 / n_axes
    return {
        int(round((90.0 - i * step) % 360)): letters[i]
        for i in range(n_axes)
    }


def _rose_sequential_label_x(n_axes: int) -> Dict[int, str]:
    """Generate 'Label A', 'Label B', … CCW starting from 90°."""
    if n_axes < 3 or n_axes > 26:
        return {}
    labels = [f"Label {chr(ord('A') + i)}" for i in range(n_axes)]
    step = 360.0 / n_axes
    return {
        int(round((90.0 - i * step) % 360)): labels[i]
        for i in range(n_axes)
    }


def _fragmented_label_merge(detections: List[Dict], outer_radius: float) -> List[Dict]:
    """Merge nearby OCR text fragments that belong to the same label.

    Rose charts often have multi-word labels split across adjacent OCR boxes
    (e.g. 'After' + 'Sales' + 'Support' → 'After-Sales Support').
    """
    if len(detections) < 2:
        return detections

    merged = []
    used = set()
    # Sort by angle, then merge nearby text pieces
    sorted_dets = sorted(detections, key=lambda d: (d["angle"], d["distance"]))

    for i, det in enumerate(sorted_dets):
        if i in used:
            continue
        group = [det]
        used.add(i)
        # Look ahead for nearby fragments
        for j in range(i + 1, len(sorted_dets)):
            if j in used:
                continue
            other = sorted_dets[j]
            # Close in angle (< 8°) AND similar distance (< 30px)
            if (angle_distance(det["angle"], other["angle"]) < 8.0
                    and abs(det["distance"] - other["distance"]) < 30
                    and text_has_alpha(det["text"])
                    and text_has_alpha(other["text"])):
                group.append(other)
                used.add(j)
        if len(group) > 1:
            # Merge texts with spaces
            merged_text = " ".join(d["text"] for d in group)
            # Use the first detection's metadata, update text and confidence
            new_det = dict(group[0])
            new_det["text"] = merged_text
            new_det["confidence"] = float(np.mean([d["confidence"] for d in group]))
            merged.append(new_det)
        else:
            merged.append(det)
    return merged


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def _try_synthetic_rose_mode(detections, outer_radius) -> Dict[int, str] | None:
    """Try direct OCR-to-grid assignment for clean synthetic rose charts.

    Returns axis_labels dict if successful, None if chart doesn't qualify.
    """
    # Collect alpha detections in label band, exclude tick values
    alpha_dets = [
        d for d in detections
        if text_has_alpha(d["text"])
        and in_label_band(d, outer_radius)
        and len(compact_text(d["text"])) >= 2  # exclude single chars
    ]
    if len(alpha_dets) < 3:
        return None

    # Sort by angle and deduplicate close-angle detections
    alpha_dets = sorted(alpha_dets, key=lambda d: d["angle"])
    unique = []
    for d in alpha_dets:
        if unique and angle_distance(d["angle"], unique[-1]["angle"]) < 5:
            # Keep higher confidence
            if d["confidence"] > unique[-1]["confidence"]:
                unique[-1] = d
        else:
            unique.append(d)

    if len(unique) < 3:
        return None

    # Compute angular gaps to infer n_axes
    angles = [d["angle"] for d in unique]
    gaps = []
    for i in range(len(angles)):
        gap = angle_distance(angles[i], angles[(i + 1) % len(angles)])
        if gap > 3:
            gaps.append(gap)
    if not gaps:
        return None

    median_gap = float(np.median(gaps))
    if median_gap < 8 or median_gap > 180:
        return None
    n_axes = max(3, min(36, round(360.0 / median_gap)))

    # Build uniform grid starting from the first detected angle
    step = 360.0 / n_axes
    start_angle = unique[0]["angle"]

    axis_labels = {}
    used = set()
    for i in range(n_axes):
        grid_angle = (start_angle + i * step) % 360
        # Find the closest unused detection
        best_idx = None
        best_dist = step * 0.6  # max 60% of step away
        for idx, d in enumerate(unique):
            if idx in used:
                continue
            dist = angle_distance(d["angle"], grid_angle)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            axis_labels[int(round(grid_angle))] = unique[best_idx]["text"]

    if len(axis_labels) >= 3:
        return axis_labels
    return None

def detect_rose(
    image_path, center, outer_radius, use_llm=True, reader=None
) -> Tuple[Dict[int, str], Dict]:
    """Detect axis labels on a rose/polar-area chart.

    Args:
        reader: Optional pre-initialized EasyOCR reader. If None, one is
                created automatically.  Reuse across charts for speed.
    """
    # Step 1: run radar detection ONCE, get all data
    if reader is None:
        reader = init_ocr()
    image_path = Path(image_path)
    # Always pass use_llm=False to radar detect_axes — the rose module
    # handles its own LLM refinement at the end.  The radar LLM calls
    # (llm_count_axes, llm_refine_labels) use radar-specific contact
    # sheets that are not appropriate for rose/polar-area charts.
    axis_labels, debug, detections = detect_axes(
        reader, image_path, center, outer_radius, use_llm=False
    )

    # ── Synthetic rose mode: direct OCR assignment ──
    # When the chart has clean, simple labels at uniform angular spacing
    # (typical of synthetic/generated rose charts), bypass the complex
    # radar binding logic and assign OCR detections directly to a uniform
    # angular grid inferred from detection gaps.
    synthetic_result = _try_synthetic_rose_mode(detections, outer_radius)
    if synthetic_result is not None:
        debug = {"n_final": len(synthetic_result),
                 "n_source": "synthetic_rose_direct",
                 "synthetic_rose_mode": True}
        # Run LLM refinement on the synthetic result when enabled.
        # Synthetic charts have accurate grids but occasional OCR
        # misreads (e.g. "DI"→"Dl", "VVUY"→"WUY") that LLM can fix.
        if use_llm:
            axes_angles = sorted(synthetic_result.keys())
            image = cv2.imread(str(image_path))
            if image is not None:
                try:
                    synthetic_result = llm_refine_labels(
                        image, image_path, center, outer_radius,
                        synthetic_result, axes_angles
                    )
                    debug["llm_refined"] = True
                except Exception:
                    pass
        return synthetic_result, debug

    # ── R0: Insufficient OCR text — check early before rose adaptations ──
    # Use n_alpha_band from the radar debug (count-band alpha, pre-crop-OCR).
    # If even the full-image OCR found almost nothing, the chart is unreadable.
    n_alpha = debug.get("n_alpha_band", 0)
    n_numeric = debug.get("n_numeric_band", 0)
    has_product = any(
        re.match(r"^(Product|Series|Item)\s+([A-Z])$", d["text"].strip(), re.IGNORECASE)
        for d in detections
    )
    has_label_x = any(
        re.match(r"^(Label|Uabel|Jabcl|Labcl)\s+([A-Z])$", d["text"], re.IGNORECASE)
        for d in detections
    )
    months_list = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    has_month = sum(1 for d in detections
                    if d["text"].strip()[:3] in months_list) >= 3
    if (n_alpha + n_numeric) < 3 and not (has_product or has_label_x or has_month):
        debug["fallback"] = True
        debug["fallback_reason"] = f"insufficient_ocr_text(n_alpha={n_alpha},n_numeric={n_numeric})"
        return {}, debug

    # Non-canonical axis count in numeric-only mode — the grid is likely
    # denser than what OCR can read (e.g. 24-angle chart where only major
    # 30° ticks are detected, giving a non-integer step).
    CANONICAL = {3, 4, 5, 6, 8, 10, 12, 16, 20, 24}
    if (debug.get("numeric_axis_mode")
            and debug.get("n_alpha_band", 0) == 0
            and debug.get("n_final", 0) not in CANONICAL):
        debug["fallback"] = True
        debug["fallback_reason"] = f"non_canonical_numeric_grid(n={debug.get('n_final')})"
        return {}, debug

    # Step 2: independently estimate n_axes for rose charts from the SAME OCR
    rose_n = _estimate_rose_axis_count_from_dets(detections, outer_radius)

    if rose_n >= 3:
        n_axes = rose_n
    else:
        n_axes = debug.get("n_final", len(axis_labels))

    labels = list(axis_labels.values()) if axis_labels else []

    # ── R1: Wedge-value filter ──
    wedge_count = sum(1 for v in labels if v != "?" and _is_wedge_value(v))
    if wedge_count >= max(len(labels) * 0.4, 2) and n_axes >= 3:
        # Only trust sequential letters if we have independent evidence
        # that the labels ARE sequential (Product X or Label X patterns).
        # Otherwise the chart likely has complex text labels and we should
        # fallback rather than generating wrong A,B,C,…
        has_product_pattern = any(
            re.match(r"^(Product|Series|Item)\s+([A-Z])$", d["text"].strip(), re.IGNORECASE)
            for d in detections
        )
        has_label_x_pattern = any(
            re.match(r"^(Label|Uabel|Jabcl|Labcl)\s+([A-Z])$", d["text"], re.IGNORECASE)
            for d in detections
        )
        if not (has_product_pattern or has_label_x_pattern):
            debug["fallback"] = True
            debug["fallback_reason"] = "rose_wedge_no_sequential_evidence"
            return {}, debug
        axis_labels = _rose_sequential_letters(n_axes)
        debug["rose_wedge_filter"] = True
        debug["rose_sequential"] = True
        return axis_labels, debug

    # ── R2: Single-letter or all-"?" → sequential (only with evidence) ──
    single_count = sum(1 for v in labels if len(v.strip()) == 1 and v.strip().isalpha())
    question_count = sum(1 for v in labels if v == "?")
    if ((single_count >= max(len(labels), 1) * 0.5 and single_count >= 2)
            or question_count >= max(len(labels), 1) * 0.5):
        if n_axes >= 3:
            # Only use sequential letters if we have Product/Label-X evidence
            has_product_pattern = any(
                re.match(r"^(Product|Series|Item)\s+([A-Z])$", d["text"].strip(), re.IGNORECASE)
                for d in detections
            )
            has_label_x_pattern = any(
                re.match(r"^(Label|Uabel|Jabcl|Labcl)\s+([A-Z])$", d["text"], re.IGNORECASE)
                for d in detections
            )
            if has_product_pattern or has_label_x_pattern:
                axis_labels = _rose_sequential_letters(n_axes)
                debug["rose_sequential"] = True
                return axis_labels, debug
            # Otherwise fallback — can't determine real labels
            debug["fallback"] = True
            debug["fallback_reason"] = "rose_sequential_no_evidence"
            return {}, debug

    # ── R3: Label-X prefix detection ──
    # When OCR reads "Uabel B", "Labcl G", etc., the chart uses
    # sequential "Label A".."Label L" labels.  Generate the full grid.
    label_x_count = sum(1 for v in labels if v != "?" and _looks_like_label_prefix(v))
    if label_x_count >= max(len(labels), 1) * 0.3 and label_x_count >= 2:
        # Force n_axes from the Label X pattern if available
        label_n = _count_label_x_from_dets(detections)
        if label_n >= 3:
            axis_labels = _rose_sequential_label_x(label_n)
            debug["rose_label_x"] = True
            return axis_labels, debug

    # ── R4: Month label detection ──
    # If OCR detected ≥3 month abbreviations (Jan-Dec), the chart uses
    # month labels.  Generate the full 12-month sequence CCW from 90°.
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_detections = [
        d for d in detections
        if d["text"].strip()[:3] in MONTHS
    ]
    if len(month_detections) >= 3:
        # Verify that detected months appear at roughly the right angles
        month_angles_correct = 0
        for d in month_detections:
            month_idx = MONTHS.index(d["text"].strip()[:3])
            expected_angle = (90.0 - month_idx * 30.0) % 360
            if angle_distance(d["angle"], expected_angle) < 45:
                month_angles_correct += 1
        if month_angles_correct >= 3:
            axis_labels = _rose_month_labels(12)
            debug["rose_month_mode"] = True
            return axis_labels, debug

    # ── LLM refinement: per-axis label verification ──
    # Skip for synthetic/generated modes (already correct without LLM).
    is_generated = debug.get("synthetic_rose_mode") or debug.get("rose_sequential") or debug.get("rose_label_x") or debug.get("rose_month_mode")
    if use_llm and axis_labels and not is_generated and not debug.get("fallback"):
        try:
            step = 360.0 / max(len(axis_labels), 1)
            axes_angles = sorted(axis_labels.keys())
            image = cv2.imread(str(image_path))
            if image is not None:
                axis_labels = llm_refine_labels(
                    image, image_path, center, outer_radius, axis_labels, axes_angles
                )
                debug["llm_refined"] = True
        except Exception:
            pass

    return axis_labels, debug


def _estimate_rose_axis_count_from_dets(dets, outer_radius) -> int:
    """Estimate axis count from already-computed OCR detections."""
    # Method 1: "Product X" patterns
    product_suffixes = set()
    for d in dets:
        text = d["text"].strip()
        m = re.match(r"^(Product|Series|Item)\s+([A-Z])$", text, re.IGNORECASE)
        if m:
            product_suffixes.add(m.group(2).upper())
    # Also count bare "Product"/"Series"/"Item" as a likely missed suffix
    bare_product_count = sum(
        1 for d in dets
        if re.match(r"^(Product|Series|Item)$", d["text"].strip(), re.IGNORECASE)
    )
    if product_suffixes and bare_product_count:
        # Infer total = detected suffixes + bare items
        product_n = len(product_suffixes) + bare_product_count
    elif product_suffixes:
        # Check for gaps in the letter sequence (A,B,D,E → gap at C)
        letters = sorted(product_suffixes)
        expected_len = ord(letters[-1]) - ord(letters[0]) + 1
        if expected_len > len(letters):
            product_n = expected_len
        else:
            product_n = len(product_suffixes)
    else:
        product_n = 0

    # Method 2: single letters
    single_letters = set()
    for d in dets:
        ctext = compact_text(d["text"])
        if len(ctext) == 1 and d["text"].isalpha():
            single_letters.add(ctext.upper())

    # Method 3: "Label X" patterns with gap inference
    label_x_suffixes = set()
    for d in dets:
        m = re.match(r"^(Label|Uabel|Jabcl|Labcl)\s+([A-Z])$", d["text"], re.IGNORECASE)
        if m:
            label_x_suffixes.add(m.group(2).upper())
    bare_label_count = sum(
        1 for d in dets
        if re.match(r"^(Label|Uabel|Jabcl|Labcl)$", d["text"].strip(), re.IGNORECASE)
    )
    if label_x_suffixes and bare_label_count:
        label_n = len(label_x_suffixes) + bare_label_count
    elif label_x_suffixes:
        letters = sorted(label_x_suffixes)
        expected_len = ord(letters[-1]) - ord(letters[0]) + 1
        if expected_len > len(label_x_suffixes):
            label_n = expected_len
        else:
            label_n = len(label_x_suffixes)
    else:
        label_n = 0

    # Method 4: alpha detections in label band
    alpha_dets = [d for d in dets if in_label_band(d, outer_radius) and text_has_alpha(d["text"])]
    alpha_angles = sorted(set(round(d["angle"] / 5.0) * 5.0 for d in alpha_dets))

    candidates = []
    if product_n >= 3:
        candidates.append(product_n)
    if len(single_letters) >= 3:
        candidates.append(len(single_letters))
    if label_n >= 3:
        candidates.append(label_n)
    if len(alpha_angles) >= 3:
        candidates.append(len(alpha_angles))

    if candidates:
        return int(round(np.median(candidates)))
    return 0
