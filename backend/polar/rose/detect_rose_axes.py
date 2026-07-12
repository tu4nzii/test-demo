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

import base64
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import requests

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.polar.radar.detect_radar_axes import (
    detect as detect_radar,
    detect_axes,
    init_ocr, run_ocr_on_image,
    compact_text, angle_distance, text_has_alpha, text_has_digit,
    in_label_band, in_count_band, candidate_quality,
    llm_refine_labels, crop_axis_region,
    title_like, boilerplate_like, metadata_like_text,
    LLM_URL, LLM_HEADERS, LLM_MODEL,
)
from backend.polar.llm_debug import save_llm_error_response


# ---------------------------------------------------------------------------
# Rose-specific n_axes estimation
# ---------------------------------------------------------------------------

CANONICAL_ROSE_AXES = {3, 4, 5, 6, 8, 9, 10, 12, 16, 20, 24}


def _is_title_or_metadata(det: Dict, image_shape: Tuple[int, int, int],
                          center: Tuple[float, float], outer_radius: float) -> bool:
    return (
        title_like(det, image_shape, center, outer_radius)
        or boilerplate_like(det, image_shape, center, outer_radius)
        or metadata_like_text(det["text"])
    )


def _fallback_debug(reason: str, **extra) -> Dict:
    debug = {"fallback": True, "fallback_reason": reason}
    debug.update(extra)
    return debug


def _usable_ocr_candidates(
    detections: List[Dict],
    image_shape: Tuple[int, int, int],
    center: Tuple[float, float],
    outer_radius: float,
) -> List[Dict]:
    candidates = []
    for det in detections:
        if _is_title_or_metadata(det, image_shape, center, outer_radius):
            continue
        if not (in_label_band(det, outer_radius) or in_count_band(det, outer_radius)):
            continue
        text = compact_text(det["text"])
        if len(text) < 1:
            continue
        if not (text_has_alpha(det["text"]) or text_has_digit(det["text"])):
            continue
        candidates.append(det)
    return candidates


def _ocr_quality_fallback_reason(candidates: List[Dict]) -> Optional[str]:
    if len(candidates) < 3:
        return f"insufficient_ocr_candidates(n={len(candidates)})"
    confidences = sorted((float(d.get("confidence", 0.0)) for d in candidates), reverse=True)
    top3_mean = float(np.mean(confidences[:3])) if len(confidences) >= 3 else 0.0
    max_conf = confidences[0] if confidences else 0.0
    if max_conf < 0.45 or top3_mean < 0.40:
        return f"low_ocr_confidence(max={max_conf:.2f},top3_mean={top3_mean:.2f})"
    return None


def _axis_label_quality_fallback_reason(axis_labels: Dict[int, str]) -> Optional[str]:
    if not axis_labels:
        return "no_axis_labels"

    values = [str(v or "").strip() for v in axis_labels.values()]
    total = len(values)
    unknown_count = sum(1 for v in values if v == "?" or not compact_text(v))
    long_fragment_count = sum(
        1 for v in values
        if len(compact_text(v)) > 18 or len(v.split()) >= 4 or any(ch in v for ch in "$()")
    )

    if unknown_count > 0:
        return f"unreliable_axis_labels(unknown={unknown_count}/{total})"
    if long_fragment_count >= max(2, int(total * 0.2)):
        return f"unreliable_axis_labels(long_fragments={long_fragment_count}/{total})"
    return None


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _parse_llm_axis_labels(answer: str) -> Dict[int, str]:
    payload = _extract_json_object(answer)
    if not payload:
        return {}
    axes = payload.get("axes")
    if not isinstance(axes, list):
        return {}

    labels = {}
    for item in axes:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label or label == "?":
            continue
        try:
            angle = float(item.get("angle"))
        except (TypeError, ValueError):
            continue
        labels[int(round(angle)) % 360] = label

    if not (3 <= len(labels) <= 32):
        return {}
    normalized = [compact_text(label) for label in labels.values()]
    label_x_letters = []
    for value in normalized:
        match = re.fullmatch(r"label([a-z])", value)
        if match:
            label_x_letters.append(match.group(1).upper())
    if len(label_x_letters) == len(labels):
        letters = sorted(label_x_letters)
        expected = [chr(ord("A") + i) for i in range(len(letters))]
        if letters == expected:
            return _rose_sequential_label_x(len(letters))

    single_letters = [value.upper() for value in normalized if len(value) == 1 and value.isalpha()]
    if len(single_letters) == len(labels):
        letters = sorted(single_letters)
        expected = [chr(ord("A") + i) for i in range(len(letters))]
        if letters == expected:
            return _rose_sequential_letters(len(letters))

    return labels


def _is_complete_sequence(labels: Dict[int, str]) -> bool:
    normalized = [compact_text(label) for label in labels.values()]
    label_x_letters = []
    for value in normalized:
        match = re.fullmatch(r"label([a-z])", value)
        if match:
            label_x_letters.append(match.group(1).upper())
    if len(label_x_letters) == len(labels):
        expected = [chr(ord("A") + i) for i in range(len(label_x_letters))]
        return sorted(label_x_letters) == expected

    single_letters = [value.upper() for value in normalized if len(value) == 1 and value.isalpha()]
    if len(single_letters) == len(labels):
        expected = [chr(ord("A") + i) for i in range(len(single_letters))]
        return sorted(single_letters) == expected
    return False


def _texts_match_loose(left: str, right: str) -> bool:
    lval = compact_text(left)
    rval = compact_text(right)
    if not lval or not rval:
        return False
    return lval == rval or (len(lval) >= 4 and lval in rval) or (len(rval) >= 4 and rval in lval)


def _llm_salvage_reject_reason(
    labels: Dict[int, str],
    detections: Optional[List[Dict]],
    image_shape: Optional[Tuple[int, int, int]],
    center: Tuple[float, float] | None,
    outer_radius: float | None,
    reason: str,
) -> Optional[str]:
    if not labels:
        return "llm_no_labels"
    if reason.startswith("no_axis_line_evidence"):
        return "llm_rejected_no_axis_line_evidence"
    if reason.startswith("non_canonical_numeric_grid"):
        return "llm_rejected_non_canonical_numeric_grid"
    if all(compact_text(label).isdigit() for label in labels.values()):
        return "llm_rejected_numeric_only_salvage"
    if _is_complete_sequence(labels):
        return None

    if not detections or image_shape is None or center is None or outer_radius is None:
        return "llm_rejected_no_ocr_validation"

    agreements = 0
    conflicts = 0
    for det in _usable_ocr_candidates(detections, image_shape, center, outer_radius):
        if float(det.get("confidence", 0.0)) < 0.65:
            continue
        matched_angle = None
        for angle, label in labels.items():
            if _texts_match_loose(det["text"], label):
                matched_angle = angle
                break
        if matched_angle is None:
            continue
        if angle_distance(det["angle"], matched_angle) <= 45:
            agreements += 1
        else:
            conflicts += 1

    if conflicts and agreements == 0:
        return f"llm_rejected_ocr_angle_conflict(conflicts={conflicts})"
    if agreements == 0:
        return "llm_rejected_no_ocr_agreement"
    return None


def _llm_salvage_rose_axes(image_path: Path, reason: str) -> Tuple[Dict[int, str], Dict]:
    try:
        with open(image_path, "rb") as file:
            b64 = base64.b64encode(file.read()).decode("utf-8")
        prompt = (
            "You are reading the axis/category labels around a rose chart or polar-area chart. "
            "Return only valid JSON in this exact schema: "
            "{\"axes\":[{\"angle\":90,\"label\":\"example\"}]}. "
            "Use visual position angles in degrees: 0 is top/12 o'clock, 90 is right/3 o'clock, "
            "180 is bottom/6 o'clock, and 270 is left/9 o'clock. "
            "Include every outer category/axis label exactly once. "
            "Exclude titles, subtitles, legends, data values, radial tick values, and annotations. "
            "If labels are arranged uniformly, still report the visual angle for each label."
        )
        payload = {
            "model": LLM_MODEL,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        resp = requests.post(LLM_URL, headers=LLM_HEADERS, json=payload, timeout=45)
        if not resp.ok:
            body_path = save_llm_error_response(resp, "rose_axis_salvage")
            return {}, {"llm_salvage_error": f"http_{resp.status_code}", "llm_error_body_path": body_path}
        payload = resp.json()
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            return {}, {"llm_salvage_error": "missing_choices", "llm_answer": str(payload)[:500]}
        answer = choices[0]["message"]["content"].strip()
        labels = _parse_llm_axis_labels(answer)
        if not labels:
            return {}, {"llm_salvage_error": "parse_failed", "llm_answer": answer[:500]}
        quality_reason = _axis_label_quality_fallback_reason(labels)
        if quality_reason:
            return {}, {
                "llm_salvage_error": quality_reason,
                "llm_answer": answer[:500],
            }
        return labels, {
            "llm_salvaged": True,
            "llm_salvage_reason": reason,
            "llm_answer": answer[:500],
            "n_final": len(labels),
            "n_source": "llm_rose_axes",
        }
    except Exception as exc:
        return {}, {"llm_salvage_error": type(exc).__name__}


def _fallback_or_llm_salvage(
    image_path: Path,
    debug: Dict,
    reason: str,
    use_llm: bool,
    detections: Optional[List[Dict]] = None,
    image_shape: Optional[Tuple[int, int, int]] = None,
    center: Tuple[float, float] | None = None,
    outer_radius: float | None = None,
) -> Tuple[Dict[int, str], Dict]:
    if use_llm:
        labels, llm_debug = _llm_salvage_rose_axes(image_path, reason)
        debug.update(llm_debug)
        if labels:
            reject_reason = _llm_salvage_reject_reason(
                labels, detections, image_shape, center, outer_radius, reason
            )
            if reject_reason:
                debug["llm_salvage_rejected"] = True
                debug["llm_salvage_reject_reason"] = reject_reason
            else:
                debug["fallback"] = False
                debug["fallback_reason"] = ""
                return labels, debug
    debug["fallback"] = True
    debug["fallback_reason"] = reason
    return {}, debug


def _count_radial_axis_lines(
    image: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
) -> int:
    if image is None or image.size == 0 or outer_radius <= 0:
        return 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    min_len = max(24, int(outer_radius * 0.22))
    max_gap = max(8, int(outer_radius * 0.05))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=max(25, int(outer_radius * 0.12)),
        minLineLength=min_len, maxLineGap=max_gap
    )
    if lines is None:
        return 0

    cx, cy = center
    angle_bins = set()
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_len:
            continue
        denom = max(length, 1.0)
        center_dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / denom
        if center_dist > max(10.0, outer_radius * 0.08):
            continue
        mid_dist = float(np.hypot((x1 + x2) / 2.0 - cx, (y1 + y2) / 2.0 - cy))
        if mid_dist > outer_radius * 1.15:
            continue
        angle = (np.degrees(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0
        angle_bins.add(int(round(angle / 10.0) * 10) % 180)
    return len(angle_bins)


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


def _count_product_from_dets(detections) -> int:
    suffixes = set()
    for d in detections:
        m = re.match(r"^(Product|Series|Item)\s+([A-Z])$", d["text"].strip(), re.IGNORECASE)
        if m:
            suffixes.add(m.group(2).upper())
    bare = sum(
        1 for d in detections
        if re.match(r"^(Product|Series|Item)$", d["text"].strip(), re.IGNORECASE)
    )
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


def _try_product_mode(detections) -> Tuple[Dict[int, str] | None, str]:
    product_n = _count_product_from_dets(detections)
    has_product = any(
        re.match(r"^(Product|Series|Item)\s*([A-Z])?$", d["text"].strip(), re.IGNORECASE)
        for d in detections
    )
    if has_product and 3 <= product_n <= 26:
        return _rose_sequential_letters(product_n), "rose_product_letters"
    return None, ""


def _try_label_x_mode(detections) -> Dict[int, str] | None:
    label_n = _count_label_x_from_dets(detections)
    if label_n >= 3 and label_n in CANONICAL_ROSE_AXES:
        return _rose_sequential_label_x(label_n)
    return None


def _try_synthetic_rose_mode(
    detections, outer_radius, image_shape=None, center=None
) -> Dict[int, str] | None:
    """Try direct OCR-to-grid assignment for clean synthetic rose charts.

    Returns axis_labels dict if successful, None if chart doesn't qualify.
    """
    # Collect alpha detections in label band, exclude tick values
    alpha_dets = []
    for d in detections:
        raw_text = d["text"].strip()
        compact = compact_text(raw_text)
        letters = [ch for ch in raw_text if ch.isalpha()]
        uppercase_ratio = (
            sum(ch.isupper() for ch in letters) / max(len(letters), 1)
            if letters else 0.0
        )
        if not (
            text_has_alpha(d["text"])
            and in_label_band(d, outer_radius)
            and 2 <= len(compact) <= 8
            and " " not in raw_text
            and uppercase_ratio >= 0.8
        ):
            continue
        if image_shape is not None and center is not None:
            if _is_title_or_metadata(d, image_shape, center, outer_radius):
                continue
            h, w = image_shape[:2]
            if d["width"] > w * 0.34:
                continue
        alpha_dets.append(d)

    if len(alpha_dets) < 4:
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

    if len(unique) < 4:
        return None

    # Compute angular gaps to infer n_axes
    angles = [d["angle"] for d in unique]
    angles_sorted = sorted(angles)
    gaps = []
    for i in range(len(angles)):
        gap = (angles_sorted[(i + 1) % len(angles_sorted)] - angles_sorted[i]) % 360.0
        if gap > 3:
            gaps.append(gap)
    if not gaps:
        return None

    median_gap = float(np.median(gaps))
    largest_gap = float(max(gaps))
    if median_gap < 8 or median_gap > 130:
        return None
    n_axes = max(3, min(36, round(360.0 / median_gap)))
    if n_axes not in CANONICAL_ROSE_AXES and len(unique) < max(6, int(n_axes * 0.5)):
        return None
    if largest_gap > median_gap * 3.2 and len(unique) < max(6, int(n_axes * 0.55)):
        return None

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

    if len(axis_labels) >= max(4, min(6, n_axes)):
        return axis_labels
    return None


def _try_fast_rose_modes(
    detections, image_shape, center, outer_radius
) -> Tuple[Dict[int, str] | None, Dict]:
    label_x_result = _try_label_x_mode(detections)
    if label_x_result is not None:
        return label_x_result, {"n_source": "rose_label_x", "rose_label_x": True}

    product_result, product_source = _try_product_mode(detections)
    if product_result is not None:
        return product_result, {"n_source": product_source, "rose_sequential": True}

    synthetic_result = _try_synthetic_rose_mode(detections, outer_radius, image_shape, center)
    if synthetic_result is not None:
        return synthetic_result, {"n_source": "synthetic_rose_direct", "synthetic_rose_mode": True}

    return None, {}

def detect_rose(
    image_path, center, outer_radius, use_llm=True, reader=None
) -> Tuple[Dict[int, str], Dict]:
    """Detect axis labels on a rose/polar-area chart.

    Args:
        reader: Optional pre-initialized EasyOCR reader. If None, one is
                created automatically.  Reuse across charts for speed.
    """
    if reader is None:
        reader = init_ocr()
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        return {}, _fallback_debug("image_read_failed")

    # Fast rose-specific pass: reliable patterned charts can be solved from
    # one full-image OCR pass. This also keeps titles/subtitles out of labels.
    full_detections = run_ocr_on_image(reader, image, center, source="full")
    axis_line_count = _count_radial_axis_lines(image, center, outer_radius)
    if axis_line_count < 3:
        return _fallback_or_llm_salvage(
            image_path,
            _fallback_debug("no_axis_line_evidence", axis_line_count=axis_line_count),
            "no_axis_line_evidence",
            use_llm,
            detections=full_detections,
            image_shape=image.shape,
            center=center,
            outer_radius=outer_radius,
        )

    fast_result, fast_debug = _try_fast_rose_modes(
        full_detections, image.shape, center, outer_radius
    )
    if fast_result is not None:
        fast_debug["n_final"] = len(fast_result)
        fast_debug["axis_line_count"] = axis_line_count
        return fast_result, fast_debug

    # Step 2: fall back to the heavier radar-style detection with crop OCR.
    # Always pass use_llm=False to radar detect_axes — the rose module
    # handles its own LLM refinement at the end.  The radar LLM calls
    # (llm_count_axes, llm_refine_labels) use radar-specific contact
    # sheets that are not appropriate for rose/polar-area charts.
    axis_labels, debug, detections = detect_axes(
        reader, image_path, center, outer_radius, use_llm=False
    )
    debug["axis_line_count"] = axis_line_count

    usable_candidates = _usable_ocr_candidates(detections, image.shape, center, outer_radius)
    ocr_fallback_reason = _ocr_quality_fallback_reason(usable_candidates)
    if ocr_fallback_reason:
        debug["usable_ocr_candidates"] = len(usable_candidates)
        return _fallback_or_llm_salvage(
            image_path, debug, ocr_fallback_reason, use_llm,
            detections=detections, image_shape=image.shape,
            center=center, outer_radius=outer_radius,
        )

    # ── Synthetic rose mode: direct OCR assignment ──
    # When the chart has clean, simple labels at uniform angular spacing
    # (typical of synthetic/generated rose charts), bypass the complex
    # radar binding logic and assign OCR detections directly to a uniform
    # angular grid inferred from detection gaps.
    synthetic_result, fast_debug = _try_fast_rose_modes(
        detections, image.shape, center, outer_radius
    )
    if synthetic_result is not None:
        debug = {"n_final": len(synthetic_result), **fast_debug}
        # Run LLM refinement on the synthetic result when enabled.
        # Synthetic charts have accurate grids but occasional OCR
        # misreads (e.g. "DI"→"Dl", "VVUY"→"WUY") that LLM can fix.
        if use_llm and debug.get("synthetic_rose_mode"):
            axes_angles = sorted(synthetic_result.keys())
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
    if (n_alpha + n_numeric) < 3 and not (has_product or has_label_x):
        reason = f"insufficient_ocr_text(n_alpha={n_alpha},n_numeric={n_numeric})"
        return _fallback_or_llm_salvage(
            image_path, debug, reason, use_llm,
            detections=detections, image_shape=image.shape,
            center=center, outer_radius=outer_radius,
        )

    # Non-canonical axis count in numeric-only mode — the grid is likely
    # denser than what OCR can read (e.g. 24-angle chart where only major
    # 30° ticks are detected, giving a non-integer step).
    CANONICAL = {3, 4, 5, 6, 8, 10, 12, 16, 20, 24}
    if (debug.get("numeric_axis_mode")
            and debug.get("n_alpha_band", 0) == 0
            and debug.get("n_final", 0) not in CANONICAL):
        reason = f"non_canonical_numeric_grid(n={debug.get('n_final')})"
        return _fallback_or_llm_salvage(
            image_path, debug, reason, use_llm,
            detections=detections, image_shape=image.shape,
            center=center, outer_radius=outer_radius,
        )

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
            return _fallback_or_llm_salvage(
                image_path, debug, "rose_wedge_no_sequential_evidence", use_llm,
                detections=detections, image_shape=image.shape,
                center=center, outer_radius=outer_radius,
            )
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
            return _fallback_or_llm_salvage(
                image_path, debug, "rose_sequential_no_evidence", use_llm,
                detections=detections, image_shape=image.shape,
                center=center, outer_radius=outer_radius,
            )

    # ── R3: Label-X prefix detection ──
    # When OCR reads "Uabel B", "Labcl G", etc., the chart uses
    # sequential "Label A".."Label L" labels.  Generate the full grid.
    label_x_count = sum(1 for v in labels if v != "?" and _looks_like_label_prefix(v))
    if label_x_count >= max(len(labels), 1) * 0.3 and label_x_count >= 2:
        # Force n_axes from the Label X pattern if available
        label_n = _count_label_x_from_dets(detections)
        if label_n >= 3:
            if label_n not in CANONICAL_ROSE_AXES:
                reason = f"non_canonical_label_x_grid(n={label_n})"
                return _fallback_or_llm_salvage(
                    image_path, debug, reason, use_llm,
                    detections=detections, image_shape=image.shape,
                    center=center, outer_radius=outer_radius,
                )
            axis_labels = _rose_sequential_label_x(label_n)
            debug["rose_label_x"] = True
            return axis_labels, debug

    # ── LLM refinement: per-axis label verification ──
    # Skip for synthetic/generated modes (already correct without LLM).
    is_generated = debug.get("synthetic_rose_mode") or debug.get("rose_sequential") or debug.get("rose_label_x")
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

    quality_reason = _axis_label_quality_fallback_reason(axis_labels)
    if quality_reason and not is_generated:
        return _fallback_or_llm_salvage(
            image_path, debug, quality_reason, use_llm,
            detections=detections, image_shape=image.shape,
            center=center, outer_radius=outer_radius,
        )

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
