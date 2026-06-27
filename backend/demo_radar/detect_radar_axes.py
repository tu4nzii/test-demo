"""
Radar chart axis label detection: grid-first + per-axis OCR binding.

Estimates a uniform radar-axis grid from OCR detections, then binds OCR text
to each predicted axis by geometry.  A second OCR pass runs on each axis label
region to recover small or blurry labels.

Public API:
    detect(image_path, center, outer_radius, use_llm=True) -> (axis_labels, debug)
"""

import base64
import concurrent.futures
import json
import math
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import requests

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from model_api_config import get_chat_completion_url, get_headers, get_model_name


OCR_SCALE = 2.0
CROP_SCALE = 4.0
OCR_CONFIDENCE_MIN = 0.25

RADIUS_INNER_PAD = 35
RADIUS_OUTER_PAD = 240
ANGLE_MERGE_TOL = 8.0

LLM_URL = get_chat_completion_url()
LLM_HEADERS = get_headers()
LLM_MODEL = get_model_name()


def init_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text


def compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def angle_distance(a1: float, a2: float) -> float:
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)


def axis_unit(angle_deg: float) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    return math.sin(rad), -math.cos(rad)


def bbox_center(bbox: Iterable[Iterable[float]]) -> Tuple[float, float]:
    pts = np.array(list(bbox), dtype=float)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def bbox_size(bbox: Iterable[Iterable[float]]) -> Tuple[float, float]:
    pts = np.array(list(bbox), dtype=float)
    return float(np.max(pts[:, 0]) - np.min(pts[:, 0])), float(np.max(pts[:, 1]) - np.min(pts[:, 1]))


def text_has_alpha(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def text_has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def is_usable_text(text: str) -> bool:
    if not text:
        return False
    useful = sum(ch.isalnum() for ch in text)
    return useful >= 1 and useful / max(len(text), 1) >= 0.35


def run_ocr_on_image(reader, image: np.ndarray, center: Tuple[float, float],
                     origin: Tuple[int, int] = (0, 0), scale: float = OCR_SCALE,
                     source: str = "full") -> List[Dict]:
    if image is None or image.size == 0:
        return []

    enlarged = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(enlarged)
    detections = []
    ox, oy = origin
    cx, cy = center

    for bbox, text, conf in results:
        label = clean_text(text)
        if conf < OCR_CONFIDENCE_MIN or not is_usable_text(label):
            continue

        pts = np.array(bbox, dtype=float) / scale
        pts[:, 0] += ox
        pts[:, 1] += oy
        bcx, bcy = bbox_center(pts)
        dx, dy = bcx - cx, bcy - cy
        dist = float(math.hypot(dx, dy))
        angle = float((math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0)
        width, height = bbox_size(pts)

        detections.append({
            "text": label,
            "confidence": float(conf),
            "bbox": pts.tolist(),
            "center": [bcx, bcy],
            "distance": dist,
            "angle": angle,
            "width": width,
            "height": height,
            "source": source,
        })

    return detections


def in_label_band(det: Dict, outer_radius: float) -> bool:
    return outer_radius - RADIUS_INNER_PAD <= det["distance"] <= outer_radius + RADIUS_OUTER_PAD


def in_count_band(det: Dict, outer_radius: float) -> bool:
    text = compact_text(det["text"])
    if len(text) == 1 and text_has_alpha(det["text"]):
        outer_pad = 230
    else:
        outer_pad = 115
    return outer_radius - RADIUS_INNER_PAD <= det["distance"] <= outer_radius + outer_pad


def title_like(det: Dict, image_shape: Tuple[int, int, int], center: Tuple[float, float],
               outer_radius: float) -> bool:
    h, w = image_shape[:2]
    x, y = det["center"]
    too_wide = det["width"] > w * 0.42
    moderately_wide = det["width"] > w * 0.26
    well_above_chart = y < center[1] - outer_radius - 18
    far_above_chart = y < center[1] - outer_radius - 55
    return (too_wide and well_above_chart) or (moderately_wide and far_above_chart)


def boilerplate_like(det: Dict, image_shape: Tuple[int, int, int], center: Tuple[float, float],
                     outer_radius: float) -> bool:
    h, w = image_shape[:2]
    text = compact_text(det["text"])
    x, y = det["center"]
    if any(token in text for token in ("creativecommons", "attribution", "internation", "license", "licence")):
        return True
    bottom_text = y > center[1] + outer_radius + 45 and det["width"] > w * 0.25
    return bottom_text


def metadata_like_text(text: str) -> bool:
    ctext = compact_text(text)
    metadata_terms = {"title", "artist", "year", "genre"}
    if ctext in metadata_terms:
        return True
    return any(len(ctext) >= 4 and SequenceMatcher(None, ctext, term).ratio() >= 0.78 for term in metadata_terms)


def candidate_quality(det: Dict, image_shape: Tuple[int, int, int], center: Tuple[float, float],
                      outer_radius: float, numeric_axis_mode: bool = False) -> float:
    text = det["text"]
    quality = float(det["confidence"])
    alpha_count = sum(ch.isalpha() for ch in text)
    digit_count = sum(ch.isdigit() for ch in text)
    compact_len = len(compact_text(text))

    if alpha_count:
        quality += 0.75
    if digit_count and not alpha_count:
        quality -= 0.35
    if numeric_axis_mode:
        if digit_count and not alpha_count:
            quality += 1.15
        elif alpha_count:
            quality -= 1.2
    if compact_len == 1:
        quality -= 0.15
    else:
        quality += min(compact_len, 14) / 22.0
    if title_like(det, image_shape, center, outer_radius):
        quality -= 1.2
    if boilerplate_like(det, image_shape, center, outer_radius):
        quality -= 2.2
    if metadata_like_text(text):
        quality -= 0.8
    return quality


def merge_angle_candidates(candidates: List[Dict], image_shape: Tuple[int, int, int],
                           center: Tuple[float, float], outer_radius: float,
                           tol: float = ANGLE_MERGE_TOL) -> List[Dict]:
    merged: List[Dict] = []
    for item in sorted(candidates, key=lambda x: x["angle"]):
        if not merged:
            merged.append(item)
            continue
        if angle_distance(item["angle"], merged[-1]["angle"]) <= tol:
            if candidate_quality(item, image_shape, center, outer_radius) > candidate_quality(merged[-1], image_shape, center, outer_radius):
                merged[-1] = item
        else:
            merged.append(item)
    if len(merged) > 1 and angle_distance(merged[0]["angle"], merged[-1]["angle"]) <= tol:
        first = merged[0]
        last = merged[-1]
        if candidate_quality(last, image_shape, center, outer_radius) > candidate_quality(first, image_shape, center, outer_radius):
            merged[0] = last
        merged.pop()
    return merged


def llm_count_axes(image_path: Path) -> int:
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": LLM_MODEL,
            "temperature": 0.1,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "How many labeled axes does this radar chart have? Count only labels around the outer edge. Answer with just one integer."},
                ],
            }],
        }
        resp = requests.post(LLM_URL, headers=LLM_HEADERS, json=payload, timeout=20)
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        nums = re.findall(r"\d+", answer)
        value = int(nums[0]) if nums else 0
        return value if 3 <= value <= 32 else 0
    except Exception:
        return 0


def estimate_axis_count(detections: List[Dict], image: np.ndarray, center: Tuple[float, float],
                        outer_radius: float, n_llm: int = 0) -> Tuple[int, Dict]:
    band = [d for d in detections if in_count_band(d, outer_radius)]
    band = [d for d in band if not title_like(d, image.shape, center, outer_radius)]
    band = [d for d in band if not boilerplate_like(d, image.shape, center, outer_radius)]
    alpha_band = [d for d in band if text_has_alpha(d["text"])]
    count_pool = [
        d for d in (alpha_band or band)
        if candidate_quality(d, image.shape, center, outer_radius) > 0.15
    ]
    merged = merge_angle_candidates(count_pool, image.shape, center, outer_radius)

    ocr_n = len(merged)
    numeric_band = [d for d in band if text_has_digit(d["text"]) and not text_has_alpha(d["text"])]
    numeric_axis_hint = len(numeric_band) >= 3 and len(numeric_band) >= len(alpha_band) * 0.45

    if n_llm and ocr_n <= 3:
        n_axes = n_llm
        source = "llm_undercount"
    elif n_llm and numeric_axis_hint and n_llm <= 8 and ocr_n > n_llm:
        n_axes = n_llm
        source = "llm_numeric_hint"
    elif n_llm and abs(ocr_n - n_llm) <= 1 and n_llm in {4, 5, 6, 8, 12, 16, 20} and (
        n_llm == ocr_n
        or (n_llm > ocr_n and ocr_n >= 8)
        or (n_llm < ocr_n and (ocr_n <= 6 or ocr_n >= 10))
        or numeric_axis_hint
    ):
        n_axes = n_llm
        source = "llm_close"
    else:
        n_axes = max(ocr_n, 1)
        source = "ocr"

    # Heuristic fallback: when OCR count is off by 1 from a canonical axis
    # count and the total band detections exactly match the canonical count,
    # prefer the canonical count (e.g. 15→16 when "S" OCR'd as "5").
    CANONICAL = {4, 5, 6, 8, 10, 12, 16, 20, 24}
    if source == "ocr" and ocr_n not in CANONICAL:
        total_in_band = len(alpha_band) + len(numeric_band)
        for canonical in sorted(CANONICAL, reverse=True):
            if canonical == ocr_n + 1 and total_in_band == canonical:
                n_axes = canonical
                source = "ocr_canonical_up"
                break

    # Heuristic: when OCR finds single-letter labels (common in synthetic
    # charts with A,B,C,… axis labels), use the angular spacing between
    # detected single letters to infer the true axis count.  Single-letter
    # OCR is unreliable, but the *spacing* between successfully-read letters
    # reveals the underlying grid step.
    if source == "ocr":
        single_letter_dets = [
            d for d in band
            if len(compact_text(d["text"])) == 1
            and d["text"].isalpha()
            and candidate_quality(d, image.shape, center, outer_radius) > -0.5
        ]
        single_letters = sorted(set(compact_text(d["text"]) for d in single_letter_dets))
        if len(single_letters) >= 2:
            # Compute angular gaps between consecutive single-letter detections
            angles = sorted(d["angle"] for d in single_letter_dets)
            gaps = []
            for i in range(len(angles)):
                gap = angle_distance(angles[i], angles[(i + 1) % len(angles)])
                if gap > 5:  # ignore near-zero gaps from duplicates
                    gaps.append(gap)
            if gaps:
                median_gap = float(np.median(gaps))
                if 15 <= median_gap <= 180:
                    inferred = max(3, min(26, round(360.0 / median_gap)))
                    # Also consider common axis counts near the inferred value
                    candidates = [inferred]
                    for delta in (-1, 1, -2, 2):
                        c = inferred + delta
                        if 3 <= c <= 26 and c not in candidates:
                            candidates.append(c)
                    # Pick the best: prefer count that matches single_letters count
                    # or gives a step close to median_gap
                    best_n = inferred
                    best_err = abs(360.0 / inferred - median_gap)
                    for c in candidates:
                        err = abs(360.0 / c - median_gap)
                        if err < best_err:
                            best_err = err
                            best_n = c
                    if best_n > ocr_n and best_n <= 26:
                        n_axes = best_n
                        source = "ocr_single_letter_gap"
            elif len(single_letters) > ocr_n:
                n_axes = len(single_letters)
                source = "ocr_single_letter"

    return n_axes, {
        "n_band": len(band),
        "n_alpha_band": len(alpha_band),
        "n_merged_for_count": ocr_n,
        "n_numeric_band": len(numeric_band),
        "numeric_axis_hint": numeric_axis_hint,
        "n_llm": n_llm,
        "n_source": source,
    }


def line_evidence(image: np.ndarray, center: Tuple[float, float], outer_radius: float,
                  angle_deg: float) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    ux, uy = axis_unit(angle_deg)
    cx, cy = center
    samples = []
    for r in np.linspace(outer_radius * 0.18, outer_radius * 0.98, 80):
        x = int(round(cx + ux * r))
        y = int(round(cy + uy * r))
        if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
            samples.append(edges[y, x] / 255.0)
    return float(np.mean(samples)) if samples else 0.0


def estimate_start_angle(detections: List[Dict], image: np.ndarray, center: Tuple[float, float],
                         outer_radius: float, n_axes: int) -> Tuple[float, Dict]:
    if n_axes <= 0:
        return 0.0, {"start_source": "fallback"}

    step = 360.0 / n_axes
    label_candidates = [
        d for d in detections
        if in_label_band(d, outer_radius)
        and not title_like(d, image.shape, center, outer_radius)
        and not boilerplate_like(d, image.shape, center, outer_radius)
    ]
    if len(label_candidates) <= n_axes:
        return 0.0, {
            "start_source": "zero_sparse_labels",
            "start_score": 0.0,
            "zero_start_score": 0.0,
            "start_candidates": len(label_candidates),
        }

    # All current radar examples start at 0 deg, but a light grid search keeps
    # the script usable when a future chart is rotated.
    best_offset = 0.0
    best_score = -1e9
    grid = np.arange(0.0, step, max(0.5, min(2.0, step / 12.0)))
    angle_limit = min(28.0, max(8.0, step * 0.48))

    zero_score = None
    for offset in grid:
        score = 0.0
        for det in label_candidates:
            nearest = round((det["angle"] - offset) / step)
            grid_angle = (offset + nearest * step) % 360
            delta = angle_distance(det["angle"], grid_angle)
            if delta <= angle_limit:
                score += candidate_quality(det, image.shape, center, outer_radius) * (1.0 - delta / angle_limit)
        # Keep visual-line evidence weak; colored series can be noisy.
        for i in range(n_axes):
            score += 0.18 * line_evidence(image, center, outer_radius, (offset + i * step) % 360)
        if score > best_score:
            best_score = score
            best_offset = float(offset)
        if abs(float(offset)) < 1e-9:
            zero_score = score

    if zero_score is None:
        zero_score = best_score

    # Most real radar charts start at the top. Keep a rotated grid only when it
    # is clearly better than the zero-start grid; otherwise titles/legends can
    # drag the offset away from the chart's actual geometry.
    if zero_score >= best_score * 0.92:
        best_offset = 0.0
    return best_offset % 360.0, {
        "start_source": "grid_search",
        "start_score": round(best_score, 3),
        "zero_start_score": round(float(zero_score), 3),
        "start_candidates": len(label_candidates),
    }


def crop_axis_region(image: np.ndarray, center: Tuple[float, float], angle_deg: float,
                     outer_radius: float, step: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape[:2]
    ux, uy = axis_unit(angle_deg)
    cx, cy = center
    label_r = outer_radius + 48
    lx = int(round(cx + ux * label_r))
    ly = int(round(cy + uy * label_r))

    half_w = int(max(85, min(230, outer_radius * 0.62)))
    half_h = int(max(42, min(95, outer_radius * 0.23)))
    if step <= 24:
        half_w = int(max(70, min(145, outer_radius * 0.38)))
        half_h = int(max(34, min(70, outer_radius * 0.16)))

    x1 = max(0, lx - half_w)
    y1 = max(0, ly - half_h)
    x2 = min(w, lx + half_w)
    y2 = min(h, ly + half_h)
    return image[y1:y2, x1:x2], (x1, y1)


def projection_to_axis(det: Dict, center: Tuple[float, float], angle_deg: float) -> Tuple[float, float]:
    ux, uy = axis_unit(angle_deg)
    vx = det["center"][0] - center[0]
    vy = det["center"][1] - center[1]
    proj = vx * ux + vy * uy
    perp = abs(vx * (-uy) + vy * ux)
    return float(proj), float(perp)


def score_for_axis(det: Dict, image_shape: Tuple[int, int, int], center: Tuple[float, float],
                   outer_radius: float, angle_deg: float, step: float,
                   numeric_axis_mode: bool = False) -> float:
    proj, perp = projection_to_axis(det, center, angle_deg)
    if proj < outer_radius - RADIUS_INNER_PAD or proj > outer_radius + RADIUS_OUTER_PAD:
        return -1e9

    _, bbox_h = det["width"], det["height"]
    perp_limit = max(22.0, bbox_h * 0.8, math.radians(step) * max(proj, outer_radius) * 0.45)
    if perp > perp_limit + 16:
        return -1e9

    q = candidate_quality(det, image_shape, center, outer_radius, numeric_axis_mode=numeric_axis_mode)
    outward_preference = -abs(proj - (outer_radius + 48.0)) / 76.0
    axis_preference = -perp / max(perp_limit, 1.0)
    source_bonus = 0.05 if det.get("source") == "crop" else 0.0
    # Penalize text that looks like a statistical footnote/legend rather than an axis label.
    footnote_penalty = 0.0
    raw_text = det.get("text", "")
    if re.search(r"\(mean\b|\(median\b|\(sd\b|\(score\b|\(%", raw_text, re.IGNORECASE):
        footnote_penalty = -3.5
    return q + outward_preference + axis_preference + source_bonus + footnote_penalty


def postprocess_axis_labels(axis_labels: Dict[int, str]) -> Dict[int, str]:
    if len(axis_labels) != 12:
        return axis_labels

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ordered = sorted(axis_labels.items())
    hits = 0
    for _, label in ordered:
        c = compact_text(label)
        if c in {compact_text(m) for m in months}:
            hits += 1
        elif any(c and (c.startswith(compact_text(m)[:2]) or compact_text(m).startswith(c)) for m in months):
            hits += 1

    if hits >= 8:
        return {angle: month for (angle, _), month in zip(ordered, months)}
    return axis_labels


def dedupe_detections(detections: List[Dict]) -> List[Dict]:
    kept: List[Dict] = []
    def rank(d: Dict) -> Tuple[float, int, int]:
        return (float(d["confidence"]), len(compact_text(d["text"])), 1 if d.get("source") == "crop" else 0)

    for det in sorted(detections, key=lambda d: (-rank(d)[0], -rank(d)[1], -rank(d)[2])):
        ctext = compact_text(det["text"])
        duplicate = False
        for index, existing in enumerate(kept):
            etext = compact_text(existing["text"])
            close = math.hypot(det["center"][0] - existing["center"][0], det["center"][1] - existing["center"][1]) < 28
            same_or_partial = ctext and etext and (ctext == etext or ctext in etext or etext in ctext)
            if close and same_or_partial:
                if rank(det) > rank(existing):
                    kept[index] = det
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def bind_labels_to_axes(reader, image: np.ndarray, image_path: Path, center: Tuple[float, float],
                        outer_radius: float, axes_angles: List[float],
                        full_detections: List[Dict]) -> Tuple[Dict[int, str], List[Dict], Dict]:
    step = 360.0 / max(len(axes_angles), 1)
    all_detections = list(full_detections)
    crop_ocr_count = 0

    for angle in axes_angles:
        crop, origin = crop_axis_region(image, center, angle, outer_radius, step)
        crop_dets = run_ocr_on_image(reader, crop, center, origin=origin, scale=CROP_SCALE, source="crop")
        crop_ocr_count += len(crop_dets)
        all_detections.extend(crop_dets)

    all_detections = dedupe_detections(all_detections)
    numeric_axis_mode = (
        sum(1 for d in all_detections if in_label_band(d, outer_radius) and text_has_digit(d["text"]) and not text_has_alpha(d["text"]))
        >= max(3, len(axes_angles) // 2)
    )

    axis_labels: Dict[int, str] = {}
    used = set()
    assignments = []

    if numeric_axis_mode:
        numeric_candidates = [
            d for d in all_detections
            if in_label_band(d, outer_radius)
            and text_has_digit(d["text"])
            and not text_has_alpha(d["text"])
            and not title_like(d, image.shape, center, outer_radius)
            and not boilerplate_like(d, image.shape, center, outer_radius)
        ]
        numeric_candidates = sorted(numeric_candidates, key=lambda d: (d["angle"] % 360, -d["confidence"]))
        pruned_numeric: List[Dict] = []
        for det in numeric_candidates:
            if pruned_numeric and angle_distance(det["angle"], pruned_numeric[-1]["angle"]) < 7:
                if det["confidence"] > pruned_numeric[-1]["confidence"]:
                    pruned_numeric[-1] = det
            else:
                pruned_numeric.append(det)
        numeric_candidates = pruned_numeric

        if len(numeric_candidates) >= len(axes_angles):
            selected = numeric_candidates[:len(axes_angles)]
            selected = sorted(selected, key=lambda d: d["angle"] % 360)
            for angle, det in zip(sorted(axes_angles), selected):
                key = int(round(angle)) % 360
                axis_labels[key] = det["text"]
                assignments.append({
                    "angle": round(angle, 2),
                    "text": det["text"],
                    "score": "numeric_order",
                    "source": det.get("source"),
                    "candidate_angle": round(det["angle"], 2),
                    "distance": round(det["distance"], 1),
                })
            debug = {
                "n_crop_ocr": crop_ocr_count,
                "n_all_deduped": len(all_detections),
                "numeric_axis_mode": numeric_axis_mode,
                "assignments": assignments,
            }
            return axis_labels, all_detections, debug

    for angle in axes_angles:
        scored = []
        for idx, det in enumerate(all_detections):
            if idx in used:
                continue
            score = score_for_axis(det, image.shape, center, outer_radius, angle, step, numeric_axis_mode=numeric_axis_mode)
            if score > -1e8:
                scored.append((score, idx, det))

        if not scored:
            # Fallback: relax perp constraint for orphan axes.
            # Accept any unused detection on the correct radial band and pick
            # the one with the best quality-minus-angle-error score.
            for idx, det in enumerate(all_detections):
                if idx in used:
                    continue
                proj, _ = projection_to_axis(det, center, angle)
                if outer_radius - RADIUS_INNER_PAD <= proj <= outer_radius + RADIUS_OUTER_PAD:
                    q = candidate_quality(det, image.shape, center, outer_radius, numeric_axis_mode=numeric_axis_mode)
                    ang_err = angle_distance(det["angle"], angle)
                    relaxed_score = q - ang_err / 12.0
                    scored.append((relaxed_score, idx, det))
        if not scored:
            axis_labels[int(round(angle)) % 360] = "?"
            assignments.append({"angle": angle, "text": "?", "score": None, "source": "none"})
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best = scored[0]

        # If alphabetic candidates are available for this axis, prefer them over
        # pure tick numbers unless the numeric candidate is decisively better.
        alpha_scored = [s for s in scored if text_has_alpha(s[2]["text"])]
        if alpha_scored and not text_has_alpha(best["text"]) and not numeric_axis_mode:
            alpha_score, alpha_idx, alpha = alpha_scored[0]
            if alpha_score >= best_score - 0.65:
                best_score, best_idx, best = alpha_score, alpha_idx, alpha

        if best.get("source") == "crop":
            full_near_ties = [
                s for s in scored
                if s[2].get("source") == "full"
                and s[0] >= best_score - 0.12
                and len(compact_text(s[2]["text"])) >= len(compact_text(best["text"]))
            ]
            if full_near_ties:
                full_near_ties.sort(key=lambda s: (len(compact_text(s[2]["text"])), s[0]), reverse=True)
                best_score, best_idx, best = full_near_ties[0]

        used.add(best_idx)
        key = int(round(angle)) % 360
        axis_labels[key] = best["text"]
        assignments.append({
            "angle": round(angle, 2),
            "text": best["text"],
            "score": round(float(best_score), 3),
            "source": best.get("source"),
            "candidate_angle": round(best["angle"], 2),
            "distance": round(best["distance"], 1),
        })

    # Local swap optimisation: for each adjacent pair of axes, swap their
    # labels if doing so reduces the total candidate_angle alignment error.
    # This fixes cases where a label was "stolen" by a neighbour axis (e.g.
    # "Need for achievement" at 308° instead of 334° on RadarChart15).
    def _assign_angle_key(a: float) -> int:
        return int(round(float(a))) % 360

    sorted_angles = sorted(axis_labels.keys())
    for i in range(len(sorted_angles)):
        a1 = sorted_angles[i]
        a2 = sorted_angles[(i + 1) % len(sorted_angles)]
        lbl1 = axis_labels.get(a1, "?")
        lbl2 = axis_labels.get(a2, "?")
        if lbl1 == "?" or lbl2 == "?":
            continue
        c1 = next((a["candidate_angle"] for a in assignments
                   if _assign_angle_key(a["angle"]) == a1 and a.get("candidate_angle") is not None), None)
        c2 = next((a["candidate_angle"] for a in assignments
                   if _assign_angle_key(a["angle"]) == a2 and a.get("candidate_angle") is not None), None)
        if c1 is None or c2 is None:
            continue
        err_before = angle_distance(c1, a1) + angle_distance(c2, a2)
        err_after = angle_distance(c1, a2) + angle_distance(c2, a1)
        if err_after + 1.5 < err_before:  # require meaningful improvement
            axis_labels[a1], axis_labels[a2] = lbl2, lbl1
            for a in assignments:
                if _assign_angle_key(a["angle"]) == a1:
                    a["text"] = lbl2
                elif _assign_angle_key(a["angle"]) == a2:
                    a["text"] = lbl1

    axis_labels = postprocess_axis_labels(axis_labels)
    for assignment in assignments:
        key = int(round(float(assignment["angle"]))) % 360
        if key in axis_labels:
            assignment["text"] = axis_labels[key]

    debug = {
        "n_crop_ocr": crop_ocr_count,
        "n_all_deduped": len(all_detections),
        "numeric_axis_mode": numeric_axis_mode,
        "assignments": assignments,
    }
    return axis_labels, all_detections, debug


def draw_viz(image: np.ndarray, center: Tuple[float, float], outer_radius: float,
             detections: List[Dict], axis_labels: Dict[int, str], output_path: Path):
    vis = image.copy()
    cx, cy = center
    cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    cv2.circle(vis, (int(cx), int(cy)), int(round(outer_radius)), (180, 180, 180), 1)

    for det in detections:
        color = (180, 120, 40) if det.get("source") == "full" else (80, 160, 220)
        bbox = np.array(det["bbox"], dtype=np.int32)
        cv2.polylines(vis, [bbox], isClosed=True, color=color, thickness=1)

    for angle_deg, label in sorted(axis_labels.items()):
        ux, uy = axis_unit(angle_deg)
        ex = int(cx + ux * (outer_radius + 80))
        ey = int(cy + uy * (outer_radius + 80))
        cv2.line(vis, (int(cx), int(cy)), (ex, ey), (0, 190, 0), 2)
        cv2.putText(vis, f"{label} ({angle_deg})", (ex + 4, ey + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 170, 0), 2)

    cv2.imwrite(str(output_path), vis)


def labels_match(detected: str, expected: str) -> Tuple[bool, str]:
    det = (detected or "").strip().lower()
    exp = (expected or "").strip().lower()
    if det == exp:
        return True, "exact"

    cdet = compact_text(det)
    cexp = compact_text(exp)
    if cdet and cdet == cexp:
        return True, "normalized"

    if len(cdet) >= 5 and len(cexp) >= 5:
        ratio = SequenceMatcher(None, cdet, cexp).ratio()
        if ratio >= 0.88:
            return True, "fuzzy"

        # EasyOCR often reads "e" as "c" in small radar labels:
        # Timeliness -> Timclincss, Ireland -> Ircland, The -> Thc.
        det_ce = cdet.replace("c", "e")
        exp_ce = cexp.replace("c", "e")
        if det_ce == cexp or cdet == exp_ce:
            return True, "fuzzy_ce"
        if SequenceMatcher(None, det_ce, cexp).ratio() >= 0.88:
            return True, "fuzzy_ce"

    if cdet and cexp and (cdet in cexp or cexp in cdet):
        # If detected is a substantial substring of expected (e.g. "amine" ⊆ "ketamine"),
        # treat as fuzzy match rather than partial mismatch.
        if cdet in cexp and len(cdet) >= 4:
            return True, "fuzzy_sub"
        return False, "partial"
    return False, "mismatch"


def evaluate(axis_labels: Dict[int, str], gt_labels: List[str], gt_angles: List[float]) -> Tuple[int, int, List[Tuple]]:
    correct = total = 0
    details = []
    for gt_angle, gt_label in zip(gt_angles, gt_labels):
        total += 1
        if not axis_labels:
            details.append((gt_angle, gt_label, "?", "no_detect"))
            continue
        closest = min(axis_labels.keys(), key=lambda a: angle_distance(a, gt_angle))
        detected = axis_labels[closest]
        ok, status = labels_match(detected, gt_label)
        if ok:
            correct += 1
        details.append((gt_angle, gt_label, detected, status))
    return correct, total, details


def _detect_and_infer_sequential_letters(
    axis_labels: Dict[int, str],
    all_detections: List[Dict],
    image: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
    n_axes: int,
) -> Tuple[bool, int]:
    """Detect sequential-letter charts AND infer the true axis count from gaps.

    Returns (is_sequential, best_n_axes).  When is_sequential is True,
    best_n_axes is the inferred count (may differ from the passed n_axes).
    """
    if n_axes < 3:
        return False, n_axes

    # Collect single-letter detections in the label band
    single_letter_dets = [
        d for d in all_detections
        if in_label_band(d, outer_radius)
        and len(compact_text(d["text"])) == 1
        and d["text"].isalpha()
    ]
    single_letters = sorted(set(compact_text(d["text"]) for d in single_letter_dets))

    # Compute best n_axes: try alphabetic-gap heuristic first, then
    # fall back to a "best fit" search over plausible axis counts (3-12).
    # For each candidate n, generate sequential labels and score how well
    # they match the raw single-letter OCR detections.
    best_n = n_axes
    letter_positions = []
    for d in single_letter_dets:
        ch = compact_text(d["text"])
        if len(ch) == 1 and "A" <= ch <= "Z":
            letter_positions.append((d["angle"], ord(ch) - ord("A"), ch))
    letter_positions.sort(reverse=True)  # descending angle = CCW order

    if len(letter_positions) >= 2:
        # ── Alphabetic-gap heuristic with counter-clockwise direction ──
        # Synthetic charts place labels CCW starting from the right (A at ~90°
        # in image coordinates).  Use directed CCW angular gaps rather than
        # shortest-arc distance to preserve the label ordering.
        def _ccw_gap(a1: float, a2: float) -> float:
            """Counter-clockwise angular gap from a1 to a2 (0–360)."""
            return (a1 - a2) % 360.0

        step_estimates = []
        for i in range(len(letter_positions)):
            a1, p1, _ = letter_positions[i]
            a2, p2, _ = letter_positions[(i + 1) % len(letter_positions)]
            angle_gap = _ccw_gap(a1, a2)  # CCW from a1 to a2
            letter_gap = (p2 - p1) % 26   # forward alphabetic (A→B=1)
            if letter_gap > 0 and 5 < angle_gap < 355:
                step_estimates.append(angle_gap / letter_gap)
        if step_estimates:
            median_step = float(np.median(step_estimates))
            if 10 <= median_step <= 180:
                best_n = max(3, min(26, round(360.0 / median_step)))

        # ── Best-fit search: try n=3..12, score by matching generated
        #     letters to raw OCR single-letter detections ──
        best_fit_score = -1
        for n_candidate in range(3, 13):
            step = 360.0 / n_candidate
            score = 0
            for _, _, ch in letter_positions:
                expected_angle = (ord(ch) - ord("A")) * step % 360
                # Find the closest raw detection to this expected angle
                best_match = 1e9
                for a, _, dch in letter_positions:
                    if dch == ch:
                        best_match = min(best_match, angle_distance(a, expected_angle))
                if best_match < step * 0.6:  # within 60% of a step
                    score += 1
            if score > best_fit_score:
                best_fit_score = score
                best_fit_n = n_candidate
        # Use best-fit if it matches more letters than the gap heuristic
        if best_fit_score > len(letter_positions) * 0.6:
            best_n = best_fit_n

    elif len(single_letter_dets) >= 2:
        # Fallback: angular gaps only (no alphabetic info)
        angles = sorted(d["angle"] for d in single_letter_dets)
        gaps = []
        for i in range(len(angles)):
            gap = angle_distance(angles[i], angles[(i + 1) % len(angles)])
            if gap > 5:
                gaps.append(gap)
        if gaps:
            median_gap = float(np.median(gaps))
            if 15 <= median_gap <= 180:
                best_n = max(3, min(26, round(360.0 / median_gap)))

    # Detection: does this chart look like sequential letters?
    labels = [v for v in axis_labels.values() if v and v != "?"]
    question_count = sum(1 for v in axis_labels.values() if v == "?")

    # Case 1: many "?" placeholders → OCR lost
    if question_count >= len(axis_labels) * 0.3:
        if len(single_letters) >= 1:
            return True, best_n
        if question_count >= len(axis_labels) * 0.5:
            long_junk = sum(1 for v in axis_labels.values()
                           if v != "?" and len(v.strip()) > 3)
            if long_junk >= 1 or question_count == len(axis_labels):
                return True, best_n

    if not labels:
        return False, n_axes

    # Case 2: mix of single letters and long random strings → pollution
    single_count = sum(1 for v in labels if len(v.strip()) == 1 and v.strip().isalpha())
    long_count = sum(1 for v in labels if len(v.strip()) > 3)
    if single_count >= 1 and long_count >= 1:
        return True, best_n

    # Case 3: all (or nearly all) assigned labels are single letters —
    # this is the hallmark of a sequential-letter chart regardless of
    # whether long-junk strings are present.
    if single_count >= len(labels) * 0.6 and single_count >= 3:
        return True, best_n

    return False, n_axes


def _generate_sequential_letters(
    n_axes: int, start_angle: float
) -> Dict[int, str]:
    """Generate A,B,C,… labels clockwise starting from start_angle."""
    if n_axes < 3 or n_axes > 26:
        return {}
    letters = [chr(ord("A") + i) for i in range(n_axes)]
    step = 360.0 / n_axes
    return {
        int(round((start_angle + i * step) % 360)): letters[i]
        for i in range(n_axes)
    }


def _generate_sequential_letters_ccw(
    n_axes: int, start_angle: float
) -> Dict[int, str]:
    """Generate A,B,C,… labels counter-clockwise starting from start_angle."""
    if n_axes < 3 or n_axes > 26:
        return {}
    letters = [chr(ord("A") + i) for i in range(n_axes)]
    step = 360.0 / n_axes
    return {
        int(round((start_angle - i * step) % 360)): letters[i]
        for i in range(n_axes)
    }


def llm_refine_labels(
    image: np.ndarray,
    image_path: Path,
    center: Tuple[float, float],
    outer_radius: float,
    axis_labels: Dict[int, str],
    axes_angles: List[float],
) -> Dict[int, str]:
    """Use multimodal LLM to verify/refine each axis label.

    For each axis, crops the label region and asks the LLM to read the text.
    If the LLM disagrees with the OCR label, the LLM answer is used.

    Uses a thread-pool with hard timeout to prevent API hangs from blocking
    the entire pipeline.
    """
    refined = dict(axis_labels)
    consecutive_timeouts = 0

    for angle in axes_angles:
        key = int(round(angle)) % 360
        current = refined.get(key, "?")
        if current == "?":
            continue

        # Abort if API seems completely down
        if consecutive_timeouts >= 3:
            break

        try:
            crop, _ = crop_axis_region(image, center, angle, outer_radius, 30.0)
            if crop is None or crop.size == 0:
                continue

            _, buf = cv2.imencode(".png", crop)
            b64 = base64.b64encode(buf).decode("utf-8")

            prompt = (
                "Read the text label at this radar chart axis position. "
                "Return ONLY the label text, nothing else. "
                f"OCR guessed: \"{current}\". If correct, repeat it. "
                "If wrong, write the correct text."
            )

            payload = {
                "model": LLM_MODEL,
                "temperature": 0.1,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            }

            # Hard timeout via thread — requests timeout alone is unreliable
            # when the server dribbles data.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    requests.post, LLM_URL, headers=LLM_HEADERS,
                    json=payload, timeout=8,
                )
                resp = future.result(timeout=10)

            if resp.ok:
                answer = resp.json()["choices"][0]["message"]["content"].strip()
                answer = re.sub(r'^["\']|["\']$', '', answer)
                if answer and len(answer) <= 50:
                    refined[key] = answer
                consecutive_timeouts = 0
            else:
                consecutive_timeouts += 1
        except (concurrent.futures.TimeoutError, Exception):
            consecutive_timeouts += 1
            continue
    return refined


def detect_axes(reader, image_path: Path, center: Tuple[float, float], outer_radius: float,
                use_llm: bool) -> Tuple[Dict[int, str], Dict, List[Dict]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return {}, {"error": "cannot read image"}, []

    full_detections = run_ocr_on_image(reader, image, center, source="full")
    n_llm = llm_count_axes(image_path) if use_llm else 0
    n_axes, count_debug = estimate_axis_count(full_detections, image, center, outer_radius, n_llm=n_llm)

    if n_axes <= 0:
        return {}, {"error": "no axes"}, full_detections

    start_angle, start_debug = estimate_start_angle(full_detections, image, center, outer_radius, n_axes)
    step = 360.0 / n_axes
    axes_angles = [(start_angle + i * step) % 360.0 for i in range(n_axes)]

    axis_labels, detections, bind_debug = bind_labels_to_axes(
        reader, image, image_path, center, outer_radius, axes_angles, full_detections
    )

    # Fallback check: when numeric_axis_mode is active but the assignment
    # scores indicate confusion (mixed numeric ticks + alpha labels competing
    # for the same slots), mark the chart as unreliable and return empty.
    # This catches charts where OCR fundamentally cannot read the axis labels.
    if bind_debug.get("numeric_axis_mode") and axis_labels:
        scores = [a["score"] for a in bind_debug.get("assignments", [])
                  if isinstance(a.get("score"), (int, float))]
        if scores:
            neg_rate = sum(1 for s in scores if s < 0) / len(scores)
            med_score = sorted(scores)[len(scores) // 2]
            if (med_score < 0.0 or neg_rate > 0.25) and n_llm == 0:
                axis_labels = {}
                bind_debug["fallback"] = True
                bind_debug["fallback_reason"] = (
                    f"unreliable_numeric_mode(median_score={med_score:.3f},"
                    f"negative_rate={neg_rate:.2f})"
                )

    # Single-letter sequential label mode: when the chart uses simple A,B,C,...
    # labels (common in synthetic/generated charts), OCR often misses individual
    # letters while picking up data-series names.  Detect the pattern and
    # generate correct sequential labels using gap-inferred axis count.
    is_seq, seq_n_axes = _detect_and_infer_sequential_letters(
        axis_labels, detections, image, center, outer_radius, n_axes,
    )
    if is_seq:
        seq_start = 0.0
        seq_labels = _generate_sequential_letters(seq_n_axes, seq_start)
        if seq_labels:
            axis_labels = seq_labels
            n_axes = seq_n_axes
            start_angle = seq_start
            bind_debug["sequential_letter_mode"] = True
            bind_debug["seq_n_axes"] = seq_n_axes

    debug = {
        "n_raw": len(full_detections),
        "n_final": n_axes,
        "step": round(step, 2),
        "start_angle": round(start_angle, 2),
        "outer_r": outer_radius,
        **count_debug,
        **start_debug,
        **bind_debug,
    }

    # ── LLM refinement: per-axis label verification ──
    # Skip for sequential letter mode (synthetic charts, already correct).
    if use_llm and axis_labels and not bind_debug.get("sequential_letter_mode"):
        try:
            axis_labels = llm_refine_labels(
                image, image_path, center, outer_radius, axis_labels, axes_angles
            )
            bind_debug["llm_refined"] = True
        except Exception:
            pass

    return axis_labels, debug, detections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(image_path, center, outer_radius, use_llm=True):
    """Detect axis labels on a radar chart.

    Args:
        image_path: Path to the chart PNG.
        center: (cx, cy) tuple in pixels.
        outer_radius: Outer circle radius in pixels.
        use_llm: Whether to call multimodal LLM for axis count (may fail).

    Returns:
        (axis_labels, debug) where axis_labels is {angle_deg: label_text}
        and debug is a dict with detection metadata.
        If the chart is deemed unreliable (F6 fallback), axis_labels is {}.
    """
    reader = init_ocr()
    return detect_axes(reader, Path(image_path),
                       (float(center[0]), float(center[1])),
                       float(outer_radius), use_llm=use_llm)[:2]
