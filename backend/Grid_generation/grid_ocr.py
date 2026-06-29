from __future__ import annotations

import argparse
import base64
from itertools import combinations
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

import cv2
import numpy as np

from grid_math import parse_numeric_label, regularity_score

OCR_ENGINE = None
OCR_ERROR: str | None = None

def predict_paddle_ocr_raw(engine, image: np.ndarray) -> tuple[list[dict[str, object]], str | None]:
    try:
        if hasattr(engine, "predict"):
            raw = engine.predict(image)
            results = collect_new_paddleocr_results(raw)
            if not results:
                results = collect_old_paddleocr_results(raw)
            return results, None
        if hasattr(engine, "ocr"):
            try:
                raw = engine.ocr(image, cls=True)
            except TypeError:
                raw = engine.ocr(image)
            return collect_old_paddleocr_results(raw), None
        return [], "PaddleOCR object has no ocr/predict method"
    except Exception as exc:  # pragma: no cover - depends on optional local package.
        return [], f"OCR failed: {exc}"

def get_ocr_engine(args: argparse.Namespace):
    global OCR_ENGINE, OCR_ERROR
    if args.no_ocr:
        return None
    if OCR_ENGINE is not None or OCR_ERROR is not None:
        return OCR_ENGINE

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover - depends on optional local package.
        OCR_ERROR = f"PaddleOCR unavailable: {exc}"
        print(f"OCR disabled: {OCR_ERROR}")
        return None

    init_attempts = [
        {
            "lang": args.ocr_lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_det_thresh": args.ocr_det_thresh,
            "text_det_box_thresh": args.ocr_det_box_thresh,
            "text_det_unclip_ratio": args.ocr_det_unclip_ratio,
            "text_det_limit_side_len": args.ocr_det_limit_side_len,
            "text_det_limit_type": args.ocr_det_limit_type,
            "return_word_box": args.ocr_return_word_box,
        },
        {
            "use_angle_cls": False,
            "lang": args.ocr_lang,
        },
        {"lang": args.ocr_lang},
    ]
    for kwargs in init_attempts:
        try:
            OCR_ENGINE = PaddleOCR(**kwargs)
            OCR_ERROR = None
            return OCR_ENGINE
        except TypeError:
            continue
        except Exception as exc:  # pragma: no cover - depends on optional local package.
            if "Unknown argument" in str(exc) or "deprecated" in str(exc):
                continue
            OCR_ERROR = f"PaddleOCR init failed: {exc}"
            print(f"OCR disabled: {OCR_ERROR}")
            return None

    OCR_ERROR = "PaddleOCR init failed: unsupported constructor arguments"
    print(f"OCR disabled: {OCR_ERROR}")
    return None

def is_bbox(value) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 2 and value.shape[0] == 4 and value.shape[1] >= 2
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
        if not all(isinstance(coord, (int, float, np.integer, np.floating)) for coord in point[:2]):
            return False
    return True

def normalize_box(box) -> list[list[float]]:
    points = np.array([[float(point[0]), float(point[1])] for point in box], dtype=np.float32)
    return points.tolist()

def parse_ocr_line(item) -> dict[str, object] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2 or not is_bbox(item[0]):
        return None
    text = ""
    score = 0.0
    payload = item[1]
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        text = str(payload[0])
        try:
            score = float(payload[1])
        except (TypeError, ValueError):
            score = 0.0
    elif isinstance(payload, str):
        text = payload
        score = 1.0
    if not text:
        return None
    return {"text": text, "score": score, "box": normalize_box(item[0])}

def collect_old_paddleocr_results(raw) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if raw is None:
        return results
    if isinstance(raw, (list, tuple)):
        parsed = parse_ocr_line(raw)
        if parsed is not None:
            return [parsed]
        for item in raw:
            results.extend(collect_old_paddleocr_results(item))
    return results

def collect_new_paddleocr_results(raw) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]

    for item in raw:
        data = None
        if isinstance(item, dict) or hasattr(item, "get"):
            data = item
        elif hasattr(item, "json"):
            try:
                data = item.json() if callable(item.json) else item.json
            except Exception:
                data = None
        elif hasattr(item, "to_dict"):
            try:
                data = item.to_dict()
            except Exception:
                data = None
        if not isinstance(data, dict):
            continue

        texts = data.get("rec_texts") or data.get("texts") or []
        scores = data.get("rec_scores") or data.get("scores") or [1.0] * len(texts)
        boxes = data.get("dt_polys") or data.get("rec_polys") or data.get("boxes") or []
        for text, score, box in zip(texts, scores, boxes):
            if not text or not is_bbox(box):
                continue
            results.append({"text": str(text), "score": float(score), "box": normalize_box(box)})
    return results

def rotate_point_back(
    x: float,
    y: float,
    crop_width: int,
    crop_height: int,
    rotation: int,
) -> tuple[float, float]:
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return y, float(crop_height - 1) - x
    if rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return float(crop_width - 1) - y, x
    if rotation == cv2.ROTATE_180:
        return float(crop_width - 1) - x, float(crop_height - 1) - y
    return x, y

def map_rotated_crop_item_to_image(
    item: dict[str, object],
    crop_rect: tuple[int, int, int, int],
    rotation: int,
    pass_name: str,
) -> dict[str, object] | None:
    x0, y0, x1, y1 = crop_rect
    crop_width = max(1, x1 - x0)
    crop_height = max(1, y1 - y0)
    box = item.get("box")
    if not is_bbox(box):
        return None
    mapped: list[list[float]] = []
    for point in box:  # type: ignore[assignment]
        px, py = rotate_point_back(float(point[0]), float(point[1]), crop_width, crop_height, rotation)
        mapped.append([px + x0, py + y0])
    copy = dict(item)
    copy["box"] = mapped
    copy["ocr_pass"] = pass_name
    copy["rotated_ocr"] = True
    return copy

def rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0

def ocr_item_rect(item: dict[str, object]) -> tuple[int, int, int, int] | None:
    rect = box_rect(item.get("box"))
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return None
    return rect

def dedupe_ocr_results(items: list[dict[str, object]]) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = []
    for item in sorted(items, key=lambda value: (bool(value.get("rotated_ocr")), -float(value.get("score", 0.0) or 0.0))):
        text = normalize_label_text(item.get("text", ""))
        rect = ocr_item_rect(item)
        if not text or rect is None:
            continue
        duplicate_index: int | None = None
        for index, old in enumerate(kept):
            old_text = normalize_label_text(old.get("text", ""))
            old_rect = ocr_item_rect(old)
            if old_rect is None or old_text != text:
                continue
            if rect_iou(rect, old_rect) >= 0.62:
                duplicate_index = index
                break
        if duplicate_index is None:
            kept.append(item)
            continue
        old = kept[duplicate_index]
        old_score = float(old.get("score", 0.0) or 0.0)
        new_score = float(item.get("score", 0.0) or 0.0)
        if new_score > old_score + 0.03 and not old.get("rotated_ocr"):
            kept[duplicate_index] = item
    return kept

def rotated_axis_crop_ocr(engine, image: np.ndarray) -> list[dict[str, object]]:
    h, w = image.shape[:2]
    crop_rects = [
        ("x_axis_bottom", (0, int(h * 0.72), w, h), [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]),
        ("y_axis_left", (0, 0, int(w * 0.30), h), [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]),
    ]
    augmented: list[dict[str, object]] = []
    for name, (x0, y0, x1, y1), rotations in crop_rects:
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        for rotation in rotations:
            rotated = cv2.rotate(crop, rotation)
            raw_items, error = predict_paddle_ocr_raw(engine, rotated)
            if error:
                continue
            for item in raw_items:
                mapped = map_rotated_crop_item_to_image(item, (x0, y0, x1, y1), rotation, name)
                if mapped is not None:
                    augmented.append(mapped)
    return augmented

def run_paddle_ocr(image: np.ndarray, args: argparse.Namespace) -> tuple[list[dict[str, object]], str | None]:
    engine = get_ocr_engine(args)
    if engine is None:
        return [], OCR_ERROR

    results, error = predict_paddle_ocr_raw(engine, image)
    if error:
        return [], error
    results = dedupe_ocr_results([*results, *rotated_axis_crop_ocr(engine, image)])

    filtered = [item for item in results if float(item["score"]) >= args.ocr_min_score]
    return classify_axis_ocr_results(filtered, image.shape[:2]), None

def classify_axis_ocr_results(
    items: list[dict[str, object]],
    image_shape: tuple[int, int],
) -> list[dict[str, object]]:
    h, w = image_shape
    classified: list[dict[str, object]] = []
    for item in items:
        box = np.array(item["box"], dtype=np.float32)
        center_x = float(box[:, 0].mean())
        center_y = float(box[:, 1].mean())
        width = float(box[:, 0].max() - box[:, 0].min())
        height = float(box[:, 1].max() - box[:, 1].min())
        angle_like_vertical = height > width * 1.5

        if center_y <= h * 0.12:
            role = "other"
        elif center_y >= h * 0.82:
            role = "x_axis"
        elif center_x <= w * 0.22:
            role = "y_axis"
        elif angle_like_vertical and center_x <= w * 0.38:
            role = "y_axis"
        else:
            role = "other"

        copy = dict(item)
        copy.update(
            {
                "role": role,
                "center": [center_x, center_y],
                "size": [width, height],
            }
        )
        classified.append(copy)
    return classified

def normalize_label_text(text: object) -> str:
    value = str(text or "").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"(?<=\d)[.,]\s+(?=\d{4}\b)", ", ", value)
    value = value.strip(" .,:;|")
    value = re.sub(r"\s*-+\s*$", "", value)
    return value

def collect_mllm_label_set(axis: dict[str, object], key: str) -> set[str]:
    values: set[str] = set()
    if not isinstance(axis, dict):
        return values
    tick_values = collect_mllm_label_set(axis, "tick_labels") if key == "axis_label" else set()
    raw = axis.get(key, [])
    if key == "axis_label":
        raw = [raw]
    if not isinstance(raw, list):
        raw = [raw]
    for item in raw:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = item
        normalized = normalize_label_text(text)
        if key == "axis_label" and normalized in tick_values:
            continue
        if normalized and normalized not in {"none", "null", "unknown", "n/a"}:
            values.add(normalized)
    return values

def numeric_label_set(values: set[str]) -> set[float]:
    result: set[float] = set()
    for text in values:
        parsed = parse_numeric_label(text)
        if parsed is not None:
            result.add(round(float(parsed), 8))
    return result

def role_geometry_score(item: dict[str, object], role: str, image_shape: tuple[int, int]) -> float:
    h, w = image_shape
    center = item.get("center", [0.0, 0.0])
    size = item.get("size", [0.0, 0.0])
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return 0.0
    if not isinstance(size, (list, tuple)) or len(size) < 2:
        size = [0.0, 0.0]
    cx = float(center[0])
    cy = float(center[1])
    width = float(size[0])
    height = float(size[1])
    vertical_text = height > width * 1.5
    if role == "x_axis":
        score = 0.0
        if cy >= h * 0.72:
            score += 0.25
        elif cy >= h * 0.62:
            score += 0.12
        if w * 0.04 <= cx <= w * 0.96:
            score += 0.08
        if not vertical_text:
            score += 0.04
        return min(0.35, score)
    if role == "y_axis":
        score = 0.0
        if cx <= w * 0.24:
            score += 0.25
        elif cx <= w * 0.36:
            score += 0.12
        if cy >= h * 0.10:
            score += 0.06
        if vertical_text:
            score += 0.06
        return min(0.35, score)
    return 0.0

def text_matches_label_set(
    text: str,
    labels: set[str],
    numeric_values: set[float],
) -> tuple[bool, str]:
    normalized = normalize_label_text(text)
    if normalized in labels:
        return True, "text"
    for label in labels:
        if numeric_label_shape_compatible(text, label):
            return True, "numeric_shape"
    parsed = parse_numeric_label(text)
    if parsed is None:
        return False, ""
    rounded = round(float(parsed), 8)
    # Numeric equality alone is intentionally conservative. OCR strings such as
    # "0000" should not be treated as the tick "0" just because they parse to 0.
    if rounded in numeric_values and digit_signature(text) in {"", "0"}:
        return True, "numeric"
    return False, ""

def digit_signature(text: str) -> str:
    return re.sub(r"\D+", "", str(text or ""))

def has_alpha(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text or "")))

def compact_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())

def bounded_edit_distance(left: str, right: str, limit: int = 2) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]

def numeric_label_shape_compatible(ocr_text: str, label_text: str) -> bool:
    if normalize_label_text(ocr_text) == normalize_label_text(label_text):
        return True
    if has_alpha(ocr_text) or has_alpha(label_text):
        return False
    ocr_num = parse_numeric_label(ocr_text)
    label_num = parse_numeric_label(label_text)
    if ocr_num is None or label_num is None:
        return False
    ocr_digits = digit_signature(ocr_text)
    label_digits = digit_signature(label_text)
    if not ocr_digits or not label_digits:
        return False
    if ocr_digits == label_digits:
        return True
    distance = bounded_edit_distance(ocr_digits, label_digits, limit=1)
    scale = max(1.0, abs(float(ocr_num)), abs(float(label_num)))
    numerically_close = abs(float(ocr_num) - float(label_num)) <= max(1e-6, scale * 0.001)
    if numerically_close:
        # Equal numeric value still needs similar visual evidence. This blocks
        # "0000" -> "0" while allowing minor OCR punctuation/leading-zero noise.
        return abs(len(ocr_digits) - len(label_digits)) <= 1 and distance <= 1
    if (
        len(ocr_digits) == len(label_digits)
        and len(ocr_digits) >= 3
        and set(ocr_digits) == {"0"}
        and label_digits.count("0") >= len(label_digits) - 1
    ):
        return distance <= 1
    return False

def mllm_axis_ticks(mllm_result: dict[str, object], axis_key: str) -> list[str]:
    if not isinstance(mllm_result, dict) or mllm_result.get("error") is not None:
        return []
    axis = mllm_result.get(axis_key, {})
    if not isinstance(axis, dict):
        return []
    raw = axis.get("tick_labels", [])
    if not isinstance(raw, list):
        return []
    ticks: list[str] = []
    for item in raw:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        if text and normalize_label_text(text) not in {"none", "null", "unknown", "n/a"}:
            ticks.append(text)
    return ticks

def collect_mllm_other_texts(mllm_result: dict[str, object]) -> set[str]:
    if not isinstance(mllm_result, dict) or mllm_result.get("error") is not None:
        return set()
    raw = mllm_result.get("other_texts", [])
    if not isinstance(raw, list):
        return set()
    values: set[str] = set()
    for item in raw:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        normalized = normalize_label_text(text)
        if normalized and normalized not in {"none", "null", "unknown", "n/a"}:
            values.add(normalized)
    return values

def text_matches_mllm_other_text(text: str, other_texts: set[str]) -> bool:
    normalized = normalize_label_text(text)
    if not normalized or normalized in {"none", "null", "unknown", "n/a"}:
        return False
    if normalized in other_texts:
        return True
    key = compact_text_key(normalized)
    if len(key) < 8:
        return False
    for other in other_texts:
        other_key = compact_text_key(other)
        if len(other_key) < 8:
            continue
        if key == other_key:
            return True
        shorter, longer = (key, other_key) if len(key) <= len(other_key) else (other_key, key)
        if len(shorter) >= 16 and shorter in longer:
            return True
    return False

def is_regression_numeric_label(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if re.search(r"[A-Za-z]", value):
        return False
    return parse_numeric_label(value) is not None

def is_informative_tick_label(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", str(text or "")))

def is_regular_numeric_tick_sequence(labels: list[str]) -> bool:
    values: list[float] = []
    for label in labels:
        if not is_regression_numeric_label(label):
            return False
        parsed = parse_numeric_label(label)
        if parsed is None:
            return False
        values.append(float(parsed))
    if len(values) < 4:
        return False
    diffs = np.diff(np.array(values, dtype=np.float64))
    if len(diffs) == 0:
        return False
    median_diff = float(np.median(diffs))
    if abs(median_diff) <= 1e-9:
        return False
    mad = float(np.median(np.abs(diffs - median_diff)))
    return mad <= max(1e-6, abs(median_diff) * 0.08)

def numeric_parse_text(text: str) -> str:
    value = str(text or "")
    value = value.replace("'", "").replace("’", "").replace("`", "")
    value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)
    return value

def tick_text_in_merged_ocr(merged_text: str, tick_text: str) -> bool:
    merged_norm = normalize_label_text(merged_text)
    tick_norm = normalize_label_text(tick_text)
    if not merged_norm or not tick_norm:
        return False
    tick_words = re.findall(r"[a-z]+", tick_norm)
    if tick_words and not all(word in merged_norm for word in tick_words):
        return False
    if tick_norm in merged_norm and not (
        parse_numeric_label(numeric_parse_text(tick_text)) is not None
        and len(re.findall(r"[-+]?\d[^\sA-Za-z]*(?:\.\d+)?", merged_text)) >= 2
    ):
        return True
    merged_num = parse_numeric_label(numeric_parse_text(merged_text))
    tick_num = parse_numeric_label(numeric_parse_text(tick_text))
    if tick_num is None:
        return False
    numbers = re.findall(r"[-+]?\d[\d,'’`]*(?:\.\d+)?", merged_text)
    for number in numbers:
        parsed = parse_numeric_label(numeric_parse_text(number))
        if parsed is None:
            continue
        scale = max(1.0, abs(float(tick_num)), abs(float(parsed)))
        if abs(float(parsed) - float(tick_num)) <= max(1e-6, min(scale * 0.02, 0.5)):
            return True
    if merged_num is not None:
        scale = max(1.0, abs(float(tick_num)), abs(float(merged_num)))
        return abs(float(merged_num) - float(tick_num)) <= max(1e-6, min(scale * 0.02, 0.5))
    return False

def merged_tick_match_indices(text: str, ticks: list[str]) -> list[int]:
    return [index for index, tick in enumerate(ticks) if tick_text_in_merged_ocr(text, tick)]

def text_exactly_matches_axis_tick(text: str, ticks: list[str]) -> bool:
    normalized = normalize_label_text(text)
    if not normalized:
        return False
    return any(normalized == normalize_label_text(tick) for tick in ticks)

def has_merge_separator_evidence(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    if re.search(r"[,;/|]+", value):
        return True
    numeric_fragments = re.findall(r"[-+]?\d[\d,'’`]*(?:\.\d+)?", value)
    if len(numeric_fragments) >= 2:
        return True
    words = re.findall(r"[A-Za-z]+", value)
    # Multiple complete words can indicate merged category labels; a single word
    # like "Southwest" must not be split just because it contains "West".
    return len(words) >= 2

def contiguous_index_runs(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ordered = sorted(set(int(index) for index in indices))
    runs: list[list[int]] = [[ordered[0]]]
    for index in ordered[1:]:
        if index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs

def best_merged_tick_run(text: str, ticks: list[str]) -> list[int]:
    if text_exactly_matches_axis_tick(text, ticks):
        return []
    normalized_ticks = [normalize_label_text(tick) for tick in ticks]
    compact_text = re.sub(r"[^a-z0-9]+", "", normalize_label_text(text))
    single_char_sequence = (
        2 <= len(compact_text) <= 4
        and len(ticks) >= 4
        and all(len(tick) == 1 and re.search(r"[a-z0-9]", tick) for tick in normalized_ticks)
    )
    if not has_merge_separator_evidence(text) and not single_char_sequence:
        return []
    matches = merged_tick_match_indices(text, ticks)
    runs = contiguous_index_runs(matches)
    if not runs and not single_char_sequence:
        return []
    text_fragment_count = len(re.findall(r"[-+]?\d[\d,'`]*(?:\.\d+)?", str(text or "")))
    if text_fragment_count <= 0:
        text_fragment_count = len(re.findall(r"\S+", str(text or "")))
    plausible_runs = [
        run
        for run in runs
        if len(run) >= 2 and (text_fragment_count <= 1 or len(run) <= text_fragment_count + 1)
    ]
    if not plausible_runs:
        ordered = sorted(set(matches))
        if single_char_sequence and len(ordered) >= 2:
            expanded = list(range(ordered[0], ordered[-1] + 1))
            if len(expanded) <= max(4, len(ordered) + 3):
                return expanded
        return []
    return max(plausible_runs, key=lambda run: (len(run), -run[0]))

def split_box_along_axis(box: list[list[float]], count: int, axis_key: str) -> list[list[list[float]]]:
    points = np.array(box, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2 or count <= 1:
        return []
    x0, y0 = float(points[:, 0].min()), float(points[:, 1].min())
    x1, y1 = float(points[:, 0].max()), float(points[:, 1].max())
    sub_boxes: list[list[list[float]]] = []
    if axis_key == "x_axis":
        anchors = np.linspace(x0, x1, count)
        gap = float(np.median(np.diff(anchors))) if count >= 2 else max(1.0, x1 - x0)
        half_width = max(3.0, gap * 0.34)
        for anchor in anchors:
            left = max(x0, float(anchor) - half_width)
            right = min(x1, float(anchor) + half_width)
            sub_boxes.append([[left, y0], [right, y0], [right, y1], [left, y1]])
    else:
        anchors = np.linspace(y0, y1, count)
        gap = float(np.median(np.diff(anchors))) if count >= 2 else max(1.0, y1 - y0)
        half_height = max(3.0, gap * 0.34)
        for anchor in anchors:
            top = max(y0, float(anchor) - half_height)
            bottom = min(y1, float(anchor) + half_height)
            sub_boxes.append([[x0, top], [x1, top], [x1, bottom], [x0, bottom]])
    return sub_boxes

def classify_single_ocr_item(item: dict[str, object], image_shape: tuple[int, int]) -> dict[str, object]:
    classified = classify_axis_ocr_results([item], image_shape)
    return classified[0] if classified else dict(item)

def split_merged_ocr_items_with_mllm(
    ocr_items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not ocr_items or not isinstance(mllm_result, dict) or mllm_result.get("error") is not None:
        return [dict(item) for item in ocr_items], []
    split_events: list[dict[str, object]] = []
    output: list[dict[str, object]] = []
    axis_ticks = {
        "x_axis": mllm_axis_ticks(mllm_result, "x_axis"),
        "y_axis": mllm_axis_ticks(mllm_result, "y_axis"),
    }
    for item in ocr_items:
        text = str(item.get("text", "") or "").strip()
        box = item.get("box")
        if not text or not isinstance(box, (list, tuple)) or len(box) < 4:
            output.append(dict(item))
            continue
        best_axis = ""
        best_indices: list[int] = []
        for axis_key, ticks in axis_ticks.items():
            if len(ticks) < 4:
                continue
            indices = best_merged_tick_run(text, ticks)
            if len(indices) > len(best_indices):
                best_axis = axis_key
                best_indices = indices
        if not best_axis:
            output.append(dict(item))
            continue
        ticks = axis_ticks[best_axis]
        if len(best_indices) < 2:
            output.append(dict(item))
            continue
        points = np.array(box, dtype=np.float32)
        width = float(points[:, 0].max() - points[:, 0].min())
        height = float(points[:, 1].max() - points[:, 1].min())
        selected_ticks = [ticks[index] for index in best_indices]
        min_span = 32.0 if len(selected_ticks) <= 2 else 48.0
        long_enough = width >= max(min_span, height * 2.2) if best_axis == "x_axis" else height >= max(min_span, width * 2.2)
        if not long_enough:
            output.append(dict(item))
            continue
        sub_boxes = split_box_along_axis(list(box), len(selected_ticks), best_axis)
        if len(sub_boxes) != len(selected_ticks):
            output.append(dict(item))
            continue
        parent_text = text
        split_event_id = f"{best_axis}:{len(split_events)}"
        for index, tick in enumerate(selected_ticks):
            child = {
                "text": tick,
                "score": round(float(item.get("score", 0.0) or 0.0) * 0.92, 3),
                "box": sub_boxes[index],
                "role": best_axis,
                "split_from_merged": True,
                "split_parent": parent_text,
                "split_event_id": split_event_id,
                "split_index": index,
                "split_global_tick_index": best_indices[index],
                "split_count": len(selected_ticks),
                "split_source": "mllm_tick_sequence",
            }
            output.append(classify_single_ocr_item(child, image_shape))
            output[-1]["role"] = best_axis
            output[-1]["split_from_merged"] = True
            output[-1]["split_parent"] = parent_text
            output[-1]["split_event_id"] = split_event_id
            output[-1]["split_index"] = index
            output[-1]["split_global_tick_index"] = best_indices[index]
            output[-1]["split_count"] = len(selected_ticks)
            output[-1]["split_source"] = "mllm_tick_sequence"
        split_events.append(
            {
                "id": split_event_id,
                "text": parent_text,
                "axis": best_axis,
                "matched_count": len(best_indices),
                "tick_count": len(selected_ticks),
                "generated": selected_ticks,
                "generated_indices": best_indices,
                "box": box,
            }
        )
    return output, split_events

def compact_label_key(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_label_text(text))

def item_axis_role(item: dict[str, object]) -> str:
    role = str(item.get("role", "") or "")
    raw_role = str(item.get("raw_role", "") or item.get("raw_role_before_cluster", "") or "")
    return role if role in {"x_axis", "y_axis"} else raw_role

def merge_split_ocr_items_with_mllm(
    ocr_items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not ocr_items or not isinstance(mllm_result, dict) or mllm_result.get("error") is not None:
        return [dict(item) for item in ocr_items], []
    h, w = image_shape
    output = [dict(item) for item in ocr_items]
    merged_by_sources: dict[tuple[int, ...], dict[str, object]] = {}
    used_sources: set[int] = set()
    events: list[dict[str, object]] = []
    for axis_key in ("x_axis", "y_axis"):
        labels = mllm_axis_ticks(mllm_result, axis_key)
        if len(labels) < 4 or is_regular_numeric_tick_sequence(labels):
            continue
        axis_type = str((mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}).get("type", "")).lower()
        if axis_type not in {"time", "date", "category"} and not is_mllm_category_axis(mllm_result, axis_key, labels):
            continue
        label_keys = [compact_label_key(label) for label in labels]
        exact_anchor_points: list[tuple[int, float]] = []
        for anchor_index, anchor_key in enumerate(label_keys):
            if not anchor_key:
                continue
            for item in output:
                if compact_label_key(item.get("text", "")) != anchor_key:
                    continue
                center = item.get("center", [0.0, 0.0])
                if not isinstance(center, (list, tuple)) or len(center) < 2:
                    continue
                exact_anchor_points.append((anchor_index, float(center[0] if axis_key == "x_axis" else center[1])))
                break
        expected_slope: float | None = None
        expected_intercept: float | None = None
        if len(exact_anchor_points) >= 3:
            slopes: list[float] = []
            for left_pos, (left_index, left_coord) in enumerate(exact_anchor_points):
                for right_index, right_coord in exact_anchor_points[left_pos + 1 :]:
                    if right_index == left_index:
                        continue
                    slopes.append((right_coord - left_coord) / float(right_index - left_index))
            if slopes:
                slope = float(np.median(np.array(slopes, dtype=np.float64)))
                if abs(slope) > 1.0:
                    intercept = float(
                        np.median(
                            np.array(
                                [coord - slope * label_index for label_index, coord in exact_anchor_points],
                                dtype=np.float64,
                            )
                        )
                    )
                    residuals = [
                        abs(coord - (slope * label_index + intercept))
                        for label_index, coord in exact_anchor_points
                    ]
                    if float(np.median(np.array(residuals, dtype=np.float64))) <= max(8.0, abs(slope) * 0.35):
                        expected_slope = slope
                        expected_intercept = intercept
        exact_existing = {
            compact_label_key(item.get("text", ""))
            for item in output
            if compact_label_key(item.get("text", "")) in set(label_keys)
        }
        candidates: list[dict[str, object]] = []
        for index, item in enumerate(output):
            if index in used_sources:
                continue
            text = str(item.get("text", "") or "").strip()
            key = compact_label_key(text)
            if not text or not key or key in exact_existing:
                continue
            role = str(item.get("role", "") or "")
            raw_role = str(item.get("raw_role", "") or item.get("raw_role_before_cluster", "") or "")
            if role not in {axis_key, "other"} and raw_role != axis_key:
                continue
            center = item.get("center", [0.0, 0.0])
            size = item.get("size", [0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            if not isinstance(size, (list, tuple)) or len(size) < 2:
                size = [0.0, 0.0]
            cx = float(center[0])
            cy = float(center[1])
            if axis_key == "x_axis" and cy < h * 0.45:
                continue
            if axis_key == "y_axis" and cx > w * 0.55:
                continue
            if not any(key in label_key or label_key in key for label_key in label_keys):
                continue
            candidates.append(
                {
                    "index": index,
                    "item": item,
                    "key": key,
                    "coord": cx if axis_key == "x_axis" else cy,
                    "perp": cy if axis_key == "x_axis" else cx,
                    "score": float(item.get("score", 0.0) or 0.0),
                }
            )
        if len(candidates) < 2:
            continue
        for label_index, label in enumerate(labels):
            label_key = label_keys[label_index]
            if not label_key or label_key in exact_existing:
                continue
            available = [candidate for candidate in candidates if int(candidate["index"]) not in used_sources]
            parts = [candidate for candidate in available if str(candidate["key"]) in label_key and str(candidate["key"]) != label_key]
            if len(parts) < 2:
                continue
            best: tuple[float, tuple[dict[str, object], ...], str] | None = None
            for left_pos, left in enumerate(parts):
                for right in parts[left_pos + 1 :]:
                    pair = (left, right)
                    for ordered in (pair, tuple(reversed(pair))):
                        combined_key = "".join(str(part["key"]) for part in ordered)
                        distance = bounded_edit_distance(combined_key, label_key, limit=max(2, int(len(label_key) * 0.25)))
                        prefix_ok = label_key.startswith(combined_key) and len(combined_key) >= max(4, int(len(label_key) * 0.68))
                        suffix_ok = label_key.endswith(combined_key) and len(combined_key) >= max(4, int(len(label_key) * 0.68))
                        if distance > 1 and not prefix_ok and not suffix_ok:
                            continue
                        keys = [str(part["key"]) for part in ordered]
                        has_alpha = any(re.search(r"[a-z]", key) for key in keys)
                        has_digit = any(re.search(r"\d", key) for key in keys)
                        if re.search(r"[a-z]", label_key) and re.search(r"\d", label_key) and not (has_alpha and has_digit):
                            continue
                        coords = [float(part["coord"]) for part in ordered]
                        perps = [float(part["perp"]) for part in ordered]
                        coord_span = max(coords) - min(coords)
                        perp_span = max(perps) - min(perps)
                        if coord_span > max(70.0, (w if axis_key == "x_axis" else h) * 0.16):
                            continue
                        if perp_span > max(55.0, (h if axis_key == "x_axis" else w) * 0.18):
                            continue
                        score = float(distance) * 10.0 + coord_span * 0.12 + perp_span * 0.04 - sum(float(part["score"]) for part in ordered)
                        if prefix_ok or suffix_ok:
                            score += 3.0
                        if expected_slope is not None and expected_intercept is not None:
                            expected_coord = expected_slope * label_index + expected_intercept
                            merged_coord = float(np.median(np.array(coords, dtype=np.float64)))
                            expected_distance = abs(merged_coord - expected_coord)
                            if expected_distance > max(45.0, abs(expected_slope) * 1.45):
                                continue
                            score += expected_distance * 0.45
                        if best is None or score < best[0]:
                            best = (score, ordered, combined_key)
            if best is None:
                continue
            _, selected, combined_key = best
            source_indexes = tuple(sorted(int(part["index"]) for part in selected))
            if any(index in used_sources for index in source_indexes):
                continue
            boxes = [np.array(output[index].get("box"), dtype=np.float32) for index in source_indexes]
            if any(box.ndim != 2 or box.shape[0] < 4 for box in boxes):
                continue
            all_points = np.concatenate(boxes, axis=0)
            x0 = float(np.min(all_points[:, 0]))
            x1 = float(np.max(all_points[:, 0]))
            y0 = float(np.min(all_points[:, 1]))
            y1 = float(np.max(all_points[:, 1]))
            merge_expected_coord: float | None = None
            if expected_slope is not None and expected_intercept is not None:
                merge_expected_coord = float(expected_slope * label_index + expected_intercept)
                if axis_key == "x_axis":
                    width = min(float(x1 - x0), max(8.0, abs(expected_slope) * 0.82))
                    cx = float(np.clip(merge_expected_coord, 0.0, float(w - 1)))
                    x0 = max(0.0, cx - width * 0.5)
                    x1 = min(float(w - 1), cx + width * 0.5)
                else:
                    height = min(float(y1 - y0), max(8.0, abs(expected_slope) * 0.82))
                    cy = float(np.clip(merge_expected_coord, 0.0, float(h - 1)))
                    y0 = max(0.0, cy - height * 0.5)
                    y1 = min(float(h - 1), cy + height * 0.5)
            merged = {
                "text": label,
                "canonical_text": label,
                "score": round(float(np.mean([float(output[index].get("score", 0.0) or 0.0) for index in source_indexes])) * 0.94, 3),
                "box": rect_box(x0, y0, x1, y1),
                "center": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
                "size": [float(x1 - x0), float(y1 - y0)],
                "role": axis_key,
                "raw_role": axis_key,
                "role_source": "mllm_split_ocr_merge",
                "role_reason": "adjacent_ocr_fragments_match_mllm_tick",
                "text_source": "mllm_axis_tick",
                "label_kind": "tick_label",
                "canonical_axis": axis_key,
                "canonical_index": label_index,
                "canonical_match_source": "split_ocr_fragment_merge",
                "merged_from_split_ocr": True,
                "merge_source_indexes": list(source_indexes),
                "merge_source_texts": [str(output[index].get("text", "") or "") for index in source_indexes],
                "merge_compact_key": combined_key,
            }
            if merge_expected_coord is not None:
                merged["merge_expected_coord"] = round(float(merge_expected_coord), 3)
                merged["bbox_regularized"] = True
                merged["bbox_regularize_source"] = "split_ocr_merge_axis_fit"
            merged_by_sources[source_indexes] = merged
            used_sources.update(source_indexes)
            events.append(
                {
                    "axis": axis_key,
                    "label_index": label_index,
                    "text": label,
                    "source_texts": merged["merge_source_texts"],
                    "source_indexes": list(source_indexes),
                    "box": merged["box"],
                }
            )
    if not merged_by_sources:
        return output, []
    source_to_group = {index: group for group in merged_by_sources for index in group}
    rebuilt: list[dict[str, object]] = []
    emitted: set[tuple[int, ...]] = set()
    for index, item in enumerate(output):
        group = source_to_group.get(index)
        if group is None:
            rebuilt.append(item)
            continue
        if group in emitted:
            continue
        rebuilt.append(merged_by_sources[group])
        emitted.add(group)
    return rebuilt, events

def box_rect(box: object) -> tuple[int, int, int, int] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        points = np.array([[float(point[0]), float(point[1])] for point in box], dtype=np.float32)
    except (TypeError, ValueError, IndexError):
        return None
    if points.ndim != 2 or points.shape[0] < 4:
        return None
    return (
        int(np.floor(float(points[:, 0].min()))),
        int(np.floor(float(points[:, 1].min()))),
        int(np.ceil(float(points[:, 0].max()))),
        int(np.ceil(float(points[:, 1].max()))),
    )

def rect_box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

def contiguous_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(active.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(active) - 1))
    return runs

def merge_close_runs(runs: list[tuple[int, int]], max_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged

def label_width_units(label: str) -> float:
    units = 0.0
    for char in str(label or ""):
        if char.isspace():
            continue
        if char in ".,:;'`":
            units += 0.35
        elif char in "-+|":
            units += 0.45
        elif char in "1Il":
            units += 0.55
        else:
            units += 1.0
    return max(1.0, units)

def choose_label_run_groups(
    runs: list[tuple[int, int]],
    count: int,
    length: int,
    labels: list[str] | None = None,
) -> tuple[list[tuple[int, int]], str] | None:
    if count <= 1 or len(runs) < count:
        return None
    labels = labels or [""] * count
    units = np.array([label_width_units(label) for label in labels], dtype=np.float64)
    best: tuple[float, list[tuple[int, int]], str] | None = None
    run_count = len(runs)
    max_extra = min(4, max(0, run_count - count))
    for start_run in range(run_count):
        for end_run in range(start_run + count - 1, run_count):
            window_count = end_run - start_run + 1
            if window_count - count > max_extra:
                continue
            split_slots = list(range(start_run, end_run))
            for seps in combinations(split_slots, count - 1):
                group_runs: list[tuple[int, int]] = []
                cursor = start_run
                for sep in seps:
                    group_runs.append((cursor, sep))
                    cursor = sep + 1
                group_runs.append((cursor, end_run))
                groups = [(runs[left][0], runs[right][1]) for left, right in group_runs]
                spans = np.array([max(1, end - start + 1) for start, end in groups], dtype=np.float64)
                scale = float(np.median(spans / units))
                expected = units * max(1.0, scale)
                width_penalty = float(np.mean(np.abs(spans - expected) / np.maximum(1.0, expected)))
                if np.any(spans < np.maximum(3.0, expected * 0.45)):
                    width_penalty += 0.8
                selected_gaps = [
                    runs[sep + 1][0] - runs[sep][1] - 1
                    for sep in seps
                ]
                median_span = float(np.median(spans))
                gap_reward = float(np.mean(selected_gaps)) / max(1.0, median_span)
                internal_gap_penalty = 0.0
                for left, right in group_runs:
                    if right <= left:
                        continue
                    inner_gaps = [
                        runs[index + 1][0] - runs[index][1] - 1
                        for index in range(left, right)
                    ]
                    if inner_gaps:
                        internal_gap_penalty += max(0.0, max(inner_gaps) - max(selected_gaps)) / max(1.0, median_span)
                dropped = start_run + (run_count - end_run - 1)
                drop_penalty = dropped * 0.22
                if start_run > 0:
                    left_gap = runs[start_run][0] - runs[start_run - 1][1] - 1
                    drop_penalty -= min(0.18, max(0.0, left_gap) / max(1.0, median_span) * 0.08)
                if end_run < run_count - 1:
                    right_gap = runs[end_run + 1][0] - runs[end_run][1] - 1
                    drop_penalty -= min(0.18, max(0.0, right_gap) / max(1.0, median_span) * 0.08)
                edge_penalty = 0.0
                if groups[0][0] <= 1:
                    edge_penalty += 0.08
                if groups[-1][1] >= length - 2:
                    edge_penalty += 0.08
                score = width_penalty + internal_gap_penalty + drop_penalty + edge_penalty - min(0.55, gap_reward * 0.22)
                if best is None or score < best[0]:
                    method = "projection_label_width_scored"
                    if dropped:
                        method += "_trimmed_edge_noise"
                    best = (score, groups, method)
    if best is None:
        return None
    return best[1], best[2]

def split_runs_into_groups(
    runs: list[tuple[int, int]],
    count: int,
    length: int,
    labels: list[str] | None = None,
) -> tuple[list[tuple[int, int]], str]:
    if count <= 1:
        return [(0, max(0, length - 1))], "single"
    scored = choose_label_run_groups(runs, count, length, labels)
    if scored is not None:
        return scored
    if len(runs) >= count:
        gaps = [
            (runs[index + 1][0] - runs[index][1] - 1, index)
            for index in range(len(runs) - 1)
        ]
        separators = sorted(index for _, index in sorted(gaps, reverse=True)[: count - 1])
        groups: list[tuple[int, int]] = []
        start_run = 0
        for sep in separators:
            groups.append((runs[start_run][0], runs[sep][1]))
            start_run = sep + 1
        groups.append((runs[start_run][0], runs[-1][1]))
        return groups, "projection_largest_gaps"

    anchors = np.linspace(0, length - 1, count + 1)
    groups = []
    for index in range(count):
        left = int(round(anchors[index]))
        right = int(round(anchors[index + 1]))
        if index:
            left += 1
        groups.append((left, max(left, right)))
    return groups, "uniform_fallback"

def threshold_text_crop(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8,
    )
    mask = cv2.bitwise_or(otsu, adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def projection_refine_boxes(
    image: np.ndarray,
    parent_box: object,
    labels: list[str],
    axis_key: str,
) -> tuple[list[list[list[float]]], dict[str, object]]:
    rect = box_rect(parent_box)
    if rect is None or len(labels) <= 1:
        return [], {"enabled": False, "reason": "invalid_parent_or_count"}
    h, w = image.shape[:2]
    x0, y0, x1, y1 = rect
    pad = 4
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad)
    y1 = min(h - 1, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return [], {"enabled": False, "reason": "empty_parent_crop"}

    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    text_mask = threshold_text_crop(crop)
    horizontal_axis = axis_key == "x_axis"
    if horizontal_axis:
        projection = np.sum(text_mask > 0, axis=0)
        threshold = max(1.0, float(np.max(projection)) * 0.08)
        runs = contiguous_runs(projection >= threshold)
        runs = merge_close_runs(runs, max_gap=max(1, crop.shape[1] // 180))
        groups, method = split_runs_into_groups(runs, len(labels), crop.shape[1], labels)
    else:
        projection = np.sum(text_mask > 0, axis=1)
        threshold = max(1.0, float(np.max(projection)) * 0.08)
        runs = contiguous_runs(projection >= threshold)
        runs = merge_close_runs(runs, max_gap=max(1, crop.shape[0] // 180))
        groups, method = split_runs_into_groups(runs, len(labels), crop.shape[0], labels)

    boxes: list[list[list[float]]] = []
    for start, end in groups:
        if horizontal_axis:
            submask = text_mask[:, max(0, start) : min(text_mask.shape[1], end + 1)]
            ys, xs = np.where(submask > 0)
            if len(xs):
                left = x0 + max(0, start) + int(xs.min()) - 2
                right = x0 + max(0, start) + int(xs.max()) + 2
                top = y0 + int(ys.min()) - 2
                bottom = y0 + int(ys.max()) + 2
            else:
                left, right = x0 + start, x0 + end
                top, bottom = y0, y1
        else:
            submask = text_mask[max(0, start) : min(text_mask.shape[0], end + 1), :]
            ys, xs = np.where(submask > 0)
            if len(ys):
                left = x0 + int(xs.min()) - 2
                right = x0 + int(xs.max()) + 2
                top = y0 + max(0, start) + int(ys.min()) - 2
                bottom = y0 + max(0, start) + int(ys.max()) + 2
            else:
                left, right = x0, x1
                top, bottom = y0 + start, y0 + end
        boxes.append(
            rect_box(
                float(max(0, left)),
                float(max(0, top)),
                float(min(w - 1, right)),
                float(min(h - 1, bottom)),
            )
        )

    return boxes, {
        "enabled": True,
        "method": method,
        "axis": axis_key,
        "label_count": len(labels),
        "ink_run_count": len(runs),
        "parent_rect": [x0, y0, x1, y1],
    }

def refine_mllm_split_boxes_by_projection(
    image: np.ndarray,
    items: list[dict[str, object]],
    split_events: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not split_events:
        return [dict(item) for item in items], []
    output = [dict(item) for item in items]
    events: list[dict[str, object]] = []
    for event in split_events:
        axis_key = str(event.get("axis", ""))
        labels = [str(value) for value in event.get("generated", []) if str(value).strip()]
        if axis_key not in {"x_axis", "y_axis"} or len(labels) <= 1:
            continue
        boxes, details = projection_refine_boxes(image, event.get("box"), labels, axis_key)
        if len(boxes) != len(labels):
            continue
        event_id = str(event.get("id", ""))
        if event_id:
            child_indexes = [
                index
                for index, item in enumerate(output)
                if item.get("split_from_merged")
                and str(item.get("split_event_id", "")) == event_id
                and str(item.get("role", "")) == axis_key
            ]
        else:
            child_indexes = [
                index
                for index, item in enumerate(output)
                if item.get("split_from_merged")
                and str(item.get("split_parent", "")) == str(event.get("text", ""))
                and str(item.get("role", "")) == axis_key
            ]
        child_indexes = sorted(child_indexes, key=lambda index: int(output[index].get("split_index", 0)))
        if len(child_indexes) != len(labels):
            continue
        for child_index, box in zip(child_indexes, boxes):
            points = np.array(box, dtype=np.float32)
            output[child_index]["box"] = box
            output[child_index]["center"] = [float(points[:, 0].mean()), float(points[:, 1].mean())]
            output[child_index]["size"] = [
                float(points[:, 0].max() - points[:, 0].min()),
                float(points[:, 1].max() - points[:, 1].min()),
            ]
            output[child_index]["bbox_refined"] = True
            output[child_index]["bbox_refine_source"] = "projection_with_label_sequence"
        events.append(
            {
                "text": event.get("text"),
                "axis": axis_key,
                "generated": labels,
                "details": details,
            }
        )
    return output, events

def set_item_box_from_center(
    item: dict[str, object],
    axis_key: str,
    coord: float,
    perp: float,
    main_size: float,
    perp_size: float,
    image_shape: tuple[int, int],
) -> None:
    h, w = image_shape
    half_main = max(2.0, float(main_size) * 0.5)
    half_perp = max(2.0, float(perp_size) * 0.5)
    if axis_key == "x_axis":
        x0 = max(0.0, coord - half_main)
        x1 = min(float(w - 1), coord + half_main)
        y0 = max(0.0, perp - half_perp)
        y1 = min(float(h - 1), perp + half_perp)
        item["center"] = [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)]
        item["size"] = [float(x1 - x0), float(y1 - y0)]
    else:
        x0 = max(0.0, perp - half_perp)
        x1 = min(float(w - 1), perp + half_perp)
        y0 = max(0.0, coord - half_main)
        y1 = min(float(h - 1), coord + half_main)
        item["center"] = [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)]
        item["size"] = [float(x1 - x0), float(y1 - y0)]
    item["box"] = rect_box(x0, y0, x1, y1)

def regularize_mllm_split_sequence_geometry(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = [dict(item) for item in items]
    events: list[dict[str, object]] = []
    for axis_key in ("x_axis", "y_axis"):
        labels = mllm_axis_ticks(mllm_result, axis_key)
        if len(labels) < 4 or not is_regular_numeric_tick_sequence(labels):
            continue
        points: list[tuple[int, int, float, float, float, float]] = []
        for item_index, item in enumerate(output):
            if not item.get("split_from_merged") or str(item.get("role", "")) != axis_key:
                continue
            raw_global_index = item.get("split_global_tick_index", item.get("canonical_index"))
            try:
                tick_index = int(raw_global_index)
            except (TypeError, ValueError):
                continue
            if tick_index < 0 or tick_index >= len(labels):
                continue
            if not same_label_value(str(item.get("text", "")), labels[tick_index]):
                continue
            center = item.get("center", [0.0, 0.0])
            size = item.get("size", [0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            if not isinstance(size, (list, tuple)) or len(size) < 2:
                continue
            coord = float(center[0] if axis_key == "x_axis" else center[1])
            perp = float(center[1] if axis_key == "x_axis" else center[0])
            main_size = float(size[0] if axis_key == "x_axis" else size[1])
            perp_size = float(size[1] if axis_key == "x_axis" else size[0])
            points.append((tick_index, item_index, coord, perp, main_size, perp_size))
        unique_indexes = {point[0] for point in points}
        if len(points) < 4 or len(unique_indexes) < 4:
            continue
        slopes: list[float] = []
        for left_index in range(len(points)):
            for right_index in range(left_index + 1, len(points)):
                index_a, _, coord_a, _, _, _ = points[left_index]
                index_b, _, coord_b, _, _, _ = points[right_index]
                if index_a == index_b:
                    continue
                slopes.append((coord_b - coord_a) / float(index_b - index_a))
        if not slopes:
            continue
        slope = float(np.median(np.array(slopes, dtype=np.float64)))
        if abs(slope) < 2.0:
            continue
        intercept = float(np.median(np.array([coord - slope * tick_index for tick_index, _, coord, _, _, _ in points], dtype=np.float64)))
        residuals = np.array([abs(coord - (intercept + slope * tick_index)) for tick_index, _, coord, _, _, _ in points], dtype=np.float64)
        median_residual = float(np.median(residuals))
        gap = abs(slope)
        if median_residual > max(12.0, gap * 0.45):
            continue
        perps = np.array([perp for _, _, _, perp, _, _ in points], dtype=np.float64)
        main_sizes = np.array([max(2.0, main_size) for _, _, _, _, main_size, _ in points], dtype=np.float64)
        perp_sizes = np.array([max(2.0, perp_size) for _, _, _, _, _, perp_size in points], dtype=np.float64)
        median_perp = float(np.median(perps))
        median_main_size = float(np.median(main_sizes))
        median_perp_size = float(np.median(perp_sizes))
        adjust_threshold = max(6.0, gap * 0.28)
        size_low = median_main_size * 0.45
        size_high = median_main_size * 1.9
        adjusted: list[dict[str, object]] = []
        for tick_index, item_index, coord, perp, main_size, perp_size in points:
            predicted = intercept + slope * tick_index
            coord_delta = abs(coord - predicted)
            perp_delta = abs(perp - median_perp)
            size_bad = main_size < size_low or main_size > size_high
            if coord_delta <= adjust_threshold and perp_delta <= max(5.0, median_perp_size) and not size_bad:
                continue
            set_item_box_from_center(
                output[item_index],
                axis_key,
                predicted,
                median_perp,
                median_main_size,
                median_perp_size,
                image_shape,
            )
            output[item_index]["bbox_regularized"] = True
            output[item_index]["bbox_regularize_source"] = "regular_numeric_mllm_sequence"
            output[item_index]["bbox_regularize_delta"] = round(float(coord_delta), 3)
            adjusted.append(
                {
                    "tick_index": tick_index,
                    "text": output[item_index].get("text"),
                    "old_coord": round(float(coord), 3),
                    "new_coord": round(float(predicted), 3),
                }
            )
        if adjusted:
            events.append(
                {
                    "axis": axis_key,
                    "method": "regular_numeric_mllm_sequence",
                    "tick_count": len(labels),
                    "evidence_count": len(points),
                    "slope": round(float(slope), 3),
                    "median_residual": round(median_residual, 3),
                    "adjusted": adjusted,
                }
            )
    return output, events

def regularize_canonical_numeric_axis_geometry(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = [dict(item) for item in items]
    events: list[dict[str, object]] = []
    for axis_key in ("x_axis", "y_axis"):
        labels = mllm_axis_ticks(mllm_result, axis_key)
        if len(labels) < 4 or not is_regular_numeric_tick_sequence(labels):
            continue
        points: list[tuple[int, int, float, float, float, float, bool, str]] = []
        for item_index, item in enumerate(output):
            if str(item.get("canonical_axis", item.get("role", ""))) != axis_key:
                continue
            if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
                continue
            raw_index = item.get("canonical_index")
            try:
                tick_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if tick_index < 0 or tick_index >= len(labels):
                continue
            if not same_label_value(str(item.get("canonical_text", item.get("text", "")) or ""), labels[tick_index]):
                continue
            center = item.get("center", [0.0, 0.0])
            size = item.get("size", [0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            if not isinstance(size, (list, tuple)) or len(size) < 2:
                continue
            coord = float(center[0] if axis_key == "x_axis" else center[1])
            perp = float(center[1] if axis_key == "x_axis" else center[0])
            main_size = float(size[0] if axis_key == "x_axis" else size[1])
            perp_size = float(size[1] if axis_key == "x_axis" else size[0])
            is_pseudo = bool(item.get("mllm_pseudo_box"))
            points.append((tick_index, item_index, coord, perp, main_size, perp_size, is_pseudo, labels[tick_index]))
        real_points = [point for point in points if not point[6]]
        if len(real_points) < 4 or len({point[0] for point in real_points}) < 4:
            continue
        slopes: list[float] = []
        for left_index in range(len(real_points)):
            for right_index in range(left_index + 1, len(real_points)):
                index_a, _, coord_a, _, _, _, _, _ = real_points[left_index]
                index_b, _, coord_b, _, _, _, _, _ = real_points[right_index]
                if index_a == index_b:
                    continue
                slopes.append((coord_b - coord_a) / float(index_b - index_a))
        if not slopes:
            continue
        slope = float(np.median(np.array(slopes, dtype=np.float64)))
        if abs(slope) < 2.0:
            continue
        intercept = float(
            np.median(np.array([coord - slope * tick_index for tick_index, _, coord, _, _, _, _, _ in real_points], dtype=np.float64))
        )
        residuals = np.array(
            [abs(coord - (intercept + slope * tick_index)) for tick_index, _, coord, _, _, _, _, _ in real_points],
            dtype=np.float64,
        )
        median_residual = float(np.median(residuals))
        gap = abs(slope)
        if median_residual > max(8.0, gap * 0.22):
            continue
        main_sizes = np.array([max(2.0, point[4]) for point in real_points], dtype=np.float64)
        median_main_size = float(np.median(main_sizes))
        size_high = max(median_main_size * 1.75, gap * 0.52)
        reliable_points = [point for point in real_points if point[4] <= size_high]
        if len(reliable_points) < 3:
            reliable_points = real_points
        perps = np.array([point[3] for point in reliable_points], dtype=np.float64)
        perp_sizes = np.array([max(2.0, point[5]) for point in reliable_points], dtype=np.float64)
        median_perp = float(np.median(perps))
        median_perp_size = float(np.median(perp_sizes))
        median_main_size = float(np.median(np.array([max(2.0, point[4]) for point in reliable_points], dtype=np.float64)))
        sizes_by_digit_count: dict[int, list[float]] = {}
        for _, _, _, _, main_size, _, _, label in reliable_points:
            digits = digit_signature(label)
            key = len(digits) if digits else max(1, len(normalize_label_text(label)))
            sizes_by_digit_count.setdefault(key, []).append(max(2.0, main_size))

        def typical_main_size(label: str) -> float:
            digits = digit_signature(label)
            key = len(digits) if digits else max(1, len(normalize_label_text(label)))
            values = sizes_by_digit_count.get(key)
            if values:
                return float(np.median(np.array(values, dtype=np.float64)))
            return median_main_size

        adjusted: list[dict[str, object]] = []
        for tick_index, item_index, coord, perp, main_size, perp_size, is_pseudo, label in points:
            predicted = intercept + slope * tick_index
            coord_delta = abs(coord - predicted)
            expected_size = typical_main_size(label)
            size_bad = main_size > max(expected_size * 1.75, gap * 0.52) or main_size < max(2.0, expected_size * 0.38)
            pseudo_delta_threshold = max(2.5, gap * 0.075)
            real_delta_threshold = max(6.0, gap * 0.16)
            should_adjust = (is_pseudo and coord_delta > pseudo_delta_threshold) or size_bad or coord_delta > real_delta_threshold
            if not should_adjust:
                continue
            set_item_box_from_center(
                output[item_index],
                axis_key,
                predicted,
                median_perp,
                expected_size,
                median_perp_size,
                image_shape,
            )
            output[item_index]["bbox_regularized"] = True
            output[item_index]["bbox_regularize_source"] = "canonical_regular_numeric_sequence"
            output[item_index]["bbox_regularize_delta"] = round(float(coord_delta), 3)
            adjusted.append(
                {
                    "tick_index": tick_index,
                    "text": output[item_index].get("text"),
                    "old_coord": round(float(coord), 3),
                    "new_coord": round(float(predicted), 3),
                    "reason": "pseudo_sequence_delta" if is_pseudo else ("loose_ocr_box" if size_bad else "sequence_delta"),
                }
            )
        if adjusted:
            events.append(
                {
                    "axis": axis_key,
                    "method": "canonical_regular_numeric_sequence",
                    "tick_count": len(labels),
                    "evidence_count": len(real_points),
                    "slope": round(float(slope), 3),
                    "median_residual": round(median_residual, 3),
                    "adjusted": adjusted,
                }
            )
    return output, events

def numeric_fragment_count(text: str) -> int:
    return len(re.findall(r"[-+]?\d+(?:\.\d+)?", str(text or "")))

def decimal_places(text: str) -> int:
    match = re.search(r"\.(\d+)", str(text or ""))
    return len(match.group(1)) if match else 0

def format_sequence_value(value: float, decimals: int) -> str:
    if decimals <= 0 and abs(value - round(value)) <= 1e-6:
        return str(int(round(value)))
    return f"{value:.{max(0, decimals)}f}".rstrip("0").rstrip(".")

def compact_digit_sequence_labels(text: str) -> list[str]:
    raw = str(text or "").strip()
    digits = [int(value) for value in re.findall(r"\d", raw)]
    if len(digits) < 2 or len(digits) > 10:
        return []
    if all(b - a == 1 for a, b in zip(digits, digits[1:])):
        return [str(value) for value in digits]
    if len(digits) == 2 and digits[1] - digits[0] == 2:
        return [str(value) for value in range(digits[0], digits[1] + 1)]
    return []

def axis_coord(item: dict[str, object], axis_key: str) -> float | None:
    center = item.get("center", [0.0, 0.0])
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return None
    return float(center[0] if axis_key == "x_axis" else center[1])

def visual_order_axis_labels(
    labels: list[str],
    mllm_result: dict[str, object],
    axis_key: str,
) -> list[tuple[int, str]]:
    axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
    order = str(axis.get("tick_order", "") if isinstance(axis, dict) else "")
    indexed = list(enumerate(labels))
    if axis_key == "y_axis" and order == "bottom_to_top":
        return list(reversed(indexed))
    if axis_key == "x_axis" and order == "right_to_left":
        return list(reversed(indexed))
    return indexed

def numeric_axis_sequence_pair_score(ocr_text: str, label: str) -> float:
    if normalize_label_text(ocr_text) == normalize_label_text(label):
        return 0.0
    if numeric_label_shape_compatible(ocr_text, label):
        return 0.22
    ocr_num = parse_numeric_label(ocr_text)
    label_num = parse_numeric_label(label)
    if ocr_num is None or label_num is None:
        return 1.2
    ocr_digits = digit_signature(ocr_text)
    label_digits = digit_signature(label)
    if ocr_digits and label_digits and abs(len(ocr_digits) - len(label_digits)) <= 1:
        distance = bounded_edit_distance(ocr_digits, label_digits, limit=2)
        if distance <= 1:
            return 0.55
        if distance <= 2:
            return 0.78
    return 1.0

def explicit_declared_numeric_axis_order(
    mllm_result: dict[str, object],
    axis_key: str,
    labels: list[str],
) -> tuple[bool, bool]:
    axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
    declared_order = str(axis.get("tick_order", "") if isinstance(axis, dict) else "").strip().lower()
    try:
        axis_confidence = float(axis.get("confidence", 0.0) if isinstance(axis, dict) else 0.0)
    except (TypeError, ValueError):
        axis_confidence = 0.0
    label_values = [parse_numeric_label(label) for label in labels]
    declared_order_numeric_consistent = True
    if declared_order and len(label_values) >= 2 and all(value is not None for value in label_values):
        numeric_delta = float(label_values[-1]) - float(label_values[0])
        if declared_order in {"left_to_right", "bottom_to_top"}:
            declared_order_numeric_consistent = numeric_delta > 0
        elif declared_order in {"right_to_left", "top_to_bottom"}:
            declared_order_numeric_consistent = numeric_delta < 0
    reliable = (
        declared_order in {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}
        and axis_confidence >= 0.85
        and declared_order_numeric_consistent
    )
    reversed_visual = (axis_key == "y_axis" and declared_order == "bottom_to_top") or (axis_key == "x_axis" and declared_order == "right_to_left")
    return reliable, reversed_visual

def regular_numeric_axis_sequence_assignments(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    axis_key: str,
    labels: list[str],
    used_indexes: set[int],
) -> list[tuple[int, int, str]]:
    if len(labels) < 4 or not is_regular_numeric_tick_sequence(labels):
        return []
    reference, tolerance = axis_perpendicular_reference(items, axis_key)
    candidates: list[tuple[float, int, dict[str, object]]] = []
    for index, item in enumerate(items):
        if index in used_indexes:
            continue
        if item.get("role") != axis_key:
            continue
        if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
            continue
        text = str(item.get("text", "") or "").strip()
        if not text or parse_numeric_label(text) is None:
            continue
        if axis_key == "y_axis" and axis_perpendicular_score(item, axis_key, reference, tolerance) < 0.35:
            continue
        coord = axis_coord(item, axis_key)
        if coord is None:
            continue
        candidates.append((coord, index, item))
    candidates.sort(key=lambda value: value[0])
    if len(candidates) < 3 or len(candidates) > len(labels):
        return []

    best: tuple[float, int, list[tuple[int, int, str]]] | None = None
    candidate_count = len(candidates)
    explicit_reliable_order, declared_prefers_reversed = explicit_declared_numeric_axis_order(mllm_result, axis_key, labels)
    preferred_visual_labels = list(reversed(list(enumerate(labels)))) if declared_prefers_reversed else list(enumerate(labels))
    label_orders = [
        ("mllm_declared_order", preferred_visual_labels),
    ] if explicit_reliable_order else [
        ("mllm_list_order", list(enumerate(labels))),
        ("mllm_reversed_order", list(reversed(list(enumerate(labels))))),
    ]
    for order_name, visual_labels in label_orders:
        for offset in range(0, len(visual_labels) - candidate_count + 1):
            selected = visual_labels[offset : offset + candidate_count]
            pair_scores: list[float] = []
            exact_or_shape = 0
            assignments: list[tuple[int, int, str]] = []
            for (_, item_index, item), (label_index, label) in zip(candidates, selected):
                text = str(item.get("text", "") or "")
                score = numeric_axis_sequence_pair_score(text, label)
                pair_scores.append(score)
                if score <= 0.22:
                    exact_or_shape += 1
                assignments.append((label_index, item_index, f"regular_numeric_axis_order:{order_name}"))
            coords = np.array([coord for coord, _, _ in candidates], dtype=np.float64)
            diffs = np.diff(coords)
            if len(diffs) >= 2:
                median_diff = float(np.median(np.abs(diffs)))
                spacing_penalty = 0.0 if median_diff <= 1e-6 else min(0.35, float(np.median(np.abs(np.abs(diffs) - median_diff))) / max(1.0, median_diff))
            else:
                spacing_penalty = 0.0
            edge_penalty = min(0.25, offset * 0.04)
            order_penalty = 0.0
            if order_name != "mllm_declared_order" and ((order_name == "mllm_reversed_order") != declared_prefers_reversed):
                order_penalty = 0.03
            total = float(np.mean(pair_scores)) + spacing_penalty + edge_penalty + order_penalty
            if exact_or_shape == 0:
                total += 0.6
            if best is None or total < best[0]:
                best = (total, exact_or_shape, assignments)
    if best is None:
        return []
    if best[0] > 0.82 or best[1] < 1:
        return []
    return best[2]

def declared_numeric_axis_order_assignments(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    axis_key: str,
    labels: list[str],
    candidate_indexes: list[int],
) -> list[tuple[int, int, str]]:
    if len(labels) < 4 or not is_regular_numeric_tick_sequence(labels):
        return []
    reliable, reversed_visual = explicit_declared_numeric_axis_order(mllm_result, axis_key, labels)
    if not reliable:
        return []
    reference, tolerance = axis_perpendicular_reference(items, axis_key)
    candidates: list[tuple[float, int, dict[str, object]]] = []
    for index in candidate_indexes:
        if index < 0 or index >= len(items):
            continue
        item = items[index]
        if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
            continue
        text = str(item.get("text", "") or "").strip()
        if not text or parse_numeric_label(text) is None:
            continue
        if axis_key == "y_axis" and axis_perpendicular_score(item, axis_key, reference, tolerance) < 0.35:
            continue
        coord = axis_coord(item, axis_key)
        if coord is None:
            continue
        candidates.append((coord, index, item))
    candidates.sort(key=lambda value: value[0])
    if len(candidates) < max(4, int(np.ceil(len(labels) * 0.55))):
        return []
    groups: list[list[tuple[float, int, dict[str, object]]]] = []
    group_gap = 8.0
    for candidate in candidates:
        if groups and abs(candidate[0] - float(np.median([item[0] for item in groups[-1]]))) <= group_gap:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    if len(groups) < max(4, int(np.ceil(len(labels) * 0.55))):
        return []
    visual_labels = list(reversed(list(enumerate(labels)))) if reversed_visual else list(enumerate(labels))
    best: tuple[float, int, list[tuple[int, int, str]]] | None = None
    slot_count = min(len(groups), len(labels))
    group_windows = [groups[offset : offset + slot_count] for offset in range(0, len(groups) - slot_count + 1)]
    for group_window in group_windows:
        group_coords = np.array([float(np.median([candidate[0] for candidate in group])) for group in group_window], dtype=np.float64)
        diffs = np.diff(group_coords)
        spacing_penalty = 0.0
        if len(diffs) >= 2:
            median_diff = float(np.median(np.abs(diffs)))
            if median_diff <= 1e-6:
                continue
            spacing_penalty = min(0.45, float(np.median(np.abs(np.abs(diffs) - median_diff))) / max(1.0, median_diff))
            max_spacing_residual = float(np.max(np.abs(np.abs(diffs) - median_diff))) / max(1.0, median_diff)
            if spacing_penalty > 0.22 or max_spacing_residual > 0.35:
                continue
        for offset in range(0, len(visual_labels) - slot_count + 1):
            selected = visual_labels[offset : offset + slot_count]
            pair_scores: list[float] = []
            chosen: list[tuple[int, int, str]] = []
            for group, (label_index, label) in zip(group_window, selected):
                best_candidate = min(
                    group,
                    key=lambda candidate: (
                        numeric_axis_sequence_pair_score(str(candidate[2].get("text", "") or ""), label),
                        -canonical_candidate_score(candidate[2], axis_key, reference, tolerance),
                    ),
                )
                pair_score = numeric_axis_sequence_pair_score(str(best_candidate[2].get("text", "") or ""), label)
                pair_scores.append(pair_score)
                chosen.append((label_index, best_candidate[1], "regular_numeric_axis_order:mllm_declared_order"))
            exact_or_shape = sum(1 for score in pair_scores if score <= 0.22)
            total = float(np.mean(pair_scores)) + spacing_penalty + min(0.25, offset * 0.04)
            if exact_or_shape < max(2, int(np.ceil(slot_count * 0.35))):
                total += 0.45
            if best is None or total < best[0]:
                best = (total, exact_or_shape, chosen)
    if best is None or best[0] > 0.82:
        return []
    return best[2]

def infer_regular_numeric_gap(
    ordered_items: list[tuple[int, dict[str, object]]],
    candidate_position: int,
    axis_key: str,
) -> list[str]:
    numeric_neighbors: list[tuple[int, float, str, float]] = []
    for pos, (_, item) in enumerate(ordered_items):
        if pos == candidate_position or item.get("split_from_merged"):
            continue
        text = str(item.get("text", "") or "").strip()
        parsed = parse_numeric_label(text)
        coord = axis_coord(item, axis_key)
        if parsed is None or coord is None:
            continue
        numeric_neighbors.append((pos, float(parsed), text, coord))

    prev_values = [value for value in numeric_neighbors if value[0] < candidate_position]
    next_values = [value for value in numeric_neighbors if value[0] > candidate_position]
    if not prev_values or not next_values:
        return []
    _, prev_value, prev_text, _ = prev_values[-1]
    _, next_value, next_text, _ = next_values[0]
    diffs = [
        numeric_neighbors[index + 1][1] - numeric_neighbors[index][1]
        for index in range(len(numeric_neighbors) - 1)
        if abs(numeric_neighbors[index + 1][1] - numeric_neighbors[index][1]) > 1e-9
    ]
    if not diffs:
        return []
    typical_step = float(np.median([abs(diff) for diff in diffs]))
    if typical_step <= 1e-9:
        return []
    direction = 1.0 if next_value > prev_value else -1.0
    step = typical_step * direction
    gap_steps = int(round((next_value - prev_value) / step))
    if gap_steps < 3 or gap_steps > 30:
        return []
    if abs((prev_value + step * gap_steps) - next_value) > max(1e-5, abs(step) * 0.15):
        return []
    decimals = max(decimal_places(prev_text), decimal_places(next_text))
    return [format_sequence_value(prev_value + step * offset, decimals) for offset in range(1, gap_steps)]

def split_numeric_gap_ocr_items(
    items: list[dict[str, object]],
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = [dict(item) for item in items]
    split_by_index: dict[int, dict[str, object]] = {}
    events: list[dict[str, object]] = []
    for axis_key in ("x_axis", "y_axis"):
        indexed = [
            (index, item)
            for index, item in enumerate(output)
            if item.get("role") == axis_key and not item.get("split_from_merged")
        ]
        indexed = [(index, item) for index, item in indexed if axis_coord(item, axis_key) is not None]
        indexed.sort(key=lambda pair: axis_coord(pair[1], axis_key) or 0.0)
        sizes = []
        for _, item in indexed:
            size = item.get("size", [0.0, 0.0])
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                along = float(size[0] if axis_key == "x_axis" else size[1])
                cross = float(size[1] if axis_key == "x_axis" else size[0])
                text = str(item.get("text", "") or "")
                if (
                    parse_numeric_label(text) is not None
                    and numeric_fragment_count(text) <= 1
                    and not compact_digit_sequence_labels(text)
                    and along > 1
                    and cross > 1
                ):
                    sizes.append(along)
        median_single_size = float(np.median(sizes)) if sizes else (12.0 if axis_key == "x_axis" else 16.0)

        for position, (original_index, item) in enumerate(indexed):
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            compact_labels = compact_digit_sequence_labels(text)
            if parse_numeric_label(text) is not None and numeric_fragment_count(text) <= 1 and not compact_labels:
                continue
            size = item.get("size", [0.0, 0.0])
            if not isinstance(size, (list, tuple)) or len(size) < 2:
                continue
            along = float(size[0] if axis_key == "x_axis" else size[1])
            cross = float(size[1] if axis_key == "x_axis" else size[0])
            compact_labels = compact_digit_sequence_labels(text)
            if not compact_labels and len(indexed) < 4:
                continue
            required_along = (
                max(24.0, median_single_size * len(compact_labels) * 0.45, cross * 2.5)
                if compact_labels
                else max(60.0, median_single_size * 3.2, cross * 4.0)
            )
            if along < required_along:
                continue
            labels = compact_labels or infer_regular_numeric_gap(indexed, position, axis_key)
            if len(labels) < 2 or along < median_single_size * len(labels) * 0.45:
                continue
            fragments = numeric_fragment_count(text)
            if fragments < 2 and not compact_labels and float(item.get("score", 1.0) or 1.0) >= 0.75:
                continue
            box = item.get("box")
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            sub_boxes = split_box_along_axis(list(box), len(labels), axis_key)
            if len(sub_boxes) != len(labels):
                continue
            children: list[dict[str, object]] = []
            for child_index, (label, child_box) in enumerate(zip(labels, sub_boxes)):
                points = np.array(child_box, dtype=np.float32)
                children.append(
                    {
                        "text": label,
                        "score": round(float(item.get("score", 0.0) or 0.0) * 0.9, 3),
                        "box": child_box,
                        "role": axis_key,
                        "split_axis": axis_key,
                        "center": [float(points[:, 0].mean()), float(points[:, 1].mean())],
                        "size": [
                            float(points[:, 0].max() - points[:, 0].min()),
                            float(points[:, 1].max() - points[:, 1].min()),
                        ],
                        "split_from_merged": True,
                        "split_parent": text,
                        "split_index": child_index,
                        "split_count": len(labels),
                        "split_source": "ocr_numeric_gap_sequence",
                    }
                )
            split_by_index[original_index] = {"children": children}
            events.append(
                {
                    "text": text,
                    "axis": axis_key,
                    "matched_count": len(labels),
                    "tick_count": len(labels),
                    "generated": labels,
                    "box": box,
                    "source": "ocr_numeric_gap_sequence",
                    "reason": "regular_numeric_neighbors_around_long_ocr_box",
                }
            )

    if not split_by_index:
        return output, []
    rebuilt: list[dict[str, object]] = []
    for index, item in enumerate(output):
        if index in split_by_index:
            rebuilt.extend(split_by_index[index]["children"])
        else:
            rebuilt.append(item)
    return rebuilt, events

def restore_numeric_gap_split_roles(items: list[dict[str, object]]) -> list[dict[str, object]]:
    restored: list[dict[str, object]] = []
    for item in items:
        copy = dict(item)
        if copy.get("split_source") == "ocr_numeric_gap_sequence":
            split_axis = str(copy.get("split_axis") or copy.get("role") or "")
            if split_axis in {"x_axis", "y_axis"} and copy.get("role") != split_axis:
                copy["role"] = split_axis
                copy["role_source"] = "ocr_numeric_gap_sequence_locked"
                copy["role_reason"] = "same_axis_regular_sequence_overrides_incomplete_mllm_tick_list"
                copy["role_confidence"] = max(float(copy.get("role_confidence", 0.0) or 0.0), 0.82)
        restored.append(copy)
    return restored

def same_label_value(left: str, right: str) -> bool:
    return numeric_label_shape_compatible(left, right)

def numeric_prefix_label_value(left: str, right: str) -> bool:
    label_numeric = parse_numeric_label(str(right or ""))
    if label_numeric is None:
        return False
    match = re.match(r"\s*([-+]?\d+(?:\.\d+)?)\b", str(left or ""))
    if not match:
        return False
    if numeric_fragment_count(left) < 2:
        return False
    prefix_numeric = parse_numeric_label(match.group(1))
    return prefix_numeric is not None and abs(float(prefix_numeric) - float(label_numeric)) <= 1e-8

def zero_tick_text(text: object) -> bool:
    value = parse_numeric_label(str(text or ""))
    return value is not None and abs(float(value)) <= 1e-9

def likely_x_axis_origin_zero_candidate(
    output: list[dict[str, object]],
    candidate_index: int,
    mllm_result: dict[str, object],
) -> bool:
    if candidate_index < 0 or candidate_index >= len(output):
        return False
    item = output[candidate_index]
    if not zero_tick_text(item.get("text")):
        return False
    if not any(zero_tick_text(text) for text in mllm_axis_ticks(mllm_result, "x_axis")):
        return False
    if not any(zero_tick_text(text) for text in mllm_axis_ticks(mllm_result, "y_axis")):
        return False
    center = item.get("center", [0.0, 0.0])
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        return False
    try:
        cx = float(center[0])
        cy = float(center[1])
    except (TypeError, ValueError):
        return False

    x_tick_items: list[dict[str, object]] = []
    for other in output:
        if other is item or other.get("role") != "x_axis" or other.get("label_kind") == "axis_label":
            continue
        if zero_tick_text(other.get("text")):
            continue
        if parse_numeric_label(str(other.get("text", "") or "")) is None:
            continue
        other_center = other.get("center", [0.0, 0.0])
        if not isinstance(other_center, (list, tuple)) or len(other_center) < 2:
            continue
        x_tick_items.append(other)
    if len(x_tick_items) < 3:
        return False

    row_values = []
    x_values = []
    heights = []
    for other in x_tick_items:
        other_center = other.get("center", [0.0, 0.0])
        size = other.get("size", [0.0, 0.0])
        try:
            row_values.append(float(other_center[1]))
            x_values.append(float(other_center[0]))
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                heights.append(float(size[1]))
        except (TypeError, ValueError):
            continue
    if len(row_values) < 3 or not x_values:
        return False
    row = float(np.median(np.array(row_values, dtype=np.float64)))
    row_tolerance = max(8.0, min(18.0, (float(np.median(np.array(heights, dtype=np.float64))) if heights else 14.0) * 0.9))
    if abs(cy - row) > row_tolerance:
        return False
    x_step = float(np.median(np.diff(np.sort(np.array(x_values, dtype=np.float64))))) if len(x_values) >= 3 else 30.0
    if cx > min(x_values) - max(8.0, x_step * 0.25):
        return False
    return True

def median_item_box_size(items: list[dict[str, object]], axis_key: str) -> tuple[float, float]:
    widths: list[float] = []
    heights: list[float] = []
    for item in items:
        if item.get("role") != axis_key:
            continue
        if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
            continue
        size = item.get("size", [0.0, 0.0])
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            rect = box_rect(item.get("box"))
            if rect is None:
                continue
            widths.append(float(rect[2] - rect[0]))
            heights.append(float(rect[3] - rect[1]))
            continue
        width = float(size[0])
        height = float(size[1])
        if width > 1 and height > 1:
            widths.append(width)
            heights.append(height)
    if widths and heights:
        return float(np.median(widths)), float(np.median(heights))
    return (34.0, 16.0) if axis_key == "x_axis" else (38.0, 16.0)

def fit_label_position(
    ticks: list[str],
    anchors: list[tuple[int, str, float]],
    target_index: int,
) -> tuple[float | None, str, float]:
    if len(anchors) < 2:
        return None, "insufficient_anchors", 0.0
    numeric_points: list[tuple[float, float]] = []
    for _, text, coord in anchors:
        parsed = parse_numeric_label(text) if is_regression_numeric_label(text) else None
        if parsed is not None:
            numeric_points.append((float(parsed), coord))
    target_numeric = parse_numeric_label(ticks[target_index]) if is_regression_numeric_label(ticks[target_index]) else None
    if target_numeric is not None and len(numeric_points) >= 2:
        xs = np.array([value for value, _ in numeric_points], dtype=np.float64)
        ys = np.array([coord for _, coord in numeric_points], dtype=np.float64)
        if float(np.max(xs) - np.min(xs)) > 1e-6:
            slope, intercept = np.polyfit(xs, ys, 1)
            predicted = float(slope * float(target_numeric) + intercept)
            residual = float(np.median(np.abs(ys - (slope * xs + intercept))))
            step = float(np.median(np.abs(np.diff(np.sort(ys))))) if len(ys) >= 3 else 20.0
            confidence = 0.72 if residual <= max(3.0, step * 0.25) else 0.58
            return predicted, "numeric_regression", confidence
    index_values = np.array([float(index) for index, _, _ in anchors], dtype=np.float64)
    coords = np.array([coord for _, _, coord in anchors], dtype=np.float64)
    if float(np.max(index_values) - np.min(index_values)) <= 1e-6:
        return None, "degenerate_index_anchors", 0.0
    slope, intercept = np.polyfit(index_values, coords, 1)
    predicted = float(slope * float(target_index) + intercept)
    residual = float(np.median(np.abs(coords - (slope * index_values + intercept))))
    step = float(np.median(np.abs(np.diff(np.sort(coords))))) if len(coords) >= 3 else 20.0
    confidence = 0.68 if residual <= max(4.0, step * 0.30) else 0.52
    return predicted, "index_regression", confidence

def regular_numeric_bulk_pseudo_supported(
    ticks: list[str],
    anchors: list[tuple[int, str, float]],
    missing_indexes: list[int],
    axis_key: str,
    image_shape: tuple[int, int],
) -> tuple[bool, dict[str, object]]:
    h, w = image_shape
    extent = float(w if axis_key == "x_axis" else h)
    by_index: dict[int, list[float]] = {}
    for index, _, coord in anchors:
        by_index.setdefault(int(index), []).append(float(coord))
    if len(by_index) < 4:
        return False, {"reason": "too_few_unique_regular_numeric_anchors", "unique_anchor_count": len(by_index)}

    anchor_indexes = sorted(by_index)
    longest_run = 1
    current_run = 1
    for previous, current in zip(anchor_indexes, anchor_indexes[1:]):
        if current == previous + 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
    if longest_run < max(4, int(np.ceil(len(anchor_indexes) * 0.65))):
        return False, {
            "reason": "regular_numeric_anchors_not_contiguous_enough",
            "unique_anchor_count": len(anchor_indexes),
            "longest_contiguous_run": longest_run,
        }

    coverage = len(anchor_indexes) / max(1, len(ticks))
    if len(anchor_indexes) < 8 and coverage < 0.35:
        return False, {
            "reason": "regular_numeric_anchor_coverage_too_low",
            "unique_anchor_count": len(anchor_indexes),
            "coverage": round(float(coverage), 3),
        }

    index_values = np.array(anchor_indexes, dtype=np.float64)
    coords = np.array([float(np.median(np.array(by_index[index], dtype=np.float64))) for index in anchor_indexes], dtype=np.float64)
    if float(np.max(index_values) - np.min(index_values)) <= 1e-6:
        return False, {"reason": "degenerate_regular_numeric_anchor_indexes"}
    slope, intercept = np.polyfit(index_values, coords, 1)
    step = abs(float(slope))
    if step < 2.0:
        return False, {"reason": "regular_numeric_anchor_step_too_small", "step": round(step, 3)}
    fitted = slope * index_values + intercept
    residuals = np.abs(coords - fitted)
    median_residual = float(np.median(residuals))
    max_residual = float(np.max(residuals))
    if median_residual > max(2.0, step * 0.18) or max_residual > max(4.0, step * 0.35):
        return False, {
            "reason": "regular_numeric_anchor_fit_too_noisy",
            "step": round(step, 3),
            "median_residual": round(median_residual, 3),
            "max_residual": round(max_residual, 3),
        }

    margin = max(12.0, step * 0.75)
    predicted = [float(slope * float(index) + intercept) for index in missing_indexes]
    outside = [value for value in predicted if value < -margin or value > extent - 1.0 + margin]
    if outside:
        return False, {
            "reason": "regular_numeric_bulk_prediction_outside_image",
            "outside_count": len(outside),
            "extent": round(extent, 3),
            "margin": round(margin, 3),
        }

    return True, {
        "reason": "regular_numeric_contiguous_anchor_fit",
        "unique_anchor_count": len(anchor_indexes),
        "coverage": round(float(coverage), 3),
        "longest_contiguous_run": longest_run,
        "step": round(step, 3),
        "median_residual": round(median_residual, 3),
        "max_residual": round(max_residual, 3),
    }

def robust_category_axis_fit(
    anchors: list[tuple[int, float, float, float, float]],
    label_count: int,
    axis_key: str,
    image_shape: tuple[int, int] | None,
) -> tuple[float, float, dict[str, object]] | None:
    unique_by_index: dict[int, list[tuple[float, float, float, float]]] = {}
    for label_index, coord, perp, width, height in anchors:
        unique_by_index.setdefault(int(label_index), []).append((float(coord), float(perp), float(width), float(height)))
    if len(unique_by_index) < max(5, min(9, int(np.ceil(label_count * 0.30)))):
        return None

    points: list[tuple[int, float]] = []
    for label_index in sorted(unique_by_index):
        coords = [value[0] for value in unique_by_index[label_index]]
        points.append((label_index, float(np.median(np.array(coords, dtype=np.float64)))))
    if len(points) < 2:
        return None

    slopes: list[float] = []
    for left_pos, (left_index, left_coord) in enumerate(points):
        for right_index, right_coord in points[left_pos + 1 :]:
            delta = right_index - left_index
            if delta == 0:
                continue
            slope = (right_coord - left_coord) / float(delta)
            if abs(slope) > 1.0:
                slopes.append(float(slope))
    if not slopes:
        return None
    slope0 = float(np.median(np.array(slopes, dtype=np.float64)))
    if abs(slope0) <= 1.0:
        return None
    intercept0 = float(np.median(np.array([coord - slope0 * index for index, coord in points], dtype=np.float64)))
    residuals0 = [abs(coord - (slope0 * index + intercept0)) for index, coord in points]
    step = abs(slope0)
    inlier_threshold = max(8.0, step * 0.38)
    inliers = [(index, coord) for (index, coord), residual in zip(points, residuals0) if residual <= inlier_threshold]
    if len(inliers) < max(5, min(8, int(np.ceil(label_count * 0.25)))):
        inlier_threshold = max(10.0, step * 0.50)
        inliers = [(index, coord) for (index, coord), residual in zip(points, residuals0) if residual <= inlier_threshold]
    if len(inliers) < max(5, min(8, int(np.ceil(label_count * 0.25)))):
        return None

    xs = np.array([index for index, _ in inliers], dtype=np.float64)
    coords = np.array([coord for _, coord in inliers], dtype=np.float64)
    if float(np.ptp(xs)) <= 0:
        return None
    slope, intercept = np.polyfit(xs, coords, 1)
    slope = float(slope)
    intercept = float(intercept)
    step = abs(slope)
    if step <= 1.0:
        return None
    inlier_residuals = np.abs(coords - (slope * xs + intercept))
    median_residual = float(np.median(inlier_residuals))
    max_residual = float(np.max(inlier_residuals))
    if median_residual > max(4.0, step * 0.24) or max_residual > max(8.0, step * 0.45):
        return None

    if image_shape is not None:
        h, w = image_shape
        extent = float(w if axis_key == "x_axis" else h)
        margin = max(10.0, step * 0.9)
        endpoints = [float(intercept), float(slope * (label_count - 1) + intercept)]
        if any(value < -margin or value > extent - 1.0 + margin for value in endpoints):
            return None

    return slope, intercept, {
        "source": "robust_category_axis_fit",
        "unique_anchor_count": len(points),
        "inlier_count": len(inliers),
        "step": round(step, 3),
        "median_residual": round(median_residual, 3),
        "max_residual": round(max_residual, 3),
    }

def add_mllm_missing_label_boxes(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = [dict(item) for item in items]
    events: list[dict[str, object]] = []
    h, w = image_shape
    for axis_key in ("x_axis", "y_axis"):
        ticks = mllm_axis_ticks(mllm_result, axis_key)
        if len(ticks) < 2:
            continue
        label_to_index = {normalize_label_text(text): index for index, text in enumerate(ticks)}
        numeric_to_index: dict[float, int] = {}
        for index, text in enumerate(ticks):
            parsed = parse_numeric_label(text) if is_regression_numeric_label(text) else None
            if parsed is not None:
                numeric_to_index[round(float(parsed), 8)] = index
        axis_items = [item for item in output if item.get("role") == axis_key]
        anchors: list[tuple[int, str, float]] = []
        matched_tick_items: list[dict[str, object]] = []
        matched_indexes: set[int] = set()
        orthogonal_centers: list[float] = []
        for item in axis_items:
            if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
                continue
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            matched_index = label_to_index.get(normalize_label_text(text))
            if matched_index is None and is_regression_numeric_label(text):
                for tick_index, tick_text in enumerate(ticks):
                    if numeric_label_shape_compatible(text, tick_text):
                        matched_index = tick_index
                        break
            if matched_index is None:
                continue
            center = item.get("center", [0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            coord = float(center[0] if axis_key == "x_axis" else center[1])
            orthogonal = float(center[1] if axis_key == "x_axis" else center[0])
            anchors.append((matched_index, ticks[matched_index], coord))
            matched_tick_items.append(item)
            matched_indexes.add(matched_index)
            orthogonal_centers.append(orthogonal)

        missing_indexes = [index for index in range(len(ticks)) if index not in matched_indexes]
        if not missing_indexes or len(anchors) < 2:
            continue
        regular_numeric_axis = is_regular_numeric_tick_sequence(ticks)
        numeric_tick_count = sum(1 for text in ticks if is_regression_numeric_label(text))
        numeric_axis = regular_numeric_axis or numeric_tick_count >= max(2, int(len(ticks) * 0.65))
        missing_count = len(missing_indexes)
        anchor_count = len(anchors)
        minimum_anchor_count = 2 if regular_numeric_axis else (3 if missing_count > 2 else 2)
        if anchor_count < minimum_anchor_count:
            for missing_index in missing_indexes:
                events.append({
                    "axis": axis_key,
                    "text": ticks[missing_index],
                    "status": "skipped",
                    "reason": "insufficient_anchor_count_for_bulk_pseudo",
                    "matched_anchor_count": anchor_count,
                })
            continue
        if not numeric_axis and missing_count > max(2, int(anchor_count * 0.25)):
            events.append({
                "axis": axis_key,
                "status": "deferred",
                "reason": "category_bulk_completion_deferred_to_canonical_axis_fit",
                "matched_anchor_count": anchor_count,
                "missing_count": missing_count,
            })
            continue
        bulk_regular_supported = False
        bulk_regular_details: dict[str, object] = {}
        if numeric_axis and regular_numeric_axis and missing_count > max(4, int(anchor_count * 0.75)):
            bulk_regular_supported, bulk_regular_details = regular_numeric_bulk_pseudo_supported(
                ticks,
                anchors,
                missing_indexes,
                axis_key,
                image_shape,
            )
        if numeric_axis and regular_numeric_axis and missing_count > max(4, int(anchor_count * 0.75)) and not bulk_regular_supported:
            for missing_index in missing_indexes:
                events.append({
                    "axis": axis_key,
                    "text": ticks[missing_index],
                    "status": "skipped",
                    "reason": "too_many_regular_numeric_ticks_missing_for_anchor_support",
                    "matched_anchor_count": anchor_count,
                    "missing_count": missing_count,
                    "bulk_regular_support": bulk_regular_details,
                })
            continue
        if numeric_axis and not regular_numeric_axis and missing_count > max(4, anchor_count):
            for missing_index in missing_indexes:
                events.append({
                    "axis": axis_key,
                    "text": ticks[missing_index],
                    "status": "skipped",
                    "reason": "too_many_numeric_ticks_missing_for_anchor_support",
                    "matched_anchor_count": anchor_count,
                    "missing_count": missing_count,
                })
            continue
        box_width, box_height = median_item_box_size(axis_items, axis_key)
        fallback_orthogonal = float(np.median(orthogonal_centers)) if orthogonal_centers else (h * 0.88 if axis_key == "x_axis" else w * 0.10)
        for missing_index in missing_indexes:
            if numeric_axis and regular_numeric_axis and anchor_count < 2 and missing_index in {0, len(ticks) - 1}:
                events.append({
                    "axis": axis_key,
                    "text": ticks[missing_index],
                    "status": "skipped",
                    "reason": "defer_regular_numeric_endpoint_to_canonical",
                    "matched_anchor_count": anchor_count,
                })
                continue
            if not is_informative_tick_label(ticks[missing_index]):
                events.append({"axis": axis_key, "text": ticks[missing_index], "status": "skipped", "reason": "low_information_tick_label"})
                continue
            predicted, method, confidence = fit_label_position(ticks, anchors, missing_index)
            if predicted is None or confidence < 0.56:
                events.append({"axis": axis_key, "text": ticks[missing_index], "status": "skipped", "reason": method})
                continue
            if axis_key == "x_axis":
                clipped = float(np.clip(predicted, 0, w - 1))
                if abs(clipped - predicted) > max(8.0, box_width * 0.6):
                    events.append({"axis": axis_key, "text": ticks[missing_index], "status": "skipped", "reason": "predicted_position_outside_image", "confidence": round(confidence, 3)})
                    continue
                cx = clipped
                cy = float(np.clip(fallback_orthogonal, 0, h - 1))
                duplicate_distance = max(6.0, box_width * 0.35)
            else:
                cx = float(np.clip(fallback_orthogonal, 0, w - 1))
                clipped = float(np.clip(predicted, 0, h - 1))
                if abs(clipped - predicted) > max(8.0, box_height * 0.8):
                    events.append({"axis": axis_key, "text": ticks[missing_index], "status": "skipped", "reason": "predicted_position_outside_image", "confidence": round(confidence, 3)})
                    continue
                cy = clipped
                duplicate_distance = max(6.0, box_height * 0.45)
            duplicate = False
            duplicate_reason = "near_existing_axis_label"
            for item in [*matched_tick_items, *axis_items]:
                center = item.get("center", [0.0, 0.0])
                if not isinstance(center, (list, tuple)) or len(center) < 2:
                    continue
                if item not in matched_tick_items:
                    item_text = str(item.get("text", "") or "")
                    if not same_label_value(item_text, ticks[missing_index]):
                        continue
                distance = abs(float(center[0] if axis_key == "x_axis" else center[1]) - (cx if axis_key == "x_axis" else cy))
                if distance <= duplicate_distance:
                    duplicate = True
                    if item not in matched_tick_items:
                        duplicate_reason = "near_existing_axis_ocr_box"
                    break
            if duplicate:
                events.append({"axis": axis_key, "text": ticks[missing_index], "status": "skipped", "reason": duplicate_reason})
                continue
            pseudo = {
                "text": ticks[missing_index],
                "score": 0.0,
                "box": rect_box(cx - box_width / 2.0, cy - box_height / 2.0, cx + box_width / 2.0, cy + box_height / 2.0),
                "role": axis_key,
                "center": [cx, cy],
                "size": [box_width, box_height],
                "raw_role": "missing",
                "role_source": "mllm_pseudo_interpolated",
                "role_confidence": round(confidence, 3),
                "mllm_pseudo_box": True,
                "pseudo_source": method,
                "pseudo_index": missing_index,
                "pseudo_count": len(ticks),
            }
            output.append(pseudo)
            events.append(
                {
                    "axis": axis_key,
                    "text": ticks[missing_index],
                    "status": "added",
                    "source": method,
                    "confidence": round(confidence, 3),
                    "center": [round(cx, 2), round(cy, 2)],
                    "matched_anchor_count": len(anchors),
                    "bulk_regular_support": bulk_regular_details if bulk_regular_supported else None,
                }
            )
    return output, events

def axis_tick_geometry_candidates(items: list[dict[str, object]], axis_key: str) -> list[int]:
    candidates: list[tuple[float, int]] = []
    for index, item in enumerate(items):
        if item.get("role") != axis_key:
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        size = item.get("size", [0.0, 0.0])
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [0.0, 0.0]
        width = float(size[0])
        height = float(size[1])
        numeric = parse_numeric_label(text) is not None or item.get("split_from_merged") or item.get("mllm_pseudo_box")
        if not numeric and axis_key == "y_axis" and height > max(30.0, width * 2.0):
            continue
        if not numeric and axis_key == "x_axis" and width > max(80.0, height * 5.0):
            continue
        coord = axis_coord(item, axis_key)
        if coord is not None:
            candidates.append((coord, index))
    return [index for _, index in sorted(candidates)]

def axis_perpendicular_reference(
    items: list[dict[str, object]],
    axis_key: str,
) -> tuple[float | None, float]:
    values: list[float] = []
    sizes: list[float] = []
    for item in items:
        if item.get("role") != axis_key:
            continue
        text = str(item.get("text", "") or "").strip()
        center = item.get("center", [0.0, 0.0])
        size = item.get("size", [0.0, 0.0])
        if not text or not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [0.0, 0.0]
        numeric = parse_numeric_label(text) is not None or item.get("split_from_merged") or item.get("mllm_pseudo_box")
        if not numeric:
            continue
        values.append(float(center[1] if axis_key == "x_axis" else center[0]))
        sizes.append(float(size[1] if axis_key == "x_axis" else size[0]))
    if not values:
        return None, 24.0
    typical_size = float(np.median(sizes)) if sizes else 10.0
    return float(np.median(values)), max(16.0, typical_size * 2.8)

def axis_perpendicular_score(
    item: dict[str, object],
    axis_key: str,
    reference: float | None,
    tolerance: float,
) -> float:
    center = item.get("center", [0.0, 0.0])
    if reference is None or not isinstance(center, (list, tuple)) or len(center) < 2:
        return 0.0
    value = float(center[1] if axis_key == "x_axis" else center[0])
    distance = abs(value - reference)
    return max(0.0, 1.0 - distance / max(1.0, tolerance))

def expanded_axis_tick_candidates(
    items: list[dict[str, object]],
    axis_key: str,
    labels: list[str],
    used_indexes: set[int],
) -> list[int]:
    base = axis_tick_geometry_candidates(items, axis_key)
    candidates = set(index for index in base if index not in used_indexes)
    reference, tolerance = axis_perpendicular_reference(items, axis_key)
    for index, item in enumerate(items):
        if index in candidates or index in used_indexes:
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        if (
            item.get("role") == axis_key
            and item.get("label_kind") != "axis_label"
            and any(same_label_value(text, label) for label in labels)
        ):
            candidates.add(index)
            continue
        # Ambiguous origin labels are often OCR-classified to the wrong axis.
        # Only borrow cross-axis geometry when it lies on this axis' baseline/column.
        if item.get("role") not in {"x_axis", "y_axis", "other"}:
            continue
        perp_score = axis_perpendicular_score(item, axis_key, reference, tolerance)
        if perp_score < 0.35:
            continue
        if not any(
            same_label_value(text, label)
            or (axis_key == "y_axis" and numeric_prefix_label_value(text, label))
            for label in labels
        ):
            continue
        coord = axis_coord(item, axis_key)
        if coord is None:
            continue
        candidates.add(index)
    return sorted(candidates, key=lambda index: axis_coord(items[index], axis_key) or 0.0)

def canonical_candidate_score(
    item: dict[str, object],
    axis_key: str,
    reference: float | None,
    tolerance: float,
) -> float:
    score = axis_perpendicular_score(item, axis_key, reference, tolerance) * 1.8
    if item.get("mllm_pseudo_box"):
        score -= 0.65
    elif item.get("role") == axis_key:
        score += 0.25
    else:
        score += 0.10
    try:
        score += min(0.12, max(0.0, float(item.get("score", 0.0) or 0.0)) * 0.08)
    except (TypeError, ValueError):
        pass
    return score

def model_axis_label_text(mllm_result: dict[str, object], axis_key: str) -> str:
    axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
    if not isinstance(axis, dict):
        return ""
    raw = axis.get("axis_label", {})
    text = str(raw.get("text", "") if isinstance(raw, dict) else raw or "").strip()
    normalized = normalize_label_text(text)
    if normalized in {"none", "null", "unknown", "n/a"}:
        return ""
    if normalized in {normalize_label_text(label) for label in mllm_axis_ticks(mllm_result, axis_key)}:
        return ""
    return text

def choose_axis_title_candidate(
    items: list[dict[str, object]],
    axis_key: str,
    used_indexes: set[int],
) -> int | None:
    best_index: int | None = None
    best_score = -1.0
    for index, item in enumerate(items):
        axis_label_role = str(item.get("axis_label_role", "") or item.get("canonical_axis", "") or "")
        is_axis_label_candidate = item.get("label_kind") == "axis_label" and axis_label_role == axis_key
        if index in used_indexes or (item.get("role") != axis_key and not is_axis_label_candidate):
            continue
        text = str(item.get("text", "") or "").strip()
        if not text or parse_numeric_label(text) is not None:
            continue
        size = item.get("size", [0.0, 0.0])
        center = item.get("center", [0.0, 0.0])
        if not isinstance(size, (list, tuple)) or len(size) < 2 or not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        width = float(size[0])
        height = float(size[1])
        score = height + max(0.0, 80.0 - float(center[0])) * 0.2 if axis_key == "y_axis" else width
        if is_axis_label_candidate:
            score += 1000.0
        if score > best_score:
            best_score = score
            best_index = index
    return best_index

def tick_label_index(text: str, labels: list[str]) -> int | None:
    normalized = normalize_label_text(text)
    for index, label in enumerate(labels):
        if normalized and normalized == normalize_label_text(label):
            return index
    for index, label in enumerate(labels):
        if numeric_label_shape_compatible(text, label):
            return index
    return None

def cluster_by_axis_band(
    candidates: list[dict[str, object]],
    axis_key: str,
    image_shape: tuple[int, int],
) -> list[list[dict[str, object]]]:
    if not candidates:
        return []
    h, w = image_shape
    tolerance = max(10.0, (h if axis_key == "x_axis" else w) * 0.025)
    clusters: list[list[dict[str, object]]] = []
    for candidate in sorted(candidates, key=lambda value: float(value["perp"])):
        perp = float(candidate["perp"])
        for cluster in clusters:
            cluster_perp = float(np.median([float(item["perp"]) for item in cluster]))
            if abs(perp - cluster_perp) <= tolerance:
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    return clusters

def cluster_sequence_score(cluster: list[dict[str, object]], axis_key: str) -> float:
    if len(cluster) < 2:
        return 0.0
    unique_indexes = sorted({int(item["label_index"]) for item in cluster})
    count_score = min(1.0, len(unique_indexes) / 5.0)

    ordered = sorted(cluster, key=lambda item: float(item["coord"]))
    label_indexes = [int(item["label_index"]) for item in ordered]
    if len(label_indexes) >= 2:
        inc = sum(1 for a, b in zip(label_indexes, label_indexes[1:]) if b >= a)
        dec = sum(1 for a, b in zip(label_indexes, label_indexes[1:]) if b <= a)
        monotonic_score = max(inc, dec) / max(1, len(label_indexes) - 1)
    else:
        monotonic_score = 0.0

    coords = sorted(float(item["coord"]) for item in cluster)
    if len(coords) >= 3:
        diffs = np.diff(np.array(coords, dtype=np.float64))
        median_diff = float(np.median(np.abs(diffs)))
        if median_diff <= 1e-6:
            regularity_score = 0.0
        else:
            mad = float(np.median(np.abs(np.abs(diffs) - median_diff)))
            regularity_score = max(0.0, 1.0 - mad / max(1.0, median_diff))
    else:
        regularity_score = 0.55

    return 0.45 * count_score + 0.35 * monotonic_score + 0.20 * regularity_score

def axis_tick_order_direction(mllm_result: dict[str, object], axis_key: str) -> str:
    axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
    if not isinstance(axis, dict):
        return "unknown"
    order = str(axis.get("tick_order", "") or "").strip().lower()
    if order in {"left_to_right", "top_to_bottom"}:
        return "increasing"
    if order in {"right_to_left", "bottom_to_top"}:
        return "decreasing"
    return "unknown"

def axis_tick_perp_tolerance(
    candidates: list[dict[str, object]],
    axis_key: str,
    image_shape: tuple[int, int],
) -> float:
    h, w = image_shape
    base = float(h if axis_key == "x_axis" else w)
    sizes = [float(item.get("perp_size", 0.0) or 0.0) for item in candidates if float(item.get("perp_size", 0.0) or 0.0) > 0]
    typical_size = float(np.median(sizes)) if sizes else 8.0
    raw = max(10.0, base * 0.014, typical_size * 1.6)
    return min(raw, max(18.0, base * 0.08))

def build_axis_tick_hypothesis_candidates(
    items: list[dict[str, object]],
    labels: list[str],
    axis_key: str,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for index, item in enumerate(items):
        if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        label_index = tick_label_index(text, labels)
        if label_index is None:
            continue
        center = item.get("center", [0.0, 0.0])
        size = item.get("size", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [0.0, 0.0]
        cx = float(center[0])
        cy = float(center[1])
        width = float(size[0])
        height = float(size[1])
        role = str(item.get("role", "other") or "other")
        if axis_key == "x_axis":
            variants = [
                ("center", cy),
                ("top_edge", cy - height * 0.5),
                ("bottom_edge", cy + height * 0.5),
            ]
            coord = cx
            coord_size = width
            perp_size = height
        else:
            variants = [
                ("center", cx),
                ("left_edge", cx - width * 0.5),
                ("right_edge", cx + width * 0.5),
            ]
            coord = cy
            coord_size = height
            perp_size = width
        for perp_mode, perp in variants:
            candidates.append(
                {
                    "item_index": index,
                    "label_index": label_index,
                    "coord": coord,
                    "perp": float(perp),
                    "coord_size": coord_size,
                    "perp_size": perp_size,
                    "perp_mode": perp_mode,
                    "role": role,
                }
            )
    return candidates

def axis_tick_hypothesis_lines(
    candidates: list[dict[str, object]],
    axis_key: str,
    image_shape: tuple[int, int],
    tolerance: float,
) -> list[tuple[float, float, str]]:
    if not candidates:
        return []
    h, w = image_shape
    coord_span = float(w if axis_key == "x_axis" else h)
    lines: list[tuple[float, float, str]] = []
    perps = sorted(float(item["perp"]) for item in candidates)
    lines.append((0.0, float(np.median(perps)), "median_horizontal_band"))
    for item in candidates:
        lines.append((0.0, float(item["perp"]), "single_point_band"))

    min_dx = max(18.0, coord_span * 0.05)
    max_abs_slope = 0.42
    pair_count = 0
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            dx = float(right["coord"]) - float(left["coord"])
            if abs(dx) < min_dx:
                continue
            slope = (float(right["perp"]) - float(left["perp"])) / dx
            if abs(slope) > max_abs_slope:
                continue
            intercept = float(left["perp"]) - slope * float(left["coord"])
            lines.append((slope, intercept, "ransac_pair_band"))
            pair_count += 1
            if pair_count >= 2000:
                break
        if pair_count >= 2000:
            break

    # Deduplicate near-identical bands so dense labels do not make scoring quadratic.
    unique: list[tuple[float, float, str]] = []
    for slope, intercept, source in lines:
        duplicate = False
        for old_slope, old_intercept, _ in unique:
            if abs(slope - old_slope) <= 0.015 and abs(intercept - old_intercept) <= max(4.0, tolerance * 0.35):
                duplicate = True
                break
        if not duplicate:
            unique.append((slope, intercept, source))
    return unique

def axis_tick_hypothesis_sequence_score(
    inliers: list[dict[str, object]],
    labels: list[str],
    axis_key: str,
    order_direction: str,
    tolerance: float,
    slope: float,
) -> float:
    if len(inliers) < 2:
        return 0.0
    unique_count = len({int(item["label_index"]) for item in inliers})
    if unique_count < 2:
        return 0.0

    target_count = max(2, min(6, len(labels)))
    count_score = min(1.0, unique_count / target_count)
    coverage_score = min(1.0, unique_count / max(1, len(labels)))

    ordered = sorted(inliers, key=lambda item: float(item["coord"]))
    label_indexes = [int(item["label_index"]) for item in ordered]
    if len(label_indexes) >= 2:
        inc = sum(1 for a, b in zip(label_indexes, label_indexes[1:]) if b >= a)
        dec = sum(1 for a, b in zip(label_indexes, label_indexes[1:]) if b <= a)
        if order_direction == "increasing":
            monotonic_score = inc / max(1, len(label_indexes) - 1)
        elif order_direction == "decreasing":
            monotonic_score = dec / max(1, len(label_indexes) - 1)
        else:
            monotonic_score = max(inc, dec) / max(1, len(label_indexes) - 1)
    else:
        monotonic_score = 0.0

    residuals = [float(item.get("residual", 0.0) or 0.0) for item in inliers]
    median_residual = float(np.median(residuals)) if residuals else tolerance
    collinear_score = max(0.0, 1.0 - median_residual / max(1.0, tolerance))

    if unique_count >= 3:
        dedup: dict[int, float] = {}
        dedup_residual: dict[int, float] = {}
        for item in inliers:
            label_index = int(item["label_index"])
            coord = float(item["coord"])
            residual = float(item.get("residual", 0.0) or 0.0)
            if label_index not in dedup or residual < dedup_residual[label_index]:
                dedup[label_index] = coord
                dedup_residual[label_index] = residual
        xs = np.array(sorted(dedup), dtype=np.float64)
        ys = np.array([dedup[int(index)] for index in xs], dtype=np.float64)
        if len(xs) >= 3 and float(np.ptp(xs)) > 0:
            coeff = np.polyfit(xs, ys, 1)
            predicted = coeff[0] * xs + coeff[1]
            step = abs(float(coeff[0]))
            median_fit_residual = float(np.median(np.abs(ys - predicted)))
            regularity_score = max(0.0, 1.0 - median_fit_residual / max(tolerance, step * 0.35, 1.0))
        else:
            regularity_score = 0.55
    else:
        regularity_score = 0.62

    current_role_support = sum(1 for item in inliers if item.get("role") == axis_key) / max(1, len(inliers))
    slope_penalty = min(0.16, abs(slope) * 0.35)
    return (
        0.24 * count_score
        + 0.23 * monotonic_score
        + 0.23 * regularity_score
        + 0.20 * collinear_score
        + 0.06 * coverage_score
        + 0.04 * current_role_support
        - slope_penalty
    )

def evaluate_axis_tick_hypothesis(
    candidates: list[dict[str, object]],
    labels: list[str],
    axis_key: str,
    image_shape: tuple[int, int],
    mllm_result: dict[str, object],
) -> dict[str, object] | None:
    if len(candidates) < 2:
        return None
    tolerance = axis_tick_perp_tolerance(candidates, axis_key, image_shape)
    order_direction = axis_tick_order_direction(mllm_result, axis_key)
    best: dict[str, object] | None = None

    for slope, intercept, source in axis_tick_hypothesis_lines(candidates, axis_key, image_shape, tolerance):
        grouped: dict[int, dict[str, object]] = {}
        for candidate in candidates:
            expected = slope * float(candidate["coord"]) + intercept
            residual = abs(float(candidate["perp"]) - expected)
            if residual > tolerance:
                continue
            scored = dict(candidate)
            scored["residual"] = residual
            label_index = int(scored["label_index"])
            previous = grouped.get(label_index)
            if previous is None:
                grouped[label_index] = scored
                continue
            previous_score = float(previous.get("residual", tolerance))
            current_score = residual
            if scored.get("role") == axis_key:
                current_score -= tolerance * 0.08
            if previous.get("role") == axis_key:
                previous_score -= tolerance * 0.08
            if current_score < previous_score:
                grouped[label_index] = scored
        inliers = list(grouped.values())
        score = axis_tick_hypothesis_sequence_score(
            inliers,
            labels,
            axis_key,
            order_direction,
            tolerance,
            slope,
        )
        if best is None or score > float(best["score"]):
            best = {
                "score": score,
                "inliers": inliers,
                "slope": slope,
                "intercept": intercept,
                "source": source,
                "tolerance": tolerance,
                "order_direction": order_direction,
            }
    return best

def axis_tick_cluster_claims(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    axis_key: str,
    image_shape: tuple[int, int],
) -> dict[int, dict[str, object]]:
    labels = mllm_axis_ticks(mllm_result, axis_key)
    if len(labels) < 2:
        return {}

    candidates = build_axis_tick_hypothesis_candidates(items, labels, axis_key)
    hypothesis = evaluate_axis_tick_hypothesis(candidates, labels, axis_key, image_shape, mllm_result)
    if not hypothesis:
        return {}
    best_cluster = list(hypothesis.get("inliers", []))
    unique_count = len({int(item["label_index"]) for item in best_cluster})
    best_score = float(hypothesis.get("score", 0.0) or 0.0)
    if unique_count < 2 or best_score < 0.56:
        return {}

    slope = float(hypothesis.get("slope", 0.0) or 0.0)
    intercept = float(hypothesis.get("intercept", 0.0) or 0.0)
    tolerance = float(hypothesis.get("tolerance", 16.0) or 16.0)
    claims: dict[int, dict[str, object]] = {}
    for item in best_cluster:
        residual = float(item.get("residual", 0.0) or 0.0)
        closeness = max(0.0, 1.0 - residual / max(1.0, tolerance))
        claims[int(item["item_index"])] = {
            "role": axis_key,
            "score": min(1.0, best_score + closeness * 0.18),
            "hypothesis_score": round(best_score, 3),
            "hypothesis_slope": round(slope, 4),
            "hypothesis_intercept": round(intercept, 2),
            "hypothesis_tolerance": round(tolerance, 2),
            "hypothesis_source": hypothesis.get("source", ""),
            "hypothesis_perp_mode": item.get("perp_mode", ""),
            "hypothesis_unique_tick_count": unique_count,
            "reason": "axis_tick_hypothesis_fit",
        }
    return claims

def apply_axis_tick_cluster_role_correction(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int],
) -> list[dict[str, object]]:
    if not items:
        return []
    all_claims: dict[int, list[dict[str, object]]] = {}
    for axis_key in ("x_axis", "y_axis"):
        for index, claim in axis_tick_cluster_claims(items, mllm_result, axis_key, image_shape).items():
            all_claims.setdefault(index, []).append(claim)
    if not all_claims:
        return items

    output = [dict(item) for item in items]
    for index, claims in all_claims.items():
        claims = sorted(claims, key=lambda claim: float(claim["score"]), reverse=True)
        best = claims[0]
        second_score = float(claims[1]["score"]) if len(claims) > 1 else 0.0
        best_score = float(best["score"])
        best_role = str(best["role"])
        current_role = str(output[index].get("role", "other"))
        if best_score < 0.62:
            continue
        if len(claims) > 1 and best_role != current_role and best_score - second_score < 0.08:
            output[index]["role_cluster_candidates"] = claims[:2]
            continue
        if current_role != best_role:
            output[index]["raw_role_before_cluster"] = current_role
            output[index]["role"] = best_role
            output[index]["role_source"] = "mllm_axis_cluster_refined"
            output[index]["role_reason"] = str(best["reason"])
            output[index]["role_confidence"] = round(best_score, 3)
        elif str(output[index].get("role_source", "")) == "ocr_geometry":
            output[index]["role_source"] = "ocr_geometry+mllm_cluster_confirmed"
            output[index]["role_reason"] = str(best["reason"])
            output[index]["role_confidence"] = max(float(output[index].get("role_confidence", 0.0) or 0.0), round(best_score, 3))
        output[index]["role_cluster_candidates"] = claims[:2]
    return output

def is_mllm_category_axis(mllm_result: dict[str, object], axis_key: str, labels: list[str]) -> bool:
    axis = mllm_result.get(axis_key, {}) if isinstance(mllm_result, dict) else {}
    axis_type = str(axis.get("type", "") if isinstance(axis, dict) else "").strip().lower()
    if axis_type in {"category", "time", "date"}:
        return True
    if not labels:
        return False
    numeric_count = sum(1 for label in labels if parse_numeric_label(label) is not None)
    return len(labels) >= 6 and numeric_count / max(1, len(labels)) <= 0.35

def category_axis_order_candidates(
    items: list[dict[str, object]],
    axis_key: str,
    used_indexes: set[int],
    labels: list[str] | None = None,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    opposite = "y_axis" if axis_key == "x_axis" else "x_axis"
    label_texts = {normalize_label_text(label) for label in labels or [] if str(label).strip()}
    category_labels_are_numeric = bool(labels) and all(parse_numeric_label(str(label)) is not None for label in labels)
    for index, item in enumerate(items):
        if index in used_indexes:
            continue
        if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
            continue
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        exact_axis_label = normalize_label_text(text) in label_texts
        if not exact_axis_label and not category_labels_are_numeric and parse_numeric_label(text) is not None:
            continue
        if item.get("mllm_pseudo_box") and not exact_axis_label:
            continue
        role = str(item.get("role", "other") or "other")
        raw_role = str(item.get("raw_role", "") or item.get("raw_role_before_cluster", "") or "")
        if role != axis_key and raw_role != axis_key and not exact_axis_label:
            continue
        if role == opposite and raw_role != axis_key:
            continue
        center = item.get("center", [0.0, 0.0])
        size = item.get("size", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [0.0, 0.0]
        cx = float(center[0])
        cy = float(center[1])
        width = float(size[0])
        height = float(size[1])
        if axis_key == "x_axis":
            coord_variants = [("center", cx)]
            coord_size = width
            perp_size = height
            perp_variants = [("center", cy)]
        else:
            coord_variants = [("center", cy)]
            coord_size = height
            perp_size = width
            perp_variants = [("center", cx)]
        for coord_mode, coord in coord_variants:
            for perp_mode, perp in perp_variants:
                candidates.append(
                    {
                        "item_index": index,
                        "coord": float(coord),
                        "perp": float(perp),
                        "coord_size": coord_size,
                        "perp_size": perp_size,
                        "coord_mode": coord_mode,
                        "perp_mode": perp_mode,
                        "role": role,
                        "raw_role": raw_role,
                        "text": text,
                        "score": float(item.get("score", 0.0) or 0.0),
                    }
                )
    return candidates

def select_category_axis_band(
    candidates: list[dict[str, object]],
    axis_key: str,
    image_shape: tuple[int, int],
) -> list[dict[str, object]]:
    if not candidates:
        return []
    h, w = image_shape
    tolerance = max(18.0, (h if axis_key == "x_axis" else w) * 0.055)
    clusters: list[list[dict[str, object]]] = []
    for candidate in sorted(candidates, key=lambda value: float(value["perp"])):
        perp = float(candidate["perp"])
        for cluster in clusters:
            cluster_perp = float(np.median([float(item["perp"]) for item in cluster]))
            if abs(perp - cluster_perp) <= tolerance:
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    if not clusters:
        return []
    return max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            float(np.ptp([float(item["coord"]) for item in cluster])) if len(cluster) > 1 else 0.0,
        ),
    )

def category_axis_order_assignments(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    axis_key: str,
    labels: list[str],
    used_indexes: set[int],
    image_shape: tuple[int, int] | None = None,
) -> list[tuple[int, int, str]]:
    if not is_mllm_category_axis(mllm_result, axis_key, labels):
        return []
    if len(labels) < 4:
        return []
    if image_shape is None:
        coords = [
            item.get("center", [0.0, 0.0])
            for item in items
            if isinstance(item.get("center", None), (list, tuple)) and len(item.get("center", [])) >= 2
        ]
        if coords:
            max_x = max(float(value[0]) for value in coords)
            max_y = max(float(value[1]) for value in coords)
            image_shape = (int(max_y + 1), int(max_x + 1))
        else:
            image_shape = (1, 1)
    candidates = category_axis_order_candidates(items, axis_key, used_indexes, labels)
    exact_assignments: list[tuple[int, int, str]] = []
    exact_used_items: set[int] = set()
    normalized_labels = [normalize_label_text(label) for label in labels]
    for label_index, normalized_label in enumerate(normalized_labels):
        if not normalized_label:
            continue
        matches = [
            candidate
            for candidate in candidates
            if int(candidate["item_index"]) not in exact_used_items
            and str(candidate.get("coord_mode", "")) == "center"
            and str(candidate.get("perp_mode", "")) == "center"
            and normalize_label_text(str(candidate.get("text", ""))) == normalized_label
        ]
        if not matches:
            continue
        best = min(
            matches,
            key=lambda candidate: (
                1 if candidate.get("role") == axis_key else 2,
                1 if items[int(candidate["item_index"])].get("mllm_pseudo_box") else 0,
                -float(candidate.get("score", 0.0) or 0.0),
            ),
        )
        item_index = int(best["item_index"])
        exact_used_items.add(item_index)
        exact_assignments.append((label_index, item_index, "exact_text_order_match"))
    if len(exact_assignments) >= max(3, int(len(labels) * 0.55)):
        ordered_exact = sorted(exact_assignments, key=lambda value: value[0])
        coords = [
            axis_coord(items[item_index], axis_key)
            for _, item_index, _ in ordered_exact
        ]
        usable_coords = [float(value) for value in coords if value is not None]
        if len(usable_coords) == len(coords):
            diffs = np.diff(np.array(usable_coords, dtype=np.float64))
            if len(diffs) == 0 or np.all(diffs >= -1.0) or np.all(diffs <= 1.0):
                return ordered_exact

    band = select_category_axis_band(candidates, axis_key, image_shape)
    if len(band) < max(6, int(len(labels) * 0.45)):
        return []
    coords = sorted(float(item["coord"]) for item in band)
    span = coords[-1] - coords[0] if len(coords) >= 2 else 0.0
    if span < max(40.0, (image_shape[1] if axis_key == "x_axis" else image_shape[0]) * 0.25):
        return []
    expected_step = span / max(1, len(labels) - 1)
    if expected_step <= 1.0:
        return []
    max_distance = max(expected_step * 0.82, 8.0)
    unused = set(range(len(band)))
    used_item_indexes: set[int] = set()
    assignments: list[tuple[int, int, str]] = []
    for label_index in range(len(labels)):
        expected = coords[0] + expected_step * label_index
        best_local: int | None = None
        best_score = float("inf")
        for local_index in list(unused):
            candidate = band[local_index]
            if int(candidate["item_index"]) in used_item_indexes:
                continue
            distance = abs(float(candidate["coord"]) - expected)
            coord_mode = str(candidate.get("coord_mode", "center") or "center")
            role_bonus = -expected_step * 0.08 if candidate.get("role") == axis_key else 0.0
            score_bonus = -min(expected_step * 0.06, max(0.0, float(candidate.get("score", 0.0))) * expected_step * 0.04)
            mode_penalty = 0.0 if coord_mode == "center" else expected_step * 0.04
            score = distance + role_bonus + score_bonus + mode_penalty
            if score < best_score:
                best_score = score
                best_local = local_index
        if best_local is None:
            continue
        if abs(float(band[best_local]["coord"]) - expected) > max_distance:
            continue
        unused.remove(best_local)
        used_item_indexes.add(int(band[best_local]["item_index"]))
        assignments.append((label_index, int(band[best_local]["item_index"]), "category_order_geometry_alignment"))
    if len(assignments) < max(5, int(len(labels) * 0.55)):
        return []
    return assignments

def add_category_axis_missing_pseudo_items(
    output: list[dict[str, object]],
    axis_key: str,
    labels: list[str],
    assignments: list[tuple[int, int, str]],
    image_shape: tuple[int, int] | None = None,
) -> list[dict[str, object]]:
    if len(labels) < 4 or len(assignments) < max(5, min(9, int(np.ceil(len(labels) * 0.30)))):
        return []
    assigned_label_indexes = {label_index for label_index, _, _ in assignments}
    missing = [index for index in range(len(labels)) if index not in assigned_label_indexes]

    anchors: list[tuple[int, float, float, float, float]] = []
    full_label_widths: list[float] = []
    full_label_heights: list[float] = []
    composite_axis_labels = any(
        re.search(r"[A-Za-z]", str(label or "")) and re.search(r"\d", str(label or ""))
        for label in labels
    )
    for label_index, item_index, _ in assignments:
        item = output[item_index]
        center = item.get("center", [0.0, 0.0])
        size = item.get("size", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [0.0, 0.0]
        cx = float(center[0])
        cy = float(center[1])
        width = max(8.0, float(size[0]))
        height = max(8.0, float(size[1]))
        coord = cx if axis_key == "x_axis" else cy
        perp = cy if axis_key == "x_axis" else cx
        anchors.append((label_index, coord, perp, width, height))
        if (
            composite_axis_labels
            and not item.get("merged_from_split_ocr")
            and not item.get("mllm_pseudo_box")
            and normalize_label_text(str(item.get("text", "") or "")) == normalize_label_text(labels[label_index])
        ):
            full_label_widths.append(width)
            full_label_heights.append(height)
    if len(anchors) < 3:
        return []

    fit = robust_category_axis_fit(anchors, len(labels), axis_key, image_shape)
    if fit is None:
        return []
    slope, intercept, fit_details = fit
    step = abs(float(slope))

    perps = [anchor[2] for anchor in anchors]
    widths = [anchor[3] for anchor in anchors]
    heights = [anchor[4] for anchor in anchors]
    perp = float(np.median(perps))
    coord_sizes = widths if axis_key == "x_axis" else heights
    perp_sizes = heights if axis_key == "x_axis" else widths
    compact_coord_sizes = [value for value in coord_sizes if value <= max(12.0, step * 0.70)]
    compact_perp_sizes = [value for value in perp_sizes if value <= max(18.0, float(np.median(perp_sizes)) * 1.35)]
    coord_size = float(np.median(compact_coord_sizes or coord_sizes))
    perp_size = float(np.median(compact_perp_sizes or perp_sizes))
    if composite_axis_labels and full_label_widths and full_label_heights:
        if axis_key == "x_axis":
            coord_size = float(np.median(np.array(full_label_widths, dtype=np.float64)))
            perp_size = float(np.median(np.array(full_label_heights, dtype=np.float64)))
        else:
            coord_size = float(np.median(np.array(full_label_heights, dtype=np.float64)))
            perp_size = float(np.median(np.array(full_label_widths, dtype=np.float64)))
    else:
        coord_size = float(np.clip(coord_size, 5.0, max(8.0, step * 0.62)))
    width = coord_size if axis_key == "x_axis" else perp_size
    height = perp_size if axis_key == "x_axis" else coord_size

    # For dense category axes, PaddleOCR often returns wide boxes spanning
    # multiple adjacent labels. Once the MLLM sequence and enough OCR anchors
    # agree on a stable axis fit, keep the semantic label but move noisy boxes
    # back onto that fitted category lattice.
    for label_index, item_index, _ in assignments:
        item = output[item_index]
        center = item.get("center", [0.0, 0.0])
        size = item.get("size", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [width, height]
        cx = float(center[0])
        cy = float(center[1])
        current_coord = cx if axis_key == "x_axis" else cy
        current_perp = cy if axis_key == "x_axis" else cx
        current_coord_size = float(size[0] if axis_key == "x_axis" else size[1])
        predicted_coord = float(slope * label_index + intercept)
        residual = abs(current_coord - predicted_coord)
        oversized = current_coord_size > max(step * 0.76, coord_size * 1.85)
        off_lattice = residual > max(3.5, step * 0.22)
        off_band = abs(current_perp - perp) > max(5.0, perp_size * 0.75)
        if not (oversized or off_lattice or off_band):
            continue
        if axis_key == "x_axis":
            new_cx = predicted_coord
            new_cy = perp
            if image_shape is not None:
                _, image_w = image_shape
                new_cx = float(np.clip(new_cx, 0, image_w - 1))
            new_box = rect_box(new_cx - width * 0.5, new_cy - height * 0.5, new_cx + width * 0.5, new_cy + height * 0.5)
            output[item_index]["center"] = [new_cx, new_cy]
        else:
            new_cx = perp
            new_cy = predicted_coord
            if image_shape is not None:
                image_h, _ = image_shape
                new_cy = float(np.clip(new_cy, 0, image_h - 1))
            new_box = rect_box(new_cx - width * 0.5, new_cy - height * 0.5, new_cx + width * 0.5, new_cy + height * 0.5)
            output[item_index]["center"] = [new_cx, new_cy]
        output[item_index]["box"] = new_box
        output[item_index]["size"] = [width, height]
        output[item_index]["bbox_regularized"] = True
        output[item_index]["bbox_regularize_source"] = "category_axis_lattice_fit"
        output[item_index]["bbox_regularize_reason"] = "wide_or_off_lattice_category_ocr_box"
        output[item_index]["bbox_regularize_previous_center"] = [cx, cy]
        output[item_index]["bbox_regularize_fit"] = fit_details

    if not missing:
        return []
    events: list[dict[str, object]] = []
    for label_index in missing:
        coord = float(slope * label_index + intercept)
        if axis_key == "x_axis":
            cx, cy = coord, perp
            if image_shape is not None:
                _, image_w = image_shape
                clipped = float(np.clip(cx, 0, image_w - 1))
                if abs(clipped - cx) <= max(8.0, step * 0.75):
                    cx = clipped
                else:
                    continue
        else:
            cx, cy = perp, coord
            if image_shape is not None:
                image_h, _ = image_shape
                clipped = float(np.clip(cy, 0, image_h - 1))
                if abs(clipped - cy) <= max(8.0, step * 0.75):
                    cy = clipped
                else:
                    continue
        pseudo = {
            "text": labels[label_index],
            "canonical_text": labels[label_index],
            "box": rect_box(cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5),
            "center": [cx, cy],
            "size": [width, height],
            "score": 0.0,
            "role": axis_key,
            "raw_role": "missing",
            "role_source": "mllm_category_order_pseudo",
            "role_confidence": 0.56,
            "role_reason": "category_axis_order_interpolation",
            "text_source": "mllm_axis_tick",
            "label_kind": "tick_label",
            "canonical_axis": axis_key,
            "canonical_index": label_index,
            "canonical_match_source": "category_order_interpolated_missing",
            "mllm_pseudo_box": True,
            "pseudo_source": "category_order_interpolation",
            "pseudo_index": label_index,
            "pseudo_count": len(labels),
            "pseudo_fit": fit_details,
        }
        output.append(pseudo)
        events.append(
            {
                "axis": axis_key,
                "kind": "tick_label",
                "index": label_index,
                "old_text": "",
                "new_text": labels[label_index],
                "source": "category_order_interpolated_missing",
                "fit": fit_details,
            }
        )
    return events

def add_regular_numeric_endpoint_pseudo_items(
    output: list[dict[str, object]],
    axis_key: str,
    labels: list[str],
    image_shape: tuple[int, int] | None = None,
) -> list[dict[str, object]]:
    if len(labels) < 3 or not is_regular_numeric_tick_sequence(labels):
        return []
    assigned: dict[int, dict[str, object]] = {}
    for item in output:
        if item.get("label_kind") != "tick_label" or item.get("canonical_axis") != axis_key:
            continue
        try:
            label_index = int(item.get("canonical_index"))
        except (TypeError, ValueError):
            continue
        if label_index < 0 or label_index >= len(labels):
            continue
        center = item.get("center", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        assigned[label_index] = item
    missing = [index for index in range(len(labels)) if index not in assigned]
    if not missing or any(index not in {0, len(labels) - 1} for index in missing) or len(missing) > 2:
        return []
    if len(assigned) < 3:
        return []

    anchors: list[tuple[float, float, float, float, float]] = []
    for label_index, item in assigned.items():
        numeric = parse_numeric_label(labels[label_index])
        center = item.get("center", [0.0, 0.0])
        size = item.get("size", [0.0, 0.0])
        if numeric is None or not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [0.0, 0.0]
        cx = float(center[0])
        cy = float(center[1])
        coord = cx if axis_key == "x_axis" else cy
        perp = cy if axis_key == "x_axis" else cx
        anchors.append((float(numeric), coord, perp, max(8.0, float(size[0])), max(8.0, float(size[1]))))
    if len(anchors) < 3:
        return []

    def nearby_opposite_axis_real_tick(label: str, cx: float, cy: float, width: float, height: float) -> dict[str, object] | None:
        other_axis = "y_axis" if axis_key == "x_axis" else "x_axis"
        threshold = max(10.0, min(28.0, max(width, height) * 1.15))
        for item in output:
            if item.get("label_kind") != "tick_label" or item.get("canonical_axis") != other_axis:
                continue
            if item.get("mllm_pseudo_box"):
                continue
            if not same_label_value(str(item.get("text", "") or ""), label):
                continue
            center = item.get("center", [0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            try:
                distance = float(np.hypot(float(center[0]) - cx, float(center[1]) - cy))
            except (TypeError, ValueError):
                continue
            if distance <= threshold:
                return item
        return None

    xs = np.array([anchor[0] for anchor in anchors], dtype=np.float64)
    coords = np.array([anchor[1] for anchor in anchors], dtype=np.float64)
    if float(np.ptp(xs)) <= 0:
        return []
    slope, intercept = np.polyfit(xs, coords, 1)
    predicted_existing = slope * xs + intercept
    residual = float(np.median(np.abs(coords - predicted_existing)))
    sorted_coords = np.sort(coords)
    step = float(np.median(np.abs(np.diff(sorted_coords)))) if len(sorted_coords) >= 3 else abs(float(slope))
    if residual > max(8.0, step * 0.35):
        return []

    perps = [anchor[2] for anchor in anchors]
    widths = [anchor[3] for anchor in anchors]
    heights = [anchor[4] for anchor in anchors]
    perp = float(np.median(perps))
    width = float(np.median(widths))
    height = float(np.median(heights))
    h, w = image_shape if image_shape is not None else (0, 0)
    events: list[dict[str, object]] = []
    for label_index in missing:
        numeric = parse_numeric_label(labels[label_index])
        if numeric is None:
            continue
        coord = float(slope * float(numeric) + intercept)
        if axis_key == "x_axis":
            cx, cy = coord, perp
            if image_shape is not None and not (-width <= cx <= w + width):
                continue
        else:
            cx, cy = perp, coord
            if image_shape is not None and not (-height <= cy <= h + height):
                continue
        conflicting_item = nearby_opposite_axis_real_tick(labels[label_index], cx, cy, width, height)
        pseudo = {
            "text": labels[label_index],
            "canonical_text": labels[label_index],
            "box": rect_box(cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5),
            "center": [cx, cy],
            "size": [width, height],
            "score": 0.0,
            "role": axis_key,
            "raw_role": "missing",
            "role_source": "mllm_regular_numeric_endpoint_pseudo",
            "role_confidence": 0.58,
            "role_reason": "regular_numeric_endpoint_extrapolation",
            "text_source": "mllm_axis_tick",
            "label_kind": "tick_label",
            "canonical_axis": axis_key,
            "canonical_index": label_index,
            "canonical_match_source": "regular_numeric_endpoint_extrapolated",
            "mllm_pseudo_box": True,
            "pseudo_source": "regular_numeric_endpoint_extrapolation",
            "pseudo_index": label_index,
            "pseudo_count": len(labels),
        }
        if conflicting_item is not None:
            pseudo["shared_origin_opposite_axis_tick"] = {
                "axis": conflicting_item.get("canonical_axis"),
                "text": conflicting_item.get("text"),
                "center": conflicting_item.get("center"),
            }
            pseudo["role_reason"] = "regular_numeric_endpoint_extrapolation_shared_origin"
        output.append(pseudo)
        events.append(
            {
                "axis": axis_key,
                "kind": "tick_label",
                "index": label_index,
                "old_text": "",
                "new_text": labels[label_index],
                "source": "regular_numeric_endpoint_extrapolated",
            }
        )
    return events

def canonicalize_items_with_mllm_text(
    items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output = [dict(item) for item in items]
    events: list[dict[str, object]] = []
    axis_used: dict[str, set[int]] = {"x_axis": set(), "y_axis": set()}
    model_axes: set[str] = set()
    globally_used: set[int] = set()
    for axis_key in ("y_axis", "x_axis"):
        labels = mllm_axis_ticks(mllm_result, axis_key)
        axis_title = model_axis_label_text(mllm_result, axis_key)
        if labels or axis_title:
            model_axes.add(axis_key)
        candidate_indexes = expanded_axis_tick_candidates(output, axis_key, labels, globally_used)
        category_assignments = category_axis_order_assignments(output, mllm_result, axis_key, labels, globally_used)
        if category_assignments:
            for label_index, candidate_index, source in category_assignments:
                label = labels[label_index]
                old_text = str(output[candidate_index].get("text", "") or "")
                old_role = output[candidate_index].get("role")
                output[candidate_index]["text"] = label
                output[candidate_index]["canonical_text"] = label
                output[candidate_index]["text_source"] = "mllm_axis_tick"
                output[candidate_index]["label_kind"] = "tick_label"
                output[candidate_index]["canonical_axis"] = axis_key
                output[candidate_index]["canonical_index"] = label_index
                output[candidate_index]["canonical_match_source"] = source
                if old_role != axis_key:
                    output[candidate_index]["raw_role_before_canonical"] = old_role
                    output[candidate_index]["role"] = axis_key
                    output[candidate_index]["role_source"] = "canonical_category_order_reassigned"
                    output[candidate_index]["role_reason"] = "category_axis_order_geometry_alignment"
                axis_used[axis_key].add(candidate_index)
                globally_used.add(candidate_index)
                if old_text != label:
                    events.append({"axis": axis_key, "kind": "tick_label", "index": label_index, "old_text": old_text, "new_text": label, "source": source})
            pseudo_events = add_category_axis_missing_pseudo_items(output, axis_key, labels, category_assignments, image_shape)
            pseudo_start = len(output) - len(pseudo_events)
            for pseudo_offset, event in enumerate(pseudo_events):
                pseudo_index = pseudo_start + pseudo_offset
                axis_used[axis_key].add(pseudo_index)
                globally_used.add(pseudo_index)
                events.append(event)
            candidate_indexes = [index for index in candidate_indexes if index not in globally_used]
        sequence_assignments = regular_numeric_axis_sequence_assignments(output, mllm_result, axis_key, labels, globally_used)
        if sequence_assignments:
            for label_index, candidate_index, source in sequence_assignments:
                label = labels[label_index]
                old_text = str(output[candidate_index].get("text", "") or "")
                output[candidate_index]["text"] = label
                output[candidate_index]["canonical_text"] = label
                output[candidate_index]["text_source"] = "mllm_axis_tick"
                output[candidate_index]["label_kind"] = "tick_label"
                output[candidate_index]["canonical_axis"] = axis_key
                output[candidate_index]["canonical_index"] = label_index
                output[candidate_index]["canonical_match_source"] = source
                if str(source).startswith("regular_numeric_axis_order:") and output[candidate_index].get("role") != axis_key:
                    output[candidate_index]["raw_role_before_canonical"] = output[candidate_index].get("role")
                    output[candidate_index]["role"] = axis_key
                    output[candidate_index]["role_source"] = "canonical_regular_numeric_order_reassigned"
                    output[candidate_index]["role_reason"] = "declared_numeric_axis_order_geometry_alignment"
                axis_used[axis_key].add(candidate_index)
                globally_used.add(candidate_index)
                if old_text != label:
                    events.append({"axis": axis_key, "kind": "tick_label", "index": label_index, "old_text": old_text, "new_text": label, "source": source})
            candidate_indexes = [index for index in candidate_indexes if index not in globally_used]
        if labels and candidate_indexes:
            unmatched_candidates = set(candidate_indexes)
            assignments: list[tuple[int, int, str]] = []
            reference, tolerance = axis_perpendicular_reference(output, axis_key)
            declared_order_assignments = declared_numeric_axis_order_assignments(output, mllm_result, axis_key, labels, candidate_indexes)
            if declared_order_assignments:
                assignments.extend(declared_order_assignments)
                for _, candidate_index, _ in declared_order_assignments:
                    unmatched_candidates.discard(candidate_index)
            else:
                for label_index, label in enumerate(labels):
                    if label_index in {int(output[index].get("canonical_index")) for index in axis_used[axis_key] if output[index].get("canonical_index") is not None}:
                        continue
                    matches: list[tuple[int, str]] = []
                    for candidate_index in list(unmatched_candidates):
                        candidate = output[candidate_index]
                        text = str(candidate.get("text", ""))
                        perp_score = axis_perpendicular_score(candidate, axis_key, reference, tolerance)
                        if axis_key == "y_axis" and perp_score < 0.35 and not candidate.get("mllm_pseudo_box"):
                            continue
                        if same_label_value(text, label):
                            matches.append((candidate_index, "text_or_numeric_match"))
                        elif axis_key == "y_axis" and numeric_prefix_label_value(text, label) and perp_score >= 0.35:
                            matches.append((candidate_index, "numeric_prefix_axis_match"))
                    if axis_key == "y_axis" and zero_tick_text(label):
                        matches = [
                            (candidate_index, source)
                            for candidate_index, source in matches
                            if not likely_x_axis_origin_zero_candidate(output, candidate_index, mllm_result)
                        ]
                    if matches:
                        best_index, source = max(
                            matches,
                            key=lambda match: canonical_candidate_score(output[match[0]], axis_key, reference, tolerance),
                        )
                        unmatched_candidates.remove(best_index)
                        if output[best_index].get("role") != axis_key:
                            if source == "text_or_numeric_match":
                                source = "cross_axis_geometry_match"
                            output[best_index]["raw_role_before_canonical"] = output[best_index].get("role")
                            output[best_index]["role"] = axis_key
                            output[best_index]["role_source"] = "canonical_cross_axis_reassigned"
                            output[best_index]["role_reason"] = "same_tick_text_near_axis_baseline_or_column"
                        assignments.append((label_index, best_index, source))
            if len(assignments) < len(labels):
                assigned_label_indexes = {label_index for label_index, _, _ in assignments}
                remaining_labels = [index for index in range(len(labels)) if index not in assigned_label_indexes]
                remaining_candidates = [index for index in candidate_indexes if index in unmatched_candidates]
                if len(remaining_labels) == len(remaining_candidates):
                    assignments.extend(
                        (label_index, candidate_index, "order_alignment")
                        for label_index, candidate_index in zip(remaining_labels, remaining_candidates)
                    )
            for label_index, candidate_index, source in assignments:
                label = labels[label_index]
                old_text = str(output[candidate_index].get("text", "") or "")
                output[candidate_index]["text"] = label
                output[candidate_index]["canonical_text"] = label
                output[candidate_index]["text_source"] = "mllm_axis_tick"
                output[candidate_index]["label_kind"] = "tick_label"
                output[candidate_index]["canonical_axis"] = axis_key
                output[candidate_index]["canonical_index"] = label_index
                output[candidate_index]["canonical_match_source"] = source
                if str(source).startswith("regular_numeric_axis_order:") and output[candidate_index].get("role") != axis_key:
                    output[candidate_index]["raw_role_before_canonical"] = output[candidate_index].get("role")
                    output[candidate_index]["role"] = axis_key
                    output[candidate_index]["role_source"] = "canonical_regular_numeric_order_reassigned"
                    output[candidate_index]["role_reason"] = "declared_numeric_axis_order_geometry_alignment"
                axis_used[axis_key].add(candidate_index)
                globally_used.add(candidate_index)
                if old_text != label:
                    events.append({"axis": axis_key, "kind": "tick_label", "index": label_index, "old_text": old_text, "new_text": label, "source": source})
        endpoint_events = add_regular_numeric_endpoint_pseudo_items(output, axis_key, labels, image_shape)
        events.extend(endpoint_events)
        if axis_title:
            title_index = choose_axis_title_candidate(output, axis_key, axis_used[axis_key])
            if title_index is not None:
                old_text = str(output[title_index].get("text", "") or "")
                old_role = output[title_index].get("role")
                output[title_index]["text"] = axis_title
                output[title_index]["canonical_text"] = axis_title
                output[title_index]["text_source"] = "mllm_axis_title"
                output[title_index]["label_kind"] = "axis_label"
                output[title_index]["canonical_axis"] = axis_key
                output[title_index]["axis_label_role"] = axis_key
                output[title_index]["raw_role_before_axis_label_other"] = old_role
                output[title_index]["role"] = "other"
                output[title_index]["role_source"] = "canonical_axis_label_as_other"
                output[title_index]["role_reason"] = "axis_label_kept_as_other_text"
                axis_used[axis_key].add(title_index)
                globally_used.add(title_index)
                if old_text != axis_title:
                    events.append({"axis": axis_key, "kind": "axis_label", "old_text": old_text, "new_text": axis_title, "source": "axis_title_candidate"})
    used_all = axis_used["x_axis"] | axis_used["y_axis"]
    for index, item in enumerate(output):
        if item.get("split_source") == "ocr_numeric_gap_sequence" and item.get("role") in {"x_axis", "y_axis"}:
            used_all.add(index)
            item.setdefault("text_source", "ocr_regular_sequence")
            item.setdefault("label_kind", "tick_label")
        if (
            item.get("mllm_pseudo_box")
            and item.get("role") in {"x_axis", "y_axis"}
            and item.get("canonical_axis") in {"x_axis", "y_axis"}
        ):
            used_all.add(index)
            item.setdefault("text_source", "mllm_pseudo_interpolated")
            item.setdefault("label_kind", "tick_label")
    for index, item in enumerate(output):
        if index in used_all:
            continue
        if item.get("role") in model_axes:
            item["raw_role_before_canonical"] = item.get("role")
            if item.get("mllm_pseudo_box"):
                item["suppressed_mllm_pseudo_box"] = True
                item["mllm_pseudo_box"] = False
            item["role"] = "other"
            item["label_kind"] = "other"
            item["role_source"] = "canonical_other_unmatched_mllm"
            item["role_reason"] = "not_matched_to_mllm_axis_text_or_locked_sequence"
        else:
            item.setdefault("label_kind", "other" if item.get("role") == "other" else "tick_label")
            item.setdefault("role_source", item.get("role_source", "ocr_geometry"))
    return output, events

def refine_ocr_roles_with_mllm(
    ocr_items: list[dict[str, object]],
    mllm_result: dict[str, object],
    image_shape: tuple[int, int],
) -> list[dict[str, object]]:
    if not ocr_items:
        return []
    if not isinstance(mllm_result, dict) or not mllm_result.get("enabled") or mllm_result.get("error"):
        return [dict(item) for item in ocr_items]

    axis_payloads = {
        "x_axis": mllm_result.get("x_axis", {}),
        "y_axis": mllm_result.get("y_axis", {}),
    }
    label_sets: dict[str, dict[str, set]] = {}
    for role, axis in axis_payloads.items():
        if not isinstance(axis, dict):
            axis = {}
        tick_labels = collect_mllm_label_set(axis, "tick_labels")
        axis_labels = collect_mllm_label_set(axis, "axis_label")
        label_sets[role] = {
            "tick": tick_labels,
            "axis_label": axis_labels,
            "tick_numeric": numeric_label_set(tick_labels),
            "axis_numeric": numeric_label_set(axis_labels),
        }

    axis_texts = (
        label_sets["x_axis"]["tick"]
        | label_sets["x_axis"]["axis_label"]
        | label_sets["y_axis"]["tick"]
        | label_sets["y_axis"]["axis_label"]
    )
    mllm_other_texts = collect_mllm_other_texts(mllm_result) - axis_texts

    legend_like_texts: set[str] = set()
    x_axis_payload = axis_payloads.get("x_axis", {})
    x_axis_confidence = 0.0
    if isinstance(x_axis_payload, dict):
        try:
            x_axis_confidence = float(x_axis_payload.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            x_axis_confidence = 0.0
    x_axis_type = str(x_axis_payload.get("type", "") if isinstance(x_axis_payload, dict) else "").strip().lower()
    if x_axis_confidence >= 0.55 and label_sets["x_axis"]["tick"] and x_axis_type != "category":
        h, _ = image_shape
        unmatched_rows: list[list[dict[str, object]]] = []
        for item in ocr_items:
            text = str(item.get("text", "") or "").strip()
            if not text or str(item.get("role", "other")) != "x_axis" or parse_numeric_label(text) is not None:
                continue
            center = item.get("center", [0.0, 0.0])
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            cy = float(center[1])
            if cy < h * 0.72:
                continue
            tick_match, _ = text_matches_label_set(
                text,
                label_sets["x_axis"]["tick"],
                label_sets["x_axis"]["tick_numeric"],
            )
            axis_match, _ = text_matches_label_set(
                text,
                label_sets["x_axis"]["axis_label"],
                label_sets["x_axis"]["axis_numeric"],
            )
            if tick_match or axis_match:
                continue
            payload = {"text": text, "y": cy}
            for row in unmatched_rows:
                row_y = float(np.median([float(value["y"]) for value in row]))
                if abs(cy - row_y) <= max(10.0, h * 0.025):
                    row.append(payload)
                    break
            else:
                unmatched_rows.append([payload])
        for row in unmatched_rows:
            if len(row) >= 2:
                legend_like_texts.update(normalize_label_text(value["text"]) for value in row)

    refined: list[dict[str, object]] = []
    for item in ocr_items:
        text = str(item.get("text", "") or "").strip()
        raw_role = str(item.get("role", "other"))
        copy = dict(item)
        copy["raw_role"] = raw_role
        copy["role_source"] = "ocr_geometry"
        copy["role_confidence"] = round(float(item.get("score", 0.0) or 0.0), 3)
        if not text:
            refined.append(copy)
            continue
        normalized_text = normalize_label_text(text)

        candidates: list[dict[str, object]] = []
        for role in ("x_axis", "y_axis"):
            axis = axis_payloads.get(role, {})
            axis_confidence = 0.0
            if isinstance(axis, dict):
                try:
                    axis_confidence = float(axis.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    axis_confidence = 0.0
            if axis_confidence < 0.50:
                continue
            labels = label_sets[role]
            axis_match, axis_match_kind = text_matches_label_set(
                text,
                labels["axis_label"],
                labels["axis_numeric"],
            )
            tick_match, tick_match_kind = text_matches_label_set(
                text,
                labels["tick"],
                labels["tick_numeric"],
            )
            if not axis_match and not tick_match:
                continue
            match_strength = 0.48 if axis_match else 0.34
            match_kind = "axis_label" if axis_match else f"tick_{tick_match_kind}"
            score = match_strength + axis_confidence * 0.28 + role_geometry_score(copy, role, image_shape)
            candidates.append(
                {
                    "role": role,
                    "score": round(min(1.0, score), 3),
                    "match": match_kind,
                    "axis_confidence": round(axis_confidence, 3),
                }
            )

        if not candidates:
            if raw_role in {"x_axis", "y_axis"} and text_matches_mllm_other_text(text, mllm_other_texts):
                copy["role"] = "other"
                copy["role_source"] = "mllm_other_text_rejected_axis"
                copy["role_confidence"] = round(float(item.get("score", 0.0) or 0.0), 3)
                copy["role_reason"] = "matched_mllm_other_text"
            elif raw_role == "x_axis" and normalized_text in legend_like_texts:
                copy["role"] = "other"
                copy["role_source"] = "mllm_rejected_legend"
                copy["role_confidence"] = round(float(item.get("score", 0.0) or 0.0), 3)
                copy["role_reason"] = "unmatched_lower_row_legend_text"
            refined.append(copy)
            continue
        candidates = sorted(candidates, key=lambda value: float(value["score"]), reverse=True)
        best = candidates[0]
        second_score = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        best_score = float(best["score"])
        best_role = str(best["role"])
        best_match = str(best.get("match", ""))
        parsed = parse_numeric_label(text)
        ambiguous_numeric = parsed is not None and len(candidates) > 1 and abs(best_score - second_score) < 0.18

        if best_match == "axis_label" and best_score >= 0.62:
            copy["role"] = "other"
            copy["role_source"] = "mllm_axis_label_as_other"
            copy["role_confidence"] = round(best_score, 3)
            copy["role_reason"] = "axis_label_kept_as_other_text"
            copy["label_kind"] = "axis_label"
            copy["canonical_axis"] = best_role
            copy["axis_label_role"] = best_role
            copy["text_source"] = "mllm_axis_title"
            copy["role_candidates"] = candidates[:2]
        elif text_matches_mllm_other_text(text, mllm_other_texts) and best_score < 0.80:
            copy["role"] = "other"
            copy["role_source"] = "mllm_other_text_rejected_axis"
            copy["role_confidence"] = round(max(float(item.get("score", 0.0) or 0.0), 0.72), 3)
            copy["role_reason"] = "matched_mllm_other_text_over_weak_axis_match"
            copy["role_candidates"] = candidates[:2]
        elif (
            best_role != raw_role
            and not ambiguous_numeric
            and best_score >= (0.62 if raw_role == "other" else 0.72)
            and best_score - second_score >= 0.10
        ):
            copy["role"] = best_role
            copy["role_source"] = "mllm_refined"
            copy["role_confidence"] = round(best_score, 3)
            copy["role_reason"] = best["match"]
            copy["role_candidates"] = candidates[:2]
        elif best_role == raw_role and best_score >= 0.60:
            copy["role_source"] = "ocr_geometry+mllm_confirmed"
            copy["role_confidence"] = round(best_score, 3)
            copy["role_reason"] = best["match"]
            copy["role_candidates"] = candidates[:2]
        else:
            copy["role_candidates"] = candidates[:2]
        refined.append(copy)
    return apply_axis_tick_cluster_role_correction(refined, mllm_result, image_shape)

def classify_axis_type(axis_items: list[dict[str, object]], role: str) -> dict[str, object]:
    values: list[dict[str, object]] = []
    for item in axis_items:
        if item.get("label_kind") == "axis_label" or item.get("text_source") == "mllm_axis_title":
            continue
        text = str(item.get("text", "")).strip()
        center = item.get("center", [0.0, 0.0])
        if not text or not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        numeric = parse_numeric_label(text)
        values.append(
            {
                "text": text,
                "numeric": numeric,
                "x": float(center[0]),
                "y": float(center[1]),
                "score": float(item.get("score", 0.0)),
                "center": [float(center[0]), float(center[1])],
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

    numeric_values = [item for item in values if item["numeric"] is not None]
    text_values = [item for item in values if item["numeric"] is None]
    count = len(values)
    numeric_count = len(numeric_values)
    text_count = len(text_values)

    if numeric_count >= max(2, count * 0.55):
        axis_type = "numeric"
    elif text_count >= max(2, count * 0.55):
        axis_type = "category"
    elif count >= 2:
        axis_type = "mixed"
    else:
        axis_type = "unknown"

    positions = [float(item["x"] if role == "x_axis" else item["y"]) for item in values]
    span = float(max(positions) - min(positions)) if len(positions) >= 2 else 0.0
    regularity = regularity_score(positions) if len(positions) >= 3 else 0.0
    confidence = 0.0
    if count:
        confidence += min(0.45, count * 0.06)
        confidence += min(0.30, span / 500.0)
        confidence += 0.15 if regularity <= 0.50 else 0.0
        confidence += 0.10 if axis_type in {"numeric", "category"} else 0.0

    return {
        "type": axis_type,
        "count": count,
        "numeric_count": numeric_count,
        "text_count": text_count,
        "span": span,
        "regularity": regularity,
        "confidence": round(min(1.0, confidence), 3),
        "ticks": values,
    }

def label_like_score(text: str) -> float:
    score = 0.0
    if len(text) >= 3:
        score += min(0.25, len(text) / 80.0)
    if " " in text:
        score += 0.18
    if any(char in text for char in "()/%"):
        score += 0.14
    if re.search(r"[A-Za-z]{3,}", text):
        score += 0.12
    if parse_numeric_label(text) is not None:
        score -= 0.50
    return score

def likely_non_axis_annotation(text: str) -> bool:
    value = normalize_label_text(text)
    if not value:
        return False
    if any(token in value for token in ("source:", "ourworldindata", "cc by", "our world in data")):
        return True
    if value in {"annual average", "monthly average"}:
        return True
    return False

def extract_ocr_axis_label(
    ocr_items: list[dict[str, object]],
    role: str,
    image_shape: tuple[int, int],
    tick_texts: set[str] | None = None,
) -> dict[str, object]:
    h, w = image_shape
    tick_texts = tick_texts or set()
    candidates: list[dict[str, object]] = []
    for item in ocr_items:
        text = str(item.get("text", "")).strip()
        if normalize_label_text(text) in tick_texts:
            continue
        role_source = str(item.get("role_source", "") or "")
        center = item.get("center", [0.0, 0.0])
        size = item.get("size", [0.0, 0.0])
        if not text or parse_numeric_label(text) is not None:
            continue
        if role_source in {"mllm_rejected_legend", "mllm_other_text_rejected_axis", "canonical_other_unmatched_mllm"}:
            continue
        if likely_non_axis_annotation(text):
            continue
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            continue
        if not isinstance(size, (list, tuple)) or len(size) < 2:
            size = [0.0, 0.0]
        cx = float(center[0])
        cy = float(center[1])
        width = float(size[0])
        height = float(size[1])
        item_role = str(item.get("role", "other"))
        score = float(item.get("score", 0.0)) * 0.45 + label_like_score(text)

        if role == "x_axis":
            near_axis = item_role == "x_axis" or (cy >= h * 0.82 and w * 0.18 <= cx <= w * 0.85)
            if not near_axis:
                continue
            score += max(0.0, min(0.20, (cy - h * 0.78) / max(1.0, h * 0.22) * 0.20))
            score += 0.08 if w * 0.22 <= cx <= w * 0.82 else 0.0
            score += 0.06 if width >= w * 0.10 else 0.0
        else:
            vertical_text = height > width * 1.3
            near_axis = item_role == "y_axis" or (vertical_text and cx <= w * 0.25)
            if not near_axis:
                continue
            if cy <= h * 0.12:
                continue
            score += max(0.0, min(0.20, (w * 0.25 - cx) / max(1.0, w * 0.25) * 0.20))
            score += 0.08 if vertical_text else 0.0
            score += 0.06 if len(text) >= 6 else 0.0

        candidates.append(
            {
                "text": text,
                "confidence": round(max(0.0, min(1.0, score)), 3),
                "ocr_score": round(float(item.get("score", 0.0)), 3),
                "center": [cx, cy],
                "size": [width, height],
                "role": item_role,
                "role_source": role_source,
            }
        )

    candidates = sorted(candidates, key=lambda value: float(value["confidence"]), reverse=True)
    best = candidates[0] if candidates else None
    if best is None:
        return {"text": "", "confidence": 0.0, "source": "ocr", "candidates": []}

    nonnumeric_count = len(candidates)
    best_text = str(best["text"])
    best_confidence = float(best["confidence"])
    short_symbol_like = len(best_text) < 3 and not any(char in best_text for char in " ()/%")
    category_like = nonnumeric_count >= 3 and len(best_text) <= 5 and not any(char in best_text for char in " ()/%")
    if best_confidence < 0.48 or category_like or short_symbol_like:
        return {
            "text": "",
            "confidence": round(best_confidence, 3),
            "source": "ocr",
            "candidates": candidates[:10],
        }
    return {
        "text": best_text,
        "confidence": round(best_confidence, 3),
        "source": "ocr",
        "candidates": candidates[:10],
    }

def build_ocr_axis_evidence(
    ocr_items: list[dict[str, object]],
    image_shape: tuple[int, int],
) -> dict[str, object]:
    h, w = image_shape
    x_items = [
        item
        for item in ocr_items
        if item.get("role") == "x_axis"
        and item.get("label_kind") != "axis_label"
        and item.get("text_source") != "mllm_axis_title"
    ]
    y_items = [
        item
        for item in ocr_items
        if item.get("role") == "y_axis"
        and item.get("label_kind") != "axis_label"
        and item.get("text_source") != "mllm_axis_title"
    ]
    x_axis = classify_axis_type(x_items, "x_axis")
    y_axis = classify_axis_type(y_items, "y_axis")
    x_tick_texts = {normalize_label_text(item.get("text", "")) for item in x_items if item.get("label_kind") != "axis_label"}
    y_tick_texts = {normalize_label_text(item.get("text", "")) for item in y_items if item.get("label_kind") != "axis_label"}
    all_tick_texts = x_tick_texts | y_tick_texts
    x_axis["axis_label"] = extract_ocr_axis_label(ocr_items, "x_axis", image_shape, all_tick_texts)
    y_axis["axis_label"] = extract_ocr_axis_label(ocr_items, "y_axis", image_shape, all_tick_texts)
    return {
        "image_size": {"width": w, "height": h},
        "x_axis": x_axis,
        "y_axis": y_axis,
    }
