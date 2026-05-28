import argparse
import contextlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from glob import glob
from statistics import median

import cv2
import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BACKEND_DIR, "evaluation", "results")
GRID_DIR = os.path.join(BACKEND_DIR, "Grid_generation")
if GRID_DIR not in sys.path:
    sys.path.insert(0, GRID_DIR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from function_calling.axis.detect_lines import detect_candidate_lines
from function_calling.axis.infer_axes import infer_axes_from_lines
from function_calling.axis.merge_lines import merge_similar_lines
from function_calling.label.extract_tick_labels_with_llm import (
    build_cache_metadata,
    cache_result_quality,
    extract_tick_labels_with_llm,
    get_cache_file_path,
    load_llm_cache,
)
from function_calling.ticks.detect_ticks import scan_pixels_for_ticks
from function_calling.ticks.filter_ticks import filter_ticks


CHART_ROOT = os.path.join(BACKEND_DIR, "charts")
TICK_CACHE_DIR = os.path.join(BACKEND_DIR, "data", "llm_cache", "tick_labels")
DEFAULT_DATASET_ID = "backend_charts"
DEFAULT_TYPES = ["line", "scatter", "bubble", "v_bar", "h_bar"]


def base_pngs(chart_type):
    paths = []
    for path in glob(os.path.join(CHART_ROOT, chart_type, "*.png")):
        name = os.path.basename(path).lower()
        if any(part in name for part in ["_grid", "grid_with_grid", "with_grid", "temp", "encode", "marked"]):
            continue
        if os.path.exists(os.path.splitext(path)[0] + ".json"):
            paths.append(path)
    return sorted(paths)


def read_json_for(image_path):
    with open(os.path.splitext(image_path)[0] + ".json", "r", encoding="utf-8-sig") as f:
        return json.load(f)


def normalize_tick(value):
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return ("str", "nan")
        return ("num", round(float(value), 6))
    text = str(value).strip()
    if text.lower() in {"nan", "na", "n/a", "null", "none"}:
        return ("str", text.lower())
    try:
        number = float(text)
        if math.isnan(number) or math.isinf(number):
            return ("str", text.lower())
        return ("num", round(number, 6))
    except ValueError:
        return ("str", " ".join(text.lower().split()))


def ticks_equal(a, b, numeric_tol=1e-2):
    na, nb = normalize_tick(a), normalize_tick(b)
    if na[0] == "num" and nb[0] == "num":
        return abs(na[1] - nb[1]) <= numeric_tol
    return na == nb


def gt_axis_type(values):
    values = list(values or [])
    if not values:
        return "unknown"
    if all(isinstance(v, (int, float)) for v in values):
        return "numeric"
    return "text"


def normalize_axis_type(value):
    text = str(value or "").lower()
    if "数值" in text or "numeric" in text:
        return "numeric"
    if "文字" in text or "文本" in text or "category" in text or "categorical" in text or "text" in text:
        return "text"
    return "unknown"


def pixel_recall(pred_pixels, gt_pixels, tol=6):
    used = set()
    matches = 0
    for pred in pred_pixels:
        best = None
        best_dist = float("inf")
        for idx, gt in enumerate(gt_pixels):
            if idx in used:
                continue
            dist = abs(float(pred) - float(gt))
            if dist < best_dist:
                best = idx
                best_dist = dist
        if best is not None and best_dist <= tol:
            used.add(best)
            matches += 1
    return matches / len(gt_pixels) if gt_pixels else 0.0


def value_recall(pred_values, gt_values):
    total = len(gt_values)
    if total == 0:
        return 0.0
    matches = sum(
        ticks_equal(pred_values[i], gt_values[i])
        for i in range(min(len(pred_values), len(gt_values)))
    )
    return matches / total


def paired_recall(pred_pixels, pred_values, gt_pixels, gt_values, pixel_tol=6):
    total = min(len(gt_pixels), len(gt_values))
    if total == 0:
        return 0.0
    matched = 0
    used = set()
    for pred_pixel, pred_value in zip(pred_pixels, pred_values):
        best = None
        best_dist = float("inf")
        for idx, gt_pixel in enumerate(gt_pixels[:total]):
            if idx in used:
                continue
            dist = abs(float(pred_pixel) - float(gt_pixel))
            if dist < best_dist:
                best = idx
                best_dist = dist
        if best is not None and best_dist <= pixel_tol and ticks_equal(pred_value, gt_values[best]):
            used.add(best)
            matched += 1
    return matched / total


def align_values_to_pixels(values, pixels):
    values = list(values or [])
    pixels = list(pixels or [])
    if len(values) > len(pixels):
        return values[: len(pixels)]
    if len(values) < len(pixels) and len(values) >= 2:
        try:
            numeric = [float(v) for v in values]
            src = np.linspace(0, 1, len(numeric))
            dst = np.linspace(0, 1, len(pixels))
            return np.interp(dst, src, numeric).tolist()
        except Exception:
            return values
    return values


def cv_tick_pixels(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    raw_lines = detect_candidate_lines(
        gray,
        canny_threshold1=30,
        canny_threshold2=100,
        hough_threshold=15,
        min_length=15,
        max_gap=15,
    )
    merged_lines = merge_similar_lines(raw_lines)
    with contextlib.redirect_stdout(io.StringIO()):
        x_axis, y_axis, _ = infer_axes_from_lines(merged_lines, (w, h), gray)
    if x_axis is None or y_axis is None:
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        x_ticks = filter_ticks(
            merge_similar_lines(
                scan_pixels_for_ticks(image, x_axis, direction="x", scan_range=20),
                angle_threshold=np.deg2rad(10),
            ),
            direction="x",
        )
        y_ticks = filter_ticks(
            merge_similar_lines(
                scan_pixels_for_ticks(image, y_axis, direction="y", scan_range=20),
                angle_threshold=np.deg2rad(10),
            ),
            direction="y",
        )
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "x_pixels": sorted([(t[0] + t[2]) // 2 for t in x_ticks]),
        "y_pixels": sorted([(t[1] + t[3]) // 2 for t in y_ticks], reverse=True),
        "image_shape": image.shape[:2],
    }


def evaluate_image(image_path, allow_api, dataset_id):
    gt = read_json_for(image_path)
    gt_x_pixels = gt.get("x_pixels", [])
    gt_y_pixels = gt.get("y_pixels", [])
    gt_x_ticks = gt.get("x_ticks", [])
    gt_y_ticks = gt.get("y_ticks", [])

    cv_result = cv_tick_pixels(image_path)
    cache_metadata = build_cache_metadata(image_path, dataset_id=dataset_id)
    cache_file = get_cache_file_path(
        image_path,
        TICK_CACHE_DIR,
        dataset_id=dataset_id,
        prompt_signature=cache_metadata["prompt_signature"],
    )
    cache_file_exists_before = os.path.exists(cache_file)
    cache_valid_before = load_llm_cache(cache_file, expected_metadata=cache_metadata) is not None
    invalid_cache_before = False
    invalid_cache_reason = None
    if cache_file_exists_before and not cache_valid_before:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                invalid_cache_reason = cache_result_quality(json.load(f))["reason"]
        except Exception as exc:
            invalid_cache_reason = f"unreadable:{exc}"
        invalid_cache_before = True
    llm = extract_tick_labels_with_llm(
        image_path,
        cache_dir=TICK_CACHE_DIR,
        allow_api=allow_api,
        dataset_id=dataset_id,
    )
    cache_valid_after = load_llm_cache(cache_file, expected_metadata=cache_metadata) is not None

    if not cv_result:
        return {
            "image_path": image_path,
            "chart_type": os.path.basename(os.path.dirname(image_path)),
            "cv_failed": True,
            "cache_hit": cache_valid_before,
            "cache_file_exists": cache_file_exists_before,
            "invalid_cache": invalid_cache_before,
            "invalid_cache_reason": invalid_cache_reason,
            "cache_created": cache_valid_after and not cache_valid_before,
            "cache_miss": bool(llm.get("cache_miss")),
            "api_failed": bool(llm.get("api_failed")),
        }

    pred_x_axis_type = normalize_axis_type(llm.get("x_axis_type"))
    pred_y_axis_type = normalize_axis_type(llm.get("y_axis_type"))
    gt_x_axis_type = gt_axis_type(gt_x_ticks)
    gt_y_axis_type = gt_axis_type(gt_y_ticks)
    x_pixels = cv_result["x_pixels"]
    y_pixels = cv_result["y_pixels"]
    x_values = align_values_to_pixels(llm.get("x_ticks", []), x_pixels)
    y_values = align_values_to_pixels(llm.get("y_ticks", []), y_pixels)

    return {
        "image_path": image_path,
        "chart_type": os.path.basename(os.path.dirname(image_path)),
        "cv_failed": False,
        "cache_hit": cache_valid_before,
        "cache_file_exists": cache_file_exists_before,
        "invalid_cache": invalid_cache_before,
        "invalid_cache_reason": invalid_cache_reason,
        "cache_hit_reported": bool(llm.get("cache_hit")),
        "cache_created": cache_valid_after and not cache_valid_before,
        "cache_miss": bool(llm.get("cache_miss")),
        "api_failed": bool(llm.get("api_failed")),
        "cache_file": llm.get("cache_file", cache_file),
        "x_pixel_recall": pixel_recall(x_pixels, gt_x_pixels),
        "y_pixel_recall": pixel_recall(y_pixels, gt_y_pixels),
        "x_value_recall": value_recall(x_values, gt_x_ticks),
        "y_value_recall": value_recall(y_values, gt_y_ticks),
        "x_pair_recall": paired_recall(x_pixels, x_values, gt_x_pixels, gt_x_ticks),
        "y_pair_recall": paired_recall(y_pixels, y_values, gt_y_pixels, gt_y_ticks),
        "x_axis_type_pred": pred_x_axis_type,
        "y_axis_type_pred": pred_y_axis_type,
        "x_axis_type_gt": gt_x_axis_type,
        "y_axis_type_gt": gt_y_axis_type,
        "x_axis_type_correct": pred_x_axis_type == gt_x_axis_type,
        "y_axis_type_correct": pred_y_axis_type == gt_y_axis_type,
        "raw_pred_x_pixels": len(cv_result["x_pixels"]),
        "raw_pred_y_pixels": len(cv_result["y_pixels"]),
        "pred_x_pixels": len(x_pixels),
        "pred_y_pixels": len(y_pixels),
        "pred_x_values": len(x_values),
        "pred_y_values": len(y_values),
        "gt_x_ticks": len(gt_x_ticks),
        "gt_y_ticks": len(gt_y_ticks),
    }


def summarize(rows):
    def avg(items, key):
        vals = [r.get(key, 0.0) for r in items if not r.get("cv_failed")]
        return float(np.mean(vals)) if vals else 0.0

    def avg_all(items, key):
        vals = [r.get(key, 0.0) for r in items]
        return float(np.mean(vals)) if vals else 0.0

    summary = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["chart_type"]].append(row)
    grouped["ALL"] = rows

    for chart_type, items in grouped.items():
        valid_axis_type_rows = [
            r for r in items
            if not r.get("cv_failed") and r.get("x_axis_type_pred") != "unknown" and r.get("y_axis_type_pred") != "unknown"
        ]
        summary[chart_type] = {
            "n": len(items),
            "cv_success_rate": sum(not r.get("cv_failed") for r in items) / len(items) if items else 0.0,
            "cache_hits": sum(r.get("cache_hit", False) for r in items),
            "cache_hits_reported": sum(r.get("cache_hit_reported", False) for r in items),
            "cache_created": sum(r.get("cache_created", False) for r in items),
            "cache_misses_no_api": sum(r.get("cache_miss", False) for r in items),
            "invalid_cache_files": sum(r.get("invalid_cache", False) for r in items),
            "api_failed": sum(r.get("api_failed", False) for r in items),
            "x_pixel_recall": avg(items, "x_pixel_recall"),
            "y_pixel_recall": avg(items, "y_pixel_recall"),
            "x_value_recall": avg(items, "x_value_recall"),
            "y_value_recall": avg(items, "y_value_recall"),
            "x_pair_recall": avg(items, "x_pair_recall"),
            "y_pair_recall": avg(items, "y_pair_recall"),
            "x_pixel_recall_all": avg_all(items, "x_pixel_recall"),
            "y_pixel_recall_all": avg_all(items, "y_pixel_recall"),
            "x_value_recall_all": avg_all(items, "x_value_recall"),
            "y_value_recall_all": avg_all(items, "y_value_recall"),
            "x_pair_recall_all": avg_all(items, "x_pair_recall"),
            "y_pair_recall_all": avg_all(items, "y_pair_recall"),
            "mean_pair_recall_all": (
                avg_all(items, "x_pair_recall") + avg_all(items, "y_pair_recall")
            ) / 2 if items else 0.0,
            "complete_pair_accuracy": (
                sum(
                    (not r.get("cv_failed"))
                    and r.get("x_pair_recall", 0.0) >= 1.0
                    and r.get("y_pair_recall", 0.0) >= 1.0
                    for r in items
                ) / len(items)
                if items else 0.0
            ),
            "axis_type_eval_n": len(valid_axis_type_rows),
            "x_axis_type_accuracy": (
                sum(r.get("x_axis_type_correct", False) for r in valid_axis_type_rows) / len(valid_axis_type_rows)
                if valid_axis_type_rows else 0.0
            ),
            "y_axis_type_accuracy": (
                sum(r.get("y_axis_type_correct", False) for r in valid_axis_type_rows) / len(valid_axis_type_rows)
                if valid_axis_type_rows else 0.0
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate CV + MLLM tick extraction with cache.")
    parser.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    parser.add_argument("--limit", type=int, default=20, help="Per-type limit. Use 0 for all images.")
    parser.add_argument("--cache-only", action="store_true", help="Do not call MLLM APIs on cache miss.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Cache namespace for this dataset/prompt run.")
    parser.add_argument(
        "--output",
        default=os.path.join(RESULTS_DIR, "cv_mllm_tick_eval.json"),
    )
    args = parser.parse_args()

    rows = []
    for chart_type in args.types:
        paths = base_pngs(chart_type)
        if args.limit > 0:
            paths = paths[: args.limit]
        for path in paths:
            rows.append(evaluate_image(path, allow_api=not args.cache_only, dataset_id=args.dataset_id))

    result = {
        "dataset_id": args.dataset_id,
        "cache_dir": TICK_CACHE_DIR,
        "cache_only": args.cache_only,
        "summary": summarize(rows),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
