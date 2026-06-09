import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict
from glob import glob
from statistics import median

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(BACKEND_DIR, "evaluation", "results")
GRID_DIR = os.path.join(BACKEND_DIR, "Grid_generation")
if GRID_DIR not in sys.path:
    sys.path.insert(0, GRID_DIR)

from function_calling.axis.detect_lines import detect_candidate_lines
from function_calling.axis.infer_axes import infer_axes_from_lines
from function_calling.axis.merge_lines import merge_similar_lines
from function_calling.ticks.detect_ticks import scan_pixels_for_ticks
from evaluate_cv_mllm_ticks import (
    DEFAULT_DATASET_ID,
    DEFAULT_TYPES as MLLM_DEFAULT_TYPES,
    evaluate_image as evaluate_cv_mllm_image,
    summarize as summarize_cv_mllm,
)


CHART_ROOT = os.path.join(BACKEND_DIR, "charts")
CARTESIAN_TYPES = ["line", "scatter", "bubble", "v_bar", "h_bar"]
POLAR_TYPES = ["radar", "rose"]
CLASSIFICATION_TYPES = [
    "bubble",
    "donut",
    "h_bar",
    "line",
    "pie",
    "radar",
    "rose",
    "scatter",
    "v_bar",
]


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
    with open(os.path.splitext(image_path)[0] + ".json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_center(meta):
    center = meta.get("center")
    if isinstance(center, dict):
        return np.array([float(center["x"]), float(center["y"])])
    if isinstance(center, list) and len(center) >= 2:
        return np.array([float(center[0]), float(center[1])])
    return None


def axis_return_for(image_path, canny):
    image = cv2.imread(image_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    raw = detect_candidate_lines(gray, canny_threshold1=canny[0], canny_threshold2=canny[1])
    merged = merge_similar_lines(raw)
    with contextlib.redirect_stdout(io.StringIO()):
        x_axis, y_axis, filtered = infer_axes_from_lines(merged, (w, h), gray)
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "filtered_lines": len(filtered),
        "both_axes": x_axis is not None and y_axis is not None,
        "one_axis": (x_axis is not None) ^ (y_axis is not None),
        "no_axis": x_axis is None and y_axis is None,
        "image": image,
    }


def pixel_recall(pred_pixels, gt_pixels, tol=6):
    if not gt_pixels:
        return 0.0
    used = set()
    matches = 0
    for pred in pred_pixels:
        best_idx = None
        best_dist = float("inf")
        for idx, gt in enumerate(gt_pixels):
            if idx in used:
                continue
            dist = abs(float(pred) - float(gt))
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None and best_dist <= tol:
            used.add(best_idx)
            matches += 1
    return matches / len(gt_pixels)


def evaluate_cartesian(canny):
    rows = []
    for chart_type in CARTESIAN_TYPES:
        for path in base_pngs(chart_type):
            meta = read_json_for(path)
            result = axis_return_for(path, canny)
            if result is None:
                continue
            row = {
                "type": chart_type,
                "path": path,
                "both_axes": result["both_axes"],
                "one_axis": result["one_axis"],
                "no_axis": result["no_axis"],
                "x_tick_recall": 0.0,
                "y_tick_recall": 0.0,
                "pred_x_ticks": 0,
                "pred_y_ticks": 0,
                "gt_x_ticks": len(meta.get("x_pixels", [])),
                "gt_y_ticks": len(meta.get("y_pixels", [])),
            }
            if result["both_axes"]:
                with contextlib.redirect_stdout(io.StringIO()):
                    x_ticks = merge_similar_lines(
                        scan_pixels_for_ticks(result["image"], result["x_axis"], direction="x", scan_range=10),
                        angle_threshold=np.deg2rad(10),
                    )
                    y_ticks = merge_similar_lines(
                        scan_pixels_for_ticks(result["image"], result["y_axis"], direction="y", scan_range=10),
                        angle_threshold=np.deg2rad(10),
                    )
                x_pixels = sorted([(t[0] + t[2]) // 2 for t in x_ticks])
                y_pixels = sorted([(t[1] + t[3]) // 2 for t in y_ticks], reverse=True)
                row.update(
                    {
                        "x_tick_recall": pixel_recall(x_pixels, meta.get("x_pixels", [])),
                        "y_tick_recall": pixel_recall(y_pixels, meta.get("y_pixels", [])),
                        "pred_x_ticks": len(x_pixels),
                        "pred_y_ticks": len(y_pixels),
                    }
                )
            rows.append(row)
    return summarize_cartesian(rows)


def summarize_cartesian(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["type"]].append(row)
    groups["ALL"] = rows
    summary = {}
    for chart_type, items in groups.items():
        found = [r for r in items if r["both_axes"]]
        summary[chart_type] = {
            "n": len(items),
            "both_axes": sum(r["both_axes"] for r in items),
            "both_axes_rate": sum(r["both_axes"] for r in items) / len(items) if items else 0.0,
            "failure_rate": sum(not r["both_axes"] for r in items) / len(items) if items else 0.0,
            "one_axis_rate": sum(r["one_axis"] for r in items) / len(items) if items else 0.0,
            "no_axis_rate": sum(r["no_axis"] for r in items) / len(items) if items else 0.0,
            "mean_x_tick_recall": float(np.mean([r["x_tick_recall"] for r in found])) if found else 0.0,
            "mean_y_tick_recall": float(np.mean([r["y_tick_recall"] for r in found])) if found else 0.0,
            "mean_pred_x_ticks": float(np.mean([r["pred_x_ticks"] for r in found])) if found else 0.0,
            "mean_pred_y_ticks": float(np.mean([r["pred_y_ticks"] for r in found])) if found else 0.0,
        }
    return {"summary": summary, "rows": rows}


def hough_circles(image_path, chart_type, param2=30, gaussian_kernel=9, sigma=2):
    image = cv2.imread(image_path)
    if image is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gaussian_kernel > 1:
        gray = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), sigma)
    h, _ = gray.shape
    if chart_type == "radar":
        min_radius, max_radius = int(h / 5), int(h / 4)
    else:
        min_radius, max_radius = int(h / 4), int(h)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=20,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    return np.round(circles[0]).astype(float).tolist()


def evaluate_polar(param2=30, gaussian_kernel=9, sigma=2):
    rows = []
    for chart_type in POLAR_TYPES:
        for path in base_pngs(chart_type):
            meta = read_json_for(path)
            center = get_center(meta)
            gt_rs = [float(r) for r in meta.get("r_pixels", [])]
            circles = hough_circles(path, chart_type, param2=param2, gaussian_kernel=gaussian_kernel, sigma=sigma)
            row = {
                "type": chart_type,
                "path": path,
                "found": bool(circles),
                "center_error": None,
                "best_radius_error": None,
                "first_radius_error": None,
                "candidate_count": len(circles),
                "r_ticks": len(gt_rs),
            }
            if circles and center is not None:
                first = circles[0]
                center_errors = [float(np.linalg.norm(np.array(c[:2]) - center)) for c in circles]
                radius_errors = [
                    min(abs(float(c[2]) - gt_r) for gt_r in gt_rs) if gt_rs else None for c in circles
                ]
                row["center_error"] = min(center_errors)
                row["first_radius_error"] = radius_errors[0] if radius_errors else None
                row["best_radius_error"] = min([e for e in radius_errors if e is not None], default=None)
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[row["type"]].append(row)
    groups["ALL"] = rows
    summary = {}
    for chart_type, items in groups.items():
        found = [r for r in items if r["found"]]
        center_errors = [r["center_error"] for r in found if r["center_error"] is not None]
        first_r_errors = [r["first_radius_error"] for r in found if r["first_radius_error"] is not None]
        best_r_errors = [r["best_radius_error"] for r in found if r["best_radius_error"] is not None]
        summary[chart_type] = {
            "n": len(items),
            "circle_found_rate": len(found) / len(items) if items else 0.0,
            "failure_rate": 1.0 - len(found) / len(items) if items else 0.0,
            "median_center_error": median(center_errors) if center_errors else None,
            "median_first_radius_error": median(first_r_errors) if first_r_errors else None,
            "median_best_radius_error": median(best_r_errors) if best_r_errors else None,
            "mean_candidate_count": float(np.mean([r["candidate_count"] for r in found])) if found else 0.0,
        }
    return {"summary": summary, "rows": rows}


def low_level_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    resized = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 50, 150)
    feats = []
    feats.extend(cv2.calcHist([gray], [0], None, [16], [0, 256]).ravel() / gray.size)
    for channel, bins, hi in [(0, 12, 180), (1, 8, 256), (2, 8, 256)]:
        feats.extend(cv2.calcHist([hsv], [channel], None, [bins], [0, hi]).ravel() / gray.size)
    feats.append(float(edges.mean() / 255.0))
    for y in range(0, 160, 40):
        for x in range(0, 160, 40):
            patch = edges[y : y + 40, x : x + 40]
            feats.append(float(patch.mean() / 255.0))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
    horiz = vert = diag = long_lines = 0
    lengths = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            lengths.append(length)
            angle = abs(np.arctan2(dy, dx))
            if angle < np.deg2rad(10) or abs(angle - np.pi) < np.deg2rad(10):
                horiz += 1
            elif abs(angle - np.pi / 2) < np.deg2rad(10):
                vert += 1
            else:
                diag += 1
            if length > 80:
                long_lines += 1
    feats.extend([horiz, vert, diag, long_lines, len(lengths), np.mean(lengths) if lengths else 0.0])
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (9, 9), 2),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=25,
        minRadius=12,
        maxRadius=80,
    )
    feats.append(0 if circles is None else len(circles[0]))
    return np.asarray(feats, dtype=np.float32)


def evaluate_classification():
    paths, labels = [], []
    for chart_type in CLASSIFICATION_TYPES:
        for path in base_pngs(chart_type):
            paths.append(path)
            labels.append(chart_type)
    x = np.vstack([low_level_features(path) for path in paths])
    y = np.asarray(labels)
    clf = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced", n_jobs=-1),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    pred = cross_val_predict(clf, x, y, cv=cv, n_jobs=None)
    report = classification_report(y, pred, labels=CLASSIFICATION_TYPES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=CLASSIFICATION_TYPES)
    return {
        "n": len(paths),
        "accuracy": accuracy_score(y, pred),
        "labels": CLASSIFICATION_TYPES,
        "per_class_recall": {label: report[label]["recall"] for label in CLASSIFICATION_TYPES},
        "per_class_precision": {label: report[label]["precision"] for label in CLASSIFICATION_TYPES},
        "confusion_matrix": cm.tolist(),
        "errors": [
            {"path": path, "true": true, "pred": predicted}
            for path, true, predicted in zip(paths, y, pred)
            if true != predicted
        ],
    }


def evaluate_cartesian_cv_mllm(types, limit, cache_only, dataset_id):
    rows = []
    for chart_type in types:
        paths = base_pngs(chart_type)
        if limit > 0:
            paths = paths[:limit]
        for path in paths:
            rows.append(
                evaluate_cv_mllm_image(
                    path,
                    allow_api=not cache_only,
                    dataset_id=dataset_id,
                )
            )
    return {
        "dataset_id": dataset_id,
        "cache_only": cache_only,
        "types": types,
        "limit_per_type": limit,
        "summary": summarize_cv_mllm(rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "axis_prior_reviewer_eval.json"))
    parser.add_argument("--include-mllm", action="store_true", help="Include CV+MLLM tick/axis-type evaluation.")
    parser.add_argument("--cache-only", action="store_true", help="When including MLLM, use cache only and do not call APIs.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="MLLM cache namespace.")
    parser.add_argument("--mllm-types", nargs="+", default=MLLM_DEFAULT_TYPES)
    parser.add_argument("--mllm-limit", type=int, default=20, help="Per-type limit for MLLM evaluation. Use 0 for all.")
    args = parser.parse_args()

    canny_pairs = [(30, 90), (50, 150), (80, 200), (100, 250)]
    polar_param2 = [20, 30, 40, 50]

    result = {
        "dataset_root": CHART_ROOT,
        "chart_classification": evaluate_classification(),
        "cartesian_default": evaluate_cartesian((50, 150)),
        "cartesian_canny_sensitivity": {
            f"{a}/{b}": evaluate_cartesian((a, b))["summary"]["ALL"] for a, b in canny_pairs
        },
        "polar_default": evaluate_polar(param2=30, gaussian_kernel=9, sigma=2),
        "polar_param2_sensitivity": {
            str(param2): evaluate_polar(param2=param2, gaussian_kernel=9, sigma=2)["summary"]["ALL"]
            for param2 in polar_param2
        },
        "fixed_parameters": {
            "cartesian": {
                "canny_threshold1": 50,
                "canny_threshold2": 150,
                "hough_rho": 1,
                "hough_theta_degrees": 1,
                "hough_threshold": 20,
                "hough_min_line_length": 20,
                "hough_max_line_gap": 20,
                "axis_angle_tolerance_degrees": 10,
                "tick_scan_range": 10,
                "tick_min_run_pixels": 3,
                "tick_dark_threshold": 230,
            },
            "polar": {
                "gaussian_kernel": "9x9",
                "gaussian_sigma": 2,
                "hough_circles_dp": 1.2,
                "hough_circles_min_dist": 100,
                "hough_circles_param1": 20,
                "hough_circles_param2": 30,
                "radar_radius_range": "height/5 to height/4",
                "rose_radius_range": "height/4 to height",
            },
        },
    }
    if args.include_mllm:
        result["cartesian_cv_mllm"] = evaluate_cartesian_cv_mllm(
            types=args.mllm_types,
            limit=args.mllm_limit,
            cache_only=args.cache_only,
            dataset_id=args.dataset_id,
        )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    preview = {
        "classification": {
            "n": result["chart_classification"]["n"],
            "accuracy": result["chart_classification"]["accuracy"],
            "errors": len(result["chart_classification"]["errors"]),
        },
        "cartesian_default": result["cartesian_default"]["summary"]["ALL"],
        "polar_default": result["polar_default"]["summary"]["ALL"],
    }
    if args.include_mllm:
        preview["cartesian_cv_mllm"] = result["cartesian_cv_mllm"]["summary"].get("ALL", {})
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
