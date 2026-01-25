import json
import os
import numpy as np

# === 参数配置 ===
# PRED_DIR = "data/debug/detect_ticks/tick_data"
# GT_ROOT_DIR = "generated_charts"
PRED_DIR = "data/debug/detect_ticks/tick_data"
GT_ROOT_DIR = "generated_charts/v_bar"
TOLERANCE = 1
# ===================================

def match_ticks(detected, ground_truth, tolerance=1):
    detected = sorted(detected)
    gt = sorted(ground_truth)
    matched = []
    errors = []
    gt_used = [False] * len(gt)

    for d in detected:
        best_match = None
        best_error = float('inf')
        for i, g in enumerate(gt):
            if gt_used[i]:
                continue
            error = abs(d - g)
            if error <= tolerance and error < best_error:
                best_match = i
                best_error = error
        if best_match is not None:
            matched.append((d, gt[best_match]))
            errors.append(d - gt[best_match])
            gt_used[best_match] = True

    return len(matched), errors, len(detected), len(ground_truth)

def accumulate_metrics(matches, total_pred, total_gt, errors, accum):
    precision = matches / total_pred if total_pred else 0
    recall = matches / total_gt if total_gt else 0
    mse = np.mean(np.array(errors)**2) if errors else 0
    max_err = max(np.abs(errors)) if errors else 0
    avg_err = np.mean(np.abs(errors)) if errors else 0

    accum["precision"].append(precision)
    accum["recall"].append(recall)
    accum["mse"].append(mse)
    accum["max_err"].append(max_err)
    accum["avg_err"].append(avg_err)

def print_summary(name, accum):
    print(f"\n{name} ticks:")
    print(f"  Avg Precision: {np.mean(accum['precision']):.4f}")
    print(f"  Avg Recall:    {np.mean(accum['recall']):.4f}")
    print(f"  Avg MSE:       {np.mean(accum['mse']):.3f}")
    print(f"  Avg Max Error: {np.mean(accum['max_err']):.3f}")
    print(f"  Avg Abs Error: {np.mean(accum['avg_err']):.3f}")

def main():
    x_accum = {"precision": [], "recall": [], "mse": [], "max_err": [], "avg_err": []}
    y_accum = {"precision": [], "recall": [], "mse": [], "max_err": [], "avg_err": []}
    count = 0

    for fname in os.listdir(PRED_DIR):
        if not fname.endswith(".json"):
            continue
        pred_path = os.path.join(PRED_DIR, fname)

        # 从文件名中提取类型和编号
        parts = fname.split("_chart")
        if len(parts) != 2:
            print(f"[Skip] 命名格式非法: {fname}")
            continue
        chart_type = parts[0]
        index = parts[1].replace(".json", "")
        gt_path = os.path.join(GT_ROOT_DIR, chart_type, f"chart{index}.json")

        if not os.path.exists(gt_path):
            print(f"[Skip] 找不到 Ground Truth: {gt_path}")
            continue

        with open(pred_path, 'r') as f:
            pred = json.load(f)
        with open(gt_path, 'r') as f:
            gt = json.load(f)

        x_match, x_errors, x_total, x_gt_total = match_ticks(pred['x_pixels'], gt['x_pixels'], TOLERANCE)
        y_match, y_errors, y_total, y_gt_total = match_ticks(pred['y_pixels'], gt['y_pixels'], TOLERANCE)

        accumulate_metrics(x_match, x_total, x_gt_total, x_errors, x_accum)
        accumulate_metrics(y_match, y_total, y_gt_total, y_errors, y_accum)
        count += 1

    print(f"[Summary]")
    print(f"Total images evaluated: {count}")
    print_summary("X-axis", x_accum)
    print_summary("Y-axis", y_accum)

if __name__ == "__main__":
    main()
