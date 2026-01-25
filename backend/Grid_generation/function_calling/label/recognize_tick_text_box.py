import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import easyocr
import json
from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
from function_calling.axis.detect_lines import detect_candidate_lines
from function_calling.axis.merge_lines import merge_similar_lines
from function_calling.axis.infer_axes import infer_axes_from_lines
import numpy as np
import re

reader = easyocr.Reader(['en'])

def get_axis_roi(image_shape, axis_line, direction='x', roi_thickness=40):
    h, w = image_shape[:2]
    if direction == 'x':
        x_start = min(axis_line[0], axis_line[2])
        x_end = max(axis_line[0], axis_line[2])
        y = (axis_line[1] + axis_line[3]) // 2
        y1 = max(0, y)
        y2 = min(h, y + roi_thickness)
        return (x_start, y1, x_end, y2)
    elif direction == 'y':
        y_start = min(axis_line[1], axis_line[3])
        y_end = max(axis_line[1], axis_line[3])
        x = (axis_line[0] + axis_line[2]) // 2
        x1 = max(0, x - roi_thickness)
        x2 = min(w, x)
        return (x1, y_start, x2, y_end)
    else:
        raise ValueError("Direction must be 'x' or 'y'")

def detect_tick_text_in_roi(image, roi_rect, score_threshold=0.3):
    x1, y1, x2, y2 = roi_rect
    roi_img = image[y1:y2, x1:x2]
    h, w = roi_img.shape[:2]

    # 动态调整放大倍数
    if max(h, w) < 100:
        scale_factor = 3
    elif max(h, w) < 200:
        scale_factor = 2
    else:
        scale_factor = 1.5

    roi_resized = cv2.resize(roi_img, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)

    # 灰度 & 二值化 & 锐化增强
    gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(binary, -1, kernel)

    results = reader.readtext(sharpened)
    boxes = []
    for bbox, text, conf in results:
        if conf < score_threshold:
            continue
        if not text.strip() or not re.search(r'\d', text):
            continue
        xs = [int(p[0] / scale_factor) for p in bbox]
        ys = [int(p[1] / scale_factor) for p in bbox]
        bx1, by1 = min(xs) + x1, min(ys) + y1
        bx2, by2 = max(xs) + x1, max(ys) + y1
        boxes.append({
            "box": [bx1, by1, bx2, by2],
            "text": text
        })
    return boxes

def process_image_for_tick_labels(image_path, name, output_dir, roi_thickness=40):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    raw_lines = detect_candidate_lines(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    merged = merge_similar_lines(raw_lines)
    x_axis, y_axis = infer_axes_from_lines(merged, (w, h))
    if x_axis is None or y_axis is None:
        print(f"[Warning] {name} 未检测到坐标轴，跳过")
        return []
    x_roi = get_axis_roi(img.shape, x_axis, 'x', roi_thickness)
    y_roi = get_axis_roi(img.shape, y_axis, 'y', roi_thickness)
    x_labels = detect_tick_text_in_roi(img, x_roi)
    y_labels = detect_tick_text_in_roi(img, y_roi)
    all_labels = x_labels + y_labels
    vis = img.copy()
    for item in all_labels:
        x1, y1, x2, y2 = item["box"]
        text = item["text"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    save_path = os.path.join(output_dir, f"{name}_tick_labels.png")
    cv2.imwrite(save_path, vis)
    print(f"[✔] {name} 检测到 tick labels: {len(all_labels)}，结果保存至: {save_path}")
    return all_labels

def main():
    output_dir = DEBUG_OUTPUT_DIRS['recognize_tick_labels']
    if CLEAR_OUTPUT_BEFORE_RUN.get('recognize_tick_labels', False):
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    for name, path in IMG_PATHS.items():
        print(f"[Info] 正在处理: {name}")
        result = process_image_for_tick_labels(path, name, output_dir)
        all_results[name] = result
    json_path = os.path.join(output_dir, "all_tick_label_boxes.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] 所有 tick label 包围盒已保存至: {json_path}")

if __name__ == '__main__':
    main()
