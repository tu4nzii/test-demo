import os
import json
import cv2
import numpy as np
from config import IMG_PATHS, DEBUG_OUTPUT_DIRS
from utils.image_io import load_image, save_image
from function_calling.axis.detect_lines import detect_candidate_lines
from function_calling.axis.merge_lines import merge_similar_lines
from function_calling.axis.infer_axes import infer_axes_from_lines

def extract_axis_rectangle(image: np.ndarray, axis_line: list[int], direction: str, thickness_threshold: int = 30) -> list[int]:
    """
    基于轴线中心，沿正交方向扫描暗像素带，估计轴线粗细。
    返回矩形框 [x1, y1, x2, y2]。
    direction: 'x' 表示横向 X 轴，'y' 表示纵向 Y 轴。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x1, y1, x2, y2 = axis_line

    # 计算中点
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if direction == 'x':
        profile = gray[:, cx]
        dark_mask = profile < 200
        y_indices = np.where(dark_mask)[0]
        y_start, y_end = (cy-1, cy+1) if len(y_indices) == 0 else (max(0, int(y_indices[0])), min(h-1, int(y_indices[-1])))
        return [int(x1), int(y_start), int(x2), int(y_end)]

    elif direction == 'y':
        profile = gray[cy, :]
        dark_mask = profile < 200
        x_indices = np.where(dark_mask)[0]
        x_start, x_end = (cx-1, cx+1) if len(x_indices) == 0 else (max(0, int(x_indices[0])), min(w-1, int(x_indices[-1])))
        return [int(x_start), int(y1), int(x_end), int(y2)]

    else:
        raise ValueError("Direction must be 'x' or 'y'")

def export_axis_rectangles():
    output_dir = DEBUG_OUTPUT_DIRS['detect_ticks']
    json_out_dir = os.path.join(output_dir, "axis_boxes")
    os.makedirs(json_out_dir, exist_ok=True)

    for name, path in IMG_PATHS.items():
        image = load_image(path)
        h, w = image.shape[:2]
        print(f"[Scan] 正在处理 {name}")

        lines = detect_candidate_lines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        merged = merge_similar_lines(lines)
        x_axis, y_axis = infer_axes_from_lines(merged, (w, h))

        if x_axis is None or y_axis is None:
            print(f"[Warning] 未能识别坐标轴: {name}")
            continue

        x_box = extract_axis_rectangle(image, x_axis, direction='x')
        y_box = extract_axis_rectangle(image, y_axis, direction='y')

        out_json = {
            "image_name": name,
            "x_axis_box": list(map(int, x_box)),
            "y_axis_box": list(map(int, y_box))
        }

        out_path = os.path.join(json_out_dir, f"{name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_json, f, indent=2)
        print(f"[Export] 导出至 {out_path}")

        # Debug 可视化输出
        vis = image.copy()
        cv2.rectangle(vis, (x_box[0], x_box[1]), (x_box[2], x_box[3]), (0, 0, 255), 2)  # 红色 X 轴框
        cv2.rectangle(vis, (y_box[0], y_box[1]), (y_box[2], y_box[3]), (255, 0, 0), 2)  # 蓝色 Y 轴框
        out_img_path = os.path.join(json_out_dir, f"vis_{name}.png")
        save_image(vis, out_img_path)
        print(f"[Debug] 可视化保存至 {out_img_path}")

if __name__ == '__main__':
    export_axis_rectangles()
