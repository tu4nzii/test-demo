# function_calling/recognize_tick_labels.py

import cv2
import pytesseract
import numpy as np

def recognize_tick_labels(image, tick_lines, direction='x', label_region=30):
    """
    识别 tick 附近的刻度值文本
    - image: 原始图像（BGR）
    - tick_lines: 已检测的 tick 线段列表 [[x1, y1, x2, y2], ...]
    - direction: 'x' 表示检测 X轴下方文本，'y' 表示 Y轴左侧文本
    - label_region: tick 线延伸方向上多大的区域用于找文本
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []

    for tick in tick_lines:
        x1, y1, x2, y2 = tick
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if direction == 'x':
            # 扫描区域：tick 下方一个矩形
            roi = gray[center_y + 2 : center_y + 2 + label_region, center_x - 15 : center_x + 15]
        else:
            # 扫描区域：tick 左边一个矩形
            roi = gray[center_y - 10 : center_y + 10, max(center_x - label_region, 0): center_x - 2]

        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            continue

        # 二值化 + OCR
        _, roi_bin = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
        config = r'--psm 7 -c tessedit_char_whitelist=0123456789.-'
        text = pytesseract.image_to_string(roi_bin, config=config).strip()

        if text:  # 非空结果
            results.append({
                'tick': tick,
                'text': text
            })

    return results


# 🧪 示例用法
def main():
    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
    from utils.image_io import load_image, save_image
    from function_calling.axis.detect_lines import detect_candidate_lines
    from function_calling.axis.merge_lines import merge_similar_lines
    from function_calling.axis.infer_axes import infer_axes_from_lines
    from function_calling.ticks.detect_ticks import scan_pixels_for_ticks
    from function_calling.ticks.filter_ticks import filter_ticks
    import os
    import shutil

    output_dir = DEBUG_OUTPUT_DIRS['recognize_tick_labels']
    # 配置：是否清空输出目录
    if CLEAR_OUTPUT_BEFORE_RUN.get('recognize_tick_labels', False):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for name, path in IMG_PATHS.items():
        img = load_image(path)
        print(f"[Info] 处理: {name} => {path}")
        h, w = img.shape[:2]

        # 获取坐标轴
        raw_lines = detect_candidate_lines(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        merged_lines = merge_similar_lines(raw_lines)
        x_axis, y_axis = infer_axes_from_lines(merged_lines, (w, h))
        if x_axis is None or y_axis is None:
            print("未检测到 X/Y 轴")
            continue

        # 原始tick + 合并
        x_raw = scan_pixels_for_ticks(img, x_axis, direction='x', scan_range=10)
        x_merged = merge_similar_lines(x_raw, angle_threshold=np.deg2rad(10))
        x_ticks = filter_ticks(x_merged, direction='x')

        y_raw = scan_pixels_for_ticks(img, y_axis, direction='y', scan_range=10)
        y_merged = merge_similar_lines(y_raw, angle_threshold=np.deg2rad(10))
        y_ticks = filter_ticks(y_merged, direction='y')


        # 识别文字
        x_labels = recognize_tick_labels(img, x_ticks, direction='x')
        y_labels = recognize_tick_labels(img, y_ticks, direction='y')

        print("X轴刻度值：")
        for item in x_labels:
            print(item)

        print("Y轴刻度值：")
        for item in y_labels:
            print(item)

        # 可视化查看效果
        vis = img.copy()
        for item in x_labels + y_labels:
            x1, y1, x2, y2 = item['tick']
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, item['text'], (x2 + 5, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        out_path = os.path.join(output_dir, f"result_{name}.png")
        save_image(vis, out_path)
        print(f"[Debug] 结果保存至: {out_path}")

if __name__ == '__main__':
    main()
