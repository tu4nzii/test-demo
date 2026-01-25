import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def scan_pixels_for_ticks(image, axis_line, direction='x', scan_range=20, min_tick_height=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    tick_lines = []

    if direction == 'x':
        axis_y = (axis_line[1] + axis_line[3]) // 2
        # x_start = max(0, min(axis_line[0], axis_line[2]) - scan_range)
        # x_end   = min(w, max(axis_line[0], axis_line[2]) + scan_range)
        x_start = min(axis_line[0], axis_line[2])
        x_end   = max(axis_line[0], axis_line[2])
        for xx in range(x_start, x_end):
            tick_start = axis_y + 1
            tick_end = min(axis_y + scan_range, h - 1)
            black_run = []

            for y in range(tick_start, tick_end):
                if gray[y, xx] < 200:
                    black_run.append(y)
                else:
                    break
            if len(black_run) >= min_tick_height:
                tick_lines.append([xx, black_run[0], xx, black_run[-1]])
                print(f"[Debug][X-axis-from-origin] Tick at x={xx}, y=({black_run[0]}-{black_run[-1]})")

    elif direction == 'y':
        axis_x = (axis_line[0] + axis_line[2]) // 2
        # y_start = max(0, min(axis_line[1], axis_line[3]) - scan_range)
        # y_end   = min(h, max(axis_line[1], axis_line[3]) + scan_range)
        y_start = min(axis_line[1], axis_line[3])
        y_end   = max(axis_line[1], axis_line[3])

        for yy in range(y_start, y_end):
            black_run = []
            for x in range(axis_x - 1, max(axis_x - scan_range, 0), -1):
                if gray[yy, x] < 200:
                    black_run.append(x)
                else:
                    break
            if len(black_run) >= min_tick_height:
                tick_lines.append([black_run[-1], yy, black_run[0], yy])
                print(f"[Debug][Y-left] Tick at y={yy}, x=({black_run[-1]}-{black_run[0]})")

    return tick_lines


def main(idx=None):
    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
    from utils.image_io import load_image, save_image
    from function_calling.axis.detect_lines import detect_candidate_lines
    from function_calling.axis.merge_lines import merge_similar_lines
    from function_calling.axis.infer_axes import infer_axes_from_lines
    import os
    import shutil
    import json

    output_dir = DEBUG_OUTPUT_DIRS['detect_ticks']
    if CLEAR_OUTPUT_BEFORE_RUN.get('detect_ticks', False):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    img_items = list(IMG_PATHS.items())
    if idx is not None:
        indices = []
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, (list, tuple)):
            indices = list(idx)
        elif isinstance(idx, str):
            for part in idx.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-')
                    indices.extend(range(int(start), int(end)+1))
                else:
                    indices.append(int(part))
        indices = [i for i in indices if 0 <= i < len(img_items)]
        img_items = [img_items[i] for i in indices]

    for name, path in img_items:
        print(f"[Info] 处理: {name} => {path}")
        img = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        raw_lines = detect_candidate_lines(gray)
        merged_lines = merge_similar_lines(raw_lines)

        # ✅ 使用新版 infer_axes 接口（含 gray）
        x_axis, y_axis, _ = infer_axes_from_lines(merged_lines, (w, h), gray)
        if x_axis is None or y_axis is None:
            print(f"[Warning] {name} 未检测到 X/Y 轴，自动跳过。\n")
            continue

        x_ticks_raw = scan_pixels_for_ticks(img, x_axis, direction='x', scan_range=10)
        y_ticks_raw = scan_pixels_for_ticks(img, y_axis, direction='y', scan_range=10)

        x_ticks = merge_similar_lines(x_ticks_raw, angle_threshold=np.deg2rad(10))
        y_ticks = merge_similar_lines(y_ticks_raw, angle_threshold=np.deg2rad(10))

        print(f"X轴检测到刻度线: {len(x_ticks)} 条")
        print(f"Y轴检测到刻度线: {len(y_ticks)} 条")

        vis = img.copy()
        cv2.line(vis, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0,0,255), 3)
        cv2.line(vis, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255,0,0), 3)
        for x1,y1,x2,y2 in x_ticks:
            cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        for x1,y1,x2,y2 in y_ticks:
            cv2.line(vis, (x1,y1), (x2,y2), (255,255,0), 2)

        save_image(vis, os.path.join(output_dir, f"result_{name}.png"))
        print(f"[Debug] 结果保存至: result_{name}.png")

        tick_output_dir = os.path.join(output_dir, "tick_data")
        os.makedirs(tick_output_dir, exist_ok=True)
        # 提取 tick 中心坐标值
        x_tick_positions = sorted(list(set([int(round((x1 + x2) / 2)) for x1, y1, x2, y2 in x_ticks])))
        y_tick_positions = sorted(list(set([int(round((y1 + y2) / 2)) for x1, y1, x2, y2 in y_ticks])), reverse=True)


        tick_data = {
            "image_name": name,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "x_pixels": x_tick_positions,
            "y_pixels": y_tick_positions
        }

        with open(os.path.join(tick_output_dir, f"{name}.json"), 'w', encoding='utf-8') as f:
            json.dump(tick_data, f, indent=2)
        print(f"[Export] Tick 数据已保存至: tick_data/{name}.json")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, default=None, help='图片索引（单个、逗号、范围）')
    args = parser.parse_args()
    main(idx=args.idx)
