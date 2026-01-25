import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def scan_pixels_for_ticks(image, axis_line, direction='x', scan_range=20, min_tick_height=3, max_tick_height=20, max_distance_to_axis=10):
    """
    基于像素扫描沿坐标轴检测短横线刻度，简化为直接扩展坐标轴范围后扫描
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    tick_lines = []
    
    if direction == 'x':
        # 计算x轴的y坐标（轴线中点y值）
        axis_y = (axis_line[1] + axis_line[3]) // 2
        
        # 扩展x轴范围：左右各加scan_range，同时限制在图像边界内
        x_start = min(axis_line[0], axis_line[2]) - scan_range
        x_end = max(axis_line[0], axis_line[2]) + scan_range
        x_start = max(0, x_start)  # 不超出左边界
        x_end = min(w, x_end)      # 不超出右边界
        
        # 直接在扩展后的范围内逐x扫描
        for xx in range(x_start, x_end):
            # 获取当前列的像素
            column = gray[:, xx]
            run_start = None  # 记录连续暗像素的起始位置
            
            for y in range(h):
                if column[y] < 200:  # 暗像素（可能是刻度线）
                    if run_start is None:
                        run_start = y
                else:  # 亮像素（结束连续暗像素）
                    if run_start is not None:
                        run_end = y - 1
                        # 检查这段连续暗像素是否符合刻度条件
                        for center in range(run_start + min_tick_height//2, run_end - min_tick_height//2 + 1):
                            if abs(center - axis_y) <= max_distance_to_axis:
                                # 计算刻度的起止点
                                tick_start = max(run_start, center - min_tick_height//2)
                                tick_end = min(run_end, center + min_tick_height//2)
                                if tick_end - tick_start + 1 >= min_tick_height:
                                    tick_lines.append([xx, tick_start, xx, tick_end])
                                break
                        run_start = None  # 重置连续暗像素记录
            
            # 处理图像底部的连续暗像素（边界情况）
            if run_start is not None:
                run_end = h - 1
                for center in range(run_start + min_tick_height//2, run_end - min_tick_height//2 + 1):
                    if abs(center - axis_y) <= max_distance_to_axis:
                        tick_start = max(run_start, center - min_tick_height//2)
                        tick_end = min(run_end, center + min_tick_height//2)
                        if tick_end - tick_start + 1 >= min_tick_height:
                            tick_lines.append([xx, tick_start, xx, tick_end])
                        break
    
    elif direction == 'y':
        # 计算y轴的x坐标（轴线中点x值）
        axis_x = (axis_line[0] + axis_line[2]) // 2
        
        # 扩展y轴范围：上下各加scan_range，限制在图像边界内
        y_start = min(axis_line[1], axis_line[3]) - scan_range
        y_end = max(axis_line[1], axis_line[3]) + scan_range
        y_start = max(0, y_start)  # 不超出上边界
        y_end = min(h, y_end)      # 不超出下边界
        
        # 直接在扩展后的范围内逐y扫描
        for yy in range(y_start, y_end):
            # 获取当前行的像素
            row = gray[yy, :]
            run_start = None  # 记录连续暗像素的起始位置
            
            for x in range(w):
                if row[x] < 200:  # 暗像素（可能是刻度线）
                    if run_start is None:
                        run_start = x
                else:  # 亮像素（结束连续暗像素）
                    if run_start is not None:
                        run_end = x - 1
                        # 检查这段连续暗像素是否符合刻度条件
                        for center in range(run_start + min_tick_height//2, run_end - min_tick_height//2 + 1):
                            if abs(center - axis_x) <= max_distance_to_axis:
                                # 计算刻度的起止点
                                tick_start = max(run_start, center - min_tick_height//2)
                                tick_end = min(run_end, center + min_tick_height//2)
                                if tick_end - tick_start + 1 >= min_tick_height:
                                    tick_lines.append([tick_start, yy, tick_end, yy])
                                break
                        run_start = None  # 重置连续暗像素记录
            
            # 处理图像右侧的连续暗像素（边界情况）
            if run_start is not None:
                run_end = w - 1
                for center in range(run_start + min_tick_height//2, run_end - min_tick_height//2 + 1):
                    if abs(center - axis_x) <= max_distance_to_axis:
                        tick_start = max(run_start, center - min_tick_height//2)
                        tick_end = min(run_end, center + min_tick_height//2)
                        if tick_end - tick_start + 1 >= min_tick_height:
                            tick_lines.append([tick_start, yy, tick_end, yy])
                        break

    return tick_lines

# 🧪 测试示例
def main(idx=None):
    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
    from utils.image_io import load_image, save_image
    from function_calling.axis.detect_lines import detect_candidate_lines
    from function_calling.axis.merge_lines import merge_similar_lines
    from function_calling.axis.infer_axes import infer_axes_from_lines
    import os
    import shutil

    output_dir = DEBUG_OUTPUT_DIRS['detect_ticks']
    # 配置：是否清空输出目录
    if CLEAR_OUTPUT_BEFORE_RUN.get('detect_ticks', False):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    img_items = list(IMG_PATHS.items())
    # 处理idx参数，支持int、list、tuple、str
    if idx is not None:
        indices = []
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, (list, tuple)):
            indices = list(idx)
        elif isinstance(idx, str):
            # 支持逗号分隔、范围
            for part in idx.split(','):
                part = part.strip()
                if '-' in part:
                    start, end = part.split('-')
                    indices.extend(range(int(start), int(end)+1))
                else:
                    indices.append(int(part))
        # 过滤越界
        indices = [i for i in indices if 0 <= i < len(img_items)]
        img_items = [img_items[i] for i in indices]
    for name, path in img_items:
        img = load_image(path)
        print(f"[Info] 处理: {name} => {path}")
        h, w = img.shape[:2]

        # 获取 X/Y 轴
        raw_lines = detect_candidate_lines(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        merged_lines = merge_similar_lines(raw_lines)
        x_axis, y_axis = infer_axes_from_lines(merged_lines, (w, h))
        if x_axis is None or y_axis is None:
            print(f"[Warning] {name} 未检测到 X/Y 轴，自动跳过。\n")
            continue
        
        # 检测原始刻度线
        x_ticks_raw = scan_pixels_for_ticks(img, x_axis, direction='x', scan_range=10)
        y_ticks_raw = scan_pixels_for_ticks(img, y_axis, direction='y', scan_range=10)

        # 合并刻度线
        x_ticks = merge_similar_lines(x_ticks_raw, angle_threshold=np.deg2rad(10))
        y_ticks = merge_similar_lines(y_ticks_raw, angle_threshold=np.deg2rad(10))

        # # 调试
        # x_ticks = x_ticks_raw
        # y_ticks = y_ticks_raw

        print(f"X轴检测到刻度线: {len(x_ticks)} 条")
        print(f"Y轴检测到刻度线: {len(y_ticks)} 条")

        vis = img.copy()
        cv2.line(vis, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0,0,255), 3)
        cv2.line(vis, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255,0,0), 3)
        for x1,y1,x2,y2 in x_ticks:
            cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        for x1,y1,x2,y2 in y_ticks:
            cv2.line(vis, (x1,y1), (x2,y2), (255,255,0), 2)
        out_path = os.path.join(output_dir, f"result_{name}.png")
        save_image(vis, out_path)
        print(f"[Debug] 结果保存至: {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=str, default=None, help='图片索引（单个、逗号、范围）')
    args = parser.parse_args()
    main(idx=args.idx)