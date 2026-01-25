# 1.手动数值调试模式，可直接编辑和测试线段合并算法的数值样例
# python -m function_calling.merge_lines --mode manual  
# 2.(默认)批量处理图片库，输出合并后线段的可视化结果到对应文件夹
# python -m function_calling.merge_lines --mode image

import numpy as np

def calculate_angle(line):
    x1, y1, x2, y2 = line
    return np.arctan2(y2 - y1, x2 - x1)

def angle_to_axis(angle):
    """
    返回角度与水平(0)和垂直(pi/2)的最小夹角
    """
    angle = abs(angle)
    angle = angle % np.pi
    return min(abs(angle), abs(np.pi/2 - angle))

def is_projection_close(line1, line2, proj_threshold=20):
    """
    判断两条线段在主方向上的投影是否重叠或接近
    """
    angle1 = abs(calculate_angle(line1)) % np.pi
    # 判断主方向
    if angle1 < np.deg2rad(45) or angle1 > np.deg2rad(135):  # 水平为主
        x1_min, x1_max = sorted([line1[0], line1[2]])
        x2_min, x2_max = sorted([line2[0], line2[2]])
        # 投影重叠或接近
        return not (x1_max + proj_threshold < x2_min or x2_max + proj_threshold < x1_min)
    else:  # 垂直为主
        y1_min, y1_max = sorted([line1[1], line1[3]])
        y2_min, y2_max = sorted([line2[1], line2[3]])
        return not (y1_max + proj_threshold < y2_min or y2_max + proj_threshold < y1_min)

def merge_group_lines(group):
    """
    合并一组线段，分为两种情况：
    1. 若有一条线段在主方向上远长于其他线段，则直接取最长的那条
    2. 否则，合并为覆盖整个区间的线段，主方向坐标取最小和最大，正交方向取中值
    """
    if len(group) == 1:
        return group[0]
    # 判断主方向
    angles = [abs(calculate_angle(l)) % np.pi for l in group]
    mean_angle = np.mean(angles)
    is_horizontal = mean_angle < np.deg2rad(45) or mean_angle > np.deg2rad(135)
    if is_horizontal:
        xs = []
        ys = []
        x_lengths = []
        for l in group:
            xs.extend([l[0], l[2]])
            ys.extend([l[1], l[3]])
            x_lengths.append(abs(l[2] - l[0]))
        max_len = max(x_lengths)
        if max_len > 1.5 * np.median(x_lengths):
            idx = x_lengths.index(max_len)
            return group[idx]
        x_min, x_max = min(xs), max(xs)
        y_med = int(np.ceil(np.median(ys)))
        return [x_min, y_med, x_max, y_med]
    else:
        xs = []
        ys = []
        y_lengths = []
        for l in group:
            xs.extend([l[0], l[2]])
            ys.extend([l[1], l[3]])
            y_lengths.append(abs(l[3] - l[1]))
        max_len = max(y_lengths)
        if max_len > 1.5 * np.median(y_lengths):
            idx = y_lengths.index(max_len)
            return group[idx]
        y_min, y_max = min(ys), max(ys)
        x_med = int(np.ceil(np.median(xs)))
        return [x_med, y_min, x_med, y_max]

def merge_similar_lines(lines, angle_threshold=np.deg2rad(10), proj_threshold=10, center_threshold=5):
    """
    合并接近平行且相邻的线段，主方向投影需重叠，中心点垂直方向距离需在阈值内。
    proj_threshold: 主方向投影重叠的阈值
    center_threshold: 正交方向中心点距离的阈值
    """
    merged_lines = []
    used = [False] * len(lines)

    for i, line_i in enumerate(lines):
        if used[i]:
            continue
        group = [line_i]
        angle_i = calculate_angle(line_i)
        for j, line_j in enumerate(lines):
            if i == j or used[j]:
                continue
            angle_j = calculate_angle(line_j)
            if abs(angle_i - angle_j) < angle_threshold:
                if is_projection_close(line_i, line_j, proj_threshold):
                    # 判断正交方向中心点距离
                    angle1 = abs(angle_i) % np.pi
                    center_i = np.mean([[line_i[0], line_i[1]], [line_i[2], line_i[3]]], axis=0)
                    center_j = np.mean([[line_j[0], line_j[1]], [line_j[2], line_j[3]]], axis=0)
                    if angle1 < np.deg2rad(45) or angle1 > np.deg2rad(135):  # 水平为主
                        # y方向中心点距离
                        if abs(center_i[1] - center_j[1]) < center_threshold:
                            group.append(line_j)
                            used[j] = True
                    else:  # 垂直为主
                        # x方向中心点距离
                        if abs(center_i[0] - center_j[0]) < center_threshold:
                            group.append(line_j)
                            used[j] = True
        merged_lines.append(merge_group_lines(group))
        used[i] = True
    # 递归/循环直到收敛
    if len(merged_lines) < len(lines):
        return merge_similar_lines(merged_lines, angle_threshold, proj_threshold, center_threshold)
    else:
        return merged_lines

# 🧪 内置测试
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='merge_lines 调试模式')
    parser.add_argument('--mode', choices=['image', 'manual'], default='image', help='选择调试模式：image（处理图片）或 manual（手动输入线段）')
    args = parser.parse_args()

    if args.mode == 'manual':
        # 手动数值调试样例
        print('【手动数值调试模式】')
        # 示例线段，可自行修改
        lines = [
            [10, 20, 100, 20],
            [105, 22, 200, 21],
            [15, 25, 90, 25],
            [300, 50, 400, 50],
            [305, 52, 390, 51],
            [50, 100, 50, 200],
            [52, 105, 52, 190],
        ]
        merged = merge_similar_lines(lines)
        print('合并前：')
        for l in lines:
            print(l)
        print('合并后：')
        for l in merged:
            print(l)
    else:
        # 图片批量处理模式
        import cv2
        from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
        from utils.image_io import load_image, save_image
        from function_calling.axis.detect_lines import detect_candidate_lines
        from function_calling.axis.infer_axes import infer_axes_from_lines
        import os
        import shutil

        output_dir = DEBUG_OUTPUT_DIRS['merge_lines']
        # 配置：是否清空输出目录
        if CLEAR_OUTPUT_BEFORE_RUN.get('merge_lines', False):
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        for name, path in IMG_PATHS.items():
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

            # 可视化合并后的线段
            vis = img.copy()
            for x1, y1, x2, y2 in merged_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.line(vis, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0,0,255), 2)
            cv2.line(vis, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255,0,0), 2)
            out_path = os.path.join(output_dir, f"result_{name}.png")
            save_image(vis, out_path)
            print(f"[Debug] 合并后线段图像已保存至: {out_path}")
