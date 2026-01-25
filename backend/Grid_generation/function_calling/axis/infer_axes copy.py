# 输入：合并后的线段列表（[x1, y1, x2, y2]），图像尺寸（w,h）；

# 逻辑：
# 1️⃣ 两线段 相交（且交点在图像范围内）；
# 2️⃣ 两线段方向 接近垂直（夹角约90°）；
# 3️⃣ 两线段长度均超过图像宽/高的1/2；
# 4️⃣ 优先返回符合条件的第一组线段对作为 (X轴, Y轴)。

import numpy as np
import math

def line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot(x2 - x1, y2 - y1)

def line_angle(line):
    x1, y1, x2, y2 = line
    return np.arctan2(y2 - y1, x2 - x1)

def lines_intersect(line1, line2):
    """判断两线段是否相交"""
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    A, B = (line1[0], line1[1]), (line1[2], line1[3])
    C, D = (line2[0], line2[1]), (line2[2], line2[3])
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def intersection_point(line1, line2):
    """计算两线段交点"""
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return x, y

def infer_axes_from_lines(lines, image_size, angle_tolerance=np.deg2rad(10)):
    w, h = image_size
    min_len_x = w * 0.5
    min_len_y = h * 0.5

    candidates = []

    for i, line1 in enumerate(lines):
        for j, line2 in enumerate(lines):
            if i >= j:
                continue
            if not lines_intersect(line1, line2):
                continue
            angle1 = line_angle(line1)
            angle2 = line_angle(line2)
            angle_diff = abs(abs(angle1 - angle2) - np.pi/2)
            if angle_diff > angle_tolerance:
                continue
            len1 = line_length(line1)
            len2 = line_length(line2)
            if len1 < min_len_x or len2 < min_len_y:
                continue
            inter = intersection_point(line1, line2)
            if inter is None or not (0 <= inter[0] <= w and 0 <= inter[1] <= h):
                continue
            candidates.append((line1, line2, inter))

    if not candidates:
        print("[Warning] 未找到符合条件的 X/Y 轴线段对")
        return None, None
    
    print(f"坐标轴候选结果：\n{candidates}\n")
    
    # 在所有候选中优先选择y轴为最靠左的竖线
    best = None
    min_x = float('inf')
    for l1, l2, inter in candidates:
        # 判断哪条是水平线，哪条是竖线
        if abs(line_angle(l1)) < angle_tolerance:
            x_axis, y_axis = l1, l2
        else:
            x_axis, y_axis = l2, l1
        # y轴竖线的最小x坐标
        y_xs = [y_axis[0], y_axis[2]]
        y_minx = min(y_xs)
        if y_minx < min_x:
            min_x = y_minx
            best = (x_axis, y_axis)
    x_axis, y_axis = best
    # 对x_axis和y_axis分别延长5像素
    def extend_line(line, pixels=5):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return [x1, y1, x2, y2]
        ex = dx / length * pixels
        ey = dy / length * pixels
        return [int(round(x1 - ex)), int(round(y1 - ey)), int(round(x2 + ex)), int(round(y2 + ey))]
    x_axis = extend_line(x_axis, 5)
    y_axis = extend_line(y_axis, 5)
    return x_axis, y_axis

# 🧪 内置测试
def main():
    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
    from utils.image_io import load_image, save_image
    import os
    import cv2
    from function_calling.axis.detect_lines import detect_candidate_lines
    from function_calling.axis.merge_lines import merge_similar_lines

    output_dir = DEBUG_OUTPUT_DIRS['infer_axes']
    import shutil
    # 配置：是否清空输出目录
    if CLEAR_OUTPUT_BEFORE_RUN.get('infer_axes', False):
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
        print(f"[Debug] 检测到 {len(merged_lines)} 条候选线段")
        x_axis, y_axis = infer_axes_from_lines(merged_lines, (w, h))
        if x_axis is None or y_axis is None:
            print(f"[Warning] {name} 未检测到 X/Y 轴，自动跳过。\n")
            continue

        print(f"检测结果：\nX轴: {x_axis}\nY轴: {y_axis}")

        vis = img.copy()
        cv2.line(vis, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 3)
        cv2.line(vis, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255, 0, 0), 3)
        out_path = os.path.join(output_dir, f"result_{name}.png")
        save_image(vis, out_path)
        print(f"[Debug] 结果保存至: {out_path}")

if __name__ == '__main__':
    main()
