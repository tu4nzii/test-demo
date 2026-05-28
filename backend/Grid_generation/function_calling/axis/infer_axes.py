# 适配本数据集

import numpy as np
import math
import cv2

def line_length(line):
    x1, y1, x2, y2 = line
    return np.hypot(x2 - x1, y2 - y1)

def line_angle(line):
    x1, y1, x2, y2 = line
    return np.arctan2(y2 - y1, x2 - x1)

def infer_axes_from_lines(lines, image_size, image_gray, angle_tolerance=np.deg2rad(10)):
    w, h = image_size

    def is_horizontal_or_vertical(line, angle_thresh=angle_tolerance):
        angle = abs(line_angle(line))
        return angle < angle_thresh or abs(angle - np.pi / 2) < angle_thresh

    def recover_longest_subsegment(line, image_gray, intensity_thresh=125, gap_tol=2):
        x1, y1, x2, y2 = line
        length = int(np.hypot(x2 - x1, y2 - y1))
        xs = np.linspace(x1, x2, length).astype(int)
        ys = np.linspace(y1, y2, length).astype(int)
        black_ranges, current = [], []
        for x, y in zip(xs, ys):
            if 0 <= x < image_gray.shape[1] and 0 <= y < image_gray.shape[0]:
                if image_gray[y, x] < intensity_thresh:
                    current.append((x, y))
                else:
                    if len(current) > gap_tol:
                        black_ranges.append(current)
                    current = []
        if len(current) > gap_tol:
            black_ranges.append(current)
        if not black_ranges:
            return line
        longest = max(black_ranges, key=len)
        recovered = [int(longest[0][0]), int(longest[0][1]), int(longest[-1][0]), int(longest[-1][1])]
        # Some exported charts render axes in anti-aliased gray rather than
        # near-black. In that case this pixel scan can shrink a valid long
        # Hough axis to a tiny dark fragment, so keep the original candidate.
        if line_length(recovered) < max(30, line_length(line) * 0.4):
            return [int(x1), int(y1), int(x2), int(y2)]
        return recovered

    def extend_line(line, pixels=5):
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return [x1, y1, x2, y2]
        ex, ey = dx / length * pixels, dy / length * pixels
        return [int(round(x1 - ex)), int(round(y1 - ey)), int(round(x2 + ex)), int(round(y2 + ey))]

    def align_x_axis_to_y_axis(x_axis, y_axis):
        h_x_min, h_x_max, h_y = horizontal_geom(x_axis)
        v_x, v_y_min, v_y_max = vertical_geom(y_axis)

        if h_x_min <= v_x <= h_x_max:
            if abs(x_axis[0] - v_x) <= abs(x_axis[2] - v_x):
                x_axis = [v_x, h_y, x_axis[2], h_y]
            else:
                x_axis = [x_axis[0], h_y, v_x, h_y]

        return x_axis, y_axis

    def horizontal_geom(line):
        x_min, x_max = sorted([line[0], line[2]])
        y = int(round((line[1] + line[3]) / 2))
        return x_min, x_max, y

    def vertical_geom(line):
        y_min, y_max = sorted([line[1], line[3]])
        x = int(round((line[0] + line[2]) / 2))
        return x, y_min, y_max

    def choose_axes_by_pair_score(filtered_lines):
        horizontal_candidates = []
        vertical_candidates = []
        for line in filtered_lines:
            angle = abs(line_angle(line))
            length = line_length(line)
            if angle < angle_tolerance and length >= w * 0.5:
                horizontal_candidates.append(line)
            elif abs(angle - np.pi / 2) < angle_tolerance and length >= h * 0.5:
                vertical_candidates.append(line)

        if not horizontal_candidates or not vertical_candidates:
            return None, None

        best_pair = None
        best_score = float("inf")
        max_corner_gap = max(30, 0.08 * max(w, h))
        max_bottom_gap = max(35, 0.12 * h)

        for h_line in horizontal_candidates:
            h_x_min, h_x_max, h_y = horizontal_geom(h_line)
            h_len = max(1, h_x_max - h_x_min)
            for v_line in vertical_candidates:
                v_x, v_y_min, v_y_max = vertical_geom(v_line)
                v_len = max(1, v_y_max - v_y_min)

                bottom_gap = abs(h_y - v_y_max)
                if bottom_gap > max_bottom_gap:
                    continue

                if h_x_min <= v_x <= h_x_max:
                    corner_gap = 0
                else:
                    corner_gap = min(abs(v_x - h_x_min), abs(v_x - h_x_max))
                if corner_gap > max_corner_gap:
                    continue

                left_gap = abs(v_x - h_x_min)
                # Prefer a real chart corner: the x-axis begins near the y-axis
                # bottom, not at a lower legend or frame line.
                score = (
                    bottom_gap * 8
                    + corner_gap * 4
                    + left_gap * 0.6
                    - h_len * 0.08
                    - v_len * 0.04
                )
                if score < best_score:
                    best_score = score
                    best_pair = (h_line, v_line)

        if best_pair:
            return best_pair
        return None, None

    # Step 1: 筛选出候选线段（水平/垂直，且长度 > 图像一半）
    filtered_lines = []
    for line in lines:
        if is_horizontal_or_vertical(line):
            angle = abs(line_angle(line))
            min_len = w * 0.5 if angle < angle_tolerance else h * 0.5
            if line_length(line) >= min_len:
                filtered_lines.append(recover_longest_subsegment(line, image_gray))

    # Step 2: 优先成对选择能形成真实左下角的 X/Y 轴，避免把图例框底边当作 X 轴。
    x_axis, y_axis = choose_axes_by_pair_score(filtered_lines)

    # Step 3: 兜底保留旧逻辑，避免极端图像上完全无结果。
    min_x = float("inf")
    max_y = float("-inf")

    if x_axis is None or y_axis is None:
        for line in filtered_lines:
            angle = abs(line_angle(line))
            length = line_length(line)
            if angle < angle_tolerance:  # 横线
                if length >= w * 0.5:
                    y_coords = [line[1], line[3]]
                    bottom_y = max(y_coords)
                    if bottom_y > max_y:
                        max_y = bottom_y
                        x_axis = line
            elif abs(angle - np.pi / 2) < angle_tolerance:  # 竖线
                if length >= h * 0.5:
                    x_coords = [line[0], line[2]]
                    left_x = min(x_coords)
                    if left_x < min_x:
                        min_x = left_x
                        y_axis = line

    if x_axis is None or y_axis is None:
        print("[Warning] 未找到有效的 X/Y 轴线段（长度不足或位置不合规）")
        return None, None, filtered_lines

    x_axis, y_axis = align_x_axis_to_y_axis(x_axis, y_axis)
    x_axis, y_axis = extend_line(x_axis), extend_line(y_axis)
    x_axis, y_axis = align_x_axis_to_y_axis(x_axis, y_axis)

    return x_axis, y_axis, filtered_lines


def main():
    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
    from utils.image_io import load_image, save_image
    import os
    import shutil
    import cv2
    from function_calling.axis.detect_lines import detect_candidate_lines
    from function_calling.axis.merge_lines import merge_similar_lines

    output_dir = DEBUG_OUTPUT_DIRS['infer_axes']
    if CLEAR_OUTPUT_BEFORE_RUN.get('infer_axes', False):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for name, path in IMG_PATHS.items():
        print(f"[Info] 处理: {name} => {path}")
        img = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        raw_lines = detect_candidate_lines(gray)
        merged_lines = merge_similar_lines(raw_lines)
        print(f"[Debug] 检测到 {len(merged_lines)} 条候选线段")

        x_axis, y_axis, filtered_lines = infer_axes_from_lines(merged_lines, (w, h), gray)
        if x_axis is None or y_axis is None:
            print(f"[Warning] {name} 未检测到 X/Y 轴，跳过。\\n")
        else:
            print(f"✅ X轴: {x_axis}\\n✅ Y轴: {y_axis}")
            vis = img.copy()
            cv2.line(vis, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 3)
            cv2.line(vis, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255, 0, 0), 3)
            save_image(vis, os.path.join(output_dir, f"result_{name}.png"))

        # vis_all = img.copy()
        # for line in filtered_lines:
        #     x1, y1, x2, y2 = line
        #     cv2.line(vis_all, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # save_image(vis_all, os.path.join(output_dir, f"all_candidates_{name}.png"))
        # print(f"[Debug] 保存候选线段图: all_candidates_{name}.png\\n")

if __name__ == '__main__':
    main()
