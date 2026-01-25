# function_calling/filter_ticks.py
import numpy as np
from function_calling.axis.merge_lines import merge_similar_lines


def filter_ticks(tick_lines, direction='x', gap_tolerance=2):
    """
    根据刻度间距模式进行过滤，保留主间距上的刻度线，
    并考虑多组候选方案以避免偏移噪点影响，优先选择长度更长且间距更均匀的序列。
    """
    # 提取中心点坐标
    if direction == 'x':
        centers = [((line[0] + line[2]) // 2) for line in tick_lines]
    else:
        centers = [((line[1] + line[3]) // 2) for line in tick_lines]
    centers = np.array(centers)

    if len(centers) < 2:
        return tick_lines

    # 计算差值
    diffs = np.diff(np.sort(centers))
    if len(diffs) == 0:
        return tick_lines

    # 统计出现频率最高的间距
    rounded = np.round(diffs / 5) * 5  # 取整以避免偏移影响
    values, counts = np.unique(rounded, return_counts=True)
    main_gap = values[np.argmax(counts)]

    print(f"[Info] 推断主要刻度间距为：{main_gap}")

    # 预排序所有tick_lines及其中心坐标
    sorted_indices = np.argsort(centers)
    tick_lines_sorted = [tick_lines[i] for i in sorted_indices]
    centers_sorted = centers[sorted_indices]

    # 枚举所有起点，从每个起点开始向后寻找满足间距的序列
    candidates = []
    n = len(tick_lines_sorted)

    for i in range(n):
        seq = [tick_lines_sorted[i]]
        seq_centers = [centers_sorted[i]]
        last_center = centers_sorted[i]

        for j in range(i + 1, n):
            delta = centers_sorted[j] - last_center
            if abs(delta - main_gap) <= gap_tolerance:
                seq.append(tick_lines_sorted[j])
                seq_centers.append(centers_sorted[j])
                last_center = centers_sorted[j]
            elif delta > main_gap + 3 * gap_tolerance:
                break

        if len(seq) >= 3:
            gaps = np.diff(seq_centers)
            var = np.var(gaps)
            mid = np.mean(seq_centers)
            dist_to_center = np.abs(mid - np.mean(centers))
            candidates.append((len(seq), -var, -dist_to_center, seq))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][3]  # 返回长度最长、间距最均匀、位置居中的序列
    else:
        return tick_lines


# 🧪 调试入口
def main():
    import cv2
    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
    from utils.image_io import load_image, save_image
    from function_calling.axis.detect_lines import detect_candidate_lines
    from function_calling.axis.merge_lines import merge_similar_lines
    from function_calling.axis.infer_axes import infer_axes_from_lines
    from function_calling.ticks.detect_ticks import scan_pixels_for_ticks
    import os
    import shutil

    output_dir = DEBUG_OUTPUT_DIRS['filter_ticks']
    # 配置：是否清空输出目录
    if CLEAR_OUTPUT_BEFORE_RUN.get('filter_ticks', False):
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
            print(f"[Warning] {name} 未检测到 X/Y 轴，自动跳过。\n")
            continue

        # 原始tick + 合并
        x_raw = scan_pixels_for_ticks(img, x_axis, direction='x', scan_range=10)
        x_merged = merge_similar_lines(x_raw, angle_threshold=np.deg2rad(10))
        x_filtered = filter_ticks(x_merged, direction='x')

        y_raw = scan_pixels_for_ticks(img, y_axis, direction='y', scan_range=10)
        y_merged = merge_similar_lines(y_raw, angle_threshold=np.deg2rad(10))
        y_filtered = filter_ticks(y_merged, direction='y')

        print(f"[Result] X轴制度数：原始={len(x_raw)} 合并={len(x_merged)} 过滤后={len(x_filtered)}")
        print(f"[Result] Y轴制度数：原始={len(y_raw)} 合并={len(y_merged)} 过滤后={len(y_filtered)}")

        # 可视化
        vis = img.copy()
        cv2.line(vis, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0,0,255), 2)
        cv2.line(vis, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255,0,0), 2)
        for x1, y1, x2, y2 in x_filtered:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in y_filtered:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

        out_path = os.path.join(output_dir, f"result_{name}.png")
        save_image(vis, out_path)
        print(f"✅ 过滤后的制度线图像已保存至：{out_path}")

if __name__ == '__main__':
    main()
