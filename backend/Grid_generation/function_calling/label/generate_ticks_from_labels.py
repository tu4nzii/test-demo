import os
import json
import cv2
import numpy as np
from config import DEBUG_OUTPUT_DIRS, IMG_PATHS
from utils.image_io import read_image
from utils.drawing import draw_points

def midpoint(box):
    x1, y1, x2, y2 = box
    return [(x1 + x2) // 2, (y1 + y2) // 2]

def project_point_to_axis(p, axis_line):
    x0, y0 = p
    x1, y1, x2, y2 = axis_line
    axis_vec = np.array([x2 - x1, y2 - y1])
    point_vec = np.array([x0 - x1, y0 - y1])
    proj_len = np.dot(axis_vec, point_vec) / np.linalg.norm(axis_vec)**2
    proj = np.array([x1, y1]) + proj_len * axis_vec
    return proj.astype(int).tolist()

def generate_ticks_from_label_boxes(name,
                                     x_axis_line=None,
                                     y_axis_line=None):
    label_json_path = os.path.join(DEBUG_OUTPUT_DIRS['recognize_tick_labels'], f"{name}.json")
    img_path = IMG_PATHS[name]
    img = read_image(img_path)

    with open(label_json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    tick_points_x = []
    tick_points_y = []

    for label in label_data:
        box = label['box']
        mid = midpoint(box)
        if x_axis_line:
            tick = project_point_to_axis(mid, x_axis_line)
            tick_points_x.append(tick)
        if y_axis_line:
            tick = project_point_to_axis(mid, y_axis_line)
            tick_points_y.append(tick)

    vis = img.copy()
    if tick_points_x:
        vis = draw_points(vis, tick_points_x, color=(0, 255, 0), radius=5)
    if tick_points_y:
        vis = draw_points(vis, tick_points_y, color=(0, 255, 255), radius=5)

    save_path = os.path.join(DEBUG_OUTPUT_DIRS['recognize_tick_labels'], f"tick_from_labels_{name}.png")
    cv2.imwrite(save_path, vis)
    print(f"✅ 虚拟 tick 点已保存到: {save_path}")

    tick_json = {
        'x_ticks_from_labels': tick_points_x,
        'y_ticks_from_labels': tick_points_y
    }
    json_path = os.path.join(DEBUG_OUTPUT_DIRS['recognize_tick_labels'], f"tick_from_labels_{name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tick_json, f, indent=2)

    print(f"✅ tick 点数据保存到: {json_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='图像文件名（不带后缀）')
    args = parser.parse_args()

    x_axis = [55, 280, 680, 280]
    y_axis = [55, 420, 55, 30]

    generate_ticks_from_label_boxes(args.name, x_axis_line=x_axis, y_axis_line=y_axis)
