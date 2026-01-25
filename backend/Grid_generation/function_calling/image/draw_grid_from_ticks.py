# function_calling/image/draw_grid_from_ticks.py

import os
import cv2
import json
import numpy as np
from glob import glob
from utils.image_io import load_image
from config import GRID_CONFIG, DEBUG_OUTPUT_DIRS


def draw_grid_from_ticks(img, x_ticks, y_ticks, x_axis, y_axis,
                         grid_color=(180, 180, 180), grid_thickness=1, grid_line_type=None,
                         show_result=True, save_path=None):
    if grid_line_type is None:
        grid_line_type = cv2.LINE_AA

    canvas = img.copy()
    cv2.line(canvas, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)  # 红色 X 轴
    cv2.line(canvas, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255, 0, 0), 2)  # 蓝色 Y 轴

    y_axis_y1 = min(y_axis[1], y_axis[3])
    y_axis_y2 = max(y_axis[1], y_axis[3])
    x_axis_x1 = min(x_axis[0], x_axis[2])
    x_axis_x2 = max(x_axis[0], x_axis[2])

    dashed = canvas.copy()
    for x1, y1, x2, y2 in x_ticks:
        cx = (x1 + x2) // 2
        for y in range(y_axis_y1, y_axis_y2, 4):
            if (y // 2) % 2 == 0:
                cv2.line(dashed, (cx, y), (cx, y + 2), grid_color, grid_thickness, grid_line_type)

    for x1, y1, x2, y2 in y_ticks:
        cy = (y1 + y2) // 2
        for x in range(x_axis_x1, x_axis_x2, 4):
            if (x // 2) % 2 == 0:
                cv2.line(dashed, (x, cy), (x + 2, cy), grid_color, grid_thickness, grid_line_type)

    canvas = dashed

    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"✅ 网格图像已保存到: {save_path}")

    if show_result:
        cv2.imshow("Grid Preview", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return canvas


if __name__ == '__main__':
    input_dir = "data/debug/detect_ticks/tick_data"
    image_dir = "data/basic_images/rect"
    output_dir = DEBUG_OUTPUT_DIRS['draw_grid_from_ticks']
    os.makedirs(output_dir, exist_ok=True)

    for json_path in glob(os.path.join(input_dir, "*.json")):
        with open(json_path, 'r') as f:
            tick_data = json.load(f)

        name = tick_data.get("image_name")
        if not name:
            continue

        image_path = os.path.join(image_dir, f"{name}.png")
        if not os.path.exists(image_path):
            continue

        img = load_image(image_path)
        x_ticks = tick_data.get("x_ticks", [])
        y_ticks = tick_data.get("y_ticks", [])
        x_axis = tick_data.get("x_axis")
        y_axis = tick_data.get("y_axis")

        save_path = os.path.join(output_dir, f"grid_preview_{name}.png")
        draw_grid_from_ticks(
            img, x_ticks, y_ticks, x_axis, y_axis,
            grid_color=GRID_CONFIG['grid_color'],
            grid_thickness=GRID_CONFIG['grid_thickness'],
            grid_line_type=GRID_CONFIG['grid_line_type'],
            show_result=GRID_CONFIG['show_result'],
            save_path=save_path
        )
