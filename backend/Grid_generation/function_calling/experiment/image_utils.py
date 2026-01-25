from PIL import Image, ImageDraw, ImageFont
from math import hypot
import numpy as np
import os

def draw_dashed_line(draw, start, end, dash_length=5, gap_length=5, fill="gray", width=1):
    x1, y1 = start
    x2, y2 = end
    total_length = hypot(x2 - x1, y2 - y1)
    dash_gap = dash_length + gap_length
    num_dashes = int(total_length // dash_gap)

    for i in range(num_dashes + 1):
        start_frac = (i * dash_gap) / total_length
        end_frac = min((i * dash_gap + dash_length) / total_length, 1)
        sx = x1 + (x2 - x1) * start_frac
        sy = y1 + (y2 - y1) * start_frac
        ex = x1 + (x2 - x1) * end_frac
        ey = y1 + (y2 - y1) * end_frac
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

def draw_crosshair_on_resized_image(
    img_path,  # str 或 Image 对象
    coords: list,
    output_path: str,
    color: str = "red",
    length: int = 18,
    thickness: int = 3
):
    if isinstance(img_path, str):
        img = Image.open(img_path).convert("RGB")
    else:
        img = img_path

    draw = ImageDraw.Draw(img)
    for (x, y) in coords:
        draw.line([(x - length // 2, y), (x + length // 2, y)], fill=color, width=thickness)
        draw.line([(x, y - length // 2), (x, y + length // 2)], fill=color, width=thickness)

    img.save(output_path)
    print(f"✅ Crosshair image saved to: {output_path}")

def convert_data_coord_to_resized_crop_pixel(
    data_coord, x_ticks, y_ticks, x_pixels, y_pixels,
    crop_origin, crop_size, resize_size
):
    from numpy import interp

    x_val, y_val = data_coord
    left, upper = crop_origin
    crop_w, crop_h = crop_size
    resize_w, resize_h = resize_size

    x_pix = interp(x_val, x_ticks, x_pixels)
    y_pix = interp(y_val, y_ticks, y_pixels)

    rel_x = x_pix - left
    rel_y = y_pix - upper

    scale_x = resize_w / crop_w
    scale_y = resize_h / crop_h
    return rel_x * scale_x, rel_y * scale_y

if __name__ == "__main__":
    # 🔧 测试 dashed line 和十字绘制
    test_img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(test_img)
    draw_dashed_line(draw, (10, 100), (190, 100), dash_length=6, gap_length=4, fill="gray", width=2)

    temp_path = "test_crosshair.png"
    draw_crosshair_on_resized_image(test_img, coords=[(100, 100), (50, 150)], output_path=temp_path)
    print(f"🖼️ 测试图像已保存: {temp_path}")
