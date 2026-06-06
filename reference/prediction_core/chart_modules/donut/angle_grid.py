"""Angle-grid image generation for donut charts."""

from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFont


def draw_angle_grid_30deg(
        cfg: dict,
        img_type: str,
        output_suffix: str,
        inner_radius: int,
        line_color: tuple = (0, 0, 0, 255),
        line_width: int = 1,
        font_size: int = 8,
        text_offset: int = 15,
        grid_line_ratio: float = 0.1,
        text_offset_ratio: float = 0.1,
) -> str:
    """Generate the clockwise angle-grid image used as the with_grid input."""
    print(f"[ANGLE_GRID] generating with_grid image: {cfg['chart_id']}")

    src_path = cfg["image_paths"][img_type]
    img = Image.open(src_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    cx, cy = cfg["center"]
    img_width, img_height = img.size

    font_size = 20 if img_width >= 1000 or img_height >= 1000 else 12
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    grid_line_length = int(inner_radius * grid_line_ratio)
    radius_text = inner_radius + grid_line_length + int(inner_radius * text_offset_ratio)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    labels = []

    for angle_deg in range(0, 360, 15):
        theta = math.radians(angle_deg - 90)

        x_start = cx + inner_radius * math.cos(theta)
        y_start = cy + inner_radius * math.sin(theta)
        x_end = cx + (inner_radius + grid_line_length) * math.cos(theta)
        y_end = cy + (inner_radius + grid_line_length) * math.sin(theta)
        draw.line([(x_start, y_start), (x_end, y_end)], fill=line_color, width=line_width)

        tx = cx + radius_text * math.cos(theta)
        ty = cy + radius_text * math.sin(theta)
        label = f"{angle_deg}{chr(176)}"

        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad = 4
        o.rounded_rectangle(
            [
                tx - text_w / 2 - pad,
                ty - text_h / 2 - pad,
                tx + text_w / 2 + pad,
                ty + text_h / 2 + pad,
            ],
            radius=4,
            fill=(255, 255, 255, 0),
        )
        labels.append((tx, ty, label, text_w, text_h))

    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0, 255))

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)

    for tx, ty, label, text_w, text_h in labels:
        draw.text(
            (tx - text_w / 2, ty - text_h / 2),
            label,
            fill=(0, 0, 0, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )

    base, ext = os.path.splitext(src_path)
    out_path = f"{base}{output_suffix}{ext}"
    img.save(out_path)
    print(f"[ANGLE_GRID] saved: {out_path}")
    return out_path
