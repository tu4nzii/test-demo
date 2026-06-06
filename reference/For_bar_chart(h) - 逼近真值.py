# Experiment Scheduler for Grid-Based Prompt Evaluation (Three Prompt-Image Settings)
from __future__ import annotations
import json
import math
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import base64
import requests
import time
import asyncio
import aiohttp
from typing import List, Union
import numpy as np
from pandas import Series
from pandas.core.dtypes.inference import is_number
import sys
import argparse   # ✅ 新增
import asyncio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors   # ✅ 新增
from numbers import Number   # ✅ 新增
from typing import Tuple, Optional



EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    # ("feedback", "grid_with_grid"),
    # ("amplifier", "grid_with_grid"), # ✅ 新增
]

# 新增函数：自动加载 charts 文件夹下的文件配置
def load_chart_configs():
    config_dir = "chart_configs"
    chart_configs = []

    for filename in os.listdir(config_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(config_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                config = json.load(f)
                chart_configs.append(config)
    return chart_configs

# === 模拟图表数据与标注点 ===
# 新增批量处理配置，支持多数据集配置
DATASET_CONFIGS = load_chart_configs()

import re
def safe_filename(name: str) -> str:
    """替换掉 Windows 不允许的文件名字符"""
    return re.sub(r'[\\/:*?"<>|]', "_", name)

# 数值型坐标值的插值映射（连续坐标轴）
def build_axis_mapping(tick_values, tick_pixels):
    return lambda v: np.interp(v, tick_values, tick_pixels)


# ========= New helper: 像素跨度推断（增强版）=========
def get_category_span(label: str,
                      tick_labels: list,
                      tick_pixels: list[int],
                      img_min: int,
                      img_max: int,
                      mode: str = "center") -> tuple[int, int]:
    """
    给定某个分类标签及其在图像上的 tick 像素位置，返回该分类项在图像上的像素范围（起止边界）。
    自动处理像素从上往下（升序）或从下往上（降序）两种情况，并返回有序的 (start, end)。

    ✅ 改进：
    - 允许 label 本身包含逗号。
    - 自动尝试模糊匹配（包含 / 相互包含）以修正模型输出略有截断的情况。
    - 输出详细匹配日志。
    """

    raw_label = label.strip()

    # Step 1️⃣：逗号分割（保留原始 label 以防回退）
    if "," in raw_label:
        parts = [p.strip() for p in raw_label.split(",")]
        label = parts[-1]
    else:
        label = raw_label

    print(f"🧪 正在匹配 label: {label} （类型：{type(label)}）")

    # Step 2️⃣：精确匹配
    if label in tick_labels:
        matched_label = label
        print(f"✅ 精确匹配成功: {matched_label}")
    else:
        # Step 3️⃣：模糊匹配
        matches = [yt for yt in tick_labels if label in yt or yt in label]
        if matches:
            matched_label = matches[0]
            print(f"⚠️ 未精确匹配，自动模糊匹配 → {matched_label}")
        else:
            # Step 4️⃣：完全匹配失败，给出完整错误上下文
            print("❌ 无法匹配 label!")
            print("tick_labels 可选项：", tick_labels)
            raise ValueError(f"Label '{label}' not found in tick_labels.")

    idx = tick_labels.index(matched_label)

    # Step 5️⃣：计算像素范围
    if mode == "center":
        center = tick_pixels[idx]
        if idx > 0:
            prev = tick_pixels[idx - 1]
            left_or_top = (prev + center) // 2
        else:
            left_or_top = img_min

        if idx < len(tick_pixels) - 1:
            next_ = tick_pixels[idx + 1]
            right_or_bottom = (center + next_) // 2
        else:
            right_or_bottom = img_max

    elif mode == "left":
        left_or_top = tick_pixels[idx]
        right_or_bottom = tick_pixels[idx + 1] if idx < len(tick_pixels) - 1 else img_max
    else:
        raise ValueError(f"Unsupported mode '{mode}', use 'center' or 'left'")

    return tuple(sorted((left_or_top, right_or_bottom)))

def build_categorical_axis_mapping_fuzzy(tick_labels, tick_pixels):
    label_to_pixel = dict(zip(tick_labels, tick_pixels))

    def mapper(label):
        print(f"🧪 正在匹配 label: {label} （类型：{type(label)}）")
        label = str(label)

        # ✅ 直接精确匹配
        if label in label_to_pixel:
            return label_to_pixel[label]

        # ✅ 特别处理：尝试提取最后一段作为 fallback
        fallback = label.split(",")[-1].strip()
        if fallback in label_to_pixel:
            print(f"🔄 使用 fallback 匹配: '{fallback}'")
            return label_to_pixel[fallback]

        # ✅ 更宽松的模糊匹配：tick_label 出现在 label 中
        for tick_label in tick_labels:
            if tick_label in label:
                print(f"🔄 模糊匹配: '{tick_label}' in '{label}'")
                return label_to_pixel[tick_label]

        print(f"⚠️ 无法匹配 label: {label}")
        return -1

    return mapper


# ========= 统一接口（水平条形图专用） =========
def get_axis_mapper(ticks, pixels, axis_type):
    """
    根据轴类型或名称返回映射函数
    兼容两种调用：
    - 旧：传 'categorical' / 'numerical'
    - 新：传 'x' / 'y' （水平条形图约定：x 数值轴，y 分类轴）
    """
    # ✅ 兼容 axis_name = "x"/"y"
    if axis_type == "x":
        return build_axis_mapping(ticks, pixels)  # 数值轴
    elif axis_type == "y":
        return build_categorical_axis_mapping_fuzzy(ticks, pixels)  # 分类轴

    # ✅ 兼容旧参数 axis_type = "categorical"/"numerical"
    elif axis_type == "categorical":
        return build_categorical_axis_mapping_fuzzy(ticks, pixels)
    elif axis_type == "numerical":
        return build_axis_mapping(ticks, pixels)

    else:
        raise ValueError(f"Unsupported axis_type '{axis_type}', must be 'x'/'y' or 'categorical'/'numerical'")

# ------------------ 视觉反馈函数（新版） ------------------
def sanitize_filename(path: str) -> str:
    """替换路径中的非法字符，保证在 Windows 下能保存"""
    folder, filename = os.path.split(path)
    safe_filename = re.sub(r'[\\/:*?"<>|]', "_", filename)
    return os.path.join(folder, safe_filename)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_color_name_approx(hex_color: str) -> str:
    """自动返回最接近的 CSS4 命名颜色"""
    r, g, b = hex_to_rgb(hex_color)
    min_dist = float('inf')
    closest_name = "unknown"
    for name, hex_code in mcolors.CSS4_COLORS.items():
        r2, g2, b2 = hex_to_rgb(hex_code)
        dist = (r - r2)**2 + (g - g2)**2 + (b - b2)**2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def generate_series_color_description(series_color: dict[str, str]) -> str:
    """生成自然语言段落，用于描述每个 series 的颜色"""
    lines = ["The chart uses specific colors for each series:"]
    for series, hex_val in series_color.items():
        rgb = hex_to_rgb(hex_val)
        color_name = get_color_name_approx(hex_val)
        lines.append(f'- "{series}" is colored **{hex_val}** (approx. {color_name}, RGB: {rgb}).')
    return "\n".join(lines)



def build_color_prompt(
    point_name: str,
    series_color: dict[str, str],
) -> str:
    # ✅ 提取逗号前的主标签部分，例如 "Total revenue, Deloitte" → "Total revenue"
    main_label = point_name.split(",", 1)[0].strip()

    # ✅ 生成颜色描述
    color_desc = generate_series_color_description(series_color)

    # ✅ 构造提示词
    return (
        f"You are given a cropped bar chart image for {point_name}.\n"
        f"**{color_desc}**.\n"
        f"Please check if there is the correct color bar segment for \"{main_label}\" visible that corresponds to the color alignment.\n"
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )



def generate_prompt(
        item_name: str,
        prompt_type: str,
        x_ticks: list,
        y_ticks: list,
        series_color: dict[str, str],  # series_name -> color
        visible_ticks: list,
        axis_types: dict = None,  # ✅ 可以传也可以不传，内部强制设定
        pred_feedback: list = None,
        feedback_round: int = 0,
        current_round: int = 1
) -> str:
    # === 默认固定设定 ===
    x_type = "numerical"
    y_type = "categorical"

    # === 从 item_name 拆出 (series_name, y_label) ===
    try:
        series_name, y_label = item_name.rsplit(",", 1)
        series_name = series_name.strip()
        y_label = y_label.strip()
    except ValueError:
        raise ValueError(f"⚠️ item_name 解析失败：'{item_name}' 不符合 'series_name, y_label' 格式")

    # === ticks 拼接 ===
    x_tick_str = ", ".join(str(x) for x in x_ticks)
    y_tick_str = ", ".join(str(y) for y in y_ticks)

    # === 颜色描述 ===
    color_desc = generate_series_color_description(series_color)

    # === 1. baseline 提示 ===
    if prompt_type == "baseline":
        base_prompt = (
            f"You are given a bar chart image. "
            f"{color_desc}\n"
            f"Your task is to predict the x coordinate for the segment labeled [{item_name}].\n"
            f"To identify the x coordinate, first locate the tick interval in which the right boundary of the segment representing [{item_name}] falls."
        )

    # === 2. amplifier 提示 ===
    elif prompt_type == "amplifier":
        # 🛠️ 处理 visible_ticks 展开 & 去重排序
        if len(visible_ticks) > 0 and isinstance(visible_ticks[0], list):
            visible_ticks = visible_ticks[0]
        x_tick_str = ", ".join(str(round(x, 2)) for x in sorted(set(visible_ticks)))

        base_prompt = f'''
        You are given a chart image. Your task is to predict the x coordinate for the segment labeled [{item_name}].
        The segment appears in the **center**, extracted from the full chart by locating the category label **"{y_label}"** on the y-axis.
        The top and bottom sides include a **horizontally drawn x-axis**, with tick values [{x_tick_str}] and grid lines.
        Your task is to estimate the **x coordinate** corresponding to the **right boundary** of the colored segment.
        The segment color indicates its category: use alignment between the legend and segment to verify the target. {color_desc}
        Instructions:
            - First, locate the x-axis tick interval in which the segment’s right boundary falls. 
            - Then, determine the relative position of the boundary within this interval. Use linear interpolation between the two tick values to estimate the precise x-axis value.
            - **Important:** Do not snap or round to the nearest tick; interpolate proportionally.
            - **Edge case:** If the segment cannot be visually detected even near the **minimum tick boundary**, output the **minimum tick value (e.g., 0)** as the coordinate.
        '''

    # === 3. grid / feedback 提示 ===
    else:
        base_prompt = f'''
        You are analyzing a bar chart that contains **reference grid lines**, where horizontal lines correspond to y-axis ticks, and vertical lines align with x-axis ticks.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        {color_desc}
        After locating the correct segment for [{item_name}], identify the position of its right edge by comparing it with the two nearest vertical grid lines on the X-axis. 
        Use linear interpolation between these two ticks to estimate the accurate X-coordinate of [{item_name}].
        '''

    # === 4. feedback 历史修正 ===
    if prompt_type == "feedback" and pred_feedback and current_round >= feedback_round:
        if isinstance(pred_feedback, list):
            pred = pred_feedback[-1]
            x = f'{pred[0]:.2f}' if isinstance(pred[0], (int, float)) else f'"{pred[0]}"'
            y = f'{pred[1]:.2f}' if isinstance(pred[1], (int, float)) else f'"{pred[1]}"'

        base_prompt = f'''
        You are analyzing a bar chart with reference grid lines.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        {color_desc}        

        The given chart shows your previous prediction for the x coordinate of [{item_name}], aligned with a red crosshair at (x = {x}, y = {y}). 
        Compare this red crosshair to the true right boundary of [{item_name}]: determine whether it is too far left, too far right, or aligned correctly. 
        After verifying, adjust the prediction proportionally to refine your estimate to get the most accurate result x coordinate of [{item_name}].
        '''

    # === 5. JSON 输出固定为 (x 数值, y 分类) ===
    base_prompt += f'''
    Only respond in this JSON format:
    {{"datapoints": [{{"{item_name}": [x, "{y_label}"]}}]}}
    '''

    return base_prompt


#OK for x===catogorical, y===numerical
def generate_overlayed_image_multi_with_mapping(
        original_img_path: str,
        pred_coords: list,
        x_ticks: list,
        y_ticks: list,
        x_pixels: list,
        y_pixels: list,
        output_path: str,
        feedback_round: int = 1,
        draw_all_preds: bool = False,
        axis_types: dict | None = None
):
    if axis_types is None:
        axis_types = {"x": "numerical", "y": "categorical"}

    x_type = axis_types.get("x", "numerical")
    y_type = axis_types.get("y", "categorical")

    # --- 像素映射器 ---
    x_mapper = get_axis_mapper(x_ticks, x_pixels, "numerical")
    y_mapper = build_categorical_axis_mapping_fuzzy(y_ticks, y_pixels)  # ✅ 模糊匹配器

    # --- 载入原图 ---
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    # --- 样式 ---
    radius = 5
    colors = ["red", "purple", "orange", "green", "blue",
              "black", "brown", "pink", "gray", "cyan"]

    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]

    for idx, coord in enumerate(coords_to_draw):
        print(f"这是用于匹配的标签内容：{coord[1]}")
        true_y_label = str(coord[1]).split(",")[-1].strip()
        print(f"真正的 y 轴标签内容：{true_y_label}")

        try:
            if x_type == "numerical" and isinstance(coord[0], str):
                raise ValueError(f"x轴为numerical，但坐标为str: {coord[0]}")
            if y_type == "numerical" and isinstance(coord[1], str):
                raise ValueError(f"y轴为numerical，但坐标为str: {coord[1]}")

            x_pixel = int(x_mapper(coord[0]))
            y_pixel = int(y_mapper(true_y_label))  # ✅ 模糊容错匹配
        except Exception as e:
            print(f"❌ 坐标映射失败: {coord} | {e}")
            continue

        color = colors[idx % len(colors)]

        # ---------- 计算当前条带的跨度 ----------
        distances = [abs(p - y_pixel) for p in y_pixels]
        nearest_idx = int(np.argmin(distances))
        if nearest_idx == 0:
            span = abs(y_pixels[1] - y_pixels[0])
        elif nearest_idx == len(y_pixels) - 1:
            span = abs(y_pixels[-1] - y_pixels[-2])
        else:
            span = (abs(y_pixels[nearest_idx] - y_pixels[nearest_idx - 1]) +
                    abs(y_pixels[nearest_idx] - y_pixels[nearest_idx + 1])) // 2

        half_span = span // 2

        draw.line((x_pixel - half_span, y_pixel, x_pixel + half_span, y_pixel),
                  fill=color, width=2)
        draw.line((x_pixel, y_pixel - half_span, x_pixel, y_pixel + half_span),
                  fill=color, width=2)

    # --- 保存 ---
    # 取到原始文件名
    output_filename = os.path.basename(output_path)

    # 只清洗文件名，不动路径
    output_filename = sanitize_filename(output_filename)

    # 提取图表ID
    chart_id = os.path.splitext(os.path.basename(original_img_path))[0]
    chart_id = chart_id.replace("_with_grid", "").replace("_grid", "")

    # 组装最终路径
    folder = os.path.join("results", chart_id, "tempy")
    os.makedirs(folder, exist_ok=True)

    # 截断长文件名（避免 Windows 260 限制）
    if len(output_filename) > 120:
        base, ext = os.path.splitext(output_filename)
        output_filename = base[:120] + ext

    output_path = os.path.join(folder, output_filename)

    # 保存图片
    img.save(output_path)
    print(f"✅ 反馈图已保存至: {output_path}")
    return output_path


# # --- 保存 ---
   #  output_filename = os.path.basename(output_path)
   #
   #  # 提取图表名（去掉 _with_grid / _grid 等后缀）
   #  chart_id = os.path.splitext(os.path.basename(original_img_path))[0]
   #  chart_id = chart_id.replace("_with_grid", "").replace("_grid", "")
   #
   #  # 组装最终路径
   #  output_path = os.path.join("results", chart_id, "tempy", output_filename)
   #
   #  output_path = sanitize_filename(output_path)
   #  folder = os.path.dirname(output_path)
   #  if folder:
   #      os.makedirs(folder, exist_ok=True)
   #
   #  os.makedirs(os.path.dirname(output_path), exist_ok=True)
   #  img.save(output_path)
   #  print(f"✅ 反馈图已保存至: {output_path}")
   #  return output_path


def crop_segment_with_dual_yaxis_ticks(
    image_path: str,
    point_name: str,
    y_value: float,
    x_label: str,
    x_ticks: list,
    x_pixels: list,
    y_ticks: list,
    y_pixels: list,
    out_size: int = 240,
    side_pad: int = 10,
    vert_pad: int = 30,
    dash_region: int = 5,
    dash_len: int = 10,
    dash_gap: int = 4,
    tick_len: int = 6,
    border_width: int = 2,
    border_color: str = "black",
    feedback_round: int = 0,
    output_path: str | None = None
) -> tuple[str, list, tuple[float, float]]:

    from PIL import Image, ImageDraw, ImageFont
    import os

    # ---------- 辅助函数：旋转文字 ----------
    def draw_rotated_text(base_img, text, center, angle, font, fill="black"):
        dummy = Image.new("RGB", (10, 10))
        dummy_draw = ImageDraw.Draw(dummy)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        pad = 10
        txt_img = Image.new("RGBA", (w + pad, h + pad), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((pad // 2, pad // 2), text, font=font, fill=fill)
        rotated = txt_img.rotate(angle, expand=1)

        rw, rh = rotated.size
        paste_x = int(center[0] - rw / 2)
        paste_y = int(center[1] - rh / 2)
        base_img.paste(rotated, (paste_x, paste_y), rotated)

    # ---------- 字体 ----------
    try:
        if os.name == "nt":
            font_path = "C:/Windows/Fonts/arial.ttf"
        else:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, size=14)
    except:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (10, 10))
    ddraw = ImageDraw.Draw(dummy)
    max_label_height = max(ddraw.textbbox((0, 0), str(v), font=font)[3] for v in y_ticks)
    required_top = dash_region + tick_len + 8 + max_label_height
    vert_pad_eff = max(vert_pad, required_top)

    # ---------- 打开原图 ----------
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # ---------- 找到 y 分类 ----------
    true_y_label = str(point_name).split(",")[-1].strip()
    idx = y_ticks.index(true_y_label)
    center_y = y_pixels[idx]

    if len(y_pixels) > 1:
        if idx == 0:
            band = abs(y_pixels[1] - y_pixels[0])
        elif idx == len(y_pixels) - 1:
            band = abs(y_pixels[-1] - y_pixels[-2])
        else:
            band = (abs(y_pixels[idx] - y_pixels[idx - 1]) +
                    abs(y_pixels[idx] - y_pixels[idx + 1])) // 2
    else:
        band = 80

    half_band = band // 2
    top_crop = max(0, center_y - half_band)
    bottom_crop = min(img_h, center_y + half_band)

    # ---------- 定位 x ----------
    pairs = sorted(zip(x_ticks, x_pixels))
    v_min, v_max = pairs[0][0], pairs[-1][0]
    p_min, p_max = pairs[0][1], pairs[-1][1]
    scale = (p_max - p_min) / (v_max - v_min) if v_max != v_min else 1.0

    x_val = max(min(y_value, v_max), v_min)
    center_x = p_min + (x_val - v_min) * scale

    if len(pairs) > 1:
        base_span_px = abs(pairs[1][1] - pairs[0][1])
    else:
        base_span_px = 20

    k = max(0, feedback_round - 1)
    half_span = max(5, base_span_px / (2 ** k))

    left_crop = max(0, int(center_x - half_span))
    right_crop = min(img_w, int(center_x + half_span))

    if right_crop <= left_crop:
        right_crop = min(img_w, left_crop + 10)

    # ---------- 裁剪 ----------
    cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))
    crop_w, crop_h = cropped.size

    # ---------- progressive zoom ----------
    zoom = 2 ** max(0, feedback_round - 1)
    new_w = int(crop_w * zoom)
    new_h = int(crop_h * zoom)

    resized = cropped.resize((new_w, new_h), resample=Image.NEAREST)

    # ---------- 自适应 canvas ----------
    label_pad_top = 70
    label_pad_bottom = 70

    canvas_w = new_w
    canvas_h = new_h + label_pad_top + label_pad_bottom

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    offset_x = 0
    offset_y = label_pad_top
    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)

    crop_top = offset_y
    crop_bottom = offset_y + new_h

    # ---------- dense ticks ----------
    dense_ticks = []
    dense_pixels = []

    for divisor in [2, 3, 4, 5, 6, 8, 10]:
        dense_ticks.clear()
        dense_pixels.clear()

        for i in range(len(pairs) - 1):
            v1, p1 = pairs[i]
            v2, p2 = pairs[i + 1]
            for j in range(divisor + 1):
                a = j / divisor
                dense_ticks.append(round(v1 + (v2 - v1) * a, 2))
                dense_pixels.append(p1 + (p2 - p1) * a)

        if sum(left_crop <= p <= right_crop for p in dense_pixels) >= 6:
            break

    # ---------- 映射：关键补丁（value,pixel）使 key 不再覆盖 ----------
    x_mapping = {}
    for v, p in zip(dense_ticks, dense_pixels):
        if not (left_crop <= p <= right_crop):
            continue
        rel = (p - left_crop) / crop_w
        pix = offset_x + int(rel * new_w)

        # ★★★ 核心修复点（1）：避免覆盖 ★★★
        x_mapping[(v, pix)] = pix

    # ---------- 绘制 ----------
    def draw_ticks(side):
        for (v, pix), x_pix in x_mapping.items():
            if side == "top":
                draw.line([(x_pix, crop_top - dash_region - tick_len),
                           (x_pix, crop_top - dash_region)], fill="black", width=1)
                center = (x_pix, crop_top - dash_region - tick_len - 25)
                draw_rotated_text(canvas, str(v), center, 45, font)
            else:
                draw.line([(x_pix, crop_bottom + dash_region),
                           (x_pix, crop_bottom + dash_region + tick_len)], fill="black", width=1)
                center = (x_pix, crop_bottom + dash_region + tick_len + 25)
                draw_rotated_text(canvas, str(v), center, -90, font)

    draw_ticks("top")

    for (v, pix), x_pix in x_mapping.items():
        y = crop_top
        while y < crop_bottom:
            draw.line([(x_pix, y), (x_pix, min(y + dash_len, crop_bottom))],
                      fill="gray", width=1)
            y += dash_len + dash_gap

    draw_ticks("bottom")

    draw.line([(offset_x, crop_top), (offset_x + new_w, crop_top)], fill=border_color, width=border_width)
    draw.line([(offset_x, crop_bottom), (offset_x + new_w, crop_bottom)], fill=border_color, width=border_width)

    # ---------- 保存 ----------
    if output_path is None:
        safe_name = safe_filename(point_name)
        output_filename = f"amplifier_crop_{safe_name}_round{feedback_round}.png"
        chart_id = os.path.splitext(os.path.basename(image_path))[0]
        chart_id = chart_id.replace("_with_grid", "").replace("_grid", "")
        output_path = os.path.join("results", chart_id, "tempy", output_filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)

    # ---------- 数值范围 ----------
    min_val = v_min + (left_crop - p_min) / scale
    max_val = v_min + (right_crop - p_min) / scale
    visible_range = (round(min_val, 4), round(max_val, 4))

    # ★★★ 核心修复点（2）：visible tick 只返回 value，而不是 (value,pixel) ★★★
    visible_ticks = [vp[0] for vp, pix in x_mapping.items()]

    return output_path, visible_ticks, visible_range


# def crop_segment_with_dual_yaxis_ticks(
#     image_path: str,
#     point_name: str,
#     y_value: float,
#     x_label: str,
#     x_ticks: list,
#     x_pixels: list,
#     y_ticks: list,
#     y_pixels: list,
#     out_size: int = 240,
#     side_pad: int = 10,
#     vert_pad: int = 30,
#     dash_region: int = 5,
#     dash_len: int = 10,
#     dash_gap: int = 4,
#     tick_len: int = 6,
#     border_width: int = 2,
#     border_color: str = "black",
#     feedback_round: int = 0,
#     output_path: str | None = None
# ) -> tuple[str, list, tuple[float, float]]:
#
#     from PIL import Image, ImageDraw, ImageFont
#     import os
#
#     # ---------- 辅助函数：保持不变 ----------
#     def draw_rotated_text(base_img, text, center, angle, font, fill="black"):
#         dummy = Image.new("RGB", (10, 10))
#         dummy_draw = ImageDraw.Draw(dummy)
#         bbox = dummy_draw.textbbox((0, 0), text, font=font)
#         w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
#
#         pad = 10
#         txt_img = Image.new("RGBA", (w+pad, h+pad), (255,255,255,0))
#         txt_draw = ImageDraw.Draw(txt_img)
#         txt_draw.text((pad//2, pad//2), text, font=font, fill=fill)
#
#         rotated = txt_img.rotate(angle, expand=1)
#         rw, rh = rotated.size
#
#         base_img.paste(rotated, (int(center[0]-rw/2), int(center[1]-rh/2)), rotated)
#
#     # ---------- 0. 字体，不变 ----------
#     try:
#         if os.name == "nt":
#             font_path = "C:/Windows/Fonts/arial.ttf"
#         else:
#             font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
#         font = ImageFont.truetype(font_path, size=14)
#     except:
#         font = ImageFont.load_default()
#
#     dummy = Image.new("RGB", (10, 10))
#     ddraw = ImageDraw.Draw(dummy)
#     max_label_height = max(ddraw.textbbox((0,0), str(v), font=font)[3] for v in y_ticks)
#     required_top = dash_region + tick_len + 8 + max_label_height
#     vert_pad_eff = max(vert_pad, required_top)
#
#     # ---------- 1. 打开图片 ----------
#     img = Image.open(image_path).convert("RGB")
#     img_w, img_h = img.size
#
#     # ---------- 2. 找 y 分类 ----------
#     true_y_label = str(point_name).split(",")[-1].strip()
#     idx = y_ticks.index(true_y_label)
#     center_y = y_pixels[idx]
#
#     if len(y_pixels) > 1:
#         if idx == 0:
#             band = abs(y_pixels[1] - y_pixels[0])
#         elif idx == len(y_pixels)-1:
#             band = abs(y_pixels[-1] - y_pixels[-2])
#         else:
#             band = (abs(y_pixels[idx]-y_pixels[idx-1]) + abs(y_pixels[idx]-y_pixels[idx+1]))//2
#     else:
#         band = 80
#
#     half_band = band//2
#     top_crop = max(0, center_y-half_band)
#     bottom_crop = min(img_h, center_y+half_band)
#
#     # ---------- 3. 找 x 中心 ----------
#     pairs = sorted(zip(x_ticks, x_pixels))
#     v_min, v_max = pairs[0][0], pairs[-1][0]
#     p_min, p_max = pairs[0][1], pairs[-1][1]
#
#     scale = (p_max-p_min)/(v_max-v_min) if v_max!=v_min else 1.0
#     x_val = max(min(y_value, v_max), v_min)
#     center_x = p_min + (x_val - v_min)*scale
#
#     if len(pairs)>1:
#         base_span_px = abs(pairs[1][1] - pairs[0][1])
#     else:
#         base_span_px = 20
#
#     k = max(0, feedback_round-1)
#     half_span = max(5, base_span_px/(2**k))
#
#     left_crop  = max(0, int(center_x-half_span))
#     right_crop = min(img_w, int(center_x+half_span))
#     if right_crop <= left_crop:
#         right_crop = min(img_w, left_crop+10)
#
#     # ---------- 4. 裁剪 ----------
#     cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))
#     crop_w, crop_h = cropped.size
#
#     # ---------- 5. progressive zoom（新的核心） ----------
#     zoom = 2 ** max(0, feedback_round-1)
#     new_w = int(crop_w * zoom)
#     new_h = int(crop_h * zoom)
#
#     resized = cropped.resize((new_w, new_h), resample=Image.NEAREST)
#
#     # ---------- 6. 画布（新的关键：canvas_w = new_w，offset_x = 0） ----------
#     label_pad_top = 70
#     label_pad_bottom = 70
#
#     canvas_w = new_w          # ❗❗❗ 不再是 out_size，保证横向不压缩
#     canvas_h = new_h + label_pad_top + label_pad_bottom
#
#     canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
#     offset_x = 0              # ❗❗❗ 不水平居中，避免网格线错位
#     offset_y = label_pad_top
#     canvas.paste(resized, (offset_x, offset_y))
#     draw = ImageDraw.Draw(canvas)
#
#     crop_top = offset_y
#     crop_bottom = offset_y + new_h
#
#     # ---------- 7. 构建 dense ticks（不变） ----------
#     dense_ticks = []
#     dense_pixels = []
#
#     for divisor in [2,3,4,5,6,8,10]:
#         dense_ticks.clear()
#         dense_pixels.clear()
#
#         for i in range(len(pairs)-1):
#             v1, p1 = pairs[i]
#             v2, p2 = pairs[i+1]
#             for j in range(divisor+1):
#                 a = j/divisor
#                 dense_ticks.append(round(v1 + (v2-v1)*a, 2))
#                 dense_pixels.append(p1 + (p2-p1)*a)
#
#         if sum(left_crop <= p <= right_crop for p in dense_pixels) >= 6:
#             break
#
#     # ---------- 8. 映射到画布坐标（保持不变） ----------
#     x_mapping = {}
#     for v,p in zip(dense_ticks, dense_pixels):
#         if p < left_crop or p > right_crop:
#             continue
#         rel = (p-left_crop)/crop_w
#         x_mapping[v] = offset_x + int(rel * new_w)
#
#     # ---------- 9. 绘制（保持原样） ----------
#     def draw_ticks(side):
#         for v,x_pix in x_mapping.items():
#             if side == "top":
#                 draw.line([(x_pix, crop_top-dash_region-tick_len),
#                            (x_pix, crop_top-dash_region)], fill="black", width=1)
#                 center = (x_pix, crop_top-dash_region-tick_len-25)
#                 draw_rotated_text(canvas, str(v), center, 45, font)
#             else:
#                 draw.line([(x_pix, crop_bottom+dash_region),
#                            (x_pix, crop_bottom+dash_region+tick_len)], fill="black", width=1)
#                 center = (x_pix, crop_bottom+dash_region+tick_len+25)
#                 draw_rotated_text(canvas, str(v), center, -90, font)
#
#     draw_ticks("top")
#     for v,x_pix in x_mapping.items():
#         y = crop_top
#         while y < crop_bottom:
#             draw.line([(x_pix, y), (x_pix, min(y+dash_len, crop_bottom))], fill="gray", width=1)
#             y += dash_len + dash_gap
#     draw_ticks("bottom")
#
#     draw.line([(offset_x, crop_top), (offset_x+new_w, crop_top)], fill=border_color, width=border_width)
#     draw.line([(offset_x, crop_bottom), (offset_x+new_w, crop_bottom)], fill=border_color, width=border_width)
#
#     # ---------- 10. 保存（保持原样） ----------
#     if output_path is None:
#         safe_name = safe_filename(point_name)
#         output_filename = f"amplifier_crop_{safe_name}_round{feedback_round}.png"
#         chart_id = os.path.splitext(os.path.basename(image_path))[0]
#         chart_id = chart_id.replace("_with_grid","").replace("_grid","")
#         output_path = os.path.join("results", chart_id, "tempy", output_filename)
#
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     canvas.save(output_path)
#
#     # ---------- 11. 计算可见区间（保持不变） ----------
#     min_val = v_min + (left_crop - p_min) / scale
#     max_val = v_min + (right_crop - p_min) / scale
#     visible_range = (round(min_val,4), round(max_val,4))
#
#     visible_ticks = [v for v,_x in x_mapping.items()]
#
#     return output_path, visible_ticks, visible_range




async def try_crop_until_bar_detected(
    image_path: str,
    point_name: str,
    y_value: float,   # ⚠️ 实际上传的是 x 数值
    x_label: str,
    x_ticks: list,
    x_pixels: list,
    y_ticks: list,
    y_pixels: list,
    judge_prompt: str,
    out_size: int = 240,
    side_pad: int = 10,
    vert_pad: int = 30,
    max_attempts: int = 10,
    output_path: Optional[str] = None,
    feedback_round: int = 0
) -> Optional[Tuple[str, list, Tuple[float, float]]]:

    # 🟩 打印 judge_prompt 方便调试
    print("\n=== 🧠 当前 judge_prompt 内容 ===")
    print(judge_prompt)
    print("=================================\n")

    # === 1. 数值轴信息（x 轴） ===
    pairs = sorted(zip(x_ticks, x_pixels), key=lambda p: p[0])
    v_min, v_max = pairs[0][0], pairs[-1][0]
    p_min, p_max = pairs[0][1], pairs[-1][1]
    scale = (p_max - p_min) / (v_max - v_min) if (v_max - v_min) != 0 else 1.0

    # 初始中心像素（由预测值映射）
    center_px = p_min + (y_value - v_min) * scale

    tick_span_px = None  # 用窗口宽度动态更新

    print(f"\n=== 🧭 开始检测目标 bar: {point_name} (初始值 {y_value:.4f}) ===")

    # === 2. 多次尝试偏移（从初始位置开始，逐步向左扫） ===
    for i in range(max_attempts + 1):

        if tick_span_px is not None:
            # 之后每次：中心左移一个「窗口宽度」
            shifted_center_px = center_px - i * tick_span_px
        else:
            # 第一次：不偏移
            shifted_center_px = center_px

        # 反推到数值空间
        shifted_val = v_min + (shifted_center_px - p_min) / scale

        # 执行裁剪（✅ 解包三个返回值）
        pred_img_path, visible_ticks, visible_range = crop_segment_with_dual_yaxis_ticks(
            image_path=image_path,
            point_name=point_name,
            y_value=shifted_val,   # 当前中心的 x 数值
            x_label=x_label,
            x_ticks=x_ticks,
            x_pixels=x_pixels,
            y_ticks=y_ticks,
            y_pixels=y_pixels,
            out_size=out_size,
            side_pad=side_pad,
            vert_pad=vert_pad,
            output_path=output_path,
            feedback_round=feedback_round
        )

        min_val, max_val = visible_range

        # === 第一次用当前窗口的值域反推窗口宽度对应的像素步长 ===
        if tick_span_px is None:
            # 窗口值域宽度 * 轴上像素/值 → 窗口像素宽度
            tick_span_px = (max_val - min_val) * scale
            print(f"🪜 动态计算窗口宽度 tick_span_px = {tick_span_px:.2f}px "
                  f"（对应值域跨度 {max_val - min_val:.4f}）")

        # 调 LLM 判断是否包含目标 bar
        exists = await call_llm_bar_existence(judge_prompt, pred_img_path)

        print(f"[尝试 {i:02d}] 偏移: {0 if tick_span_px is None else i * tick_span_px:.1f}px | "
              f"中心值: {shifted_val:.4f} | "
              f"窗口: ({min_val:.4f}, {max_val:.4f}) | "
              f"{'✅ 检测到目标 bar!' if exists else '❌ 未检测到'}")

        if exists:
            print(f"🎯 成功检测到目标 bar 于尝试 {i}（中心 {shifted_val:.4f}）")
            print(f"输出图像: {pred_img_path}\n")
            return pred_img_path, visible_ticks, visible_range

    print("⚠️ 未检测到 bar，已尝试多轮偏移。")
    return None


import aiohttp, base64, json, re, asyncio, os
from typing import Tuple
from datetime import datetime

# ======== 全局控制参数 ======== #

url = "https://api.vveai.com/v1/chat/completions"
# === 实验配置 ===
REPEAT_TIMES = 1
MAX_ATTEMPTS = 10 # 每个点最多尝试10次来获得3次成功预测
SEM_LIMIT = 4
sem = asyncio.Semaphore(SEM_LIMIT)
BASE_TIMEOUT = aiohttp.ClientTimeout(total=180, connect=30, sock_connect=30, sock_read=120)
MAX_RETRIES = 3
_session: aiohttp.ClientSession | None = None

# ======== 多 Key 自动切换配置 ======== #
API_KEYS = [
    "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    "sk-2nzrUYD0JWLFzopWF477111f78E746AbAcA9Ed8534C3A481",
    "sk-CiD5WVUNIkBeXDgYB46b90C06aD24636BcEaBaFa993970C4",
    "sk-WvF4fU10VeOkfFMq579610Fc01E8496d827d0d3e04C44d0a",
    "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb",
]
key_index = 0

def get_headers() -> dict:
    """获取当前 key 对应的 headers"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_index]}"
    }

def rotate_key() -> None:
    """切换到下一个 key"""
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"🔑 已切换至新的 API Key [{key_index + 1}/{len(API_KEYS)}]")


async def call_llm_response(prompt: str, image_path: str, point_name: str) -> Tuple[float, float]:
    """增强稳定版：自动轮换 Key + 多阶段超时 + 安全 JSON 解析"""
    global _session

    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)

    img_size_kb = os.path.getsize(image_path) / 1024
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "gemini-2.0-flash",  # 可切换 gemini-2.5-flash / gpt-4o
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.0
    }

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start_t = datetime.now()
                async with _session.post(url, headers=get_headers(), json=payload) as response:
                    if response.status == 429:
                        print(f"🚫 [{point_name}] 频率超限，切换 Key 重试…")
                        rotate_key(); await asyncio.sleep(3); continue

                    text_buf = None
                    for phase in range(1, 6):
                        try:
                            text_buf = await asyncio.wait_for(response.text(), timeout=30)
                            break
                        except asyncio.TimeoutError:
                            print(f"💤 [{point_name}] 阶段 {phase}/5 超时等待中…")
                            if phase == 5:
                                raise asyncio.TimeoutError
                            await asyncio.sleep(1.5 * phase)

                    if not text_buf:
                        print(f"⚠️ [{point_name}] 阶段性超时，切换 Key。")
                        rotate_key(); continue

                    if response.status != 200:
                        print(f"⚠️ [{point_name}] HTTP {response.status}: {text_buf[:200]}")
                        await asyncio.sleep(2 ** attempt)
                        continue

                    # ===== 统一处理模型返回 ===== #
                    text = text_buf.strip()

                    # ① 优先尝试结构化解析
                    result = safe_json_loads(text)
                    content = None
                    if result and "choices" in result:
                        content = result["choices"][0]["message"]["content"]
                    elif isinstance(result, dict):
                        # 有时直接返回 {"datapoints": {...}}
                        content = json.dumps(result)
                    else:
                        content = text

                    # ② 从 content 中提取 JSON 块
                    json_match = re.search(r"(\{.*\}|\[.*\])", content, re.DOTALL)
                    json_str = json_match.group(1) if json_match else None

                    if not json_str:
                        print(f"⚠️ [{point_name}] 未检测到 JSON，原始输出：{content[:300]}")
                        return (-1, -1)

                    # ③ 容错解析
                    coords_json = safe_json_loads(json_str)
                    if not coords_json:
                        print(f"⚠️ [{point_name}] 无法解析模型返回，原文片段：{content[:300]}")
                        return (-1, -1)

                    # ④ 坐标提取
                    coords = extract_coords(coords_json, point_name)
                    elapsed = datetime.now() - start_t
                    print(f"✅ [{point_name}] 成功解析 | 用时: {elapsed} | 图像 {img_size_kb:.1f} KB")
                    return coords

            except asyncio.TimeoutError:
                print(f"⏳ [{point_name}] 超时（第 {attempt}/{MAX_RETRIES} 次） | {datetime.now() - start_t}")
                rotate_key(); await asyncio.sleep(3 * attempt)
                continue

            except aiohttp.ClientConnectionError as e:
                print(f"🌐 [{point_name}] 网络异常：{e} → session 重建")
                if _session and not _session.closed:
                    await _session.close()
                _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)
                rotate_key(); await asyncio.sleep(5)
                continue

            except Exception as e:
                print(f"❌ [{point_name}] 未知错误：{type(e).__name__} - {e}")
                await asyncio.sleep(2)
                continue

        print(f"❌ [{point_name}] 连续 {MAX_RETRIES} 次失败，放弃。")
        rotate_key()
        return (-1, -1)



async def call_llm_bar_existence(prompt: str, image_path: str) -> bool:
    """判断图中是否存在 bar —— 支持多 Key 自动切换"""
    global _session

    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)

    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": "gemini-2.0-flash", #gemini-2.5-flash-lite; gpt-4o
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.0
    }

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with _session.post(url, headers=get_headers(), json=payload) as response:
                    if response.status == 429:
                        print("🚫 请求频率超限，切换 Key 重试…")
                        rotate_key()
                        await asyncio.sleep(2)
                        continue

                    text = await asyncio.wait_for(response.text(), timeout=60)
                    if response.status != 200:
                        print(f"⚠️ 存在性请求 HTTP {response.status}: {text[:200]}")
                        await asyncio.sleep(2)
                        continue

                    # ---- 解析内容 ----
                    try:
                        result = json.loads(text)
                        content = result["choices"][0]["message"]["content"].strip().lower()

                        # 优先解析 JSON 格式
                        try:
                            json_str = next(s for s in content.splitlines() if s.strip().startswith("{"))
                            parsed = json.loads(json_str)
                            return parsed.get("exists", False)
                        except Exception:
                            pass

                        # 关键词回退判断
                        return "yes" in content and "no" not in content

                    except Exception as e:
                        print(f"⚠️ LLM 判断解析异常：{e}")
                        await asyncio.sleep(2)
                        continue

            except asyncio.TimeoutError:
                print(f"⏳ 存在性判断超时（第 {attempt}/{MAX_RETRIES} 次）")
                rotate_key()
                await asyncio.sleep(2 * attempt)
                continue

            except aiohttp.ClientConnectionError as e:
                print(f"🌐 存在性判断连接异常：{e}，尝试重建 session")
                if _session and not _session.closed:
                    await _session.close()
                _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)
                rotate_key()
                await asyncio.sleep(3)
                continue

        print("❌ 存在性判断连续失败，放弃。")
        rotate_key()
        return False


import json, re

# ======== 容错 JSON 解析 ======== #
# ======== 容错 JSON 解析（增强版） ======== #
def safe_json_loads(s: str):
    """
    尝试从模型输出中解析 JSON，自动清理常见异常：
    - 反引号、```json 包裹
    - 尾随逗号
    - 全角/花括号引号替换
    - Markdown 或解释性语句截断
    - 额外右括号或右花括号
    """
    import re, json

    if not s:
        return None

    s = s.strip()

    # 清除 Markdown 包裹
    s = re.sub(r'^```(?:json)?|```$', '', s, flags=re.I).strip('`')

    # 替换全角引号与中文引号
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # 去掉尾随逗号
    s = re.sub(r',\s*([}\]])', r'\1', s)

    # 去掉模型可能输出的解释前缀
    if not s.startswith("{") and "{" in s:
        s = s.split("{", 1)[-1]
        s = "{" + s

    # ✅ 平衡括号数量：有时模型多输出一个 "}}"
    def balance_brackets(txt):
        while txt.count('{') < txt.count('}'):
            txt = txt.rstrip('}')
        while txt.count('[') < txt.count(']'):
            txt = txt.rstrip(']')
        return txt

    s = balance_brackets(s)

    # 第一次尝试解析
    try:
        return json.loads(s)
    except Exception:
        # 再尝试提取最外层 JSON 结构
        m = re.search(r'(\{.*\})', s, re.DOTALL)
        if m:
            candidate = balance_brackets(m.group(1))
            try:
                return json.loads(candidate)
            except Exception:
                # 最后尝试：截断到最后一个完整右括号
                last_brace = candidate.rfind('}')
                if last_brace != -1:
                    try:
                        return json.loads(candidate[: last_brace + 1])
                    except Exception:
                        pass
    return None


# ======== 坐标提取函数（兼容 dict / list） ======== #
def extract_coords(coords_json, point_name: str):
    """
    从模型返回的 JSON 中提取指定 point_name 的坐标。
    兼容多种结构与异常情况。
    """
    import re

    if coords_json is None:
        print(f"⚠️ [extract_coords] 空 JSON")
        return (-1, -1)

    # 如果顶层是 list，统一包成 dict
    if isinstance(coords_json, list):
        coords_json = {"datapoints": coords_json}

    # 自动推断 datapoints 区域
    if isinstance(coords_json, dict):
        datapoints = coords_json.get("datapoints", coords_json)
    else:
        print(f"⚠️ [extract_coords] 非法结构: {type(coords_json).__name__}")
        return (-1, -1)

    # 标准形式一：dict
    if isinstance(datapoints, dict):
        for k, v in datapoints.items():
            if isinstance(k, str) and point_name.strip().lower() in k.strip().lower():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    # 尝试将数值字符串转为 float
                    try:
                        x = float(v[0]) if isinstance(v[0], str) and re.match(r"^[\d.]+$", v[0]) else v[0]
                    except Exception:
                        x = v[0]
                    return (x, v[1])
        print(f"⚠️ [extract_coords] 未匹配到 '{point_name}'，尝试模糊匹配")

    # 标准形式二：list
    elif isinstance(datapoints, list):
        for item in datapoints:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if isinstance(k, str) and point_name.strip().lower() in k.strip().lower():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        try:
                            x = float(v[0]) if isinstance(v[0], str) and re.match(r"^[\d.]+$", v[0]) else v[0]
                        except Exception:
                            x = v[0]
                        return (x, v[1])

    print(f"⚠️ [extract_coords] 未找到匹配项 | point_name={point_name} | 返回结构={type(coords_json).__name__}")
    return (-1, -1)


# --- 新增：仅保留 grid+with_grid 的最后一轮预测 ---
def filter_final_round_for_feedback(df: pd.DataFrame) -> pd.DataFrame:
    multi_round_prompt_types = ["amplifier", "feedback", "feedback_crop",
                                "feedback_crop_final", "color_feedback"]

    df["round_index"] = df.groupby(
        ["chart_id", "point", "prompt_type", "image_type"]
    ).cumcount()

    mask_multi_feedback = df["prompt_type"].isin(multi_round_prompt_types)

    df["max_round_index"] = df.groupby(
        ["chart_id", "point", "prompt_type", "image_type"]
    )["round_index"].transform("max")

    df_filtered = df[
        ~mask_multi_feedback | (df["round_index"] == df["max_round_index"])
    ].drop(columns=["max_round_index"])

    return df_filtered


def compute_mae(pred: Tuple, gt: Tuple, axis_types: dict) -> Union[float, None]:
    """
    计算 MAE，水平条形图：X 数值轴，Y 分类轴
    pred, gt = (x_val, y_val)
    """
    x_val, y_val = pred
    x_gt, y_gt = gt

    # --- X 数值轴 ---
    if axis_types.get("x") == "numerical":
        if pd.notna(x_val) and pd.notna(x_gt):
            return round(abs(float(x_val) - float(x_gt)), 4)

    # --- Y 分类轴 ---
    if axis_types.get("y") == "categorical":
        return 0.0 if y_val == y_gt else 1.0

    return None


def compute_re(pred: Tuple, gt: Tuple, axis_types: dict) -> Tuple[float, float]:
    """
    计算相对误差：只对数值轴计算
    """
    x_val, y_val = pred
    x_gt, y_gt = gt

    # --- X 数值轴 ---
    if (axis_types.get("x") == "numerical" and
        pd.notna(x_val) and pd.notna(x_gt) and x_gt != 0):
        x_re = abs(float(x_val) - float(x_gt)) / (abs(float(x_gt)) + 1e-6)
    else:
        x_re = -1

    # --- Y 分类轴（不算） ---
    y_re = -1

    return round(x_re, 4), round(y_re, 4)


# ====== 绘图 ======
def evaluate_results(df: pd.DataFrame, result_dir: str = "."):
    os.makedirs(result_dir, exist_ok=True)

    df = filter_final_round_for_feedback(df)

    # ========== 类型转换，确保 pred_x 与 gt_x 可数值运算 ==========
    df["gt_x"] = pd.to_numeric(df["gt_x"], errors="coerce")
    df["pred_x"] = pd.to_numeric(df["pred_x"], errors="coerce")

    # ========== 标注有效性 ==========
    if "pred_x" in df.columns:
        df["x_valid"] = df["pred_x"].apply(lambda x: pd.notna(x))
    else:
        df["x_valid"] = False

    # ========== 每行计算 X 相对误差 ==========
    def compute_x_re(row):
        if row["x_valid"] and row["gt_x"] != 0:
            return abs(row["pred_x"] - row["gt_x"]) / abs(row["gt_x"])
        return None

    df["x_re"] = df.apply(compute_x_re, axis=1)

    # ========== 汇总 X 轴指标 ==========
    summary = df.groupby(["prompt_type", "image_type"]).apply(lambda group: pd.Series({
        "avg_x_mae": (group[group["x_valid"]]["pred_x"] - group[group["x_valid"]]["gt_x"]).abs().mean(),
        "avg_x_re": group[group["x_valid"]]["x_re"].mean(),
        "valid_x_count": group["x_valid"].sum(),
        "avg_error_over_vrange": group["error_over_vrange"].mean(),

    })).reset_index()

    summary_path = os.path.join(result_dir, "axis_level_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("📄 已保存 X 轴评估表 axis_level_summary.csv")

    # ========== 保存完整记录 ==========
    df.to_csv(os.path.join(result_dir, "full_results_with_xre.csv"), index=False)
    print("📄 已保存完整记录表 full_results_with_xre.csv")

    # ========== 合并绘制 MAE、相对误差 与 error_over_vrange ==========
    x = range(len(summary))
    labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
    bar_width = 0.4

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # —— X轴 MAE —— (左轴)
    bars1 = ax1.bar(x, summary["avg_x_mae"], width=bar_width, label="X MAE", color="#76D7C4")
    ax1.set_ylabel("MAE (X axis)")
    ax1.set_xlabel("Prompt + Image Setting")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20)
    ax1.tick_params(axis='y')

    # 在柱子上标注 MAE
    for i, mae in enumerate(summary["avg_x_mae"]):
        if not pd.isna(mae):
            ax1.text(i, mae + 0.02, f"{mae:.2f}", ha='center', va='bottom', fontsize=9)

    # —— X轴相对误差 与 MAE/v_range —— (右轴)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["avg_x_re"], label="X Relative Error", color="#F1948A", marker="o", linewidth=2)
    ax2.plot(x, summary["avg_error_over_vrange"], label="MAE / v-range", color="#9B59B6", marker="s", linewidth=2,
             linestyle="--")

    ax2.set_ylabel("Relative / Normalized Error (X axis)")
    ax2.tick_params(axis='y')

    # 在点上标注 RE
    for i, re in enumerate(summary["avg_x_re"]):
        if not pd.isna(re):
            ax2.text(i, re + 0.01, f"{re:.2f}", ha='center', va='bottom', fontsize=9, color="#F1948A")

    # 在点上标注 MAE/v-range
    for i, val in enumerate(summary["avg_error_over_vrange"]):
        if not pd.isna(val):
            ax2.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=9, color="#9B59B6")

    # —— 图例合并 ——
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")

    plt.title("X Axis MAE, Relative Error, and Normalized MAE (MAE/v-range)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "axis_level_mae_re_vrange_combined.png"))
    print("📊 已保存合并图 axis_level_mae_re_vrange_combined.png")
    plt.close()

    # # ========== 合并绘制 MAE & 相对误差 ==========
    # x = range(len(summary))
    # labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
    # bar_width = 0.4
    #
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    #
    # # —— X轴 MAE —— (左轴)
    # bars1 = ax1.bar(x, summary["avg_x_mae"], width=bar_width, label="X MAE", color="#76D7C4")
    # ax1.set_ylabel("MAE (X axis)")
    # ax1.set_xlabel("Prompt + Image Setting")
    # ax1.set_xticks(list(x))
    # ax1.set_xticklabels(labels, rotation=20)
    # ax1.tick_params(axis='y')
    #
    # # 在柱子上标注 MAE
    # for i, mae in enumerate(summary["avg_x_mae"]):
    #     if not pd.isna(mae):
    #         ax1.text(i, mae + 0.02, f"{mae:.2f}", ha='center', va='bottom', fontsize=9)
    #
    # # —— X轴相对误差 —— (右轴)
    # ax2 = ax1.twinx()
    # ax2.plot(x, summary["avg_x_re"], label="X Relative Error", color="#F1948A", marker="o", linewidth=2)
    # ax2.set_ylabel("Relative Error (X axis)")
    # ax2.tick_params(axis='y')
    #
    # # 在点上标注 RE
    # for i, re in enumerate(summary["avg_x_re"]):
    #     if not pd.isna(re):
    #         ax2.text(i, re + 0.01, f"{re:.2f}", ha='center', va='bottom', fontsize=9, color="#F1948A")
    #
    # # —— 图例合并 ——
    # lines, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")
    #
    # plt.title("X Axis MAE & Relative Error by Prompt+Image Setting")
    # plt.tight_layout()
    # plt.savefig(os.path.join(result_dir, "axis_level_mae_re_combined.png"))
    # print("📊 已保存合并图 axis_level_mae_re_combined.png")
    # plt.close()


# -*- coding: utf-8 -*-
"""
run_experiment_batch.py  ——  完整批量异步实验脚本（保留全部业务逻辑）
-------------------------------------------------------------
* 支持 --batch-size 控制并发批量大小
* 支持 --chart-ids 选择性运行指定图表
* 结果按 chart_id 拆分子目录保存，并调用 evaluate_results()
* 完全沿用原有 FEEDBACK / 裁剪 / adaptive‑crop 等内部实现，不删除任何细节
"""

def build_feedback_final_preds(records, axis_types: dict = None):
    """
    从所有 records 中提取 feedback 阶段的最后一轮预测结果，
    并根据数值轴 (x 或 y) 自动取出对应的数值存储到 pred_y 字段。

    返回结构：
    {
      (chart_id, point_name): {"pred_y": float数值}
    }
    """
    result = {}
    df = pd.DataFrame(records)
    feedback_df = df[df["prompt_type"] == "feedback"]

    if axis_types is None:
        # 默认竖直柱状图 (y 为数值轴)
        axis_types = {"x": "categorical", "y": "numerical"}

    for (chart_id, point_name), gdf in feedback_df.groupby(["chart_id", "point"]):
        last_row = gdf.sort_values("run").iloc[-1]

        # 根据 axis_types 确定数值轴
        if axis_types.get("y") == "numerical":
            num_val = last_row["pred_y"]
        elif axis_types.get("x") == "numerical":
            num_val = last_row["pred_x"]
        else:
            print(f"⚠️ 未知轴配置，跳过 {chart_id}-{point_name}")
            continue

        if isinstance(num_val, (int, float)) and not pd.isna(num_val):
            result[(chart_id, point_name)] = {"pred_y": num_val}

    return result


# ----------------------------------------------------------------------------------
# 主实验函数
# ----------------------------------------------------------------------------------
async def run_experiment(batch_size: int | None = None,
                         chart_ids: List[str] | None = None):
    """
    批量实验入口

    :param batch_size:  每批次并发处理的数据集数量；None 表示一次性处理全部
    :param chart_ids:   仅处理指定的 chart_id 列表；None 表示全部
    """
    # ----------------------- 0) 数据集过滤 -----------------------
    datasets = DATASET_CONFIGS.copy()
    if chart_ids:
        datasets = [ds for ds in datasets if ds["chart_id"] in chart_ids]

    if not datasets:
        print("⚠️  未找到符合条件的 chart 配置，退出。")
        return

    feedback_final_results: Dict[tuple[str, str], Tuple] = {}
    records: List[Dict] = []
    feedback_final_preds: Dict[tuple, dict] = {}  # ✅ 在 run_batch 定义之前声明

    # ----------------------- 1) 单数据集执行 ----------------------
    async def run_for_dataset(dataset, feedback_final_preds, run_amplifier=True):
        nonlocal records, feedback_final_results
        axis_types = {"x": "numerical","y": "categorical"}  # 全局设定

        pred_coords = []
        for group_label, sub_points in dataset["data_points"].items():
            # ✅ 【在这里加】v_range 计算逻辑
            if "x_ticks" in dataset and dataset["x_ticks"]:
                v_min, v_max = min(dataset["x_ticks"]), max(dataset["x_ticks"])
                v_range = float(v_max - v_min) if v_max != v_min else None
            else:
                all_vals = [v for grp in dataset["data_points"].values() for v in grp.values()]
                v_range = float(max(all_vals) - min(all_vals)) if all_vals else None

            for sub_label, x_value in sub_points.items():  # ⚠️ y_value 改成 x_value，更符合实际
                point_name = f"{group_label}, {sub_label}"  # y 轴分类标签
                gt = (x_value, sub_label)  # ✅ gt: (数值x, 分类y)
                pred_coords.append((point_name, x_value))  # coord[0] 是分类名，coord[1] 是数值

                FEEDBACK_START_ROUND = 2
                MAX_FEEDBACK_POINTS = 3

                for prompt_type, image_type in EXPERIMENT_TYPES:
                    # Phase 2 时仅跑 amplifier；Phase 1 时不跑 amplifier
                    if run_amplifier:
                        if prompt_type != "amplifier":
                            continue
                    else:
                        if prompt_type == "amplifier":
                            continue

                    valid_runs = 0
                    total_attempts = 0
                    image_path = dataset["image_paths"][image_type]
                    history_preds: List[Tuple] = []
                    last_pred: Tuple | None = None

                    while valid_runs < REPEAT_TIMES:
                        pred_img_path = image_path  # 默认使用原图

                        if prompt_type == "amplifier":
                            current_round = valid_runs + 1

                            if current_round == 1:
                                feedback_key = (dataset["chart_id"], point_name)
                                feedback_pred = feedback_final_preds.get(feedback_key)

                                if feedback_pred is None or not isinstance(feedback_pred["pred_y"], (int, float)):
                                    print(
                                        f"⚠️ Skipping amplifier round 1: no valid feedback prediction for {point_name}")
                                    continue

                                x_label = point_name.split(",")[-1].strip()
                                y_value = feedback_pred["pred_y"]

                            else:
                                if last_pred is None or not isinstance(last_pred[0], (int, float)):
                                    print(
                                        f"⚠️ Skipping amplifier round {current_round}: no valid last prediction for {point_name}")
                                    continue

                                x_label = point_name.split(",")[-1].strip()
                                y_value = last_pred[0]  # ✅ 改这里，用 x 轴数值

                            # 不需要自己拼 tempy 路径了，直接交给函数返回完整路径
                            res = await try_crop_until_bar_detected(
                                image_path=dataset["image_paths"]["grid_with_grid"],
                                point_name=point_name,
                                y_value=y_value,
                                x_label=x_label,
                                x_ticks=dataset["x_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_ticks=dataset["y_ticks"],
                                y_pixels=dataset["y_pixels"],
                                judge_prompt=build_color_prompt(
                                    point_name=point_name,
                                    series_color=dataset["series_color"],
                                ),
                                feedback_round=valid_runs + 1  # ✅ 传入
                            )

                            if res is None:
                                print("跳过该预测点")
                            else:
                                pred_img_path, visible_ticks, visible_range = res  # ✅ 返回的就是 results/chart_id/tempy/... 路径

                        if (
                                prompt_type == "feedback"
                                and last_pred is not None
                                and valid_runs + 1 >= FEEDBACK_START_ROUND
                        ):
                            safe_point_name = safe_filename(point_name)   # ✅ 修复

                            pred_img_path = generate_overlayed_image_multi_with_mapping(
                                original_img_path=dataset["image_paths"]["grid_with_grid"],
                                pred_coords=history_preds,
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                output_path=f"overlay_{dataset['chart_id']}_{safe_point_name}_"
                                            f"{prompt_type}_{image_type}_run{valid_runs + 1}.png",  # 只传文件名
                                feedback_round=valid_runs + 1,
                                draw_all_preds=False,
                                axis_types={"x": "numerical", "y": "categorical"},
                            )


                        prompt = generate_prompt(
                            item_name=point_name,
                            prompt_type=prompt_type,
                            x_ticks=dataset["x_ticks"],
                            y_ticks=dataset["y_ticks"],
                            series_color=dataset["series_color"],
                            axis_types={"x": "numerical", "y": "categorical"},
                            pred_feedback=history_preds[-2:] if len(history_preds) >= 1 else None,
                            feedback_round=FEEDBACK_START_ROUND,
                            current_round=valid_runs + 1,
                            visible_ticks=visible_ticks if prompt_type == "amplifier" else None,
                        )

                        print("\n==============================")
                        print(f"📌 Round {valid_runs + 1} | Point: {point_name} | "
                              f"Type: {prompt_type} - {image_type}")
                        print(f"🖼️  使用图像路径: {pred_img_path}")
                        print("📋 Prompt 内容如下：\n")
                        print(prompt)
                        print("==============================\n")

                        pred = await call_llm_response(prompt, pred_img_path, point_name)

                        if pred == (-1, -1):
                            print(f"❌ 第 {total_attempts + 1} 次预测失败 "
                                  f"[{prompt_type} - {image_type}] @ {point_name}")
                            total_attempts += 1
                            await asyncio.sleep(7)
                            if total_attempts >= 47:
                                print("⚠️ 尝试上限，跳过该点。")
                                break
                            continue

                        x_val, y_val = pred

                        # 按照数值轴来判断有效性
                        if axis_types["x"] == "numerical":
                            valid_numeric = pd.notna(x_val)
                        elif axis_types["y"] == "numerical":
                            valid_numeric = pd.notna(y_val)
                        else:
                            valid_numeric = False

                        if valid_numeric:
                            last_pred = pred
                            history_preds.append(pred)
                            mae = compute_mae(pred, gt, axis_types)
                            x_re, y_re = compute_re(pred, gt, axis_types)
                        else:
                            print(f"⚠️ 无效预测 (非数值轴): {pred}")
                            mae, x_re, y_re = None, -1, -1

                        # x_val, y_val = pred
                        # last_pred = pred  # 保持 (x_val, y_val)
                        # history_preds.append(pred)
                        #
                        # if pd.notna(y_val):  # 数值轴是 y，直接判断
                        #     mae = compute_mae(pred, gt, axis_types)
                        #     x_re, y_re = compute_re(pred, gt, axis_types)
                        # else:
                        #     mae, x_re, y_re = None, -1, -1

                        img_w, img_h = Image.open(image_path).size
                        x_map = get_axis_mapper(dataset["x_ticks"], dataset["x_pixels"], "numerical")
                        y_map = get_axis_mapper(dataset["y_ticks"], dataset["y_pixels"], "categorical")

                        try:
                            px_rel_x = (abs(x_map(pred[0]) - x_map(gt[0])) / img_w if is_number(pred[0]) else -1)
                            px_rel_y = (abs(y_map(pred[1]) - y_map(gt[1])) / img_h if is_number(pred[1]) else -1)
                        except Exception as e:
                            print(f"⚠️ Pixel mapping error: {e}")
                            px_rel_x = px_rel_y = -1

                        if mae is not None and v_range not in (None, 0):
                            error_over_vrange = mae / v_range
                        else:
                            error_over_vrange = None

                        records.append({
                            "chart_id": dataset["chart_id"],
                            "point": point_name,
                            "prompt_type": prompt_type,
                            "image_type": image_type,
                            "run": valid_runs + 1,
                            "image_path": pred_img_path,
                            "gt_x": gt[0],
                            "gt_y": gt[1],
                            "pred_x": x_val,
                            "pred_y": y_val,
                            "mae": mae,
                            "error_over_vrange": error_over_vrange,  # ✅ 新增
                            "pixel_rel_x": px_rel_x,
                            "pixel_rel_y": px_rel_y,
                            "x_re": x_re,
                            "y_re": y_re,
                        })

                        # records.append({
                        #     "chart_id": dataset["chart_id"],
                        #     "point": point_name,
                        #     "prompt_type": prompt_type,
                        #     "image_type": image_type,
                        #     "run": valid_runs + 1,
                        #     "image_path": pred_img_path,
                        #     "gt_x": gt[0],
                        #     "gt_y": gt[1],
                        #     "pred_x": x_val,  # 即使是分类，也存下来
                        #     "pred_y": y_val,
                        #     "mae": mae,
                        #     "pixel_rel_x": px_rel_x,
                        #     "pixel_rel_y": px_rel_y,
                        #     "x_re": x_re,
                        #     "y_re": y_re,
                        # })

                        valid_runs += 1
                        total_attempts += 1
                        print(f"✅ 成功 {valid_runs}/{REPEAT_TIMES} [{prompt_type} - {image_type}] @ {point_name}")

                        if prompt_type == "feedback" and history_preds:
                            feedback_final_results[(dataset["chart_id"], point_name)] = history_preds[-1]
                            safe_name = safe_filename(point_name)   # ✅ 修复

                            final_img_path = generate_overlayed_image_multi_with_mapping(
                                original_img_path=dataset["image_paths"]["grid_with_grid"],
                                pred_coords=history_preds,
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                output_path=f"final_overlay_{dataset['chart_id']}_{safe_name}_"
                                            f"{prompt_type}_{image_type}.png",  # 只传文件名
                                feedback_round=valid_runs,
                                draw_all_preds=True,
                                axis_types={"x": "numerical", "y": "categorical"},
                            )

                            # final_img_path = (
                            #     f"tempy/final_overlay_{dataset['chart_id']}_{safe_name}_"
                            #     f"{prompt_type}_{image_type}.png"
                            # )
                            os.makedirs("tempy", exist_ok=True)
                            generate_overlayed_image_multi_with_mapping(
                                original_img_path=dataset["image_paths"]["grid_with_grid"],
                                pred_coords=history_preds,
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                output_path=final_img_path,
                                feedback_round=valid_runs,
                                draw_all_preds=True,
                                axis_types={"x": "numerical", "y": "categorical"},
                            )

    # ----------------------- 2) 分批并发调度 ---------------------
    records: List[Dict] = []
    feedback_final_results: Dict[tuple[str, str], Tuple] = {}

    # 初始为空（第一阶段 amplifier 不使用）
    feedback_final_preds: Dict[tuple, dict] = {}

    async def run_batch(batch_ds: List[Dict], feedback_final_preds, run_amplifier=True):
        await asyncio.gather(*[run_for_dataset(ds, feedback_final_preds, run_amplifier) for ds in batch_ds])

    print("🚀 Phase 1: Running all prompts except amplifier...")
    if batch_size and batch_size > 0:
        for bi in range((len(datasets) + batch_size - 1) // batch_size):
            s, e = bi * batch_size, (bi + 1) * batch_size
            await run_batch(datasets[s:e], feedback_final_preds, run_amplifier=False)
    else:
        await run_batch(datasets, feedback_final_preds, run_amplifier=False)

    feedback_final_preds = build_feedback_final_preds(records, axis_types={"x": "numerical", "y": "categorical"})

    print("🚀 Phase 2: Running amplifier only...")
    if batch_size and batch_size > 0:
        for bi in range((len(datasets) + batch_size - 1) // batch_size):
            s, e = bi * batch_size, (bi + 1) * batch_size
            await run_batch(datasets[s:e], feedback_final_preds, run_amplifier=True)
    else:
        await run_batch(datasets, feedback_final_preds, run_amplifier=True)

    # ----------------------- 3) 结果落盘 ------------------------
    if not records:
        print("⚠️ 无实验记录生成，结束。")
        return

    df = pd.DataFrame(records)

    for chart_id, gdf in df.groupby("chart_id"):
        result_dir = os.path.join("results", chart_id)
        os.makedirs(result_dir, exist_ok=True)

        csv_path = os.path.join(result_dir, "experiment_results.csv")
        gdf.to_csv(csv_path, index=False)
        print(f"✅ 保存 {chart_id} 结果 → {csv_path}")

        final_df = filter_final_round_for_feedback(gdf)
        evaluate_results(final_df, result_dir=result_dir)
        print(f"📊 {chart_id} 评估图生成完毕")

    print("🎉 全部批量实验完成。")

# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图表数据批量处理工具")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="每批并发处理的数据集数量")
    parser.add_argument("--chart-ids", nargs="+", default=None,
                        help="仅处理指定 chart_id，如 --chart-ids chart04 chart11")
    args = parser.parse_args()

    asyncio.run(run_experiment(
        batch_size=args.batch_size,
        # chart_ids=["h_bar_001"]
        chart_ids=args.chart_ids,
    ))



