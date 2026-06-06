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


# ========== 配置项 ========== #
api_key = "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d"
url = "https://api.vveai.com/v1/chat/completions"
# url = "https://api.v3.cm"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# === 实验配置 ===
REPEAT_TIMES = 1
MAX_ATTEMPTS = 10  # 每个点最多尝试10次来获得3次成功预测

EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("amplifier", "grid_with_grid"), # ✅ 新增
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



# 区分轴的类型为类别型还是数值型
def build_axis_mapping(tick_values, tick_pixels):
    """
    数值型坐标值的插值映射（连续坐标轴）
    """
    return lambda v: np.interp(v, tick_values, tick_pixels)

# ========= New helper: 像素跨度推断 =========
def get_category_span(label: str,
                      tick_labels: list,
                      tick_pixels: list[int],
                      img_min: int,
                      img_max: int,
                      mode: str = "center") -> tuple[int, int]:
    """
    给定某个分类标签及其在图像上的 tick 像素位置，返回该分类项在图像上的像素范围（起止边界）。
    自动处理像素从上往下（升序）或从下往上（降序）两种情况，并返回有序的 (start, end)。
    """
    # 预处理：如果 label 是拼接形式（如 'A, B'），取最后一个
    if "," in label:
        parts = [p.strip() for p in label.split(",")]
        label = parts[-1]  # ✅ 使用最后一段


    if label not in tick_labels:
        raise ValueError(f"Label '{label}' not found in tick_labels.")

    print(f"🧪 自动提取最后一段作为 label: {label}")
    idx = tick_labels.index(label)

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

def get_axis_mapper(ticks, pixels, axis_type):
    """
    根据轴类型（numerical 或 categorical）返回合适的映射函数
    """
    if axis_type == "categorical":
        return build_categorical_axis_mapping_fuzzy(ticks, pixels)
    else:
        return build_axis_mapping(ticks, pixels)

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
    color_desc = generate_series_color_description(series_color)
    return (
        f"You are given a cropped bar chart image for {point_name}. "
        f"Please check if there is the correct color bar segment for {point_name} visible that corresponds to the color alignment. **{color_desc}**. "
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )



def generate_prompt(
    item_name: str,
    prompt_type: str,
    x_ticks: list,
    y_ticks: list,
    series_color: dict[str, str],  # ⬅️ 新增参数：series_name -> color（如 "Apple": "red"）
    visible_ticks: list,
    axis_types: dict,
    pred_feedback: list = None,
    feedback_round: int = 0,
    current_round: int = 1
) -> str:
    x_type = axis_types.get("x", "numerical")
    y_type = axis_types.get("y", "numerical")

    x_label = item_name.split(",")[-1].strip() if x_type == "categorical" else None
    y_label = item_name.split(",")[0].strip() if y_type == "categorical" and "," in item_name else None

    x_tick_str = ", ".join(str(x) for x in x_ticks)
    y_tick_str = ", ".join(str(y) for y in y_ticks)

    try:
        series_name,x_label = item_name.rsplit(",", 1)
        x_label = x_label.strip()
        series_name = series_name.strip()
    except ValueError:
        raise ValueError(f"⚠️ item_name 解析失败：'{item_name}' 不包含两个部分。请确保格式为 'group_label, sub_label'")

    color_desc = generate_series_color_description(series_color)

    # try:
    #     x_label, series_name = item_name.rsplit(",", 1)
    #     x_label = x_label.strip()
    #     series_name = series_name.strip()
    # except ValueError:
    #     raise ValueError(f"⚠️ item_name 解析失败：'{item_name}' 不包含两个部分。请确保格式为 'group_label, sub_label'")

    # # ===== 特例处理：amplifier 第一轮强制使用 baseline 提示模版 =====
    # if prompt_type == "amplifier" and current_round == 1:
    #     print(f"🌀 第一次 amplifier round（当前轮 {current_round}），使用 baseline prompt 模板：{item_name}")
    #     prompt_type = "baseline"
    # else:
    #     prompt_type = prompt_type

    # === 1. 基础结构 ===
    if prompt_type == "baseline":
    # if prompt_type in ["baseline","amplifier"]:
        base_prompt = f"You are given a bar chart image. Your task is to predict the y coordinate for [{item_name}]. {color_desc}\n"

    elif prompt_type == "amplifier":
        # 拆解标签
        # series_name, y_label = item_name.split(",")
        # y_label = x_label.strip()
        # series_name = series_name.strip()

        x_label, series_name = item_name.split(",")
        x_label = x_label.strip()
        series_name = series_name.strip()

        # x_tick_str = ", ".join(str(x) for x in x_ticks)
        # x_tick_str = visible_ticks
        # 🛠️ 如果是嵌套列表，则展开
        if len(visible_ticks) > 0 and isinstance(visible_ticks[0], list):
            visible_ticks = visible_ticks[0]

        # ✅ 去重 + 排序 + 格式化为字符串
        x_tick_str = ", ".join(str(round(x, 2)) for x in sorted(set(visible_ticks)))

        base_prompt = f'''
        You are given a chart image. Your task is to predict the y coordinate for the segment labeled [{item_name}].
        The segment appears on the **center**, extracted from the full chart by locating the category label **"{x_label}"** on the x-axis.
        The left and right sides include a **vertically drawn y-axis**, with tick values [{x_tick_str}] and grid lines.
        Your task is to estimate the **y coordinate** corresponding to the **top boundary** of the colored segment.
        The segment color indicates its category: use alignment between the legend and segment to verify the target. {color_desc}
        Instructions:
            Your should predict the y coordinate for [{item_name}] by:
            - First, locate the y-axis tick interval in which the segment’s top boundary falls. 
            - Then, determine the relative position of the boundary within this interval. Use linear interpolation between the two tick values to estimate the precise y-axis value of the top boundary.
            *Important:** 
            - *After identifying the two horizontal reference lines that enclose the segment’s top boundary, you should compute that boundary’s proportional vertical distance between the two lines accurately as much as possible and **report the exact interpolated y-value**, so that the predicted value can be accurately aligned with the true value. Do **not !** snap or round this value to the nearest tick (e.g., multiples of 5 or 10). 
            - *Example: If the boundary is 63 % of the distance from 100 to 200, output **163**.
        
        - Follow the instructions above to predict the y coordinate for [{item_name}], which corresponds to the top boundary y-axis value of the segment.
        '''

    else:  # "grid" or "feedback"
        base_prompt = f'''
        You are analyzing a bar chart that contains **reference grid lines**, where horizontal lines correspond to y-axis ticks, and vertical lines align with x-axis ticks.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        {color_desc}
        '''



    if prompt_type in ["grid", "feedback"]:
        if x_type == "categorical":
            base_prompt += (f'Once the correct segment for [{item_name}] is located, determine the y position of its top edge, that is, the top boundary of the stacked bar segment by\n'
                            f'comparing it with the nearest Y-axis tick marks and their corresponding horizontal grid lines.\n'
                            f'Then, interpolate between the known Y-axis values using its relative vertical distance within the interval to estimate the precise Y coordinate.\n')
        if y_type == "categorical":
            base_prompt += f"After locating the correct segment for [{item_name}], identify the position of its right edge by comparing it with the two nearest vertical grid lines on the X-axis. Use linear interpolation between these two ticks to estimate the accurate X-coordinate of [{item_name}].\n"


    # === 3. 若是反馈类型则追加预测历史 ===
    if prompt_type == "feedback" and pred_feedback and current_round >= feedback_round:
        if isinstance(pred_feedback, list):
                pred = pred_feedback[-1]
                x = f'{pred[0]:.2f}' if isinstance(pred[0], (int, float)) else f'"{pred[0]}"'
                y = f'{pred[1]:.2f}' if isinstance(pred[1], (int, float)) else f'"{pred[1]}"'
        base_prompt = f'''
        You are analyzing a staked bar chart that contains **reference grid lines**, where horizontal lines correspond to y-axis ticks, and vertical lines align with x-axis ticks.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        {color_desc}        
        
        The given chart shows your previous prediction for the y coordinate of [{item_name}], aligned with a red crosshair at (x = {x}, y = {y}). 
        Compare this red crosshair to the true top boundary position of [{item_name}]: determine whether it is on the top, below, or exactly aligned with the actual top boundary of the segment. 
        To be clear, the color of the segment indicates its category for [{item_name}]: use alignment between the legend and segment to verify the target.
        After verifying the alignment, correct the offset direction of the red crosshair to refine your prediction accordingly.
        Use the red crosshair of the feedback marker as a reference to adjust the offset between its predicted position and the true position, so that the updated visual feedback horizontal lines align precisely — indicating an accurate prediction.
        Do not just read the value of the reference line as the true position, but also verify its position relative to the other tick marks and the corresponding horizontal grid lines.
        '''

    # === 4. JSON 输出格式 ===
    if x_type == "categorical" and y_type != "categorical":
        base_prompt += f'''
        Only respond in this JSON format:
        {{"datapoints": [{{"{item_name}": ["{x_label}", y]}}]}}
        '''
    elif y_type == "categorical" and x_type != "categorical":
        base_prompt += f'''
        Only respond in this JSON format, in which the x value must be a numeric value only, without any additional text, units, or symbols:
        {{"datapoints": [{{"{item_name}": [x, "{y_label}"]}}]}}
        '''
    elif x_type == "categorical" and y_type == "categorical":
        base_prompt += f'''
        Only respond in this JSON format:
        {{"datapoints": [{{"{item_name}": ["{x_label}", "{y_label}"]}}]}}
        '''
    else:
        base_prompt += f'''
        Only respond in this JSON format:
        {{"datapoints": [{{"{item_name}": [x, y]}}]}}
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
        axis_types = {"x": "numerical", "y": "numerical"}

    x_type = axis_types.get("x", "numerical")
    y_type = axis_types.get("y", "numerical")

    # --- 像素映射器 ---
    x_mapper = get_axis_mapper(x_ticks, x_pixels, "categorical")
    y_mapper = get_axis_mapper(y_ticks, y_pixels, "numerical")

    # --- 载入原图 ---
    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    # --- 样式 ---
    radius = 5                       # 仅用于放置文本时的偏移
    colors = ["red", "purple", "orange", "green", "blue",
              "black", "brown", "pink", "gray", "cyan"]

    # --- 只取最近一次预测（除非要求画全部） ---
    coords_to_draw = pred_coords if draw_all_preds else pred_coords[-1:]

    # --- 开始绘制 ---
    for idx, coord in enumerate(coords_to_draw):
        # ✅ 提取真正的 x label（sub_label）
        # x_label_full = str(coord[1])
        # print(f'这是用于匹配的标签内容：{coord}')
        print(f'这是用于匹配的标签内容：{coord[0]}')
        # print(f'这是用于匹配的标签内容：{coord[1]}')
        true_x_label = coord[0].split(",")[-1].strip()
        print(f'真正的 x 轴标签内容：{true_x_label}')


        try:
            if x_type == "numerical" and isinstance(true_x_label, str):
                raise ValueError(f"x轴为numerical，但坐标为str: {true_x_label}")
            if y_type == "numerical" and isinstance(coord[1], str):
                raise ValueError(f"y轴为numerical，但坐标为str: {coord[1]}")

            y_pixel = int(y_mapper(coord[1]))
            x_pixel = int(x_mapper(true_x_label))
        except Exception as e:
            print(f"❌ 坐标映射失败: {coord} | {e}")
            continue

        color = colors[idx % len(colors)]


        # ---------- 参考线截断范围 ----------
        # 水平线端点
        if y_type == "numerical":
            if x_type == "categorical":
                left, right = get_category_span(true_x_label, x_ticks, x_pixels,
                                                img_min=0, img_max=img_w)
            else:
                left, right = 0, img_w
        # 垂直线端点
        if x_type == "numerical":
            if y_type == "categorical":
                top, bottom = get_category_span(coord[1], y_ticks, y_pixels,
                                                img_min=0, img_max=img_h)
            else:
                top, bottom = 0, img_h
        # -----------------------------------

        # ===== 水平实线 =====
        if y_type == "numerical":
            draw.line((left, y_pixel, right, y_pixel), fill=color, width=1)

            # ===== 垂直实线（与水平线同长度） =====
            top_v = y_pixel - (right - left) // 2
            bottom_v = y_pixel + (right - left) // 2
            top_v = max(0, top_v)
            bottom_v = min(img_h, bottom_v)
            draw.line((x_pixel, top_v, x_pixel, bottom_v), fill=color, width=1)


            y_val = coord[1]
            y_text = f"{y_val:.1f}" if isinstance(y_val, Number) else str(y_val)
            # draw.text((left + 2, y_pixel - 8), y_text, fill=color)
            from PIL import ImageFont

            font = ImageFont.load_default()
            bbox = font.getbbox(y_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x_text = left - text_width - 4
            y_text_pos = y_pixel - text_height // 2
            draw.text((x_text, y_text_pos), y_text, fill=color, font=font)

        # 若 x 轴为数值轴，也保留垂直实线（整幅高）
        if x_type == "numerical":
            draw.line((x_pixel, 0, x_pixel, img_h), fill=color, width=1)

        # # ===== 点编号文本 =====
        # draw.text((x_pixel + radius + 2, y_pixel - radius),
        #           f"P{label_index}", fill=color)

    # --- 保存 ---
    output_filename = os.path.basename(output_path)  # ✅ 取文件名
    chart_id = os.path.splitext(os.path.basename(original_img_path))[0]
    chart_id = chart_id.replace("_with_grid", "").replace("_grid", "").replace("_clean", "")
    output_path = os.path.join("results", chart_id, "feedback", output_filename)  # ✅ 拼接到 results/chart_id/tempy

    output_path = sanitize_filename(output_path)
    folder = os.path.dirname(output_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    img.save(output_path)
    print(f"✅ 反馈图已保存至: {output_path}")
    return output_path  # ✅ 新增：返回完整路径

    # # --- 保存 ---
    # output_filename = os.path.basename(output_path)  # ✅ 取文件名
    # chart_id = os.path.splitext(os.path.basename(original_img_path))[0]  # ✅ 图表名称
    # output_path = os.path.join("results", chart_id, "tempy", output_filename)  # ✅ 拼接到 results/chart_id/tempy
    #
    # output_path = sanitize_filename(output_path)
    # folder = os.path.dirname(output_path)
    # if folder:
    #     os.makedirs(folder, exist_ok=True)
    #
    # img.save(output_path)
    # print(f"✅ 反馈图已保存至: {output_path}")
    # return output_path  # ✅ 新增：返回完整路径

# def crop_segment_with_dual_yaxis_ticks(
#     image_path: str,
#     point_name: str,        # x 为分类轴的分类标签
#     y_value: float,         # 纵向条：y 为数值轴（中心值）
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
#     output_path: str | None = None,
#     enforce_min_vertical_coverage: bool = False,
#     min_canvas_height_ratio: float = 0.5,
#     round_idx: int = 1,            # progressive zoom 轮次（从 1 开始更直观）
#     return_meta: bool = False,
#     value_half_span: float | None = None,
#     target_grid_step_px: int = 50,
#     clamp_grid_px: tuple[int, int] = (40, 60),
#     auto_span: bool = False,
#     span_min: float | None = None,
#     span_max: float | None = None,
#     m_max: int = 10,
#     stabilize_band: tuple[int, int] = (45, 55)
# ):
#     """
#     ✅竖向 bar chart 的最终稳定版（完全可替换原函数）
#     - progressive zoom
#     - canvas 自动扩宽
#     - dense tick 去重（value 3 位小数）
#     - orig_px 限制在裁剪范围
#     - 不修改绘图风格
#     """
#
#     from PIL import Image, ImageDraw, ImageFont
#     import os, math, re
#
#     # ------------------------------------------------------------
#     # Formatting & utilities
#     # ------------------------------------------------------------
#     def _format_tick_val(v: float) -> str:
#         fv = float(v)
#         if fv.is_integer():
#             return str(int(fv))
#         s = str(round(fv, 3))
#         if "." in s:
#             s = s.rstrip("0").rstrip(".")
#         return s
#
#     # ---------------- safe filename ----------------
#     if "safe_filename" in globals():
#         safe_filename = globals()["safe_filename"]
#     else:
#         def safe_filename(name: str):
#             return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]
#
#     if "sanitize_filename" in globals():
#         sanitize_filename = globals()["sanitize_filename"]
#     else:
#         def sanitize_filename(path: str):
#             return re.sub(r'[<>:"/\\|?*]+', "_", path)
#
#     # ------------------------------------------------------------
#     # helpers
#     # ------------------------------------------------------------
#     def _median_diff(vals):
#         diffs = sorted(abs(b - a) for a, b in zip(vals[:-1], vals[1:]) if b != a)
#         if not diffs:
#             return None
#         m = len(diffs) // 2
#         return diffs[m] if len(diffs) % 2 else 0.5 * (diffs[m - 1] + diffs[m])
#
#     def _val_to_px(y_val, v_min, p_min, scale):
#         return p_min - (y_val - v_min) * scale
#
#     def _px_to_val(px, v_min, p_min, scale):
#         return v_min + (p_min - px) / scale
#
#     # ------------------------------------------------------------
#     # 0. Font
#     # ------------------------------------------------------------
#     try:
#         if os.name == "nt":
#             font_path = "C:/Windows/Fonts/arial.ttf"
#         else:
#             font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
#         font = ImageFont.truetype(font_path, size=int(out_size * 0.06))
#     except:
#         font = ImageFont.load_default()
#
#     dummy = Image.new("RGB", (10, 10))
#     ddraw = ImageDraw.Draw(dummy)
#
#     # ------------------------------------------------------------
#     # 1. Load image
#     # ------------------------------------------------------------
#     img = Image.open(image_path).convert("RGB")
#     img_w, img_h = img.size
#
#     if x_label not in x_ticks:
#         raise ValueError(f"x_label {x_label} 不在 x_ticks 中")
#     idx = x_ticks.index(x_label)
#     center_x = x_pixels[idx]
#
#     # ------------------------------------------------------------
#     # 1A. horizontal crop
#     # ------------------------------------------------------------
#     shrink_ratio = 0.5
#     if idx > 0:
#         left_dist = int((center_x - x_pixels[idx - 1]) * 0.5 * shrink_ratio)
#     else:
#         left_dist = int((x_pixels[idx + 1] - center_x) * 0.5 * shrink_ratio)
#
#     if idx < len(x_pixels) - 1:
#         right_dist = int((x_pixels[idx + 1] - center_x) * 0.5 * shrink_ratio)
#     else:
#         right_dist = int((center_x - x_pixels[idx - 1]) * 0.5 * shrink_ratio)
#
#     left_crop = max(center_x - left_dist, 0)
#     right_crop = min(center_x + right_dist, img_w)
#
#     # ------------------------------------------------------------
#     # 1B. vertical crop（based on Δv0、round_idx 或 auto_span）
#     # ------------------------------------------------------------
#     if len(y_ticks) < 2:
#         raise ValueError("至少需要两个 y_ticks")
#
#     pairs = sorted(zip(y_ticks, y_pixels), key=lambda p: p[0])
#     v_min, v_max = pairs[0][0], pairs[-1][0]
#     p_min, p_max = pairs[0][1], pairs[-1][1]
#     scale = (p_min - p_max) / (v_max - v_min)
#     Δv0 = _median_diff([v for v, _ in pairs])
#
#     # default crop
#     center_px = _val_to_px(y_value, v_min, p_min, scale)
#     tick_span_px = abs(pairs[1][1] - pairs[0][1])
#     half_span_px_default = tick_span_px / 4
#
#     top_crop = int(center_px - half_span_px_default)
#     bottom_crop = int(center_px + half_span_px_default)
#     top_crop = max(0, min(img_h - 1, top_crop))
#     bottom_crop = max(0, min(img_h, bottom_crop))
#
#     vhs_used = None
#
#     # progressive value-span
#     if Δv0 is not None and round_idx >= 1 and not auto_span and value_half_span is None:
#         level_val = max(0, round_idx - 1)
#         vhs_used = Δv0 / (2 ** level_val)
#         v_top = max(v_min, y_value - vhs_used)
#         v_bot = min(v_max, y_value + vhs_used)
#
#         top_crop = int(max(0, min(img_h - 1, _val_to_px(v_top, v_min, p_min, scale))))
#         bottom_crop = int(max(0, min(img_h,     _val_to_px(v_bot, v_min, p_min, scale))))
#         if top_crop > bottom_crop:
#             top_crop, bottom_crop = bottom_crop, top_crop
#
#     crop_h = max(1, bottom_crop - top_crop)
#
#     # ------------------------------------------------------------
#     # 2. crop from original
#     # ------------------------------------------------------------
#     img_cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))
#     crop_w, crop_h_eff = img_cropped.size
#
#     # ------------------------------------------------------------
#     # 3. Progressive Zoom (×2)
#     # ------------------------------------------------------------
#     zoom = 2 ** max(0, round_idx - 1)
#     new_w = int(crop_w * zoom)
#     new_h = int(crop_h * zoom)
#
#     resized = img_cropped.resize((new_w, new_h), resample=Image.NEAREST)
#
#     # ------------------------------------------------------------
#     # 4. canvas（宽度 = new_w、高度 = new_h）
#     # ------------------------------------------------------------
#     canvas_w = new_w
#     canvas_h = new_h
#
#     canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
#     offset_x = 0
#     offset_y = 0
#     canvas.paste(resized, (offset_x, offset_y))
#
#     draw = ImageDraw.Draw(canvas)
#
#     # 角落标注
#     if round_idx:
#         draw.text((offset_x + 4, offset_y + 4), f"R{round_idx}", fill="black", font=font)
#
#     # ------------------------------------------------------------
#     # 5. compute visible value range
#     # ------------------------------------------------------------
#     v_region_min = _px_to_val(top_crop, v_min, p_min, scale)
#     v_region_max = _px_to_val(top_crop + crop_h, v_min, p_min, scale)
#     if v_region_min > v_region_max:
#         v_region_min, v_region_max = v_region_max, v_region_min
#
#     # ------------------------------------------------------------
#     # 6. dense tick generation
#     # ------------------------------------------------------------
#     dense_vals = []
#
#     if Δv0 is None:
#         dense_vals = [v_region_min, v_region_max]
#     else:
#         S = new_h / crop_h
#         scale_abs = abs(scale) if scale != 0 else 1e-6
#         step_ideal = target_grid_step_px / (scale_abs * S)
#
#         candidate_steps = [Δv0 / (2 ** k) for k in range(5)]
#         tick_step = min(candidate_steps, key=lambda s: abs(s - step_ideal))
#
#         coarse_min = math.floor(v_region_min / Δv0) * Δv0 - Δv0
#         coarse_max = math.ceil(v_region_max / Δv0) * Δv0 + Δv0
#
#         keep_lo = v_region_min - tick_step
#         keep_hi = v_region_max + tick_step
#
#         v = coarse_min
#         while v <= coarse_max + 1e-9:
#             if keep_lo <= v <= keep_hi:
#                 dense_vals.append(round(v, 3))   # 精度由 6 → 3
#             v += tick_step
#
#         if not dense_vals:
#             dense_vals = [v_region_min, v_region_max]
#
#     # ------------------------------------------------------------
#     # 6B. Map ticks to pixel positions（加入裁剪区限制）
#     # ------------------------------------------------------------
#     y_mapping = {}
#
#     for v in dense_vals:
#         orig_px = _val_to_px(v, v_min, p_min, scale)
#
#         # 关键修复：强制限制在裁剪区内
#         orig_px = max(top_crop, min(bottom_crop, orig_px))
#
#         rel = (orig_px - top_crop) / crop_h
#         y_pix = offset_y + int(rel * new_h)
#
#         v_round = round(v, 3)
#         y_mapping[(v_round, y_pix)] = y_pix
#
#     crop_left = offset_x
#     crop_right = offset_x + new_w
#
#     # ------------------------------------------------------------
#     # 6C. draw ticks & grid
#     # ------------------------------------------------------------
#     drawn_tick_values = set()
#
#     def draw_side(side: str):
#         for (y_val, y_pix), pix_val in y_mapping.items():
#
#             if not (offset_y <= y_pix <= offset_y + new_h):
#                 continue
#
#             text = _format_tick_val(y_val)
#             tw, th = ddraw.textbbox((0, 0), text, font=font)[2:]
#
#             if (y_pix - th // 2) < 0 or (y_pix + th // 2) > canvas_h:
#                 continue
#
#             if side == "left":
#                 x = crop_left - dash_region
#                 while x < crop_left:
#                     x_end = min(x + dash_len, crop_left)
#                     draw.line([(x, y_pix), (x_end, y_pix)], fill="gray", width=1)
#                     x += dash_len + dash_gap
#
#                 tick_x0 = crop_left - dash_region - tick_len
#                 tick_x1 = crop_left - dash_region
#                 draw.line([(tick_x0, y_pix), (tick_x1, y_pix)], fill="black", width=1)
#
#                 draw.text((tick_x0 - 4 - tw, y_pix - th // 2), text, fill="black", font=font)
#
#                 drawn_tick_values.add(y_val)
#
#             else:
#                 if y_val not in drawn_tick_values:
#                     continue
#                 x = crop_right
#                 end_limit = crop_right + dash_region
#                 while x < end_limit:
#                     x_end = min(x + dash_len, end_limit)
#                     draw.line([(x, y_pix), (x_end, y_pix)], fill="gray", width=1)
#                     x += dash_len + dash_gap
#
#                 tick_x0 = crop_right + dash_region
#                 tick_x1 = tick_x0 + tick_len
#                 draw.line([(tick_x0, y_pix), (tick_x1, y_pix)], fill="black", width=1)
#
#     # 左侧（文字）
#     draw_side("left")
#
#     # 横向虚线
#     for y_val in sorted(drawn_tick_values):
#         key_val = round(y_val, 3)
#         matched = [(v, ypix) for (v, ypix) in y_mapping.keys() if v == key_val]
#         if not matched:
#             continue
#         v_match, y_pix = matched[0]
#
#         x = crop_left
#         while x < crop_right:
#             x_end = min(x + dash_len, crop_right)
#             draw.line([(x, y_pix), (x_end, y_pix)], fill="gray", width=1)
#             x += dash_len + dash_gap
#
#     # 右侧 tick
#     draw_side("right")
#
#     # ------------------------------------------------------------
#     # 7. border
#     # ------------------------------------------------------------
#     draw.line([(crop_left, offset_y), (crop_left, offset_y + new_h)], fill=border_color, width=border_width)
#     draw.line([(crop_right, offset_y), (crop_right, offset_y + new_h)], fill=border_color, width=border_width)
#
#     # ------------------------------------------------------------
#     # 8. save
#     # ------------------------------------------------------------
#     if output_path is None:
#         safe_name = safe_filename(point_name)
#         chart_id = os.path.splitext(os.path.basename(image_path))[0]
#         chart_id = chart_id.replace("_with_grid", "").replace("_grid", "")
#         output_path = os.path.join("results", chart_id, "tempy",
#                                    f"amplifier_crop_{safe_name}_r{round_idx}.png")
#
#     output_path = sanitize_filename(output_path)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     canvas.save(output_path)
#
#     # ------------------------------------------------------------
#     # 9. return
#     # ------------------------------------------------------------
#     visible_ticks = []
#     for (v, y_pix), _px in y_mapping.items():
#         if offset_y <= y_pix <= offset_y + new_h:
#             visible_ticks.append(v)
#
#     if return_meta:
#         return output_path, visible_ticks, {
#             "round_idx": round_idx,
#             "visible_ticks": visible_ticks,
#             "value_min": v_region_min,
#             "value_max": v_region_max,
#             "zoom": zoom,
#             "crop_rect_orig": [left_crop, top_crop, right_crop, bottom_crop],
#             "resize_WH": [new_w, new_h],
#         }
#
#     return output_path, visible_ticks

def crop_segment_with_dual_yaxis_ticks(
    image_path: str,
    point_name: str,        # x 为分类轴的分类标签
    y_value: float,         # 纵向条：y 为数值轴（中心值）
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
    output_path: str | None = None,
    enforce_min_vertical_coverage: bool = False,   # ✅ 保留参数，但语义对齐横向版本
    min_canvas_height_ratio: float = 1.0,         # ✅ 相对“默认窗口高度”，而不是整图高度
    round_idx: int = 1,            # progressive zoom 轮次（从 1 开始）
    return_meta: bool = False,
    value_half_span: float | None = None,
    target_grid_step_px: int = 50,
    clamp_grid_px: tuple[int, int] = (40, 60),    # 目前未启用，保留接口
    auto_span: bool = False,
    span_min: float | None = None,
    span_max: float | None = None,
    m_max: int = 10,
    stabilize_band: tuple[int, int] = (45, 55)
):
    """
    ✅ 竖向 bar chart 版本（规则对齐横向版）：
      - y 为数值轴：按 Δv0 / 2^(round_idx-1) 做 progressive value-span 裁剪
      - x 为分类轴：在横向上只裁到当前 bar 附近，左右少量留白
      - zoom 只放大裁剪区域，不改变 value-span
      - 左右双 y 轴 + 水平虚线网格，与横向版本风格一致（只是转了个方向）
    """
    from PIL import Image, ImageDraw, ImageFont
    import os, math, re

    # ---------------- Formatting & utilities ----------------
    def _format_tick_val(v: float) -> str:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        s = str(round(fv, 3))
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    if "safe_filename" in globals():
        safe_filename = globals()["safe_filename"]
    else:
        def safe_filename(name: str):
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]

    if "sanitize_filename" in globals():
        sanitize_filename = globals()["sanitize_filename"]
    else:
        def sanitize_filename(path: str):
            return re.sub(r'[<>:"/\\|?*]+', "_", path)

    def _median_diff(vals):
        diffs = sorted(abs(b - a) for a, b in zip(vals[:-1], vals[1:]) if b != a)
        if not diffs:
            return None
        m = len(diffs) // 2
        return diffs[m] if len(diffs) % 2 else 0.5 * (diffs[m - 1] + diffs[m])

    def _val_to_px(y_val, v_min, p_min, scale):
        # 数值 -> 像素（和横向版本一致：v 越大，像素越“往上/下”取决于原图）
        return p_min - (y_val - v_min) * scale

    def _px_to_val(px, v_min, p_min, scale):
        return v_min + (p_min - px) / scale

    # ---------------- 0. Font ----------------
    try:
        if os.name == "nt":
            font_path = "C:/Windows/Fonts/arial.ttf"
        else:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, size=int(out_size * 0.06))
    except:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (10, 10))
    ddraw = ImageDraw.Draw(dummy)

    # ---------------- 1. Load image ----------------
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    if x_label not in x_ticks:
        raise ValueError(f"x_label {x_label} 不在 x_ticks 中")
    idx = x_ticks.index(x_label)
    center_x = x_pixels[idx]

    # ---------------- 1A. horizontal crop（对齐横向规则） ----------------
    # 横向只围绕当前 bar：左右取相邻 tick 中点的缩减版本
    shrink_ratio = 1  # 和横向版本一致：只取中间一小段
    if idx > 0:
        left_dist = int((center_x - x_pixels[idx - 1]) * 0.5 * shrink_ratio)
    else:
        left_dist = int((x_pixels[idx + 1] - center_x) * 0.5 * shrink_ratio)

    if idx < len(x_pixels) - 1:
        right_dist = int((x_pixels[idx + 1] - center_x) * 0.5 * shrink_ratio)
    else:
        right_dist = int((center_x - x_pixels[idx - 1]) * 0.5 * shrink_ratio)

    left_crop = max(center_x - left_dist, 0)
    right_crop = min(center_x + right_dist, img_w)

    # ---------------- 1B. vertical crop（对齐横向 value-span 规则） ----------------
    if len(y_ticks) < 2:
        raise ValueError("至少需要两个 y_ticks")

    pairs = sorted(zip(y_ticks, y_pixels), key=lambda p: p[0])
    v_min, v_max = pairs[0][0], pairs[-1][0]
    p_min, p_max = pairs[0][1], pairs[-1][1]

    scale = (p_min - p_max) / (v_max - v_min)
    Δv0 = _median_diff([v for v, _ in pairs])  # base tick step (value domain)

    # progressive half-width in VALUE domain
    # round1 → Δv0
    # round2 → Δv0/2
    # round3 → Δv0/4
    half_val = Δv0 / (2 ** (round_idx - 1))

    # ---- correct symmetric pixel-domain cropping ----

    scale_abs = abs(scale)

    center_px = _val_to_px(y_value, v_min, p_min, scale)

    pixel_half = half_val * scale_abs

    top_px_val = center_px - pixel_half
    bot_px_val = center_px + pixel_half

    # clamp only to image boundaries
    top_crop = max(0, int(top_px_val))
    bottom_crop = min(img_h, int(bot_px_val))

    crop_h = bottom_crop - top_crop

    # ---------------- 2. crop from original ----------------
    img_cropped = img.crop((left_crop, top_crop, right_crop, bottom_crop))
    crop_w, crop_h_eff = img_cropped.size  # crop_h_eff 仅保留，方便调试

    # ---------------- 3. Progressive Zoom (×2^(round_idx-1)) ----------------
    zoom = 2 ** max(0, round_idx)
    new_w = int(crop_w * zoom)
    new_h = int(crop_h * zoom)
    resized = img_cropped.resize((new_w, new_h), resample=Image.NEAREST)

    # ---------------- 4. 当前裁剪区域对应的 value 范围 ----------------
    v_region_min = _px_to_val(top_crop, v_min, p_min, scale)
    v_region_max = _px_to_val(top_crop + crop_h, v_min, p_min, scale)
    if v_region_min > v_region_max:
        v_region_min, v_region_max = v_region_max, v_region_min

    # ---------------- 5. dense tick 生成（固定 2/4/8 分 Δv0） ----------------
    dense_vals = []
    if Δv0 is None:
        # 没法估计原始 tick 间距，就只用上下边界兜底
        dense_vals = [v_region_min, v_region_max]
    else:
        # round_idx = 0 或 None 时，当作 baseline：不加密（step = Δv0）
        if round_idx is None or round_idx <= 0:
            div_k = 0
        else:
            # round_idx = 1 → Δv0 / 2
            # round_idx = 2 → Δv0 / 4
            # round_idx = 3 → Δv0 / 8 ...
            div_k = round_idx

        tick_step = Δv0 / (2 ** div_k) if div_k > 0 else Δv0

        # 以原始 tick 步长 Δv0 为基准，确定覆盖当前裁剪区域的值域
        v_start = math.floor(v_region_min / Δv0) * Δv0
        v_end   = math.ceil(v_region_max / Δv0) * Δv0

        # 为了让网格线稍微超出一点裁剪区域，向两头各多走半步
        v = v_start - tick_step
        while v <= v_end + tick_step + 1e-9:
            dense_vals.append(round(v, 3))
            v += tick_step

        if not dense_vals:
            dense_vals = [v_region_min, v_region_max]


    # # ---------------- 5. dense tick 生成（逻辑对齐横向版） ----------------
    # dense_vals = []
    # if Δv0 is None:
    #     dense_vals = [v_region_min, v_region_max]
    # else:
    #     S = new_h / crop_h                       # 放大后的缩放因子
    #     scale_abs = abs(scale) if scale != 0 else 1e-6
    #     step_ideal = target_grid_step_px / (scale_abs * S)  # 希望“屏幕上”每格 ~ target_grid_step_px 像素
    #
    #     candidate_steps = [Δv0 / (2 ** k) for k in range(5)]
    #     tick_step = min(candidate_steps, key=lambda s: abs(s - step_ideal))
    #
    #     coarse_min = math.floor(v_region_min / Δv0) * Δv0 - Δv0
    #     coarse_max = math.ceil(v_region_max / Δv0) * Δv0 + Δv0
    #
    #     keep_lo = v_region_min - tick_step
    #     keep_hi = v_region_max + tick_step
    #
    #     v = coarse_min
    #     while v <= coarse_max + 1e-9:
    #         if keep_lo <= v <= keep_hi:
    #             dense_vals.append(round(v, 3))
    #         v += tick_step
    #
    #     if not dense_vals:
    #         dense_vals = [v_region_min, v_region_max]

    # ---------------- 6. 估计最大刻度文字宽度，用于扩宽 canvas（左右双轴） ----------------
    max_text_w = 0
    for v in dense_vals:
        text = _format_tick_val(v)
        bbox = ddraw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        max_text_w = max(max_text_w, w)

    axis_side_width = dash_region + tick_len + max_text_w + side_pad

    canvas_w = int(axis_side_width * 2 + new_w)  # 左右各一个 y 轴区域
    canvas_h = new_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # 把裁剪图像贴在中间，左右预留 y 轴刻度 & 虚线区域
    crop_left = int(axis_side_width)
    crop_right = crop_left + new_w
    offset_x = crop_left
    offset_y = 0

    canvas.paste(resized, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)

    # 角落标注当前轮次
    if round_idx:
        draw.text((crop_left + 4, offset_y + 4), f"R{round_idx}", fill="black", font=font)

    # ---------------- 7. 将 dense_vals 映射到当前 canvas 的 y 像素 ----------------
    y_mapping = {}
    for v in dense_vals:
        orig_px = _val_to_px(v, v_min, p_min, scale)
        orig_px = max(top_crop, min(bottom_crop, orig_px))
        rel = (orig_px - top_crop) / crop_h
        y_pix = offset_y + int(rel * new_h)
        v_round = round(v, 3)
        y_mapping[(v_round, y_pix)] = y_pix

    # ---------------- 8. 画左右 y 轴刻度 + 水平网格 ----------------
    drawn_tick_values = set()

    def draw_side(side: str):
        for (y_val, y_pix), _ in y_mapping.items():
            if not (offset_y <= y_pix <= offset_y + new_h):
                continue

            text = _format_tick_val(y_val)
            bbox = ddraw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            if (y_pix - th // 2) < 0 or (y_pix + th // 2) > canvas_h:
                continue

            if side == "left":
                # 左侧虚线 & 刻度
                x = crop_left - dash_region
                while x < crop_left:
                    x_end = min(x + dash_len, crop_left)
                    draw.line([(x, y_pix), (x_end, y_pix)], fill="gray", width=1)
                    x += dash_len + dash_gap

                tick_x0 = crop_left - dash_region - tick_len
                tick_x1 = crop_left - dash_region
                draw.line([(tick_x0, y_pix), (tick_x1, y_pix)], fill="black", width=1)

                text_x = tick_x0 - 4 - tw
                text_y = y_pix - th // 2
                draw.text((text_x, text_y), text, fill="black", font=font)

                drawn_tick_values.add(y_val)

            else:
                # 右侧只画在左侧已经画过的 value 上
                if y_val not in drawn_tick_values:
                    continue

                x = crop_right
                end_limit = crop_right + dash_region
                while x < end_limit:
                    x_end = min(x + dash_len, end_limit)
                    draw.line([(x, y_pix), (x_end, y_pix)], fill="gray", width=1)
                    x += dash_len + dash_gap

                tick_x0 = crop_right + dash_region
                tick_x1 = tick_x0 + tick_len
                draw.line([(tick_x0, y_pix), (tick_x1, y_pix)], fill="black", width=1)

    # 左侧 y 轴 + 刻度文字
    draw_side("left")

    # 水平虚线网格（在条形图区域内部）
    for y_val in sorted(drawn_tick_values):
        key_val = round(y_val, 3)
        matched = [(v, ypix) for (v, ypix) in y_mapping.keys() if v == key_val]
        if not matched:
            continue
        _, y_pix = matched[0]

        x = crop_left
        while x < crop_right:
            x_end = min(x + dash_len, crop_right)
            draw.line([(x, y_pix), (x_end, y_pix)], fill="gray", width=1)
            x += dash_len + dash_gap

    # 右侧 y 轴刻度
    draw_side("right")

    # ---------------- 9. border（和横向版风格一致） ----------------
    draw.line([(crop_left, offset_y), (crop_left, offset_y + new_h)], fill=border_color, width=border_width)
    draw.line([(crop_right, offset_y), (crop_right, offset_y + new_h)], fill=border_color, width=border_width)

    # ---------------- 10. save ----------------
    if output_path is None:
        safe_name = safe_filename(point_name)
        chart_id = os.path.splitext(os.path.basename(image_path))[0]
        chart_id = chart_id.replace("_with_grid", "").replace("_grid", "")
        output_path = os.path.join("results", chart_id, "crop",
                                   f"amplifier_crop_{safe_name}_r{round_idx}.png")

    output_path = sanitize_filename(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)

    # ---------------- 11. return ----------------
    visible_ticks = []
    for (v, y_pix), _ in y_mapping.items():
        if offset_y <= y_pix <= offset_y + new_h:
            visible_ticks.append(v)

    # 去重 + 排序，让后续 prompt 构造更稳
    visible_ticks = sorted(set(visible_ticks))

    if return_meta:
        return output_path, visible_ticks, {
            "round_idx": round_idx,
            "visible_ticks": visible_ticks,
            "value_min": v_region_min,
            "value_max": v_region_max,
            "zoom": zoom,
            "crop_rect_orig": [left_crop, top_crop, right_crop, bottom_crop],
            "resize_WH": [new_w, new_h],
        }

    return output_path, visible_ticks


from typing import Tuple, Optional
async def try_crop_until_bar_detected(
    image_path: str,
    point_name: str,
    y_value: float,
    x_label: str,
    x_ticks: list,
    x_pixels: list,
    y_ticks: list,
    y_pixels: list,
    judge_prompt: str,
    out_size: int = 240,
    side_pad: int = 10,
    vert_pad: int = 30,
    max_attempts: int = 20,
    output_path: Optional[str] = None,
    round_idx: int = 0,   # ✅ 当前 amplifier 轮次（1,2,3…）
) -> Optional[Tuple[str, list]]:

    # ---- 1) 数值轴映射参数 ----
    pairs = sorted(zip(y_ticks, y_pixels), key=lambda p: p[0])
    v_min, v_max = pairs[0][0], pairs[-1][0]
    p_min, p_max = pairs[0][1], pairs[-1][1]
    scale = (p_min - p_max) / (v_max - v_min)   # 值→像素, >0
    tick_span_px = abs(pairs[1][1] - pairs[0][1])

    # 原始预测值对应的像素位置
    center_px = p_min - (y_value - v_min) * scale

    # ---- 2) 只向“下方”偏移：0, +1tick, +2tick, ... ----
    for i in range(0, max_attempts + 1):
        # 像素上往下偏移 i 个 tick 间距
        shifted_center_px = center_px + i * tick_span_px

        # 反推对应的数值 y
        shifted_y_val = v_min + (p_min - shifted_center_px) / scale

        # 超出值域就没必要再试
        if shifted_y_val < v_min or shifted_y_val > v_max:
            continue

        # ---- 3) 以偏移后的中心值做裁剪 ----
        pred_img_path, visible_ticks = crop_segment_with_dual_yaxis_ticks(
            image_path=image_path,
            point_name=point_name,
            y_value=shifted_y_val,          # ✅ 关键：用偏移后的值
            x_label=x_label,
            x_ticks=x_ticks,
            x_pixels=x_pixels,
            y_ticks=y_ticks,
            y_pixels=y_pixels,
            out_size=out_size,
            enforce_min_vertical_coverage=True,
            min_canvas_height_ratio=1.0,
            round_idx=round_idx,            # 只标当前 amplifier 轮次
        )

        # ---- 4) 问 LLM：这张图里有目标 bar 吗？ ----
        exists = await call_llm_bar_existence(judge_prompt, pred_img_path)
        if exists:
            # 找到了，就用当前这张图进入坐标预测阶段
            return pred_img_path, visible_ticks

    print("⚠️ 未检测到 bar，向下偏移多轮仍失败。")
    return None


async def call_llm_bar_existence(prompt: str, image_path: str) -> bool:
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": "gemini-2.5-flash", #gemini-2.0-flash; gpt-4o

        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            try:
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip().lower()

                # ✅ 尝试解析 JSON 格式回答
                try:
                    json_str = next(s for s in content.splitlines() if s.strip().startswith("{"))
                    parsed = json.loads(json_str)
                    return parsed.get("exists", False)
                except Exception:
                    pass

                # ✅ 若 JSON 解析失败，回退至关键词判断
                return "yes" in content and "no" not in content

            except Exception as e:
                print(f"❌ LLM 判断异常：{e}")
                return False


# async def call_llm_response(prompt: str, image_path: str, point_name: str) -> Tuple[float, float]:
#     with open(image_path, "rb") as img_file:
#         base64_image = base64.b64encode(img_file.read()).decode("utf-8")
#
#     payload = {
#         "model": "gemini-2.0-flash", # gemini-2.0-flash; gpt-4o;qwen-vl-max;gpt-4o
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}} ,
#                     {"type": "text", "text": prompt}
#                 ]
#             }
#         ],
#         "temperature": 0
#     }
#
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, headers=headers, json=payload) as response:
#             try:
#                 result = await response.json()
#                 if "error" in result and result["error"]["code"] == 429:  # 检测到 429 错误
#                     print(f"❌ 请求频率超限，错误信息: {result['error']['message']}")
#                     return (-1, -1)  # 返回一个特殊值来指示出错
#             except Exception as e:
#                 print(f"❌ JSON 解析失败：{e}")
#                 print("Raw response:", await response.text())
#                 return (-1, -1)
#
#             # 检查返回结果是否包含 'choices' 键
#             if "choices" not in result:
#                 print("❌ API 返回不包含 'choices' 字段，返回内容如下：")
#                 print(json.dumps(result, indent=2))
#                 return (-1, -1)
#
#             content = result["choices"][0]["message"]["content"]
#             try:
#                 json_str = next(s for s in content.splitlines() if s.strip().startswith("{") or s.strip().startswith("["))
#                 coords_json = json.loads(json_str)
#
#                 if isinstance(coords_json, list):
#                     for item in coords_json:
#                         if item.get("label") == point_name:
#                             return tuple(item["point"])
#                 elif "datapoints" in coords_json:
#                     for item in coords_json["datapoints"]:
#                         if point_name in item:
#                             return tuple(item[point_name])
#             except Exception:
#                 print(f"⚠️ Failed to parse model response: {content}")
#             return (-1, -1)

import json
import base64
import aiohttp

# ---------------- JSON 修复器 ---------------- #
def repair_json_str(bad_json: str) -> str:
    """
    尝试修复不完整 JSON：
    - 缺失末尾 } 或 ]
    - 嵌套结构未闭合
    - LLM 输出中断
    """
    txt = bad_json.strip()

    # 1) 先尝试直接解析
    try:
        json.loads(txt)
        return txt
    except:
        pass

    # 2) 根据内容判断是否为 { 开头的对象
    if txt.startswith("{"):
        # 如果缺右大括号
        if txt.count("{") > txt.count("}"):
            repaired = txt + "}"
            try:
                json.loads(repaired)
                return repaired
            except:
                pass

        # 如果缺右中括号（常见于 {"datapoints":[ ）
        if txt.count("[") > txt.count("]"):
            repaired = txt + "]}"
            try:
                json.loads(repaired)
                return repaired
            except:
                pass

    # 3) 如果以 ] 结尾，可能缺一个 }
    if txt.endswith("]"):
        repaired = txt + "}"
        try:
            json.loads(repaired)
            return repaired
        except:
            pass

    # 4) 暴力兜底：补 "]}"
    repaired = txt + "]}"
    try:
        json.loads(repaired)
        return repaired
    except:
        pass

    # 5) 实在不行 → 返回原文（让上层报错）
    return txt


# ---------------- 主函数 ---------------- #
async def call_llm_response(prompt: str, image_path: str, point_name: str):
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": "gemini-2.5-flash",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:

            # ---------- 处理网络 JSON ---------- #
            try:
                result = await response.json()
                if "error" in result and result["error"].get("code") == 429:
                    print(f"❌ 请求频率超限: {result['error']['message']}")
                    return (-1, -1)
            except Exception as e:
                print("❌ API 返回 JSON 解析失败：", e)
                print("Raw content = ", await response.text())
                return (-1, -1)

            if "choices" not in result:
                print("❌ API 返回不包含 'choices'：", json.dumps(result, indent=2))
                return (-1, -1)

            # ---------- 提取 LLM 文本 ---------- #
            content = result["choices"][0]["message"]["content"]

            # ---------- 找到可能 JSON 行 ---------- #
            try:
                json_str = next(
                    s for s in content.splitlines()
                    if s.strip().startswith("{") or s.strip().startswith("[")
                )
            except StopIteration:
                print("❌ 未找到 JSON 结构：", content)
                return (-1, -1)

            # ---------- 修复 JSON ---------- #
            json_str_repaired = repair_json_str(json_str)

            try:
                coords_json = json.loads(json_str_repaired)
            except Exception as e:
                print("❌ 修复后仍解析失败：", e)
                print("JSON = ", json_str_repaired)
                print("原始内容 = ", content)
                return (-1, -1)

            # ---------- 解析最终坐标（严格符合你指定的 JSON 格式） ---------- #

            def parse_datapoint(coords_json, point_name):
                """
                解析结构:
                {
                    "datapoints": [
                        { "<point_name>": ["<x_label>", y] }
                    ]
                }
                """

                datapoints = coords_json.get("datapoints", None)
                if not isinstance(datapoints, list):
                    return None

                for item in datapoints:
                    if point_name not in item:
                        continue

                    val = item[point_name]

                    # ---- Case A：LLM 正确返回 ["Chest freezers", y] ----
                    if isinstance(val, list) and len(val) == 2:
                        x_label, y_val = val[0], val[1]

                        # --- 强制转换 y 值 ---
                        try:
                            y_float = float(y_val)
                        except:
                            # 尝试从字符串中提取数字（例如 "y=23.5" 或 "23.5%"）
                            import re
                            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(y_val))
                            if nums:
                                y_float = float(nums[0])
                            else:
                                print(f"❌ 无法从 '{y_val}' 解析 y 数值，返回 (-1,-1)")
                                return (-1, -1)

                        return (x_label, y_float)

                    # ---- Case B：LLM 错误返回 float（如 23.7） ----
                    if isinstance(val, (int, float)):
                        x_label = point_name.split(",")[0].strip()
                        return (x_label, float(val))

                    print(f"⚠️ 无法解析 datapoint: {val}")
                    return None

                return None  # 找不到 point_name

            # 优先按你指定的格式解析
            if isinstance(coords_json, dict):
                parsed = parse_datapoint(coords_json, point_name)
                if parsed is not None:
                    return parsed

            print("⚠️ Model JSON 不包含目标 point：", content)
            return (-1, -1)

            # # ---------- 解析最终坐标 ---------- #
            # # 1) 格式类型一：{"datapoints":[{ "label":[x,y] }]}
            # if isinstance(coords_json, dict) and "datapoints" in coords_json:
            #     for item in coords_json["datapoints"]:
            #         if point_name in item:
            #             return tuple(item[point_name])
            #
            # # 2) 格式类型二：[ { "label": "...", "point":[x,y] } ]
            # if isinstance(coords_json, list):
            #     for item in coords_json:
            #         if item.get("label") == point_name:
            #             return tuple(item["point"])
            #
            # print("⚠️ Model JSON 不包含目标 point：", content)
            # return (-1, -1)


async def call_llm_with_retry(prompt: str, image_path: str, point_name: str) -> Tuple[float, float]:
    for attempt in range(MAX_ATTEMPTS):  # 保证最多尝试 MAX_ATTEMPTS 次
        coords = await call_llm_response(prompt, image_path, point_name)
        if coords != (-1, -1):
            return coords

        # 如果返回的是 -1，说明请求失败
        print("❌ 请求失败，等待并重试...")
        await asyncio.sleep(7)  # 等待7秒后再尝试
    print(f"⚠️ {point_name} 无法成功预测，已尝试 {MAX_ATTEMPTS} 次")
    return (-1, -1)


# --- 新增：仅保留 grid+with_grid 的最后一轮预测 ---
def filter_final_round_for_feedback(df: pd.DataFrame) -> pd.DataFrame:
    # 定义多轮反馈类型的 prompt
    multi_round_prompt_types = ["amplifier","feedback", "feedback_crop", "feedback_crop_final", "color_feedback"]

    # 为每组配置添加轮次编号
    df["round_index"] = df.groupby(["chart_id", "point", "prompt_type", "image_type"]).cumcount()

    # 标记哪些记录属于需要筛选“最后一轮”的多轮反馈
    mask_multi_feedback = df["prompt_type"].isin(multi_round_prompt_types)

    # 获取每组配置的最大轮次索引
    df["max_round_index"] = df.groupby(["chart_id", "point", "prompt_type", "image_type"])["round_index"].transform("max")

    # 保留：非多轮反馈的记录；或多轮反馈中为最后一轮的记录
    df_filtered = df[~mask_multi_feedback | (df["round_index"] == df["max_round_index"])].drop(columns=["max_round_index"])

    return df_filtered


def compute_mae(pred: Tuple, gt: Tuple, axis_types: dict) -> Union[float, None]:
    """
    计算 MAE，区分数值轴和分类轴
    pred, gt = (x_val, y_val)
    axis_types: {"x": "categorical"/"numerical", "y": "categorical"/"numerical"}
    """
    x_val, y_val = pred
    x_gt, y_gt = gt

    # --- y 轴 ---
    if axis_types.get("y") == "numerical":
        if pd.notna(y_val) and pd.notna(y_gt):
            return round(abs(float(y_val) - float(y_gt)), 4)
    elif axis_types.get("y") == "categorical":
        return 0.0 if y_val == y_gt else 1.0

    # --- x 轴 ---
    if axis_types.get("x") == "numerical":
        if pd.notna(x_val) and pd.notna(x_gt):
            return round(abs(float(x_val) - float(x_gt)), 4)
    elif axis_types.get("x") == "categorical":
        return 0.0 if x_val == x_gt else 1.0

    return None


def compute_re(pred: Tuple, gt: Tuple, axis_types: dict) -> Tuple[float, float]:
    """
    计算相对误差，数值轴可算，相对误差 categorical 轴返回 -1
    """
    x_val, y_val = pred
    x_gt, y_gt = gt

    # --- x 轴 ---
    if axis_types.get("x") == "numerical" and pd.notna(x_val) and pd.notna(x_gt) and x_gt != 0:
        x_re = abs(float(x_val) - float(x_gt)) / (abs(float(x_gt)) + 1e-6)
    else:
        x_re = -1

    # --- y 轴 ---
    if axis_types.get("y") == "numerical" and pd.notna(y_val) and pd.notna(y_gt) and y_gt != 0:
        y_re = abs(float(y_val) - float(y_gt)) / (abs(float(y_gt)) + 1e-6)
    else:
        y_re = -1

    return round(x_re, 4), round(y_re, 4)

def evaluate_results(df: pd.DataFrame, result_dir: str = "."):
    os.makedirs(result_dir, exist_ok=True)

    df = filter_final_round_for_feedback(df)

    # ========== 类型转换 ==========
    df["gt_y"] = pd.to_numeric(df["gt_y"], errors="coerce")
    df["pred_y"] = pd.to_numeric(df["pred_y"], errors="coerce")

    # ========== 有效性 ==========
    df["y_valid"] = df["pred_y"].notna()

    # ========== 原始 Y 相对误差 ==========
    def compute_y_re(row):
        if row["y_valid"] and row["gt_y"] != 0:
            return abs(row["pred_y"] - row["gt_y"]) / abs(row["gt_y"])
        return None

    df["y_re"] = df.apply(compute_y_re, axis=1)

    # ========== ⭐ 新增：Normalized Y Error (|Δy| / y_range) ==========
    if "y_range" in df.columns:
        df["y_abs_err"] = (df["pred_y"] - df["gt_y"]).abs()
        df["y_err_over_range"] = df["y_abs_err"] / df["y_range"]
    else:
        df["y_err_over_range"] = np.nan

    # ========== 汇总 Y 轴指标 ==========
    summary = df.groupby(["prompt_type", "image_type"]).apply(lambda group: pd.Series({
        "avg_y_mae": (group[group["y_valid"]]["pred_y"] - group[group["y_valid"]]["gt_y"]).abs().mean(),
        "avg_y_re": group[group["y_valid"]]["y_re"].mean(),

        # ⭐ 新增：Normalized Y Error
        "avg_y_err_over_range": group["y_err_over_range"].mean(),

        "valid_y_count": group["y_valid"].sum()
    })).reset_index()

    summary_path = os.path.join(result_dir, "axis_level_summary.csv")
    summary.to_csv(summary_path, index=False)

    df.to_csv(os.path.join(result_dir, "full_results_with_yre.csv"), index=False)

    # ========== 绘图 ==========
    x = range(len(summary))
    labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
    bar_width = 0.4

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # —— Y轴 MAE —— (左轴)
    bars1 = ax1.bar(x, summary["avg_y_mae"], width=bar_width, label="Y MAE", color="#76D7C4")
    ax1.set_ylabel("MAE (Y axis)")
    ax1.set_xlabel("Prompt + Image Setting")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20)
    ax1.tick_params(axis='y')

    for i, mae in enumerate(summary["avg_y_mae"]):
        if not pd.isna(mae):
            ax1.text(i, mae + 0.02, f"{mae:.2f}", ha='center', fontsize=9)

    # —— Y轴相对误差 —— (右轴)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["avg_y_re"], label="Y Relative Error", color="#F1948A", marker="o", linewidth=2)
    ax2.set_ylabel("Relative Error (Y axis)")
    ax2.tick_params(axis='y')

    for i, re in enumerate(summary["avg_y_re"]):
        if not pd.isna(re):
            ax2.text(i, re + 0.01, f"{re:.2f}", ha='center', fontsize=9, color="#F1948A")

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")

    plt.title("Y Axis MAE & Relative Error by Prompt+Image Setting")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "axis_level_mae_re_combined.png"))
    plt.close()

    # ========== ⭐ 绘制 Normalized Y Error ==========
    plt.figure(figsize=(10, 6))
    plt.bar(x, summary["avg_y_err_over_range"], width=0.4, color="#AF7AC5")
    for i, v in enumerate(summary["avg_y_err_over_range"]):
        if not pd.isna(v):
            plt.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=9)

    plt.xticks(x, labels, rotation=20)
    plt.ylabel("|ΔY| / (max_tick - min_tick)")
    plt.title("Normalized Y Error (Per Prompt+Image Setting)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "axis_level_y_err_over_range.png"))
    plt.close()


# ====== 绘图 ======
# def evaluate_results(df: pd.DataFrame, result_dir: str = "."):
#     os.makedirs(result_dir, exist_ok=True)
#
#     df = filter_final_round_for_feedback(df)
#
#     # ========== 类型转换，确保 pred_y 与 gt_y 可数值运算 ==========
#     df["gt_y"] = pd.to_numeric(df["gt_y"], errors="coerce")
#     df["pred_y"] = pd.to_numeric(df["pred_y"], errors="coerce")
#
#     # ========== 标注有效性 ==========
#     if "pred_y" in df.columns:
#         df["y_valid"] = df["pred_y"].apply(lambda x: pd.notna(x))
#     else:
#         df["y_valid"] = False
#
#     # ========== 每行计算 y 相对误差 ==========
#     def compute_y_re(row):
#         if row["y_valid"] and row["gt_y"] != 0:
#             return abs(row["pred_y"] - row["gt_y"]) / abs(row["gt_y"])
#         return None
#
#     df["y_re"] = df.apply(compute_y_re, axis=1)
#
#     # ========== 汇总 Y 轴指标 ==========
#     summary = df.groupby(["prompt_type", "image_type"]).apply(lambda group: pd.Series({
#         "avg_y_mae": (group[group["y_valid"]]["pred_y"] - group[group["y_valid"]]["gt_y"]).abs().mean(),
#         "avg_y_re": group[group["y_valid"]]["y_re"].mean(),
#         "valid_y_count": group["y_valid"].sum()
#     })).reset_index()
#
#     summary_path = os.path.join(result_dir, "axis_level_summary.csv")
#     summary.to_csv(summary_path, index=False)
#     print("📄 已保存 Y 轴评估表 axis_level_summary.csv")
#
#     # ========== 保存完整记录 ==========
#     df.to_csv(os.path.join(result_dir, "full_results_with_yre.csv"), index=False)
#     print("📄 已保存完整记录表 full_results_with_yre.csv")
#
#     # ========== 合并绘制 MAE & 相对误差 ==========
#     x = range(len(summary))
#     labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
#     bar_width = 0.4
#
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#
#     # —— Y轴 MAE —— (左轴)
#     bars1 = ax1.bar(x, summary["avg_y_mae"], width=bar_width, label="Y MAE", color="#76D7C4")
#     ax1.set_ylabel("MAE (Y axis)")
#     ax1.set_xlabel("Prompt + Image Setting")
#     ax1.set_xticks(list(x))
#     ax1.set_xticklabels(labels, rotation=20)
#     ax1.tick_params(axis='y')
#
#     # 在柱子上标注 MAE
#     for i, mae in enumerate(summary["avg_y_mae"]):
#         if not pd.isna(mae):
#             ax1.text(i, mae + 0.02, f"{mae:.2f}", ha='center', va='bottom', fontsize=9)
#
#     # —— Y轴相对误差 —— (右轴)
#     ax2 = ax1.twinx()
#     bars2 = ax2.plot(x, summary["avg_y_re"], label="Y Relative Error", color="#F1948A", marker="o", linewidth=2)
#     ax2.set_ylabel("Relative Error (Y axis)")
#     ax2.tick_params(axis='y')
#
#     # 在点上标注 RE
#     for i, re in enumerate(summary["avg_y_re"]):
#         if not pd.isna(re):
#             ax2.text(i, re + 0.01, f"{re:.2f}", ha='center', va='bottom', fontsize=9, color="#F1948A")
#
#     # —— 图例合并 ——
#     lines, labels1 = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")
#
#     plt.title("Y Axis MAE & Relative Error by Prompt+Image Setting")
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "axis_level_mae_re_combined.png"))
#     print("📊 已保存合并图 axis_level_mae_re_combined.png")
#     plt.close()

# -*- coding: utf-8 -*-
"""
run_experiment_batch.py  ——  完整批量异步实验脚本（保留全部业务逻辑）
-------------------------------------------------------------
* 支持 --batch-size 控制并发批量大小
* 支持 --chart-ids 选择性运行指定图表
* 结果按 chart_id 拆分子目录保存，并调用 evaluate_results()
* 完全沿用原有 FEEDBACK / 裁剪 / adaptive‑crop 等内部实现，不删除任何细节
"""

def build_feedback_final_preds(records: list[dict]) -> dict:
    """
    从 records 中提取每个 (chart_id, point) 的 feedback 最后一轮结果
    """
    final_preds = {}
    for r in records:
        if r["prompt_type"] == "feedback":
            key = (r["chart_id"], r["point"])
            if key not in final_preds or r.get("round_index", 0) > final_preds[key].get("round_index", -1):
                final_preds[key] = r
    return final_preds

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

    # ----------------------- 1) 单数据集执行 ----------------------
    async def run_for_dataset(dataset, feedback_final_preds, run_amplifier=True):
        nonlocal records, feedback_final_results
        axis_types = {"x": "categorical","y": "numerical"}  # 全局设定

        pred_coords = []
        for group_label, sub_points in dataset["data_points"].items():
            for sub_label, y_value in sub_points.items():
                point_name = f"{group_label}, {sub_label}"  # ⚠️ 用英文逗号，便于后续 split
                gt = (None,y_value)
                pred_coords.append((point_name,y_value))  # 注意：point_name 是 coord[0]

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
                            MAX_AMPLIFIER_ROUNDS = 3
                            feedback_key = (dataset["chart_id"], point_name)
                            feedback_pred = feedback_final_preds.get(feedback_key)

                            # 🟡 没有 feedback 结果则跳过
                            if feedback_pred is None or not isinstance(feedback_pred["pred_y"], (int, float)):
                                print(f"⚠️ Skipping amplifier: no valid feedback prediction for {point_name}")
                                continue

                            # ✅ 第一轮以 feedback 最终预测值为起点
                            y_curr = feedback_pred["pred_y"]
                            x_label = point_name.split(",")[-1].strip()
                            last_pred = None
                            history_preds = []

                            for amp_round in range(1, MAX_AMPLIFIER_ROUNDS + 1):
                                print(f"\n⚙️ Amplifier Round {amp_round} @ {point_name}")

                                # 1️⃣ 基于当前 y_curr 裁剪出新区域
                                res = await try_crop_until_bar_detected(
                                    image_path=dataset["image_paths"]["grid_with_grid"],
                                    point_name=point_name,
                                    y_value=y_curr,
                                    x_label=x_label,
                                    x_ticks=dataset["x_ticks"],
                                    x_pixels=dataset["x_pixels"],
                                    y_ticks=dataset["y_ticks"],
                                    y_pixels=dataset["y_pixels"],
                                    judge_prompt=build_color_prompt(
                                        point_name=point_name,
                                        series_color=dataset["series_color"],
                                    ),
                                    round_idx=amp_round,
                                )

                                if res is None:
                                    print(f"⚠️ Amplifier Round {amp_round} 裁剪失败，提前终止。")
                                    break

                                pred_img_path, visible_ticks = res  # ✅ 每轮裁剪后的图像路径
                                print(f"✅ Amplifier 图像已保存: {pred_img_path}")

                                # 2️⃣ 构造 prompt
                                prompt = generate_prompt(
                                    item_name=point_name,
                                    prompt_type="amplifier",
                                    x_ticks=dataset["x_ticks"],
                                    y_ticks=dataset["y_ticks"],
                                    series_color=dataset["series_color"],
                                    axis_types={"x": "categorical", "y": "numerical"},
                                    visible_ticks=visible_ticks,
                                    current_round=amp_round,
                                )

                                print("\n==============================")
                                print(f"📌 Amplifier Round {amp_round} | Point: {point_name}")
                                print(f"🖼️  使用图像路径: {pred_img_path}")
                                print("📋 Prompt 内容如下：\n")
                                print(prompt)
                                print("==============================\n")

                                # 3️⃣ 调用模型预测
                                pred = await call_llm_response(prompt, pred_img_path, point_name)
                                if pred == (-1, -1):
                                    print(f"❌ Amplifier Round {amp_round} 模型预测失败 @ {point_name}")
                                    break

                                x_val, y_val = pred
                                print(f"✅ Amplifier Round {amp_round} 预测结果: y={y_val:.4f}")

                                # 4️⃣ 更新状态
                                y_curr = y_val
                                last_pred = pred
                                history_preds.append(pred)

                                # ---- 绝对误差 ----
                                x_abs_err = abs(x_val - gt[0]) if isinstance(x_val, (int, float)) else float("nan")
                                y_abs_err = abs(y_val - gt[1]) if isinstance(y_val, (int, float)) else float("nan")

                                # ---- 轴范围 ----
                                x_ticks = dataset["x_ticks"]
                                y_ticks = dataset["y_ticks"]

                                def is_numeric_list(lst):
                                    try:
                                        return all(
                                            isinstance(v, (int, float)) or str(v).replace('.', '', 1).isdigit() for v in
                                            lst)
                                    except:
                                        return False

                                # x 轴（可能是分类轴）
                                if is_numeric_list(x_ticks):
                                    x_min, x_max = min(map(float, x_ticks)), max(map(float, x_ticks))
                                    x_range = x_max - x_min
                                else:
                                    x_range = float("nan")

                                # y 轴（始终数值）
                                y_min, y_max = min(map(float, y_ticks)), max(map(float, y_ticks))
                                y_range = y_max - y_min

                                # ---- 归一化误差 ----
                                x_err_over_range = x_abs_err / x_range if (
                                            x_range == x_range and x_range != 0) else float("nan")
                                y_err_over_range = y_abs_err / y_range if y_range != 0 else float("nan")

                                xy_err_over_range = (
                                    (x_err_over_range + y_err_over_range) / 2
                                    if not (np.isnan(x_err_over_range) or np.isnan(y_err_over_range))
                                    else float("nan")
                                )

                                # 5️⃣ 写入记录
                                records.append({
                                    "chart_id": dataset["chart_id"],
                                    "point": point_name,
                                    "prompt_type": "amplifier",
                                    "image_type": image_type,
                                    "run": amp_round,
                                    "image_path": pred_img_path,

                                    "gt_x": gt[0],
                                    "gt_y": gt[1],
                                    "pred_x": x_val,
                                    "pred_y": y_val,

                                    "mae": compute_mae(pred, gt, axis_types),
                                    "x_re": compute_re(pred, gt, axis_types)[0],
                                    "y_re": compute_re(pred, gt, axis_types)[1],

                                    # ⭐ 新增
                                    "x_abs_err": x_abs_err,
                                    "y_abs_err": y_abs_err,
                                    "x_range": x_range,
                                    "y_range": y_range,
                                    "x_err_over_range": x_err_over_range,
                                    "y_err_over_range": y_err_over_range,
                                    "xy_err_over_range": xy_err_over_range,
                                })

                                # records.append({
                                #     "chart_id": dataset["chart_id"],
                                #     "point": point_name,
                                #     "prompt_type": "amplifier",
                                #     "image_type": image_type,
                                #     "run": amp_round,
                                #     "image_path": pred_img_path,
                                #     "gt_x": gt[0],
                                #     "gt_y": gt[1],
                                #     "pred_x": x_val,
                                #     "pred_y": y_val,
                                #     "mae": compute_mae(pred, gt, axis_types),
                                #     "x_re": compute_re(pred, gt, axis_types)[0],
                                #     "y_re": compute_re(pred, gt, axis_types)[1],
                                # })

                            # ✅ 最后一轮的预测结果保留到 feedback_final_results 供后续使用
                            if history_preds:
                                feedback_final_results[(dataset["chart_id"], point_name)] = history_preds[-1]

                            # ✅ 跳出 while valid_runs 的循环（每点只跑一组 amplifier）
                            break

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
                                axis_types={"x": "categorical", "y": "numerical"},
                            )


                        prompt = generate_prompt(
                            item_name=point_name,
                            prompt_type=prompt_type,
                            x_ticks=dataset["x_ticks"],
                            y_ticks=dataset["y_ticks"],
                            series_color=dataset["series_color"],
                            axis_types={"x": "categorical", "y": "numerical"},
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
                        last_pred = pred  # 保持 (x_val, y_val)
                        history_preds.append(pred)

                        if pd.notna(y_val):  # 数值轴是 y，直接判断
                            mae = compute_mae(pred, gt, axis_types)
                            x_re, y_re = compute_re(pred, gt, axis_types)
                        else:
                            mae, x_re, y_re = None, -1, -1

                        # ======== 像素相对误差 ========
                        img_w, img_h = Image.open(image_path).size
                        x_map = get_axis_mapper(dataset["x_ticks"], dataset["x_pixels"], "categorical")
                        y_map = get_axis_mapper(dataset["y_ticks"], dataset["y_pixels"], "numerical")

                        try:
                            px_rel_x = (abs(x_map(pred[0]) - x_map(gt[0])) / img_w if is_number(pred[0]) else -1)
                            px_rel_y = (abs(y_map(pred[1]) - y_map(gt[1])) / img_h if is_number(pred[1]) else -1)
                        except Exception as e:
                            print(f"⚠️ Pixel mapping error: {e}")
                            px_rel_x = px_rel_y = -1

                        # ======== ⭐ 关键新增：数值误差 + 轴范围 + 归一化误差 ========
                        # 数值绝对误差
                        x_abs_err = abs(x_val - gt[0]) if is_number(x_val) else float("nan")
                        y_abs_err = abs(y_val - gt[1]) if is_number(y_val) else float("nan")

                        # --- x_range：仅当 x_ticks 全为数值时才计算 ---
                        x_ticks = dataset["x_ticks"]
                        y_ticks = dataset["y_ticks"]

                        def all_numeric(lst):
                            try:
                                return all(
                                    isinstance(v, (int, float)) or str(v).replace('.', '', 1).isdigit() for v in lst)
                            except:
                                return False

                        # x 轴（可能是分类轴）
                        if all_numeric(x_ticks):
                            x_min, x_max = min(map(float, x_ticks)), max(map(float, x_ticks))
                            x_range = x_max - x_min
                        else:
                            x_range = float("nan")  # 分类轴不计算范围

                        # y 轴（一定是数值轴）
                        y_min, y_max = min(map(float, y_ticks)), max(map(float, y_ticks))
                        y_range = y_max - y_min

                        x_err_over_range = x_abs_err / x_range if x_range != 0 else float("nan")
                        y_err_over_range = y_abs_err / y_range if y_range != 0 else float("nan")

                        if np.isnan(x_err_over_range) or np.isnan(y_err_over_range):
                            xy_err_over_range = float("nan")
                        else:
                            xy_err_over_range = (x_err_over_range + y_err_over_range) / 2.0

                        # ======== 最终记录 ========
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
                            "pixel_rel_x": px_rel_x,
                            "pixel_rel_y": px_rel_y,
                            "x_re": x_re,
                            "y_re": y_re,

                            # ⭐ 新增：用于 normalized error 的全部字段
                            "x_abs_err": x_abs_err,
                            "y_abs_err": y_abs_err,
                            "x_range": x_range,
                            "y_range": y_range,
                            "x_err_over_range": x_err_over_range,
                            "y_err_over_range": y_err_over_range,
                            "xy_err_over_range": xy_err_over_range,
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
                                axis_types={"x": "categorical", "y": "numerical"},
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
                                axis_types={"x": "categorical", "y": "numerical"},
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

    feedback_final_preds = build_feedback_final_preds(records)

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
        chart_ids=["v_bar_044"]
        # chart_ids=args.chart_ids,
    ))



