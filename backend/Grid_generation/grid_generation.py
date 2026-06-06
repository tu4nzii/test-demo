import os
import cv2
import numpy as np
import json
import logging
import sys
import asyncio
import aiohttp
import base64
import itertools
import math
from glob import glob

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_LOG_DIR = os.path.join(BACKEND_DIR, "data", "logs")
LLM_CACHE_DIR = os.path.join(BACKEND_DIR, "data", "llm_cache")
TICK_LABEL_CACHE_DIR = os.path.join(LLM_CACHE_DIR, "tick_labels")
COLOR_CACHE_DIR = os.path.join(LLM_CACHE_DIR, "colors")

# 配置日志系统
log_path = os.path.join(RUNTIME_LOG_DIR, "grid_generation.log")
try:
    # 确保日志目录存在
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
except Exception as e:
    # 如果日志配置失败，至少确保有控制台输出
    print(f"警告: 无法配置日志文件，将只输出到控制台: {e}")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ========== 大模型API配置项 ==========
api_key = "sk-f08TVXG8bJyobMmFvOgh09Bn93vFiuRX8j5iNuSSYQLmqgBd"
url = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

async def call_llm_recognize_ticks(image_path, direction, tick_regions):
    """
    使用大模型识别图像中的刻度值
    """
    # 图像读取与编码
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # 构造提示词
    direction_text = "X轴" if direction == 'x' else "Y轴"
    prompt = f"请识别图片中{direction_text}上的所有刻度值。请仔细查看{direction_text}上的数字标签，只返回所有可见的数字，每个数字占一行。请严格按照这个格式返回：\n```\n数字1\n数字2\n数字3\n...\n```\n不要包含任何其他文字说明。"

    # 构造请求体
    payload = {
        "model": "internvl3-78b",
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

    # 发送请求
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            try:
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # 提取数字
                import re
                numbers = []
                # 尝试从代码块中提取
                code_blocks = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", content)
                if code_blocks:
                    text_content = code_blocks[0]
                else:
                    text_content = content
                
                # 提取所有数字
                for line in text_content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            # 尝试直接转换为数字
                            num = float(line)
                            numbers.append(num)
                        except ValueError:
                            # 尝试提取行中的数字
                            nums = re.findall(r"-?\d+\.?\d*", line)
                            for n in nums:
                                try:
                                    numbers.append(float(n))
                                except ValueError:
                                    continue
                
                logger.debug(f"大模型识别到{direction_text}刻度值: {numbers}")
                return numbers
            except Exception as e:
                logger.error(f"大模型识别失败: {e}")
                return []

def recognize_tick_labels_with_llm(img, ticks, direction, temp_dir=None):
    """
    使用大模型替代OCR识别刻度标签
    """
    # 创建临时文件来保存图像
    import tempfile
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, img)
        
        # 运行异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            tick_values = loop.run_until_complete(call_llm_recognize_ticks(temp_path, direction, ticks))
        finally:
            loop.close()
        
        # 将识别的数值与刻度位置匹配
        result = []
        if direction == 'x' and tick_values:
            # X轴刻度按位置排序
            sorted_ticks = sorted(ticks, key=lambda t: (t[0] + t[2]) // 2)
            # 确保数值数量与刻度数量匹配
            if len(tick_values) > len(sorted_ticks):
                # 如果数值过多，选择最接近数量的
                tick_values = tick_values[:len(sorted_ticks)]
            elif len(tick_values) < len(sorted_ticks):
                # 如果数值不足，使用已有数值插值
                import numpy as np
                if len(tick_values) >= 2:
                    # 插值生成更多数值
                    x_positions = np.linspace(0, 1, len(sorted_ticks))
                    orig_positions = np.linspace(0, 1, len(tick_values))
                    interpolated = np.interp(x_positions, orig_positions, tick_values)
                    tick_values = interpolated.tolist()
            
            # 创建结果列表
            for i, tick in enumerate(sorted_ticks):
                if i < len(tick_values):
                    result.append({
                        'tick': tick,
                        'text': str(tick_values[i])
                    })
        
        elif direction == 'y' and tick_values:
            # Y轴刻度按位置排序（从下到上）
            sorted_ticks = sorted(ticks, key=lambda t: (t[1] + t[3]) // 2, reverse=True)
            # 确保数值数量与刻度数量匹配
            if len(tick_values) > len(sorted_ticks):
                tick_values = tick_values[:len(sorted_ticks)]
            elif len(tick_values) < len(sorted_ticks):
                # 如果数值不足，使用已有数值插值
                import numpy as np
                if len(tick_values) >= 2:
                    # 插值生成更多数值
                    y_positions = np.linspace(0, 1, len(sorted_ticks))
                    orig_positions = np.linspace(0, 1, len(tick_values))
                    interpolated = np.interp(y_positions, orig_positions, tick_values)
                    tick_values = interpolated.tolist()
            
            # 创建结果列表
            for i, tick in enumerate(sorted_ticks):
                if i < len(tick_values):
                    result.append({
                        'tick': tick,
                        'text': str(tick_values[i])
                    })
        
        return result
    except Exception as e:
        logger.error(f"使用大模型识别刻度标签时出错: {e}")
        return []
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

# 导入所需的功能模块
from function_calling.axis.detect_lines import detect_candidate_lines
from function_calling.axis.merge_lines import merge_similar_lines
from function_calling.axis.infer_axes import infer_axes_from_lines
from function_calling.ticks.detect_ticks import scan_pixels_for_ticks
from function_calling.ticks.filter_ticks import filter_ticks
from function_calling.label.recognize_tick_labels import recognize_tick_labels
from function_calling.label.extract_tick_labels_with_llm import extract_tick_labels_with_llm
from function_calling.color.extract_chart_colors import extract_chart_series_color, extract_point_chart_items
from function_calling.image.draw_grid_from_ticks import draw_grid_from_ticks
from utils.image_io import load_image, save_image


NUMERIC_AXIS_TYPE = "\u6570\u503c\u8f74"
TEXT_AXIS_TYPE = "\u6587\u5b57\u8f74"


def normalize_axis_repair_hint(axis_repair_hint):
    hint = axis_repair_hint if isinstance(axis_repair_hint, dict) else {}

    def as_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "missing"}
        return False

    x_axis_missing = as_bool(hint.get("x_axis_missing", hint.get("x", False)))
    y_axis_missing = as_bool(hint.get("y_axis_missing", hint.get("y", False)))
    return {
        "x_axis_missing": x_axis_missing,
        "y_axis_missing": y_axis_missing,
        "x_ticks_missing": as_bool(hint.get("x_ticks_missing", False)) or x_axis_missing,
        "y_ticks_missing": as_bool(hint.get("y_ticks_missing", False)) or y_axis_missing,
        "confidence": hint.get("confidence", 0),
        "reason": str(hint.get("reason", "") or ""),
    }


def axis_repair_enabled(axis_repair_hint):
    hint = normalize_axis_repair_hint(axis_repair_hint)
    return any(
        hint.get(key)
        for key in ("x_axis_missing", "y_axis_missing", "x_ticks_missing", "y_ticks_missing")
    )


def _line_len(line):
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


def _horizontal_axis_from_lines(lines, w, h):
    best = None
    best_score = float("-inf")
    for line in lines or []:
        x1, y1, x2, y2 = [int(v) for v in line]
        if abs(y1 - y2) > 8:
            continue
        length = abs(x2 - x1)
        if length < max(30, w * 0.25):
            continue
        y = int(round((y1 + y2) / 2))
        score = length + y * 0.8
        if score > best_score:
            best_score = score
            best = [min(x1, x2), y, max(x1, x2), y]
    return best


def _vertical_axis_from_lines(lines, w, h):
    best = None
    best_score = float("-inf")
    for line in lines or []:
        x1, y1, x2, y2 = [int(v) for v in line]
        if abs(x1 - x2) > 8:
            continue
        length = abs(y2 - y1)
        if length < max(30, h * 0.25):
            continue
        x = int(round((x1 + x2) / 2))
        score = length + (w - x) * 0.35
        if score > best_score:
            best_score = score
            best = [x, max(y1, y2), x, min(y1, y2)]
    return best


def _bar_boxes(img, chart_type):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 35, 40]), np.array([179, 255, 255]))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    min_area = max(20, w * h * 0.0003)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < min_area:
            continue
        if chart_type == "h_bar":
            if bw < 12 or bh < 4 or bw < bh * 2:
                continue
        elif chart_type == "v_bar":
            if bh < 12 or bw < 4 or bh < bw * 2:
                continue
        else:
            continue
        boxes.append((int(x), int(y), int(bw), int(bh)))
    if chart_type == "h_bar":
        if len(boxes) >= 3:
            median_height = float(np.median([box[3] for box in boxes]))
            boxes = [box for box in boxes if box[3] >= max(4, median_height * 0.45)]
        return sorted(boxes, key=lambda box: box[1])
    if len(boxes) >= 3:
        median_width = float(np.median([box[2] for box in boxes]))
        boxes = [box for box in boxes if box[2] >= max(4, median_width * 0.45)]
    return sorted(boxes, key=lambda box: box[0])


def _axis_y(axis):
    return int(round((axis[1] + axis[3]) / 2))


def _axis_x(axis):
    return int(round((axis[0] + axis[2]) / 2))


def _dedupe_pixels(pixels, tolerance=4):
    values = sorted(int(round(pixel)) for pixel in pixels)
    groups = []
    for value in values:
        if not groups or abs(value - groups[-1][-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [int(round(float(np.median(group)))) for group in groups]


def _tick_group_count(ticks, direction):
    if not ticks:
        return 0
    if direction == "x":
        pixels = [(tick[0] + tick[2]) / 2 for tick in ticks]
    else:
        pixels = [(tick[1] + tick[3]) / 2 for tick in ticks]
    return len(_dedupe_pixels(pixels))


def _is_bar_category_axis(chart_type, direction):
    chart_type = (chart_type or "").lower()
    direction = (direction or "").lower()
    return (chart_type == "h_bar" and direction == "y") or (
        chart_type == "v_bar" and direction == "x"
    )


def _required_tick_count(chart_type, direction):
    return 1 if _is_bar_category_axis(chart_type, direction) else 2


def _horizontal_gridline_plot_span(lines, x_axis, image_shape):
    if x_axis is None:
        return None

    h, w = image_shape[:2]
    axis_y = _axis_y(x_axis)
    axis_left, axis_right = sorted([int(x_axis[0]), int(x_axis[2])])
    candidates = []
    min_length = max(40, (axis_right - axis_left) * 0.45, w * 0.2)
    for line in lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(y1 - y2) > 4:
            continue
        left, right = sorted([x1, x2])
        length = right - left
        if length < min_length:
            continue
        y = int(round((y1 + y2) / 2))
        if y < h * 0.05 or y > axis_y + max(8, h * 0.01):
            continue
        candidates.append((left, right, y))

    if not candidates:
        return None

    left = int(round(float(np.median([item[0] for item in candidates]))))
    right = int(np.percentile([item[1] for item in candidates], 90))
    y_pixels = _dedupe_pixels([item[2] for item in candidates])
    if axis_y not in y_pixels:
        y_pixels.append(axis_y)
    y_pixels = sorted(_dedupe_pixels(y_pixels), reverse=True)
    return left, right, y_pixels


def repair_missing_axes(img, merged_lines, x_axis, y_axis, chart_type, axis_repair_hint):
    hint = normalize_axis_repair_hint(axis_repair_hint)
    if not axis_repair_enabled(hint):
        return x_axis, y_axis, []

    chart_type = (chart_type or "").lower()
    h, w = img.shape[:2]
    boxes = _bar_boxes(img, chart_type)

    if x_axis is None and not hint["x_axis_missing"]:
        x_axis = _horizontal_axis_from_lines(merged_lines, w, h)
    if y_axis is None and not hint["y_axis_missing"]:
        y_axis = _vertical_axis_from_lines(merged_lines, w, h)

    if not boxes:
        return x_axis, y_axis, boxes

    left = min(box[0] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    top = min(box[1] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)

    if chart_type == "h_bar":
        if hint["y_axis_missing"]:
            bottom_y = _axis_y(x_axis) if x_axis is not None else min(h - 1, bottom + max(8, int(np.median([b[3] for b in boxes]))))
            top_y = max(0, top)
            y_axis = [left, bottom_y, left, top_y]
        if hint["x_axis_missing"]:
            axis_x = _axis_x(y_axis) if y_axis is not None else left
            axis_y = max(y_axis[1], y_axis[3]) if y_axis is not None else min(h - 1, bottom + 8)
            x_axis = [axis_x, axis_y, right, axis_y]

    elif chart_type == "v_bar":
        if hint["x_axis_missing"]:
            axis_y = max(box[1] + box[3] for box in boxes)
            axis_x = _axis_x(y_axis) if y_axis is not None else left
            x_axis = [axis_x, axis_y, right, axis_y]
        if hint["y_axis_missing"]:
            grid_span = _horizontal_gridline_plot_span(merged_lines, x_axis, img.shape)
            axis_y = _axis_y(x_axis) if x_axis is not None else bottom
            if grid_span:
                grid_left, grid_right, grid_y_pixels = grid_span
                axis_x = grid_left
                top_y = min(grid_y_pixels)
                if x_axis is not None:
                    x_axis = [axis_x, axis_y, max(grid_right, x_axis[2], x_axis[0]), axis_y]
            else:
                axis_x = min(x_axis[0], x_axis[2]) if x_axis is not None else left
                top_y = top
            y_axis = [axis_x, axis_y, axis_x, top_y]

    return x_axis, y_axis, boxes


def _synthetic_bar_tick_pixels(chart_type, direction, boxes):
    if not boxes:
        return []
    if chart_type == "h_bar" and direction == "y":
        return sorted([int(round(y + h / 2)) for _, y, _, h in boxes], reverse=True)
    if chart_type == "v_bar" and direction == "x":
        return sorted([int(round(x + w / 2)) for x, _, w, _ in boxes])
    return []


def synthesize_tick_pixels_for_missing_axis(
    chart_type,
    direction,
    axis,
    boxes,
    tick_values,
    axis_repair_hint,
):
    hint = normalize_axis_repair_hint(axis_repair_hint)
    if not (hint.get(f"{direction}_axis_missing") or hint.get(f"{direction}_ticks_missing")):
        return []

    chart_type = (chart_type or "").lower()
    bar_pixels = _synthetic_bar_tick_pixels(chart_type, direction, boxes)
    if bar_pixels:
        return bar_pixels

    if axis is None or len(tick_values or []) < 2:
        return []

    count = len(tick_values)
    if direction == "x":
        start, end = sorted([int(axis[0]), int(axis[2])])
    else:
        low, high = sorted([int(axis[1]), int(axis[3])])
        start, end = high, low
    return [int(round(value)) for value in np.linspace(start, end, count)]


def ticks_from_pixels(pixels, axis, direction):
    if axis is None:
        return []
    ticks = []
    if direction == "x":
        y = _axis_y(axis)
        for x in pixels:
            ticks.append([int(x), y + 1, int(x), y + 6])
    else:
        x = _axis_x(axis)
        for y in pixels:
            ticks.append([x - 6, int(y), x - 1, int(y)])
    return ticks


def infer_tick_pixels_from_gridlines(lines, x_axis, y_axis, direction, image_shape):
    """Infer tick positions from full plot gridlines when short tick marks are absent."""
    if x_axis is None or y_axis is None:
        return []

    h, w = image_shape[:2]
    axis_y = _axis_y(x_axis)
    axis_x = _axis_x(y_axis)
    plot_left = max(0, min(axis_x, min(x_axis[0], x_axis[2])))
    plot_right = min(w - 1, max(x_axis[0], x_axis[2]))
    plot_top = max(0, min(y_axis[1], y_axis[3]))
    plot_bottom = min(h - 1, max(axis_y, y_axis[1], y_axis[3]))
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)
    border_margin_x = max(6, int(round(plot_width * 0.015)))
    border_margin_y = max(6, int(round(plot_height * 0.03)))

    pixels = []
    for line in lines or []:
        x1, y1, x2, y2 = [int(v) for v in line]
        if direction == "y":
            if abs(y1 - y2) > 4:
                continue
            y = int(round((y1 + y2) / 2))
            if y <= plot_top + border_margin_y or y >= plot_bottom - border_margin_y:
                continue
            left, right = sorted([x1, x2])
            overlap = max(0, min(right, plot_right) - max(left, plot_left))
            if overlap >= plot_width * 0.35:
                pixels.append(y)
        else:
            if abs(x1 - x2) > 4:
                continue
            x = int(round((x1 + x2) / 2))
            if x <= plot_left + border_margin_x or x >= plot_right - border_margin_x:
                continue
            top, bottom = sorted([y1, y2])
            overlap = max(0, min(bottom, plot_bottom) - max(top, plot_top))
            if overlap >= plot_height * 0.35:
                pixels.append(x)

    if not pixels:
        return []

    pixels = sorted(set(int(p) for p in pixels))
    if direction == "y":
        pixels = sorted(pixels, reverse=True)
    return pixels


def _group_projection_peaks(scores, *, min_score=0.08, max_gap=3):
    peak_indices = [index for index, score in enumerate(scores) if score >= min_score]
    if not peak_indices:
        return []

    groups = []
    for index in peak_indices:
        if not groups or index - groups[-1][-1] > max_gap:
            groups.append([index])
        else:
            groups[-1].append(index)

    peaks = []
    for group in groups:
        weights = np.array([float(scores[index]) for index in group])
        if float(weights.sum()) <= 0:
            center = int(round(sum(group) / len(group)))
        else:
            center = int(round(float(np.average(group, weights=weights))))
        peaks.append((center, float(max(weights)), len(group)))
    return peaks


def infer_point_chart_grid_pixels_by_projection(img, x_axis, y_axis, direction, expected_count=None):
    """Detect faint dashed plot gridlines by projecting light gray pixels.

    This is a fallback for point charts where Hough-based tick detection locks on
    to bubble/text fragments instead of the real dashed grid. It deliberately
    looks only inside the plot area and is used only after the ordinary tick
    result is found to be suspicious.
    """
    if img is None or x_axis is None or y_axis is None:
        return []

    h, w = img.shape[:2]
    axis_x = _axis_x(y_axis)
    axis_y = _axis_y(x_axis)
    plot_top = max(0, min(int(y_axis[1]), int(y_axis[3])))
    plot_bottom = min(h - 1, max(axis_y, int(y_axis[1]), int(y_axis[3])))
    if plot_bottom - plot_top < max(40, h * 0.15):
        return []

    b, g, r = cv2.split(img)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    gray_grid_mask = (maxc - minc <= 12) & (maxc >= 180) & (maxc <= 245)

    direction = (direction or "").lower()
    if direction == "x":
        y1 = max(0, plot_top)
        y2 = min(h, plot_bottom + 1)
        x1 = max(0, axis_x - 12)
        x2 = min(w, int(round(w * 0.88)))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=0)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.08, max_gap=3)
        pixels = [x1 + center for center, score, width in peaks if width <= 10]
        pixels = sorted(_dedupe_pixels(pixels))
    elif direction == "y":
        x_grid = infer_point_chart_grid_pixels_by_projection(
            img, x_axis, y_axis, "x", expected_count=None
        )
        x1 = min(x_grid) if len(x_grid) >= 2 else max(0, axis_x - 4)
        x2 = max(x_grid) if len(x_grid) >= 2 else min(w - 1, max(int(x_axis[0]), int(x_axis[2])))
        x1 = max(0, x1)
        x2 = min(w, x2 + 1)
        y1 = max(0, plot_top)
        y2 = min(h, plot_bottom - max(18, int(round((plot_bottom - plot_top) * 0.06))))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=1)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.08, max_gap=3)
        pixels = [y1 + center for center, score, width in peaks if width <= 12]
        pixels = sorted(_dedupe_pixels(pixels), reverse=True)
    else:
        return []

    if expected_count and len(pixels) > expected_count:
        if direction == "x":
            pixels = pixels[:expected_count]
        else:
            pixels = pixels[:expected_count]
    return pixels


def infer_point_chart_grid_pixels_for_missing_axes(img, direction):
    """Infer candidate plot gridlines without trusting detected axis lines.

    Real-world point charts often omit both axis strokes while keeping a light
    background grid. This projection is only used inside the missing-axis repair
    path, so regular dataset charts with visible axes keep the established flow.
    """
    if img is None:
        return []

    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    gray_grid_mask = (maxc - minc <= 12) & (maxc >= 180) & (maxc <= 245)

    direction = (direction or "").lower()
    if direction == "x":
        x1 = max(0, int(round(w * 0.04)))
        x2 = min(w, int(round(w * 0.88)))
        y1 = max(0, int(round(h * 0.20)))
        y2 = min(h, int(round(h * 0.80)))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=0)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.075, max_gap=3)
        pixels = [x1 + center for center, score, width in peaks if width <= 12]
        return sorted(_dedupe_pixels(pixels))

    if direction == "y":
        x_candidates = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
        if len(x_candidates) >= 2:
            x1 = max(0, min(x_candidates) - 2)
            x2 = min(w, max(x_candidates) + 3)
        else:
            x1 = max(0, int(round(w * 0.04)))
            x2 = min(w, int(round(w * 0.88)))
        y1 = max(0, int(round(h * 0.20)))
        y2 = min(h, int(round(h * 0.82)))
        if y2 <= y1 or x2 <= x1:
            return []
        scores = gray_grid_mask[y1:y2, x1:x2].mean(axis=1)
        if len(scores) >= 3:
            scores = np.convolve(scores, np.ones(3) / 3, mode="same")
        peaks = _group_projection_peaks(scores, min_score=0.075, max_gap=3)
        pixels = [y1 + center for center, score, width in peaks if width <= 14]
        return sorted(_dedupe_pixels(pixels), reverse=True)

    return []


def _finite_numeric_sequence(values):
    numeric = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numeric.append(number)
    return numeric


def _fit_tick_pixels(values, pixels, scale):
    numeric = np.array(values, dtype=float)
    candidate_pixels = np.array(pixels, dtype=float)
    if len(numeric) != len(candidate_pixels) or len(numeric) < 2:
        return float("inf")
    if scale == "log":
        if np.any(numeric <= 0):
            return float("inf")
        numeric = np.log10(numeric)
    try:
        coeff = np.polyfit(numeric, candidate_pixels, 1)
        predicted = np.polyval(coeff, numeric)
    except Exception:
        return float("inf")
    span = max(1.0, float(candidate_pixels.max() - candidate_pixels.min()))
    return float(np.sqrt(np.mean((predicted - candidate_pixels) ** 2)) / span)


def _values_prefer_log_scale(values):
    numeric = _finite_numeric_sequence(values)
    if not numeric or len(numeric) < 3 or any(value <= 0 for value in numeric):
        return False
    linear_gaps = np.diff(numeric)
    log_gaps = np.diff(np.log10(numeric))
    if np.any(linear_gaps == 0) or np.any(log_gaps == 0):
        return False
    linear_cv = float(np.std(np.abs(linear_gaps)) / max(1e-9, np.mean(np.abs(linear_gaps))))
    log_cv = float(np.std(np.abs(log_gaps)) / max(1e-9, np.mean(np.abs(log_gaps))))
    return linear_cv > 0.35 and log_cv < max(0.35, linear_cv * 0.55)


def select_projected_tick_pixels_for_values(projected_pixels, tick_values, direction):
    """Choose the gridline subset that best matches the visible tick labels."""
    numeric = _finite_numeric_sequence(tick_values)
    if not numeric or len(numeric) < 2:
        return [], "linear"

    direction = (direction or "").lower()
    pixels = sorted(_dedupe_pixels(projected_pixels or []), reverse=(direction == "y"))
    target_count = len(numeric)
    if len(pixels) < target_count:
        return [], "linear"
    if len(pixels) == target_count:
        scale = axis_scale_from_ticks_and_pixels(numeric, pixels)
        return pixels, scale

    scales = ["linear"]
    if _values_prefer_log_scale(numeric):
        scales.insert(0, "log")

    best = None
    total_combinations = math.comb(len(pixels), target_count)
    if total_combinations > 20000:
        # Keep the search bounded while preserving plot extremes and quantiles.
        quantile_indices = {
            int(round(value))
            for value in np.linspace(0, len(pixels) - 1, min(len(pixels), target_count * 4))
        }
        pixels = [pixels[index] for index in sorted(quantile_indices)]

    for combo in itertools.combinations(pixels, target_count):
        for scale in scales:
            score = _fit_tick_pixels(numeric, combo, scale)
            if best is None or score < best[0]:
                best = (score, scale, list(combo))

    if best is None:
        return [], "linear"
    return best[2], best[1]


def axis_scale_from_ticks_and_pixels(tick_values, pixel_positions):
    numeric = _finite_numeric_sequence(tick_values)
    if not numeric or len(numeric) < 3 or len(pixel_positions or []) != len(numeric):
        return "linear"
    if not _values_prefer_log_scale(numeric):
        return "linear"
    linear_score = _fit_tick_pixels(numeric, pixel_positions, "linear")
    log_score = _fit_tick_pixels(numeric, pixel_positions, "log")
    if log_score < linear_score * 0.65:
        return "log"
    return "linear"


def _gray_grid_mask(img):
    b, g, r = cv2.split(img)
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    return (maxc - minc <= 12) & (maxc >= 180) & (maxc <= 245)


def _mask_groups(values):
    groups = []
    for value in values:
        value = int(value)
        if not groups or value - groups[-1][-1] > 3:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [(group[0], group[-1], len(group)) for group in groups if len(group) >= 2]


def _regular_dash_span(groups):
    if len(groups) < 6:
        return None
    centers = [int(round((left + right) / 2)) for left, right, _ in groups]
    best = []
    for start_index in range(len(groups)):
        seq = [start_index]
        last_center = centers[start_index]
        for index in range(start_index + 1, len(groups)):
            gap = centers[index] - last_center
            if 5 <= gap <= 11:
                seq.append(index)
                last_center = centers[index]
            elif gap > 16:
                break
        if len(seq) > len(best):
            best = seq
    if len(best) < 6:
        return None
    return groups[best[0]][0], groups[best[-1]][1]


def _regular_dash_left_in_window(groups, min_left, max_left):
    centers = [int(round((left + right) / 2)) for left, right, _ in groups]
    for start_index, (left, _, _) in enumerate(groups):
        if left < min_left or left > max_left:
            continue
        seq_len = 1
        last_center = centers[start_index]
        for index in range(start_index + 1, len(groups)):
            gap = centers[index] - last_center
            if 5 <= gap <= 11:
                seq_len += 1
                last_center = centers[index]
            elif gap > 16:
                break
        if seq_len >= 6:
            return int(left)
    return None


def infer_point_chart_plot_bounds_from_horizontal_grid(img, y_pixels, x_pixels):
    """Estimate plot left/right from the dashed horizontal grid, not tick pixels."""
    if img is None or len(y_pixels or []) < 2:
        return None
    h, w = img.shape[:2]
    mask = _gray_grid_mask(img)
    spans = []
    left_edges = []
    tick_min = min(x_pixels or [0])
    tick_max = max(x_pixels or [w - 1])
    min_left = max(0, int(round(w * 0.04)))
    max_left = max(min_left, tick_min - max(35, int(round(w * 0.05))))
    for y in y_pixels:
        y = int(round(y))
        band = mask[max(0, y - 2) : min(h, y + 3), :]
        if band.size == 0:
            continue
        scores = band.mean(axis=0)
        groups = _mask_groups(np.where(scores >= 0.2)[0])
        left_edge = _regular_dash_left_in_window(groups, min_left, max_left)
        if left_edge is not None:
            left_edges.append(left_edge)
        span = _regular_dash_span(groups)
        if span is not None:
            spans.append(span)
    if not spans and not left_edges:
        return None

    left_candidates = [
        left for left, right in spans
        if left < tick_min - max(20, int(round(w * 0.03)))
    ]
    right_candidates = [
        right for left, right in spans
        if right > tick_max + max(10, int(round(w * 0.015)))
    ]
    plot_left = int(round(float(np.median(left_edges or left_candidates or [min(left for left, _ in spans)]))))
    plot_right = int(round(float(np.median(right_candidates or [max(right for _, right in spans)]))))
    if plot_right <= plot_left:
        return None
    return plot_left, plot_right


def _vertical_grid_centers_near_x(mask, x_pixels, y_low, y_high):
    h, w = mask.shape[:2]
    centers = []
    for x in x_pixels or []:
        x = int(round(x))
        band = mask[:, max(0, x - 2) : min(w, x + 3)]
        if band.size == 0:
            continue
        scores = band.mean(axis=1)
        groups = _mask_groups(np.where(scores >= 0.2)[0])
        for top, bottom, width in groups:
            center = int(round((top + bottom) / 2))
            if y_low <= center <= y_high and width <= 10:
                centers.append(center)
    return centers


def _supported_center_near_expected(centers, expected, tolerance=7):
    if not centers:
        return None
    groups = []
    for center in sorted(int(c) for c in centers):
        if not groups or center - groups[-1][-1] > tolerance:
            groups.append([center])
        else:
            groups[-1].append(center)
    groups = [group for group in groups if len(group) >= 3]
    if not groups:
        return None
    best = min(groups, key=lambda group: abs(float(np.median(group)) - expected))
    return int(round(float(np.median(best))))


def infer_point_chart_plot_vertical_bounds_from_grid(img, y_pixels, x_pixels):
    """Estimate plot top/bottom from vertical grid extent around selected x ticks."""
    if img is None or len(y_pixels or []) < 2 or len(x_pixels or []) < 2:
        return None
    sorted_y = sorted(int(round(value)) for value in y_pixels)
    gaps = [sorted_y[index + 1] - sorted_y[index] for index in range(len(sorted_y) - 1)]
    gaps = [gap for gap in gaps if gap > 4]
    if not gaps:
        return None
    gap = float(np.median(gaps))
    top_tick = min(sorted_y)
    bottom_tick = max(sorted_y)
    expected_top = top_tick - gap
    expected_bottom = bottom_tick + gap
    mask = _gray_grid_mask(img)

    top_centers = _vertical_grid_centers_near_x(
        mask,
        x_pixels,
        max(0, int(round(top_tick - gap * 1.35))),
        int(round(top_tick - gap * 0.25)),
    )
    bottom_centers = _vertical_grid_centers_near_x(
        mask,
        x_pixels,
        int(round(bottom_tick + gap * 0.25)),
        min(mask.shape[0] - 1, int(round(bottom_tick + gap * 1.25))),
    )
    plot_top = _supported_center_near_expected(top_centers, expected_top)
    plot_bottom = _supported_center_near_expected(bottom_centers, expected_bottom)
    if plot_top is None and plot_bottom is None:
        return None
    plot_top = int(plot_top if plot_top is not None else top_tick)
    plot_bottom = int(plot_bottom if plot_bottom is not None else bottom_tick)
    if plot_bottom <= plot_top:
        return None
    return plot_top, plot_bottom


def point_chart_tick_pixels_are_suspicious(pixel_positions, axis, direction, expected_count):
    if axis is None or expected_count is None or expected_count < 2:
        return False
    if len(pixel_positions or []) < 2:
        return True

    if direction == "x":
        axis_span = abs(int(axis[2]) - int(axis[0]))
        pixel_span = max(pixel_positions) - min(pixel_positions)
    else:
        axis_span = abs(int(axis[3]) - int(axis[1]))
        pixel_span = max(pixel_positions) - min(pixel_positions)

    if axis_span <= 0:
        return False
    if len(pixel_positions) != expected_count and pixel_span < axis_span * 0.85:
        return True
    return pixel_span < axis_span * 0.35


def _regular_grid_pixels(values, *, min_count=4):
    pixels = sorted(_dedupe_pixels(values))
    if len(pixels) < min_count:
        return []

    gaps = [pixels[index + 1] - pixels[index] for index in range(len(pixels) - 1)]
    usable_gaps = [gap for gap in gaps if gap >= 8]
    if not usable_gaps:
        return []
    median_gap = float(np.median(usable_gaps))
    if median_gap <= 0:
        return []

    groups = [[pixels[0]]]
    for previous, current in zip(pixels, pixels[1:]):
        gap = current - previous
        if median_gap * 0.45 <= gap <= median_gap * 2.25:
            groups[-1].append(current)
        else:
            groups.append([current])

    groups = [group for group in groups if len(group) >= min_count]
    if not groups:
        return []
    return max(groups, key=len)


def refine_point_chart_axes_from_gridlines(lines, x_axis, y_axis, chart_type, image_shape):
    """Recover scatter/bubble plot bounds when an internal gridline is mistaken for an axis.

    Some real-world bubble charts render faint axes and stronger internal grid
    lines. The generic axis scorer can then choose an internal vertical gridline
    as the y-axis. This correction is intentionally conservative: it only runs
    for point charts with a regular grid and only changes axes when the inferred
    plot boundary is substantially different from the current axis.
    """
    chart_type = (chart_type or "").lower()
    if chart_type not in {"scatter", "bubble"} or x_axis is None or y_axis is None:
        return x_axis, y_axis, False

    h, w = image_shape[:2]
    verticals = []
    for line in lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(x1 - x2) > 4:
            continue
        top, bottom = sorted([y1, y2])
        length = bottom - top
        if length < h * 0.45:
            continue
        if top > h * 0.35 or bottom < h * 0.55:
            continue
        x = int(round((x1 + x2) / 2))
        verticals.append((x, top, bottom))

    x_grid = _regular_grid_pixels([item[0] for item in verticals], min_count=4)
    if len(x_grid) < 4:
        return x_axis, y_axis, False

    grid_left, grid_right = min(x_grid), max(x_grid)
    grid_width = grid_right - grid_left
    if grid_width < w * 0.25:
        return x_axis, y_axis, False

    relevant_verticals = [item for item in verticals if grid_left - 3 <= item[0] <= grid_right + 3]
    vertical_top = int(np.percentile([item[1] for item in relevant_verticals], 25))
    vertical_bottom = int(np.percentile([item[2] for item in relevant_verticals], 50))

    horizontals = []
    for line in lines or []:
        x1, y1, x2, y2 = [int(value) for value in line]
        if abs(y1 - y2) > 4:
            continue
        left, right = sorted([x1, x2])
        y = int(round((y1 + y2) / 2))
        if y < vertical_top - 8 or y > vertical_bottom + max(12, int(h * 0.05)):
            continue
        overlap = max(0, min(right, grid_right) - max(left, grid_left))
        if overlap < grid_width * 0.55:
            continue
        horizontals.append((left, right, y))

    y_grid = _regular_grid_pixels([item[2] for item in horizontals], min_count=4)
    if len(y_grid) < 4:
        return x_axis, y_axis, False

    candidate_top = min(y_grid)
    candidate_bottom = max(y_grid)
    candidate_left = grid_left
    candidate_right = int(np.median([item[1] for item in horizontals if item[2] in set(y_grid)]))

    current_left = _axis_x(y_axis)
    current_bottom = _axis_y(x_axis)
    left_shift = abs(current_left - candidate_left)
    bottom_shift = abs(current_bottom - candidate_bottom)
    if left_shift < max(18, grid_width * 0.06) and bottom_shift < max(10, h * 0.025):
        return x_axis, y_axis, False

    if candidate_right <= candidate_left or candidate_bottom <= candidate_top:
        return x_axis, y_axis, False

    repaired_x_axis = [candidate_left, candidate_bottom, candidate_right, candidate_bottom]
    repaired_y_axis = [candidate_left, candidate_bottom, candidate_left, candidate_top]
    return repaired_x_axis, repaired_y_axis, True


def apply_bar_geometry_repair_hint(
    chart_type,
    axis_repair_hint,
    boxes,
    x_tick_count=0,
    y_tick_count=0,
):
    """Enable repair only for obvious bar/tick mismatches.

    The MLLM hint stays the primary switch. This guard catches cases where the
    model says axes are present but CV finds far fewer category ticks than bars.
    """
    hint = normalize_axis_repair_hint(axis_repair_hint)
    chart_type = (chart_type or "").lower()
    if chart_type not in {"h_bar", "v_bar"} or len(boxes or []) < 1:
        return hint

    min_expected_ticks = max(1, int(np.ceil(len(boxes) * 0.6)))
    if chart_type == "h_bar":
        if y_tick_count < min_expected_ticks:
            hint["y_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {y_tick_count} y ticks for {len(boxes)} bars"
        if x_tick_count < 2:
            hint["x_axis_missing"] = True
            hint["x_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {x_tick_count} numeric x ticks"
    elif chart_type == "v_bar":
        if x_tick_count < min_expected_ticks:
            hint["x_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {x_tick_count} x ticks for {len(boxes)} bars"
        if y_tick_count < 2:
            hint["y_axis_missing"] = True
            hint["y_ticks_missing"] = True
            hint["reason"] = (
                hint.get("reason") or ""
            ) + f" | geometry repair: {y_tick_count} numeric y ticks"
    return hint


def _numeric_sequence(values):
    numeric = []
    for value in values or []:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return None
    return numeric


def _numeric_ticks_from_unit_labels(values):
    numeric = []
    for value in values or []:
        if isinstance(value, (int, float)):
            numeric.append(float(value))
            continue
        text = str(value).replace(",", "").strip()
        match = __import__("re").search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            numeric.append(float(match.group(0)))
        except ValueError:
            return None
    return numeric if len(numeric) == len(values or []) else None


def coerce_chart_axis_numeric_ticks(chart_type, axis, tick_values, axis_type):
    """Value axes often show numeric ticks with units, e.g. '$25' or '65 gr'."""
    chart_type = (chart_type or "").lower()
    axis = (axis or "").lower()
    value_axis = (
        chart_type in {"scatter", "bubble"}
        or (chart_type in {"v_bar", "line"} and axis == "y")
        or (chart_type == "h_bar" and axis == "x")
    )
    if not value_axis:
        return tick_values, axis_type
    numeric = _numeric_ticks_from_unit_labels(tick_values)
    if numeric is None or len(numeric) < 2:
        return tick_values, axis_type
    return numeric, NUMERIC_AXIS_TYPE


def add_missing_numeric_axis_endpoints(direction, axis, pixel_positions, tick_values):
    """Add missing numeric endpoint pixels when tick labels include them."""
    if axis is None or len(pixel_positions) < 2 or len(tick_values or []) <= len(pixel_positions):
        return pixel_positions

    numeric = _numeric_sequence(tick_values)
    if not numeric or len(numeric) < 2:
        return pixel_positions

    pixels = list(pixel_positions)
    gaps = np.diff(sorted(pixels))
    if len(gaps) == 0:
        return pixels
    median_gap = float(np.median(gaps))
    tolerance = max(6.0, median_gap * 0.3)

    if direction == "x":
        start, end = sorted([int(axis[0]), int(axis[2])])
        first_gap = pixels[0] - start
        if len(numeric) > len(pixels) and first_gap > 0 and abs(first_gap - median_gap) <= tolerance:
            pixels.insert(0, start)
        last_gap = end - pixels[-1]
        if len(numeric) > len(pixels) and last_gap > 0 and abs(last_gap - median_gap) <= tolerance:
            pixels.append(end)
    else:
        low, high = sorted([int(axis[1]), int(axis[3])])
        first_gap = high - pixels[0]
        if len(numeric) > len(pixels) and first_gap > 0 and abs(first_gap - median_gap) <= tolerance:
            pixels.insert(0, high)
        last_gap = pixels[-1] - low
        if len(numeric) > len(pixels) and last_gap > 0 and abs(last_gap - median_gap) <= tolerance:
            pixels.append(low)

    return pixels


def draw_basic_grid(img, x_pixels, y_pixels, x_axis, y_axis):
    """
    绘制基础网格 - 只延伸短横线形成网格图
    """
    canvas = img.copy()
    # 绘制坐标轴
    cv2.line(canvas, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 2)
    cv2.line(canvas, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (255, 0, 0), 2)
    
    # 绘制水平网格线（Y方向）
    x_min, x_max = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    y_min, y_max = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
    
    # 绘制垂直网格线
    for x_pix in x_pixels:
        cv2.line(canvas, (x_pix, y_min), (x_pix, y_max), (180, 180, 180), 1, cv2.LINE_AA)
    
    # 绘制水平网格线
    for y_pix in y_pixels:
        cv2.line(canvas, (x_min, y_pix), (x_max, y_pix), (180, 180, 180), 1, cv2.LINE_AA)
    
    return canvas

def calculate_max_decimal_places(ticks):
    """
    计算一组刻度的最大小数位数
    对输入的所有刻度值，先消除误差，再计算最大小数位
    """
    max_decimal = 0
    for tick in ticks:
        if isinstance(tick, (int, float)) or (isinstance(tick, str) and tick.replace('.', '', 1).isdigit()):
            try:
                # 先转换为浮点数
                if isinstance(tick, str):
                    num = float(tick)
                else:
                    num = tick
                # 消除浮点数误差
                formatted = f"{num:.12f}"
                # 计算小数位数
                decimal_places = count_decimal_places(formatted)
                max_decimal = max(max_decimal, decimal_places)
            except:
                pass
    return max_decimal

def format_tick_value(value, decimal_places):
    """
    格式化刻度值，保留指定的小数位数
    关键：使用字符串格式化消除浮点数误差
    """
    logger.debug(f"格式化前原始值: {value}, 目标小数位数: {decimal_places}")
    
    try:
        # 转换为浮点数
        if isinstance(value, str):
            num = float(value)
        else:
            num = float(value)
        
        # 第一步：误差消除
        # 使用字符串格式化到12位小数，再四舍五入到decimal_places+1位
        num = round(float(f"{num:.12f}"), decimal_places + 1)
        logger.debug(f"误差消除后: {num}")
        
        # 第二步：强制格式化到指定小数位数
        if decimal_places > 0:
            formatted = f"{num:.{decimal_places}f}"
        else:
            formatted = f"{int(round(num))}"
        
        logger.debug(f"格式化后: {formatted}")
        return formatted
    except Exception as e:
        logger.error(f"格式化刻度值时出错: {e}")
        # 文字轴或其他类型，直接返回
        return str(value)

def draw_encrypted_grid(
    img,
    x_pixels,
    y_pixels,
    x_ticks_encrypted,
    y_ticks_encrypted,
    x_axis,
    y_axis,
    x_axis_type="数值轴",
    y_axis_type="数值轴",
    base_x_pixels=None,
    base_y_pixels=None,
):
    """
    绘制加密网格 - 在基础网格上添加加密刻度线、文本框和文本
    只对数值轴加密生成的部分添加网格线、文本框和文本
    文字轴不加密
    """
    canvas = draw_basic_grid(
        img,
        base_x_pixels if base_x_pixels is not None else x_pixels,
        base_y_pixels if base_y_pixels is not None else y_pixels,
        x_axis,
        y_axis,
    )

    # Draw encrypted grid lines only at inserted midpoint positions. Original
    # grid lines are already rendered by draw_basic_grid above.
    grid_overlay = canvas.copy()
    encrypted_grid_color = (0, 0, 255)
    encrypted_grid_alpha = 0.35
    x_min, x_max = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
    y_min, y_max = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])

    drawn_x_grid_lines = 0
    if x_axis_type == "数值轴":
        for i, x_pix in enumerate(x_pixels):
            if i % 2 == 1:
                cv2.line(grid_overlay, (int(x_pix), y_min), (int(x_pix), y_max), encrypted_grid_color, 1, cv2.LINE_AA)
                drawn_x_grid_lines += 1

    drawn_y_grid_lines = 0
    if y_axis_type == "数值轴":
        for i, y_pix in enumerate(y_pixels):
            if i % 2 == 1:
                cv2.line(grid_overlay, (x_min, int(y_pix)), (x_max, int(y_pix)), encrypted_grid_color, 1, cv2.LINE_AA)
                drawn_y_grid_lines += 1

    cv2.addWeighted(grid_overlay, encrypted_grid_alpha, canvas, 1 - encrypted_grid_alpha, 0, canvas)
    logger.debug(f"成功绘制加密网格线: X轴{drawn_x_grid_lines}条, Y轴{drawn_y_grid_lines}条")
    
    # 绘制加密刻度文本标签，优化显示效果
    try:
        # 优化文本样式，减小字体大小避免重叠
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3  # 减小字体大小
        font_color = (0, 0, 255)  # 红色文本
        thickness = 1  # 细线
        padding = 3  # 减小内边距
        
        logger.debug(f"准备绘制加密刻度文本: X轴像素点数量={len(x_pixels)}, 加密刻度数量={len(x_ticks_encrypted)}")
        logger.debug(f"Y轴像素点数量={len(y_pixels)}, 加密刻度数量={len(y_ticks_encrypted)}")
        logger.debug(f"X轴类型: {x_axis_type}, Y轴类型: {y_axis_type}")
        
        # 计算X轴的最大小数位数
        x_max_decimal = calculate_max_decimal_places(x_ticks_encrypted)
        logger.debug(f"X轴最大小数位数: {x_max_decimal}")
        
        # 计算Y轴的最大小数位数
        y_max_decimal = calculate_max_decimal_places(y_ticks_encrypted)
        logger.debug(f"Y轴最大小数位数: {y_max_decimal}")
        
        # 为X轴绘制加密刻度文本（只对数字轴加密部分）
        drawn_x_texts = 0
        if x_axis_type == "数值轴":
            x_min, x_max_val = min(x_axis[0], x_axis[2]), max(x_axis[0], x_axis[2])
            x_axis_y = max(y_axis[1], y_axis[3])  # X轴的Y坐标
            
            # 确保x_pixels和x_ticks_encrypted长度匹配
            if len(x_pixels) == len(x_ticks_encrypted):
                # 加密刻度是在原始刻度之间插入的，所以偶数索引是原始刻度，奇数索引是加密生成的
                for i in range(len(x_pixels)):
                    # 只处理加密生成的刻度（奇数索引）
                    if i % 2 == 1:
                        x_pix = x_pixels[i]
                        # 检查索引是否有效
                        if i < len(x_ticks_encrypted):
                            tick_value = x_ticks_encrypted[i]
                            
                            # 确保值有效并格式化
                            if tick_value is not None:
                                # 格式化刻度值，保留X轴的最大小数位数
                                text = format_tick_value(tick_value, x_max_decimal)
                                
                                # 获取文本大小
                                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                                
                                # 计算文本位置 - 放在X轴上方，确保足够空间
                                text_x = x_pix - text_size[0] // 2
                                # 确保文本在X轴上方有足够空间，避免重叠
                                text_y = x_axis_y + 16  # 适当间距
                                
                                # 边界检查，确保不与图表内容重叠
                                chart_content_margin = 50  # 图表内容边缘距离
                                if (0 <= text_x and text_x + text_size[0] <= canvas.shape[1] and \
                                   0 <= text_y - text_size[1] - padding and text_y + padding <= canvas.shape[0] and \
                                   text_y >= chart_content_margin):  # 确保在图表内容下方
                                    # 使用半透明背景，减少对图表的遮挡
                                    overlay = canvas.copy()
                                    cv2.rectangle(overlay, 
                                                (text_x - padding, text_y - text_size[1] - padding),
                                                (text_x + text_size[0] + padding, text_y + padding),
                                                (255, 255, 255), -1)
                                    # 添加透明度
                                    alpha = 0.7  # 透明度因子
                                    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
                                    # 添加细边框
                                    cv2.rectangle(canvas, 
                                                (text_x - padding, text_y - text_size[1] - padding),
                                                (text_x + text_size[0] + padding, text_y + padding),
                                                (0, 0, 0), 1)
                                    # 绘制红色文本
                                    cv2.putText(canvas, text, (text_x, text_y), 
                                                font, font_scale, font_color, thickness, cv2.LINE_AA)
                                    drawn_x_texts += 1
        
        # 为Y轴绘制加密刻度文本（只对数字轴加密部分）
        drawn_y_texts = 0
        if y_axis_type == "数值轴":
            y_min, y_max_val = min(y_axis[1], y_axis[3]), max(y_axis[1], y_axis[3])
            y_axis_x = min(x_axis[0], x_axis[2])  # Y轴的X坐标
            
            # 确保y_pixels和y_ticks_encrypted长度匹配
            if len(y_pixels) == len(y_ticks_encrypted):
                # 加密刻度是在原始刻度之间插入的，所以偶数索引是原始刻度，奇数索引是加密生成的
                for i in range(len(y_pixels)):
                    # 只处理加密生成的刻度（奇数索引）
                    if i % 2 == 1:
                        y_pix = y_pixels[i]
                        # 检查索引是否有效
                        if i < len(y_ticks_encrypted):
                            tick_value = y_ticks_encrypted[i]
                            
                            # 确保值有效并格式化
                            if tick_value is not None:
                                # 格式化刻度值，保留Y轴的最大小数位数
                                text = format_tick_value(tick_value, y_max_decimal)
                                
                                # 获取文本大小
                                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                                
                                # 计算文本位置 - 放在Y轴左侧，避免与图表重叠
                                text_x = y_axis_x - text_size[0] - 8  # 放在Y轴左侧
                                text_y = y_pix + text_size[1] // 2
                                
                                # 边界检查，确保不与图表内容重叠
                                chart_content_margin = 50  # 图表内容边缘距离
                                if (0 <= text_y and text_y - text_size[1] - padding >= 0 and \
                                   text_x - padding >= 0 and text_x + text_size[0] + padding <= canvas.shape[1] and \
                                   text_x <= canvas.shape[1] - chart_content_margin):  # 确保在图表内容左侧
                                    # 使用半透明背景，减少对图表的遮挡
                                    overlay = canvas.copy()
                                    cv2.rectangle(overlay, 
                                                (text_x - padding, text_y - text_size[1] - padding),
                                                (text_x + text_size[0] + padding, text_y + padding),
                                                (255, 255, 255), -1)
                                    # 添加透明度
                                    alpha = 0.7  # 透明度因子
                                    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
                                    # 添加细边框
                                    cv2.rectangle(canvas, 
                                                (text_x - padding, text_y - text_size[1] - padding),
                                                (text_x + text_size[0] + padding, text_y + padding),
                                                (0, 0, 0), 1)
                                    # 绘制红色文本
                                    cv2.putText(canvas, text, (text_x, text_y), 
                                                font, font_scale, font_color, thickness, cv2.LINE_AA)
                                    drawn_y_texts += 1
        
        logger.debug(f"成功绘制加密刻度文本: X轴{drawn_x_texts}个, Y轴{drawn_y_texts}个")
        
        # 不再添加水印，避免干扰图表
                    
    except Exception as e:
        logger.error(f"绘制加密刻度文本时出错: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
    
    return canvas

def count_decimal_places(value):
    """
    计算浮点数的小数位数
    改用字符串方式计算，处理浮点数误差
    """
    try:
        # 转换为浮点数
        if isinstance(value, str):
            value = float(value)
        
        # 先消除浮点数误差
        s = f"{value:.12f}"
        
        # 处理科学计数法
        if 'e' in s.lower():
            # 转换为普通小数形式
            parts = s.lower().split('e')
            num = float(parts[0])
            exp = int(parts[1])
            s = f"{num * (10 ** exp):.12f}"
        
        # 去掉末尾的0
        if '.' in s:
            # 分割整数和小数部分
            int_part, dec_part = s.split('.')
            # 去掉小数部分末尾的0
            dec_part = dec_part.rstrip('0')
            # 如果小数部分为空，返回0
            if not dec_part:
                return 0
            # 返回小数部分长度
            return len(dec_part)
        return 0
    except:
        return 0

def process_chart(image_path, output_dir, chart_type_override=None, chart_id_override=None, axis_repair_hint=None):
    """
    处理单个图表，生成两种网格图像和刻度信息
    1. _grid: 基础网格 - 短横线延伸形成网格图
    2. _with_grid: 加密网格 - 在基础网格上添加加密刻度和文本
    """
    logger.info(f"处理图像: {image_path}")
    logger.debug(f"输出目录: {output_dir}")
    chart_id = chart_id_override or os.path.splitext(os.path.basename(image_path))[0]
    chart_type = (chart_type_override or os.path.basename(os.path.dirname(image_path))).lower()
    axis_repair_hint = normalize_axis_repair_hint(axis_repair_hint)
    repair_applied = {
        "x_axis": False,
        "y_axis": False,
        "x_ticks": False,
        "y_ticks": False,
        "hint": axis_repair_hint,
    }
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    try:
        # 修复Windows路径问题 - 使用绝对路径和规范化
        image_path = os.path.abspath(os.path.normpath(image_path))
        logger.debug(f"规范化后的绝对图像路径: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"图像文件不存在: {image_path}")
            return None
        
        # 检查文件大小
        file_size = os.path.getsize(image_path)
        logger.debug(f"图像文件大小: {file_size} 字节")
        
        # 在Windows上，尝试不同的编码方式
        # 使用双反斜杠确保路径正确
        alt_path = image_path.replace('/', '\\')
        logger.debug(f"Windows格式路径: {alt_path}")
        
        # 尝试使用numpy和cv2.imdecode处理中文路径问题
        try:
            import numpy as np
            # 读取文件内容
            img_data = np.fromfile(image_path, dtype=np.uint8)
            # 解码图像
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if img is not None:
                logger.info(f"成功加载图像，形状: {img.shape}")
            else:
                logger.error(f"cv2.imdecode失败，但文件存在且可读")
                logger.debug(f"文件头: {img_data[:12].hex()}")
                return None
        except Exception as e:
            logger.error(f"使用numpy和cv2.imdecode加载图像时出错: {str(e)}")
            return None
        
        h, w = img.shape[:2]
        logger.debug(f"图像尺寸: {w}x{h}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. 检测直线，调整参数以提高检测率
        logger.debug("开始检测直线...")
        raw_lines = detect_candidate_lines(
            gray, 
            canny_threshold1=30,  # 降低阈值以检测更多边缘
            canny_threshold2=100,
            hough_threshold=15,    # 降低阈值
            min_length=15,         # 缩短最小长度
            max_gap=15             # 增加最大间隙
        )
        logger.debug(f"检测到 {len(raw_lines)} 条原始直线")
        
        if not raw_lines:
            logger.warning(f"未检测到直线: {image_path}")
            return None
        
        # 2. 合并相似直线
        merged_lines = merge_similar_lines(raw_lines)
        logger.debug(f"合并后得到 {len(merged_lines)} 条直线")
        
        # 3. 推断坐标轴
        logger.debug("开始推断坐标轴...")
        try:
            x_axis, y_axis, _ = infer_axes_from_lines(merged_lines, (w, h), gray)
        except Exception as e:
            logger.error(f"推断坐标轴时出错: {e}")
            # 尝试直接检测坐标轴
            x_axis, y_axis = None, None
            
            # 检测底部的水平线作为X轴
            for line in merged_lines:
                x1, y1, x2, y2 = line
                # 水平线且接近底部
                if abs(y1 - y2) < 10 and max(y1, y2) > h * 0.7:
                    if x_axis is None or (max(x2, x1) - min(x1, x2)) > (x_axis[2] - x_axis[0]):
                        x_axis = line
            
            # 检测左侧的垂直线作为Y轴
            for line in merged_lines:
                x1, y1, x2, y2 = line
                # 垂直线且接近左侧
                if abs(x1 - x2) < 10 and min(x1, x2) < w * 0.3:
                    if y_axis is None or (max(y2, y1) - min(y1, y2)) > (y_axis[3] - y_axis[1]):
                        y_axis = line
        
        before_repair_x_axis, before_repair_y_axis = x_axis, y_axis
        if axis_repair_enabled(axis_repair_hint):
            x_axis, y_axis, repair_boxes = repair_missing_axes(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                axis_repair_hint,
            )
            repair_applied["x_axis"] = axis_repair_hint.get("x_axis_missing") and x_axis is not None
            repair_applied["y_axis"] = axis_repair_hint.get("y_axis_missing") and y_axis is not None
        else:
            repair_boxes = []

        if x_axis is None or y_axis is None:
            logger.warning(f"未检测到 X/Y 轴: {image_path}")
            return None
        
        logger.debug(f"检测到坐标轴: X轴={x_axis}, Y轴={y_axis}")
        
        # 4. 检测刻度线
        logger.debug("开始检测刻度线...")
        x_axis, y_axis, point_axes_refined = refine_point_chart_axes_from_gridlines(
            merged_lines, x_axis, y_axis, chart_type, img.shape
        )
        if point_axes_refined:
            repair_applied["axis_refined_from_gridlines"] = True
            logger.debug(
                "Point-chart axes refined from gridlines: X轴=%s, Y轴=%s",
                x_axis,
                y_axis,
            )

        x_raw_ticks = scan_pixels_for_ticks(img, x_axis, direction='x', scan_range=20)
        y_raw_ticks = scan_pixels_for_ticks(img, y_axis, direction='y', scan_range=20)
        if chart_type in {"scatter", "bubble"}:
            if len(x_raw_ticks) < 2:
                x_grid_pixels = infer_tick_pixels_from_gridlines(
                    merged_lines, x_axis, y_axis, "x", img.shape
                )
                if len(x_grid_pixels) >= 2:
                    x_raw_ticks = ticks_from_pixels(x_grid_pixels, x_axis, "x")
                    logger.debug(
                        "Inferred %s X tick positions from plot gridlines for %s chart.",
                        len(x_grid_pixels),
                        chart_type,
                    )
            if len(y_raw_ticks) < 2:
                y_grid_pixels = infer_tick_pixels_from_gridlines(
                    merged_lines, x_axis, y_axis, "y", img.shape
                )
                if len(y_grid_pixels) >= 2:
                    y_raw_ticks = ticks_from_pixels(y_grid_pixels, y_axis, "y")
                    logger.debug(
                        "Inferred %s Y tick positions from plot gridlines for %s chart.",
                        len(y_grid_pixels),
                        chart_type,
                    )
        if chart_type in {"h_bar", "v_bar"} and not repair_boxes:
            repair_boxes = _bar_boxes(img, chart_type)
        geometry_hint = apply_bar_geometry_repair_hint(
            chart_type,
            axis_repair_hint,
            repair_boxes,
            x_tick_count=_tick_group_count(x_raw_ticks, "x"),
            y_tick_count=_tick_group_count(y_raw_ticks, "y"),
        )
        if axis_repair_enabled(geometry_hint) and geometry_hint != axis_repair_hint:
            axis_repair_hint = geometry_hint
            repair_applied["hint"] = axis_repair_hint
            x_axis, y_axis, repair_boxes = repair_missing_axes(
                img,
                merged_lines,
                x_axis,
                y_axis,
                chart_type,
                axis_repair_hint,
            )
            repair_applied["x_axis"] = axis_repair_hint.get("x_axis_missing") and x_axis is not None
            repair_applied["y_axis"] = axis_repair_hint.get("y_axis_missing") and y_axis is not None
            logger.debug(
                "Bar geometry repair enabled: x_axis_missing=%s y_axis_missing=%s x_ticks_missing=%s y_ticks_missing=%s",
                axis_repair_hint.get("x_axis_missing"),
                axis_repair_hint.get("y_axis_missing"),
                axis_repair_hint.get("x_ticks_missing"),
                axis_repair_hint.get("y_ticks_missing"),
            )
        logger.debug(f"检测到 X轴刻度 {len(x_raw_ticks)} 个, Y轴刻度 {len(y_raw_ticks)} 个")
        
        if axis_repair_enabled(axis_repair_hint):
            if axis_repair_hint.get("x_ticks_missing"):
                x_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type, "x", x_axis, repair_boxes, [], axis_repair_hint
                )
                if len(x_pixels) >= _required_tick_count(chart_type, "x"):
                    x_raw_ticks = ticks_from_pixels(x_pixels, x_axis, "x")
                    repair_applied["x_ticks"] = bool(x_raw_ticks)
            if axis_repair_hint.get("y_ticks_missing"):
                y_pixels = synthesize_tick_pixels_for_missing_axis(
                    chart_type, "y", y_axis, repair_boxes, [], axis_repair_hint
                )
                if len(y_pixels) >= _required_tick_count(chart_type, "y"):
                    y_raw_ticks = ticks_from_pixels(y_pixels, y_axis, "y")
                    repair_applied["y_ticks"] = bool(y_raw_ticks)

        if (
            (len(x_raw_ticks) < _required_tick_count(chart_type, "x") and not axis_repair_hint.get("x_ticks_missing"))
            or (len(y_raw_ticks) < _required_tick_count(chart_type, "y") and not axis_repair_hint.get("y_ticks_missing"))
        ):
            logger.warning(f"未检测到足够的刻度线: {image_path}")
            return None
        
        # 5. 合并和过滤刻度线
        x_merged_ticks = merge_similar_lines(x_raw_ticks, angle_threshold=np.deg2rad(10))
        y_merged_ticks = merge_similar_lines(y_raw_ticks, angle_threshold=np.deg2rad(10))
        
        x_filtered_ticks = filter_ticks(x_merged_ticks, direction='x')
        y_filtered_ticks = filter_ticks(y_merged_ticks, direction='y')

        if (
            _is_bar_category_axis(chart_type, "x")
            and _tick_group_count(x_filtered_ticks, "x") < _tick_group_count(x_merged_ticks, "x")
        ):
            x_filtered_ticks = x_merged_ticks
        if (
            _is_bar_category_axis(chart_type, "y")
            and _tick_group_count(y_filtered_ticks, "y") < _tick_group_count(y_merged_ticks, "y")
        ):
            y_filtered_ticks = y_merged_ticks
        
        logger.debug(f"过滤后: X轴刻度 {len(x_filtered_ticks)} 个, Y轴刻度 {len(y_filtered_ticks)} 个")
        
        if (
            (len(x_filtered_ticks) < _required_tick_count(chart_type, "x") and not axis_repair_hint.get("x_ticks_missing"))
            or (len(y_filtered_ticks) < _required_tick_count(chart_type, "y") and not axis_repair_hint.get("y_ticks_missing"))
        ):
            logger.warning(f"未检测到有效的刻度线: {image_path}")
            return None
    
    except Exception as e:
        logger.error(f"前序处理出错: {e}")
        return None
    
    # 6. 使用model_processor.py中的函数提取刻度标签和颜色
    logger.debug("开始使用模型提取刻度标签和颜色...")
    
    # 处理刻度标签
    ticks_result = extract_tick_labels_with_llm(
        image_path,
        cache_dir=TICK_LABEL_CACHE_DIR,
        chart_type_override=chart_type,
    )
    if ticks_result.get("api_failed"):
        logger.warning(
            "LLM tick labels unavailable after retries; skip chart instead of using positional fallback: %s",
            ticks_result.get("failure_reason", "unknown"),
        )
        return None

    x_ticks_values = ticks_result.get("x_ticks", [])
    y_ticks_values = ticks_result.get("y_ticks", [])
    x_axis_type = ticks_result.get("x_axis_type", "数值轴")
    y_axis_type = ticks_result.get("y_axis_type", "数值轴")
    x_ticks_values, x_axis_type = coerce_chart_axis_numeric_ticks(
        chart_type, "x", x_ticks_values, x_axis_type
    )
    y_ticks_values, y_axis_type = coerce_chart_axis_numeric_ticks(
        chart_type, "y", y_ticks_values, y_axis_type
    )
    
    # 处理图例颜色
    if chart_type in {"scatter", "bubble"}:
        colors_result = extract_point_chart_items(image_path)
    else:
        colors_result = extract_chart_series_color(image_path)
    if isinstance(colors_result, list):
        colors_data = colors_result
    else:
        colors_data = [{"name": "Series 1", "color": str(colors_result)}]
    
    logger.debug(f"模型处理结果: X轴{len(x_ticks_values)}个刻度, Y轴{len(y_ticks_values)}个刻度, {len(colors_data)}个颜色")
    
    # 初始化刻度数据列表
    x_ticks_data = []  # 数值轴存储为float，文字轴存储为字符串
    x_pixels_data = []
    y_ticks_data = []  # 数值轴存储为float，文字轴存储为字符串
    y_pixels_data = []
    
    # 计算检测到的刻度线的中心位置
    x_pixel_positions = sorted([(t[0] + t[2]) // 2 for t in x_filtered_ticks])
    y_pixel_positions = sorted([(t[1] + t[3]) // 2 for t in y_filtered_ticks], reverse=True)
    x_axis_scale = "linear"
    y_axis_scale = "linear"

    if (
        chart_type in {"scatter", "bubble"}
        and axis_repair_enabled(axis_repair_hint)
        and (
            axis_repair_hint.get("x_axis_missing")
            or axis_repair_hint.get("y_axis_missing")
            or axis_repair_hint.get("x_ticks_missing")
            or axis_repair_hint.get("y_ticks_missing")
        )
    ):
        projected_x_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "x")
        projected_y_pixels = infer_point_chart_grid_pixels_for_missing_axes(img, "y")
        selected_x_pixels, selected_x_scale = select_projected_tick_pixels_for_values(
            projected_x_pixels,
            x_ticks_values,
            "x",
        )
        selected_y_pixels, selected_y_scale = select_projected_tick_pixels_for_values(
            projected_y_pixels,
            y_ticks_values,
            "y",
        )
        if len(selected_x_pixels) == len(x_ticks_values or []) and len(selected_x_pixels) >= 2:
            x_pixel_positions = selected_x_pixels
            x_axis_scale = selected_x_scale
            repair_applied["x_ticks"] = True
            repair_applied["x_ticks_refined_from_missing_axis_grid"] = True
            logger.debug(
                "Point-chart missing-axis X ticks selected from projection: %s scale=%s",
                x_pixel_positions,
                x_axis_scale,
            )
        if len(selected_y_pixels) == len(y_ticks_values or []) and len(selected_y_pixels) >= 2:
            y_pixel_positions = selected_y_pixels
            y_axis_scale = selected_y_scale
            repair_applied["y_ticks"] = True
            repair_applied["y_ticks_refined_from_missing_axis_grid"] = True
            logger.debug(
                "Point-chart missing-axis Y ticks selected from projection: %s scale=%s",
                y_pixel_positions,
                y_axis_scale,
            )
        if len(x_pixel_positions) >= 2 and len(y_pixel_positions) >= 2:
            plot_bounds = infer_point_chart_plot_bounds_from_horizontal_grid(
                img,
                y_pixel_positions,
                x_pixel_positions,
            )
            if plot_bounds:
                plot_left, plot_right = plot_bounds
                repair_applied["plot_bounds_refined_from_horizontal_grid"] = True
            else:
                plot_left = int(min(x_pixel_positions))
                plot_right = int(max(x_pixel_positions))
            vertical_bounds = infer_point_chart_plot_vertical_bounds_from_grid(
                img,
                y_pixel_positions,
                x_pixel_positions,
            )
            if vertical_bounds:
                plot_top, plot_bottom = vertical_bounds
                repair_applied["plot_vertical_bounds_refined_from_grid"] = True
            else:
                plot_bottom = int(max(y_pixel_positions))
                plot_top = int(min(y_pixel_positions))
            if plot_right > plot_left and plot_bottom > plot_top:
                x_axis = [plot_left, plot_bottom, plot_right, plot_bottom]
                y_axis = [plot_left, plot_bottom, plot_left, plot_top]
                repair_applied["x_axis"] = True
                repair_applied["y_axis"] = True
                repair_applied["axis_refined_from_missing_point_grid"] = True

    if chart_type in {"h_bar", "v_bar"} and axis_repair_enabled(axis_repair_hint):
        if x_axis_type == NUMERIC_AXIS_TYPE:
            x_pixel_positions = add_missing_numeric_axis_endpoints(
                "x", x_axis, x_pixel_positions, x_ticks_values
            )
        if y_axis_type == NUMERIC_AXIS_TYPE:
            y_pixel_positions = add_missing_numeric_axis_endpoints(
                "y", y_axis, y_pixel_positions, y_ticks_values
            )

    if axis_repair_enabled(axis_repair_hint) and not repair_applied.get("axis_refined_from_missing_point_grid"):
        if axis_repair_hint.get("x_ticks_missing"):
            repaired_x_pixels = synthesize_tick_pixels_for_missing_axis(
                chart_type, "x", x_axis, repair_boxes, x_ticks_values, axis_repair_hint
            )
            if len(repaired_x_pixels) >= _required_tick_count(chart_type, "x"):
                x_pixel_positions = repaired_x_pixels
                repair_applied["x_ticks"] = True
        if axis_repair_hint.get("y_ticks_missing"):
            repaired_y_pixels = synthesize_tick_pixels_for_missing_axis(
                chart_type, "y", y_axis, repair_boxes, y_ticks_values, axis_repair_hint
            )
            if len(repaired_y_pixels) >= _required_tick_count(chart_type, "y"):
                y_pixel_positions = repaired_y_pixels
                repair_applied["y_ticks"] = True

    if (
        chart_type in {"scatter", "bubble"}
        and repair_applied.get("axis_refined_from_gridlines")
        and not repair_applied.get("axis_refined_from_missing_point_grid")
    ):
        if x_axis_type == NUMERIC_AXIS_TYPE and len(x_ticks_values or []) >= 2:
            x_start, x_end = sorted([int(x_axis[0]), int(x_axis[2])])
            x_pixel_positions = [int(round(value)) for value in np.linspace(x_start, x_end, len(x_ticks_values))]
            repair_applied["x_ticks_refined_from_axis"] = True
        if y_axis_type == NUMERIC_AXIS_TYPE and len(y_ticks_values or []) >= 2:
            y_low, y_high = sorted([int(y_axis[1]), int(y_axis[3])])
            y_pixel_positions = [int(round(value)) for value in np.linspace(y_high, y_low, len(y_ticks_values))]
            repair_applied["y_ticks_refined_from_axis"] = True

    if chart_type in {"scatter", "bubble"}:
        if (
            x_axis_type == NUMERIC_AXIS_TYPE
            and point_chart_tick_pixels_are_suspicious(
                x_pixel_positions, x_axis, "x", len(x_ticks_values or [])
            )
        ):
            projected_x_pixels = infer_point_chart_grid_pixels_by_projection(
                img, x_axis, y_axis, "x", expected_count=len(x_ticks_values or [])
            )
            if len(projected_x_pixels) >= max(2, min(len(x_ticks_values or []), 3)):
                x_pixel_positions = projected_x_pixels
                repair_applied["x_ticks_refined_from_projection_grid"] = True
                logger.debug(
                    "Point-chart X ticks refined from projection grid: %s",
                    x_pixel_positions,
                )
        if (
            y_axis_type == NUMERIC_AXIS_TYPE
            and point_chart_tick_pixels_are_suspicious(
                y_pixel_positions, y_axis, "y", len(y_ticks_values or [])
            )
        ):
            projected_y_pixels = infer_point_chart_grid_pixels_by_projection(
                img, x_axis, y_axis, "y", expected_count=len(y_ticks_values or [])
            )
            if len(projected_y_pixels) >= max(2, min(len(y_ticks_values or []), 3)):
                y_pixel_positions = projected_y_pixels
                repair_applied["y_ticks_refined_from_projection_grid"] = True
                logger.debug(
                    "Point-chart Y ticks refined from projection grid: %s",
                    y_pixel_positions,
                )

    if (
        axis_repair_hint.get("x_ticks_missing")
        and x_axis_type != NUMERIC_AXIS_TYPE
        and len(x_ticks_values) < len(x_pixel_positions)
    ):
        missing_count = len(x_pixel_positions) - len(x_ticks_values)
        x_ticks_values = list(x_ticks_values) + [f"category_{i + 1}" for i in range(missing_count)]

    if (
        axis_repair_hint.get("y_ticks_missing")
        and y_axis_type != NUMERIC_AXIS_TYPE
        and len(y_ticks_values) < len(y_pixel_positions)
    ):
        missing_count = len(y_pixel_positions) - len(y_ticks_values)
        y_ticks_values = [f"category_{i + 1}" for i in range(missing_count)] + list(y_ticks_values)

    # Local fallback for offline/dev runs: when the LLM cannot return usable
    # tick labels, keep the image-processing pipeline alive by assigning
    # positional numeric ticks to every detected tick mark.
    if len(x_ticks_values) < 2 and len(x_pixel_positions) >= 2:
        logger.warning("LLM X tick labels unavailable; using positional fallback ticks.")
        x_ticks_values = list(range(len(x_pixel_positions)))
        x_axis_type = "数值轴"

    if len(y_ticks_values) < 2 and len(y_pixel_positions) >= 2:
        logger.warning("LLM Y tick labels unavailable; using positional fallback ticks.")
        y_ticks_values = list(range(len(y_pixel_positions)))
        y_axis_type = "数值轴"
    
    # 匹配X轴刻度
    if x_ticks_values and x_pixel_positions:
        # 确保数值数量与刻度数量匹配
        if len(x_ticks_values) > len(x_pixel_positions):
            x_ticks_values = x_ticks_values[:len(x_pixel_positions)]
        elif len(x_ticks_values) < len(x_pixel_positions):
            # 如果数值不足，使用已有数值插值
            if len(x_ticks_values) >= 2:
                import numpy as np
                # 尝试转换为数值进行插值
                try:
                    numeric_values = [float(v) for v in x_ticks_values]
                    x_positions = np.linspace(0, 1, len(x_pixel_positions))
                    orig_positions = np.linspace(0, 1, len(numeric_values))
                    interpolated = np.interp(x_positions, orig_positions, numeric_values)
                    x_ticks_values = interpolated.tolist()
                except:
                    # 如果是文字轴，无法插值，保持原样
                    pass
        
        # 匹配刻度值和像素位置
        for i, (tick_value, pixel_pos) in enumerate(zip(x_ticks_values, x_pixel_positions)):
            if x_axis_type == "数值轴":
                try:
                    value = float(tick_value)
                    x_ticks_data.append(value)
                except ValueError:
                    # 如果是数值轴但值不是数字，转换为字符串
                    x_ticks_data.append(str(tick_value))
            else:
                # 文字轴，直接存储为字符串
                x_ticks_data.append(str(tick_value))
            x_pixels_data.append(pixel_pos)
    
    # 匹配Y轴刻度
    if y_ticks_values and y_pixel_positions:
        # 确保数值数量与刻度数量匹配
        if len(y_ticks_values) > len(y_pixel_positions):
            y_ticks_values = y_ticks_values[:len(y_pixel_positions)]
        elif len(y_ticks_values) < len(y_pixel_positions):
            # 如果数值不足，使用已有数值插值
            if len(y_ticks_values) >= 2:
                import numpy as np
                # 尝试转换为数值进行插值
                try:
                    numeric_values = [float(v) for v in y_ticks_values]
                    y_positions = np.linspace(0, 1, len(y_pixel_positions))
                    orig_positions = np.linspace(0, 1, len(numeric_values))
                    interpolated = np.interp(y_positions, orig_positions, numeric_values)
                    y_ticks_values = interpolated.tolist()
                except:
                    # 如果是文字轴，无法插值，保持原样
                    pass
        
        # 匹配刻度值和像素位置
        for i, (tick_value, pixel_pos) in enumerate(zip(y_ticks_values, y_pixel_positions)):
            if y_axis_type == "数值轴":
                try:
                    value = float(tick_value)
                    y_ticks_data.append(value)
                except ValueError:
                    # 如果是数值轴但值不是数字，转换为字符串
                    y_ticks_data.append(str(tick_value))
            else:
                # 文字轴，直接存储为字符串
                y_ticks_data.append(str(tick_value))
            y_pixels_data.append(pixel_pos)
    
    if (
        len(x_ticks_data) < _required_tick_count(chart_type, "x")
        or len(y_ticks_data) < _required_tick_count(chart_type, "y")
    ):
        logger.warning(f"有效刻度数量不足: {image_path}")
        return None
    
    logger.debug(f"最终有效刻度: X轴={len(x_ticks_data)}个, Y轴={len(y_ticks_data)}个")
    
    # 生成加密刻度和对应的加密像素位置
    logger.debug("生成加密刻度和对应的加密像素位置...")
    
    # 判断是否为数值轴
    is_x_numeric = x_axis_type == "数值轴"
    is_y_numeric = y_axis_type == "数值轴"
    if is_x_numeric:
        x_axis_scale = axis_scale_from_ticks_and_pixels(x_ticks_data, x_pixels_data)
    else:
        x_axis_scale = "linear"
    if is_y_numeric:
        y_axis_scale = axis_scale_from_ticks_and_pixels(y_ticks_data, y_pixels_data)
    else:
        y_axis_scale = "linear"
    
    # 生成加密刻度（只对数字轴加密）
    x_ticks_encrypted = generate_encrypted_ticks(
        x_ticks_data,
        is_numeric_axis=is_x_numeric,
        axis_scale=x_axis_scale,
    )
    y_ticks_encrypted = generate_encrypted_ticks(
        y_ticks_data,
        is_numeric_axis=is_y_numeric,
        axis_scale=y_axis_scale,
    )
    
    # 生成对应的加密像素位置
    # 对于数字轴，需要插入中间像素位置；对于文字轴，不需要
    if is_x_numeric:
        x_pixels_encrypted = []
        for i in range(len(x_ticks_data)):
            # 添加原始像素位置
            x_pixels_encrypted.append(x_pixels_data[i])
            # 如果不是最后一个点，计算并添加中间像素位置
            if i < len(x_ticks_data) - 1:
                mid_pixel = (x_pixels_data[i] + x_pixels_data[i + 1]) // 2
                x_pixels_encrypted.append(mid_pixel)
    else:
        # 文字轴，不插入中间像素位置
        x_pixels_encrypted = x_pixels_data.copy()
    
    if is_y_numeric:
        y_pixels_encrypted = []
        for i in range(len(y_ticks_data)):
            # 添加原始像素位置
            y_pixels_encrypted.append(y_pixels_data[i])
            # 如果不是最后一个点，计算并添加中间像素位置
            if i < len(y_ticks_data) - 1:
                mid_pixel = (y_pixels_data[i] + y_pixels_data[i + 1]) // 2
                y_pixels_encrypted.append(mid_pixel)
    else:
        # 文字轴，不插入中间像素位置
        y_pixels_encrypted = y_pixels_data.copy()
    
    logger.debug(f"生成加密刻度: X轴={len(x_ticks_encrypted)}个, Y轴={len(y_ticks_encrypted)}个")
    logger.debug(f"生成加密像素位置: X轴={len(x_pixels_encrypted)}个, Y轴={len(y_pixels_encrypted)}个")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成基础网格图像 (_grid)
    basic_grid_path = os.path.join(output_dir, f"{chart_id}_grid.png")
    try:
        basic_grid_img = draw_basic_grid(img, x_pixels_data, y_pixels_data, x_axis, y_axis)
        # 保存基础网格图像
        try:
            success, encoded_img = cv2.imencode('.png', basic_grid_img)
            if success:
                encoded_img.tofile(basic_grid_path)
                logger.debug(f"基础网格图像已保存到: {basic_grid_path}")
            else:
                logger.error(f"无法编码基础网格图像: {basic_grid_path}")
        except Exception as e:
            logger.error(f"保存基础网格图像时出错: {str(e)}")
    except Exception as e:
        logger.error(f"绘制基础网格时出错: {e}")
    
    # 生成加密网格图像 (_with_grid)
    encrypted_grid_path = os.path.join(output_dir, f"{chart_id}_with_grid.png")
    try:
        # 使用加密像素位置和加密刻度绘制加密网格
        # 传递轴类型参数，只对数字轴加密部分添加文本
        encrypted_grid_img = draw_encrypted_grid(
            img, 
            x_pixels_encrypted, 
            y_pixels_encrypted, 
            x_ticks_encrypted, 
            y_ticks_encrypted, 
            x_axis, 
            y_axis,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
            base_x_pixels=x_pixels_data,
            base_y_pixels=y_pixels_data,
        )
        # 保存加密网格图像
        try:
            success, encoded_img = cv2.imencode('.png', encrypted_grid_img)
            if success:
                encoded_img.tofile(encrypted_grid_path)
                logger.debug(f"加密网格图像已保存到: {encrypted_grid_path}")
                # 验证文件是否成功保存
                if os.path.exists(encrypted_grid_path):
                    logger.debug(f"加密网格文件大小: {os.path.getsize(encrypted_grid_path)} 字节")
                else:
                    logger.warning(f"加密网格图像文件未找到: {encrypted_grid_path}")
            else:
                logger.error(f"无法编码加密网格图像: {encrypted_grid_path}")
        except Exception as e:
            logger.error(f"保存加密网格图像时出错: {str(e)}")
    except Exception as e:
        logger.error(f"绘制加密网格时出错: {e}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
    
    # 保存刻度信息（包含_with_grid相关数据）
    tick_data = {
        "chart_id": chart_id,
        "x_ticks": x_ticks_data,
        "y_ticks": y_ticks_data,
        "x_pixels": x_pixels_data,
        "y_pixels": y_pixels_data,
        "x_ticks_encrypted": x_ticks_encrypted,
        "y_ticks_encrypted": y_ticks_encrypted,
        "x_pixels_encrypted": x_pixels_encrypted,
        "y_pixels_encrypted": y_pixels_encrypted,
        "x_axis_type": x_axis_type,
        "y_axis_type": y_axis_type,
        "x_axis_scale": x_axis_scale,
        "y_axis_scale": y_axis_scale,
        "image_path": image_path,
        "basic_grid_path": basic_grid_path,
        "encrypted_grid_path": encrypted_grid_path,
        "colors": colors_data,
        "axis_repair": repair_applied,
    }
    
    output_json_path = os.path.join(output_dir, f"{chart_id}_ticks.json")
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(tick_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"刻度信息已保存到: {output_json_path}")
    except Exception as e:
        logger.error(f"保存JSON时出错: {e}")
    
    logger.info(f"处理完成: {chart_id}")
    return tick_data

def generate_encrypted_ticks(original_ticks, is_numeric_axis=True, axis_scale="linear"):
    """
    根据原始刻度生成加密刻度（在原刻度之间插入中间值）
    对于数字轴，插入中间值；对于文字轴，不插入中间值
    """
    encrypted_ticks = []
    if not is_numeric_axis:
        # 文字轴，不加密，直接返回原值
        return original_ticks.copy()
    
    # 数字轴，插入中间值
    for i in range(len(original_ticks)):
        encrypted_ticks.append(original_ticks[i])
        if i < len(original_ticks) - 1:
            # 检查是否为数值类型
            try:
                val1 = float(original_ticks[i])
                val2 = float(original_ticks[i + 1])
                if axis_scale == "log" and val1 > 0 and val2 > 0:
                    mid_value = math.sqrt(val1 * val2)
                    if max(abs(val1), abs(val2)) >= 100:
                        mid_value = round(mid_value)
                    elif max(abs(val1), abs(val2)) >= 1:
                        mid_value = round(mid_value, 4)
                else:
                    mid_value = (val1 + val2) / 2
                # 消除中间值的浮点数误差
                mid_value = round(float(f"{mid_value:.12f}"), 10)
                encrypted_ticks.append(mid_value)
            except (ValueError, TypeError):
                # 如果不是数值，不插入中间值
                pass
    return encrypted_ticks


def batch_process_charts(input_dirs, output_base_dir):
    """
    批量处理指定目录下的所有图表
    """
    for input_dir in input_dirs:
        # 确保路径使用正确的编码
        input_dir = os.path.abspath(input_dir)
        if not os.path.exists(input_dir):
            logger.error(f"目录不存在: {input_dir}")
            continue
        
        # 获取图表类型（bubble或scatter）
        chart_type = os.path.basename(input_dir)
        output_dir = os.path.join(output_base_dir, f"{chart_type}_with_grid")
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"处理目录: {input_dir}, 输出目录: {output_dir}")
        
        # 获取所有PNG图像
        image_paths = glob(os.path.join(input_dir, "*.png"))
        # 过滤掉可能包含'grid'的文件名，避免重复处理
        image_paths = [p for p in image_paths if 'grid' not in os.path.basename(p).lower()]
        logger.info(f"找到 {len(image_paths)} 张 {chart_type} 图表")
        
        # 逐个处理图像
        success_count = 0
        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:  # 每处理10个文件输出一次进度
                logger.info(f"进度: {i+1}/{len(image_paths)}")
            
            try:
                result = process_chart(image_path, output_dir)
                if result:
                    success_count += 1
            except Exception as e:
                logger.error(f"处理文件时出错: {image_path}, 错误: {e}")
        
        logger.info(f"{chart_type} 处理完成: {success_count}/{len(image_paths)} 成功")
    
    return success_count


def main():
    logger.info("开始执行网格生成程序...")
    
    # 定义输入和输出路径，使用原始字符串避免转义问题
    charts_base_dir = r"D:\home work\Agent.paper\test demo\backend\Grid_generation\generated_charts\test1110"
    
    output_base_dir = r"D:\home work\Agent.paper\test demo\backend\Grid_generation\generated_charts_with_grid"
    
    # 处理所有文件模式
    test_mode = True  # 设置为True处理少量测试文件
    
    if test_mode:
        logger.info("进入测试模式，只处理少量文件...")
        
        # 选择几个测试文件
        charts_base_dir = r"\Users\98185\Desktop\Grid_generation\generated_charts\test1110"
        test_files = []
        
        # 直接查找test1113目录下的所有PNG文件
        if os.path.exists(charts_base_dir):
            png_files = [f for f in os.listdir(charts_base_dir) if f.endswith('.png') and 'grid' not in f.lower()]
            # 添加所有找到的PNG文件
            test_files = [os.path.join(charts_base_dir, f) for f in png_files]
        
        logger.info(f"找到测试文件: {test_files}")
        
        for test_file in test_files:
            if os.path.exists(test_file):
                logger.info(f"测试文件处理: {test_file}")
                chart_type = os.path.basename(os.path.dirname(test_file))
                test_output_dir = os.path.join(output_base_dir, f"{chart_type}_with_grid")
                os.makedirs(test_output_dir, exist_ok=True)
                result = process_chart(test_file, test_output_dir)
                logger.info(f"测试文件处理结果: {'成功' if result else '失败'}")
        
        logger.info("测试模式处理完成！请检查生成的图像是否包含加密刻度文本")
    else:
        # 检查输入目录是否存在
        if not os.path.exists(charts_base_dir):
            logger.error(f"图表基础目录不存在: {charts_base_dir}")
            return
        
        logger.info(f"图表基础目录: {charts_base_dir}")
        
        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)
        logger.info(f"输出基础目录: {output_base_dir}")
        
        # 获取所有子目录作为图表类型
        chart_types = []
        for item in os.listdir(charts_base_dir):
            item_path = os.path.join(charts_base_dir, item)
            if os.path.isdir(item_path):
                chart_types.append(item)
        
        logger.info(f"找到 {len(chart_types)} 种图表类型: {', '.join(chart_types)}")
        
        # 单独处理每种图表类型，以便分别统计成功数量
        total_success = 0
        
        for chart_type in chart_types:
            chart_dir = os.path.join(charts_base_dir, chart_type)
            logger.info(f"开始处理 {chart_type} 图表...")
            
            output_dir = os.path.join(output_base_dir, f"{chart_type}_with_grid")
            os.makedirs(output_dir, exist_ok=True)
            
            # 先测试一个文件
            try:
                chart_files = [f for f in os.listdir(chart_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and 'grid' not in f.lower()]
                if chart_files:
                    test_file = os.path.join(chart_dir, chart_files[0])
                    logger.info(f"测试处理 {chart_type} 文件: {test_file}")
                    test_result = process_chart(test_file, output_dir)
                    if test_result:
                        logger.info("测试文件处理成功！")
                    else:
                        logger.warning("测试文件处理失败，可能需要调整参数")
            except Exception as e:
                logger.error(f"测试 {chart_type} 处理时出错: {e}")
            
            # 批量处理当前图表类型
            chart_success = batch_process_charts([chart_dir], output_base_dir)
            logger.info(f"{chart_type} 处理完成，成功处理 {chart_success} 个文件")
            total_success += chart_success
        
        logger.info(f"所有图表处理完成！总计成功处理: {total_success} 个文件")


if __name__ == "__main__":
    logger.debug("网格生成脚本启动")
    main()
    logger.debug("网格生成脚本结束")
