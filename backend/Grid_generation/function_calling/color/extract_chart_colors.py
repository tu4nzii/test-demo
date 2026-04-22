import cv2
import numpy as np
from collections import Counter
import json
import sys
import os
import base64
import re
import requests

# API配置
API_URL = "https://api.vveai.com/v1/chat/completions"
API_KEYS = [
    "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    "sk-2nzrUYD0JWLFzopWF477111f78E746AbAcA9Ed8534C3A481",
    "sk-CiD5WVUNIkBeXDgYB46b90C06aD24636BcEaBaFa993970C4",
    "sk-WvF4fU10VeOkfFMq579610Fc01E8496d827d0d3e04C44d0a",
    "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
]
key_index = 0

def rotate_key():
    """切换到下一个 key"""
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"🔑 已切换至新的 API Key [{key_index + 1}/{len(API_KEYS)}]")

def chat_with_gemini(messages: list) -> str:
    """与Gemini进行对话（同步版本）"""
    payload = {
        "model": "gemini-2.5-pro",
        "messages": messages,
        "temperature": 0.7
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_index]}"
    }

    for attempt in range(1, 4):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)

            if response.status_code == 429:
                print(f"🚫 请求频率超限，切换 Key 重试...")
                rotate_key()
                continue

            if response.status_code != 200:
                print(f"⚠️ HTTP {response.status_code}: {response.text[:200]}")
                continue

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                print(f"⚠️ 响应格式错误: {result}")

        except Exception as e:
            print(f"❌ 第 {attempt} 次尝试失败: {e}")
            continue

    print("❌ 所有尝试均失败")
    return "抱歉，我暂时无法回应您的请求。"

def count_legend_items(image_path: str) -> int:
    """判断图表中的图例数量"""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图像文件")

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        messages = [
            {"role": "system", "content": "你是一个图表分析专家，请分析提供的图表图像。"},
            {"role": "user", "content": [
                {"type": "text", "text": "请分析这个图表中的图例数量。图表类型是直角坐标系的（柱状图、折线图、散点图等）。请只返回数字，不要添加任何额外文字。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ]

        response = chat_with_gemini(messages)
        legend_count = int(re.search(r'\d+', response).group())
        return legend_count
    except Exception as e:
        print(f"❌ 无法识别图例数量: {e}")
        return 1

def recognize_legend_items(image_path: str) -> list:
    """识别图表中的所有图例项及其颜色"""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图像文件")

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        messages = [
            {"role": "system", "content": "你是一个图表分析专家，请分析提供的图表图像并识别图例颜色。"},
            {"role": "user", "content": [
                {"type": "text", "text": '''请分析这个图表，识别所有的图例项及其颜色。

请返回一个JSON格式的响应，包含一个数组'legend_items'，每个元素是一个对象，包含：
- name: 图例的名称（如果无法识别，使用"系列1"、"系列2"等默认名称）
- color: 图例对应的颜色（必须是标准的十六进制颜色代码，如#1f77b4）

重要要求：
1. 只返回JSON格式，不要添加任何额外文字说明
2. 颜色格式必须是标准的十六进制颜色代码（#RRGGBB格式）
3. 如果图表只有一个系列，返回一个元素；如果有多个系列，返回多个元素

示例格式：
{"legend_items": [{"name": "系列1", "color": "#1f77b4"}, {"name": "系列2", "color": "#ff7f0e"}]}'''},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ]

        response = chat_with_gemini(messages)

        response = response.strip()
        if '{' in response and '}' in response:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            response = response[json_start:json_end]

        result = json.loads(response)
        return result.get('legend_items', [])
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        return []
    except Exception as e:
        print(f"❌ 无法识别图例项: {e}")
        return []

def extract_roi_for_histogram(image_path, legend_count):
    """根据图例数量提取用于统计颜色直方图的ROI"""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图像文件")

        h, w, _ = image.shape

        if legend_count == 1:
            roi = image
        else:
            _, buffer = cv2.imencode('.png', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            messages = [
                {"role": "system", "content": "你是一个图表分析专家，请分析提供的图表图像。"},
                {"role": "user", "content": [
                    {"type": "text", "text": '''请分析这个图表，确定图表主体数据区域的边界坐标。

请返回一个JSON格式的响应，包含以下字段：
- x1: 区域左上角x坐标
- y1: 区域左上角y坐标
- x2: 区域右下角x坐标
- y2: 区域右下角y坐标

确保返回的坐标在图像范围内，且只包含图表的数据点/柱状图等主体元素。'''},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]}
            ]

            response = chat_with_gemini(messages)

            response = response.strip()
            if '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                response = response[json_start:json_end]

            roi_coords = json.loads(response)
            x1 = max(0, roi_coords.get('x1', int(w * 0.1)))
            y1 = max(0, roi_coords.get('y1', int(h * 0.1)))
            x2 = min(w, roi_coords.get('x2', int(w * 0.9)))
            y2 = min(h, roi_coords.get('y2', int(h * 0.9)))

            roi = image[y1:y2, x1:x2]

        return roi
    except Exception as e:
        print(f"❌ 无法提取ROI区域: {e}")
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x1, y1 = int(w * 0.1), int(h * 0.1)
        x2, y2 = int(w * 0.9), int(h * 0.9)
        return image[y1:y2, x1:x2]

def select_chart_series_color(image_path: str, candidate_colors: list) -> str:
    """让AI选择最适合的图表系列颜色"""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图像文件")

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        hex_colors = [bgr_to_hex(color) for color in candidate_colors if bgr_to_hex(color) is not None]

        messages = [
            {"role": "system", "content": "你是一个图表分析专家，请分析提供的图表图像并选择最适合的系列颜色。"},
            {"role": "user", "content": [
                {"type": "text", "text": f"请分析这个图表，并从提供的候选颜色中选择最适合作为本图系列颜色的颜色。\n\n图表类型是直角坐标系的（柱状图、折线图、散点图等）。\n\n候选颜色：{', '.join(hex_colors)}\n\n请只返回选中的颜色的16进制值，不要添加任何额外文字。"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ]

        selected_color = chat_with_gemini(messages)

        selected_color = selected_color.strip().strip('#')
        if len(selected_color) == 6:
            return f"#{selected_color}"
        else:
            return hex_colors[0]
    except Exception as e:
        print(f"❌ 无法选择图表系列颜色: {e}")
        return "#000000"

def compute_color_histogram(image):
    """计算图像的颜色直方图"""
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        return hist_h, hist_s, hist_v
    except Exception as e:
        print(f"❌ 无法计算颜色直方图: {e}")
        return None, None, None

def filter_colors_by_threshold(image, threshold=0.01):
    """根据阈值过滤图像中的颜色"""
    try:
        if image is None:
            return []
        pixels = image.reshape(-1, 3)
        color_counts = Counter(tuple(pixel) for pixel in pixels)
        total_pixels = len(pixels)
        filtered_colors = [color for color, count in color_counts.items() if count / total_pixels > threshold]
        return filtered_colors
    except Exception as e:
        print(f"❌ 无法过滤颜色: {e}")
        return []

def bgr_to_hex(bgr_color):
    """将BGR颜色转换为十六进制颜色代码"""
    try:
        b, g, r = bgr_color
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception as e:
        print(f"❌ 无法转换颜色格式: {e}")
        return None

def extract_chart_series_color(image_path):
    """提取图表中数据系列的主要颜色（同步版本）"""
    try:
        print(f"📊 处理图表: {image_path}")

        print("🔍 AI识别图例项及颜色...")
        legend_items = recognize_legend_items(image_path)

        if legend_items and len(legend_items) > 0:
            print(f"✅ 成功提取{len(legend_items)}个图例项及颜色")
            for item in legend_items:
                print(f"   {item.get('name', 'Unknown')}: {item.get('color', 'N/A')}")
            return legend_items
        else:
            print("⚠️ AI识别失败，使用默认颜色")
            return [{'name': '系列1', 'color': '#1f77b4'}]

    except Exception as e:
        print(f"❌ 提取图表颜色失败: {e}")
        return [{'name': '系列1', 'color': '#1f77b4'}]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("请输入测试图表路径: ")

    if not os.path.exists(image_path):
        print(f"❌ 图表文件不存在: {image_path}")
        sys.exit(1)

    series_colors = extract_chart_series_color(image_path)

    if series_colors:
        print(f"\n📋 最终结果:")
        if len(series_colors) == 1:
            print(f"   图表系列颜色: {series_colors[0]['color']}")
        else:
            for i, item in enumerate(series_colors, 1):
                print(f"   系列{i}: {item['name']} - {item['color']}")
