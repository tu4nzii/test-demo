import cv2
import numpy as np
from collections import Counter
import json
import sys
import os
import base64
import re
import requests
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_api_config import get_api_key, get_chat_completion_url, get_model_name

def _build_chat_completions_url(raw_url: str) -> str:
    url = (raw_url or "").strip().rstrip("/")
    if not url:
        return get_chat_completion_url()
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _split_env_keys(raw_keys: str) -> list:
    return [key.strip() for key in re.split(r"[,\s;]+", raw_keys or "") if key.strip()]


# API閰嶇疆
API_URL = _build_chat_completions_url(
    os.getenv("COLOR_LLM_API_URL")
    or os.getenv("COLOR_LLM_BASE_URL")
    or os.getenv("MLLM_API_URL")
    or os.getenv("MLLM_BASE_URL")
    or get_chat_completion_url()
)
ENV_API_KEYS = _split_env_keys(
    os.getenv("COLOR_LLM_API_KEYS")
    or os.getenv("COLOR_LLM_API_KEY")
    or os.getenv("MLLM_API_KEYS")
    or os.getenv("MLLM_API_KEY")
    or ""
)
DEFAULT_API_KEYS = [
    get_api_key()
]
API_KEYS = ENV_API_KEYS or DEFAULT_API_KEYS
key_index = 0
LLM_MODEL = os.getenv("COLOR_LLM_MODEL") or os.getenv("MLLM_MODEL") or get_model_name()
LLM_TEMPERATURE = float(os.getenv("COLOR_LLM_TEMPERATURE", "0.7"))
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("COLOR_LLM_TIMEOUT_SECONDS", "180"))
LLM_MAX_ATTEMPTS = int(os.getenv("COLOR_LLM_MAX_ATTEMPTS", "8"))
LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("COLOR_LLM_RETRY_BACKOFF_SECONDS", "2"))

def rotate_key():
    """鍒囨崲鍒颁笅涓€涓?key"""
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"馃攽 宸插垏鎹㈣嚦鏂扮殑 API Key [{key_index + 1}/{len(API_KEYS)}]")

def chat_with_gemini(messages: list) -> str:
    """涓嶨emini杩涜瀵硅瘽锛堝悓姝ョ増鏈級"""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": LLM_TEMPERATURE
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_index]}"
    }

    retryable_status = {429, 500, 502, 503, 504}

    for attempt in range(1, LLM_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT_SECONDS)

            if response.status_code in retryable_status:
                print(f"鈿狅笍 HTTP {response.status_code}: {response.text[:200]}")
                rotate_key()
                if attempt < LLM_MAX_ATTEMPTS:
                    wait_seconds = min(LLM_RETRY_BACKOFF_SECONDS * attempt, 20)
                    print(f"鈴?绛夊緟 {wait_seconds:.1f}s 鍚庨噸璇?[{attempt}/{LLM_MAX_ATTEMPTS}]...")
                    time.sleep(wait_seconds)
                continue

            if response.status_code != 200:
                print(f"鈿狅笍 HTTP {response.status_code}: {response.text[:200]}")
                return "The model API request failed."

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                print(f"鈿狅笍 鍝嶅簲鏍煎紡閿欒: {result}")
                if attempt < LLM_MAX_ATTEMPTS:
                    time.sleep(min(LLM_RETRY_BACKOFF_SECONDS * attempt, 20))

        except Exception as e:
            print(f"鉂?绗?{attempt} 娆″皾璇曞け璐? {e}")
            if attempt < LLM_MAX_ATTEMPTS:
                time.sleep(min(LLM_RETRY_BACKOFF_SECONDS * attempt, 20))
            continue

    print("鉂?鎵€鏈夊皾璇曞潎澶辫触")
    return "The model API request failed."

def count_legend_items(image_path: str) -> int:
    """鍒ゆ柇鍥捐〃涓殑鍥句緥鏁伴噺"""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("鏃犳硶璇诲彇鍥惧儚鏂囦欢")

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        messages = [
            {"role": "system", "content": "You are a chart analysis assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Count the legend items in this chart. Return only one integer, with no explanation."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ]

        response = chat_with_gemini(messages)
        legend_count = int(re.search(r'\d+', response).group())
        return legend_count
    except Exception as e:
        print(f"鉂?鏃犳硶璇嗗埆鍥句緥鏁伴噺: {e}")
        return 1

def recognize_legend_items(image_path: str) -> list:
    """璇嗗埆鍥捐〃涓殑鎵€鏈夊浘渚嬮」鍙婂叾棰滆壊"""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("鏃犳硶璇诲彇鍥惧儚鏂囦欢")

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        messages = [
            {"role": "system", "content": "You are a chart analysis assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": """Identify all legend items and their colors.
Return strict JSON only:
{"legend_items": [{"name": "Series 1", "color": "#1f77b4"}]}
Use #RRGGBB colors. Preserve visible legend names when readable."""},
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
        print(f"鉂?JSON瑙ｆ瀽閿欒: {e}")
        return []
    except Exception as e:
        print(f"鉂?鏃犳硶璇嗗埆鍥句緥椤? {e}")
        return []

def recognize_point_items(image_path: str) -> list:
    """Recognize labels attached to scatter/bubble data points."""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to read image file")
        height, width = image.shape[:2]
        if max(width, height) < 1400:
            scale = min(3.0, 1400 / max(width, height))
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise chart-reading assistant. Extract visible "
                    "labels attached to scatter or bubble data points."
                ),
            },
            {"role": "user", "content": [
                {"type": "text", "text": """
Identify every visible data-point label in this scatter/bubble chart.

Return only JSON in this exact shape:
{"point_items":[{"name":"US","color":"#8e8ef0"}]}

Rules:
1. Include labels printed inside or next to plotted bubbles/points, including tiny labels and labels in overlapping marker clusters.
2. Preserve the label text exactly as shown.
3. Exclude chart title, subtitle/source, axis titles, axis tick labels, grid labels, annotations, menu text, and watermark text.
4. Do not infer labels that are not visibly printed on the chart.
5. If the marker color is hard to determine, use null for color.
6. Look carefully near overlapping bubbles before finalizing the list.
7. Keep the visual left-to-right/top-to-bottom order as much as possible.
"""},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]},
        ]

        response = chat_with_gemini(messages).strip()
        if '{' in response and '}' in response:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            response = response[json_start:json_end]

        result = json.loads(response)
        items = result.get('point_items', [])
        if not isinstance(items, list):
            return []
        return _clean_point_items(items)
    except json.JSONDecodeError as e:
        print(f"閴?Point label JSON鐟欙絾鐎介柨娆掝嚖: {e}")
        return []
    except Exception as e:
        print(f"閴?閺冪姵纭剁拠鍡楀焼閺佺増宓侀悙瑙勭垼缁? {e}")
        return []

def _clean_point_items(items: list) -> list:
    cleaned = []
    seen = set()
    excluded = {
        "source",
        "highcharts",
        "safe fat intake",
        "safe sugar intake",
        "daily fat intake",
        "daily sugar intake",
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        lowered = name.lower()
        if any(token in lowered for token in excluded):
            continue
        if lowered in seen:
            continue
        color = item.get("color")
        if isinstance(color, str):
            color = color.strip()
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                color = None
        else:
            color = None
        cleaned.append({"name": name, "color": color})
        seen.add(lowered)
    return cleaned

def extract_roi_for_histogram(image_path, legend_count):
    """鏍规嵁鍥句緥鏁伴噺鎻愬彇鐢ㄤ簬缁熻棰滆壊鐩存柟鍥剧殑ROI"""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("鏃犳硶璇诲彇鍥惧儚鏂囦欢")

        h, w, _ = image.shape

        if legend_count == 1:
            roi = image
        else:
            _, buffer = cv2.imencode('.png', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            messages = [
                {"role": "system", "content": "You are a chart analysis assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Return strict JSON for the main plot area bounds: {\"x1\":0,\"y1\":0,\"x2\":100,\"y2\":100}. Coordinates must be within the image and exclude legends when possible."},
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
        print(f"鉂?鏃犳硶鎻愬彇ROI鍖哄煙: {e}")
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x1, y1 = int(w * 0.1), int(h * 0.1)
        x2, y2 = int(w * 0.9), int(h * 0.9)
        return image[y1:y2, x1:x2]

def select_chart_series_color(image_path: str, candidate_colors: list) -> str:
    """Ask the model to select the best series color."""
    try:
        image_path = os.path.normpath(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("鏃犳硶璇诲彇鍥惧儚鏂囦欢")

        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        hex_colors = [bgr_to_hex(color) for color in candidate_colors if bgr_to_hex(color) is not None]

        messages = [
            {"role": "system", "content": "You are a chart analysis assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Choose the candidate color that best represents the plotted data series in this chart. Candidates: {', '.join(hex_colors)}. Return only one #RRGGBB value."},
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
        print(f"鉂?鏃犳硶閫夋嫨鍥捐〃绯诲垪棰滆壊: {e}")
        return "#000000"

def compute_color_histogram(image):
    """Calculate an image color histogram."""
    try:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        return hist_h, hist_s, hist_v
    except Exception as e:
        print(f"鉂?鏃犳硶璁＄畻棰滆壊鐩存柟鍥? {e}")
        return None, None, None

def filter_colors_by_threshold(image, threshold=0.01):
    """Filter image colors by a frequency threshold."""
    try:
        if image is None:
            return []
        pixels = image.reshape(-1, 3)
        color_counts = Counter(tuple(pixel) for pixel in pixels)
        total_pixels = len(pixels)
        filtered_colors = [color for color, count in color_counts.items() if count / total_pixels > threshold]
        return filtered_colors
    except Exception as e:
        print(f"鉂?鏃犳硶杩囨护棰滆壊: {e}")
        return []

def bgr_to_hex(bgr_color):
    """Convert a BGR color to a hex color."""
    try:
        b, g, r = bgr_color
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception as e:
        print(f"鉂?鏃犳硶杞崲棰滆壊鏍煎紡: {e}")
        return None

def extract_chart_series_color(image_path):
    """Extract the main data-series color from a chart."""
    try:
        print(f"馃搳 澶勭悊鍥捐〃: {image_path}")

        print("馃攳 AI璇嗗埆鍥句緥椤瑰強棰滆壊...")
        legend_items = recognize_legend_items(image_path)

        if legend_items and len(legend_items) > 0:
            print(f"Extracted {len(legend_items)} legend items")
            for item in legend_items:
                print(f"   {item.get('name', 'Unknown')}: {item.get('color', 'N/A')}")
            return legend_items
        else:
            print("Legend recognition failed; using a default color")
            return [{'name': '绯诲垪1', 'color': '#1f77b4'}]

    except Exception as e:
        print(f"鉂?鎻愬彇鍥捐〃棰滆壊澶辫触: {e}")
        return [{'name': '绯诲垪1', 'color': '#1f77b4'}]

def extract_point_chart_items(image_path):
    """Extract prediction targets for scatter/bubble charts.

    Point charts often label each marker directly and have no legend. In that
    case, generic legend extraction can hallucinate names, so prefer visible
    point labels and only fall back to legend/color extraction if none are found.
    """
    point_items = recognize_point_items(image_path)
    if point_items:
        print(f"Detected {len(point_items)} point labels")
        return point_items
    return extract_chart_series_color(image_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("璇疯緭鍏ユ祴璇曞浘琛ㄨ矾寰? ")

    if not os.path.exists(image_path):
        print(f"鉂?鍥捐〃鏂囦欢涓嶅瓨鍦? {image_path}")
        sys.exit(1)

    series_colors = extract_chart_series_color(image_path)

    if series_colors:
        print(f"\n馃搵 鏈€缁堢粨鏋?")
        if len(series_colors) == 1:
            print(f"   鍥捐〃绯诲垪棰滆壊: {series_colors[0]['color']}")
        else:
            for i, item in enumerate(series_colors, 1):
                print(f"   绯诲垪{i}: {item['name']} - {item['color']}")
