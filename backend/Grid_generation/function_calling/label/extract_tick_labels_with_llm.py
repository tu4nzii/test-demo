# -*- coding: utf-8 -*-
"""
使用Gemini API提取图表刻度标签
支持轴类型判断和刻度值识别
"""

import os
import sys
import base64
import re
import json
import hashlib
import math
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np

# 导入gemini调用模块（gemini_calls.py在项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从 f:\program\test-demo\backend\Grid_generation\function_calling\label 向上3级到达项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def _build_chat_completions_url(raw_url: str) -> str:
    url = (raw_url or "").strip().rstrip("/")
    if not url:
        return "https://api.vveai.com/v1/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _split_env_keys(raw_keys: str) -> List[str]:
    return [key.strip() for key in re.split(r"[,\s;]+", raw_keys or "") if key.strip()]


# API配置
API_URL = _build_chat_completions_url(
    os.getenv("TICK_LLM_API_URL")
    or os.getenv("TICK_LLM_BASE_URL")
    or os.getenv("MLLM_API_URL")
    or os.getenv("MLLM_BASE_URL")
    or "https://api.vveai.com/v1"
)
ENV_API_KEYS = _split_env_keys(
    os.getenv("TICK_LLM_API_KEYS")
    or os.getenv("TICK_LLM_API_KEY")
    or os.getenv("MLLM_API_KEYS")
    or os.getenv("MLLM_API_KEY")
    or ""
)
DEFAULT_API_KEYS = [
    "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    "sk-2nzrUYD0JWLFzopWF477111f78E746AbAcA9Ed8534C3A481",
    "sk-CiD5WVUNIkBeXDgYB46b90C06aD24636BcEaBaFa993970C4",
    "sk-WvF4fU10VeOkfFMq579610Fc01E8496d827d0d3e04C44d0a",
    "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
]
API_KEYS = ENV_API_KEYS or DEFAULT_API_KEYS
key_index = 0
LLM_MODEL = os.getenv("TICK_LLM_MODEL") or os.getenv("MLLM_MODEL") or "gemini-2.5-pro"
LLM_TEMPERATURE = float(os.getenv("TICK_LLM_TEMPERATURE", "0"))
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("TICK_LLM_TIMEOUT_SECONDS", "180"))
LLM_MAX_ATTEMPTS = int(os.getenv("TICK_LLM_MAX_ATTEMPTS", "8"))
LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("TICK_LLM_RETRY_BACKOFF_SECONDS", "2"))
TICK_CACHE_SCHEMA_VERSION = "tick-mllm-v11"
TICK_SYSTEM_PROMPT = (
    "浣犳槸涓€涓笓涓氱殑鍥捐〃鍒嗘瀽涓撳锛屾搮闀胯瘑鍒浘琛ㄤ腑鐨勫潗鏍囪酱鍜屽埢搴︽爣绛俱€?"
)

TICK_SYSTEM_PROMPT = (
    "You are a precise chart-reading assistant. Extract only visible axis tick labels, "
    "preserve their order, and do not infer data values or legend text as ticks."
)


def rotate_key():
    """切换到下一个 key"""
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"🔑 已切换至新的 API Key [{key_index + 1}/{len(API_KEYS)}]")

def chat_with_gemini(messages: list) -> Optional[str]:
    """与Gemini进行对话（同步版本）"""
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
                print(f"⚠️ HTTP {response.status_code}: {response.text[:200]}")
                rotate_key()
                if attempt < LLM_MAX_ATTEMPTS:
                    wait_seconds = min(LLM_RETRY_BACKOFF_SECONDS * attempt, 20)
                    print(f"⏳ 等待 {wait_seconds:.1f}s 后重试 [{attempt}/{LLM_MAX_ATTEMPTS}]...")
                    time.sleep(wait_seconds)
                continue
            
            if response.status_code != 200:
                print(f"⚠️ HTTP {response.status_code}: {response.text[:200]}")
                return None
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                print(f"⚠️ 响应格式错误: {result}")
                if attempt < LLM_MAX_ATTEMPTS:
                    time.sleep(min(LLM_RETRY_BACKOFF_SECONDS * attempt, 20))
                
        except Exception as e:
            print(f"❌ 第 {attempt} 次尝试失败: {e}")
            if attempt < LLM_MAX_ATTEMPTS:
                time.sleep(min(LLM_RETRY_BACKOFF_SECONDS * attempt, 20))
            continue
    
    print("❌ 所有尝试均失败")
    return None


def get_cache_file_path(image_path: str, cache_dir: str) -> str:
    """
    根据图像路径生成缓存文件路径
    
    Args:
        image_path: 图像文件路径
        cache_dir: 缓存目录
    
    Returns:
        缓存文件路径
    """
    # 使用图像路径的hash作为文件名
    image_hash = hashlib.md5(image_path.encode('utf-8')).hexdigest()
    abs_path = os.path.abspath(image_path)
    try:
        stat = os.stat(abs_path)
        cache_key = f"tick-v2|{abs_path}|{stat.st_size}|{int(stat.st_mtime)}"
    except OSError:
        cache_key = f"tick-v2|{abs_path}"
    image_hash = hashlib.md5(cache_key.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"{image_hash}.json")
    return cache_file


def load_llm_cache(cache_file: str) -> Optional[Dict]:
    """
    从缓存文件加载LLM识别结果
    
    Args:
        cache_file: 缓存文件路径
    
    Returns:
        识别结果字典，如果文件不存在则返回None
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        print(f"[Warning] 读取缓存文件失败: {e}")
        return None


def save_llm_cache(cache_file: str, result: Dict, image_path: str, x_response: str = "", y_response: str = "") -> None:
    """
    保存LLM识别结果到缓存文件
    
    Args:
        cache_file: 缓存文件路径
        result: 识别结果字典
        image_path: 原始图像路径
        x_response: X轴LLM原始响应（可选）
        y_response: Y轴LLM原始响应（可选）
    """
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        cache_data = {
            "image_path": image_path,
            "x_ticks": result.get("x_ticks", []),
            "y_ticks": result.get("y_ticks", []),
            "x_axis_type": result.get("x_axis_type", "未知"),
            "y_axis_type": result.get("y_axis_type", "未知"),
            "x_llm_response": x_response,  # 保存原始响应以便调试
            "y_llm_response": y_response,
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Info] LLM识别结果已缓存: {cache_file}")
    except Exception as e:
        print(f"[Warning] 保存缓存文件失败: {e}")


def encode_image_to_base64(image_path: str) -> str:
    """
    将图像编码为base64字符串
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        base64编码的字符串
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_axis_crop_to_base64(image_path: str, direction: str) -> str:
    image_path = os.path.normpath(image_path)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return encode_image_to_base64(image_path)
    h, w = image.shape[:2]
    chart_type = os.path.basename(os.path.dirname(image_path)).lower()
    if direction.lower() == "x":
        crop = image[int(h * 0.52):h, 0:w]
    elif chart_type == "h_bar":
        crop = image[0:int(h * 0.82), 0:int(w * 0.68)]
    else:
        crop = image[0:h, 0:int(w * 0.34)]
    ok, buffer = cv2.imencode(".png", crop)
    if not ok:
        return encode_image_to_base64(image_path)
    return base64.b64encode(buffer).decode("utf-8")


def build_tick_extraction_prompt(direction: str = 'x') -> str:
    """
    构造提示词，增加轴类型判断和顺序要求
    
    Args:
        direction: 'x' 表示X轴，'y' 表示Y轴
    
    Returns:
        提示词字符串
    """
    # X轴从左到右，Y轴从下到上
    if direction.lower() == 'x':
        direction_text = "X轴"
        order_text = "从左到右"
    else:
        direction_text = "Y轴"
        order_text = "从下到上"
    
    prompt = f"""请识别图片中{direction_text}上的所有刻度值和轴类型。

**步骤1: 轴类型判断**
请首先判断该轴是"数值轴"还是"文字轴"：
- 数值轴：刻度为纯数字，如1, 2, 3.5, -4等
- 文字轴：刻度为文字、字母或年份、月份等非数值信息，如'Jan', 'Feb', '2020', 'A', 'B', 'Company A'等

**步骤2: 刻度值识别**
请识别{direction_text}上的所有刻度值，严格按照{order_text}的顺序识别和返回。

**输出格式示例：**

如果是数值轴：
```
轴类型:数值轴
刻度值:
```
数字1
数字2
...
```
```

如果是文字轴：
```
轴类型:文字轴
刻度值:
```
文字1
文字2
...
```
```

**重要要求：**
1. 刻度值必须按照{order_text}的顺序列出，不要颠倒顺序。
2. 对于长文本标签(如公司名称、地名、产品名称等)，请完整识别整个标签文本，不要截断。
3. 每个刻度值必须在一行内完整显示，不要使用换行符(\n)或任何其他特殊字符分隔。
4. 如果标签文本很长，请保持原样完整返回，不要添加换行符或分段。
5. 不要包含任何其他文字说明，只返回轴类型和刻度值。"""
    return prompt


def build_tick_extraction_prompt(direction: str = "x", chart_type: str = "") -> str:
    axis_name = "X axis" if direction.lower() == "x" else "Y axis"
    order = "left to right" if direction.lower() == "x" else "bottom to top"
    if chart_type == "h_bar" and direction.lower() == "y":
        return """
Read only the Y-axis category tick labels in this horizontal bar chart.

Return strict JSON with this schema:
{
  "axis_type": "text",
  "ticks": [
    {"position": 0, "label": "top-most category label"},
    {"position": 1, "label": "next category label below it"}
  ]
}

Rules:
1. Position 0 must be the label beside the top-most horizontal bar.
2. Increase position by 1 as labels move downward; the last item is beside the bottom-most bar.
3. Do not read the numeric scale at the bottom as Y-axis ticks.
4. Preserve category labels exactly, including punctuation and parenthesized text.
5. Return JSON only, with no Markdown and no explanation.
"""
    return f"""
Read the visible tick labels on the {axis_name} only.

Return strict JSON with this schema:
{{
  "axis_type": "numeric" or "text",
  "ticks": ["label1", "label2", "..."]
}}

Rules:
1. The ticks must be ordered {order}.
2. Include every visible tick label on this axis, including intermediate labels.
3. Do not include axis titles, legend labels, data labels, point labels, or grid values from the other axis.
4. Preserve text labels exactly. For numeric labels, keep signs, decimals, commas, and percentages if shown.
5. If labels are partially occluded, return only labels that are readable.
6. Decide axis_type by the role of the axis, not only by whether labels can be parsed as numbers.
7. Use "numeric" only for a continuous quantitative scale where inserted intermediate tick values would be meaningful on the chart.
8. Use "text" for discrete categories, names, IDs, dates, months, quarters, or observation periods. Calendar years on a line/bar chart are usually time-point labels, so classify them as text unless the axis is clearly a continuous numeric scale.
9. Do not convert time labels or category labels into numbers; preserve them as strings.
10. Return JSON only, with no Markdown and no explanation.
"""


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(data: Dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_tick_prompt_signature(chart_type: str = "") -> str:
    payload = {
        "cache_schema": TICK_CACHE_SCHEMA_VERSION,
        "chart_type": chart_type,
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
        "system_prompt": TICK_SYSTEM_PROMPT,
        "x_prompt": build_tick_extraction_prompt("x", chart_type),
        "y_prompt": build_tick_extraction_prompt("y", chart_type),
    }
    return stable_json_hash(payload)


def build_cache_metadata(image_path: str, dataset_id: str = "default") -> Dict:
    abs_path = os.path.abspath(image_path)
    chart_type = os.path.basename(os.path.dirname(abs_path)).lower()
    try:
        image_hash = sha256_file(abs_path)
    except OSError:
        image_hash = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
    return {
        "cache_schema": TICK_CACHE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "chart_type": chart_type,
        "image_sha256": image_hash,
        "prompt_signature": build_tick_prompt_signature(chart_type),
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
    }


def _axis_type_is_unknown(axis_type: object) -> bool:
    text = str(axis_type or "").strip().lower()
    return not text or "unknown" in text or "未知" in text or "未" in text


def _looks_like_api_failure(text: object) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return True
    failure_markers = [
        "暂时无法",
        "无法回应",
        "所有尝试均失败",
        "timed out",
        "timeout",
        "read timed out",
        "api failed",
        "request failed",
        "sorry",
        "apolog",
    ]
    return any(marker in value for marker in failure_markers)


def cache_result_quality(cache_data: Dict) -> Dict:
    if cache_data.get("cache_status") and cache_data.get("cache_status") != "ok":
        return {"valid": False, "reason": cache_data.get("failure_reason", "cache_status_not_ok")}

    x_response = cache_data.get("x_llm_response", "")
    y_response = cache_data.get("y_llm_response", "")
    if _looks_like_api_failure(x_response) or _looks_like_api_failure(y_response):
        return {"valid": False, "reason": "api_failure_response"}

    x_ticks = cache_data.get("x_ticks", [])
    y_ticks = cache_data.get("y_ticks", [])
    if not isinstance(x_ticks, list) or not isinstance(y_ticks, list):
        return {"valid": False, "reason": "malformed_ticks"}

    if not x_ticks and not y_ticks:
        return {"valid": False, "reason": "empty_ticks"}

    return {"valid": True, "reason": "ok"}


def llm_axis_result_is_valid(result: Dict) -> bool:
    if result.get("status") != "ok":
        return False
    if _looks_like_api_failure(result.get("raw_response", "")):
        return False
    return bool(result.get("ticks")) or not _axis_type_is_unknown(result.get("axis_type"))


def _clean_hbar_category_tick(value):
    text = str(value).strip()
    return re.sub(r"\s+[-–—]\s*$", "", text).strip()


def _finite_numeric_values(values):
    numeric = []
    for value in values or []:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        numeric.append(number)
    return numeric


def _replace_nonfinite_ticks(values):
    cleaned = []
    changed = False
    for value in values or []:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned.append("nan")
            changed = True
        else:
            cleaned.append(value)
    return cleaned, changed


def get_cache_file_path(image_path: str, cache_dir: str, dataset_id: str = "default", prompt_signature: str = None) -> str:
    abs_path = os.path.abspath(image_path)
    chart_type = os.path.basename(os.path.dirname(abs_path)).lower()
    try:
        image_hash = sha256_file(abs_path)
    except OSError:
        image_hash = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
    cache_key = {
        "schema": TICK_CACHE_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "chart_type": chart_type,
        "image_sha256": image_hash,
        "prompt_signature": prompt_signature or build_tick_prompt_signature(chart_type),
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
    }
    return os.path.join(cache_dir, f"{stable_json_hash(cache_key)}.json")


def load_llm_cache(cache_file: str, expected_metadata: Optional[Dict] = None) -> Optional[Dict]:
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        if expected_metadata:
            cached_metadata = cache_data.get("metadata", {})
            for key, value in expected_metadata.items():
                if cached_metadata.get(key) != value:
                    return None
        quality = cache_result_quality(cache_data)
        if not quality["valid"]:
            print(f"[Info] 忽略无效LLM缓存({quality['reason']}): {cache_file}")
            return None
        return cache_data
    except Exception as e:
        print(f"[Warning] 读取LLM缓存失败: {e}")
        return None


def save_llm_cache(
    cache_file: str,
    result: Dict,
    image_path: str,
    x_response: str = "",
    y_response: str = "",
    metadata: Optional[Dict] = None,
) -> None:
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        cache_data = {
            "metadata": metadata or {},
            "cache_status": result.get("cache_status", "ok"),
            "image_path": image_path,
            "x_ticks": result.get("x_ticks", []),
            "y_ticks": result.get("y_ticks", []),
            "x_axis_type": result.get("x_axis_type", "unknown"),
            "y_axis_type": result.get("y_axis_type", "unknown"),
            "x_llm_response": x_response,
            "y_llm_response": y_response,
            "x_status": result.get("x_status", "ok"),
            "y_status": result.get("y_status", "ok"),
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"[Info] LLM识别结果已缓存: {cache_file}")
    except Exception as e:
        print(f"[Warning] 保存LLM缓存失败: {e}")


def extract_axis_ticks_with_llm(image_path: str, direction: str = 'x') -> Dict:
    """
    使用LLM识别指定轴的刻度值和类型（同步版本）
    
    Args:
        image_path: 图像文件路径
        direction: 'x' 表示X轴，'y' 表示Y轴
    
    Returns:
        包含axis_type、ticks和raw_response的字典，格式: {"axis_type": "数值轴"或"文字轴", "ticks": [...], "raw_response": "..."}
    """
    try:
        # 读取图像并编码为base64
        image_base64 = encode_axis_crop_to_base64(image_path, direction)
        
        chart_type = os.path.basename(os.path.dirname(image_path)).lower()
        # 构建提示词
        prompt = build_tick_extraction_prompt(direction, chart_type)
        
        # 构建包含图像的消息
        messages = [
            {
                "role": "system",
                "content": TICK_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        # 调用Gemini API（同步版本）
        response = chat_with_gemini(messages)
        if not response:
            return {
                "axis_type": "unknown",
                "ticks": [],
                "raw_response": "",
                "status": "api_failed",
            }

        # 解析响应
        result = parse_llm_response(response, direction)
        if chart_type == "h_bar" and direction.lower() == "y" and result.get("ticks"):
            result["ticks"] = [_clean_hbar_category_tick(tick) for tick in reversed(result["ticks"])]
        else:
            numeric_ticks = _finite_numeric_values(result.get("ticks", []))
            if direction.lower() == "y" and numeric_ticks and len(numeric_ticks) >= 2:
                if numeric_ticks[0] > numeric_ticks[-1]:
                    result["ticks"] = list(reversed(result["ticks"]))
        result["raw_response"] = response  # 保存原始响应
        result["status"] = "ok"
        return result
        
    except Exception as e:
        print(f"[Error] LLM识别{direction}轴刻度失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "axis_type": "unknown",
            "ticks": [],
            "raw_response": "",
            "status": "exception",
        }


def parse_llm_response(response_text: str, direction: str) -> Dict:
    """
    解析LLM响应，提取轴类型和刻度值
    
    Args:
        response_text: LLM返回的文本
        direction: 轴方向（用于错误提示）
    
    Returns:
        包含axis_type和ticks的字典
    """
    result = {
        "axis_type": "未知",
        "ticks": []
    }
    
    try:
        # 提取轴类型
        axis_type_match = re.search(r'轴类型[：:]\s*(数值轴|文字轴)', response_text)
        if axis_type_match:
            result["axis_type"] = axis_type_match.group(1)
        
        # 提取刻度值部分
        # 查找刻度值区域（在"刻度值:"之后）
        ticks_section_match = re.search(r'刻度值[：:]\s*\n?```\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if ticks_section_match:
            ticks_text = ticks_section_match.group(1).strip()
        else:
            # 如果没有找到代码块，尝试查找"刻度值:"之后的所有行
            ticks_section_match = re.search(r'刻度值[：:]\s*\n(.*?)(?=\n\n|\Z)', response_text, re.DOTALL)
            if ticks_section_match:
                ticks_text = ticks_section_match.group(1).strip()
            else:
                # 如果还是找不到，尝试提取所有非空行
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                # 找到"刻度值:"之后的行
                start_idx = -1
                for i, line in enumerate(lines):
                    if '刻度值' in line:
                        start_idx = i + 1
                        break
                if start_idx > 0:
                    ticks_text = '\n'.join(lines[start_idx:])
                else:
                    ticks_text = ""
        
        # 解析刻度值列表
        if ticks_text:
            ticks = []
            for line in ticks_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('```'):
                    # 尝试转换数值
                    try:
                        # 如果是数值，转换为float
                        tick_value = float(line)
                        ticks.append(tick_value)
                    except ValueError:
                        # 如果是文字，保持原样
                        ticks.append(line)
            
            result["ticks"] = ticks
        
        # 如果没有找到刻度值，尝试从整个响应中提取
        if not result["ticks"]:
            # 尝试提取所有可能的数值或文字
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not any(keyword in line for keyword in ['轴类型', '刻度值', '```', '---']):
                    try:
                        tick_value = float(line)
                        result["ticks"].append(tick_value)
                    except ValueError:
                        if len(line) > 0:
                            result["ticks"].append(line)
        
    except Exception as e:
        print(f"[Warning] 解析LLM响应失败: {e}")
        print(f"[Debug] 响应内容: {response_text[:500]}")
    
    return result


def parse_llm_response(response_text: str, direction: str) -> Dict:
    result = {"axis_type": "unknown", "ticks": []}
    text = (response_text or "").strip()
    if not text:
        return result

    try:
        match = re.search(r"\{[\s\S]*\}", text)
        payload = json.loads(match.group(0) if match else text)
        axis_type = str(payload.get("axis_type", "unknown")).strip().lower()
        is_numeric_axis = axis_type in {"numeric", "number", "value"}
        result["axis_type"] = "数值轴" if is_numeric_axis else "文字轴"
        ticks = payload.get("ticks", [])
        if isinstance(ticks, list):
            if ticks and all(isinstance(item, dict) for item in ticks):
                def tick_position(item):
                    for key in ("position", "index", "rank", "from_bottom"):
                        try:
                            return float(item[key])
                        except (KeyError, TypeError, ValueError):
                            continue
                    return float("inf")

                ticks = sorted(ticks, key=tick_position)
                ticks = [
                    item.get("label", item.get("text", item.get("tick", "")))
                    for item in ticks
                ]
            cleaned = []
            has_text_value = False
            for item in ticks:
                value = str(item).strip()
                if not value:
                    continue
                if not is_numeric_axis:
                    cleaned.append(value)
                    has_text_value = True
                    continue
                lower_value = value.lower()
                if lower_value in {"nan", "na", "n/a", "null", "none"}:
                    cleaned.append(value)
                    has_text_value = True
                    continue
                try:
                    numeric_value = float(value.replace(",", "").replace("−", "-").replace("–", "-").rstrip("%"))
                    if math.isnan(numeric_value) or math.isinf(numeric_value):
                        cleaned.append(value)
                        has_text_value = True
                    else:
                        cleaned.append(numeric_value)
                except ValueError:
                    cleaned.append(value)
                    has_text_value = True
            result["ticks"] = cleaned
            if has_text_value:
                result["axis_type"] = "文字轴"
            return result
    except Exception:
        pass

    # Fallback for old or non-JSON responses.
    lines = [
        line.strip().strip("-* ")
        for line in text.replace("```json", "```").splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    for line in lines:
        lowered = line.lower()
        if "axis" in lowered or "tick" in lowered or "刻度" in line or "类型" in line:
            continue
        try:
            result["ticks"].append(float(line.replace(",", "").replace("−", "-").replace("–", "-").rstrip("%")))
        except ValueError:
            result["ticks"].append(line)
    if result["ticks"] and _as_numeric_list(result["ticks"]) is not None:
        result["axis_type"] = "数值轴"
    elif result["ticks"]:
        result["axis_type"] = "文字轴"
    return result


def _as_numeric_list(values):
    numeric = []
    for value in values or []:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return None
    return numeric


def _legacy_extract_tick_labels_with_llm_unused(
    image_path: str,
    cache_dir: Optional[str] = None,
    allow_api: bool = True,
    dataset_id: str = "default",
) -> Dict:
    """
    使用LLM识别图表的X轴和Y轴刻度标签（支持缓存，同步版本）
    
    Args:
        image_path: 图像文件路径
        cache_dir: 缓存目录（如果为None，则不使用缓存）
    
    Returns:
        包含x_ticks和y_ticks的字典，格式: {"x_ticks": [...], "y_ticks": [...]}
    """
    # 检查缓存
    if cache_dir:
        cache_file = get_cache_file_path(image_path, cache_dir)
        cached_result = load_llm_cache(cache_file)
        if cached_result:
            print(f"[Info] 从缓存加载LLM识别结果: {cache_file}")
            return {
                "x_ticks": cached_result.get("x_ticks", []),
                "y_ticks": cached_result.get("y_ticks", []),
                "x_axis_type": cached_result.get("x_axis_type", "未知"),
                "y_axis_type": cached_result.get("y_axis_type", "未知")
            }

    if not allow_api:
        return {
            "x_ticks": [],
            "y_ticks": [],
            "x_axis_type": "unknown",
            "y_axis_type": "unknown",
            "cache_miss": True,
        }

    print(f"[Info] 开始使用LLM识别刻度标签: {image_path}")
    
    # 分别识别X轴和Y轴
    x_result = extract_axis_ticks_with_llm(image_path, direction='x')
    y_result = extract_axis_ticks_with_llm(image_path, direction='y')
    
    # 获取原始响应（如果可用）
    x_response = x_result.get("raw_response", "")
    y_response = y_result.get("raw_response", "")
    
    result = {
        "x_ticks": x_result.get("ticks", []),
        "y_ticks": y_result.get("ticks", []),
        "x_axis_type": x_result.get("axis_type", "未知"),
        "y_axis_type": y_result.get("axis_type", "未知")
    }
    
    print(f"[Info] X轴识别结果: 类型={result['x_axis_type']}, 刻度数={len(result['x_ticks'])}")
    print(f"[Info] Y轴识别结果: 类型={result['y_axis_type']}, 刻度数={len(result['y_ticks'])}")
    
    # 保存到缓存
    if cache_dir:
        cache_file = get_cache_file_path(image_path, cache_dir)
        save_llm_cache(cache_file, result, image_path, x_response, y_response)
    
    return result


def extract_tick_labels_with_llm(
    image_path: str,
    cache_dir: Optional[str] = None,
    allow_api: bool = True,
    dataset_id: str = "default",
) -> Dict:
    """Extract X/Y tick labels with a prompt- and dataset-aware MLLM cache."""
    cache_file = None
    metadata = None
    if cache_dir:
        metadata = build_cache_metadata(image_path, dataset_id=dataset_id)
        cache_file = get_cache_file_path(
            image_path,
            cache_dir,
            dataset_id=dataset_id,
            prompt_signature=metadata["prompt_signature"],
        )
        cached_result = load_llm_cache(cache_file, expected_metadata=metadata)
        if cached_result:
            print(f"[Info] 从缓存加载LLM识别结果: {cache_file}")
            chart_type = os.path.basename(os.path.dirname(image_path)).lower()
            x_ticks, x_had_nonfinite = _replace_nonfinite_ticks(cached_result.get("x_ticks", []))
            y_ticks, y_had_nonfinite = _replace_nonfinite_ticks(cached_result.get("y_ticks", []))
            if chart_type == "h_bar":
                y_ticks = [_clean_hbar_category_tick(tick) for tick in y_ticks]
            else:
                numeric_y = _finite_numeric_values(y_ticks)
                if numeric_y and len(numeric_y) >= 2 and numeric_y[0] > numeric_y[-1]:
                    y_ticks = list(reversed(y_ticks))
            return {
                "x_ticks": x_ticks,
                "y_ticks": y_ticks,
                "x_axis_type": "文字轴" if x_had_nonfinite else cached_result.get("x_axis_type", "unknown"),
                "y_axis_type": "文字轴" if y_had_nonfinite else cached_result.get("y_axis_type", "unknown"),
                "cache_hit": True,
                "cache_miss": False,
                "cache_file": cache_file,
                "cache_status": cached_result.get("cache_status", "ok"),
            }

    if not allow_api:
        return {
            "x_ticks": [],
            "y_ticks": [],
            "x_axis_type": "unknown",
            "y_axis_type": "unknown",
            "cache_hit": False,
            "cache_miss": True,
            "cache_file": cache_file,
        }

    print(f"[Info] 开始使用LLM识别刻度标签: {image_path}")
    x_result = extract_axis_ticks_with_llm(image_path, direction="x")
    y_result = extract_axis_ticks_with_llm(image_path, direction="y")

    result = {
        "x_ticks": x_result.get("ticks", []),
        "y_ticks": y_result.get("ticks", []),
        "x_axis_type": x_result.get("axis_type", "unknown"),
        "y_axis_type": y_result.get("axis_type", "unknown"),
        "x_status": x_result.get("status", "unknown"),
        "y_status": y_result.get("status", "unknown"),
        "cache_hit": False,
        "cache_miss": False,
        "cache_file": cache_file,
    }
    print(f"[Info] X轴识别结果: 类型={result['x_axis_type']}, 刻度数={len(result['x_ticks'])}")
    print(f"[Info] Y轴识别结果: 类型={result['y_axis_type']}, 刻度数={len(result['y_ticks'])}")

    if not (llm_axis_result_is_valid(x_result) and llm_axis_result_is_valid(y_result)):
        result["api_failed"] = True
        result["cache_status"] = "invalid"
        result["failure_reason"] = f"x={result['x_status']};y={result['y_status']}"
        print(f"[Warning] LLM结果无效，不写入正式缓存: {result['failure_reason']}")
        return result

    result["cache_status"] = "ok"
    if cache_dir and cache_file:
        save_llm_cache(
            cache_file,
            result,
            image_path,
            x_result.get("raw_response", ""),
            y_result.get("raw_response", ""),
            metadata=metadata,
        )

    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试LLM刻度标签提取')
    parser.add_argument('--image', type=str, required=True, help='图像文件路径')
    args = parser.parse_args()
    
    result = extract_tick_labels_with_llm(args.image)
    print("\n识别结果:")
    print(f"X轴类型: {result['x_axis_type']}")
    print(f"X轴刻度: {result['x_ticks']}")
    print(f"Y轴类型: {result['y_axis_type']}")
    print(f"Y轴刻度: {result['y_ticks']}")
