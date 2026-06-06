# -*- coding: utf-8 -*-
"""
浣跨敤Gemini API鎻愬彇鍥捐〃鍒诲害鏍囩
鏀寔杞寸被鍨嬪垽鏂拰鍒诲害鍊艰瘑鍒?
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

# 娣诲姞椤圭洰璺緞
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np

# 瀵煎叆gemini璋冪敤妯″潡锛坓emini_calls.py鍦ㄩ」鐩牴鐩綍锛?
current_dir = os.path.dirname(os.path.abspath(__file__))
# 浠?f:\program\test-demo\backend\Grid_generation\function_calling\label 鍚戜笂3绾у埌杈鹃」鐩牴鐩綍
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


def _split_env_keys(raw_keys: str) -> List[str]:
    return [key.strip() for key in re.split(r"[,\s;]+", raw_keys or "") if key.strip()]


# API閰嶇疆
API_URL = _build_chat_completions_url(
    os.getenv("TICK_LLM_API_URL")
    or os.getenv("TICK_LLM_BASE_URL")
    or os.getenv("MLLM_API_URL")
    or os.getenv("MLLM_BASE_URL")
    or get_chat_completion_url()
)
ENV_API_KEYS = _split_env_keys(
    os.getenv("TICK_LLM_API_KEYS")
    or os.getenv("TICK_LLM_API_KEY")
    or os.getenv("MLLM_API_KEYS")
    or os.getenv("MLLM_API_KEY")
    or ""
)
DEFAULT_API_KEYS = [
    get_api_key()
]
API_KEYS = ENV_API_KEYS or DEFAULT_API_KEYS
key_index = 0
LLM_MODEL = os.getenv("TICK_LLM_MODEL") or os.getenv("MLLM_MODEL") or get_model_name()
LLM_TEMPERATURE = float(os.getenv("TICK_LLM_TEMPERATURE", "0"))
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("TICK_LLM_TIMEOUT_SECONDS", "180"))
LLM_MAX_ATTEMPTS = int(os.getenv("TICK_LLM_MAX_ATTEMPTS", "8"))
LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("TICK_LLM_RETRY_BACKOFF_SECONDS", "2"))
TICK_CACHE_SCHEMA_VERSION = "tick-mllm-v11"
TICK_SYSTEM_PROMPT = (
    "娴ｇ姵妲告稉鈧稉顏冪瑩娑撴氨娈戦崶鎹愩€冮崚鍡樼€芥稉鎾愁啀閿涘本鎼梹鑳槕閸掝偄娴樼悰銊よ厬閻ㄥ嫬娼楅弽鍥叡閸滃苯鍩㈡惔锔界垼缁涗勘鈧?"
)

TICK_SYSTEM_PROMPT = (
    "You are a precise chart-reading assistant. Extract only visible axis tick labels, "
    "preserve their order, and do not infer data values or legend text as ticks."
)


def rotate_key():
    """鍒囨崲鍒颁笅涓€涓?key"""
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"馃攽 宸插垏鎹㈣嚦鏂扮殑 API Key [{key_index + 1}/{len(API_KEYS)}]")

def chat_with_gemini(messages: list) -> Optional[str]:
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
                return None
            
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
    return None


def get_cache_file_path(image_path: str, cache_dir: str) -> str:
    """
    鏍规嵁鍥惧儚璺緞鐢熸垚缂撳瓨鏂囦欢璺緞
    
    Args:
        image_path: 鍥惧儚鏂囦欢璺緞
        cache_dir: 缂撳瓨鐩綍
    
    Returns:
        缂撳瓨鏂囦欢璺緞
    """
    # 浣跨敤鍥惧儚璺緞鐨刪ash浣滀负鏂囦欢鍚?
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
    浠庣紦瀛樻枃浠跺姞杞絃LM璇嗗埆缁撴灉
    
    Args:
        cache_file: 缂撳瓨鏂囦欢璺緞
    
    Returns:
        璇嗗埆缁撴灉瀛楀吀锛屽鏋滄枃浠朵笉瀛樺湪鍒欒繑鍥濶one
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        print(f"[Warning] 璇诲彇缂撳瓨鏂囦欢澶辫触: {e}")
        return None


def save_llm_cache(cache_file: str, result: Dict, image_path: str, x_response: str = "", y_response: str = "") -> None:
    """
    淇濆瓨LLM璇嗗埆缁撴灉鍒扮紦瀛樻枃浠?
    
    Args:
        cache_file: 缂撳瓨鏂囦欢璺緞
        result: 璇嗗埆缁撴灉瀛楀吀
        image_path: 鍘熷鍥惧儚璺緞
        x_response: X杞碙LM鍘熷鍝嶅簲锛堝彲閫夛級
        y_response: Y杞碙LM鍘熷鍝嶅簲锛堝彲閫夛級
    """
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        cache_data = {
            "image_path": image_path,
            "x_ticks": result.get("x_ticks", []),
            "y_ticks": result.get("y_ticks", []),
            "x_axis_type": result.get("x_axis_type", "鏈煡"),
            "y_axis_type": result.get("y_axis_type", "鏈煡"),
            "x_llm_response": x_response,  # 淇濆瓨鍘熷鍝嶅簲浠ヤ究璋冭瘯
            "y_llm_response": y_response,
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Info] LLM璇嗗埆缁撴灉宸茬紦瀛? {cache_file}")
    except Exception as e:
        print(f"[Warning] 淇濆瓨缂撳瓨鏂囦欢澶辫触: {e}")


def encode_image_to_base64(image_path: str) -> str:
    """
    灏嗗浘鍍忕紪鐮佷负base64瀛楃涓?
    
    Args:
        image_path: 鍥惧儚鏂囦欢璺緞
    
    Returns:
        base64缂栫爜鐨勫瓧绗︿覆
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _bar_base_type(chart_type: str = "") -> str:
    chart_type = (chart_type or "").lower()
    if chart_type in {"h_bar", "h_stacked_bar"}:
        return "h_bar"
    if chart_type in {"v_bar", "v_stacked_bar"}:
        return "v_bar"
    return chart_type


def encode_axis_crop_to_base64(image_path: str, direction: str, chart_type: str = "") -> str:
    image_path = os.path.normpath(image_path)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return encode_image_to_base64(image_path)
    h, w = image.shape[:2]
    chart_type = _bar_base_type(chart_type or os.path.basename(os.path.dirname(image_path)))
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


def build_tick_extraction_prompt(direction: str = "x", chart_type: str = "") -> str:
    axis_name = "X axis" if direction.lower() == "x" else "Y axis"
    order = "left to right" if direction.lower() == "x" else "bottom to top"
    if _bar_base_type(chart_type) == "h_bar" and direction.lower() == "y":
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


def build_cache_metadata(image_path: str, dataset_id: str = "default", chart_type_override: str = "") -> Dict:
    abs_path = os.path.abspath(image_path)
    chart_type = (chart_type_override or os.path.basename(os.path.dirname(abs_path))).lower()
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
    return not text or "unknown" in text or "none" in text


def _looks_like_api_failure(text: object) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return True
    failure_markers = [
        "鏆傛椂鏃犳硶",
        "鏃犳硶鍥炲簲",
        "鎵€鏈夊皾璇曞潎澶辫触",
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
    return re.sub(r"\s+[-\u2010-\u2015]\s*$", "", text).strip()


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


def get_cache_file_path(
    image_path: str,
    cache_dir: str,
    dataset_id: str = "default",
    prompt_signature: str = None,
    chart_type_override: str = "",
) -> str:
    abs_path = os.path.abspath(image_path)
    chart_type = (chart_type_override or os.path.basename(os.path.dirname(abs_path))).lower()
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
            print(f"[Info] 蹇界暐鏃犳晥LLM缂撳瓨({quality['reason']}): {cache_file}")
            return None
        return cache_data
    except Exception as e:
        print(f"[Warning] 璇诲彇LLM缂撳瓨澶辫触: {e}")
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
        print(f"[Info] LLM璇嗗埆缁撴灉宸茬紦瀛? {cache_file}")
    except Exception as e:
        print(f"[Warning] 淇濆瓨LLM缂撳瓨澶辫触: {e}")


def extract_axis_ticks_with_llm(image_path: str, direction: str = 'x', chart_type_override: str = "") -> Dict:
    """Extract axis tick labels and axis type with the configured MLLM."""
    try:
        # 璇诲彇鍥惧儚骞剁紪鐮佷负base64
        chart_type = (chart_type_override or os.path.basename(os.path.dirname(image_path))).lower()
        image_base64 = encode_axis_crop_to_base64(image_path, direction, chart_type=chart_type)
        prompt = build_tick_extraction_prompt(direction, chart_type)
        
        # 鏋勫缓鎻愮ず璇?
        prompt = build_tick_extraction_prompt(direction, chart_type)
        
        # 鏋勫缓鍖呭惈鍥惧儚鐨勬秷鎭?
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
        
        # 璋冪敤Gemini API锛堝悓姝ョ増鏈級
        response = chat_with_gemini(messages)
        if not response:
            return {
                "axis_type": "unknown",
                "ticks": [],
                "raw_response": "",
                "status": "api_failed",
            }

        # 瑙ｆ瀽鍝嶅簲
        result = parse_llm_response(response, direction)
        if _bar_base_type(chart_type) == "h_bar" and direction.lower() == "y" and result.get("ticks"):
            result["ticks"] = [_clean_hbar_category_tick(tick) for tick in reversed(result["ticks"])]
        else:
            numeric_ticks = _finite_numeric_values(result.get("ticks", []))
            if direction.lower() == "y" and numeric_ticks and len(numeric_ticks) >= 2:
                if numeric_ticks[0] > numeric_ticks[-1]:
                    result["ticks"] = list(reversed(result["ticks"]))
        result["raw_response"] = response  # 淇濆瓨鍘熷鍝嶅簲
        result["status"] = "ok"
        return result
        
    except Exception as e:
        print(f"[Error] LLM璇嗗埆{direction}杞村埢搴﹀け璐? {e}")
        import traceback
        traceback.print_exc()
        return {
            "axis_type": "unknown",
            "ticks": [],
            "raw_response": "",
            "status": "exception",
        }


def parse_llm_response(response_text: str, direction: str) -> Dict:
    text = (response_text or "").strip()
    result = {"axis_type": "unknown", "ticks": []}
    if not text:
        return result

    try:
        match = re.search(r"\{[\s\S]*\}", text)
        payload = json.loads(match.group(0) if match else text)
        axis_type = str(payload.get("axis_type", "unknown")).strip().lower()
        result["axis_type"] = "numeric" if axis_type in {"numeric", "number", "value"} else "text"
        ticks = payload.get("ticks", [])
        if isinstance(ticks, list):
            if ticks and all(isinstance(item, dict) for item in ticks):
                ticks = sorted(ticks, key=lambda item: float(item.get("position", item.get("index", 1e9))))
                ticks = [item.get("label", item.get("text", item.get("tick", ""))) for item in ticks]
            result["ticks"] = [_normalize_tick_value(item, result["axis_type"]) for item in ticks if str(item).strip()]
            return result
    except Exception:
        pass

    for line in text.replace("```json", "```").splitlines():
        line = line.strip().strip("-* ")
        lowered = line.lower()
        if not line or line.startswith("```") or "axis" in lowered or "tick" in lowered:
            continue
        result["ticks"].append(_normalize_tick_value(line, "numeric"))
    result["axis_type"] = "numeric" if _as_numeric_list(result["ticks"]) is not None else "text"
    return result


def _normalize_tick_value(value, axis_type: str):
    text = str(value).strip()
    if axis_type != "numeric":
        return text
    try:
        return float(text.replace(",", "").rstrip("%"))
    except ValueError:
        return text


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
    """Extract X/Y tick labels with optional cache support."""
    # 妫€鏌ョ紦瀛?
    if cache_dir:
        cache_file = get_cache_file_path(image_path, cache_dir)
        cached_result = load_llm_cache(cache_file)
        if cached_result:
            print(f"[Info] 浠庣紦瀛樺姞杞絃LM璇嗗埆缁撴灉: {cache_file}")
            return {
                "x_ticks": cached_result.get("x_ticks", []),
                "y_ticks": cached_result.get("y_ticks", []),
                "x_axis_type": cached_result.get("x_axis_type", "鏈煡"),
                "y_axis_type": cached_result.get("y_axis_type", "鏈煡")
            }

    if not allow_api:
        return {
            "x_ticks": [],
            "y_ticks": [],
            "x_axis_type": "unknown",
            "y_axis_type": "unknown",
            "cache_miss": True,
        }

    print(f"[Info] 寮€濮嬩娇鐢↙LM璇嗗埆鍒诲害鏍囩: {image_path}")
    
    # 鍒嗗埆璇嗗埆X杞村拰Y杞?
    x_result = extract_axis_ticks_with_llm(image_path, direction='x')
    y_result = extract_axis_ticks_with_llm(image_path, direction='y')
    
    # 鑾峰彇鍘熷鍝嶅簲锛堝鏋滃彲鐢級
    x_response = x_result.get("raw_response", "")
    y_response = y_result.get("raw_response", "")
    
    result = {
        "x_ticks": x_result.get("ticks", []),
        "y_ticks": y_result.get("ticks", []),
        "x_axis_type": x_result.get("axis_type", "鏈煡"),
        "y_axis_type": y_result.get("axis_type", "鏈煡")
    }
    
    print(f"[Info] X杞磋瘑鍒粨鏋? 绫诲瀷={result['x_axis_type']}, 鍒诲害鏁?{len(result['x_ticks'])}")
    print(f"[Info] Y杞磋瘑鍒粨鏋? 绫诲瀷={result['y_axis_type']}, 鍒诲害鏁?{len(result['y_ticks'])}")
    
    # 淇濆瓨鍒扮紦瀛?
    if cache_dir:
        cache_file = get_cache_file_path(image_path, cache_dir)
        save_llm_cache(cache_file, result, image_path, x_response, y_response)
    
    return result


def extract_tick_labels_with_llm(
    image_path: str,
    cache_dir: Optional[str] = None,
    allow_api: bool = True,
    dataset_id: str = "default",
    chart_type_override: str = "",
) -> Dict:
    """Extract X/Y tick labels with a prompt- and dataset-aware MLLM cache."""
    cache_file = None
    metadata = None
    chart_type = (chart_type_override or os.path.basename(os.path.dirname(image_path))).lower()
    if cache_dir:
        metadata = build_cache_metadata(image_path, dataset_id=dataset_id, chart_type_override=chart_type)
        cache_file = get_cache_file_path(
            image_path,
            cache_dir,
            dataset_id=dataset_id,
            prompt_signature=metadata["prompt_signature"],
            chart_type_override=chart_type,
        )
        cached_result = load_llm_cache(cache_file, expected_metadata=metadata)
        if cached_result:
            print(f"[Info] 浠庣紦瀛樺姞杞絃LM璇嗗埆缁撴灉: {cache_file}")
            x_ticks, x_had_nonfinite = _replace_nonfinite_ticks(cached_result.get("x_ticks", []))
            y_ticks, y_had_nonfinite = _replace_nonfinite_ticks(cached_result.get("y_ticks", []))
            if _bar_base_type(chart_type) == "h_bar":
                y_ticks = [_clean_hbar_category_tick(tick) for tick in y_ticks]
            else:
                numeric_y = _finite_numeric_values(y_ticks)
                if numeric_y and len(numeric_y) >= 2 and numeric_y[0] > numeric_y[-1]:
                    y_ticks = list(reversed(y_ticks))
            return {
                "x_ticks": x_ticks,
                "y_ticks": y_ticks,
                "x_axis_type": "text" if x_had_nonfinite else cached_result.get("x_axis_type", "unknown"),
                "y_axis_type": "text" if y_had_nonfinite else cached_result.get("y_axis_type", "unknown"),
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

    print(f"[Info] 寮€濮嬩娇鐢↙LM璇嗗埆鍒诲害鏍囩: {image_path}")
    x_result = extract_axis_ticks_with_llm(image_path, direction="x", chart_type_override=chart_type)
    y_result = extract_axis_ticks_with_llm(image_path, direction="y", chart_type_override=chart_type)

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
    print(f"[Info] X杞磋瘑鍒粨鏋? 绫诲瀷={result['x_axis_type']}, 鍒诲害鏁?{len(result['x_ticks'])}")
    print(f"[Info] Y杞磋瘑鍒粨鏋? 绫诲瀷={result['y_axis_type']}, 鍒诲害鏁?{len(result['y_ticks'])}")

    if not (llm_axis_result_is_valid(x_result) and llm_axis_result_is_valid(y_result)):
        result["api_failed"] = True
        result["cache_status"] = "invalid"
        result["failure_reason"] = f"x={result['x_status']};y={result['y_status']}"
        print(f"[Warning] LLM缁撴灉鏃犳晥锛屼笉鍐欏叆姝ｅ紡缂撳瓨: {result['failure_reason']}")
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
    
    parser = argparse.ArgumentParser(description='娴嬭瘯LLM鍒诲害鏍囩鎻愬彇')
    parser.add_argument('--image', type=str, required=True, help='鍥惧儚鏂囦欢璺緞')
    args = parser.parse_args()
    
    result = extract_tick_labels_with_llm(args.image)
    print("\n璇嗗埆缁撴灉:")
    print(f"X杞寸被鍨? {result['x_axis_type']}")
    print(f"X杞村埢搴? {result['x_ticks']}")
    print(f"Y杞寸被鍨? {result['y_axis_type']}")
    print(f"Y杞村埢搴? {result['y_ticks']}")
