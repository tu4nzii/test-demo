# -*- coding: utf-8 -*-
"""Extract chart tick labels with the configured MLLM."""

import os
import sys
import base64
import re
import json
import hashlib
import math
from datetime import datetime
from typing import Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gemini_calls import FAILURE_TEXT, chat_with_gemini_sync
from model_api_config import get_model_name

LLM_MODEL = os.getenv("TICK_LLM_MODEL") or os.getenv("MLLM_MODEL") or get_model_name()
LLM_TEMPERATURE = float(os.getenv("TICK_LLM_TEMPERATURE", "0"))
LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("TICK_LLM_TIMEOUT_SECONDS", "180"))
LLM_MAX_ATTEMPTS = int(os.getenv("TICK_LLM_MAX_ATTEMPTS", "8"))
LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("TICK_LLM_RETRY_BACKOFF_SECONDS", "2"))
TICK_CACHE_SCHEMA_VERSION = "tick-mllm-v16"
TICK_SYSTEM_PROMPT = (
    "You are a precise chart-reading assistant. Extract only visible axis tick labels, "
    "preserve their order, and do not infer data values or legend text as ticks."
)


def chat_with_gemini(messages: list) -> Optional[str]:
    """Call the project-wide Gemini/OpenAI-compatible MLLM helper."""
    content = chat_with_gemini_sync(
        messages,
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        timeout_seconds=LLM_REQUEST_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_ATTEMPTS,
        retry_backoff_seconds=LLM_RETRY_BACKOFF_SECONDS,
    )
    return None if content == FAILURE_TEXT else content


def get_cache_file_path(image_path: str, cache_dir: str) -> str:
    """Build a cache file path from image identity and prompt metadata."""
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
    """Load cached MLLM tick extraction result."""
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        print(f"[Warning] Failed to read cache file: {e}")
        return None


def save_llm_cache(cache_file: str, result: Dict, image_path: str, x_response: str = "", y_response: str = "") -> None:
    """Save LLM recognition result to cache."""
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        cache_data = {
            "image_path": image_path,
            "x_ticks": result.get("x_ticks", []),
            "y_ticks": result.get("y_ticks", []),
            "x_axis_type": result.get("x_axis_type", "unknown"),
            "y_axis_type": result.get("y_axis_type", "unknown"),
            "x_llm_response": x_response,
            "y_llm_response": y_response,
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Info] Cached LLM recognition result: {cache_file}")
    except Exception as e:
        print(f"[Warning] Failed to save cache file: {e}")


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file as a base64 string."""
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
    if chart_type in {"scatter", "bubble"}:
        return encode_image_to_base64(image_path)
    if direction.lower() == "x" and chart_type == "h_bar":
        crop = image
    elif direction.lower() == "x":
        crop = image[int(h * 0.52):h, 0:w]
    elif chart_type == "h_bar":
        crop = image[0:int(h * 0.82), 0:int(w * 0.68)]
    else:
        left = image[0:h, 0:int(w * 0.42)]
        right = image[0:h, int(w * 0.58):w]
        gap = np.full((h, max(8, int(w * 0.02)), 3), 255, dtype=np.uint8)
        crop = np.concatenate([left, gap, right], axis=1)
    ok, buffer = cv2.imencode(".png", crop)
    if not ok:
        return encode_image_to_base64(image_path)
    return base64.b64encode(buffer).decode("utf-8")


def build_tick_extraction_prompt(direction: str = "x", chart_type: str = "") -> str:
    axis_name = "X axis" if direction.lower() == "x" else "Y axis"
    order = "left to right" if direction.lower() == "x" else "bottom to top"
    chart_type = (chart_type or "").lower()
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
    point_chart_rules = ""
    if chart_type in {"scatter", "bubble"}:
        point_chart_rules = """
Point-chart rules:
- Read ticks from the main plotting area's coordinate axes only.
- Ignore color bars, size legends, series legends, captions, source notes, and their tick labels.
- If a color scale or legend appears below the plot, do not treat it as the X axis.
- Include the full visible range of the main axis, including endpoint ticks at the left/right or bottom/top of the plot.
"""
    line_chart_rules = ""
    if chart_type == "line":
        line_chart_rules = """
Line-chart rules:
- Read only labels printed on the outer axes or plot-frame tick positions.
- Ignore numeric labels printed next to data points, curve annotations, callouts, tooltips, and legend entries.
- Ignore right-side series labels unless they are aligned as a numeric axis scale with multiple regular tick labels.
- If the x-axis is a time/category axis such as months, dates, or years, return those visible axis labels as text in order.
- If dense minor ticks are present but only a few tick labels are printed, return only the printed tick labels.
"""
    bar_chart_rules = ""
    if _bar_base_type(chart_type) in {"v_bar", "h_bar"}:
        bar_chart_rules = """
Bar-chart category-axis rules:
- For grouped or multi-level category axes, read only the primary tick labels directly aligned with bars or bar groups.
- Ignore secondary grouping labels such as years printed below quarters, cohort headers, panel labels, axis titles, and legends.
- If labels such as Q1 Q2 Q3 Q4 repeat under multiple years, return only the repeated Q labels in their visible order, not the year labels.
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
4. For Y axis labels, inspect both the left and right sides of the plot; some charts place the numeric Y-axis labels on the right.
5. Preserve text labels exactly. For numeric labels, keep signs, decimals, commas, and percentages if shown.
6. If labels are partially occluded, return only labels that are readable.
7. Decide axis_type by the role of the axis, not only by whether labels can be parsed as numbers.
8. Use "numeric" only for a continuous quantitative scale where inserted intermediate tick values would be meaningful on the chart.
9. Use "text" for discrete categories, names, IDs, dates, months, quarters, or observation periods. Calendar years on a line/bar chart are usually time-point labels, so classify them as text unless the axis is clearly a continuous numeric scale.
10. Do not convert time labels or category labels into numbers; preserve them as strings.
11. Return JSON only, with no Markdown and no explanation.
{point_chart_rules}
{line_chart_rules}
{bar_chart_rules}
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
        "unable to respond",
        "all attempts failed",
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
            print(f"[Info] Ignore invalid LLM cache ({quality['reason']}): {cache_file}")
            return None
        return cache_data
    except Exception as e:
        print(f"[Warning] Failed to read LLM cache: {e}")
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
        print(f"[Info] Cached LLM recognition result: {cache_file}")
    except Exception as e:
        print(f"[Warning] Failed to save LLM cache: {e}")


def extract_axis_ticks_with_llm(image_path: str, direction: str = 'x', chart_type_override: str = "") -> Dict:
    """Extract axis tick labels and axis type with the configured MLLM."""
    try:
        chart_type = (chart_type_override or os.path.basename(os.path.dirname(image_path))).lower()
        image_base64 = encode_axis_crop_to_base64(image_path, direction, chart_type=chart_type)
        prompt = build_tick_extraction_prompt(direction, chart_type)
        
        prompt = build_tick_extraction_prompt(direction, chart_type)
        
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
        
        response = chat_with_gemini(messages)
        if not response:
            return {
                "axis_type": "unknown",
                "ticks": [],
                "raw_response": "",
                "status": "api_failed",
            }

        result = parse_llm_response(response, direction)
        if _bar_base_type(chart_type) == "h_bar" and direction.lower() == "y" and result.get("ticks"):
            result["ticks"] = [_clean_hbar_category_tick(tick) for tick in reversed(result["ticks"])]
        else:
            numeric_ticks = _finite_numeric_values(result.get("ticks", []))
            if direction.lower() == "y" and numeric_ticks and len(numeric_ticks) >= 2:
                if numeric_ticks[0] > numeric_ticks[-1]:
                    result["ticks"] = list(reversed(result["ticks"]))
        result["raw_response"] = response
        result["status"] = "ok"
        return result
        
    except Exception as e:
        print(f"[Error] LLM failed to extract {direction}-axis ticks: {e}")
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
    if cache_dir:
        cache_file = get_cache_file_path(image_path, cache_dir)
        cached_result = load_llm_cache(cache_file)
        if cached_result:
            print(f"[Info] Loaded LLM tick result from cache: {cache_file}")
            return {
                "x_ticks": cached_result.get("x_ticks", []),
                "y_ticks": cached_result.get("y_ticks", []),
                "x_axis_type": cached_result.get("x_axis_type", "unknown"),
                "y_axis_type": cached_result.get("y_axis_type", "unknown")
            }

    if not allow_api:
        return {
            "x_ticks": [],
            "y_ticks": [],
            "x_axis_type": "unknown",
            "y_axis_type": "unknown",
            "cache_miss": True,
        }

    print(f"[Info] Start LLM tick extraction: {image_path}")
    
    x_result = extract_axis_ticks_with_llm(image_path, direction='x')
    y_result = extract_axis_ticks_with_llm(image_path, direction='y')
    
    x_response = x_result.get("raw_response", "")
    y_response = y_result.get("raw_response", "")
    
    result = {
        "x_ticks": x_result.get("ticks", []),
        "y_ticks": y_result.get("ticks", []),
        "x_axis_type": x_result.get("axis_type", "unknown"),
        "y_axis_type": y_result.get("axis_type", "unknown")
    }
    
    print(f"[Info] X-axis tick result: type={result['x_axis_type']}, count={len(result['x_ticks'])}")
    print(f"[Info] Y-axis tick result: type={result['y_axis_type']}, count={len(result['y_ticks'])}")
    
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
            print(f"[Info] Loaded LLM tick result from cache: {cache_file}")
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

    print(f"[Info] Start LLM tick extraction: {image_path}")
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
    print(f"[Info] X-axis tick result: type={result['x_axis_type']}, count={len(result['x_ticks'])}")
    print(f"[Info] Y-axis tick result: type={result['y_axis_type']}, count={len(result['y_ticks'])}")

    if not (llm_axis_result_is_valid(x_result) and llm_axis_result_is_valid(y_result)):
        result["api_failed"] = True
        result["cache_status"] = "invalid"
        result["failure_reason"] = f"x={result['x_status']};y={result['y_status']}"
        print(f"[Warning] Invalid LLM result; skip writing formal cache: {result['failure_reason']}")
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


def _bar_value_cache_path(image_path: str, cache_dir: str, chart_type: str) -> str:
    abs_path = os.path.abspath(image_path)
    try:
        image_hash = sha256_file(abs_path)
    except OSError:
        image_hash = hashlib.sha256(abs_path.encode("utf-8")).hexdigest()
    payload = {
        "schema": "bar-value-labels-v1",
        "chart_type": chart_type,
        "image_sha256": image_hash,
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
    }
    return os.path.join(cache_dir, f"{stable_json_hash(payload)}.json")


def _parse_bar_value_response(response_text: str) -> List[float]:
    text = (response_text or "").strip()
    if not text:
        return []
    values = []
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        payload = json.loads(match.group(0) if match else text)
        raw_values = payload.get("values", [])
        if isinstance(raw_values, list):
            if raw_values and all(isinstance(item, dict) for item in raw_values):
                raw_values = sorted(
                    raw_values,
                    key=lambda item: float(item.get("position", item.get("index", 1e9))),
                )
                raw_values = [
                    item.get("value", item.get("label", item.get("text", "")))
                    for item in raw_values
                ]
            for item in raw_values:
                match_value = re.search(
                    r"[-+]?\d[\d,]*(?:\.\d+)?",
                    str(item).replace("\u2212", "-"),
                )
                if match_value:
                    values.append(float(match_value.group(0).replace(",", "")))
    except Exception:
        pass
    if values:
        return values

    for match_value in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", text.replace("\u2212", "-")):
        try:
            values.append(float(match_value.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def extract_bar_value_labels_with_llm(
    image_path: str,
    cache_dir: Optional[str] = None,
    allow_api: bool = True,
    chart_type_override: str = "",
) -> Dict:
    """Read numeric data labels printed at bar ends when an axis has no ticks."""
    chart_type = (chart_type_override or os.path.basename(os.path.dirname(image_path))).lower()
    cache_file = None
    if cache_dir:
        cache_file = _bar_value_cache_path(image_path, cache_dir, chart_type)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached.get("values"), list):
                    return {"values": cached.get("values", []), "cache_hit": True}
            except Exception:
                pass

    if not allow_api:
        return {"values": [], "cache_hit": False, "cache_miss": True}

    order = "left to right" if _bar_base_type(chart_type) == "v_bar" else "top to bottom"
    prompt = f"""
Read only numeric data labels printed on or immediately beside bar ends in this bar chart.

Return strict JSON only:
{{
  "values": [
    {{"position": 0, "value": "first visible bar-end numeric label"}},
    {{"position": 1, "value": "next visible bar-end numeric label"}}
  ]
}}

Rules:
1. Order values {order}.
2. Use only labels that directly annotate bar lengths/heights, such as labels above vertical bars or at the end of horizontal bars.
3. Do not read axis tick labels, category labels, legend text, titles, source notes, or percentages in captions.
4. Preserve the numeric text, but units such as m, %, or kg may remain attached.
5. If there are no bar-end numeric labels, return {{"values": []}}.
"""
    messages = [
        {"role": "system", "content": "You are a precise chart-reading assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(image_path)}"},
                },
            ],
        },
    ]
    response = chat_with_gemini(messages)
    values = _parse_bar_value_response(response or "")
    result = {"values": values, "raw_response": response or "", "cache_hit": False}
    if cache_file:
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save bar value cache: {e}")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test LLM tick label extraction')
    parser.add_argument('--image', type=str, required=True, help='Image file path')
    args = parser.parse_args()
    
    result = extract_tick_labels_with_llm(args.image)
    print("\nRecognition result:")
    print(f"X-axis type: {result['x_axis_type']}")
    print(f"X-axis ticks: {result['x_ticks']}")
    print(f"Y-axis type: {result['y_axis_type']}")
    print(f"Y-axis ticks: {result['y_ticks']}")
