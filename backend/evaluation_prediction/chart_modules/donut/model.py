"""Model calls and response normalization for donut charts."""

from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import sys
import threading
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from ..circular_model_config import get_chat_completion_urls, get_headers, get_model_name


LEGACY_URLS = [
    "http://localhost:8100/v1/chat/completions",
    "http://localhost:8101/v1/chat/completions",
    "http://localhost:8102/v1/chat/completions",
    "http://localhost:8103/v1/chat/completions",
    "http://localhost:8104/v1/chat/completions",
]

API_URLS = get_chat_completion_urls(LEGACY_URLS)
LLM_MODEL = get_model_name()
_api_index = random.randint(0, len(API_URLS) - 1)
_api_lock = threading.Lock()


def safe_print(*values: Any) -> None:
    text = " ".join(str(value) for value in values)
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


def get_next_api_url() -> str:
    global _api_index
    with _api_lock:
        url = API_URLS[_api_index]
        _api_index = (_api_index + 1) % len(API_URLS)
        return url


def normalize_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    return value


def _same_label(left: Any, right: Any) -> bool:
    return str(left or "").strip().casefold() == str(right or "").strip().casefold()


def _value_from_datapoints(datapoints: Any, item_name: str | None = None) -> Any:
    if not isinstance(datapoints, list):
        raise ValueError("Invalid 'datapoints' format")

    fallback_value = None
    for datapoint in datapoints:
        if not isinstance(datapoint, dict):
            continue

        if item_name:
            for key, value in datapoint.items():
                if _same_label(key, item_name):
                    return normalize_value(value)
            name = datapoint.get("name") or datapoint.get("label")
            if _same_label(name, item_name):
                if "percentage" in datapoint:
                    return normalize_value(datapoint["percentage"])
                if "value" in datapoint:
                    return normalize_value(datapoint["value"])
                if "start_angle" in datapoint and "end_angle" in datapoint:
                    return normalize_value(
                        {
                            "start_angle": datapoint["start_angle"],
                            "end_angle": datapoint["end_angle"],
                        }
                    )
            continue

        if "percentage" in datapoint or "value" in datapoint:
            return normalize_value(datapoint.get("percentage", datapoint.get("value")))
        if "start_angle" in datapoint and "end_angle" in datapoint:
            return normalize_value(
                {
                    "start_angle": datapoint["start_angle"],
                    "end_angle": datapoint["end_angle"],
                }
            )
        if datapoint:
            fallback_value = normalize_value(list(datapoint.values())[0])

    if fallback_value is not None and not item_name:
        return fallback_value
    raise ValueError(f"No datapoint found for target {item_name!r}")


async def call_llm_once(prompt: str, image_path: str, item_name: str | None = None) -> dict[str, float] | float | None:
    timeout = ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as sess:
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        ],
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.0,
            }

            current_url = get_next_api_url()
            safe_print(f"Using API URL: {current_url}")

            async with sess.post(current_url, json=payload, headers=get_headers()) as resp:
                res = await resp.json()
        except asyncio.TimeoutError:
            safe_print("Request timed out.")
            return None
        except Exception as exc:
            if "resp" in locals():
                try:
                    safe_print("JSON decode error", await resp.text())
                except Exception:
                    pass
            safe_print(f"Request failed: {exc}")
            return None

    safe_print(f"Full API Response: {json.dumps(res, indent=2, ensure_ascii=False)}")

    if "response" in res:
        txt = res["response"]
        safe_print(f"Extracted response from 'response' field: {txt}")
    elif "choices" in res:
        txt = res["choices"][0]["message"]["content"]
        safe_print(f"Extracted response from 'choices' field: {txt}")
    else:
        safe_print("API response missing expected fields", res)
        return None

    try:
        safe_print(f"Model Output Raw:\n{txt}\n")
        txt = re.sub(r"^```json", "", txt)
        txt = re.sub(r"```$", "", txt).strip()
        result = json.loads(txt)

        datapoints_key = "datapoints"
        if datapoints_key not in result:
            for key in result:
                if "datapoint" in key.lower():
                    datapoints_key = key
                    safe_print(f"Found misspelled key: {key}, using it as datapoints source")
                    break

        if datapoints_key in result and isinstance(result[datapoints_key], list):
            return _value_from_datapoints(result[datapoints_key], item_name)
        return result
    except Exception as exc:
        safe_print(f"Parse fail: {exc}\nFull text after cleaning:\n{txt}")
        return None


async def call_llm_with_retry(prompt: str, image_path: str, max_attempts: int = 10) -> Any:
    for _ in range(max_attempts):
        ans = await call_llm_once(prompt, image_path)
        if ans is not None:
            return ans
        await asyncio.sleep(6)
    return None
