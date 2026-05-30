"""Model calls and response normalization for pie charts."""

from __future__ import annotations

import asyncio
import base64
import random
from typing import Any

import aiohttp
from aiohttp import ClientTimeout

from prediction_core.model_config import get_chat_completion_urls, get_headers, get_model_name

from .json_parser import parse_model_output


LEGACY_URLS = [
    "http://localhost:8100/v1/chat/completions",
    "http://localhost:8101/v1/chat/completions",
    "http://localhost:8102/v1/chat/completions",
    "http://localhost:8103/v1/chat/completions",
    "http://localhost:8104/v1/chat/completions",
    "http://localhost:8105/v1/chat/completions",
    "http://localhost:8106/v1/chat/completions",
    "http://localhost:8107/v1/chat/completions",
    "http://localhost:8108/v1/chat/completions",
    "http://localhost:8109/v1/chat/completions",
    "http://localhost:8110/v1/chat/completions",
    "http://localhost:8111/v1/chat/completions",
    "http://localhost:8112/v1/chat/completions",
    "http://localhost:8113/v1/chat/completions",
    "http://localhost:8114/v1/chat/completions",
    "http://localhost:8115/v1/chat/completions",
]

URLS = get_chat_completion_urls(LEGACY_URLS)
LLM_MODEL = get_model_name()


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


async def call_llm_once(prompt: str, image_path: str) -> dict[str, float] | float | bool | None:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    timeout = ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as sess:
        try:
            selected_url = random.choice(URLS)
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                ],
                "max_tokens": 512,
            }

            async with sess.post(selected_url, headers=get_headers(), json=payload) as resp:
                res = await resp.json()
        except asyncio.TimeoutError:
            print("Request timed out.")
            return None
        except Exception as exc:
            if "resp" in locals():
                try:
                    print("JSON decode error", await resp.text())
                except Exception:
                    pass
            print(f"Request failed: {exc}")
            return None

    if "choices" in res:
        txt = res["choices"][0]["message"]["content"]
    elif "response" in res:
        txt = res["response"]
        if txt.startswith("```json") and txt.endswith("```"):
            txt = txt[7:-3].strip()
    else:
        print("API response missing expected fields", res)
        return None

    try:
        parsed = await parse_model_output(txt, prompt)
        if not parsed:
            raise ValueError("Failed to parse model output")

        if "datapoints" not in parsed:
            if "contains" in parsed and len(parsed) == 1:
                return bool(parsed["contains"])
            if "crosses_zero" in parsed and len(parsed) == 1:
                return bool(parsed["crosses_zero"])
            parsed["datapoints"] = []

        datapoints = parsed.get("datapoints", [])
        if not datapoints or not isinstance(datapoints[0], dict):
            raise ValueError("Invalid 'datapoints' format")

        value = normalize_value(list(datapoints[0].values())[0])
        if isinstance(value, dict):
            if "start_angle" in value and "end_angle" in value:
                return value
            raise ValueError("Missing start_angle or end_angle in dict")
        if isinstance(value, (float, int, bool)):
            return float(value)
        raise ValueError(f"Unsupported value type after normalization: {type(value)}")
    except Exception as exc:
        print(f"Parse fail: {exc}\nFull text:\n{txt}")
        return None


async def call_llm_with_retry(prompt: str, image_path: str, max_attempts: int = 10) -> Any:
    for _ in range(max_attempts):
        ans = await call_llm_once(prompt, image_path)
        if ans is not None:
            return ans
        await asyncio.sleep(6)
    return None
