"""Model calls and response normalization for pie charts."""

from __future__ import annotations

import asyncio
import base64
import random
import re
import sys
from typing import Any

from aiohttp import ClientTimeout

from gemini_calls import FAILURE_TEXT, chat_with_gemini

from ..circular_model_config import get_chat_completion_urls, get_model_name

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


def safe_print(*values: Any) -> None:
    text = " ".join(str(value) for value in values)
    try:
        sys.stdout.write(text + "\n")
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + "\n")


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


def _extract_json_object(text: str) -> str:
    cleaned = re.sub(r"^```json", "", text.strip())
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned


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


async def call_llm_once(prompt: str, image_path: str, item_name: str | None = None) -> dict[str, float] | float | bool | None:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    timeout = ClientTimeout(total=300)
    selected_url = random.choice(URLS)
    request_urls = [selected_url] + [url for url in URLS if url != selected_url]
    txt = await chat_with_gemini(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        model=LLM_MODEL,
        max_tokens=2048,
        temperature=0.0,
        urls=request_urls,
        timeout=timeout,
    )
    if txt == FAILURE_TEXT:
        safe_print("Request failed: model API request failed.")
        return None
    if txt.startswith("```json") and txt.endswith("```"):
        txt = txt[7:-3].strip()

    try:
        txt = _extract_json_object(txt)
        parsed = await parse_model_output(txt, prompt)
        if not parsed:
            raise ValueError("Failed to parse model output")

        if "datapoints" not in parsed:
            if "contains" in parsed and len(parsed) == 1:
                return bool(parsed["contains"])
            if "crosses_zero" in parsed and len(parsed) == 1:
                return bool(parsed["crosses_zero"])
            parsed["datapoints"] = []

        value = _value_from_datapoints(parsed.get("datapoints", []), item_name)
        if isinstance(value, dict):
            if "start_angle" in value and "end_angle" in value:
                return value
            raise ValueError("Missing start_angle or end_angle in dict")
        if isinstance(value, (float, int, bool)):
            return float(value)
        raise ValueError(f"Unsupported value type after normalization: {type(value)}")
    except Exception as exc:
        safe_print(f"Parse fail: {exc}\nFull text:\n{txt}")
        return None


async def call_llm_with_retry(prompt: str, image_path: str, max_attempts: int = 10) -> Any:
    for _ in range(max_attempts):
        ans = await call_llm_once(prompt, image_path)
        if ans is not None:
            return ans
        await asyncio.sleep(6)
    return None
