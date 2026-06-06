"""Async model helpers for radar charts."""

from __future__ import annotations

import asyncio
import base64
import os
import random
from typing import Tuple

import aiohttp
import cv2

from reference.prediction_core.json_utils import extract_json_response, validate_xy
from reference.prediction_core.model_config import get_chat_completion_urls, get_headers, get_model_name


LEGACY_URLS = ["http://localhost:8508/v1/chat/completions"]
api_urls = get_chat_completion_urls(LEGACY_URLS)
headers = get_headers()
llm_model = get_model_name()


def encode_cv2_to_base64(image):
    retval, buffer = cv2.imencode(".png", image)
    if not retval:
        return None
    return base64.b64encode(buffer).decode("utf-8")


def read_file_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


async def call_llm_response_async(
    session: aiohttp.ClientSession,
    prompt: str,
    image_b64: str,
    item_name: str,
    model_name=llm_model,
) -> Tuple[float | None, float | None]:
    if not image_b64:
        return (None, None)

    max_retries = 15
    base_delay = 1

    for attempt in range(max_retries):
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                "max_tokens": 512,
            }

            selected_url = random.choice(api_urls)
            async with session.post(selected_url, json=payload, headers=headers, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                    elif "response" in result:
                        content = result["response"]
                    else:
                        print(f"Unexpected response format: {result}")
                        continue

                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]

                    coords_json = extract_json_response(content)
                    if isinstance(coords_json, dict) and "datapoints" in coords_json:
                        if isinstance(coords_json["datapoints"], list):
                            for item in coords_json["datapoints"]:
                                if not isinstance(item, dict):
                                    continue
                                if item_name in item:
                                    coords = item[item_name]
                                    if validate_xy(coords):
                                        return tuple(coords)
                    return (None, None)

                if response.status in {429} or response.status >= 500:
                    await asyncio.sleep(base_delay * (2 ** attempt))
                else:
                    await asyncio.sleep(base_delay)
        except Exception:
            await asyncio.sleep(base_delay * (2 ** attempt))

    return (None, None)


async def check_find_point_async(session: aiohttp.ClientSession, image_b64: str, color: str) -> str:
    max_retries = 15
    retry_delay = 1
    prompt = f"判断图片中是否包含颜色为{color}的点,记住是一个圆点，若包含，返回True，不包含，返回False,并说出原因"

    for _ in range(max_retries):
        try:
            payload = {
                "model": llm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                "max_tokens": 512,
            }

            selected_url = random.choice(api_urls)
            async with session.post(selected_url, json=payload, headers=headers, timeout=60) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                    elif "response" in result:
                        content = result["response"]
                    else:
                        print(f"Unexpected response format: {result}")
                        continue
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    return "True" if "True" in content else "False"
                await asyncio.sleep(retry_delay)
        except Exception:
            await asyncio.sleep(retry_delay)

    return "False"
