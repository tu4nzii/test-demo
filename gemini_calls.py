# -*- coding: utf-8 -*-
"""
Compatibility helper for older code that imports gemini_calls.py.

The actual model endpoint, model name, and API key are managed by
model_api_config.py.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiohttp

from model_api_config import get_chat_completion_urls, get_headers, get_model_name


BASE_TIMEOUT = aiohttp.ClientTimeout(
    total=int(os.getenv("MLLM_TIMEOUT_SECONDS", "180")),
    connect=30,
    sock_connect=30,
    sock_read=120,
)
MAX_RETRIES = int(os.getenv("MLLM_MAX_RETRIES", "3"))


def _extract_response_text(result: dict[str, Any]) -> str:
    if "choices" in result and result["choices"]:
        message = result["choices"][0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content)

    if "candidates" in result and result["candidates"]:
        parts = result["candidates"][0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))

    return ""


async def chat_with_gemini(messages: list) -> str:
    """Backward-compatible name for a unified chat-completions request."""
    payload = {
        "model": os.getenv("MLLM_MODEL") or get_model_name(),
        "messages": messages,
        "temperature": float(os.getenv("MLLM_TEMPERATURE", "0.7")),
    }
    urls = get_chat_completion_urls()
    retryable_status = {429, 500, 502, 503, 504}

    async with aiohttp.ClientSession(timeout=BASE_TIMEOUT) as session:
        for attempt in range(1, MAX_RETRIES + 1):
            url = urls[(attempt - 1) % len(urls)]
            try:
                async with session.post(url, headers=get_headers(), json=payload) as response:
                    text = await response.text()
                    if response.status in retryable_status:
                        print(f"Retryable model API HTTP {response.status}: {text[:200]}")
                        await asyncio.sleep(min(2 * attempt, 20))
                        continue
                    if response.status != 200:
                        print(f"Model API HTTP {response.status}: {text[:200]}")
                        return "The model API request failed."

                    result = json.loads(text)
                    content = _extract_response_text(result)
                    if content:
                        return content

                    print(f"Unexpected model API response: {text[:200]}")
                    await asyncio.sleep(min(2 * attempt, 20))
            except Exception as e:
                print(f"Model API attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(min(2 * attempt, 20))

    print("All model API attempts failed.")
    return "The model API request failed."


async def run_chat():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer clearly and concisely.",
        }
    ]

    while True:
        try:
            user_input = input("User: ").strip()
            if user_input.lower() in {"quit", "exit"}:
                break
            if user_input.lower() == "clear":
                messages = messages[:1]
                print("Conversation cleared.")
                continue
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            response = await chat_with_gemini(messages)
            print(f"AI: {response}")
            messages.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            break


async def test_all_keys():
    response = await chat_with_gemini([{"role": "user", "content": "ping"}])
    print(response)


if __name__ == "__main__":
    asyncio.run(test_all_keys())
