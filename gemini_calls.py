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
import time
from typing import Any, Iterable

import aiohttp
import requests

from model_api_config import get_chat_completion_urls, get_headers, get_model_name


BASE_TIMEOUT = aiohttp.ClientTimeout(
    total=int(os.getenv("MLLM_TIMEOUT_SECONDS", "180")),
    connect=30,
    sock_connect=30,
    sock_read=120,
)
MAX_RETRIES = int(os.getenv("MLLM_MAX_RETRIES", "3"))
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
FAILURE_TEXT = "The model API request failed."


def _extract_response_text(result: dict[str, Any]) -> str:
    response_text = result.get("response")
    if isinstance(response_text, str):
        return response_text

    text = result.get("text")
    if isinstance(text, str):
        return text

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


def _normalize_urls(urls: Iterable[str] | None = None) -> list[str]:
    normalized = []
    for raw_url in list(urls or get_chat_completion_urls()):
        url = str(raw_url or "").strip().rstrip("/")
        if not url:
            continue
        if not url.endswith("/chat/completions"):
            url = f"{url}/chat/completions"
        normalized.append(url)
    return normalized or get_chat_completion_urls()


def _build_payload(
    messages: list,
    *,
    model: str | None = None,
    temperature: float | int | str | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model or os.getenv("MLLM_MODEL") or get_model_name(),
        "messages": messages,
        "temperature": float(
            os.getenv("MLLM_TEMPERATURE", "0.7") if temperature is None else temperature
        ),
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    if response_format is not None:
        payload["response_format"] = response_format
    if extra_payload:
        payload.update(extra_payload)
    return payload


async def chat_completion_request(
    messages: list,
    *,
    model: str | None = None,
    temperature: float | int | str | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    extra_payload: dict[str, Any] | None = None,
    urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: aiohttp.ClientTimeout | None = None,
    max_retries: int | None = None,
) -> dict[str, Any] | None:
    """Send one OpenAI-compatible chat-completions request with retries."""
    payload = _build_payload(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_payload=extra_payload,
    )
    request_urls = _normalize_urls(urls)
    attempts = int(max_retries or MAX_RETRIES)
    request_timeout = timeout or BASE_TIMEOUT

    async with aiohttp.ClientSession(timeout=request_timeout) as session:
        for attempt in range(1, attempts + 1):
            url = request_urls[(attempt - 1) % len(request_urls)]
            try:
                async with session.post(url, headers=headers or get_headers(), json=payload) as response:
                    text = await response.text()
                    if response.status in RETRYABLE_STATUS:
                        print(f"Retryable model API HTTP {response.status}: {text[:200]}")
                        await asyncio.sleep(min(2 * attempt, 20))
                        continue
                    if response.status != 200:
                        print(f"Model API HTTP {response.status}: {text[:200]}")
                        return None

                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        print(f"Model API returned non-JSON response: {text[:200]}")
                        return None

                    if _extract_response_text(result):
                        return result

                    print(f"Unexpected model API response: {text[:200]}")
                    await asyncio.sleep(min(2 * attempt, 20))
            except Exception as e:
                print(f"Model API attempt {attempt} failed: {e}")
                if attempt < attempts:
                    await asyncio.sleep(min(2 * attempt, 20))

    print("All model API attempts failed.")
    return None


async def chat_with_gemini(messages: list, **kwargs: Any) -> str:
    """Backward-compatible name for a unified chat-completions request."""
    result = await chat_completion_request(messages, **kwargs)
    if not result:
        return FAILURE_TEXT
    return _extract_response_text(result) or FAILURE_TEXT


def chat_completion_request_sync(
    messages: list,
    *,
    model: str | None = None,
    temperature: float | int | str | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    extra_payload: dict[str, Any] | None = None,
    urls: Iterable[str] | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: int | float | None = None,
    max_retries: int | None = None,
    retry_backoff_seconds: float | None = None,
) -> dict[str, Any] | None:
    """Synchronous variant for existing CV modules that are not async."""
    payload = _build_payload(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        extra_payload=extra_payload,
    )
    request_urls = _normalize_urls(urls)
    attempts = int(max_retries or MAX_RETRIES)
    request_timeout = timeout_seconds or int(os.getenv("MLLM_TIMEOUT_SECONDS", "180"))
    retry_backoff = float(retry_backoff_seconds or os.getenv("MLLM_RETRY_BACKOFF_SECONDS", "2"))

    for attempt in range(1, attempts + 1):
        url = request_urls[(attempt - 1) % len(request_urls)]
        try:
            response = requests.post(
                url,
                headers=headers or get_headers(),
                json=payload,
                timeout=request_timeout,
            )
            if response.status_code in RETRYABLE_STATUS:
                print(f"Retryable model API HTTP {response.status_code}: {response.text[:200]}")
                if attempt < attempts:
                    time.sleep(min(retry_backoff * attempt, 20))
                continue
            if response.status_code != 200:
                print(f"Model API HTTP {response.status_code}: {response.text[:200]}")
                return None
            try:
                result = response.json()
            except ValueError:
                print(f"Model API returned non-JSON response: {response.text[:200]}")
                return None
            if _extract_response_text(result):
                return result
            print(f"Unexpected model API response: {response.text[:200]}")
            if attempt < attempts:
                time.sleep(min(retry_backoff * attempt, 20))
        except Exception as e:
            print(f"Model API attempt {attempt} failed: {e}")
            if attempt < attempts:
                time.sleep(min(retry_backoff * attempt, 20))

    print("All model API attempts failed.")
    return None


def chat_with_gemini_sync(messages: list, **kwargs: Any) -> str:
    """Synchronous text-returning wrapper around the unified model API call."""
    result = chat_completion_request_sync(messages, **kwargs)
    if not result:
        return FAILURE_TEXT
    return _extract_response_text(result) or FAILURE_TEXT


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
