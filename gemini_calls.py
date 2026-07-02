# -*- coding: utf-8 -*-
"""
Compatibility helper for older code that imports gemini_calls.py.

The actual model endpoint, model name, and API key are managed by
model_api_config.py.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
import uuid
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
_MODAL_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "modal_call_context",
    default={},
)
_LAST_MODAL_CALL_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "last_modal_call_id",
    default=None,
)


def set_modal_call_context(context: dict[str, Any] | None) -> contextvars.Token:
    return _MODAL_CONTEXT.set(dict(context or {}))


def reset_modal_call_context(token: contextvars.Token) -> None:
    _MODAL_CONTEXT.reset(token)


def get_last_modal_call_id() -> str | None:
    return _LAST_MODAL_CALL_ID.get()


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


def _text_prompt_from_messages(messages: list) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
    return "\n".join(part for part in parts if part)


def _rough_token_count(text: str) -> int:
    if not text:
        return 0
    # Lightweight fallback when the OpenAI-compatible endpoint does not return usage.
    return max(1, (len(text) + 3) // 4)


def _usage(result: dict[str, Any] | None) -> dict[str, int | None]:
    if not isinstance(result, dict):
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    input_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("promptTokenCount")
    )
    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("candidatesTokenCount")
    )
    total_tokens = usage.get("total_tokens") or usage.get("totalTokenCount")
    return {
        "input_tokens": int(input_tokens) if isinstance(input_tokens, (int, float)) else None,
        "output_tokens": int(output_tokens) if isinstance(output_tokens, (int, float)) else None,
        "total_tokens": int(total_tokens) if isinstance(total_tokens, (int, float)) else None,
    }


def _write_modal_log(entry: dict[str, Any]) -> None:
    log_path = os.getenv("GT_MODAL_CALL_LOG_PATH", "").strip()
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Could not write modal call log: {exc}")


def _write_modal_full_log(entry: dict[str, Any]) -> None:
    log_path = os.getenv("GT_MODAL_FULL_LOG_PATH", "").strip()
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Could not write full modal call log: {exc}")


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
    modal_context: dict[str, Any] | None = None,
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
    text_prompt = _text_prompt_from_messages(messages)
    prompt_text_tokens = _rough_token_count(text_prompt)
    started_at = time.perf_counter()
    context = {**_MODAL_CONTEXT.get(), **(modal_context or {})}
    call_id = str(uuid.uuid4())
    _LAST_MODAL_CALL_ID.set(call_id)
    retry_events: list[dict[str, Any]] = []

    def write_logs(entry: dict[str, Any], *, result: dict[str, Any] | None = None, response_text: str | None = None) -> None:
        summary_entry = {
            **context,
            **entry,
            "call_id": call_id,
            "text_prompt_tokens": prompt_text_tokens,
            "model": payload.get("model"),
        }
        _write_modal_log(summary_entry)
        _write_modal_full_log(
            {
                **summary_entry,
                "request_payload": payload,
                "request_urls": request_urls,
                "retry_events": retry_events,
                "response_json": result,
                "response_text": response_text,
            }
        )

    async with aiohttp.ClientSession(timeout=request_timeout) as session:
        for attempt in range(1, attempts + 1):
            url = request_urls[(attempt - 1) % len(request_urls)]
            try:
                async with session.post(url, headers=headers or get_headers(), json=payload) as response:
                    text = await response.text()
                    if response.status in RETRYABLE_STATUS:
                        retry_events.append({"attempt": attempt, "url": url, "status": response.status, "text": text})
                        print(f"Retryable model API HTTP {response.status}: {text[:200]}")
                        await asyncio.sleep(min(2 * attempt, 20))
                        continue
                    if response.status != 200:
                        print(f"Model API HTTP {response.status}: {text[:200]}")
                        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
                        write_logs(
                            {
                                "attempts": attempt,
                                "input_tokens": None,
                                "output_tokens": None,
                                "total_tokens": None,
                                "request_duration_ms": duration_ms,
                                "url": url,
                                "success": False,
                                "error": f"HTTP {response.status}",
                                "response_status": response.status,
                            },
                            response_text=text,
                        )
                        return None

                    try:
                        result = json.loads(text)
                    except json.JSONDecodeError:
                        print(f"Model API returned non-JSON response: {text[:200]}")
                        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
                        write_logs(
                            {
                                "attempts": attempt,
                                "input_tokens": None,
                                "output_tokens": None,
                                "total_tokens": None,
                                "request_duration_ms": duration_ms,
                                "url": url,
                                "success": False,
                                "error": "non_json_response",
                                "response_status": response.status,
                            },
                            response_text=text,
                        )
                        return None

                    response_text = _extract_response_text(result)
                    if response_text:
                        duration_ms = round((time.perf_counter() - started_at) * 1000, 3)
                        usage = _usage(result)
                        write_logs(
                            {
                                "attempts": attempt,
                                "input_tokens": usage["input_tokens"],
                                "output_tokens": usage["output_tokens"],
                                "total_tokens": usage["total_tokens"],
                                "request_duration_ms": duration_ms,
                                "url": url,
                                "success": True,
                                "raw_prediction": response_text,
                            },
                            result=result,
                            response_text=response_text,
                        )
                        return result

                    print(f"Unexpected model API response: {text[:200]}")
                    retry_events.append({"attempt": attempt, "url": url, "status": response.status, "text": text})
                    await asyncio.sleep(min(2 * attempt, 20))
            except Exception as e:
                retry_events.append({"attempt": attempt, "url": url, "error": str(e)})
                print(f"Model API attempt {attempt} failed: {e}")
                if attempt < attempts:
                    await asyncio.sleep(min(2 * attempt, 20))

    print("All model API attempts failed.")
    write_logs(
        {
            "attempts": attempts,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "request_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "success": False,
        }
    )
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
    modal_context: dict[str, Any] | None = None,
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
    text_prompt = _text_prompt_from_messages(messages)
    prompt_text_tokens = _rough_token_count(text_prompt)
    started_at = time.perf_counter()
    context = {**_MODAL_CONTEXT.get(), **(modal_context or {})}
    call_id = str(uuid.uuid4())
    _LAST_MODAL_CALL_ID.set(call_id)
    retry_events: list[dict[str, Any]] = []

    def write_logs(entry: dict[str, Any], *, result: dict[str, Any] | None = None, response_text: str | None = None) -> None:
        summary_entry = {
            **context,
            **entry,
            "call_id": call_id,
            "text_prompt_tokens": prompt_text_tokens,
            "model": payload.get("model"),
        }
        _write_modal_log(summary_entry)
        _write_modal_full_log(
            {
                **summary_entry,
                "request_payload": payload,
                "request_urls": request_urls,
                "retry_events": retry_events,
                "response_json": result,
                "response_text": response_text,
            }
        )

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
                retry_events.append({"attempt": attempt, "url": url, "status": response.status_code, "text": response.text})
                print(f"Retryable model API HTTP {response.status_code}: {response.text[:200]}")
                if attempt < attempts:
                    time.sleep(min(retry_backoff * attempt, 20))
                continue
            if response.status_code != 200:
                print(f"Model API HTTP {response.status_code}: {response.text[:200]}")
                write_logs(
                    {
                        "attempts": attempt,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "request_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                        "url": url,
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "response_status": response.status_code,
                    },
                    response_text=response.text,
                )
                return None
            try:
                result = response.json()
            except ValueError:
                print(f"Model API returned non-JSON response: {response.text[:200]}")
                write_logs(
                    {
                        "attempts": attempt,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "request_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                        "url": url,
                        "success": False,
                        "error": "non_json_response",
                        "response_status": response.status_code,
                    },
                    response_text=response.text,
                )
                return None
            response_text = _extract_response_text(result)
            if response_text:
                usage = _usage(result)
                write_logs(
                    {
                        "attempts": attempt,
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "total_tokens": usage["total_tokens"],
                        "request_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                        "url": url,
                        "success": True,
                        "raw_prediction": response_text,
                    },
                    result=result,
                    response_text=response_text,
                )
                return result
            print(f"Unexpected model API response: {response.text[:200]}")
            retry_events.append({"attempt": attempt, "url": url, "status": response.status_code, "text": response.text})
            if attempt < attempts:
                time.sleep(min(retry_backoff * attempt, 20))
        except Exception as e:
            retry_events.append({"attempt": attempt, "url": url, "error": str(e)})
            print(f"Model API attempt {attempt} failed: {e}")
            if attempt < attempts:
                time.sleep(min(retry_backoff * attempt, 20))

    print("All model API attempts failed.")
    write_logs(
        {
            "attempts": attempts,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "request_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "success": False,
        }
    )
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
