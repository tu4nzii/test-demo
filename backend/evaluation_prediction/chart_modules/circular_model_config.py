"""Model config for pie/donut extraction."""

from __future__ import annotations

import os
import re


DEFAULT_BASE_URL = "https://api.vveai.com/v1"
DEFAULT_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_API_KEYS = [
    "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    "sk-2nzrUYD0JWLFzopWF477111f78E746AbAcA9Ed8534C3A481",
    "sk-CiD5WVUNIkBeXDgYB46b90C06aD24636BcEaBaFa993970C4",
    "sk-WvF4fU10VeOkfFMq579610Fc01E8496d827d0d3e04C44d0a",
    "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb",
]


def get_model_name() -> str:
    return (
        os.getenv("CIRCULAR_CHART_MODEL_NAME")
        or os.getenv("COLOR_LLM_MODEL")
        or os.getenv("MLLM_MODEL")
        or DEFAULT_MODEL_NAME
    )


def get_chat_completion_urls(legacy_urls=None) -> list[str]:
    del legacy_urls
    raw_urls = os.getenv("CIRCULAR_CHART_API_URLS") or os.getenv("COLOR_LLM_API_URLS") or ""
    if raw_urls:
        return [_chat_url(item) for item in _split(raw_urls)]
    raw_base = (
        os.getenv("CIRCULAR_CHART_API_URL")
        or os.getenv("CIRCULAR_CHART_BASE_URL")
        or os.getenv("COLOR_LLM_API_URL")
        or os.getenv("COLOR_LLM_BASE_URL")
        or os.getenv("MLLM_API_URL")
        or os.getenv("MLLM_BASE_URL")
        or DEFAULT_BASE_URL
    )
    return [_chat_url(raw_base)]


def get_chat_completion_url() -> str:
    return get_chat_completion_urls()[0]


def get_headers() -> dict[str, str]:
    keys = _api_keys()
    headers = {"Content-Type": "application/json"}
    if keys:
        headers["Authorization"] = f"Bearer {keys[0]}"
    return headers


def _api_keys() -> list[str]:
    raw = (
        os.getenv("CIRCULAR_CHART_API_KEYS")
        or os.getenv("CIRCULAR_CHART_API_KEY")
        or os.getenv("COLOR_LLM_API_KEYS")
        or os.getenv("COLOR_LLM_API_KEY")
        or os.getenv("MLLM_API_KEYS")
        or os.getenv("MLLM_API_KEY")
        or ""
    )
    return _split(raw) or DEFAULT_API_KEYS


def _split(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;\s]+", raw or "") if item.strip()]


def _chat_url(raw_url: str) -> str:
    url = (raw_url or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"
