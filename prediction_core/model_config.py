"""Central model configuration for chart value prediction.

Defaults point at the temporary OpenAI-compatible endpoint requested for the
current experiments. Set CHART_USE_LEGACY_PIXTRAL=1 to keep using each script's
local Pixtral endpoint pool.
"""

from __future__ import annotations

import os
from typing import Iterable


DEFAULT_BASE_URL = "http://dsiclab-model.ic.h3i.buaa.edu.cn/v1"
DEFAULT_MODEL_NAME = "gpt-5.3-codex"
DEFAULT_API_KEY = "sk-CbLDZcUuoj5NphrQfMQqh1ltqNBTkg85n7nMSFrsyxex5SOb"


def get_base_url() -> str:
    return os.getenv("CHART_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def get_model_name() -> str:
    return os.getenv("CHART_MODEL_NAME", DEFAULT_MODEL_NAME)


def get_api_key() -> str:
    return os.getenv("CHART_API_KEY") or os.getenv("OPENAI_API_KEY", DEFAULT_API_KEY)


def get_chat_completion_url() -> str:
    return f"{get_base_url()}/chat/completions"


def _split_urls(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def use_legacy_pixtral() -> bool:
    return os.getenv("CHART_USE_LEGACY_PIXTRAL", "").lower() in {"1", "true", "yes", "on"}


def get_chat_completion_urls(legacy_urls: Iterable[str] | None = None) -> list[str]:
    env_urls = os.getenv("CHART_API_URLS", "")
    if env_urls:
        return _split_urls(env_urls)
    legacy = list(legacy_urls or [])
    if use_legacy_pixtral() and legacy:
        return legacy
    return [get_chat_completion_url()]


def get_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = get_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
