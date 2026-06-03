"""Central OpenAI-compatible model configuration for this project.

Set CHART_MODEL_PROFILE to switch all project MLLM calls together:
- dsiclab_gpt54: current default, using the BUAA endpoint and gpt-5.4.
- vveai_gpt41: original vveai endpoint with gpt-4.1.
- vveai_gemini: original vveai endpoint with gemini-2.5-pro.

Any profile value can still be overridden by CHART_BASE_URL, CHART_MODEL_NAME,
and CHART_API_KEY. Set CHART_USE_LEGACY_PIXTRAL=1 to keep using each script's
local Pixtral endpoint pool where those legacy pools exist.
"""

from __future__ import annotations

import os
from typing import Iterable


MODEL_PROFILES = {
    "dsiclab_gpt54": {
        "base_url": "http://dsiclab-model.ic.h3i.buaa.edu.cn/v1",
        "model_name": "gpt-5.4",
        "api_key": "sk-CbLDZcUuoj5NphrQfMQqh1ltqNBTkg85n7nMSFrsyxex5SOb",
    },
    "vveai_gpt41": {
        "base_url": "https://api.vveai.com/v1",
        "model_name": "gpt-4.1",
        "api_key": "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    },
    "vveai_gemini": {
        "base_url": "https://api.vveai.com/v1",
        "model_name": "gemini-2.5-pro",
        "api_key": "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    },
}

DEFAULT_PROFILE = "dsiclab_gpt54"


def get_profile_name() -> str:
    return os.getenv("CHART_MODEL_PROFILE", DEFAULT_PROFILE).strip() or DEFAULT_PROFILE


def get_profile() -> dict[str, str]:
    return MODEL_PROFILES.get(get_profile_name(), MODEL_PROFILES[DEFAULT_PROFILE])


def get_base_url() -> str:
    return os.getenv("CHART_BASE_URL", get_profile()["base_url"]).rstrip("/")


def get_model_name() -> str:
    return os.getenv("CHART_MODEL_NAME", get_profile()["model_name"])


def get_api_key() -> str:
    return os.getenv("CHART_API_KEY") or os.getenv("OPENAI_API_KEY") or get_profile()["api_key"]


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
