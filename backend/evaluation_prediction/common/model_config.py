"""Project-wide OpenAI-compatible model API configuration.

This module is the backend-owned source of truth for model endpoint settings.
Use ``CHART_MODEL_PROFILE`` to switch profiles, or override individual fields
with ``CHART_BASE_URL``, ``CHART_MODEL_NAME``, and ``CHART_API_KEY``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


_GPT54_PROFILE = {
    "base_url": "http://dsiclab-model.ic.h3i.buaa.edu.cn/v1",
    "model_name": "gpt-5.4",
    "api_key": "",
}

_VVEAI_GEMINI_PROFILE = {
    "base_url": "https://api.vveai.com/v1",
    "model_name": "gemini-2.5-flash-lite",
    "api_key": "",
}

_WORK_PROFILE = {
    "base_url": "https://api.vveai.com/v1",
    "model_name": "gemini-2.5-flash-lite",
    "api_key": "",
}

MODEL_PROFILES = {
    "gpt54": _GPT54_PROFILE,
    "dsiclab_gpt54": _GPT54_PROFILE,
    "vveai_gpt41": {
        "base_url": "https://api.vveai.com/v1",
        "model_name": "gpt-4.1",
        "api_key": "",
    },
    "vveai_gemini": _VVEAI_GEMINI_PROFILE,
    "gemini": _VVEAI_GEMINI_PROFILE,
    "work": _WORK_PROFILE,
    "vveai_work": _WORK_PROFILE,
}

DEFAULT_PROFILE = "gemini"
LOCAL_SECRET_PATHS = (
    Path(__file__).with_name("model_secrets.local.json"),
    Path(__file__).resolve().parents[3] / "model_secrets.local.json",
)


def get_profile_name() -> str:
    return os.getenv("CHART_MODEL_PROFILE", DEFAULT_PROFILE).strip() or DEFAULT_PROFILE


def get_profile() -> dict[str, str]:
    return MODEL_PROFILES.get(get_profile_name(), MODEL_PROFILES[DEFAULT_PROFILE])


def _load_local_secrets() -> dict:
    for path in LOCAL_SECRET_PATHS:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _get_local_secret(field: str) -> str:
    secrets = _load_local_secrets()
    profiles = secrets.get("profiles", {})
    profile = profiles.get(get_profile_name(), {}) if isinstance(profiles, dict) else {}
    value = profile.get(field) if isinstance(profile, dict) else None
    if value:
        return str(value)
    value = secrets.get(field)
    return str(value) if value else ""


def get_base_url() -> str:
    return os.getenv("CHART_BASE_URL", get_profile()["base_url"]).rstrip("/")


def get_model_name() -> str:
    return os.getenv("CHART_MODEL_NAME", get_profile()["model_name"])


def get_api_key() -> str:
    return (
        os.getenv("CHART_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or _get_local_secret("api_key")
        or get_profile()["api_key"]
    )


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
