"""Circular-chart model config, delegated to the project-wide config."""

from __future__ import annotations

from typing import Iterable

from model_api_config import (
    get_api_key,
    get_base_url,
    get_chat_completion_url,
    get_chat_completion_urls,
    get_headers,
    get_model_name,
)


def get_circular_chat_completion_urls(legacy_urls: Iterable[str] | None = None) -> list[str]:
    return get_chat_completion_urls(legacy_urls)
