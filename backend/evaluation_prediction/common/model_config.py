"""Model configuration for backend-local chart value prediction.

This module deliberately delegates to ``prediction_core.model_config`` so the
backend prediction flow uses the same endpoint, model name, API key fallback,
and legacy Pixtral switches as the standalone evaluation flow.
"""

from __future__ import annotations

import sys
from typing import Iterable

from .paths import PROJECT_ROOT


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction_core.model_config import (  # noqa: E402
    get_api_key as _core_get_api_key,
    get_base_url as _core_get_base_url,
    get_chat_completion_url as _core_get_chat_completion_url,
    get_chat_completion_urls as _core_get_chat_completion_urls,
    get_headers as _core_get_headers,
    get_model_name as _core_get_model_name,
    use_legacy_pixtral as _core_use_legacy_pixtral,
)


def get_base_url() -> str:
    return _core_get_base_url()


def get_model_name() -> str:
    return _core_get_model_name()


def get_api_key() -> str:
    return _core_get_api_key()


def get_chat_completion_url() -> str:
    return _core_get_chat_completion_url()


def use_legacy_pixtral() -> bool:
    return _core_use_legacy_pixtral()


def get_chat_completion_urls(legacy_urls: Iterable[str] | None = None) -> list[str]:
    return _core_get_chat_completion_urls(legacy_urls)


def get_headers() -> dict[str, str]:
    return _core_get_headers()
