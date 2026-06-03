"""Backend bridge to the project-wide model API configuration."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction_core.model_config import (  # noqa: E402
    get_api_key,
    get_base_url,
    get_chat_completion_url,
    get_chat_completion_urls,
    get_headers,
    get_model_name,
    get_profile_name,
    use_legacy_pixtral,
)


__all__ = [
    "get_api_key",
    "get_base_url",
    "get_chat_completion_url",
    "get_chat_completion_urls",
    "get_headers",
    "get_model_name",
    "get_profile_name",
    "use_legacy_pixtral",
]
