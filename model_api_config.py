"""Compatibility import for project-wide model API configuration."""

from prediction_core.model_config import (
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
