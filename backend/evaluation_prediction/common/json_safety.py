"""JSON safety helpers for experiment artifacts and API payloads."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def sanitize_json_value(value: Any) -> Any:
    """Return a JSON-compliant copy with NaN/Infinity converted to null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    return value


def safe_read_text(path: Path) -> str:
    """Read UTF-8 text while tolerating a Windows BOM."""
    return path.read_text(encoding="utf-8-sig")
