from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def save_llm_error_response(response: Any, context: str) -> str | None:
    """Print and persist non-200 LLM responses for debugging provider rejects."""

    try:
        status_code = int(getattr(response, "status_code", 0) or 0)
    except (TypeError, ValueError):
        status_code = 0
    body = str(getattr(response, "text", "") or "")
    url = str(getattr(response, "url", "") or "")
    headers = dict(getattr(response, "headers", {}) or {})

    root = Path(
        os.getenv(
            "POLAR_LLM_ERROR_DIR",
            str(Path(__file__).resolve().parents[1] / "data" / "polar" / "llm_errors"),
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_context = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in context)[:80]
    path = root / f"{timestamp}_{safe_context}_http_{status_code}.json"

    payload = {
        "timestamp": timestamp,
        "context": context,
        "status_code": status_code,
        "url": url,
        "response_headers": headers,
        "response_body": body,
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to save LLM error response: {exc}")
        path = None

    print(f"LLM request failed in {context}: HTTP {status_code}")
    if path is not None:
        print(f"LLM response body saved to: {path}")
    print("LLM response body:")
    print(body if body else "<empty>")
    return str(path) if path is not None else None
