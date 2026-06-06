"""Runtime knobs shared by chart prediction scripts."""

from __future__ import annotations

import os


def get_repeat_times(default: int = 3) -> int:
    raw = os.getenv("CHART_REPEAT_TIMES", str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"CHART_REPEAT_TIMES must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError("CHART_REPEAT_TIMES must be >= 1")
    return value
