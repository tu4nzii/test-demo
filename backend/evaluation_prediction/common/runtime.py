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


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def get_amplifier_rounds(default: int = 3) -> int:
    """Number of chained local-refinement rounds for chart readers."""
    return _positive_int_from_env("CHART_AMPLIFIER_ROUNDS", default)


def get_bar_amplifier_rounds(default: int = 3) -> int:
    """Number of chained amplifier refinement rounds for h_bar/v_bar."""
    raw = os.getenv("CHART_BAR_AMPLIFIER_ROUNDS")
    if raw is not None:
        return _positive_int_from_env("CHART_BAR_AMPLIFIER_ROUNDS", default)
    return get_amplifier_rounds(default)
