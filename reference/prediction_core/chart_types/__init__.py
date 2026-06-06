"""Chart type specifications grouped by coordinate system."""

from __future__ import annotations

from .cartesian import CARTESIAN_SPECS
from .polar import POLAR_SPECS


CHART_SPECS = {**CARTESIAN_SPECS, **POLAR_SPECS}


def get_spec(chart_type: str):
    key = chart_type.strip().lower()
    if key not in CHART_SPECS:
        valid = ", ".join(sorted(CHART_SPECS))
        raise KeyError(f"Unknown chart type: {chart_type!r}. Valid types: {valid}")
    return CHART_SPECS[key]
