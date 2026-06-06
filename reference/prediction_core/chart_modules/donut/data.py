"""Data loading for donut charts."""

from __future__ import annotations

from reference.prediction_core.chart_io import load_json_configs


def load_chart_configs() -> list[dict]:
    return load_json_configs("chart_configs")
