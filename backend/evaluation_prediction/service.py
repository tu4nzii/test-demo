"""Service helpers for running backend-generated prediction flows."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


SUPPORTED_PREDICTION_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}
SUPPORTED_BAR_TYPES = {"v_bar", "h_bar"}


async def run_bar_prediction_async(
    chart_type: str,
    config_json_path: str | Path,
    *,
    batch_size: int | None = 1,
) -> list[dict[str, Any]]:
    if chart_type == "v_bar":
        from .chart_modules.v_bar.runner import run_experiment
    elif chart_type == "h_bar":
        from .chart_modules.h_bar.runner import run_experiment
    else:
        raise ValueError(f"Unsupported bar prediction chart type: {chart_type}")

    return await run_experiment(
        batch_size=batch_size,
        config_paths=[str(config_json_path)],
    )


async def run_prediction_async(
    chart_type: str,
    config_json_path: str | Path,
    *,
    batch_size: int | None = 1,
) -> list[dict[str, Any]]:
    if chart_type == "v_bar":
        from .chart_modules.v_bar.runner import run_experiment
    elif chart_type == "h_bar":
        from .chart_modules.h_bar.runner import run_experiment
    elif chart_type == "line":
        from .chart_modules.line.runner import run_experiment
    elif chart_type == "scatter":
        from .chart_modules.scatter.data import PointChartConfig
        from .chart_modules.scatter.runner import run_experiment

        return await run_experiment(
            PointChartConfig(chart_type="scatter", result_dir_name="scatter", mark_name="circle"),
            batch_size=batch_size,
            config_paths=[str(config_json_path)],
        )
    elif chart_type == "bubble":
        from .chart_modules.bubble.data import PointChartConfig
        from .chart_modules.bubble.runner import run_experiment

        return await run_experiment(
            PointChartConfig(chart_type="bubble", result_dir_name="bubble", mark_name="bubble"),
            batch_size=batch_size,
            config_paths=[str(config_json_path)],
        )
    else:
        raise ValueError(f"Unsupported prediction chart type: {chart_type}")

    return await run_experiment(
        batch_size=batch_size,
        config_paths=[str(config_json_path)],
    )


def run_bar_prediction(
    chart_type: str,
    config_json_path: str | Path,
    *,
    batch_size: int | None = 1,
) -> list[dict[str, Any]]:
    return asyncio.run(run_bar_prediction_async(chart_type, config_json_path, batch_size=batch_size))


def run_prediction(
    chart_type: str,
    config_json_path: str | Path,
    *,
    batch_size: int | None = 1,
) -> list[dict[str, Any]]:
    return asyncio.run(run_prediction_async(chart_type, config_json_path, batch_size=batch_size))
