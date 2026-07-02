"""Service helpers for running backend-generated prediction flows."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any


SUPPORTED_PREDICTION_TYPES = {
    "v_bar",
    "h_bar",
    "line",
    "scatter",
    "bubble",
    "pie",
    "donut",
    "radar",
    "rose",
}
SUPPORTED_BAR_TYPES = {"v_bar", "h_bar"}


def sync_results_root_from_env() -> None:
    """Keep already-imported runners aligned with the experiment output root."""
    raw = os.getenv("EVALUATION_PREDICTION_RESULTS_ROOT")
    if not raw:
        return
    root = Path(raw).expanduser().resolve()

    from .common import paths as common_paths

    common_paths.RESULTS_ROOT = root
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("backend.evaluation_prediction") and not module_name.startswith("evaluation_prediction"):
            continue
        if hasattr(module, "RESULTS_ROOT"):
            setattr(module, "RESULTS_ROOT", root)
        if module_name.endswith(".chart_modules.line.visual") and hasattr(module, "RESULT_ROOT"):
            setattr(module, "RESULT_ROOT", root / "line")


def normalize_prediction_type(chart_type: str) -> str:
    return str(chart_type or "").lower()


async def run_bar_prediction_async(
    chart_type: str,
    config_json_path: str | Path,
    *,
    batch_size: int | None = 1,
) -> list[dict[str, Any]]:
    sync_results_root_from_env()
    chart_type = normalize_prediction_type(chart_type)
    if chart_type == "v_bar":
        from .chart_modules.v_bar.runner import run_experiment

        return await run_experiment(
            batch_size=batch_size,
            config_paths=[str(config_json_path)],
            chart_type=chart_type,
        )
    elif chart_type == "h_bar":
        from .chart_modules.h_bar.runner import run_experiment

        return await run_experiment(
            batch_size=batch_size,
            config_paths=[str(config_json_path)],
            chart_type=chart_type,
        )
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
    sync_results_root_from_env()
    chart_type = normalize_prediction_type(chart_type)
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
    elif chart_type == "pie":
        from .chart_modules.pie.runner import run_experiment
    elif chart_type == "donut":
        from .chart_modules.donut.runner import run_experiment
    elif chart_type == "radar":
        from .chart_modules.radar.runner import run_experiment
    elif chart_type == "rose":
        from .chart_modules.rose.runner import run_experiment
    else:
        raise ValueError(f"Unsupported prediction chart type: {chart_type}")

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "config_paths": [str(config_json_path)],
    }
    if chart_type in {"v_bar", "h_bar"}:
        kwargs["chart_type"] = chart_type
    return await run_experiment(**kwargs)


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
