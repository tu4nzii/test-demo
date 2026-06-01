"""Evaluation entry points for scatter charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.evaluation_utils import save_xy_results, xy_mae, xy_relative_error


def compute_mae(pred: tuple[Any, Any], gt: tuple[Any, Any]) -> float | None:
    return xy_mae(pred, gt)


def compute_relative_error(pred: tuple[Any, Any], gt: tuple[Any, Any]) -> tuple[float | None, float | None]:
    return xy_relative_error(pred, gt)


def save_results(records: list[dict[str, Any]], result_dir: Path) -> None:
    save_xy_results(records, result_dir, chart_label="scatter")
