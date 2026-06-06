"""Evaluation entry points for donut charts."""

from __future__ import annotations

import pandas as pd

from reference.prediction_core.evaluation_utils import polar_mae, polar_percent_relative_error, save_polar_summary_and_plot


def compute_mae(pred: float | None, gt: float) -> float | None:
    return polar_mae(pred, gt)


def compute_relative_error(pred: float | None, gt: float) -> float | None:
    return polar_percent_relative_error(pred, gt)


def save_summary_and_plot(df_single_chart: pd.DataFrame, out_dir: str, chart_id: str) -> None:
    save_polar_summary_and_plot(df_single_chart, out_dir, chart_id)
