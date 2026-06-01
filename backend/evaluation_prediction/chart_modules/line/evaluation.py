"""Evaluation entry points for line charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.evaluation_utils import numeric_mae, numeric_relative_error, save_axis_results


def compute_mae(pred_y: Any, gt_y: Any) -> float | None:
    return numeric_mae(pred_y, gt_y)


def compute_relative_error(pred_y: Any, gt_y: Any) -> float | None:
    return numeric_relative_error(pred_y, gt_y)


def save_results(records: list[dict[str, Any]], result_dir: Path) -> None:
    save_axis_results(
        records,
        result_dir,
        chart_label="line",
        axis="y",
        pred_column="pred_y",
        final_csv="full_results_with_yre.csv",
        summary_csv="mae_summary.csv",
        plot_file="error_comparison_plot.png",
        plot_title="Line Y Axis MAE & Relative Error by Prompt+Image Setting",
        bar_color="#D98880",
        line_color="#5DADE2",
    )
