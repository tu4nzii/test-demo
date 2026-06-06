"""Evaluation entry points for horizontal bar charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reference.prediction_core.evaluation_utils import numeric_mae, numeric_relative_error, save_axis_results


def compute_mae(pred_x: Any, gt_x: Any) -> float | None:
    return numeric_mae(pred_x, gt_x)


def compute_relative_error(pred_x: Any, gt_x: Any) -> float | None:
    return numeric_relative_error(pred_x, gt_x)


def save_results(records: list[dict[str, Any]], result_dir: Path) -> None:
    save_axis_results(
        records,
        result_dir,
        chart_label="h_bar",
        axis="x",
        pred_column="pred_x",
        final_csv="full_results_with_xre.csv",
        summary_csv="axis_level_summary.csv",
        plot_file="axis_level_mae_re_combined.png",
        plot_title="X Axis MAE & Relative Error by Prompt+Image Setting",
        bar_color="#76D7C4",
        line_color="#F1948A",
    )
