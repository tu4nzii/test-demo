"""Shared evaluation helpers used by chart-specific runners."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reference.prediction_core.chart_io import ensure_dir


DEFAULT_ROUND_GROUPS = ("chart_id", "point", "prompt_type", "image_type")
XY_ROUND_GROUPS = ("chart_id", "point_name", "prompt_type", "image_type")


def numeric_mae(pred: Any, gt: Any, digits: int = 4) -> float | None:
    try:
        return round(abs(float(pred) - float(gt)), digits)
    except Exception:
        return None


def numeric_relative_error(pred: Any, gt: Any, digits: int = 4) -> float | None:
    try:
        gt_value = float(gt)
        if gt_value == 0:
            return None
        return round(abs(float(pred) - gt_value) / abs(gt_value), digits)
    except Exception:
        return None


def xy_mae(pred: tuple[Any, Any], gt: tuple[Any, Any]) -> float | None:
    try:
        return round(abs(float(pred[0]) - float(gt[0])) + abs(float(pred[1]) - float(gt[1])), 4)
    except Exception:
        return None


def xy_relative_error(pred: tuple[Any, Any], gt: tuple[Any, Any]) -> tuple[float | None, float | None]:
    try:
        pred_x, pred_y = float(pred[0]), float(pred[1])
        gt_x, gt_y = float(gt[0]), float(gt[1])
        x_re = None if gt_x == 0 else round(abs(pred_x - gt_x) / abs(gt_x), 4)
        y_re = None if gt_y == 0 else round(abs(pred_y - gt_y) / abs(gt_y), 4)
        return x_re, y_re
    except Exception:
        return None, None


def final_rounds(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    group_keys = list(group_cols)
    df["round_index"] = df.groupby(group_keys).cumcount()
    df["max_round_index"] = df.groupby(group_keys)["round_index"].transform("max")
    return df[df["round_index"] == df["max_round_index"]].drop(columns=["max_round_index"])


def save_axis_results(
    records: list[dict[str, Any]],
    result_dir: Path,
    *,
    chart_label: str,
    axis: str,
    pred_column: str,
    final_csv: str,
    summary_csv: str,
    plot_file: str,
    plot_title: str,
    bar_color: str,
    line_color: str,
) -> None:
    ensure_dir(result_dir)
    df = pd.DataFrame(records)
    if df.empty:
        print(f"[{chart_label} evaluation] No records generated.")
        return

    axis_key = axis.lower()
    axis_name = axis.upper()
    df.to_csv(result_dir / "experiment_results.csv", index=False)
    final_df = final_rounds(df, DEFAULT_ROUND_GROUPS)
    final_df.to_csv(result_dir / final_csv, index=False)

    summary = final_df.groupby(["prompt_type", "image_type"]).agg(
        **{
            f"avg_{axis_key}_mae": ("mae", "mean"),
            f"avg_{axis_key}_re": (f"{axis_key}_re", "mean"),
            f"valid_{axis_key}_count": (
                pred_column,
                lambda values: pd.to_numeric(values, errors="coerce").notna().sum(),
            ),
        }
    ).reset_index()
    summary.to_csv(result_dir / summary_csv, index=False)

    if summary.empty:
        return

    labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
    x_values = range(len(summary))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x_values, summary[f"avg_{axis_key}_mae"], width=0.4, label=f"{axis_name} MAE", color=bar_color)
    ax1.set_ylabel(f"MAE ({axis_name} axis)")
    ax1.set_xlabel("Prompt + Image Setting")
    ax1.set_xticks(list(x_values))
    ax1.set_xticklabels(labels, rotation=20)

    ax2 = ax1.twinx()
    ax2.plot(
        x_values,
        summary[f"avg_{axis_key}_re"],
        label=f"{axis_name} Relative Error",
        color=line_color,
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel(f"Relative Error ({axis_name} axis)")

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(result_dir / plot_file)
    plt.close(fig)


def save_xy_results(records: list[dict[str, Any]], result_dir: Path, *, chart_label: str) -> None:
    ensure_dir(result_dir)
    df = pd.DataFrame(records)
    if df.empty:
        print(f"[{chart_label} evaluation] No records generated.")
        return

    df.to_csv(result_dir / "experiment_results.csv", index=False)
    final_df = final_rounds(df, XY_ROUND_GROUPS)
    final_df.to_csv(result_dir / "full_results_with_yre.csv", index=False)

    summary = final_df.groupby(["prompt_type", "image_type"]).agg(
        avg_mae=("mae", "mean"),
        avg_x_mae=("x_abs_err", "mean"),
        avg_y_mae=("y_abs_err", "mean"),
        avg_x_re=("x_re", "mean"),
        avg_y_re=("y_re", "mean"),
        avg_px_rel_x=("pixel_rel_x", "mean"),
        avg_px_rel_y=("pixel_rel_y", "mean"),
        avg_xy_err_over_range=("xy_err_over_range", "mean"),
        valid_count=("pred_x", lambda values: pd.to_numeric(values, errors="coerce").notna().sum()),
    ).reset_index()
    summary.to_csv(result_dir / "mae_summary.csv", index=False)
    summary[["prompt_type", "image_type", "avg_mae"]].to_csv(result_dir / "prompt_comparison.csv", index=False)

    if summary.empty:
        return

    labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
    x_values = range(len(summary))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x_values, summary["avg_mae"], width=0.35, label="MAE", color="#D98880")
    ax1.set_ylabel("MAE")
    ax1.set_xticks(list(x_values))
    ax1.set_xticklabels(labels, rotation=25)

    ax2 = ax1.twinx()
    ax2.plot(x_values, summary["avg_x_re"], marker="o", label="X Relative Error", color="#2E86C1")
    ax2.plot(x_values, summary["avg_y_re"], marker="o", label="Y Relative Error", color="#28B463")
    ax2.set_ylabel("Relative Error")

    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, loc="upper left")
    plt.title("Average MAE and Relative Errors by Prompt+Image Setting")
    plt.tight_layout()
    plt.savefig(result_dir / "error_comparison_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    ax.bar([i - width / 2 for i in x_values], summary["avg_x_mae"], width=width, label="X MAE", color="#85C1E9")
    ax.bar([i + width / 2 for i in x_values], summary["avg_y_mae"], width=width, label="Y MAE", color="#F5B7B1")
    ax.set_xticks(list(x_values))
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("MAE per axis")
    ax.legend()
    plt.tight_layout()
    plt.savefig(result_dir / "y_mae_comparison_plot.png")
    plt.close(fig)


def polar_mae(pred: float | None, gt: float) -> float | None:
    if pred is None:
        return None
    return round(abs(pred - gt), 3)


def polar_percent_relative_error(pred: float | None, gt: float) -> float | None:
    if pred is None or gt == 0:
        return None
    return round(abs(pred - gt) / gt * 100, 2)


def pie_angle_relative_error(pred: float | None, gt: float) -> float | None:
    if pred is None or gt == 0:
        return None
    return round(abs(pred - gt / 360), 2)


def save_polar_summary_and_plot(df_single_chart: pd.DataFrame, out_dir: str, chart_id: str) -> None:
    summary = (
        df_single_chart
        .groupby(["prompt_type", "image_type"], sort=True)
        .agg(avg_mae=("mae", "mean"), avg_re=("rel_err", "mean"))
        .sort_index()
    )

    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"))

    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
    ax2 = ax1.twinx()
    x_values = np.arange(len(summary))
    bar_width = 0.4

    bars1 = ax1.bar(x_values - bar_width / 2, summary["avg_mae"], width=bar_width, label="MAE")
    bars2 = ax2.bar(
        x_values + bar_width / 2,
        summary["avg_re"],
        width=bar_width,
        label="Relative Error (%)",
        hatch="//",
        alpha=0.7,
    )

    ax1.set_ylabel("Average MAE")
    ax2.set_ylabel("Average Relative Error (%)")
    ax1.set_xticks(x_values)
    ax1.set_xticklabels([f"{pt}\n{it}" for pt, it in summary.index], rotation=15, ha="right")

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.title(f"{chart_id} - MAE vs Relative Error")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mae_relerr_plot.png"), bbox_inches="tight")
    plt.close(fig)
