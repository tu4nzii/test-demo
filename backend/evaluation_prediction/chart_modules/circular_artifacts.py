"""Artifacts for pie/donut prediction runs.

The standalone prediction_core flow writes feedback/amplifier images plus CSV
summaries. Backend prediction does not use GT, so MAE/relative-error fields are
kept empty while prediction coverage and value summaries are still saved.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw

from ..common.chart_io import ensure_dir


def save_circular_artifacts(
    *,
    dataset: dict[str, Any],
    chart_type: str,
    result_dir: Path,
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ensure_dir(result_dir)
    all_records = _merge_prediction_records(dataset, records, predictions)
    pd.DataFrame(all_records).to_csv(result_dir / "experiment_results.csv", index=False)
    _save_summary_and_plot(all_records, result_dir, str(dataset.get("chart_id", "")))
    _save_feedback_images(dataset, result_dir, predictions)
    _save_amplifier_images(dataset, chart_type, result_dir, predictions)
    return all_records


def _merge_prediction_records(
    dataset: dict[str, Any],
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = [dict(record) for record in records]
    existing = {
        (
            str(record.get("point")),
            str(record.get("prompt_type")),
            str(record.get("image_type")),
        )
        for record in merged
    }
    for item in predictions:
        key = (str(item.get("label") or item.get("id")), str(item.get("prompt_type")), str(item.get("image_type")))
        if key in existing:
            continue
        merged.append(
            {
                "chart_id": dataset.get("chart_id"),
                "point": item.get("label") or item.get("id"),
                "prompt_type": item.get("prompt_type") or item.get("extraction_source") or "prediction",
                "image_type": item.get("image_type"),
                "run": 1,
                "image_path": item.get("image_path"),
                "gt": None,
                "pred": item.get("value"),
                "pred_pct": item.get("value"),
                "percentage": item.get("percentage"),
                "start_angle": item.get("start_angle"),
                "end_angle": item.get("end_angle"),
                "mae": None,
                "rel_err": None,
                "raw_prediction": "",
            }
        )
    return merged


def _save_summary_and_plot(records: list[dict[str, Any]], result_dir: Path, chart_id: str) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        (result_dir / "summary.csv").write_text("", encoding="utf-8")
        return

    df["pred_numeric"] = pd.to_numeric(df.get("pred_pct", df.get("pred")), errors="coerce")
    if "mae" not in df:
        df["mae"] = None
    if "rel_err" not in df:
        df["rel_err"] = None
    summary = (
        df.groupby(["prompt_type", "image_type"], dropna=False)
        .agg(
            object_count=("point", "nunique"),
            record_count=("point", "count"),
            valid_prediction_count=("pred_numeric", lambda values: values.notna().sum()),
            avg_prediction=("pred_numeric", "mean"),
            avg_mae=("mae", "mean"),
            avg_re=("rel_err", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(result_dir / "summary.csv", index=False)

    if summary.empty:
        return

    labels = summary.apply(lambda row: f"{row['prompt_type']}\n{row['image_type']}", axis=1)
    x_values = list(range(len(summary)))
    fig, ax1 = plt.subplots(figsize=(max(7, len(summary) * 1.4), 4.5), dpi=150)
    ax1.bar(x_values, summary["valid_prediction_count"], color="#4C78A8", alpha=0.8, label="Valid predictions")
    ax1.set_ylabel("Valid Prediction Count")
    ax1.set_xticks(x_values)
    ax1.set_xticklabels(labels, rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x_values, summary["avg_prediction"], color="#F58518", marker="o", label="Average value")
    ax2.set_ylabel("Average Predicted Share")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    plt.title(f"{chart_id} Prediction Summary")
    plt.tight_layout()
    plt.savefig(result_dir / "mae_relerr_plot.png", bbox_inches="tight")
    plt.close(fig)


def _save_feedback_images(dataset: dict[str, Any], result_dir: Path, predictions: list[dict[str, Any]]) -> None:
    source = _path_or_none(_with_grid_path(dataset)) or _path_or_none(_no_grid_path(dataset))
    if source is None:
        return
    center, radius = _circle_geometry(dataset, source)
    out_dir = ensure_dir(result_dir / "feedback_img")
    for item in predictions:
        start = _number_or_none(item.get("start_angle"))
        end = _number_or_none(item.get("end_angle"))
        if start is None or end is None:
            continue
        label = str(item.get("label") or item.get("id") or "segment")
        out_path = out_dir / f"{_safe_name(label)}_feedback.png"
        _draw_angle_feedback(source, out_path, center, radius, start, end)


def _save_amplifier_images(dataset: dict[str, Any], chart_type: str, result_dir: Path, predictions: list[dict[str, Any]]) -> None:
    source = _path_or_none(_no_grid_path(dataset)) or _path_or_none(_with_grid_path(dataset))
    if source is None:
        return
    center, radius = _circle_geometry(dataset, source)
    inner_radius = int(radius * 0.45) if chart_type == "donut" else 0
    out_dir = ensure_dir(result_dir / "amplifier_img")
    for item in predictions:
        start = _number_or_none(item.get("start_angle"))
        end = _number_or_none(item.get("end_angle"))
        if start is None or end is None:
            continue
        label = str(item.get("label") or item.get("id") or "segment")
        out_path = out_dir / f"{_safe_name(label)}_amp1.png"
        _crop_sector(source, out_path, center, inner_radius, radius, start, end)


def _draw_angle_feedback(
    source: Path,
    out_path: Path,
    center: tuple[int, int],
    radius: int,
    start_angle: float,
    end_angle: float,
) -> None:
    with Image.open(source).convert("RGBA") as base:
        img = base.copy()
    draw = ImageDraw.Draw(img)
    cx, cy = center
    line_len = max(12, int(radius * 0.08))
    for angle in (start_angle, end_angle):
        theta = math.radians(angle - 90.0)
        x1 = cx + (radius - line_len) * math.cos(theta)
        y1 = cy + (radius - line_len) * math.sin(theta)
        x2 = cx + (radius + line_len) * math.cos(theta)
        y2 = cy + (radius + line_len) * math.sin(theta)
        draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0, 255), width=3)
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.arc(bbox, start=(start_angle - 90.0) % 360, end=(end_angle - 90.0) % 360, fill=(255, 0, 0, 255), width=3)
    img.save(out_path)


def _crop_sector(
    source: Path,
    out_path: Path,
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
    start_angle: float,
    end_angle: float,
) -> None:
    with Image.open(source).convert("RGBA") as base:
        bbox = _sector_bbox(center, outer_radius, base.size)
        crop = base.crop(bbox)
        local_center = (center[0] - bbox[0], center[1] - bbox[1])
        mask = Image.new("L", crop.size, 0)
        draw = ImageDraw.Draw(mask)
        points = [local_center]
        span = (end_angle - start_angle) % 360
        steps = max(12, int(span // 3) + 1)
        for index in range(steps + 1):
            angle = start_angle + span * index / steps
            theta = math.radians(angle - 90.0)
            points.append(
                (
                    local_center[0] + outer_radius * math.cos(theta),
                    local_center[1] + outer_radius * math.sin(theta),
                )
            )
        draw.polygon(points, fill=255)
        if inner_radius > 0:
            draw.ellipse(
                [
                    local_center[0] - inner_radius,
                    local_center[1] - inner_radius,
                    local_center[0] + inner_radius,
                    local_center[1] + inner_radius,
                ],
                fill=0,
            )
        result = Image.new("RGBA", crop.size, (255, 255, 255, 0))
        result.paste(crop, (0, 0), mask)
        result.save(out_path)


def _sector_bbox(center: tuple[int, int], radius: int, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    cx, cy = center
    pad = max(8, int(radius * 0.08))
    left = max(0, cx - radius - pad)
    top = max(0, cy - radius - pad)
    right = min(width, cx + radius + pad)
    bottom = min(height, cy + radius + pad)
    return left, top, right, bottom


def _circle_geometry(dataset: dict[str, Any], source: Path) -> tuple[tuple[int, int], int]:
    with Image.open(source) as img:
        width, height = img.size
    raw_center = dataset.get("center")
    if isinstance(raw_center, dict):
        center = (int(raw_center.get("x", width // 2)), int(raw_center.get("y", height // 2)))
    elif isinstance(raw_center, (list, tuple)) and len(raw_center) >= 2:
        center = (int(raw_center[0]), int(raw_center[1]))
    else:
        center = (width // 2, height // 2)
    raw_radius = dataset.get("r_pixels") or dataset.get("radius")
    if isinstance(raw_radius, (list, tuple)):
        radius_values = [_number_or_none(value) for value in raw_radius]
        radius = int(max([value for value in radius_values if value is not None] or [min(width, height) * 0.35]))
    else:
        radius = int(_number_or_none(raw_radius) or min(width, height) * 0.35)
    return center, max(1, radius)


def _with_grid_path(dataset: dict[str, Any]) -> Any:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    return image_paths.get("with_grid") or image_paths.get("grid_with_grid") or dataset.get("encrypted_grid_path")


def _no_grid_path(dataset: dict[str, Any]) -> Any:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    return image_paths.get("no_grid") or dataset.get("image_path")


def _path_or_none(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).resolve()
    return path if path.exists() else None


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number else None
    except Exception:
        return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "segment"
