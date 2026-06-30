from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

GRID_MODULE_DIR = Path(__file__).resolve().parents[2] / "Grid_generation"
if str(GRID_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_MODULE_DIR))

from grid_math import parse_numeric_label


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DATASET_CATEGORY_TYPES = {
    "bubble": "bubble",
    "donut": "donut",
    "hbar": "h_bar",
    "line": "line",
    "pie": "pie",
    "radar": "radar",
    "rose": "rose",
    "scatter": "scatter",
    "vbar": "v_bar",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def try_load_json(path: Path) -> Any | None:
    try:
        return load_json(path)
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return parse_numeric_label(str(value if value is not None else ""))


def image_short_side(image_path: Path | None) -> int | None:
    if image_path is None or not image_path.exists():
        return None
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if image is None or image.ndim < 2:
        return None
    h, w = image.shape[:2]
    return int(min(h, w))


def grid_pixel_threshold(record: dict[str, Any], explicit_threshold: float | None = None) -> float | None:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
    for key in ("source_image", "uploaded_image", "encrypted_grid"):
        path = copied.get(key)
        short_side = image_short_side(Path(str(path))) if path else None
        if short_side is not None:
            return float(math.ceil(short_side * 0.05))
    return None


def normalize_text(value: Any) -> str:
    return " ".join(str(value if value is not None else "").strip().casefold().replace(",", "").split())


def normalize_chart_type(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"h_bar", "horizontal_bar", "h_stacked_bar", "horizontal_stacked_bar"}:
        return "h_bar"
    if text in {"v_bar", "vertical_bar", "v_stacked_bar", "vertical_stacked_bar"}:
        return "v_bar"
    if "bubble" in text:
        return "bubble"
    if "scatter" in text or "scatetr" in text:
        return "scatter"
    if "line" in text:
        return "line"
    if "donut" in text:
        return "donut"
    if "pie" in text:
        return "pie"
    if "radar" in text:
        return "radar"
    if "rose" in text or "nightingale" in text:
        return "rose"
    if "stacked" in text and ("hbar" in text or "xbar" in text or "horizontal" in text):
        return "h_bar"
    if "stacked" in text and "bar" in text:
        return "v_bar"
    if "hbar" in text or "xbar" in text or "horizontal" in text:
        return "h_bar"
    if "bar" in text:
        return "v_bar"
    return text or "unknown"


def chart_family(value: Any) -> str:
    chart_type = normalize_chart_type(value)
    if chart_type in {"v_bar", "h_bar"}:
        return "bar"
    if chart_type in {"bubble", "scatter"}:
        return "point"
    return chart_type


def chart_type_matches(pred_type: Any, gt_type: Any) -> bool:
    pred = normalize_chart_type(pred_type)
    gt = normalize_chart_type(gt_type)
    if pred == gt:
        return True
    return {pred, gt} == {"bubble", "scatter"}


def category_folder_type(folder_name: str) -> str | None:
    prefix = re.split(r"[_\-\s]+", folder_name.strip().lower(), maxsplit=1)[0]
    return DATASET_CATEGORY_TYPES.get(prefix)


def build_final_realdataset_category_index(dataset_root: Path) -> dict[str, str]:
    root = dataset_root / "Final-RealDataset"
    index: dict[str, str] = {}
    if not root.exists():
        return index
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir() or group_dir.name == "ALL":
            continue
        chart_type = category_folder_type(group_dir.name)
        if not chart_type:
            continue
        for folder_name in ("chart", "charts"):
            chart_dir = group_dir / folder_name
            if not chart_dir.exists():
                continue
            for image_path in chart_dir.rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                index[image_path.name.lower()] = chart_type
                index[image_path.stem.lower()] = chart_type
    return index


def infer_gt_type(
    dataset_relative: str,
    dataset_root: Path | None = None,
    final_realdataset_index: dict[str, str] | None = None,
) -> str:
    rel = Path(dataset_relative)
    parts = rel.parts
    if len(parts) >= 2:
        dataset = parts[0]
        group = parts[1]
        if group != "ALL":
            chart_type = category_folder_type(group)
            if chart_type:
                return chart_type
        elif dataset == "Final-RealDataset":
            index = final_realdataset_index
            if index is None and dataset_root is not None:
                index = build_final_realdataset_category_index(dataset_root)
            if index:
                filename_type = index.get(rel.name.lower())
                if filename_type:
                    return filename_type
                stem_type = index.get(rel.stem.lower())
                if stem_type:
                    return stem_type
    return normalize_chart_type(dataset_relative)


def source_config_path(dataset_root: Path, dataset_relative: str) -> Path | None:
    rel = Path(dataset_relative)
    parts = rel.parts
    if not parts:
        return None
    dataset = parts[0]
    inner = Path(*parts[1:])
    image_stem = inner.stem
    root = dataset_root / dataset
    candidates: list[Path] = []
    if dataset == "Final-RealDataset":
        candidates.extend(
            [
                root / "ALL" / "chart_config" / f"{image_stem}_encrypted.json",
                root / "ALL" / "chart_config" / f"{image_stem}.json",
            ]
        )
    elif len(inner.parts) >= 2:
        group = inner.parts[0]
        candidates.extend(
            [
                root / group / "chart_config" / f"{image_stem}.json",
                root / group / "chart_configs" / f"{image_stem}.json",
                root / group / "chart_config" / f"{image_stem}_encrypted.json",
                root / group / "chart_configs" / f"{image_stem}_encrypted.json",
            ]
        )
    candidates.extend(root.rglob(f"{image_stem}.json"))
    candidates.extend(root.rglob(f"{image_stem}_encrypted.json"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def artifact_payload(record: dict[str, Any]) -> dict[str, Any]:
    copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
    payload: dict[str, Any] = {}
    for key in ("ticks_json",):
        path = copied.get(key)
        if path and Path(str(path)).exists():
            try:
                payload.update(load_json(Path(str(path))))
            except Exception:
                pass
    artifact_dir = Path(str(copied.get("encrypted_grid") or "")).parent
    if artifact_dir.exists():
        for path in sorted(artifact_dir.glob("*_grid_failure.json")):
            try:
                failure_report = load_json(path)
                if isinstance(failure_report, dict):
                    payload["grid_failure"] = failure_report
                    payload["grid_failure_path"] = str(path)
                    break
            except Exception:
                payload["grid_failure"] = {
                    "failed": True,
                    "reason": "invalid_failure_report",
                }
                payload["grid_failure_path"] = str(path)
                break
        for path in sorted(artifact_dir.glob("*_final*_bindings.json")):
            try:
                payload["final_bindings"] = load_json(path)
                payload["final_bindings_path"] = str(path)
                break
            except Exception:
                continue
        for path in sorted(artifact_dir.glob("*_image.json")):
            try:
                payload.update(load_json(path))
                break
            except Exception:
                continue
    return payload


def numeric_tolerance(values: list[Any]) -> float:
    nums = sorted(value for item in values if (value := safe_float(item)) is not None)
    if len(nums) >= 2:
        gaps = [abs(right - left) for left, right in zip(nums, nums[1:]) if abs(right - left) > 1e-12]
        if gaps:
            gaps = sorted(gaps)
            mid = len(gaps) // 2
            median_gap = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2
            return max(1e-6, float(median_gap) * 0.01)
    return 1e-6


def pixel_tolerance(pixels: list[Any]) -> float:
    vals = sorted(float(value) for value in pixels)
    if len(vals) >= 2:
        gaps = [abs(right - left) for left, right in zip(vals, vals[1:]) if abs(right - left) > 1e-6]
        if gaps:
            gaps = sorted(gaps)
            mid = len(gaps) // 2
            median_gap = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2
            return max(8.0, min(36.0, float(median_gap) * 0.42))
    return 18.0


def label_matches(gt_label: Any, pred_label: Any, tolerance: float) -> bool:
    gt_num = safe_float(gt_label)
    pred_num = safe_float(pred_label)
    if gt_num is not None and pred_num is not None:
        return abs(gt_num - pred_num) <= tolerance
    return normalize_text(gt_label) == normalize_text(pred_label)


def all_numeric(values: list[Any]) -> bool:
    return bool(values) and all(safe_float(value) is not None for value in values)


def final_binding_ticks(pred: dict[str, Any], axis: str) -> list[dict[str, Any]]:
    final_bindings = pred.get("final_bindings")
    if isinstance(final_bindings, dict):
        axis_data = final_bindings.get(f"{axis}_axis")
        tick_bindings = axis_data.get("tick_bindings", []) if isinstance(axis_data, dict) else []
        ticks: list[dict[str, Any]] = []
        for binding in tick_bindings:
            if not isinstance(binding, dict) or binding.get("position") is None:
                continue
            try:
                position = float(binding.get("position"))
            except (TypeError, ValueError):
                continue
            label = binding.get("label", "")
            ticks.append(
                {
                    "position": position,
                    "label": label,
                    "numeric": binding.get("numeric", safe_float(label)),
                }
            )
        return ticks

    pred_ticks = pred.get(f"{axis}_ticks") if isinstance(pred.get(f"{axis}_ticks"), list) else []
    pred_pixels = pred.get(f"{axis}_pixels") if isinstance(pred.get(f"{axis}_pixels"), list) else []
    ticks = []
    for label, pixel in zip(pred_ticks, pred_pixels):
        try:
            position = float(pixel)
        except (TypeError, ValueError):
            continue
        ticks.append({"position": position, "label": label, "numeric": safe_float(label)})
    return ticks


def nearest_gt_tick(gt_ticks: list[Any], gt_pixels: list[Any], position: float) -> tuple[Any, float] | None:
    if not gt_ticks or not gt_pixels:
        return None
    tolerance = pixel_tolerance(gt_pixels)
    pairs = [(tick, float(pixel)) for tick, pixel in zip(gt_ticks, gt_pixels)]
    gt_label, gt_pixel = min(pairs, key=lambda item: abs(item[1] - float(position)))
    if abs(gt_pixel - float(position)) > tolerance:
        return None
    return gt_label, gt_pixel


def has_cartesian_ticks(gt: dict[str, Any]) -> bool:
    return any(
        isinstance(gt.get(f"{axis}_ticks"), list)
        and isinstance(gt.get(f"{axis}_pixels"), list)
        and bool(gt.get(f"{axis}_ticks"))
        and bool(gt.get(f"{axis}_pixels"))
        for axis in ("x", "y")
    )


def is_numeric_value_axis(chart_type: str, axis: str) -> bool:
    normalized = normalize_chart_type(chart_type)
    if normalized == "v_bar":
        return axis == "y"
    if normalized == "h_bar":
        return axis == "x"
    if normalized in {"line", "scatter", "bubble"}:
        return axis in {"x", "y"}
    return False


def tick_metrics(
    gt: dict[str, Any],
    pred: dict[str, Any],
    axis: str,
    pixel_threshold: float | None,
    count_numeric_values: bool = True,
) -> dict[str, Any]:
    gt_ticks = gt.get(f"{axis}_ticks") if isinstance(gt.get(f"{axis}_ticks"), list) else []
    gt_pixels = gt.get(f"{axis}_pixels") if isinstance(gt.get(f"{axis}_pixels"), list) else []
    pred_ticks = final_binding_ticks(pred, axis)
    num_tol = numeric_tolerance(gt_ticks)
    metric_threshold = float(pixel_threshold) if pixel_threshold is not None else pixel_tolerance(gt_pixels)

    numeric_total = 0
    numeric_matched = 0
    numeric_correct = 0
    numeric_error_sum = 0.0
    position_matched = 0
    position_error_sum = 0.0
    position_max_error = 0.0
    label_total = len(pred_ticks)
    label_matched = 0
    label_correct = 0
    axis_is_numeric = count_numeric_values and all_numeric(gt_ticks)

    for pred_tick in pred_ticks:
        match = nearest_gt_tick(gt_ticks, gt_pixels, float(pred_tick["position"]))
        if match is None:
            if axis_is_numeric:
                numeric_total += 1
            continue
        gt_tick, gt_pixel = match
        position_error = abs(float(gt_pixel) - float(pred_tick["position"]))
        position_matched += 1
        position_error_sum += position_error
        position_max_error = max(position_max_error, position_error)
        label_matched += 1
        if label_matches(gt_tick, pred_tick.get("label"), num_tol):
            label_correct += 1
        if not count_numeric_values:
            continue
        gt_number = safe_float(gt_tick)
        if gt_number is None:
            continue
        numeric_total += 1
        pred_number = safe_float(pred_tick.get("numeric"))
        if pred_number is None:
            pred_number = safe_float(pred_tick.get("label"))
        if pred_number is None:
            continue
        numeric_matched += 1
        numeric_error_sum += position_error
        if position_error <= metric_threshold:
            numeric_correct += 1

    return {
        "numeric_total": numeric_total,
        "numeric_matched": numeric_matched,
        "numeric_correct": numeric_correct,
        "numeric_error_sum": numeric_error_sum,
        "tick_position_matched": position_matched,
        "position_error_sum": position_error_sum,
        "tick_position_max_error_px": position_max_error if position_matched else None,
        "label_total": label_total,
        "label_matched": label_matched,
        "label_correct": label_correct,
    }


def rgb_from_hex(value: Any) -> tuple[int, int, int] | None:
    text = str(value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        return None
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None


def flatten_colors(value: Any, prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(item, str):
                result[name] = item
            elif isinstance(item, dict):
                result.update(flatten_colors(item, name))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("name") and item.get("color"):
                result[str(item["name"])] = str(item["color"])
    return result


def color_metrics(gt: dict[str, Any], pred: dict[str, Any], threshold: float) -> dict[str, Any]:
    gt_colors = flatten_colors(gt.get("series_color"))
    pred_colors = flatten_colors(pred.get("series_color")) or flatten_colors(pred.get("colors"))
    total = len(gt_colors)
    correct = 0
    for gt_name, gt_color in gt_colors.items():
        gt_rgb = rgb_from_hex(gt_color)
        if gt_rgb is None:
            continue
        gt_norm = normalize_text(gt_name)
        candidates = [
            pred_color
            for pred_name, pred_color in pred_colors.items()
            if normalize_text(pred_name) == gt_norm or normalize_text(pred_name).split("/")[-1] == gt_norm.split("/")[-1]
        ]
        for pred_color in candidates:
            pred_rgb = rgb_from_hex(pred_color)
            if pred_rgb is None:
                continue
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(gt_rgb, pred_rgb)))
            if dist <= threshold:
                correct += 1
                break
    return {"legend_color_total": total, "legend_color_correct": correct}


def add_counts(target: dict[str, float], metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            target[key] += value


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = (str(row["dataset"]), str(row["gt_type"]))
        grouped[key]["sample_count"] += 1
        if (
            row.get("status") in {"success", "skipped_success_cache", "recovered_from_grid_reference"}
            and not row.get("grid_eval_skipped")
        ):
            grouped[key]["processed_count"] += 1
        for metric_key in (
            "numeric_total",
            "numeric_matched",
            "numeric_correct",
            "numeric_error_sum",
            "tick_position_matched",
            "position_error_sum",
            "label_total",
            "label_matched",
            "label_correct",
            "legend_color_total",
            "legend_color_correct",
            "chart_type_total",
            "chart_type_correct",
            "chart_family_correct",
        ):
            grouped[key][metric_key] += float(row.get(metric_key) or 0)
        grouped[key]["tick_position_max_error_px"] = max(
            grouped[key].get("tick_position_max_error_px", 0.0),
            float(row.get("tick_position_max_error_px") or 0.0),
        )
    summary = []
    overall: dict[str, float] = defaultdict(float)
    for (dataset, gt_type), values in sorted(grouped.items()):
        summary.append(summary_row(dataset, gt_type, values))
        for key, value in values.items():
            overall[key] = max(overall[key], value) if key == "tick_position_max_error_px" else overall[key] + value
    summary.append(summary_row("overall", "overall", overall))
    return summary


def summary_row(dataset: str, gt_type: str, values: dict[str, float]) -> dict[str, Any]:
    numeric_matched = values.get("numeric_matched", 0)
    numeric_total = values.get("numeric_total", 0)
    position_matched = values.get("tick_position_matched", 0)
    label_total = values.get("label_total", 0)
    color_total = values.get("legend_color_total", 0)
    type_total = values.get("chart_type_total", 0)
    return {
        "dataset": dataset,
        "gt_type": gt_type,
        "sample_count": int(values.get("sample_count", 0)),
        "processed_count": int(values.get("processed_count", 0)),
        "tick_value_mae_px": values.get("numeric_error_sum", 0) / numeric_matched if numeric_matched else None,
        "tick_value_accuracy_2px": values.get("numeric_correct", 0) / numeric_total if numeric_total else None,
        "numeric_total": int(numeric_total),
        "numeric_matched": int(numeric_matched),
        "tick_position_mae_px": values.get("position_error_sum", 0) / position_matched if position_matched else None,
        "tick_position_matched": int(position_matched),
        "tick_position_max_error_px": values.get("tick_position_max_error_px") if position_matched else None,
        "label_name_accuracy": values.get("label_correct", 0) / label_total if label_total else None,
        "label_total": int(label_total),
        "label_matched": int(values.get("label_matched", 0)),
        "legend_color_accuracy": values.get("legend_color_correct", 0) / color_total if color_total else None,
        "legend_color_total": int(color_total),
        "chart_type_accuracy": values.get("chart_type_correct", 0) / type_total if type_total else None,
        "chart_family_accuracy": values.get("chart_family_correct", 0) / type_total if type_total else None,
    }


def infer_grid_chart_type(dataset_relative: str) -> str:
    rel = Path(dataset_relative)
    text = dataset_relative.replace("\\", "/")
    name = rel.stem
    if text.startswith("Final-RealDataset/"):
        if name.startswith(("BarChart", "GroupedBarChart")):
            return "bar"
        if name.startswith("StackedBarChart"):
            return "stackedbar"
        if name.startswith("LineGraph"):
            return "line"
        if name.startswith(("ScatterPlot", "Scatterplot")):
            return "scatter"
        if name.startswith(("BubbleChart", "Bubblechart")):
            return "bubble"
        return normalize_chart_type(name)
    if "vBar_50/" in text:
        return "v_bar"
    if "hBar_50/" in text:
        return "h_bar"
    if "Line_50/" in text:
        return "line"
    if "Scatetr_50/" in text or "Scatter_50/" in text:
        return "scatter"
    if "Bubble_50/" in text:
        return "bubble"
    return normalize_chart_type(text)


def summarize_grid_effect(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grid_rows = [row for row in rows if row.get("grid_eval_included")]
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in grid_rows:
        key = (str(row["dataset"]), str(row.get("grid_type") or row["gt_type"]))
        grouped[key]["sample_count"] += 1
        if (
            row.get("status") in {"success", "skipped_success_cache", "recovered_from_grid_reference"}
            and not row.get("grid_eval_skipped")
        ):
            grouped[key]["processed_count"] += 1
        for metric_key in (
            "numeric_total",
            "numeric_matched",
            "numeric_correct",
            "numeric_error_sum",
            "tick_position_matched",
            "position_error_sum",
            "label_total",
            "label_matched",
            "label_correct",
        ):
            grouped[key][metric_key] += float(row.get(metric_key) or 0)
        grouped[key]["tick_position_max_error_px"] = max(
            grouped[key].get("tick_position_max_error_px", 0.0),
            float(row.get("tick_position_max_error_px") or 0.0),
        )

    summary = []
    overall: dict[str, float] = defaultdict(float)
    for (dataset, chart_type), values in sorted(grouped.items()):
        summary.append(summary_row(dataset, chart_type, values))
        for key, value in values.items():
            overall[key] = max(overall[key], value) if key == "tick_position_max_error_px" else overall[key] + value
    summary.append(summary_row("overall", "overall", overall))
    return summary


def load_reference_grid_rows(reference_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    summary_path = reference_root / "tick_eval_grid_summary.json"
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not summary_path.exists():
        return rows
    for row in load_json(summary_path):
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset"))
        chart_type = str(row.get("chart_type"))
        rows[(dataset, chart_type)] = {
            "dataset": dataset,
            "gt_type": chart_type,
            "sample_count": row.get("sample_count"),
            "processed_count": row.get("evaluated_count"),
            "tick_value_mae_px": row.get("tick_value_mae"),
            "tick_value_accuracy_2px": row.get("tick_value_accuracy"),
            "numeric_total": row.get("numeric_total"),
            "numeric_matched": row.get("numeric_matched"),
            "label_name_accuracy": row.get("label_name_accuracy"),
            "label_total": row.get("label_total"),
        }
    return rows


def write_grid_comparison_report(
    path: Path,
    current_summary: list[dict[str, Any]],
    reference_root: Path,
    threshold_label: str,
) -> None:
    reference_rows = load_reference_grid_rows(reference_root)
    current_by_key = {(str(row["dataset"]), str(row["gt_type"])): row for row in current_summary}
    keys = sorted(current_by_key, key=lambda item: (item[0] != "overall", item[0], item[1]))
    lines = [
        "# Grid Effect Comparison",
        "",
        f"- Current output: `backend/evaluation/recheck_outputs/vishintprompt_full_grid_encryption_latest`",
        f"- Grid reference: `{reference_root}`",
        f"- Tick matching: same reconstructed-grid/final-bindings logic as `F:\\program\\grid`.",
        f"- Tick-value accuracy threshold: `{threshold_label}`.",
        "",
        "| Dataset | Type | Current samples | Current processed | Grid samples | Grid processed | Current MAE(px) | Grid MAE(px) | Current Acc | Grid Acc | Current Label Acc | Grid Label Acc |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in keys:
        current = current_by_key.get(key, {})
        reference = reference_rows.get(key, {})
        lines.append(
            "| {dataset} | {kind} | {current_samples} | {current_processed} | {ref_samples} | {ref_processed} | {current_mae} | {ref_mae} | {current_acc} | {ref_acc} | {current_label} | {ref_label} |".format(
                dataset=key[0],
                kind=key[1],
                current_samples=current.get("sample_count", "-"),
                current_processed=current.get("processed_count", "-"),
                ref_samples=reference.get("sample_count", "-"),
                ref_processed=reference.get("processed_count", "-"),
                current_mae=format_num(current.get("tick_value_mae_px")),
                ref_mae=format_num(reference.get("tick_value_mae_px")),
                current_acc=format_pct(current.get("tick_value_accuracy_2px")),
                ref_acc=format_pct(reference.get("tick_value_accuracy_2px")),
                current_label=format_pct(current.get("label_name_accuracy")),
                ref_label=format_pct(reference.get("label_name_accuracy")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_pct(value: Any) -> str:
    return "-" if value is None else f"{float(value) * 100:.2f}%"


def format_num(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def parse_color_threshold_overrides(value: str | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    if not value:
        return overrides
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid color threshold override: {item!r}")
        key, raw_threshold = item.split("=", 1)
        normalized = normalize_chart_type(key.strip())
        overrides[normalized] = float(raw_threshold.strip())
    return overrides


def color_threshold_for_type(chart_type: Any, default: float, overrides: dict[str, float]) -> float:
    normalized = normalize_chart_type(chart_type)
    return float(overrides.get(normalized, default))


def format_color_threshold_policy(default: float, overrides: dict[str, float]) -> str:
    if not overrides:
        return f"{default:g}"
    items = ", ".join(f"{key}={value:g}" for key, value in sorted(overrides.items()))
    return f"default={default:g}; {items}"


TABLE_COLUMNS = (
    "| 数据集 | 类别 | 样本数 | 已处理 | Tick MAE(px) | Tick Acc@2px | "
    "Pos MAE(px) | 图例颜色准确率 | 标签准确率 | 图表分类准确率 | 图表家族准确率 |"
)
TABLE_RULE = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
CARTESIAN_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}
POLAR_TYPES = {"pie", "donut", "radar", "rose"}
TYPE_ORDER = {
    "v_bar": 0,
    "h_bar": 1,
    "line": 2,
    "scatter": 3,
    "bubble": 4,
    "rose": 5,
    "pie": 6,
    "donut": 7,
    "radar": 8,
}


def summary_metric_counts(row: dict[str, Any]) -> dict[str, float]:
    numeric_total = float(row.get("numeric_total") or 0)
    numeric_matched = float(row.get("numeric_matched") or 0)
    position_matched = float(row.get("tick_position_matched") or 0)
    label_total = float(row.get("label_total") or 0)
    color_total = float(row.get("legend_color_total") or 0)
    sample_count = float(row.get("sample_count") or 0)
    return {
        "sample_count": sample_count,
        "processed_count": float(row.get("processed_count") or 0),
        "numeric_total": numeric_total,
        "numeric_matched": numeric_matched,
        "numeric_correct": float(row.get("tick_value_accuracy_2px") or 0) * numeric_total,
        "numeric_error_sum": float(row.get("tick_value_mae_px") or 0) * numeric_matched,
        "tick_position_matched": position_matched,
        "position_error_sum": float(row.get("tick_position_mae_px") or 0) * position_matched,
        "label_total": label_total,
        "label_correct": float(row.get("label_name_accuracy") or 0) * label_total,
        "legend_color_total": color_total,
        "legend_color_correct": float(row.get("legend_color_accuracy") or 0) * color_total,
        "chart_type_total": sample_count,
        "chart_type_correct": float(row.get("chart_type_accuracy") or 0) * sample_count,
        "chart_family_correct": float(row.get("chart_family_accuracy") or 0) * sample_count,
        "tick_position_max_error_px": float(row.get("tick_position_max_error_px") or 0),
    }


def add_summary_counts(target: dict[str, float], row: dict[str, Any]) -> None:
    counts = summary_metric_counts(row)
    for key, value in counts.items():
        if key == "tick_position_max_error_px":
            target[key] = max(target.get(key, 0.0), value)
        else:
            target[key] = target.get(key, 0.0) + value


def row_from_summary_counts(dataset: str, category: str, counts: dict[str, float]) -> dict[str, Any]:
    row = summary_row(dataset, category, counts)
    row["dataset"] = dataset
    row["category"] = category
    return row


def report_row(row: dict[str, Any], category_key: str = "category") -> str:
    return (
        "| {dataset} | {category} | {sample_count} | {processed_count} | {mae} | {tick_acc} | "
        "{position_mae} | {color_acc} | {label_acc} | {type_acc} | {family_acc} |"
    ).format(
        dataset=row["dataset"],
        category=row[category_key],
        sample_count=row["sample_count"],
        processed_count=row["processed_count"],
        mae=format_num(row.get("tick_value_mae_px")),
        tick_acc=format_pct(row.get("tick_value_accuracy_2px")),
        position_mae=format_num(row.get("tick_position_mae_px")),
        color_acc=format_pct(row.get("legend_color_accuracy")),
        label_acc=format_pct(row.get("label_name_accuracy")),
        type_acc=format_pct(row.get("chart_type_accuracy")),
        family_acc=format_pct(row.get("chart_family_accuracy")),
    )


def format_status(value: Any) -> str:
    status_map = {
        "success": "成功",
        "skipped_success_cache": "缓存成功",
        "recovered_from_grid_reference": "参考恢复成功",
        "failed": "失败",
    }
    return status_map.get(str(value or ""), str(value or "-"))


def format_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "-":
        return "-"
    reason_map = {
        "x_axis_semantic_only_low_ocr_support": "x轴仅语义候选，OCR支持不足",
        "y_axis_semantic_only_low_ocr_support": "y轴仅语义候选，OCR支持不足",
        "x_axis_semantic_only_repeated_labels": "x轴仅语义候选，标签重复",
        "y_axis_semantic_only_repeated_labels": "y轴仅语义候选，标签重复",
        "x_axis_selected_candidate_invalid": "x轴选中候选无效",
        "y_axis_selected_candidate_invalid": "y轴选中候选无效",
        "x_axis_selected_candidate_has_invalid_reasons": "x轴选中候选存在无效原因",
        "y_axis_selected_candidate_has_invalid_reasons": "y轴选中候选存在无效原因",
        "x_axis_selected_axis_severely_undercovered": "x轴覆盖严重不足",
        "y_axis_selected_axis_severely_undercovered": "y轴覆盖严重不足",
        "x_axis_dense_axis_low_unique_ocr_support": "x轴密集轴唯一OCR支持不足",
        "y_axis_dense_axis_low_unique_ocr_support": "y轴密集轴唯一OCR支持不足",
        "x_axis_numeric_selected_low_ocr_support": "x轴数值轴OCR支持不足",
        "y_axis_numeric_selected_low_ocr_support": "y轴数值轴OCR支持不足",
        "x_axis_numeric_selected_large_ocr_distance": "x轴数值轴OCR距离过大",
        "y_axis_numeric_selected_large_ocr_distance": "y轴数值轴OCR距离过大",
        "grid_generation_failed": "网格生成失败",
        "invalid_or_unreadable_gt_json": "GT JSON无效或无法读取",
    }
    parts = [part for part in text.split("+") if part]
    if parts and all(part in reason_map for part in parts):
        return "；".join(reason_map[part] for part in parts)
    return reason_map.get(text, text.replace("|", "/"))


def category_for_type(chart_type: str) -> str:
    normalized = normalize_chart_type(chart_type)
    if normalized in CARTESIAN_TYPES:
        return "直角系"
    if normalized in POLAR_TYPES:
        return "极坐标"
    return "其他"


def category_summary_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    type_rows = [row for row in summary if row.get("dataset") != "overall"]
    dataset_order = []
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals_by_dataset: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    grand_by_category: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    grand_total: dict[str, float] = defaultdict(float)

    for row in type_rows:
        dataset = str(row["dataset"])
        if dataset not in dataset_order:
            dataset_order.append(dataset)
        category = category_for_type(str(row["gt_type"]))
        add_summary_counts(grouped[(dataset, category)], row)
        add_summary_counts(totals_by_dataset[dataset], row)
        add_summary_counts(grand_by_category[category], row)
        add_summary_counts(grand_total, row)

    rows: list[dict[str, Any]] = []
    for dataset in dataset_order:
        for category in ("直角系", "极坐标", "其他"):
            if (dataset, category) in grouped:
                rows.append(row_from_summary_counts(dataset, category, grouped[(dataset, category)]))
        rows.append(row_from_summary_counts(dataset, "总计", totals_by_dataset[dataset]))
    for category in ("直角系", "极坐标", "其他"):
        if category in grand_by_category:
            rows.append(row_from_summary_counts("总计", category, grand_by_category[category]))
    rows.append(row_from_summary_counts("总计", "总计", grand_total))
    return rows


def metric_item_count(row: dict[str, Any], metric_key: str) -> int:
    if metric_key == "tick_value_mae_px":
        return int(row.get("numeric_matched") or 0)
    if metric_key == "tick_value_accuracy_2px":
        return int(row.get("numeric_total") or 0)
    if metric_key == "tick_position_mae_px":
        return int(row.get("tick_position_matched") or 0)
    if metric_key == "legend_color_accuracy":
        return int(row.get("legend_color_total") or 0)
    if metric_key == "label_name_accuracy":
        return int(row.get("label_total") or 0)
    if metric_key in {"chart_type_accuracy", "chart_family_accuracy"}:
        return int(row.get("sample_count") or 0)
    return 0


def metric_value_text(metric_key: str, value: Any) -> str:
    if "accuracy" in metric_key:
        return format_pct(value)
    return format_num(value)


def worst_metric_rows(type_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_specs = [
        ("Tick MAE(px)", "tick_value_mae_px", "max", "数值轴 tick 的像素误差最大"),
        ("Tick Acc@2px", "tick_value_accuracy_2px", "min", "数值轴 tick 在 2px 容差下的错误率最高"),
        ("Pos MAE(px)", "tick_position_mae_px", "max", "tick 位置匹配后的像素误差最大"),
        ("图例颜色准确率", "legend_color_accuracy", "min", "图例/系列颜色匹配错误率最高"),
        ("标签准确率", "label_name_accuracy", "min", "tick 标签文本匹配错误率最高"),
        ("图表分类准确率", "chart_type_accuracy", "min", "图表类型分类错误率最高"),
        ("图表家族准确率", "chart_family_accuracy", "min", "图表家族分类错误率最高"),
    ]
    rows: list[dict[str, Any]] = []
    for label, key, direction, meaning in metric_specs:
        candidates = [
            row for row in type_rows
            if row.get(key) is not None and metric_item_count(row, key) > 0
        ]
        if not candidates:
            continue
        if direction == "max":
            worst = max(candidates, key=lambda row: float(row.get(key) or 0))
        else:
            worst = min(candidates, key=lambda row: float(row.get(key) or 0))
        value = worst.get(key)
        rows.append(
            {
                "metric": label,
                "dataset": worst.get("dataset"),
                "type": worst.get("gt_type"),
                "value": metric_value_text(key, value),
                "items": metric_item_count(worst, key),
                "meaning": meaning,
            }
        )
    return rows


def sample_metric_count(row: dict[str, Any], metric_key: str) -> int:
    if metric_key == "tick_value_mae_px":
        return int(row.get("numeric_matched") or 0)
    if metric_key == "tick_value_accuracy_2px":
        return int(row.get("numeric_total") or 0)
    if metric_key == "tick_position_mae_px":
        return int(row.get("tick_position_matched") or 0)
    if metric_key == "legend_color_accuracy":
        return int(row.get("legend_color_total") or 0)
    if metric_key == "label_name_accuracy":
        return int(row.get("label_total") or 0)
    if metric_key in {"chart_type_accuracy", "chart_family_accuracy"}:
        return int(row.get("chart_type_total") or 0)
    return 0


def sample_metric_value(row: dict[str, Any], metric_key: str) -> float | None:
    if metric_key == "tick_value_mae_px":
        matched = float(row.get("numeric_matched") or 0)
        return float(row.get("numeric_error_sum") or 0) / matched if matched else None
    if metric_key == "tick_value_accuracy_2px":
        total = float(row.get("numeric_total") or 0)
        return float(row.get("numeric_correct") or 0) / total if total else None
    if metric_key == "tick_position_mae_px":
        matched = float(row.get("tick_position_matched") or 0)
        return float(row.get("position_error_sum") or 0) / matched if matched else None
    if metric_key == "legend_color_accuracy":
        total = float(row.get("legend_color_total") or 0)
        return float(row.get("legend_color_correct") or 0) / total if total else None
    if metric_key == "label_name_accuracy":
        total = float(row.get("label_total") or 0)
        return float(row.get("label_correct") or 0) / total if total else None
    if metric_key == "chart_type_accuracy":
        total = float(row.get("chart_type_total") or 0)
        return float(row.get("chart_type_correct") or 0) / total if total else None
    if metric_key == "chart_family_accuracy":
        total = float(row.get("chart_family_total") or row.get("chart_type_total") or 0)
        return float(row.get("chart_family_correct") or 0) / total if total else None
    return None


def worst_cartesian_sample_rows(details: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    good_statuses = {"success", "skipped_success_cache", "recovered_from_grid_reference"}
    included = [
        row
        for row in details
        if normalize_chart_type(row.get("gt_type")) in CARTESIAN_TYPES
        and row.get("grid_eval_included")
        and not row.get("grid_eval_skipped")
        and row.get("status") in good_statuses
    ]
    metric_specs = [
        ("Tick MAE(px)", "tick_value_mae_px", "max", "数值轴 tick 像素误差"),
        ("Tick Acc@2px", "tick_value_accuracy_2px", "min", "数值轴 tick 2px 容差准确率"),
        ("Pos MAE(px)", "tick_position_mae_px", "max", "tick 位置像素误差"),
        ("标签准确率", "label_name_accuracy", "min", "tick 标签文本准确率"),
        ("图例颜色准确率", "legend_color_accuracy", "min", "图例/系列颜色准确率"),
        ("图表分类准确率", "chart_type_accuracy", "min", "图表类型分类结果"),
    ]
    rows: list[dict[str, Any]] = []
    for label, key, direction, meaning in metric_specs:
        candidates: list[tuple[float, dict[str, Any]]] = []
        for row in included:
            value = sample_metric_value(row, key)
            if value is None or sample_metric_count(row, key) <= 0:
                continue
            candidates.append((value, row))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=(direction == "max"))
        for rank, (value, row) in enumerate(candidates[:limit], start=1):
            rows.append(
                {
                    "metric": label,
                    "rank": rank,
                    "sample": row.get("dataset_relative", "-"),
                    "dataset": row.get("dataset", "-"),
                    "type": row.get("gt_type", "-"),
                    "pred_type": row.get("pred_type", "-"),
                    "value": metric_value_text(key, value),
                    "items": sample_metric_count(row, key),
                    "meaning": meaning,
                }
            )
    return rows


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    details_path: Path,
    threshold_label: str,
    color_threshold_label: str,
    details: list[dict[str, Any]],
) -> None:
    type_rows = [row for row in summary if row.get("dataset") != "overall"]
    type_rows = sorted(
        type_rows,
        key=lambda row: (
            str(row.get("dataset")),
            TYPE_ORDER.get(str(row.get("gt_type")), 100),
            str(row.get("gt_type")),
        ),
    )
    overall = next((row for row in summary if row.get("dataset") == "overall"), None)
    failed_rows = [
        row
        for row in details
        if row.get("status") not in {"success", "skipped_success_cache", "recovered_from_grid_reference"}
        or row.get("grid_eval_skipped")
    ]
    lines = [
        "# VisHintPrompt 最新评估报告",
        "",
        "- 生成端输入：只使用系统生成结果。",
        "- 评估端输入：数据集 JSON 只在离线打分脚本中读取。",
        "- Tick 指标：使用与 `F:\\program\\grid` 一致的重建网格/final-bindings 匹配逻辑。",
        "- Tick MAE 和 Tick Acc@2px 只统计数值轴；分类轴/文字轴不参与这两个指标。",
        f"- Tick 准确率阈值：{threshold_label}。",
        f"- 图例颜色准确率阈值：{color_threshold_label}（RGB 欧氏距离）。",
        "- 图表分类准确率：`bubble` 与 `scatter` 按点图族互认。",
        f"- 明细文件：`{details_path}`",
        "",
        "## 类别汇总",
        "",
        TABLE_COLUMNS,
        TABLE_RULE,
    ]
    for row in category_summary_rows(summary):
        lines.append(report_row(row))

    lines.extend(["", "## 类型明细", "", TABLE_COLUMNS.replace("类别", "类型"), TABLE_RULE])
    for row in type_rows:
        detail_row = dict(row)
        detail_row["category"] = detail_row["gt_type"]
        lines.append(report_row(detail_row))

    if overall:
        overall_row = dict(overall)
        overall_row["dataset"] = "总计"
        overall_row["category"] = "全量数据集"
        lines.extend(["", "## 总体结果", "", TABLE_COLUMNS.replace("类别", "范围"), TABLE_RULE, report_row(overall_row)])

    worst_rows = worst_metric_rows(type_rows)
    if worst_rows:
        lines.extend(
            [
                "",
                "## 误差最大项说明",
                "",
                "| 指标 | 误差最大/表现最差项 | 当前值 | 参与计算项数 | 说明 |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in worst_rows:
            lines.append(
                "| {metric} | {dataset} / `{kind}` | {value} | {items} | {meaning} |".format(
                    metric=row["metric"],
                    dataset=row["dataset"],
                    kind=row["type"],
                    value=row["value"],
                    items=row["items"],
                    meaning=row["meaning"],
                )
            )

    cartesian_worst_rows = worst_cartesian_sample_rows(details)
    if cartesian_worst_rows:
        lines.extend(
            [
                "",
                "## 直角坐标系已计入样本中表现最差的图表",
                "",
                "- 这里仅统计未剔除、已参与直角坐标系指标计算的样本；失败样本和 `grid_eval_skipped` 样本不放入此表。",
                "- 每个指标分别列出 Top 5，便于定位是 tick、label、颜色还是分类造成的主要问题。",
                "",
                "| 指标 | 排名 | 样本 | GT类型 | 预测类型 | 当前值 | 参与项数 | 说明 |",
                "| --- | ---: | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in cartesian_worst_rows:
            lines.append(
                "| {metric} | {rank} | `{sample}` | `{kind}` | `{pred}` | {value} | {items} | {meaning} |".format(
                    metric=row["metric"],
                    rank=row["rank"],
                    sample=row["sample"],
                    kind=row["type"],
                    pred=row["pred_type"],
                    value=row["value"],
                    items=row["items"],
                    meaning=row["meaning"],
                )
            )

    if failed_rows:
        lines.extend(
            [
                "",
                "## 未计入已处理指标的样本",
                "",
                "| 数据集 | 样本 | 状态 | 原因 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in sorted(failed_rows, key=lambda item: str(item.get("dataset_relative"))):
            reason = row.get("grid_eval_skip_reason") or row.get("gt_eval_error") or row.get("error") or "-"
            lines.append(
                "| {dataset} | `{sample}` | {status} | {reason} |".format(
                    dataset=row.get("dataset", "-"),
                    sample=row.get("dataset_relative", "-"),
                    status=format_status(row.get("status", "-")),
                    reason=format_reason(reason),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("backend/datasets/VisHintPrompt_datasets"))
    parser.add_argument("--batch-root", type=Path, default=Path("backend/evaluation/recheck_outputs/vishintprompt_full_grid_encryption_latest"))
    parser.add_argument("--output", type=Path, default=Path("backend/evaluation/results/vishintprompt_latest_metrics"))
    parser.add_argument("--grid-reference-root", type=Path, default=Path(r"F:\program\grid\VisHintPrompt_full_eval"))
    parser.add_argument("--pixel-threshold", type=float, default=None)
    parser.add_argument("--color-threshold", type=float, default=45.0)
    parser.add_argument(
        "--color-threshold-by-type",
        default="",
        help="Comma-separated RGB-distance overrides, e.g. v_bar=120,line=120,scatter=80.",
    )
    args = parser.parse_args()
    threshold_label = (
        f"{args.pixel_threshold:g}px"
        if args.pixel_threshold is not None
        else "grid dynamic threshold, ceil(short_side * 0.05)"
    )
    color_threshold_overrides = parse_color_threshold_overrides(args.color_threshold_by_type)
    color_threshold_label = format_color_threshold_policy(args.color_threshold, color_threshold_overrides)

    manifest_path = args.batch_root / "manifest.json"
    manifest = load_json(manifest_path)
    final_realdataset_index = build_final_realdataset_category_index(args.dataset_root)
    rows: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        dataset_relative = str(record.get("dataset_relative") or "")
        gt_type = infer_gt_type(dataset_relative, args.dataset_root, final_realdataset_index)
        pred_type = normalize_chart_type(record.get("chart_type"))
        row = {
            "dataset": str(record.get("dataset") or dataset_relative.split("/", 1)[0]),
            "dataset_relative": dataset_relative,
            "status": record.get("status"),
            "gt_type": gt_type,
            "pred_type": pred_type,
            "chart_type_total": 1,
            "chart_type_correct": 1 if chart_type_matches(pred_type, gt_type) else 0,
            "chart_family_correct": 1 if chart_family(pred_type) == chart_family(gt_type) else 0,
        }
        gt_path = source_config_path(args.dataset_root, dataset_relative)
        pred = artifact_payload(record)
        row["grid_type"] = gt_type
        if gt_path and gt_path.exists() and pred:
            gt = try_load_json(gt_path)
            if not isinstance(gt, dict):
                row["gt_config"] = str(gt_path)
                row["gt_eval_error"] = "invalid_or_unreadable_gt_json"
                rows.append(row)
                continue
            row["grid_eval_included"] = has_cartesian_ticks(gt)
            failure_report = pred.get("grid_failure") if isinstance(pred.get("grid_failure"), dict) else {}
            if row["grid_eval_included"] and failure_report.get("failed"):
                row["grid_eval_skipped"] = True
                row["grid_eval_skip_reason"] = failure_report.get("reason", "grid_generation_failed")
                row["grid_failure_path"] = pred.get("grid_failure_path")
                rows.append(row)
                continue
            row["pixel_error_threshold_px"] = grid_pixel_threshold(record, args.pixel_threshold)
            x_metrics = tick_metrics(
                gt,
                pred,
                "x",
                row["pixel_error_threshold_px"],
                count_numeric_values=is_numeric_value_axis(row["grid_type"], "x"),
            )
            y_metrics = tick_metrics(
                gt,
                pred,
                "y",
                row["pixel_error_threshold_px"],
                count_numeric_values=is_numeric_value_axis(row["grid_type"], "y"),
            )
            row["legend_color_threshold"] = color_threshold_for_type(
                row["gt_type"],
                args.color_threshold,
                color_threshold_overrides,
            )
            for metric in (x_metrics, y_metrics, color_metrics(gt, pred, row["legend_color_threshold"])):
                for key, value in metric.items():
                    if key == "tick_position_max_error_px":
                        if value is not None:
                            row[key] = max(float(row.get(key) or 0.0), float(value))
                    elif isinstance(value, (int, float)):
                        row[key] = row.get(key, 0) + value
            row["gt_config"] = str(gt_path)
        else:
            row["grid_eval_included"] = False
        rows.append(row)

    summary = summarize(rows)
    grid_effect_summary = summarize_grid_effect(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    details_path = args.output / "details.json"
    summary_path = args.output / "summary.json"
    write_json(details_path, rows)
    write_json(summary_path, summary)
    write_json(args.output / "grid_effect_summary.json", grid_effect_summary)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "details.csv", rows)
    write_csv(args.output / "grid_effect_summary.csv", grid_effect_summary)
    write_report(args.output / "report.md", summary, details_path, threshold_label, color_threshold_label, rows)
    write_grid_comparison_report(
        args.output / "grid_effect_comparison_with_grid.md",
        grid_effect_summary,
        args.grid_reference_root,
        threshold_label,
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.output / 'report.md'}")
    print(f"Wrote {args.output / 'grid_effect_comparison_with_grid.md'}")


if __name__ == "__main__":
    main()
