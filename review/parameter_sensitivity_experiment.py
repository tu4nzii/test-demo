"""Offline parameter sensitivity experiment for reviewer responses.

This script evaluates only the low-level candidate generation stage affected by
Gaussian smoothing and Canny/Hough parameters. Ground truth JSON files are read
only for offline scoring; they are not used by the runtime generation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "backend" / "datasets" / "VisHintPrompt_datasets"
FULL_REPORT_GRID_EFFECT_SUMMARY = (
    ROOT / "backend" / "evaluation" / "results" / "vishintprompt_full_latest_report" / "grid_effect_summary.csv"
)

CARTESIAN_FOLDERS = {
    "vbar": "v_bar",
    "hbar": "h_bar",
    "line": "line",
    "scatter": "scatter",
    "bubble": "bubble",
}

GAUSSIAN_SETTINGS = [
    ("none", 0, 0.0),
    ("g3_s0", 3, 0.0),
    ("g5_s1", 5, 1.0),
]

CANNY_SETTINGS = [
    ("canny_20_80", 20, 80),
    ("canny_30_100", 30, 100),
    ("canny_50_150", 50, 150),
    ("canny_70_210", 70, 210),
]

HoughSetting = tuple[str, int, int, int]
HOUGH_BASELINE: HoughSetting = ("hough15_l20_g20", 15, 20, 20)
HOUGH_SENSITIVITY: list[HoughSetting] = [
    ("hough10_l20_g20", 10, 20, 20),
    HOUGH_BASELINE,
    ("hough20_l20_g20", 20, 20, 20),
]


@dataclass(frozen=True)
class Sample:
    dataset: str
    chart_type: str
    config_path: Path
    image_path: Path
    x_pixels: list[float]
    y_pixels: list[float]
    x_ticks: list[object]
    y_ticks: list[object]


def read_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def normalize_folder_type(folder_name: str) -> str | None:
    key = "".join(ch for ch in folder_name.lower() if ch.isalpha())
    for prefix, chart_type in CARTESIAN_FOLDERS.items():
        if key.startswith(prefix):
            return chart_type
    return None


def resolve_image_path(category_dir: Path, config: dict) -> Path | None:
    rel = (config.get("image_paths") or {}).get("no_grid")
    candidates: list[Path] = []
    if rel:
        rel_path = Path(str(rel))
        candidates.extend(
            [
                category_dir / rel_path,
                category_dir / rel_path.name,
                category_dir / "chart" / rel_path.name,
                category_dir / "charts" / rel_path.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if rel:
        matches = sorted(category_dir.rglob(Path(str(rel)).name))
        matches = [p for p in matches if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if matches:
            return matches[0]
    return None


def load_samples(limit_per_type: int | None = None) -> list[Sample]:
    samples: list[Sample] = []
    for dataset_dir in sorted(DATASET_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        per_type_count: dict[str, int] = {}
        for category_dir in sorted(dataset_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name.lower() == "all":
                continue
            chart_type = normalize_folder_type(category_dir.name)
            if chart_type is None:
                continue
            config_dirs = [category_dir / "chart_configs", category_dir / "chart_config"]
            for config_dir in config_dirs:
                if not config_dir.exists():
                    continue
                for config_path in sorted(config_dir.glob("*.json")):
                    if limit_per_type is not None and per_type_count.get(chart_type, 0) >= limit_per_type:
                        continue
                    try:
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    image_path = resolve_image_path(category_dir, config)
                    x_pixels = config.get("x_pixels") or []
                    y_pixels = config.get("y_pixels") or []
                    if image_path is None or not x_pixels or not y_pixels:
                        continue
                    samples.append(
                        Sample(
                            dataset=dataset_dir.name,
                            chart_type=chart_type,
                            config_path=config_path,
                            image_path=image_path,
                            x_pixels=[float(v) for v in x_pixels],
                            y_pixels=[float(v) for v in y_pixels],
                            x_ticks=list(config.get("x_ticks") or []),
                            y_ticks=list(config.get("y_ticks") or []),
                        )
                    )
                    per_type_count[chart_type] = per_type_count.get(chart_type, 0) + 1
    return samples


def is_numeric_tick(value: object) -> bool:
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value).replace(",", "").strip())
        return True
    except Exception:
        return False


def numeric_axis_pixels(sample: Sample) -> tuple[list[float], list[float]]:
    x_numeric = sample.x_pixels if sample.x_ticks and all(is_numeric_tick(v) for v in sample.x_ticks) else []
    y_numeric = sample.y_pixels if sample.y_ticks and all(is_numeric_tick(v) for v in sample.y_ticks) else []
    if sample.chart_type == "v_bar":
        x_numeric = []
    if sample.chart_type == "h_bar":
        y_numeric = []
    return x_numeric, y_numeric


def detect_lines(
    image: np.ndarray,
    gaussian_kernel: int,
    gaussian_sigma: float,
    canny_low: int,
    canny_high: int,
    hough_threshold: int,
    min_line_length: int,
    max_line_gap: int,
) -> list[list[int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gaussian_kernel > 1:
        gray = cv2.GaussianBlur(gray, (gaussian_kernel, gaussian_kernel), gaussian_sigma)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return [line[0].astype(int).tolist() for line in lines]


def line_positions(lines: Iterable[list[int]]) -> tuple[list[float], list[float]]:
    vertical: list[float] = []
    horizontal: list[float] = []
    for x1, y1, x2, y2 in lines:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 10:
            continue
        if abs(dy) <= max(2.0, 0.08 * abs(dx)):
            horizontal.append((y1 + y2) / 2.0)
        elif abs(dx) <= max(2.0, 0.08 * abs(dy)):
            vertical.append((x1 + x2) / 2.0)
    return vertical, horizontal


def nearest_error(positions: list[float], target: float) -> float | None:
    if not positions:
        return None
    return min(abs(pos - target) for pos in positions)


def target_set_error(positions: list[float], targets: list[float]) -> float | None:
    if not positions or not targets:
        return None
    return min(abs(pos - target) for pos in positions for target in targets)


def recall_at(positions: list[float], targets: list[float], tolerance: float) -> tuple[int, int]:
    if not targets:
        return 0, 0
    matched = 0
    for target in targets:
        err = nearest_error(positions, target)
        if err is not None and err <= tolerance:
            matched += 1
    return matched, len(targets)


def mean_coord_error(positions: list[float], targets: list[float]) -> list[float]:
    errors: list[float] = []
    for target in targets:
        err = nearest_error(positions, target)
        if err is not None:
            errors.append(err)
    return errors


def evaluate_setting(samples: list[Sample], setting: dict) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        image = read_image(sample.image_path)
        if image is None:
            continue
        lines = detect_lines(
            image,
            setting["gaussian_kernel"],
            setting["gaussian_sigma"],
            setting["canny_low"],
            setting["canny_high"],
            setting["hough_threshold"],
            setting["min_line_length"],
            setting["max_line_gap"],
        )
        vertical, horizontal = line_positions(lines)
        x_axis_y = max(sample.y_pixels)
        y_axis_x_candidates = [min(sample.x_pixels), max(sample.x_pixels)]
        x_axis_err = nearest_error(horizontal, x_axis_y)
        y_axis_err = target_set_error(vertical, y_axis_x_candidates)
        numeric_x, numeric_y = numeric_axis_pixels(sample)
        x_match3, x_total = recall_at(vertical, numeric_x, 3.0)
        y_match3, y_total = recall_at(horizontal, numeric_y, 3.0)
        x_match5, _ = recall_at(vertical, numeric_x, 5.0)
        y_match5, _ = recall_at(horizontal, numeric_y, 5.0)
        numeric_errors = mean_coord_error(vertical, numeric_x) + mean_coord_error(horizontal, numeric_y)
        row = {
            "setting": setting["name"],
            "gaussian": setting["gaussian"],
            "canny": setting["canny"],
            "hough": setting["hough"],
            "dataset": sample.dataset,
            "chart_type": sample.chart_type,
            "image": str(sample.image_path.relative_to(ROOT)),
            "line_candidates": len(lines),
            "vertical_candidates": len(vertical),
            "horizontal_candidates": len(horizontal),
            "x_axis_error_px": x_axis_err,
            "y_axis_error_px": y_axis_err,
            "x_axis_hit_5px": x_axis_err is not None and x_axis_err <= 5.0,
            "y_axis_hit_5px": y_axis_err is not None and y_axis_err <= 5.0,
            "both_axes_hit_5px": (
                x_axis_err is not None
                and y_axis_err is not None
                and x_axis_err <= 5.0
                and y_axis_err <= 5.0
            ),
            "numeric_tickline_match_3px": x_match3 + y_match3,
            "numeric_tickline_match_5px": x_match5 + y_match5,
            "numeric_tickline_total": x_total + y_total,
            "numeric_coord_mae_px": mean(numeric_errors) if numeric_errors else None,
        }
        rows.append(row)
    return rows


def safe_mean(values: Iterable[float | int | bool | None]) -> float | None:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return None
    return mean(cleaned)


def summarize_rows(rows: list[dict]) -> dict:
    total_numeric = sum(int(row["numeric_tickline_total"]) for row in rows)
    matched3 = sum(int(row["numeric_tickline_match_3px"]) for row in rows)
    matched5 = sum(int(row["numeric_tickline_match_5px"]) for row in rows)
    return {
        "samples": len(rows),
        "avg_line_candidates": safe_mean(row["line_candidates"] for row in rows),
        "x_axis_hit_5px": safe_mean(row["x_axis_hit_5px"] for row in rows),
        "y_axis_hit_5px": safe_mean(row["y_axis_hit_5px"] for row in rows),
        "both_axes_hit_5px": safe_mean(row["both_axes_hit_5px"] for row in rows),
        "numeric_tickline_recall_3px": matched3 / total_numeric if total_numeric else None,
        "numeric_tickline_recall_5px": matched5 / total_numeric if total_numeric else None,
        "numeric_coord_mae_px": safe_mean(row["numeric_coord_mae_px"] for row in rows),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def load_full_pipeline_metrics() -> dict:
    if not FULL_REPORT_GRID_EFFECT_SUMMARY.exists():
        return {}
    try:
        with FULL_REPORT_GRID_EFFECT_SUMMARY.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    overall = next((row for row in rows if row.get("dataset") == "overall"), None)
    if not overall:
        return {}

    def f(value):
        try:
            return float(value)
        except Exception:
            return None

    cart = {
        "samples": int(float(overall.get("sample_count") or 0)),
        "processed": int(float(overall.get("processed_count") or 0)),
        "tick_value_mae_px": f(overall.get("tick_value_mae_px")),
        "tick_value_accuracy_2px": f(overall.get("tick_value_accuracy_2px")),
        "tick_position_mae_px": f(overall.get("tick_position_mae_px")),
        "label_name_accuracy": f(overall.get("label_name_accuracy")),
    }
    return {"cartesian_from_current_report": cart, "source": str(FULL_REPORT_GRID_EFFECT_SUMMARY)}


def write_markdown(path: Path, summary_rows: list[dict], baseline_by_type: list[dict], metadata: dict) -> None:
    baseline = next(row for row in summary_rows if row["setting"] == "g3_s0__canny_30_100__hough15_l20_g20")
    current_runtime_baseline = next(
        row for row in summary_rows if row["setting"] == "none__canny_30_100__hough15_l20_g20"
    )
    baseline_avg = float(current_runtime_baseline["avg_line_candidates"] or 0.0)

    def change_from_baseline(value: float | None) -> str:
        if value is None or baseline_avg <= 0:
            return "-"
        return f"{((float(value) - baseline_avg) / baseline_avg) * 100:+.2f}%"

    def interpretation(setting: str) -> str:
        if setting == "none__canny_30_100__hough15_l20_g20":
            return "Legacy Canny/Hough diagnostic baseline."
        if "hough10" in setting:
            return "Substantially more candidates in the legacy detector."
        if "hough20" in setting:
            return "Substantially fewer candidates in the legacy detector."
        if setting.startswith("g3_s0") or setting.startswith("g5_s1"):
            return "More candidates than the no-blur legacy baseline."
        if "70_210" in setting:
            return "Fewer candidates than the legacy baseline."
        return "Similar candidate volume to the selected baseline."

    lines = [
        "# Parameter Sensitivity Experiment",
        "",
        "This folder contains an offline reproducibility experiment for the reviewer response.",
        "Ground truth JSON files are used only for scoring in this experiment; they are not used by the runtime generation pipeline.",
        "",
        "Scope note: this experiment isolates the legacy low-level Canny/Hough candidate generator. It is not the active Cartesian runtime path and is not an end-to-end rerun of the current Cartesian grid reconstruction pipeline. The current runtime pipeline uses enhanced-grid-first mask reconstruction, constructs three grid candidates (`combined_mask`, `tick_supplement`, and `semantic_guide`), applies score-based selection and exit checks, and writes `final_bindings`; the full-pipeline metrics below come from that latest pipeline report.",
        "",
        "## Dataset",
        "",
        f"- Dataset root: `{metadata['dataset_root']}`",
        f"- Cartesian samples evaluated: {metadata['sample_count']}",
        f"- Types: {', '.join(metadata['chart_types'])}",
        "",
        "## Active Cartesian Runtime Parameters",
        "",
        "The current system path calls `_process_chart_with_enhanced_grid_only` and `grid_line_filter.process_image`. Its fixed parameters are:",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| Neutral grid mask | saturation <= 70; gray range [95, 255]; local contrast >= 7 |",
        "| Optional dark candidates | disabled by default; dark cutoff 80 when enabled |",
        "| Morphological line length | min_line_frac 0.055 of image width/height; lower bound 15 px |",
        "| Gap closing | gap_frac 0.006 of image width/height; lower bound 3 px |",
        "| Component thickness filter | max_thickness_frac 0.008 of shorter side; lower bound 3 px |",
        "| Grid geometry reconstruction | min_grid_span_frac 0.18; min_grid_lines 2; cluster_tolerance 3 px; grid_thickness 1 px |",
        "| Tick supplement from dark axis/tick evidence | tick_dark_cutoff 150 |",
        "| OCR filtering | ocr_min_score 0.45; det_thresh 0.35; det_box_thresh 0.60; det_unclip_ratio 1.15; det_limit_side_len 960 |",
        "",
        "## Legacy Canny/Hough Diagnostic Settings",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        "| Gaussian smoothing before Canny | none for the legacy line detector; `(3,3), sigma=0` only for local OCR crop thresholding |",
        "| Gaussian settings tested in the sweep | none; `(3,3), sigma=0`; `(5,5), sigma=1` |",
        "| Canny thresholds | 30 / 100 |",
        "| Probabilistic Hough threshold | 15 |",
        "| Hough min line length / max gap | 20 px / 20 px |",
        "| Legacy tick scan range | 20 px |",
        "| Legacy tick merge angle tolerance | 10 degrees |",
        "",
        "## Main Result",
        "",
        "This reviewer-facing summary reports candidate-volume stability. Internal diagnostic columns such as candidate hits near GT axis positions are retained only in `parameter_sensitivity_samples.csv` and `parameter_sensitivity_summary.csv`.",
        "",
        "| Setting | Avg Hough line candidates | Change from baseline | Interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {setting} | {avg} | {change} | {interpretation} |".format(
                setting=row["setting"],
                avg=num(row["avg_line_candidates"]),
                change=change_from_baseline(row.get("avg_line_candidates")),
                interpretation=interpretation(str(row["setting"])),
            )
        )
    lines.extend(
        [
            "",
            "## Baseline By Chart Type",
            "",
            "| Type | Samples | Avg Hough line candidates |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in baseline_by_type:
        lines.append(
            "| {chart_type} | {samples} | {avg} |".format(
                chart_type=row["chart_type"],
                samples=row["samples"],
                avg=num(row["avg_line_candidates"]),
            )
        )
    full_metrics = metadata.get("full_pipeline_metrics") or {}
    cart = full_metrics.get("cartesian_from_current_report")
    if cart:
        lines.extend(
            [
                "",
                "## Current Full-Pipeline Reference",
                "",
                "The latest full-pipeline report is included as the end-to-end Cartesian evidence. It evaluates the active enhanced-grid-first three-candidate scoring/exit pipeline and uses generated `final_bindings`.",
                f"- Cartesian samples: {cart['samples']}",
                f"- Cartesian processed samples: {cart['processed']}",
                f"- Cartesian Tick MAE: {num(cart.get('tick_value_mae_px'))} px",
                f"- Cartesian Tick Acc@2px: {pct(cart.get('tick_value_accuracy_2px'))}",
                f"- Cartesian Tick position MAE: {num(cart.get('tick_position_mae_px'))} px",
                f"- Cartesian label accuracy: {pct(cart.get('label_name_accuracy'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `parameter_sensitivity_samples.csv`: per-sample, per-setting measurements.",
            "- `parameter_sensitivity_summary.csv`: aggregate measurements for each parameter setting.",
            "- `parameter_sensitivity_baseline_by_type.csv`: baseline results by chart type.",
            "- `parameter_sensitivity_summary.json`: machine-readable metadata and summary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "review"))
    parser.add_argument("--limit-per-type", type=int, default=0, help="0 means all available samples.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limit = None if args.limit_per_type <= 0 else args.limit_per_type
    samples = load_samples(limit_per_type=limit)
    if not samples:
        raise SystemExit("No cartesian samples found.")

    settings: list[dict] = []
    for gaussian_name, kernel, sigma in GAUSSIAN_SETTINGS:
        for canny_name, low, high in CANNY_SETTINGS:
            settings.append(
                {
                    "name": f"{gaussian_name}__{canny_name}__{HOUGH_BASELINE[0]}",
                    "gaussian": gaussian_name,
                    "gaussian_kernel": kernel,
                    "gaussian_sigma": sigma,
                    "canny": canny_name,
                    "canny_low": low,
                    "canny_high": high,
                    "hough": HOUGH_BASELINE[0],
                    "hough_threshold": HOUGH_BASELINE[1],
                    "min_line_length": HOUGH_BASELINE[2],
                    "max_line_gap": HOUGH_BASELINE[3],
                }
            )
    for hough_name, threshold, min_len, max_gap in HOUGH_SENSITIVITY:
        if hough_name == HOUGH_BASELINE[0]:
            continue
        settings.append(
            {
                "name": f"none__canny_30_100__{hough_name}",
                "gaussian": "none",
                "gaussian_kernel": 0,
                "gaussian_sigma": 0.0,
                "canny": "canny_30_100",
                "canny_low": 30,
                "canny_high": 100,
                "hough": hough_name,
                "hough_threshold": threshold,
                "min_line_length": min_len,
                "max_line_gap": max_gap,
            }
        )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for setting in settings:
        rows = evaluate_setting(samples, setting)
        all_rows.extend(rows)
        summary = summarize_rows(rows)
        summary_rows.append({"setting": setting["name"], **summary})

    baseline_name = "none__canny_30_100__hough15_l20_g20"
    baseline_rows = [row for row in all_rows if row["setting"] == baseline_name]
    baseline_by_type: list[dict] = []
    for chart_type in sorted({sample.chart_type for sample in samples}):
        type_rows = [row for row in baseline_rows if row["chart_type"] == chart_type]
        baseline_by_type.append({"chart_type": chart_type, **summarize_rows(type_rows)})

    write_csv(
        output_dir / "parameter_sensitivity_samples.csv",
        all_rows,
        [
            "setting",
            "gaussian",
            "canny",
            "hough",
            "dataset",
            "chart_type",
            "image",
            "line_candidates",
            "vertical_candidates",
            "horizontal_candidates",
            "x_axis_error_px",
            "y_axis_error_px",
            "x_axis_hit_5px",
            "y_axis_hit_5px",
            "both_axes_hit_5px",
            "numeric_tickline_match_3px",
            "numeric_tickline_match_5px",
            "numeric_tickline_total",
            "numeric_coord_mae_px",
        ],
    )
    write_csv(
        output_dir / "parameter_sensitivity_summary.csv",
        summary_rows,
        [
            "setting",
            "samples",
            "avg_line_candidates",
            "x_axis_hit_5px",
            "y_axis_hit_5px",
            "both_axes_hit_5px",
            "numeric_tickline_recall_3px",
            "numeric_tickline_recall_5px",
            "numeric_coord_mae_px",
        ],
    )
    write_csv(
        output_dir / "parameter_sensitivity_baseline_by_type.csv",
        baseline_by_type,
        [
            "chart_type",
            "samples",
            "avg_line_candidates",
            "x_axis_hit_5px",
            "y_axis_hit_5px",
            "both_axes_hit_5px",
            "numeric_tickline_recall_3px",
            "numeric_tickline_recall_5px",
            "numeric_coord_mae_px",
        ],
    )

    metadata = {
        "dataset_root": str(DATASET_ROOT),
        "sample_count": len(samples),
        "chart_types": sorted({sample.chart_type for sample in samples}),
        "settings": settings,
        "summary": summary_rows,
        "baseline_by_type": baseline_by_type,
        "full_pipeline_metrics": load_full_pipeline_metrics(),
        "note": "GT is used only for offline scoring in this review experiment.",
    }
    (output_dir / "parameter_sensitivity_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(output_dir / "parameter_sensitivity_report.md", summary_rows, baseline_by_type, metadata)

    print(json.dumps({"samples": len(samples), "settings": len(settings), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
