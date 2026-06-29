"""Evaluate pie and donut circle extraction with no-GT fallback gates.

Pie metrics:
  - center error
  - outer radius error

Donut metrics:
  - center error
  - inner radius error
  - outer radius error

Fallback decisions are made before metric scoring and do not inspect GT
errors.  They are based on image readability, detector failures, tiny chart
size/radius, and the donut detector's concentric-ring reliability gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
POLAR_DATA = BACKEND / "data" / "polar"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.polar.pie.demo_pie_circle_find_1 import PieCircleDetector  # noqa: E402
from backend.polar.donut.demo_donut_circle_find_1 import DonutCircleDetector  # noqa: E402


REAL_PIE_DIR = BACKEND / "real" / "PieChart-11 & DonutChart-14" / "PieChart-11-final"
REAL_DONUT_DIR = BACKEND / "real" / "PieChart-11 & DonutChart-14" / "DonutChart-14-final"
SYNTH_PIE_DIR = BACKEND / "real" / "pie"
SYNTH_DONUT_DIR = BACKEND / "real" / "donut"
OUTPUT_DIR = POLAR_DATA / "output" / "pie_donut_circle_eval"
TOLERANCE_RATIO = 0.05


@dataclass
class CircleEvalRow:
    dataset: str
    chart_type: str
    chart_id: str
    image_path: str
    json_path: str
    fallback: bool
    fallback_reason: str
    detection_source: str
    short_side: int
    tolerance_px: float
    tolerance_pass: bool | None
    center_error_px: float | None
    center_error_ratio: float | None
    outer_radius_error_px: float | None
    outer_radius_error_ratio: float | None
    inner_radius_error_px: float | None
    inner_radius_error_ratio: float | None
    pred_center: list[float] | None
    pred_inner_radius: float | None
    pred_outer_radius: float | None
    gt_center: list[float] | None
    gt_inner_radius: float | None
    gt_outer_radius: float | None
    notes: str


def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is {type(value).__name__}, expected object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def imread(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def normalize_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def normalize_center(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        x = normalize_number(value.get("x"))
        y = normalize_number(value.get("y"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x = normalize_number(value[0])
        y = normalize_number(value[1])
    else:
        return None
    if x is None or y is None:
        return None
    return [x, y]


def normalize_r_pixels(value: Any, chart_type: str) -> tuple[float | None, float | None]:
    if chart_type == "pie":
        if isinstance(value, (list, tuple)) and value:
            outer = normalize_number(value[-1])
        else:
            outer = normalize_number(value)
        return None, outer
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        inner = normalize_number(value[0])
        outer = normalize_number(value[1])
        return inner, outer
    return None, None


def natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def resolve_image(json_path: Path, data: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    for suffix in (".png", ".jpg", ".jpeg"):
        candidates.append(json_path.with_suffix(suffix))
        candidates.append(json_path.with_name(f"{json_path.stem}_no_grid{suffix}"))
    image_paths = data.get("image_paths") if isinstance(data.get("image_paths"), dict) else {}
    direct = data.get("image")
    if isinstance(direct, str) and direct:
        path = Path(direct)
        candidates.append(path if path.is_absolute() else json_path.parent / path)
        candidates.append(ROOT / direct)
        candidates.append(BACKEND / direct)
    for key in ("no_grid", "image", "with_grid"):
        value = image_paths.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            candidates.append(path if path.is_absolute() else json_path.parent / path)
            candidates.append(ROOT / value)
            candidates.append(BACKEND / value)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def iter_jsons(chart_type: str, dataset: str) -> list[Path]:
    if chart_type == "pie":
        directory = REAL_PIE_DIR if dataset == "real" else SYNTH_PIE_DIR
    else:
        directory = REAL_DONUT_DIR if dataset == "real" else SYNTH_DONUT_DIR
    return sorted(
        [path for path in directory.glob("*.json") if not path.stem.endswith("_attributes")],
        key=natural_key,
    )


def fallback_row(
    dataset: str,
    chart_type: str,
    chart_id: str,
    image_path: str,
    json_path: Path,
    reason: str,
    short_side: int = 0,
    tolerance_px: float = 0.0,
    gt_center: list[float] | None = None,
    gt_inner: float | None = None,
    gt_outer: float | None = None,
    pred_center: list[float] | None = None,
    pred_inner: float | None = None,
    pred_outer: float | None = None,
    detection_source: str = "",
    notes: str = "",
) -> CircleEvalRow:
    return CircleEvalRow(
        dataset=dataset,
        chart_type=chart_type,
        chart_id=chart_id,
        image_path=image_path,
        json_path=str(json_path.resolve()),
        fallback=True,
        fallback_reason=reason,
        detection_source=detection_source,
        short_side=short_side,
        tolerance_px=round(float(tolerance_px), 4),
        tolerance_pass=None,
        center_error_px=None,
        center_error_ratio=None,
        outer_radius_error_px=None,
        outer_radius_error_ratio=None,
        inner_radius_error_px=None,
        inner_radius_error_ratio=None,
        pred_center=pred_center,
        pred_inner_radius=pred_inner,
        pred_outer_radius=pred_outer,
        gt_center=gt_center,
        gt_inner_radius=gt_inner,
        gt_outer_radius=gt_outer,
        notes=notes,
    )


def tiny_chart_reason(short_side: int, outer_radius: float | None) -> str:
    if short_side < 80:
        return f"circle_quality_failed:tiny_image(short_side={short_side})"
    if outer_radius is not None and outer_radius < max(20.0, short_side * 0.08):
        return f"circle_quality_failed:tiny_outer_radius(r={outer_radius:.1f})"
    return ""


def evaluate_one(
    json_path: Path,
    chart_type: str,
    dataset: str,
    output_dir: Path,
    tolerance_ratio: float,
) -> CircleEvalRow:
    chart_id = json_path.stem
    data: dict[str, Any] = {}
    image_path: Path | None = None
    gt_center = None
    gt_inner = None
    gt_outer = None
    pred_center = None
    pred_inner = None
    pred_outer = None
    short_side = 0
    tolerance_px = 0.0

    try:
        data = read_json(json_path)
        chart_id = str(data.get("chart_id") or json_path.stem)
        gt_center = normalize_center(data.get("center") or data.get("pred_coords"))
        gt_inner, gt_outer = normalize_r_pixels(data.get("r_pixels"), chart_type)
        if gt_center is None or gt_outer is None or (chart_type == "donut" and gt_inner is None):
            return fallback_row(
                dataset, chart_type, chart_id, "", json_path,
                "metadata_failed:missing_center_or_radius",
                gt_center=gt_center, gt_inner=gt_inner, gt_outer=gt_outer,
            )

        image_path = resolve_image(json_path, data)
        if image_path is None:
            return fallback_row(
                dataset, chart_type, chart_id, "", json_path,
                "image_failed:not_found",
                gt_center=gt_center, gt_inner=gt_inner, gt_outer=gt_outer,
            )

        image = imread(image_path)
        if image is None:
            return fallback_row(
                dataset, chart_type, chart_id, str(image_path), json_path,
                "image_failed:unreadable",
                gt_center=gt_center, gt_inner=gt_inner, gt_outer=gt_outer,
            )
        short_side = int(min(image.shape[:2]))
        tolerance_px = short_side * tolerance_ratio

        pre_reason = tiny_chart_reason(short_side, None)
        if pre_reason:
            return fallback_row(
                dataset, chart_type, chart_id, str(image_path), json_path,
                pre_reason,
                short_side=short_side, tolerance_px=tolerance_px,
                gt_center=gt_center, gt_inner=gt_inner, gt_outer=gt_outer,
            )

        run_dir = output_dir / chart_type / dataset / "detections"
        if chart_type == "pie":
            result = PieCircleDetector().process_single_image(str(image_path), str(run_dir))
            pred_center = [float(result["center"]["x"]), float(result["center"]["y"])]
            pred_outer = float(result["radius"])
            detection_source = str(result.get("detection_source") or "")
            edge_support = float(result.get("edge_support", 0.0))
            fill_support = float(result.get("fill_support", 0.0))
            notes = (
                f"edge_support={edge_support:.4f};"
                f"fill_support={fill_support:.4f}"
            )
            if edge_support < 0.45:
                return fallback_row(
                    dataset, chart_type, chart_id, str(image_path), json_path,
                    f"circle_quality_failed:low_pie_edge_support({edge_support:.2f})",
                    short_side=short_side, tolerance_px=tolerance_px,
                    gt_center=gt_center, gt_inner=gt_inner, gt_outer=gt_outer,
                    pred_center=pred_center, pred_inner=pred_inner, pred_outer=pred_outer,
                    detection_source=detection_source, notes=notes,
                )
        else:
            result = DonutCircleDetector().process_single_image(str(image_path), str(run_dir))
            pred_center = [float(result["center"][0]), float(result["center"][1])]
            pred_inner = float(result["inner_radius"])
            pred_outer = float(result["outer_radius"])
            detection_source = str(result.get("detection_source") or "")
            notes = (
                f"inner_transition={float(result.get('inner_transition', 0.0)):.4f};"
                f"outer_transition={float(result.get('outer_transition', 0.0)):.4f};"
                f"outer_radius_consistency={float(result.get('outer_radius_consistency', 0.0)):.4f}"
            )

        post_reason = tiny_chart_reason(short_side, pred_outer)
        if post_reason:
            return fallback_row(
                dataset, chart_type, chart_id, str(image_path), json_path,
                post_reason,
                short_side=short_side, tolerance_px=tolerance_px,
                gt_center=gt_center, gt_inner=gt_inner, gt_outer=gt_outer,
                pred_center=pred_center, pred_inner=pred_inner, pred_outer=pred_outer,
                detection_source=detection_source, notes=notes,
            )

        center_error = math.hypot(pred_center[0] - gt_center[0], pred_center[1] - gt_center[1])
        outer_error = abs(float(pred_outer) - float(gt_outer))
        inner_error = abs(float(pred_inner) - float(gt_inner)) if chart_type == "donut" else None
        all_errors = [center_error, outer_error] + ([inner_error] if inner_error is not None else [])
        tolerance_pass = all(error <= tolerance_px for error in all_errors)

        return CircleEvalRow(
            dataset=dataset,
            chart_type=chart_type,
            chart_id=chart_id,
            image_path=str(image_path),
            json_path=str(json_path.resolve()),
            fallback=False,
            fallback_reason="",
            detection_source=detection_source,
            short_side=short_side,
            tolerance_px=round(float(tolerance_px), 4),
            tolerance_pass=bool(tolerance_pass),
            center_error_px=round(float(center_error), 4),
            center_error_ratio=round(float(center_error / short_side), 6),
            outer_radius_error_px=round(float(outer_error), 4),
            outer_radius_error_ratio=round(float(outer_error / short_side), 6),
            inner_radius_error_px=round(float(inner_error), 4) if inner_error is not None else None,
            inner_radius_error_ratio=round(float(inner_error / short_side), 6) if inner_error is not None else None,
            pred_center=[round(float(pred_center[0]), 4), round(float(pred_center[1]), 4)],
            pred_inner_radius=round(float(pred_inner), 4) if pred_inner is not None else None,
            pred_outer_radius=round(float(pred_outer), 4),
            gt_center=[round(float(gt_center[0]), 4), round(float(gt_center[1]), 4)],
            gt_inner_radius=round(float(gt_inner), 4) if gt_inner is not None else None,
            gt_outer_radius=round(float(gt_outer), 4),
            notes=notes,
        )
    except Exception as error:
        reason = str(error)
        if "exploded" in reason or "common circle" in reason:
            reason = "circle_quality_failed:exploded_or_nonconcentric_ring"
        elif "Unable to find reliable concentric donut boundaries" in reason:
            reason = "circle_quality_failed:no_reliable_donut_boundaries"
        elif "Unable to find a safe pie circle" in reason:
            reason = "circle_quality_failed:no_safe_pie_circle"
        elif "Unable to isolate" in reason:
            reason = "circle_quality_failed:no_colored_region"
        else:
            reason = f"exception:{type(error).__name__}:{reason}"
        return fallback_row(
            dataset, chart_type, chart_id,
            str(image_path) if image_path else "",
            json_path,
            reason,
            short_side=short_side, tolerance_px=tolerance_px,
            gt_center=gt_center, gt_inner=gt_inner, gt_outer=gt_outer,
            pred_center=pred_center, pred_inner=pred_inner, pred_outer=pred_outer,
        )


def mean(values: list[float]) -> float | None:
    return round(float(sum(values) / len(values)), 4) if values else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return round(float(values[mid]), 4)
    return round(float((values[mid - 1] + values[mid]) / 2), 4)


def summarize(rows: list[CircleEvalRow]) -> dict[str, Any]:
    valid = [row for row in rows if not row.fallback]
    fallback = [row for row in rows if row.fallback]
    tolerance_fail = [row for row in valid if row.tolerance_pass is False]
    reasons: dict[str, int] = {}
    for row in fallback:
        reasons[row.fallback_reason] = reasons.get(row.fallback_reason, 0) + 1

    def stats(field: str) -> dict[str, float | None]:
        values = [
            float(getattr(row, field))
            for row in valid
            if getattr(row, field) is not None
        ]
        return {
            "mean": mean(values),
            "median": median(values),
            "max": round(float(max(values)), 4) if values else None,
        }

    return {
        "total": len(rows),
        "fallback_count": len(fallback),
        "fallback_rate": round(len(fallback) / len(rows), 4) if rows else 0.0,
        "success_count": len(valid),
        "tolerance_fail_count": len(tolerance_fail),
        "fallback_reasons": reasons,
        "center_error_px": stats("center_error_px"),
        "outer_radius_error_px": stats("outer_radius_error_px"),
        "inner_radius_error_px": stats("inner_radius_error_px"),
        "max_tolerance_px": round(max((row.tolerance_px for row in rows), default=0.0), 4),
    }


def write_csv(path: Path, rows: list[CircleEvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CircleEvalRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, summaries: dict[str, Any], rows_by_key: dict[str, list[CircleEvalRow]]) -> None:
    lines = [
        "# Pie/Donut Circle Extraction Evaluation",
        "",
        "Fallback 机制：读图/标注缺失、无可靠圆、图表过小、donut 环不共圆或疑似爆炸时，不进入误差统计。",
        "Pie 指标为圆心与外半径；donut 指标为圆心、内半径与外半径。容差为图像短边的 5%。",
        "",
    ]
    for key, summary in summaries.items():
        lines.extend([
            f"## {key}",
            "",
            f"- total: {summary['total']}",
            f"- fallback: {summary['fallback_count']} ({summary['fallback_rate']:.2%})",
            f"- success: {summary['success_count']}",
            f"- tolerance_fail: {summary['tolerance_fail_count']}",
            f"- center_error_px: {summary['center_error_px']}",
            f"- outer_radius_error_px: {summary['outer_radius_error_px']}",
            f"- inner_radius_error_px: {summary['inner_radius_error_px']}",
            "",
        ])
        fallback_rows = [row for row in rows_by_key[key] if row.fallback]
        if fallback_rows:
            lines.append("Fallback 图表：")
            for row in fallback_rows:
                lines.append(f"- {row.chart_id}: {row.fallback_reason}")
            lines.append("")
        failed_rows = [row for row in rows_by_key[key] if row.tolerance_pass is False]
        if failed_rows:
            lines.append("未通过容差图表：")
            for row in failed_rows:
                lines.append(
                    f"- {row.chart_id}: center={row.center_error_px}, "
                    f"inner={row.inner_radius_error_px}, outer={row.outer_radius_error_px}"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_group(chart_type: str, dataset: str, args: argparse.Namespace) -> list[CircleEvalRow]:
    jsons = iter_jsons(chart_type, dataset)
    if args.only:
        jsons = [path for path in jsons if args.only.lower() in path.stem.lower()]
    if args.limit:
        jsons = jsons[: args.limit]
    rows = []
    print(f"[{chart_type}/{dataset}] evaluating {len(jsons)} charts")
    for index, json_path in enumerate(jsons, 1):
        row = evaluate_one(
            json_path,
            chart_type,
            dataset,
            args.output_dir,
            args.tolerance_ratio,
        )
        rows.append(row)
        status = "fallback" if row.fallback else "success"
        detail = row.fallback_reason if row.fallback else f"pass={row.tolerance_pass}"
        print(f"  [{index}/{len(jsons)}] {row.chart_id}: {status} {detail}")
    return rows


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Evaluate pie/donut circle extraction.")
    parser.add_argument("--chart-type", choices=["pie", "donut", "all"], default="all")
    parser.add_argument("--dataset", choices=["real", "synth", "all"], default="real")
    parser.add_argument("--tolerance-ratio", type=float, default=TOLERANCE_RATIO)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    chart_types = ["pie", "donut"] if args.chart_type == "all" else [args.chart_type]
    datasets = ["real", "synth"] if args.dataset == "all" else [args.dataset]

    rows_by_key: dict[str, list[CircleEvalRow]] = {}
    summaries: dict[str, Any] = {}
    for chart_type in chart_types:
        for dataset in datasets:
            key = f"{chart_type}_{dataset}"
            rows = run_group(chart_type, dataset, args)
            rows_by_key[key] = rows
            summaries[key] = summarize(rows)
            write_csv(args.output_dir / f"{key}_circle_eval.csv", rows)
            write_json(args.output_dir / f"{key}_circle_eval.json", {
                "summary": summaries[key],
                "rows": [asdict(row) for row in rows],
            })

    write_json(args.output_dir / "pie_donut_circle_eval_summary.json", summaries)
    write_markdown(args.output_dir / "pie_donut_circle_eval_summary.md", summaries, rows_by_key)
    print("\nSummary:")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"\nOutput: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
