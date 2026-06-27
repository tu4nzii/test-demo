"""
Evaluate radar axis label detection — full pipeline for real & synthetic charts.

Workflow:
  1. Load GT (center, r_pixels, theta_ticks, theta_angles) from JSON.
  2. Call detect_radar_axes.detect() — runs OCR pipeline with internal F6 fallback.
  3. If detection signals unreliability → record as "fallback".
  4. Otherwise evaluate detected labels against GT.
  5. Report per-chart + summary in CSV, JSON, and Markdown.

Usage:
  python evaluate_radar.py --dataset all
  python evaluate_radar.py --dataset real --only RadarChart20
  python evaluate_radar.py --dataset synthetic --limit 10
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from demo_radar.detect_radar_axes import (
    detect, evaluate, labels_match, angle_distance, compact_text,
)


REAL_RADAR_DIR = BACKEND / "real" / "RadarChart-18 & RoseChart-6" / "RadarChart-18-final"
SYNTH_RADAR_DIR = BACKEND / "real" / "radar"
OUTPUT_DIR = ROOT / "data" / "output" / "radar_axes_eval"
POLYGON_REAL_NUMBERS = {1, 5, 6, 8, 16, 17, 18, 23}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvalRow:
    dataset: str
    chart_id: str
    image_path: str
    json_path: str
    fallback: bool
    fallback_reason: str
    n_axes_gt: int
    n_axes_pred: int | None
    correct: int
    total: int
    accuracy: float | None
    median_score: float | None
    negative_rate: float | None
    numeric_axis_mode: bool | None
    n_source: str
    pred_labels: dict = field(default_factory=dict)
    mismatches: list = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is {type(value).__name__}, expected object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def parse_json_field(value: Any) -> Any:
    """Parse a field that may be a JSON-encoded string (synthetic dataset)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def chart_number(path: Path, data: dict[str, Any] | None = None) -> int | None:
    candidates = [path.stem]
    if data and data.get("chart_id") is not None:
        candidates.append(str(data["chart_id"]))
    for candidate in candidates:
        digits = "".join(ch for ch in candidate if ch.isdigit())
        if digits:
            return int(digits)
    return None


def resolve_image(json_path: Path, data: dict[str, Any]) -> Path | None:
    image_paths = data.get("image_paths") if isinstance(data.get("image_paths"), dict) else {}
    candidates: list[Path] = []
    for key in ("no_grid", "image", "with_grid"):
        value = image_paths.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            candidates.append(path if path.is_absolute() else json_path.parent / path)
    for suffix in (".png", ".jpg", ".jpeg"):
        candidates.append(json_path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def iter_dataset_jsons(dataset: str) -> list[Path]:
    if dataset == "real":
        return sorted(REAL_RADAR_DIR.glob("RadarChart*.json"),
                      key=lambda p: chart_number(p) or 0)
    paths = []
    for path in sorted(SYNTH_RADAR_DIR.glob("radar_*.json")):
        if path.stem.endswith("_attributes"):
            continue
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_one(json_path: Path, dataset: str, args: argparse.Namespace) -> EvalRow:
    chart_id = json_path.stem
    data: dict[str, Any] = {}
    image_path_str = ""

    pred_labels: dict = {}
    mismatches: list = []
    n_axes_pred = None
    correct = 0
    total = 0
    accuracy = None
    median_score = None
    negative_rate = None
    numeric_axis_mode = None
    n_source = ""
    notes: list[str] = []

    try:
        data = read_json(json_path)
        chart_id = str(data.get("chart_id") or json_path.stem)
        image = resolve_image(json_path, data)
        if image is None:
            raise ValueError("image_not_found")
        image_path_str = str(image)

        # Parse fields (synthetic dataset stores lists as JSON strings)
        center_raw = parse_json_field(data.get("center"))
        r_pixels_raw = parse_json_field(data.get("r_pixels"))
        gt_labels_raw = parse_json_field(data.get("theta_ticks", data.get("labels", [])))
        gt_angles_raw = parse_json_field(data.get("theta_angles", []))

        center = tuple(float(v) for v in center_raw) if (
            isinstance(center_raw, (list, tuple)) and len(center_raw) >= 2
        ) else None
        r_pixels = [float(v) for v in r_pixels_raw] if isinstance(r_pixels_raw, list) else []
        gt_labels = [str(v) for v in gt_labels_raw] if isinstance(gt_labels_raw, list) else []
        gt_angles = [float(v) for v in gt_angles_raw] if isinstance(gt_angles_raw, list) else []

        total = len(gt_angles)

        if center is None or len(r_pixels) < 1 or not gt_angles:
            raise ValueError("missing_groundtruth_metadata")

        # ── Polygon radar exclusion ──
        if dataset == "real" and chart_number(json_path, data) in POLYGON_REAL_NUMBERS:
            return EvalRow(
                dataset=dataset, chart_id=chart_id,
                image_path=image_path_str, json_path=str(json_path.resolve()),
                fallback=True, fallback_reason="polygon_radar_excluded",
                n_axes_gt=total, n_axes_pred=None,
                correct=0, total=total, accuracy=None,
                median_score=None, negative_rate=None,
                numeric_axis_mode=None, n_source="",
                pred_labels={}, mismatches=[], notes="known polygon radar",
            )

        outer_r = float(r_pixels[-1])

        # ── Run axis label detection ──
        with contextlib.redirect_stdout(io.StringIO()):
            axis_labels, debug = detect(str(image), center, outer_r, use_llm=True)

        n_axes_pred = debug.get("n_final")
        n_source = str(debug.get("n_source", ""))
        numeric_axis_mode = bool(debug.get("numeric_axis_mode"))

        # ── Score statistics ──
        scores = [a["score"] for a in debug.get("assignments", [])
                  if isinstance(a.get("score"), (int, float))]
        if scores:
            median_score = float(sorted(scores)[len(scores) // 2])
            negative_rate = sum(1 for s in scores if s < 0) / len(scores)

        pred_labels = {str(k): v for k, v in sorted(axis_labels.items())}

        # ── F6 fallback check ──
        if debug.get("fallback"):
            return EvalRow(
                dataset=dataset, chart_id=chart_id,
                image_path=image_path_str, json_path=str(json_path.resolve()),
                fallback=True,
                fallback_reason=debug.get("fallback_reason", "f6_unreliable"),
                n_axes_gt=total, n_axes_pred=n_axes_pred,
                correct=0, total=total, accuracy=None,
                median_score=median_score, negative_rate=negative_rate,
                numeric_axis_mode=numeric_axis_mode,
                n_source=n_source,
                pred_labels=pred_labels, mismatches=[], notes="",
            )

        # ── No labels detected ──
        if not axis_labels:
            return EvalRow(
                dataset=dataset, chart_id=chart_id,
                image_path=image_path_str, json_path=str(json_path.resolve()),
                fallback=True, fallback_reason="no_labels_detected",
                n_axes_gt=total, n_axes_pred=n_axes_pred,
                correct=0, total=total, accuracy=None,
                median_score=median_score, negative_rate=negative_rate,
                numeric_axis_mode=numeric_axis_mode,
                n_source=n_source,
                pred_labels={}, mismatches=[], notes="",
            )

        # ── Evaluate against GT ──
        correct, _, details = evaluate(axis_labels, gt_labels, gt_angles)
        accuracy = round(100 * correct / max(total, 1), 1)

        mismatches = [
            {"angle": a, "gt": g, "detected": d, "status": s}
            for a, g, d, s in details
            if s not in {"exact", "normalized", "fuzzy", "fuzzy_ce", "fuzzy_sub"}
        ]

        return EvalRow(
            dataset=dataset, chart_id=chart_id,
            image_path=image_path_str, json_path=str(json_path.resolve()),
            fallback=False, fallback_reason="",
            n_axes_gt=total, n_axes_pred=n_axes_pred,
            correct=correct, total=total, accuracy=accuracy,
            median_score=median_score, negative_rate=negative_rate,
            numeric_axis_mode=numeric_axis_mode,
            n_source=n_source,
            pred_labels=pred_labels, mismatches=mismatches,
            notes=";".join(notes),
        )

    except Exception as exc:
        return EvalRow(
            dataset=dataset, chart_id=chart_id,
            image_path=image_path_str, json_path=str(json_path.resolve()),
            fallback=True,
            fallback_reason=f"exception:{type(exc).__name__}:{exc}",
            n_axes_gt=total, n_axes_pred=None,
            correct=0, total=total, accuracy=None,
            median_score=None, negative_rate=None,
            numeric_axis_mode=None, n_source="",
            pred_labels={}, mismatches=[], notes="",
        )


# ---------------------------------------------------------------------------
# Summary & Output
# ---------------------------------------------------------------------------

def summarize(rows: list[EvalRow]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset in sorted({row.dataset for row in rows}):
        group = [row for row in rows if row.dataset == dataset]
        fallback = [row for row in group if row.fallback]
        evaluated = [row for row in group if not row.fallback]

        def values(field: str) -> list[float]:
            out = []
            for row in evaluated:
                value = getattr(row, field)
                if isinstance(value, (int, float)) and value is not None:
                    out.append(float(value))
            return out

        def stats(field: str) -> dict[str, float | None]:
            vals = values(field)
            if not vals:
                return {"mean": None, "median": None, "max": None}
            return {
                "mean": round(float(np.mean(vals)), 2),
                "median": round(float(np.median(vals)), 2),
                "max": round(float(np.max(vals)), 2),
            }

        reasons: dict[str, int] = {}
        for row in fallback:
            reasons[row.fallback_reason] = reasons.get(row.fallback_reason, 0) + 1

        total_c = sum(r.correct for r in evaluated)
        total_t = sum(r.total for r in evaluated)

        summary[dataset] = {
            "total_charts": len(group),
            "fallback_count": len(fallback),
            "fallback_rate": round(len(fallback) / len(group), 4) if group else 0.0,
            "evaluated_count": len(evaluated),
            "total_axes_evaluated": total_t,
            "total_axes_correct": total_c,
            "accuracy": round(100 * total_c / max(total_t, 1), 1),
            "accuracy_per_chart": stats("accuracy"),
            "fallback_reasons": reasons,
        }
    return summary


def write_csv(path: Path, rows: list[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    skip_fields = {"pred_labels", "mismatches"}
    fieldnames = [f for f in EvalRow.__dataclass_fields__ if f not in skip_fields]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            for f in skip_fields:
                d.pop(f, None)
            writer.writerow(d)


def write_markdown(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Radar Axis Label Detection — Full Evaluation",
        "",
        "## Fallback Mechanism (F6)",
        "",
        "A chart is routed to fallback and excluded when any of the following holds:",
        "",
        "1. Known polygon radar chart in the real set: `1, 5, 6, 8, 16, 17, 18, 23`.",
        "2. Required metadata (`center`, `r_pixels`, `theta_ticks`, `theta_angles`) is missing.",
        "3. Detection pipeline returns no usable axis labels.",
        "4. **F6 unreliability gate**: `numeric_axis_mode=True` AND (median score < 0 OR >25% negative scores) AND LLM unavailable.",
        "",
        "F6 uses generation-side evidence only — no GT.",
        "",
        "## Core Detection Mechanisms",
        "",
        "- Canonical axis count correction (off-by-one OCR counts)",
        "- Substring match tolerance (≥4-char substrings like `amine`⊆`ketamine`)",
        "- Footnote/legend text penalty (`(mean`, `(median`, etc.)",
        "- Relaxed perpendicular fallback for orphan axes",
        "- Adjacent axis swap optimisation (local angular error minimisation)",
        "",
        "## Summary",
        "",
    ]
    for dataset, item in summary.items():
        lines.extend([
            f"### {dataset}",
            "",
            f"- Total charts: {item['total_charts']}",
            f"- Fallback count/rate: {item['fallback_count']} / {item['fallback_rate']:.2%}",
            f"- Evaluated charts: {item['evaluated_count']}",
            f"- Total axes evaluated: {item['total_axes_evaluated']}",
            f"- Total axes correct: {item['total_axes_correct']}",
            f"- **Accuracy: {item['accuracy']:.1f}%**",
            f"- Per-chart accuracy: {item['accuracy_per_chart']}",
            "",
            "Fallback reasons:",
        ])
        if item["fallback_reasons"]:
            for reason, count in sorted(item["fallback_reasons"].items()):
                lines.append(f"- `{reason}`: {count}")
        else:
            lines.append("- None")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["real", "synthetic", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit charts per dataset.")
    parser.add_argument("--only", type=str, help="Filter by chart id/stem substring.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    configure_stdio()
    args = parse_args()
    selected = ["real", "synthetic"] if args.dataset == "all" else [args.dataset]
    rows: list[EvalRow] = []

    for dataset in selected:
        paths = iter_dataset_jsons(dataset)
        if args.only:
            needle = args.only.lower()
            paths = [p for p in paths if needle in p.stem.lower()]
        if args.limit:
            paths = paths[:args.limit]
        print(f"[{dataset}] evaluating {len(paths)} charts")

        for index, path in enumerate(paths, start=1):
            row = evaluate_one(path, dataset, args)
            rows.append(row)

            if row.fallback:
                tag = f"FALLBACK ({row.fallback_reason[:50]})"
            else:
                tag = f"{row.correct:2d}/{row.total:2d} ({row.accuracy:.0f}%)"

            print(f"  [{index:3d}/{len(paths)}] {row.chart_id:25s} {tag}")

            if not row.fallback and row.mismatches:
                for m in row.mismatches[:3]:
                    print(f"         {m['angle']:6.1f}deg: GT='{m['gt']}' -> '{m['detected']}' ({m['status']})")
                if len(row.mismatches) > 3:
                    print(f"         ... and {len(row.mismatches) - 3} more mismatches")

    # ── Summarise ──
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name_part = args.dataset

    csv_path = args.output_dir / f"radar_axes_eval_{name_part}.csv"
    json_path = args.output_dir / f"radar_axes_eval_{name_part}.json"
    md_path = args.output_dir / f"radar_axes_eval_{name_part}.md"

    write_csv(csv_path, rows)
    write_json(json_path, {"summary": summary, "rows": [asdict(row) for row in rows]})
    write_markdown(md_path, summary, args)

    # ── Console summary ──
    print(f"\n{'=' * 70}")
    for dataset, item in summary.items():
        print(f"\n[{dataset}]")
        print(f"  Charts evaluated: {item['evaluated_count']}/{item['total_charts']}")
        print(f"  Fallback:         {item['fallback_count']} ({item['fallback_rate']:.1%})")
        print(f"  Accuracy:         {item['total_axes_correct']}/{item['total_axes_evaluated']} "
              f"({item['accuracy']:.1f}%)")
        print(f"  Per-chart acc:    mean={item['accuracy_per_chart'].get('mean','?')}%, "
              f"median={item['accuracy_per_chart'].get('median','?')}%")
        if item["fallback_reasons"]:
            for reason, count in sorted(item["fallback_reasons"].items()):
                print(f"    {reason}: {count}")

    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
