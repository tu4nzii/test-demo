"""Run one polar chart through the simple end-to-end pipeline.

Pipeline:
  input JSON/image
  -> axis/circle detection with runtime fallback
  -> encrypted grid JSON/PNG for radar/rose if detection is retained
  -> optional radar value evaluation

This script is intentionally single-chart first.  Batch scripts remain useful
for paper tables, but this is the readable debugging path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
POLAR_DATA = BACKEND / "data" / "polar"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.polar.encryption.prepare_real_evaluation_gt_encryption import write_dataset_item  # noqa: E402
from backend.polar.evaluation import evaluate_pie_donut_circle_extraction as circle_eval  # noqa: E402
from backend.polar.evaluation import evaluate_radar_grid_extraction as radar_eval  # noqa: E402
from backend.polar.evaluation import evaluate_rose_grid_extraction as rose_eval  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def resolve_chart_image(json_path: Path, explicit_image: Path | None = None) -> Path:
    if explicit_image is not None:
        return explicit_image.resolve()

    data = read_json(json_path)
    direct = data.get("image")
    if direct:
        candidate = Path(str(direct))
        if candidate.exists():
            return candidate.resolve()
        candidate = json_path.parent / str(direct)
        if candidate.exists():
            return candidate.resolve()

    image_paths = data.get("image_paths") or {}
    for key in ("no_grid", "with_grid", "grid_with_grid"):
        rel = image_paths.get(key)
        if not rel:
            continue
        candidate = json_path.parent / str(rel)
        if candidate.exists():
            return candidate.resolve()

    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate.resolve()
        candidate = json_path.with_name(f"{json_path.stem}_no_grid{suffix}")
        if candidate.exists():
            return candidate.resolve()
        candidate = json_path.with_name(f"{json_path.stem}_with_grid{suffix}")
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"image not found for {json_path}")


def default_output_dir(chart_type: str, json_path: Path) -> Path:
    return POLAR_DATA / "single_runs" / chart_type / json_path.stem


def eval_args(output_dir: Path, tick_mode: str, tolerance_ratio: float, min_axis_clusters: int) -> argparse.Namespace:
    return SimpleNamespace(
        tick_mode=tick_mode,
        tolerance_ratio=tolerance_ratio,
        min_axis_clusters=min_axis_clusters,
        output_dir=output_dir,
        synth_dir=None,
    )


def run_radar_or_rose(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    json_path = args.json.resolve()
    image_path = resolve_chart_image(json_path, args.image)
    axis_dir = run_dir / "01_axis_detection"
    encrypt_dir = run_dir / "02_encrypted"

    namespace = eval_args(axis_dir, args.tick_mode, args.tolerance_ratio, args.min_axis_clusters)
    if args.chart_type == "radar":
        row = radar_eval.evaluate_one(json_path, args.dataset, namespace)
    else:
        row = rose_eval.evaluate_one(json_path, args.dataset, namespace)

    row_data = asdict(row)
    write_json(axis_dir / "axis_eval.json", row_data)

    summary: dict[str, Any] = {
        "chart_type": args.chart_type,
        "input_json": str(json_path),
        "input_image": str(image_path),
        "run_dir": str(run_dir),
        "fallback": row.fallback,
        "fallback_reason": row.fallback_reason,
        "axis_eval": str(axis_dir / "axis_eval.json"),
        "encrypted_json": None,
        "encrypted_image": None,
        "value_eval": None,
    }

    if row.fallback:
        write_json(run_dir / "pipeline_summary.json", summary)
        return summary

    encrypted = write_dataset_item(
        source_json=json_path,
        source_image=image_path,
        out_dir=encrypt_dir,
        chart_type=args.chart_type,
        output_stem=json_path.stem,
    )
    write_json(encrypt_dir / "encryption_summary.json", encrypted)
    summary["encrypted_json"] = encrypted["output_json"]
    summary["encrypted_image"] = encrypted["output_grid"]

    if args.chart_type == "radar" and args.run_value_eval:
        output_path = run_dir / "03_value_eval" / f"{json_path.stem}_value_eval.json"
        command = [
            sys.executable,
            "backend/polar/value_eval/run_real_radar_value_evaluation.py",
            "--json-dir",
            str(encrypt_dir),
            "--charts",
            json_path.stem,
            "--output",
            str(output_path),
        ]
        if args.max_points > 0:
            command.extend(["--max-points", str(args.max_points)])
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.check_call(command, cwd=ROOT, env=env)
        summary["value_eval"] = str(output_path)

    write_json(run_dir / "pipeline_summary.json", summary)
    return summary


def run_pie_or_donut(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    json_path = args.json.resolve()
    image_path = resolve_chart_image(json_path, args.image)
    eval_dir = run_dir / "01_circle_detection"
    row = circle_eval.evaluate_one(
        json_path=json_path,
        chart_type=args.chart_type,
        dataset=args.dataset,
        output_dir=eval_dir,
        tolerance_ratio=args.tolerance_ratio,
    )
    row_data = asdict(row)
    write_json(eval_dir / "circle_eval.json", row_data)
    summary = {
        "chart_type": args.chart_type,
        "input_json": str(json_path),
        "input_image": str(image_path),
        "run_dir": str(run_dir),
        "fallback": row.fallback,
        "fallback_reason": row.fallback_reason,
        "circle_eval": str(eval_dir / "circle_eval.json"),
        "detection_preview_dir": str(eval_dir / args.chart_type / args.dataset / "detections"),
        "encrypted_json": None,
        "encrypted_image": None,
        "notes": "pie/donut use circle-prior evaluation; no polar grid encryption stage is required.",
    }
    write_json(run_dir / "pipeline_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart-type", choices=["radar", "rose", "pie", "donut"], required=True)
    parser.add_argument("--json", type=Path, required=True, help="Input chart JSON.")
    parser.add_argument("--image", type=Path, help="Optional input image. If omitted, it is resolved from JSON.")
    parser.add_argument("--dataset", default="real", help="Dataset label used by fallback rules and output paths.")
    parser.add_argument("--output-dir", type=Path, help="Run output directory.")
    parser.add_argument("--tick-mode", choices=["gt-nearest", "algorithm"], default="gt-nearest")
    parser.add_argument("--tolerance-ratio", type=float, default=0.05)
    parser.add_argument("--min-axis-clusters", type=int, default=2)
    parser.add_argument("--run-value-eval", action="store_true", help="Only for radar; may call the LLM API.")
    parser.add_argument("--max-points", type=int, default=0, help="Optional value-eval smoke-test limit.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    run_dir = (args.output_dir or default_output_dir(args.chart_type, args.json)).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.chart_type in {"radar", "rose"}:
        summary = run_radar_or_rose(args, run_dir)
    else:
        summary = run_pie_or_donut(args, run_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
