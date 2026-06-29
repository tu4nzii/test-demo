"""Run real radar value evaluation from prepared GT-encrypted JSON files.

This is a thin wrapper around ``backend/polar/radar/demo_evaluation_radar_1 copy.py``.
It exists because that evaluator expects image paths to be relative to the
current working directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_JSON_DIR = ROOT_DIR / "backend" / "data" / "polar" / "real_evaluation_data" / "radar"
DEFAULT_EVALUATOR = ROOT_DIR / "backend" / "polar" / "radar" / "demo_evaluation_radar_1 copy.py"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_evaluator_class(evaluator_path: Path):
    spec = importlib.util.spec_from_file_location("real_radar_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import evaluator: {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RadarChartEvaluator


def discover_chart_jsons(json_dir: Path, chart_filters: list[str] | None = None) -> list[Path]:
    filters = {value.lower() for value in chart_filters or []}
    chart_paths: list[Path] = []
    for path in sorted(json_dir.glob("*.json")):
        data = load_json(path)
        if data.get("chart_type") != "radar":
            continue
        chart_id = str(data.get("chart_id", ""))
        names = {path.stem.lower(), chart_id.lower()}
        if filters and not (names & filters):
            continue
        chart_paths.append(path)
    return chart_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prepared real radar value evaluation.")
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--charts", nargs="*", help="Optional chart stems or chart_id values.")
    parser.add_argument("--max-points", type=int, default=0, help="Limit data points per chart for smoke tests.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <json-dir>/real_radar_value_results.json.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list charts, do not call the model.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    json_dir = args.json_dir.resolve()
    evaluator_path = args.evaluator.resolve()
    output_path = (args.output or (json_dir / "real_radar_value_results.json")).resolve()

    chart_paths = discover_chart_jsons(json_dir, args.charts)
    if not chart_paths:
        print(f"[error] No radar chart JSON found in {json_dir}")
        return 1

    print(f"[charts] {len(chart_paths)}")
    for path in chart_paths:
        data = load_json(path)
        print(f"  - {path.name} ({data.get('chart_id', path.stem)})")
    if args.dry_run:
        return 0

    if args.max_points > 0:
        os.environ["REAL_POLAR_EVAL_MAX_POINTS"] = str(args.max_points)
        print(f"[max-points] {args.max_points} per chart")
    else:
        os.environ.pop("REAL_POLAR_EVAL_MAX_POINTS", None)

    evaluator_cls = load_evaluator_class(evaluator_path)
    evaluator = evaluator_cls()

    previous_cwd = Path.cwd()
    try:
        os.chdir(json_dir)
        for index, path in enumerate(chart_paths, start=1):
            print(f"\n[{index}/{len(chart_paths)}] running {path.name}")
            evaluator.process_single_image(path.name)
    finally:
        os.chdir(previous_cwd)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(str(output_path))
    print(f"[done] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
