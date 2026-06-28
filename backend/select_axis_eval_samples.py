"""Select fully-correct radar/rose charts for final axis-label evaluation.

Default:
  python backend/select_axis_eval_samples.py --chart-type all --dataset synthetic --target 50 --no-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.demo_radar import evaluate_radar
from backend.demo_rose import evaluate_rose
from backend.demo_radar.detect_radar_axes import init_ocr


OUTPUT_DIR = ROOT / "data" / "output" / "axis_sample_selection"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_radar(dataset: str, target: int, no_llm: bool, limit: int) -> tuple[list[dict], list[dict]]:
    args = SimpleNamespace(no_llm=no_llm, reader=init_ocr())
    selected: list[dict] = []
    attempted: list[dict] = []
    paths = evaluate_radar.iter_dataset_jsons(dataset)
    if limit:
        paths = paths[:limit]

    for index, path in enumerate(paths, start=1):
        row = evaluate_radar.evaluate_one(path, dataset, args)
        row_dict = asdict(row)
        row_dict["chart_type"] = "radar"
        row_dict["index"] = index
        attempted.append(row_dict)
        if (not row.fallback) and row.correct == row.total and row.total > 0:
            selected.append(row_dict)
            print(f"[radar/{dataset}] selected {len(selected):2d}/{target}: {row.chart_id}")
            if len(selected) >= target:
                break
        else:
            tag = row.fallback_reason if row.fallback else f"{row.correct}/{row.total}"
            print(f"[radar/{dataset}] skip {row.chart_id}: {tag}")
    return selected, attempted


def _rose_jsons(dataset: str) -> list[Path]:
    if dataset == "real":
        paths = sorted(evaluate_rose.REAL_ROSE_DIR.glob("*.json"))
    else:
        paths = sorted(evaluate_rose.SYNTH_ROSE_DIR.glob("*.json"))
    return [path for path in paths if not path.stem.endswith("_attributes")]


def _select_rose(dataset: str, target: int, no_llm: bool, limit: int) -> tuple[list[dict], list[dict]]:
    reader = init_ocr()
    selected: list[dict] = []
    attempted: list[dict] = []
    paths = _rose_jsons(dataset)
    if limit:
        paths = paths[:limit]

    for index, path in enumerate(paths, start=1):
        row = evaluate_rose.evaluate_chart(path, reader=reader, use_llm=not no_llm)
        row["dataset"] = dataset
        row["chart_type"] = "rose"
        row["index"] = index
        row["json_path"] = str(path.resolve())
        row["image_path"] = str(path.with_suffix(".png").resolve())
        attempted.append(row)
        if (not row.get("skipped")) and (not row.get("fallback")) and row["correct"] == row["total"] and row["total"] > 0:
            selected.append(row)
            print(f"[rose/{dataset}] selected {len(selected):2d}/{target}: {row['name']}")
            if len(selected) >= target:
                break
        else:
            tag = row.get("reason") or row.get("fallback_reason") or f"{row.get('correct', 0)}/{row.get('total', 0)}"
            print(f"[rose/{dataset}] skip {row['name']}: {tag}")
    return selected, attempted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart-type", choices=["radar", "rose", "all"], default="all")
    parser.add_argument("--dataset", choices=["real", "synthetic"], default="synthetic")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-llm", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_all: list[dict] = []
    attempted_all: list[dict] = []

    if args.chart_type in ("radar", "all"):
        selected, attempted = _select_radar(args.dataset, args.target, args.no_llm, args.limit)
        selected_all.extend(selected)
        attempted_all.extend(attempted)
        _write_csv(args.output_dir / f"radar_{args.dataset}_selected_{args.target}.csv", selected)
        _write_json(args.output_dir / f"radar_{args.dataset}_attempted.json", attempted)

    if args.chart_type in ("rose", "all"):
        selected, attempted = _select_rose(args.dataset, args.target, args.no_llm, args.limit)
        selected_all.extend(selected)
        attempted_all.extend(attempted)
        _write_csv(args.output_dir / f"rose_{args.dataset}_selected_{args.target}.csv", selected)
        _write_json(args.output_dir / f"rose_{args.dataset}_attempted.json", attempted)

    _write_json(args.output_dir / f"axis_sample_selection_{args.chart_type}_{args.dataset}.json", {
        "target": args.target,
        "selected_count": len(selected_all),
        "attempted_count": len(attempted_all),
        "selected": selected_all,
        "attempted": attempted_all,
    })
    print(f"\nOutput: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
