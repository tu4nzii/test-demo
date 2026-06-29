"""Stable entry point for polar/circular axis-prior evaluation.

This wrapper keeps README commands independent from the historical demo file
names. It delegates to the current evaluator implementations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def run(command: list[str]) -> int:
    print("[run]", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart-type", choices=["radar", "rose", "pie", "donut", "pie-donut", "all"], default="all")
    parser.add_argument("--dataset", choices=["real", "real_corrected", "synthetic", "synth", "all"], default="all")
    parser.add_argument("--tick-mode", choices=["gt-nearest", "algorithm"], default="gt-nearest")
    args = parser.parse_args()

    py = sys.executable
    status = 0

    if args.chart_type in {"radar", "all"}:
        radar_dataset = args.dataset
        if radar_dataset == "real_corrected":
            if args.chart_type == "all":
                radar_dataset = "real"
            else:
                raise SystemExit("--dataset real_corrected is only supported by rose; use --dataset real for radar.")
        if radar_dataset == "synth":
            radar_dataset = "synthetic"
        status |= run([
            py,
            "backend/polar/evaluation/evaluate_radar_grid_extraction.py",
            "--dataset",
            radar_dataset,
            "--tick-mode",
            args.tick_mode,
        ])

    if args.chart_type in {"rose", "all"}:
        rose_dataset = args.dataset
        if rose_dataset == "synthetic":
            rose_dataset = "synth"
        status |= run([
            py,
            "backend/polar/evaluation/evaluate_rose_grid_extraction.py",
            "--dataset",
            rose_dataset,
            "--tick-mode",
            args.tick_mode,
        ])

    if args.chart_type in {"pie", "donut", "pie-donut", "all"}:
        circular_dataset = args.dataset
        if circular_dataset == "synthetic":
            circular_dataset = "synth"
        circular_type = "all" if args.chart_type in {"pie-donut", "all"} else args.chart_type
        status |= run([
            py,
            "backend/polar/evaluation/evaluate_pie_donut_circle_extraction.py",
            "--chart-type",
            circular_type,
            "--dataset",
            circular_dataset,
        ])

    return int(bool(status))


if __name__ == "__main__":
    raise SystemExit(main())
