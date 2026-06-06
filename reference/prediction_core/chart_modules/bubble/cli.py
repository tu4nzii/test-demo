"""CLI entry for the modular bubble runner."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "prediction_core").is_dir():
        sys.path.insert(0, str(_parent))
        break

from reference.prediction_core.chart_modules.bubble.data import PointChartConfig
from reference.prediction_core.chart_modules.bubble.runner import run_experiment


CONFIG = PointChartConfig(chart_type="bubble", result_dir_name="results_bubble_Pixtral", mark_name="bubble")


def main() -> int:
    parser = argparse.ArgumentParser(description="Bubble chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment(CONFIG, batch_size=args.batch_size, chart_ids=args.chart_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
