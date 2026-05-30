"""CLI entry for the modular scatter runner."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "prediction_core").is_dir():
        sys.path.insert(0, str(_parent))
        break

from prediction_core.chart_modules.scatter.data import PointChartConfig
from prediction_core.chart_modules.scatter.runner import run_experiment


CONFIG = PointChartConfig(chart_type="scatter", result_dir_name="results_scatter_Pixtral", mark_name="circle")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scatter chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment(CONFIG, batch_size=args.batch_size, chart_ids=args.chart_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
