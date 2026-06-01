"""CLI entry for backend-local scatter prediction."""

from __future__ import annotations

import argparse
import asyncio

from .data import PointChartConfig
from .runner import run_experiment


CONFIG = PointChartConfig(chart_type="scatter", result_dir_name="scatter", mark_name="circle")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scatter chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    parser.add_argument("--config-json", nargs="+", default=None)
    args = parser.parse_args()
    asyncio.run(
        run_experiment(
            CONFIG,
            batch_size=args.batch_size,
            chart_ids=args.chart_ids,
            config_paths=args.config_json,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
