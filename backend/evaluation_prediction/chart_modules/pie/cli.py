"""CLI entry for pie chart prediction."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSET_WORKDIR = Path(__file__).resolve().parents[2] / "assets" / "pie"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pie chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    args = parser.parse_args()

    os.chdir(ASSET_WORKDIR)
    from backend.evaluation_prediction.chart_modules.pie.runner import run_experiment

    asyncio.run(run_experiment(batch_size=args.batch_size, chart_ids=args.chart_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
