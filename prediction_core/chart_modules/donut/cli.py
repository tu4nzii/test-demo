"""CLI entry for donut chart prediction."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "prediction_core").is_dir():
        sys.path.insert(0, str(_parent))
        break

ASSET_WORKDIR = Path(__file__).resolve().parents[2] / "assets" / "donut"


def main() -> int:
    parser = argparse.ArgumentParser(description="Donut chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    args = parser.parse_args()

    os.chdir(ASSET_WORKDIR)
    from prediction_core.chart_modules.donut.runner import run_experiment

    asyncio.run(run_experiment(batch_size=args.batch_size, chart_ids=args.chart_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
