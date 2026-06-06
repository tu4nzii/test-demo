"""CLI entry for the modular line runner."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


for _parent in Path(__file__).resolve().parents:
    if (_parent / "prediction_core").is_dir():
        sys.path.insert(0, str(_parent))
        break

from reference.prediction_core.chart_modules.line.runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Line chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment(batch_size=args.batch_size, chart_ids=args.chart_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
