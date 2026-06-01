"""CLI entry for the modular horizontal bar runner."""

from __future__ import annotations

import argparse
import asyncio
from .runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Horizontal bar chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    parser.add_argument("--config-json", nargs="+", default=None)
    args = parser.parse_args()
    asyncio.run(run_experiment(batch_size=args.batch_size, chart_ids=args.chart_ids, config_paths=args.config_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
