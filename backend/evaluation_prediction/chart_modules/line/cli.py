"""CLI entry for backend-local line chart value prediction."""

from __future__ import annotations

import argparse
import asyncio

from .runner import run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description="Line chart value prediction")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    parser.add_argument(
        "--config-json",
        nargs="+",
        default=None,
        help="Backend-generated *_image.json or *_image_ticks.json files to use as prediction input.",
    )
    args = parser.parse_args()
    asyncio.run(
        run_experiment(
            batch_size=args.batch_size,
            chart_ids=args.chart_ids,
            config_paths=args.config_json,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
