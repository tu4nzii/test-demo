"""Unified CLI for the latest chart value-prediction scripts."""

from __future__ import annotations

import argparse

from .chart_types import CHART_SPECS, get_spec
from .execution.adapter import RunRequest, describe_request, run_backend
from .specs import PROJECT_ROOT


def list_entries() -> None:
    for entry in CHART_SPECS.values():
        rel_script = entry.script.relative_to(PROJECT_ROOT)
        print(f"{entry.chart_type:8s} {entry.coordinate_system:10s} {rel_script}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chart value prediction through one entry point.")
    parser.add_argument("chart_type", nargs="?", choices=sorted(CHART_SPECS))
    parser.add_argument("--chart-ids", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="List registered latest scripts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without running it.")
    args = parser.parse_args()

    if args.list:
        list_entries()
        return 0
    if not args.chart_type:
        parser.error("chart_type is required unless --list is used")

    entry = get_spec(args.chart_type)
    request = RunRequest(
        spec=entry,
        chart_ids=args.chart_ids,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(describe_request(request))
        return 0
    return run_backend(request)


if __name__ == "__main__":
    raise SystemExit(main())
