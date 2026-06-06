"""Run one temporary single-object end-to-end test per chart type."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from reference.prediction_core.chart_types import CHART_SPECS, get_spec
from reference.prediction_core.execution.adapter import RunRequest, describe_request, run_backend
from reference.prediction_core.testing.single_object import TemporarySingleObjectDataset


@dataclass
class E2EResult:
    chart_type: str
    chart_id: str
    object_name: str
    exit_code: int
    elapsed_seconds: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one single-object E2E test per chart type.")
    parser.add_argument(
        "--chart-types",
        nargs="+",
        default=sorted(CHART_SPECS),
        choices=sorted(CHART_SPECS),
        help="Chart types to test. Defaults to all registered types.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeat-times", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None)
    return parser.parse_args()


def run_one(chart_type: str, batch_size: int, dry_run: bool) -> E2EResult:
    spec = get_spec(chart_type)
    with TemporarySingleObjectDataset(spec) as trimmed:
        request = RunRequest(
            spec=spec,
            chart_ids=[trimmed.chart_id],
            batch_size=batch_size,
            dry_run=dry_run,
        )
        print(f"\n=== {chart_type}: {trimmed.chart_id} / {trimmed.object_name} ===")
        started = time.monotonic()
        if dry_run:
            print(describe_request(request))
            exit_code = 0
        else:
            exit_code = run_backend(request)
        elapsed = time.monotonic() - started
        return E2EResult(
            chart_type=chart_type,
            chart_id=trimmed.chart_id,
            object_name=trimmed.object_name,
            exit_code=exit_code,
            elapsed_seconds=round(elapsed, 2),
        )


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("CHART_REPEAT_TIMES", str(args.repeat_times))

    results: list[E2EResult] = []
    for chart_type in args.chart_types:
        result = run_one(chart_type, args.batch_size, args.dry_run)
        results.append(result)
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {chart_type} exit_code={result.exit_code} elapsed={result.elapsed_seconds}s")
        if not result.ok and not args.continue_on_failure:
            break

    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(
            json.dumps([asdict(result) | {"ok": result.ok} for result in results], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    failed = [result for result in results if not result.ok]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
