"""Check that a current axis-evaluation CSV reproduces a saved fallback manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row.get("chart_id", ""): row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    current = read_csv_rows(args.csv)
    failures: list[str] = []

    for item in manifest.get("charts", []):
        chart_id = item.get("chart_id", "")
        row = current.get(chart_id)
        if row is None:
            failures.append(f"{chart_id}: missing from current CSV")
            continue

        expected_fallback = bool(item.get("expected_fallback"))
        actual_fallback = str(row.get("fallback", "")).lower() == "true"
        if expected_fallback != actual_fallback:
            failures.append(f"{chart_id}: fallback expected {expected_fallback}, got {actual_fallback}")
            continue

        prefix = item.get("expected_reason_prefix") or ""
        actual_reason = row.get("fallback_reason", "") or ""
        if expected_fallback and prefix and not actual_reason.startswith(prefix):
            failures.append(f"{chart_id}: reason expected prefix {prefix!r}, got {actual_reason!r}")

    if failures:
        print("[fallback-repro] FAILED")
        for failure in failures:
            print(" -", failure)
        return 1

    print(f"[fallback-repro] OK: {len(manifest.get('charts', []))} charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
