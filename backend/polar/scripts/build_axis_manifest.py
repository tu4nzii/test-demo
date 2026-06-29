"""Build a fallback reproduction manifest from an axis-evaluation CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def build_manifest(csv_path: Path, chart_type: str, dataset_label: str, policy_version: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fallback = str(row.get("fallback", "")).lower() == "true"
            reason = row.get("fallback_reason", "") or ""
            rows.append({
                "chart_id": row.get("chart_id", ""),
                "chart_type": chart_type,
                "dataset": row.get("dataset", dataset_label),
                "image_path": row.get("image_path", ""),
                "json_path": row.get("json_path", ""),
                "expected_fallback": fallback,
                "expected_reason_prefix": reason,
                "expected_tolerance_pass": str(row.get("tolerance_pass", "")).lower() == "true" if not fallback else None,
            })
    return {
        "manifest_version": 1,
        "fallback_policy_version": policy_version,
        "source_csv": str(csv_path),
        "chart_type": chart_type,
        "dataset": dataset_label,
        "charts": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--chart-type", choices=["radar", "rose", "pie", "donut"], required=True)
    parser.add_argument("--dataset", default="real")
    parser.add_argument("--policy-version", default="polar_axis_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.csv, args.chart_type, args.dataset, args.policy_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[manifest] {args.output} ({len(manifest['charts'])} charts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
