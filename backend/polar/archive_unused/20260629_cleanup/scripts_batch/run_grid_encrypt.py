"""Stable entry point for preparing encrypted polar chart JSON/PNG files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chart-type", choices=["radar", "rose", "all"], default="all")
    parser.add_argument("--mode", choices=["gt"], default="gt", help="Currently stable for GT-controlled real evaluation.")
    parser.add_argument("--output-dir", default="backend/data/polar/real_evaluation_data")
    args = parser.parse_args()

    command = [
        sys.executable,
        "backend/polar/encryption/prepare_real_evaluation_gt_encryption.py",
        "--output-dir",
        args.output_dir,
        "--chart-type",
        args.chart_type,
    ]
    print("[run]", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
