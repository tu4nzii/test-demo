"""Organize GT experiment result folders by chart type.

Default mode is a dry run. Use --apply to move folders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


SUPPORTED_TYPES = {
    "bubble",
    "scatter",
    "line",
    "v_bar",
    "h_bar",
    "pie",
    "donut",
    "radar",
    "rose",
}


def safe_path_fragment(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("._ ") or "chart"


def default_root() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return Path(os.getenv("CHART_GT_RESULTS_DIR", str(project_root / "gt_runs"))).expanduser()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def detect_chart_type(chart_dir: Path) -> str | None:
    if chart_dir.name in SUPPORTED_TYPES:
        return None

    candidates: list[Path] = []
    candidates.extend(chart_dir.glob("*/gt_config.json"))
    candidates.extend(chart_dir.glob("*/*_gt_experiment_result.json"))
    candidates.extend(chart_dir.glob("*.json"))
    for path in candidates:
        payload = read_json(path)
        chart_type = str(payload.get("chart_type") or "").strip().lower()
        if chart_type in SUPPORTED_TYPES:
            return chart_type

    name = chart_dir.name.lower()
    if name.startswith("h_bar"):
        return "h_bar"
    if name.startswith("v_bar"):
        return "v_bar"
    if name.startswith("bubble") or name.startswith("bubblechart"):
        return "bubble"
    if name.startswith("scatter"):
        return "scatter"
    if name.startswith("line") or name.startswith("linechart"):
        return "line"
    if name.startswith("pie"):
        return "pie"
    if name.startswith("donut"):
        return "donut"
    if name.startswith("radar"):
        return "radar"
    if name.startswith("rose"):
        return "rose"
    if name.startswith("barchart"):
        return "v_bar"
    if name.startswith("groupedbarchart"):
        return "v_bar"
    return None


def ensure_child(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise RuntimeError(f"Refuse to operate outside result root: {path}")


def organize(root: Path, *, apply: bool) -> tuple[int, int, int]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    moved = 0
    skipped = 0
    conflicts = 0
    for chart_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        chart_type = detect_chart_type(chart_dir)
        if not chart_type:
            skipped += 1
            print(f"SKIP unknown/already-category: {chart_dir.name}")
            continue

        target = root / safe_path_fragment(chart_type) / chart_dir.name
        if chart_dir.resolve() == target.resolve():
            skipped += 1
            print(f"SKIP already organized: {chart_dir.name}")
            continue
        if target.exists():
            conflicts += 1
            print(f"CONFLICT target exists: {chart_dir} -> {target}")
            continue

        ensure_child(chart_dir, root)
        ensure_child(target.parent, root)
        print(f"{'MOVE' if apply else 'DRY'} {chart_dir} -> {target}")
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(chart_dir), str(target))
        moved += 1

    return moved, skipped, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize GT experiment result folders by chart type.")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--apply", action="store_true", help="Move folders. Without this flag, only print the plan.")
    args = parser.parse_args()

    moved, skipped, conflicts = organize(args.root, apply=args.apply)
    print(f"summary moved={moved} skipped={skipped} conflicts={conflicts} apply={args.apply}")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
