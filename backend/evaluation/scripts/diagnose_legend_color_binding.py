"""Diagnose legend/series color binding failures for generated outputs.

This is an offline evaluation helper. It may read dataset GT JSON only for
scoring/diagnostics and must not be used by the generation path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from evaluation.scripts.evaluate_vishintprompt_latest_metrics import (  # noqa: E402
    artifact_payload,
    flatten_colors,
    normalize_text,
    source_config_path,
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def rgb_from_hex(value: Any) -> tuple[int, int, int] | None:
    text = str(value or "").strip()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        return None
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return None


def color_distance(left: str, right: str) -> float | None:
    left_rgb = rgb_from_hex(left)
    right_rgb = rgb_from_hex(right)
    if left_rgb is None or right_rgb is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_rgb, right_rgb)))


def best_color_match(gt_color: str, pred_colors: dict[str, str]) -> tuple[str, str, float] | None:
    best: tuple[str, str, float] | None = None
    for pred_name, pred_color in pred_colors.items():
        distance = color_distance(gt_color, pred_color)
        if distance is None:
            continue
        item = (pred_name, pred_color, distance)
        if best is None or item[2] < best[2]:
            best = item
    return best


def diagnose_item(
    gt_name: str,
    gt_color: str,
    pred_colors: dict[str, str],
    *,
    threshold: float,
) -> dict[str, Any]:
    gt_norm = normalize_text(gt_name)
    name_candidates = {
        pred_name: pred_color
        for pred_name, pred_color in pred_colors.items()
        if normalize_text(pred_name) == gt_norm
        or normalize_text(pred_name).split("/")[-1] == gt_norm.split("/")[-1]
    }
    if not pred_colors:
        return {"status": "missing_pred_colors"}

    if name_candidates:
        distances = [
            (pred_name, pred_color, color_distance(gt_color, pred_color))
            for pred_name, pred_color in name_candidates.items()
        ]
        distances = [item for item in distances if item[2] is not None]
        if not distances:
            return {"status": "name_exists_invalid_color"}
        pred_name, pred_color, distance = min(distances, key=lambda item: item[2])
        if distance <= threshold:
            return {
                "status": "correct",
                "pred_name": pred_name,
                "pred_color": pred_color,
                "distance": distance,
            }
        best = best_color_match(gt_color, pred_colors)
        status = "name_exists_color_far"
        if best and normalize_text(best[0]) != gt_norm and best[2] <= threshold:
            status = "likely_swapped_binding"
        return {
            "status": status,
            "pred_name": pred_name,
            "pred_color": pred_color,
            "distance": distance,
            "best_color_match": best,
        }

    best = best_color_match(gt_color, pred_colors)
    if best and best[2] <= threshold:
        return {
            "status": "name_missing_color_close",
            "best_color_match": best,
        }
    return {
        "status": "name_missing_color_far",
        "best_color_match": best,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("backend/datasets/VisHintPrompt_datasets"))
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--color-threshold", type=float, default=45.0)
    args = parser.parse_args()

    manifest = load_json(args.batch_root / "manifest.json")
    rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        dataset_relative = str(record.get("dataset_relative") or "")
        gt_path = source_config_path(args.dataset_root, dataset_relative)
        if gt_path is None or not gt_path.exists():
            continue
        gt = load_json(gt_path)
        pred = artifact_payload(record)
        gt_colors = flatten_colors(gt.get("series_color"))
        pred_colors = flatten_colors(pred.get("series_color")) or flatten_colors(pred.get("colors"))
        if not gt_colors:
            continue

        item_rows = []
        for gt_name, gt_color in gt_colors.items():
            diagnosis = diagnose_item(gt_name, gt_color, pred_colors, threshold=args.color_threshold)
            status = str(diagnosis.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            item_rows.append(
                {
                    "gt_name": gt_name,
                    "gt_color": gt_color,
                    **diagnosis,
                }
            )
        rows.append(
            {
                "dataset_relative": dataset_relative,
                "chart_type": record.get("chart_type"),
                "status": record.get("status"),
                "gt_count": len(gt_colors),
                "pred_count": len(pred_colors),
                "pred_colors": pred_colors,
                "items": item_rows,
            }
        )

    summary = {
        "batch_root": str(args.batch_root),
        "dataset_root": str(args.dataset_root),
        "color_threshold": args.color_threshold,
        "sample_count": len(rows),
        "status_counts": status_counts,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "legend_color_diagnosis.json").write_text(
        json.dumps({"summary": summary, "records": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Legend Color Binding Diagnosis",
        "",
        f"- Batch root: `{args.batch_root}`",
        f"- Color threshold: `{args.color_threshold}`",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "| Sample | GT | Pred Count | Item Statuses |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        statuses = ", ".join(
            f"{item['gt_name']}={item['status']}"
            for item in row["items"]
        )
        lines.append(
            f"| {row['dataset_relative']} | {row['gt_count']} | {row['pred_count']} | {statuses} |"
        )
    (args.output / "legend_color_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.output / 'legend_color_diagnosis.json'}")
    print(f"Wrote {args.output / 'legend_color_diagnosis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
