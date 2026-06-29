"""Report legend-color accuracy under multiple RGB-distance tolerances.

This is an offline scoring helper. It reads GT only for evaluation and never
feeds GT into generation or model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from evaluation.scripts.diagnose_legend_color_binding import diagnose_item  # noqa: E402
from evaluation.scripts.evaluate_vishintprompt_latest_metrics import (  # noqa: E402
    artifact_payload,
    flatten_colors,
    source_config_path,
)


CARTESIAN_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}
POLAR_TYPES = {"pie", "donut", "radar", "rose"}
TYPE_ORDER = {
    "v_bar": 0,
    "h_bar": 1,
    "line": 2,
    "scatter": 3,
    "bubble": 4,
    "rose": 5,
    "pie": 6,
    "donut": 7,
    "radar": 8,
}
STATUS_CN = {
    "correct": "名称匹配且颜色正确",
    "missing_pred_colors": "预测缺失颜色",
    "name_exists_invalid_color": "名称匹配但颜色无效",
    "name_exists_color_far": "名称匹配但颜色偏差大",
    "likely_swapped_binding": "疑似颜色绑定错位",
    "name_missing_color_close": "名称不匹配但颜色接近",
    "name_missing_color_far": "名称不匹配且颜色偏差大",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def try_read_json(path: Path) -> Any | None:
    try:
        return read_json(path)
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_type(dataset_relative: Any) -> str | None:
    text = str(dataset_relative or "").replace("\\", "/")
    name = Path(text).stem
    if "/hBar_50/" in text:
        return "h_bar"
    if "/vBar_50/" in text:
        return "v_bar"
    if "/Line_50/" in text:
        return "line"
    if "/Scatetr_50/" in text or "/Scatter_50/" in text:
        return "scatter"
    if "/Bubble_50/" in text:
        return "bubble"
    if "/Pie_50/" in text:
        return "pie"
    if "/Donut_50/" in text:
        return "donut"
    if name.startswith(("BarChart", "GroupedBarChart", "StackedBarChart")):
        return "v_bar"
    if name.startswith("LineGraph"):
        return "line"
    if name.startswith(("ScatterPlot", "Scatterplot")):
        return "scatter"
    if name.startswith(("BubbleChart", "Bubblechart")):
        return "bubble"
    if name.startswith("PieChart"):
        return "pie"
    if name.startswith("DonutChart"):
        return "donut"
    if name.startswith("Radar"):
        return "radar"
    if "rose" in name.lower() or "nightingale" in name.lower():
        return "rose"
    return None


def category_for_type(chart_type: str | None) -> str:
    if chart_type in CARTESIAN_TYPES:
        return "直角系"
    if chart_type in POLAR_TYPES:
        return "极坐标"
    return "其他"


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def pct(correct: int, total: int) -> str:
    return "-" if total == 0 else f"{correct}/{total} ({correct / total * 100:.2f}%)"


def status_text(status: str) -> str:
    return STATUS_CN.get(status, status)


def metric_counts_for_threshold(
    gt_colors: dict[str, str],
    pred_colors: dict[str, str],
    threshold: float,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for gt_name, gt_color in gt_colors.items():
        diagnosis = diagnose_item(gt_name, gt_color, pred_colors, threshold=threshold)
        counts[str(diagnosis.get("status") or "unknown")] += 1
    return dict(counts)


def strict_correct(counts: dict[str, int]) -> int:
    return int(counts.get("correct", 0))


def color_close_or_swapped(counts: dict[str, int]) -> int:
    return int(
        counts.get("correct", 0)
        + counts.get("name_missing_color_close", 0)
        + counts.get("likely_swapped_binding", 0)
    )


def choose_recommended_threshold(rows: list[dict[str, Any]]) -> float | None:
    """Pick the smallest threshold that reaches 80% strict accuracy.

    If no threshold reaches 80%, pick the highest strict-accuracy threshold.
    """
    if not rows:
        return None
    for row in rows:
        if row["strict_accuracy"] >= 0.8:
            return float(row["threshold"])
    best = max(rows, key=lambda item: (item["strict_accuracy"], -float(item["threshold"])))
    return float(best["threshold"])


def build_report(
    batch_root: Path,
    dataset_root: Path,
    output: Path,
    thresholds: list[float],
) -> None:
    manifest = read_json(batch_root / "manifest.json")
    samples: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        dataset_relative = str(record.get("dataset_relative") or "")
        gt_path = source_config_path(dataset_root, dataset_relative)
        if gt_path is None or not gt_path.exists():
            continue
        gt = try_read_json(gt_path)
        if not isinstance(gt, dict):
            continue
        pred = artifact_payload(record)
        gt_colors = flatten_colors(gt.get("series_color"))
        pred_colors = flatten_colors(pred.get("series_color")) or flatten_colors(pred.get("colors"))
        if not gt_colors:
            continue
        chart_type = infer_type(dataset_relative) or str(record.get("chart_type") or "unknown")
        samples.append(
            {
                "dataset": str(record.get("dataset") or dataset_relative.split("/", 1)[0]),
                "dataset_relative": dataset_relative,
                "chart_type": chart_type,
                "category": category_for_type(chart_type),
                "gt_colors": gt_colors,
                "pred_colors": pred_colors,
            }
        )

    grouped: dict[tuple[str, str, str, float], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall: dict[tuple[str, float], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    status_detail: dict[tuple[str, float], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        for threshold in thresholds:
            counts = metric_counts_for_threshold(sample["gt_colors"], sample["pred_colors"], threshold)
            keys = [
                (sample["dataset"], sample["category"], "category"),
                (sample["dataset"], sample["chart_type"], "type"),
                ("总计", sample["category"], "category"),
                ("总计", sample["chart_type"], "type"),
                ("总计", "总计", "overall"),
            ]
            for dataset, group, kind in keys:
                add_counts(grouped[(dataset, group, kind, threshold)], counts)
            add_counts(overall[(sample["category"], threshold)], counts)
            add_counts(status_detail[(sample["category"], threshold)], counts)

    def row_from_counts(label: str, threshold: float, counts: dict[str, int]) -> dict[str, Any]:
        total = sum(counts.values())
        strict = strict_correct(counts)
        close = color_close_or_swapped(counts)
        return {
            "label": label,
            "threshold": threshold,
            "total": total,
            "strict_correct": strict,
            "strict_accuracy": strict / total if total else 0.0,
            "close_or_swapped": close,
            "close_or_swapped_accuracy": close / total if total else 0.0,
            "status_counts": dict(counts),
        }

    rows: list[dict[str, Any]] = []
    for (dataset, group, kind, threshold), counts in grouped.items():
        row = row_from_counts(group, threshold, counts)
        row.update({"dataset": dataset, "group": group, "kind": kind})
        rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "legend_color_tolerance_summary.json",
        {
            "batch_root": str(batch_root),
            "dataset_root": str(dataset_root),
            "thresholds": thresholds,
            "sample_count": len(samples),
            "rows": rows,
        },
    )

    def threshold_table(scope: str, group: str, kind: str) -> list[str]:
        lines = [
            "| RGB距离阈值 | 严格名称+颜色准确率 | 颜色接近或疑似错位 | 诊断项数 |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for threshold in thresholds:
            counts = grouped.get((scope, group, kind, threshold), {})
            total = sum(counts.values())
            lines.append(
                f"| {threshold:g} | {pct(strict_correct(counts), total)} | "
                f"{pct(color_close_or_swapped(counts), total)} | {total} |"
            )
        return lines

    lines = [
        "# 图例颜色容差敏感性报告",
        "",
        "- 口径：主评估仍要求名称匹配；本报告只改变 RGB 欧氏距离阈值。",
        "- “颜色接近或疑似错位”是诊断参考，不等同于主指标。",
        "- GT 只用于离线打分，不进入生成端或模型输入。",
        "",
        "## 全量",
        "",
        *threshold_table("总计", "总计", "overall"),
        "",
        "## 直角系",
        "",
        *threshold_table("总计", "直角系", "category"),
        "",
        "## 极坐标",
        "",
        *threshold_table("总计", "极坐标", "category"),
        "",
        "## 直角系按类型",
        "",
        "| 类型 | 推荐阈值 | " + " | ".join(f"阈值{t:g}" for t in thresholds) + " |",
        "| --- | ---: | " + " | ".join("---:" for _ in thresholds) + " |",
    ]

    for chart_type in ["v_bar", "h_bar", "line", "scatter", "bubble"]:
        type_rows = []
        for threshold in thresholds:
            counts = grouped.get(("总计", chart_type, "type", threshold), {})
            total = sum(counts.values())
            strict = strict_correct(counts)
            type_rows.append(
                {
                    "threshold": threshold,
                    "strict_accuracy": strict / total if total else 0.0,
                    "text": pct(strict, total),
                }
            )
        recommended = choose_recommended_threshold(type_rows)
        lines.append(
            f"| {chart_type} | {recommended:g} | "
            + " | ".join(row["text"] for row in type_rows)
            + " |"
        )

    lines.extend(
        [
            "",
            "## 各阈值下的问题分布（直角系）",
            "",
        ]
    )
    for threshold in thresholds:
        counts = grouped.get(("总计", "直角系", "category", threshold), {})
        lines.extend(
            [
                f"### 阈值 {threshold:g}",
                "",
                "| 诊断类型 | 数量 |",
                "| --- | ---: |",
            ]
        )
        for status, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {status_text(status)} | {count} |")
        lines.append("")

    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("backend/datasets/VisHintPrompt_datasets"))
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("backend/evaluation/recheck_outputs/vishintprompt_full_grid_encryption_latest"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/evaluation/results/legend_color_prompt_probe/tolerance"),
    )
    parser.add_argument("--thresholds", default="45,60,80,100,120")
    args = parser.parse_args()
    thresholds = [float(item) for item in str(args.thresholds).split(",") if item.strip()]
    build_report(args.batch_root, args.dataset_root, args.output, thresholds)
    print(f"Wrote {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
