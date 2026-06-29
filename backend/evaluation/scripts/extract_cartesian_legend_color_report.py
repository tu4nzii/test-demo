"""Extract Cartesian-only legend color diagnosis into a focused report."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


CARTESIAN_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}
STATUS_CN = {
    "correct": "正确",
    "missing_pred_colors": "预测缺失颜色",
    "name_exists_invalid_color": "名称匹配但颜色无效",
    "name_exists_color_far": "名称匹配但颜色偏差大",
    "likely_swapped_binding": "疑似颜色绑定错位",
    "name_missing_color_close": "名称不匹配但颜色接近",
    "name_missing_color_far": "名称不匹配且颜色偏差大",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_cartesian_type(dataset_relative: Any) -> str | None:
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
    if name.startswith(("BarChart", "GroupedBarChart", "StackedBarChart")):
        return "v_bar"
    if name.startswith("LineGraph"):
        return "line"
    if name.startswith(("ScatterPlot", "Scatterplot")):
        return "scatter"
    if name.startswith(("BubbleChart", "Bubblechart")):
        return "bubble"
    return None


def status_text(value: Any) -> str:
    return STATUS_CN.get(str(value or ""), str(value or "-"))


def accuracy(counts: dict[str, int]) -> tuple[int, int, float]:
    total = sum(counts.values())
    correct = counts.get("correct", 0)
    return correct, total, (correct / total * 100 if total else 0.0)


def close_or_swapped_accuracy(counts: dict[str, int]) -> tuple[int, int, float]:
    total = sum(counts.values())
    correct = (
        counts.get("correct", 0)
        + counts.get("name_missing_color_close", 0)
        + counts.get("likely_swapped_binding", 0)
    )
    return correct, total, (correct / total * 100 if total else 0.0)


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def item_counts(row: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in row.get("items", []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def fmt_acc(values: tuple[int, int, float]) -> str:
    correct, total, percent = values
    return f"{correct}/{total} ({percent:.2f}%)" if total else "-"


def fmt_dist(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return "-"


def build_report(
    diagnosis_path: Path,
    samples_root: Path,
    output_root: Path,
    max_low_samples: int,
) -> None:
    diagnosis = read_json(diagnosis_path)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    total_counts: dict[str, int] = {}
    type_counts: dict[str, dict[str, int]] = {}
    for row in diagnosis.get("records", []):
        if not isinstance(row, dict):
            continue
        chart_type = infer_cartesian_type(row.get("dataset_relative"))
        if chart_type not in CARTESIAN_TYPES:
            continue
        counts = item_counts(row)
        if not counts:
            continue
        add_counts(total_counts, counts)
        add_counts(type_counts.setdefault(chart_type, {}), counts)
        records.append({**row, "cartesian_type": chart_type, "item_counts": counts})

    lines = [
        "# 直角系图例颜色诊断",
        "",
        "- 范围：`v_bar`、`h_bar`、`line`、`scatter`、`bubble`。",
        "- 这里是离线评估视角：GT 只用于打分，不进入生成端或模型输入。",
        "",
        "## 总览",
        "",
        f"- 严格名称+颜色正确：{fmt_acc(accuracy(total_counts))}",
        f"- 颜色接近或疑似绑定错位：{fmt_acc(close_or_swapped_accuracy(total_counts))}",
        "",
        "| 诊断类型 | 数量 |",
        "| --- | ---: |",
    ]
    for status, count in sorted(total_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {status_text(status)} | {count} |")

    lines.extend(
        [
            "",
            "## 按类型汇总",
            "",
            "| 类型 | 严格正确 | 颜色接近或疑似绑定错位 | 诊断项数 | 主要问题 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for chart_type in ["v_bar", "h_bar", "line", "scatter", "bubble"]:
        counts = type_counts.get(chart_type, {})
        problems = ", ".join(
            f"{status_text(status)}:{count}"
            for status, count in sorted(counts.items(), key=lambda item: -item[1])[:3]
        ) or "-"
        lines.append(
            f"| {chart_type} | {fmt_acc(accuracy(counts))} | "
            f"{fmt_acc(close_or_swapped_accuracy(counts))} | {sum(counts.values())} | {problems} |"
        )

    lines.extend(
        [
            "",
            "## 低分样本明细",
            "",
            "| 样本 | 类型 | GT项数 | 预测项数 | 严格正确 | 问题分布 |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    def strict_ratio(row: dict[str, Any]) -> tuple[float, int, str]:
        counts = row["item_counts"]
        correct, total, _ = accuracy(counts)
        ratio = correct / total if total else 1.0
        return ratio, -total, str(row.get("dataset_relative"))

    for row in sorted(records, key=strict_ratio)[:max_low_samples]:
        counts = row["item_counts"]
        correct, total, _ = accuracy(counts)
        problems = ", ".join(
            f"{status_text(status)}:{count}"
            for status, count in sorted(counts.items(), key=lambda item: -item[1])
            if status != "correct"
        ) or "-"
        lines.append(
            f"| `{row.get('dataset_relative')}` | {row['cartesian_type']} | "
            f"{row.get('gt_count', 0)} | {row.get('pred_count', 0)} | {correct}/{total} | {problems} |"
        )

    lines.extend(
        [
            "",
            "## 抽样可视化",
            "",
            "下面只保留刚刚专用提示词抽样中的直角系样本。",
            "",
        ]
    )
    for payload_path in sorted(samples_root.glob("*/legend_color_probe.json")):
        payload = read_json(payload_path)
        chart_type = infer_cartesian_type(payload.get("dataset_relative"))
        if chart_type not in CARTESIAN_TYPES:
            continue
        source_dir = payload_path.parent
        sample_dir = output_root / source_dir.name
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        shutil.copytree(source_dir, sample_dir)
        rel = sample_dir.relative_to(output_root).as_posix()

        lines.extend(
            [
                f"### {payload['dataset_relative']}",
                "",
                f"- 类型：`{chart_type}`",
                f"- 当前缓存颜色准确率：{payload['current_accuracy']['text']}",
                f"- 专用提示词颜色准确率：{payload['probe_accuracy']['text']}",
                "",
            ]
        )
        for name, title in (
            ("source.png", "原图"),
            ("image_with_grid.png", "灰色网格"),
            ("image_with_grid_color.png", "彩色网格"),
        ):
            if (sample_dir / name).exists():
                lines.append(f"![{title}]({rel}/{name})")
        lines.extend(
            [
                "",
                "| 名称 | GT颜色 | 当前缓存颜色/状态/距离 | 专用提示词颜色/状态/距离 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in payload.get("comparison", []):
            lines.append(
                "| {name} | `{gt}` | `{old}` / {old_status} / {old_dist} | "
                "`{new}` / {new_status} / {new_dist} |".format(
                    name=str(item.get("name", "")).replace("|", "/"),
                    gt=item.get("gt_color") or "-",
                    old=item.get("old_color") or "-",
                    old_status=status_text(item.get("old_status")),
                    old_dist=fmt_dist(item.get("old_distance")),
                    new=item.get("new_color") or "-",
                    new_status=status_text(item.get("new_status")),
                    new_dist=fmt_dist(item.get("new_distance")),
                )
            )
        lines.extend(["", f"- 完整 JSON：`{rel}/legend_color_probe.json`", ""])

    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnosis",
        type=Path,
        default=Path("backend/evaluation/results/legend_color_prompt_probe/legend_color_diagnosis.json"),
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("backend/evaluation/results/legend_color_prompt_probe/samples"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/evaluation/results/legend_color_prompt_probe/cartesian_only"),
    )
    parser.add_argument("--max-low-samples", type=int, default=80)
    args = parser.parse_args()
    build_report(args.diagnosis, args.samples_root, args.output, args.max_low_samples)
    print(f"Wrote {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
