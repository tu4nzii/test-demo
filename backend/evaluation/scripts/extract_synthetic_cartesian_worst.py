from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


CARTESIAN_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}
GOOD_STATUSES = {"success", "skipped_success_cache", "recovered_from_grid_reference"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(value: str, limit: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    cleaned = cleaned.strip("._") or "sample"
    return cleaned[:limit]


def copy_if_exists(src: Any, dst: Path) -> str | None:
    if not src:
        return None
    src_path = Path(str(src))
    if not src_path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return dst.name


def manifest_by_relative(path: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(path)
    return {
        str(record.get("dataset_relative")): record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and record.get("dataset_relative")
    }


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def metric_count(row: dict[str, Any], metric: str) -> int:
    if metric == "tick_mae":
        return int(row.get("numeric_matched") or 0)
    if metric == "tick_acc":
        return int(row.get("numeric_total") or 0)
    if metric == "pos_mae":
        return int(row.get("tick_position_matched") or 0)
    if metric == "label_acc":
        return int(row.get("label_total") or 0)
    if metric == "legend_color_acc":
        return int(row.get("legend_color_total") or 0)
    if metric == "chart_type_acc":
        return int(row.get("chart_type_total") or 0)
    return 0


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric == "tick_mae":
        matched = float(row.get("numeric_matched") or 0)
        return float(row.get("numeric_error_sum") or 0) / matched if matched else None
    if metric == "tick_acc":
        total = float(row.get("numeric_total") or 0)
        return float(row.get("numeric_correct") or 0) / total if total else None
    if metric == "pos_mae":
        matched = float(row.get("tick_position_matched") or 0)
        return float(row.get("position_error_sum") or 0) / matched if matched else None
    if metric == "label_acc":
        total = float(row.get("label_total") or 0)
        return float(row.get("label_correct") or 0) / total if total else None
    if metric == "legend_color_acc":
        total = float(row.get("legend_color_total") or 0)
        return float(row.get("legend_color_correct") or 0) / total if total else None
    if metric == "chart_type_acc":
        total = float(row.get("chart_type_total") or 0)
        return float(row.get("chart_type_correct") or 0) / total if total else None
    return None


def metric_text(metric: str, value: float | None) -> str:
    if metric.endswith("_acc"):
        return pct(value)
    return num(value)


def worst_rows(rows: list[dict[str, Any]], metric: str, direction: str, limit: int) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        value = metric_value(row, metric)
        if value is None or metric_count(row, metric) <= 0:
            continue
        candidates.append((value, row))
    candidates.sort(key=lambda item: item[0], reverse=(direction == "max"))
    return [
        {
            "rank": index,
            "metric": metric,
            "value": value,
            "value_text": metric_text(metric, value),
            "count": metric_count(row, metric),
            "row": row,
        }
        for index, (value, row) in enumerate(candidates[:limit], start=1)
    ]


def build_report(details_path: Path, manifest_path: Path, output_root: Path, limit: int) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    details = read_json(details_path)
    manifest = manifest_by_relative(manifest_path)
    rows = [
        row
        for row in details
        if row.get("dataset") == "Sy.Dataset"
        and row.get("gt_type") in CARTESIAN_TYPES
        and row.get("status") in GOOD_STATUSES
        and row.get("grid_eval_included")
        and not row.get("grid_eval_skipped")
    ]
    excluded = [
        row
        for row in details
        if row.get("dataset") == "Sy.Dataset"
        and row.get("gt_type") in CARTESIAN_TYPES
        and row not in rows
    ]

    metric_specs = [
        ("Tick MAE(px)", "tick_mae", "max", "数值轴 tick 像素误差最大"),
        ("Tick Acc@2px", "tick_acc", "min", "数值轴 tick 在 2px 容差内准确率最低"),
        ("Pos MAE(px)", "pos_mae", "max", "tick 位置像素误差最大"),
        ("标签准确率", "label_acc", "min", "tick 标签文本匹配准确率最低"),
        ("图例颜色准确率", "legend_color_acc", "min", "图例/系列颜色匹配准确率最低"),
        ("图表分类准确率", "chart_type_acc", "min", "图表类型分类准确率最低"),
    ]

    sections: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    for label, metric, direction, description in metric_specs:
        items = worst_rows(rows, metric, direction, limit)
        sections.append(
            {
                "label": label,
                "metric": metric,
                "direction": direction,
                "description": description,
                "items": items,
            }
        )
        for item in items:
            rel = str(item["row"].get("dataset_relative"))
            selected.setdefault(rel, item["row"])

    sample_payloads: list[dict[str, Any]] = []
    for index, (dataset_relative, row) in enumerate(sorted(selected.items()), start=1):
        record = manifest.get(dataset_relative, {})
        sample_dir = output_root / f"{index:03d}_{safe_name(Path(dataset_relative).stem)}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
        images = {
            "source": copy_if_exists(
                record.get("source_image") or copied.get("source_image") or copied.get("uploaded_image"),
                sample_dir / "source.png",
            ),
            "image_grid": copy_if_exists(copied.get("basic_grid"), sample_dir / "image_grid.png"),
            "image_with_grid": copy_if_exists(copied.get("encrypted_grid"), sample_dir / "image_with_grid.png"),
            "image_with_grid_color": copy_if_exists(
                copied.get("colored_grid"),
                sample_dir / "image_with_grid_color.png",
            ),
            "ticks": copy_if_exists(copied.get("ticks_json"), sample_dir / "ticks.json"),
        }
        payload = {
            "index": index,
            "sample_dir": sample_dir.name,
            "dataset_relative": dataset_relative,
            "gt_type": row.get("gt_type"),
            "pred_type": row.get("pred_type"),
            "status": row.get("status"),
            "metrics": {
                "tick_mae": metric_value(row, "tick_mae"),
                "tick_acc": metric_value(row, "tick_acc"),
                "pos_mae": metric_value(row, "pos_mae"),
                "label_acc": metric_value(row, "label_acc"),
                "legend_color_acc": metric_value(row, "legend_color_acc"),
                "chart_type_acc": metric_value(row, "chart_type_acc"),
            },
            "copied_images": images,
            "detail": row,
            "record": record,
        }
        write_json(sample_dir / "sample.json", payload)
        sample_payloads.append(payload)

    lines = [
        "# 合成数据集直角坐标系效果较差样本",
        "",
        f"- 来源明细：`{details_path}`",
        f"- Manifest：`{manifest_path}`",
        "- 数据集：`Sy.Dataset`",
        "- 范围：`v_bar`、`h_bar`、`line`、`scatter`、`bubble`",
        "- 口径：只统计已参与评估的样本；失败样本和 `grid_eval_skipped` 样本单独说明，不混入最差样本表。",
        f"- 已参与评估样本数：{len(rows)}",
        f"- 未纳入最差表样本数：{len(excluded)}",
        f"- 去重后复制样本数：{len(sample_payloads)}",
        "",
    ]

    for section in sections:
        lines.extend(
            [
                f"## {section['label']}",
                "",
                f"- 排序依据：{section['description']}",
                "",
                "| 排名 | 样本 | GT类型 | 预测类型 | 当前值 | 参与项数 | 查看目录 |",
                "| ---: | --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in section["items"]:
            row = item["row"]
            rel = str(row.get("dataset_relative"))
            sample = next((payload for payload in sample_payloads if payload["dataset_relative"] == rel), None)
            sample_dir = sample["sample_dir"] if sample else "-"
            lines.append(
                f"| {item['rank']} | `{rel}` | `{row.get('gt_type')}` | `{row.get('pred_type')}` | "
                f"{item['value_text']} | {item['count']} | `{sample_dir}` |"
            )
        lines.append("")

    if excluded:
        lines.extend(
            [
                "## 未纳入最差表的合成直角系样本",
                "",
                "| 样本 | GT类型 | 预测类型 | 状态 | 原因 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in sorted(excluded, key=lambda item: str(item.get("dataset_relative"))):
            reason = row.get("grid_eval_skip_reason") or row.get("gt_eval_error") or row.get("error") or "-"
            if reason == "-" and not row.get("grid_eval_included"):
                reason = "未进入直角系网格指标计算"
            lines.append(
                f"| `{row.get('dataset_relative')}` | `{row.get('gt_type')}` | `{row.get('pred_type')}` | "
                f"{row.get('status')} | {str(reason).replace('|', '/')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 去重样本索引",
            "",
            "| 序号 | 样本 | GT类型 | 预测类型 | Tick MAE | Tick Acc | Pos MAE | 标签准确率 | 图例颜色准确率 | 分类准确率 | 目录 |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for payload in sample_payloads:
        metrics = payload["metrics"]
        lines.append(
            f"| {payload['index']} | `{payload['dataset_relative']}` | `{payload['gt_type']}` | `{payload['pred_type']}` | "
            f"{num(metrics['tick_mae'])} | {pct(metrics['tick_acc'])} | {num(metrics['pos_mae'])} | "
            f"{pct(metrics['label_acc'])} | {pct(metrics['legend_color_acc'])} | {pct(metrics['chart_type_acc'])} | "
            f"`{payload['sample_dir']}` |"
        )

    lines.extend(["", "## 可视化预览", ""])
    for payload in sample_payloads:
        rel_dir = payload["sample_dir"]
        lines.extend(
            [
                f"### {payload['index']}. {payload['dataset_relative']}",
                "",
                f"- GT类型：`{payload['gt_type']}`；预测类型：`{payload['pred_type']}`",
            ]
        )
        images = payload["copied_images"]
        if images.get("source"):
            lines.append(f"![原图]({rel_dir}/{images['source']})")
        if images.get("image_grid"):
            lines.append(f"![原生网格]({rel_dir}/{images['image_grid']})")
        if images.get("image_with_grid"):
            lines.append(f"![加密网格]({rel_dir}/{images['image_with_grid']})")
        if images.get("image_with_grid_color"):
            lines.append(f"![彩色网格]({rel_dir}/{images['image_with_grid_color']})")
        lines.extend(["", f"- 完整 JSON：`{rel_dir}/sample.json`", ""])

    write_json(
        output_root / "worst_samples.json",
        {
            "dataset": "Sy.Dataset",
            "scope": "cartesian",
            "included_count": len(rows),
            "excluded_count": len(excluded),
            "unique_sample_count": len(sample_payloads),
            "sections": sections,
            "samples": sample_payloads,
        },
    )
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--details",
        type=Path,
        default=Path("backend/evaluation/results/vishintprompt_full_latest_report/details.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("backend/evaluation/recheck_outputs/vishintprompt_full_grid_encryption_latest/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/evaluation/results/vishintprompt_full_latest_report/synthetic_cartesian_worst"),
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    build_report(args.details, args.manifest, args.output, args.limit)
    print(f"Wrote {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
