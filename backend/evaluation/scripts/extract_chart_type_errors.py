"""Extract chart-type misclassification samples into a focused report."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")[:80] or "sample"


def copy_if_exists(src: Any, dst: Path) -> str | None:
    if not src:
        return None
    path = Path(str(src))
    if not path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst.name


def manifest_by_relative(path: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(path)
    return {
        str(record.get("dataset_relative")): record
        for record in manifest.get("records", [])
        if isinstance(record, dict)
    }


def classify_error(gt_type: str, pred_type: str, status: str) -> str:
    if pred_type in {"unknown", "", "none"} or status == "failed":
        return "模型/流程失败，未返回有效类型"
    if {gt_type, pred_type} <= {"v_bar", "h_bar"}:
        return "柱状图方向误判"
    if "stacked" in pred_type and gt_type in {"v_bar", "h_bar"}:
        return "普通/堆叠柱状图区分错误"
    if gt_type in {"scatter", "bubble"} and pred_type in {"scatter", "bubble"}:
        return "散点图/气泡图区分错误"
    if gt_type in {"pie", "donut", "rose", "radar"} or pred_type in {"pie", "donut", "rose", "radar"}:
        return "极坐标/环饼玫瑰相关误判"
    return "其他类型误判"


def fmt_pct(value: float, total: int) -> str:
    return "-" if total == 0 else f"{value / total * 100:.2f}%"


def build_report(details_path: Path, manifest_path: Path, output_root: Path) -> None:
    rows = read_json(details_path)
    manifest = manifest_by_relative(manifest_path)
    errors = [
        row
        for row in rows
        if row.get("chart_type_total") and not row.get("chart_type_correct")
    ]
    output_root.mkdir(parents=True, exist_ok=True)

    confusion = Counter((str(row.get("gt_type")), str(row.get("pred_type"))) for row in errors)
    by_reason = Counter(
        classify_error(str(row.get("gt_type")), str(row.get("pred_type")), str(row.get("status")))
        for row in errors
    )
    by_dataset = Counter(str(row.get("dataset")) for row in errors)
    by_gt = Counter(str(row.get("gt_type")) for row in errors)

    detail_payload: list[dict[str, Any]] = []
    lines = [
        "# 图表分类错误样本",
        "",
        f"- 来源明细：`{details_path}`",
        f"- 错误样本数：{len(errors)}",
        "- 这里使用离线评估结果，不涉及重新调用模型。",
        "",
        "## 错误类型概览",
        "",
        "| 错误类别 | 数量 | 占错误样本比例 |",
        "| --- | ---: | ---: |",
    ]
    for label, count in by_reason.most_common():
        lines.append(f"| {label} | {count} | {fmt_pct(count, len(errors))} |")

    lines.extend(["", "## 按数据集", "", "| 数据集 | 错误数 |", "| --- | ---: |"])
    for dataset, count in by_dataset.most_common():
        lines.append(f"| {dataset} | {count} |")

    lines.extend(["", "## 按 GT 类型", "", "| GT类型 | 错误数 |", "| --- | ---: |"])
    for gt_type, count in by_gt.most_common():
        lines.append(f"| `{gt_type}` | {count} |")

    lines.extend(
        [
            "",
            "## 混淆对",
            "",
            "| GT类型 | 预测类型 | 数量 |",
            "| --- | --- | ---: |",
        ]
    )
    for (gt_type, pred_type), count in confusion.most_common():
        lines.append(f"| `{gt_type}` | `{pred_type}` | {count} |")

    lines.extend(
        [
            "",
            "## 样本明细",
            "",
            "| 序号 | 样本 | GT类型 | 预测类型 | 状态 | 错误类别 | 图像目录 |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for index, row in enumerate(errors, start=1):
        dataset_relative = str(row.get("dataset_relative") or "")
        record = manifest.get(dataset_relative, {})
        sample_dir = output_root / f"{index:02d}_{safe_name(Path(dataset_relative).stem)}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
        source = (
            record.get("source_image")
            or copied.get("source_image")
            or copied.get("uploaded_image")
        )
        source_name = copy_if_exists(source, sample_dir / "source.png")
        grid_name = copy_if_exists(copied.get("encrypted_grid"), sample_dir / "image_with_grid.png")
        color_grid_name = copy_if_exists(copied.get("colored_grid"), sample_dir / "image_with_grid_color.png")
        basic_grid_name = copy_if_exists(copied.get("basic_grid"), sample_dir / "image_grid.png")

        gt_type = str(row.get("gt_type"))
        pred_type = str(row.get("pred_type"))
        status = str(row.get("status"))
        reason = classify_error(gt_type, pred_type, status)
        rel_dir = sample_dir.relative_to(output_root).as_posix()

        payload = {
            "dataset": row.get("dataset"),
            "dataset_relative": dataset_relative,
            "status": status,
            "gt_type": gt_type,
            "pred_type": pred_type,
            "chart_family_correct": row.get("chart_family_correct"),
            "error_category": reason,
            "source_image": str(source) if source else None,
            "copied_images": {
                "source": source_name,
                "image_with_grid": grid_name,
                "image_with_grid_color": color_grid_name,
                "image_grid": basic_grid_name,
            },
            "record": record,
            "detail": row,
        }
        (sample_dir / "classification_error.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        detail_payload.append(payload)
        lines.append(
            f"| {index} | `{dataset_relative}` | `{gt_type}` | `{pred_type}` | "
            f"{status} | {reason} | `{rel_dir}` |"
        )

    lines.extend(["", "## 可视化索引", ""])
    for index, payload in enumerate(detail_payload, start=1):
        rel_dir = Path(f"{index:02d}_{safe_name(Path(str(payload['dataset_relative'])).stem)}").as_posix()
        lines.extend(
            [
                f"### {index}. {payload['dataset_relative']}",
                "",
                f"- GT类型：`{payload['gt_type']}`",
                f"- 预测类型：`{payload['pred_type']}`",
                f"- 错误类别：{payload['error_category']}",
                f"- 处理状态：`{payload['status']}`",
                "",
            ]
        )
        copied_images = payload["copied_images"]
        if copied_images.get("source"):
            lines.append(f"![原图]({rel_dir}/{copied_images['source']})")
        if copied_images.get("image_grid"):
            lines.append(f"![原生网格]({rel_dir}/{copied_images['image_grid']})")
        if copied_images.get("image_with_grid"):
            lines.append(f"![加密网格]({rel_dir}/{copied_images['image_with_grid']})")
        if copied_images.get("image_with_grid_color"):
            lines.append(f"![彩色网格]({rel_dir}/{copied_images['image_with_grid_color']})")
        lines.extend(["", f"- 完整 JSON：`{rel_dir}/classification_error.json`", ""])

    (output_root / "chart_type_errors.json").write_text(
        json.dumps(
            {
                "error_count": len(errors),
                "by_reason": dict(by_reason),
                "by_dataset": dict(by_dataset),
                "by_gt": dict(by_gt),
                "confusion": [
                    {"gt_type": gt, "pred_type": pred, "count": count}
                    for (gt, pred), count in confusion.most_common()
                ],
                "records": detail_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
        default=Path("backend/evaluation/results/vishintprompt_full_latest_report/chart_type_errors"),
    )
    args = parser.parse_args()
    build_report(args.details, args.manifest, args.output)
    print(f"Wrote {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
