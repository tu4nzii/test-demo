from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


CARTESIAN_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_name(value: str, limit: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return (cleaned.strip("._") or "sample")[:limit]


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


def classify_error(gt_type: str, pred_type: str, status: str) -> str:
    if pred_type in {"unknown", "", "none"} or status == "failed":
        return "模型/流程失败，未返回有效类型"
    if {gt_type, pred_type} <= {"scatter", "bubble"}:
        return "散点图/气泡图区分错误"
    if {gt_type, pred_type} <= {"v_bar", "h_bar"}:
        return "柱状图方向误判"
    return "其他类别误判"


def build_report(details_path: Path, manifest_path: Path, output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    details = read_json(details_path)
    manifest = manifest_by_relative(manifest_path)
    errors = [
        row
        for row in details
        if row.get("dataset") == "Sy.Dataset"
        and row.get("gt_type") in CARTESIAN_TYPES
        and row.get("chart_type_total")
        and not row.get("chart_type_correct")
    ]
    confusion = Counter((str(row.get("gt_type")), str(row.get("pred_type"))) for row in errors)
    by_gt = Counter(str(row.get("gt_type")) for row in errors)
    by_reason = Counter(
        classify_error(str(row.get("gt_type")), str(row.get("pred_type")), str(row.get("status")))
        for row in errors
    )

    payloads: list[dict[str, Any]] = []
    for index, row in enumerate(errors, start=1):
        dataset_relative = str(row.get("dataset_relative") or "")
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
        gt_type = str(row.get("gt_type"))
        pred_type = str(row.get("pred_type"))
        payload = {
            "index": index,
            "sample_dir": sample_dir.name,
            "dataset_relative": dataset_relative,
            "gt_type": gt_type,
            "pred_type": pred_type,
            "status": row.get("status"),
            "error_category": classify_error(gt_type, pred_type, str(row.get("status"))),
            "copied_images": images,
            "detail": row,
            "record": record,
        }
        write_json(sample_dir / "classification_error.json", payload)
        payloads.append(payload)

    lines = [
        "# 合成数据集直角坐标系分类错误样本",
        "",
        f"- 来源明细：`{details_path}`",
        f"- Manifest：`{manifest_path}`",
        "- 数据集：`Sy.Dataset`",
        "- 范围：`v_bar`、`h_bar`、`line`、`scatter`、`bubble`",
        "- 口径：只统计图表类别预测错误，即 `gt_type != pred_type`。",
        f"- 错误样本数：{len(errors)}",
        "",
        "## 错误类型概览",
        "",
        "| 错误类别 | 数量 |",
        "| --- | ---: |",
    ]
    for reason, count in by_reason.most_common():
        lines.append(f"| {reason} | {count} |")

    lines.extend(["", "## 按 GT 类型", "", "| GT类型 | 错误数 |", "| --- | ---: |"])
    for gt_type, count in by_gt.most_common():
        lines.append(f"| `{gt_type}` | {count} |")

    lines.extend(["", "## 混淆对", "", "| GT类型 | 预测类型 | 数量 |", "| --- | --- | ---: |"])
    for (gt_type, pred_type), count in confusion.most_common():
        lines.append(f"| `{gt_type}` | `{pred_type}` | {count} |")

    lines.extend(
        [
            "",
            "## 样本明细",
            "",
            "| 序号 | 样本 | GT类型 | 预测类型 | 状态 | 错误类别 | 查看目录 |",
            "| ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for payload in payloads:
        lines.append(
            f"| {payload['index']} | `{payload['dataset_relative']}` | `{payload['gt_type']}` | "
            f"`{payload['pred_type']}` | {payload['status']} | {payload['error_category']} | "
            f"`{payload['sample_dir']}` |"
        )

    lines.extend(["", "## 可视化预览", ""])
    for payload in payloads:
        rel_dir = payload["sample_dir"]
        lines.extend(
            [
                f"### {payload['index']}. {payload['dataset_relative']}",
                "",
                f"- GT类型：`{payload['gt_type']}`；预测类型：`{payload['pred_type']}`",
                f"- 错误类别：{payload['error_category']}",
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
        lines.extend(["", f"- 完整 JSON：`{rel_dir}/classification_error.json`", ""])

    write_json(
        output_root / "classification_errors.json",
        {
            "dataset": "Sy.Dataset",
            "scope": "cartesian",
            "error_count": len(errors),
            "by_reason": dict(by_reason),
            "by_gt": dict(by_gt),
            "confusion": [
                {"gt_type": gt_type, "pred_type": pred_type, "count": count}
                for (gt_type, pred_type), count in confusion.most_common()
            ],
            "records": payloads,
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
        default=Path(
            "backend/evaluation/results/vishintprompt_full_latest_report/"
            "synthetic_cartesian_classification_errors"
        ),
    )
    args = parser.parse_args()
    build_report(args.details, args.manifest, args.output)
    print(f"Wrote {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
