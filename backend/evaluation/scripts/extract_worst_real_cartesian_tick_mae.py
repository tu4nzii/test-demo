from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


CARTESIAN_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}
GOOD_STATUSES = {"success", "skipped_success_cache", "recovered_from_grid_reference"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_if_exists(src: Any, dst: Path) -> str | None:
    if not src:
        return None
    path = Path(str(src))
    if not path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst.name


def tick_mae(row: dict[str, Any]) -> float | None:
    matched = float(row.get("numeric_matched") or 0)
    if matched <= 0:
        return None
    return float(row.get("numeric_error_sum") or 0) / matched


def ratio(row: dict[str, Any], numerator: str, denominator: str) -> float | None:
    denom = float(row.get(denominator) or 0)
    if denom <= 0:
        return None
    return float(row.get(numerator) or 0) / denom


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def main() -> int:
    details_path = Path("backend/evaluation/results/vishintprompt_full_latest_report/details.json")
    manifest_path = Path("backend/evaluation/recheck_outputs/vishintprompt_full_grid_encryption_latest/manifest.json")
    output_root = Path("backend/evaluation/results/vishintprompt_full_latest_report/worst_real_cartesian_tick_mae")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    details = read_json(details_path)
    manifest = read_json(manifest_path)
    records = {
        str(record.get("dataset_relative")): record
        for record in manifest.get("records", [])
        if isinstance(record, dict) and record.get("dataset_relative")
    }

    candidates = []
    for row in details:
        value = tick_mae(row)
        if value is None:
            continue
        if row.get("dataset") != "Final-RealDataset":
            continue
        if row.get("gt_type") not in CARTESIAN_TYPES:
            continue
        if row.get("status") not in GOOD_STATUSES:
            continue
        if not row.get("grid_eval_included") or row.get("grid_eval_skipped"):
            continue
        candidates.append((value, row))

    if not candidates:
        raise RuntimeError("No included Final-RealDataset Cartesian samples with numeric tick metrics.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_mae, selected = candidates[0]
    dataset_relative = str(selected.get("dataset_relative"))
    record = records.get(dataset_relative, {})
    copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
    files = {
        "source": copy_if_exists(
            record.get("source_image") or copied.get("source_image") or copied.get("uploaded_image"),
            output_root / "source.png",
        ),
        "image_grid": copy_if_exists(copied.get("basic_grid"), output_root / "image_grid.png"),
        "image_with_grid": copy_if_exists(copied.get("encrypted_grid"), output_root / "image_with_grid.png"),
        "image_with_grid_color": copy_if_exists(
            copied.get("colored_grid"),
            output_root / "image_with_grid_color.png",
        ),
        "ticks": copy_if_exists(copied.get("ticks_json"), output_root / "ticks.json"),
    }

    tick_acc = ratio(selected, "numeric_correct", "numeric_total")
    pos_mae = ratio(selected, "position_error_sum", "tick_position_matched")
    label_acc = ratio(selected, "label_correct", "label_total")
    summary = {
        "selected_reason": "Final-RealDataset Cartesian included sample with maximum Tick MAE.",
        "dataset_relative": dataset_relative,
        "gt_type": selected.get("gt_type"),
        "pred_type": selected.get("pred_type"),
        "status": selected.get("status"),
        "tick_mae_px": selected_mae,
        "tick_acc_at_2px": tick_acc,
        "numeric_matched": selected.get("numeric_matched"),
        "numeric_total": selected.get("numeric_total"),
        "numeric_correct": selected.get("numeric_correct"),
        "pos_mae_px": pos_mae,
        "label_acc": label_acc,
        "files": files,
        "detail": selected,
        "record": record,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 真实直角系 Tick MAE 最差样本",
        "",
        f"- 样本：`{dataset_relative}`",
        "- 口径：Final-RealDataset / 直角系 / 未剔除且已计入评估 / Tick MAE 最大",
        f"- GT类型：`{selected.get('gt_type')}`；预测类型：`{selected.get('pred_type')}`",
        f"- Tick MAE(px)：{selected_mae:.3f}",
        f"- Tick Acc@2px：{fmt_pct(tick_acc)}",
        (
            f"- 数值 tick：matched={selected.get('numeric_matched')} / "
            f"total={selected.get('numeric_total')} / correct={selected.get('numeric_correct')}"
        ),
        f"- Pos MAE(px)：{pos_mae:.3f}" if pos_mae is not None else "- Pos MAE(px)：-",
        f"- 标签准确率：{fmt_pct(label_acc)}",
        "",
    ]
    if files["source"]:
        lines.append("![原图](source.png)")
    if files["image_grid"]:
        lines.append("![原生网格](image_grid.png)")
    if files["image_with_grid"]:
        lines.append("![加密网格](image_with_grid.png)")
    if files["image_with_grid_color"]:
        lines.append("![彩色网格](image_with_grid_color.png)")
    lines.extend(["", "- 完整信息：`summary.json`"])
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(output_root)
    print(json.dumps({key: summary[key] for key in (
        "dataset_relative",
        "tick_mae_px",
        "tick_acc_at_2px",
        "numeric_matched",
        "numeric_total",
        "numeric_correct",
        "pos_mae_px",
        "label_acc",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
