"""Audit the current Cartesian full-pipeline evidence for reviewer response.

This script intentionally does not run the old axis/tick scanner. It inspects
the latest full grid-generation/evaluation artifacts produced by the current
runtime pipeline:

- three candidate grids: combined_mask, tick_supplement, semantic_guide
- score-first priority decisions and exit/failure reports
- final bindings used by the evaluation metrics
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LATEST_RECHECK = ROOT / "backend" / "evaluation" / "recheck_outputs" / "vishintprompt_full_grid_encryption_latest"
LATEST_REPORT = ROOT / "backend" / "evaluation" / "results" / "vishintprompt_full_latest_report"
OUTPUT_DIR = ROOT / "review"

CARTESIAN_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble"}
EXPECTED_SOURCES = {"combined_mask", "tick_supplement", "semantic_guide"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except Exception:
        return None


def summarize_grid_effect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_json(LATEST_REPORT / "summary.json")
    cart_rows = [row for row in rows if row.get("gt_type") in CARTESIAN_TYPES]
    typed_rows = []
    for row in cart_rows:
        typed_rows.append(
            {
                "dataset": row.get("dataset"),
                "gt_type": row.get("gt_type"),
                "sample_count": int(float(row.get("sample_count") or 0)),
                "processed_count": int(float(row.get("processed_count") or 0)),
                "tick_value_mae_px": safe_float(row.get("tick_value_mae_px")),
                "tick_value_accuracy_2px": safe_float(row.get("tick_value_accuracy_2px")),
                "label_name_accuracy": safe_float(row.get("label_name_accuracy")),
            }
        )
    numeric_total = sum(float(row.get("numeric_total") or 0) for row in cart_rows)
    numeric_matched = sum(float(row.get("numeric_matched") or 0) for row in cart_rows)
    numeric_correct = sum(
        (safe_float(row.get("tick_value_accuracy_2px")) or 0.0)
        * float(row.get("numeric_total") or 0)
        for row in cart_rows
    )
    numeric_error_sum = sum(
        (safe_float(row.get("tick_value_mae_px")) or 0.0)
        * float(row.get("numeric_matched") or 0)
        for row in cart_rows
    )
    position_matched = sum(float(row.get("tick_position_matched") or 0) for row in cart_rows)
    position_error_sum = sum(
        (safe_float(row.get("tick_position_mae_px")) or 0.0)
        * float(row.get("tick_position_matched") or 0)
        for row in cart_rows
    )
    label_total = sum(float(row.get("label_total") or 0) for row in cart_rows)
    label_correct = sum(
        (safe_float(row.get("label_name_accuracy")) or 0.0)
        * float(row.get("label_total") or 0)
        for row in cart_rows
    )
    overall_summary = {
        "sample_count": int(sum(float(row.get("sample_count") or 0) for row in cart_rows)),
        "processed_count": int(sum(float(row.get("processed_count") or 0) for row in cart_rows)),
        "tick_value_mae_px": (numeric_error_sum / numeric_matched) if numeric_matched else None,
        "tick_value_accuracy_2px": (numeric_correct / numeric_total) if numeric_total else None,
        "tick_position_mae_px": (position_error_sum / position_matched) if position_matched else None,
        "label_name_accuracy": (label_correct / label_total) if label_total else None,
    }
    return typed_rows, overall_summary


def inspect_decision_file(path: Path) -> dict[str, Any]:
    data = read_json(path)
    x_scores = data.get("x_scores") if isinstance(data.get("x_scores"), dict) else {}
    y_scores = data.get("y_scores") if isinstance(data.get("y_scores"), dict) else {}
    x_sources = set(x_scores)
    y_sources = set(y_scores)
    return {
        "path": str(path.relative_to(ROOT)),
        "selection_policy": data.get("selection_policy"),
        "mllm_used": bool(data.get("mllm_used")),
        "x_choice": data.get("x_axis_vertical_grid_choice"),
        "y_choice": data.get("y_axis_horizontal_grid_choice"),
        "x_reason": data.get("x_axis_reason"),
        "y_reason": data.get("y_axis_reason"),
        "x_sources": sorted(x_sources),
        "y_sources": sorted(y_sources),
        "has_all_three_x": EXPECTED_SOURCES.issubset(x_sources),
        "has_all_three_y": EXPECTED_SOURCES.issubset(y_sources),
    }


def inspect_failure_file(path: Path) -> dict[str, Any]:
    try:
        data = read_json(path)
    except Exception:
        data = {}
    return {
        "path": str(path.relative_to(ROOT)),
        "failed": data.get("failed", True),
        "reason": data.get("reason") or data.get("grid_eval_skip_reason") or data.get("error") or "",
    }


def summarize_artifacts() -> dict[str, Any]:
    decision_paths = sorted(LATEST_RECHECK.rglob("*_grid_priority_decision.json"))
    final_binding_paths = sorted(LATEST_RECHECK.rglob("*_final*_bindings.json"))
    candidate_binding_paths = sorted(LATEST_RECHECK.rglob("*_priority*_grid_bindings.json"))
    failure_paths = sorted(LATEST_RECHECK.rglob("*_grid_failure.json"))
    selection_paths = sorted(LATEST_RECHECK.rglob("*_final*_selection.json"))

    decisions = [inspect_decision_file(path) for path in decision_paths]
    failures = [inspect_failure_file(path) for path in failure_paths]
    actual_failures = [
        row
        for row in failures
        if row.get("failed") is True or (str(row.get("reason") or "").strip() not in {"", "ok"})
    ]
    x_choice_counts = Counter(str(row.get("x_choice")) for row in decisions)
    y_choice_counts = Counter(str(row.get("y_choice")) for row in decisions)
    policy_counts = Counter(str(row.get("selection_policy")) for row in decisions)

    complete_decisions = [
        row for row in decisions if row.get("has_all_three_x") and row.get("has_all_three_y")
    ]

    return {
        "latest_recheck_dir": str(LATEST_RECHECK),
        "decision_count": len(decision_paths),
        "decision_with_all_three_sources": len(complete_decisions),
        "final_binding_file_count": len(final_binding_paths),
        "candidate_binding_file_count": len(candidate_binding_paths),
        "selection_file_count": len(selection_paths),
        "grid_status_report_file_count": len(failure_paths),
        "actual_failure_or_exit_report_count": len(actual_failures),
        "mllm_used_count": sum(1 for row in decisions if row.get("mllm_used")),
        "selection_policy_counts": dict(policy_counts),
        "x_choice_counts": dict(x_choice_counts),
        "y_choice_counts": dict(y_choice_counts),
        "sample_decisions": decisions[:10],
        "sample_failures": actual_failures[:20],
    }


def write_markdown(path: Path, metrics_rows: list[dict[str, Any]], overall: dict[str, Any], artifacts: dict[str, Any]) -> None:
    lines = [
        "# Current Cartesian Full-Pipeline Evidence",
        "",
        "This report verifies that the Cartesian evidence used in the reviewer response comes from the current full pipeline, not from the legacy axis/tick scanning path.",
        "",
        "## Pipeline Evidence",
        "",
        f"- Latest recheck directory: `{artifacts['latest_recheck_dir']}`",
        f"- Priority decision files: {artifacts['decision_count']}",
        f"- Decisions containing all three sources (`combined_mask`, `tick_supplement`, `semantic_guide`) for both axes: {artifacts['decision_with_all_three_sources']}",
        f"- Candidate grid binding files: {artifacts['candidate_binding_file_count']}",
        f"- Final binding files: {artifacts['final_binding_file_count']}",
        f"- Final selection files: {artifacts['selection_file_count']}",
        f"- Grid status report files: {artifacts['grid_status_report_file_count']}",
        f"- Actual failure/exit reports: {artifacts['actual_failure_or_exit_report_count']}",
        f"- MLLM arbitration used after score prefill: {artifacts['mllm_used_count']}",
        "",
        "Selection policies observed:",
        "",
    ]
    for policy, count in sorted(artifacts["selection_policy_counts"].items()):
        lines.append(f"- `{policy}`: {count}")
    lines.extend(
        [
            "",
            "Axis source choices:",
            "",
            "| Source | X-axis choice count | Y-axis choice count |",
            "| --- | ---: | ---: |",
        ]
    )
    all_sources = sorted(set(artifacts["x_choice_counts"]) | set(artifacts["y_choice_counts"]))
    for source in all_sources:
        lines.append(
            f"| `{source}` | {artifacts['x_choice_counts'].get(source, 0)} | {artifacts['y_choice_counts'].get(source, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Full-Pipeline Cartesian Metrics",
            "",
            "| Dataset | Type | Samples | Processed | Tick MAE(px) | Tick Acc@2px | Label Acc |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics_rows:
        lines.append(
            "| {dataset} | {gt_type} | {sample_count} | {processed_count} | {tick_mae} | {tick_acc} | {label_acc} |".format(
                dataset=row["dataset"],
                gt_type=row["gt_type"],
                sample_count=row["sample_count"],
                processed_count=row["processed_count"],
                tick_mae=num(row["tick_value_mae_px"]),
                tick_acc=pct(row["tick_value_accuracy_2px"]),
                label_acc=pct(row["label_name_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "Overall Cartesian full-pipeline summary:",
            "",
            f"- Samples: {overall['sample_count']}",
            f"- Processed: {overall['processed_count']}",
            f"- Tick MAE: {num(overall['tick_value_mae_px'])} px",
            f"- Tick Acc@2px: {pct(overall['tick_value_accuracy_2px'])}",
            f"- Tick position MAE: {num(overall['tick_position_mae_px'])} px",
            f"- Label accuracy: {pct(overall['label_name_accuracy'])}",
            "",
            "## Interpretation",
            "",
            "The parameter sensitivity sweep in this review folder is only a legacy low-level Canny/Hough candidate-generator diagnostic. The final Cartesian results above come from the active enhanced-grid-first runtime artifacts: three candidate grids are scored, unreliable cases can produce failure/exit reports, and evaluation reads generated final bindings.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_zh(path: Path, metrics_rows: list[dict[str, Any]], overall: dict[str, Any], artifacts: dict[str, Any]) -> None:
    lines = [
        "# 当前直角系全链路证据",
        "",
        "本报告用于确认审稿回复中的直角系证据来自当前完整流程，而不是旧版坐标轴/tick 扫描路径。",
        "",
        "## 流程证据",
        "",
        f"- 最新 recheck 目录：`{artifacts['latest_recheck_dir']}`",
        f"- priority decision 文件数：{artifacts['decision_count']}",
        f"- x/y 两个方向都包含三类来源（`combined_mask`、`tick_supplement`、`semantic_guide`）的 decision 数：{artifacts['decision_with_all_three_sources']}",
        f"- 候选网格 binding 文件数：{artifacts['candidate_binding_file_count']}",
        f"- final binding 文件数：{artifacts['final_binding_file_count']}",
        f"- final selection 文件数：{artifacts['selection_file_count']}",
        f"- 网格状态报告文件数：{artifacts['grid_status_report_file_count']}",
        f"- 实际 failure/exit 报告数：{artifacts['actual_failure_or_exit_report_count']}",
        f"- score prefill 后使用 MLLM 仲裁的次数：{artifacts['mllm_used_count']}",
        "",
        "观察到的 selection policy：",
        "",
    ]
    for policy, count in sorted(artifacts["selection_policy_counts"].items()):
        lines.append(f"- `{policy}`：{count}")
    lines.extend(
        [
            "",
            "轴方向来源选择统计：",
            "",
            "| 来源 | X 轴方向选择次数 | Y 轴方向选择次数 |",
            "| --- | ---: | ---: |",
        ]
    )
    all_sources = sorted(set(artifacts["x_choice_counts"]) | set(artifacts["y_choice_counts"]))
    for source in all_sources:
        lines.append(
            f"| `{source}` | {artifacts['x_choice_counts'].get(source, 0)} | {artifacts['y_choice_counts'].get(source, 0)} |"
        )
    lines.extend(
        [
            "",
            "## 直角系完整流程指标",
            "",
            "| 数据集 | 类型 | 样本数 | 已处理 | Tick MAE(px) | Tick Acc@2px | Label Acc |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics_rows:
        lines.append(
            "| {dataset} | {gt_type} | {sample_count} | {processed_count} | {tick_mae} | {tick_acc} | {label_acc} |".format(
                dataset=row["dataset"],
                gt_type=row["gt_type"],
                sample_count=row["sample_count"],
                processed_count=row["processed_count"],
                tick_mae=num(row["tick_value_mae_px"]),
                tick_acc=pct(row["tick_value_accuracy_2px"]),
                label_acc=pct(row["label_name_accuracy"]),
            )
        )
    lines.extend(
        [
            "",
            "直角系完整流程总体结果：",
            "",
            f"- 样本数：{overall['sample_count']}",
            f"- 已处理：{overall['processed_count']}",
            f"- Tick MAE：{num(overall['tick_value_mae_px'])} px",
            f"- Tick Acc@2px：{pct(overall['tick_value_accuracy_2px'])}",
            f"- Tick position MAE：{num(overall['tick_position_mae_px'])} px",
            f"- Label accuracy：{pct(overall['label_name_accuracy'])}",
            "",
            "## 解释",
            "",
            "`review/` 中的参数敏感性扫描只是一项旧版低层 Canny/Hough 候选生成器诊断。上面的最终直角系结果来自当前 active enhanced-grid-first 运行时 artifacts：三套候选网格经过 score 筛选，不可靠样本产生 failure/exit 报告，评估读取生成端输出的 final bindings。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_rows, overall = summarize_grid_effect()
    artifacts = summarize_artifacts()
    output = {
        "metrics_rows": metrics_rows,
        "overall": overall,
        "artifacts": artifacts,
    }
    (OUTPUT_DIR / "current_cartesian_full_pipeline_evidence.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(OUTPUT_DIR / "current_cartesian_full_pipeline_evidence.md", metrics_rows, overall, artifacts)
    write_markdown_zh(OUTPUT_DIR / "current_cartesian_full_pipeline_evidence_zh.md", metrics_rows, overall, artifacts)
    print(
        json.dumps(
            {
                "decision_count": artifacts["decision_count"],
                "decision_with_all_three_sources": artifacts["decision_with_all_three_sources"],
                "final_binding_file_count": artifacts["final_binding_file_count"],
                "grid_status_report_file_count": artifacts["grid_status_report_file_count"],
                "actual_failure_or_exit_report_count": artifacts["actual_failure_or_exit_report_count"],
                "overall": overall,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
