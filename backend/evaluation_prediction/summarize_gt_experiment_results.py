"""Summarize current GT experiment result directories."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.getenv("CHART_GT_RESULTS_DIR", str(PROJECT_ROOT / "gt_runs"))).expanduser()
SUMMARY_CSV = RESULTS_ROOT / "current_experiment_results.csv"
SUMMARY_MD = RESULTS_ROOT / "current_experiment_results.md"


def main() -> None:
    runs = collect_latest_runs()
    rows = [summarize_run(path) for path in runs]
    rows = [row for row in rows if row]
    rows.sort(key=lambda row: (str(row.get("chart_type")), str(row.get("chart_name"))))
    write_csv(rows, SUMMARY_CSV)
    write_markdown(rows, SUMMARY_MD)
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {SUMMARY_MD}")
    print(f"chart_runs={len(rows)}")


def collect_latest_runs() -> list[Path]:
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in RESULTS_ROOT.rglob("*_gt_experiment_result.json"):
        try:
            rel = path.relative_to(RESULTS_ROOT)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) < 4:
            continue
        chart_type, chart_name = parts[0], parts[1]
        grouped[(chart_type, chart_name)].append(path)

    latest: list[Path] = []
    for paths in grouped.values():
        latest.append(max(paths, key=lambda item: item.stat().st_mtime))
    return latest


def summarize_run(path: Path) -> dict[str, Any] | None:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    run_dir = path.parent
    metrics = payload.get("gt_metrics") if isinstance(payload.get("gt_metrics"), dict) else read_json(run_dir / "gt_metrics.json")
    final_metrics = (
        payload.get("full_flow_final_metrics")
        if isinstance(payload.get("full_flow_final_metrics"), dict)
        else summarize_csv_metrics(run_dir)
    )
    stage_coverage = payload.get("stage_coverage") if isinstance(payload.get("stage_coverage"), dict) else read_json(run_dir / "stage_coverage.json")
    records = metrics.get("records") if isinstance(metrics, dict) and isinstance(metrics.get("records"), list) else []
    stage_metrics = summarize_stage_records(records)

    chart_type = str(payload.get("chart_type") or infer_part(path, 0))
    chart_name = str(payload.get("chart_id") or infer_part(path, 1))
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    return {
        "chart_type": chart_type,
        "chart_name": chart_name,
        "model_name": payload.get("model_name"),
        "sample_id": payload.get("sample_id"),
        "run_dir": str(run_dir),
        "updated_at": format_mtime(path),
        "record_count": summary.get("record_count") or metrics.get("record_count"),
        "avg_RE": clean_number(summary.get("avg_RE") or metrics.get("avg_RE")),
        "avg_RNE": clean_number(summary.get("avg_RNE") or metrics.get("avg_RNE")),
        "full_flow_final_record_count": summary.get("full_flow_final_record_count") or final_metrics.get("record_count"),
        "full_flow_final_avg_RE": clean_number(summary.get("full_flow_final_avg_RE") or final_metrics.get("avg_RE")),
        "full_flow_final_avg_RNE": clean_number(summary.get("full_flow_final_avg_RNE") or final_metrics.get("avg_RNE")),
        "baseline_avg_RE": stage_metrics.get("baseline_avg_RE"),
        "baseline_avg_RNE": stage_metrics.get("baseline_avg_RNE"),
        "grid_avg_RNE": stage_metrics.get("grid_avg_RNE"),
        "feedback_avg_RNE": stage_metrics.get("feedback_avg_RNE"),
        "amplifier_avg_RNE": stage_metrics.get("amplifier_avg_RNE"),
        "missing_stage_object_count": count_items(stage_coverage, "missing_stage_objects"),
        "missing_valid_full_flow_object_count": count_items(stage_coverage, "missing_valid_full_flow_objects"),
        "stage_call_violation_object_count": count_items(stage_coverage, "stage_call_violation_objects"),
    }


def summarize_stage_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        stage = str(record.get("prompt_type") or record.get("stage") or "").strip()
        if stage:
            by_stage[stage].append(record)
    for stage in ("baseline", "grid", "feedback", "amplifier"):
        stage_records = by_stage.get(stage, [])
        result[f"{stage}_avg_RE"] = average(record.get("RE") for record in stage_records)
        result[f"{stage}_avg_RNE"] = average(record.get("RNE") for record in stage_records)
    return result


def summarize_csv_metrics(run_dir: Path) -> dict[str, Any]:
    paths = list((run_dir / "process_files").rglob("full_flow_final_predictions.csv"))
    values_re: list[float] = []
    values_rne: list[float] = []
    count = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                count += 1
                for key in ("RE", "re", "x_re", "y_re", "r_re"):
                    value = number_or_none(row.get(key))
                    if value is not None:
                        values_re.append(value)
                        break
                for key in ("RNE", "x_rne", "y_rne", "r_rne"):
                    value = number_or_none(row.get(key))
                    if value is not None:
                        values_rne.append(value)
                        break
    return {
        "record_count": count,
        "avg_RE": average(values_re),
        "avg_RNE": average(values_rne),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "chart_type",
        "chart_name",
        "model_name",
        "sample_id",
        "updated_at",
        "record_count",
        "avg_RE",
        "avg_RNE",
        "full_flow_final_record_count",
        "full_flow_final_avg_RE",
        "full_flow_final_avg_RNE",
        "baseline_avg_RE",
        "baseline_avg_RNE",
        "grid_avg_RNE",
        "feedback_avg_RNE",
        "amplifier_avg_RNE",
        "missing_stage_object_count",
        "missing_valid_full_flow_object_count",
        "stage_call_violation_object_count",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("chart_type"))].append(row)

    lines = [
        "# 当前 GT 实验结果汇总",
        "",
        "本文件由 `backend/evaluation_prediction/summarize_gt_experiment_results.py` 生成。每个图表对象只取当前最新 run。",
        "",
        "说明：`final avg RNE` 只统计已有 full-flow final 的对象；`final 优于 baseline` 只统计 baseline RNE 和 final RNE 都存在、且 final RNE 更低的对象。RE 在 GT 接近 0 时会被分母放大，论文对比建议优先同时查看 RNE。",
        "",
        "## 按类型汇总",
        "",
        "| 类型 | 图表数 | 有 final 对象 | final 优于 baseline | final avg RNE | final avg RE | baseline avg RNE | 缺失 full-flow 对象 | 调用轮次违规对象 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for chart_type in sorted(by_type):
        items = by_type[chart_type]
        lines.append(
            "| {chart_type} | {count} | {with_final} | {better} | {final_rne} | {final_re} | {baseline_rne} | {missing} | {violations} |".format(
                chart_type=chart_type,
                count=len(items),
                with_final=sum(1 for item in items if number_or_none(item.get("full_flow_final_avg_RNE")) is not None),
                better=count_final_better_than_baseline(items),
                final_rne=fmt(average(item.get("full_flow_final_avg_RNE") for item in items)),
                final_re=fmt(average(item.get("full_flow_final_avg_RE") for item in items)),
                baseline_rne=fmt(average(item.get("baseline_avg_RNE") for item in items)),
                missing=sum(int(item.get("missing_valid_full_flow_object_count") or 0) for item in items),
                violations=sum(int(item.get("stage_call_violation_object_count") or 0) for item in items),
            )
        )
    lines.extend(
        [
            "",
            "## 最新对象明细",
            "",
            "| 类型 | 图表 | 模型 | final RNE | final RE | baseline RNE | 缺失 full-flow | 轮次违规 | 更新时间 |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {chart_type} | {chart_name} | {model} | {final_rne} | {final_re} | {baseline_rne} | {missing} | {violations} | {updated} |".format(
                chart_type=row.get("chart_type") or "",
                chart_name=row.get("chart_name") or "",
                model=row.get("model_name") or "",
                final_rne=fmt(row.get("full_flow_final_avg_RNE")),
                final_re=fmt(row.get("full_flow_final_avg_RE")),
                baseline_rne=fmt(row.get("baseline_avg_RNE")),
                missing=row.get("missing_valid_full_flow_object_count") or 0,
                violations=row.get("stage_call_violation_object_count") or 0,
                updated=row.get("updated_at") or "",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_part(path: Path, index: int) -> str:
    try:
        return path.relative_to(RESULTS_ROOT).parts[index]
    except Exception:
        return ""


def count_items(mapping: Any, key: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    value = mapping.get(key)
    return len(value) if isinstance(value, list) else 0


def count_final_better_than_baseline(items: list[dict[str, Any]]) -> int:
    count = 0
    for item in items:
        final_rne = number_or_none(item.get("full_flow_final_avg_RNE"))
        baseline_rne = number_or_none(item.get("baseline_avg_RNE"))
        if final_rne is not None and baseline_rne is not None and final_rne < baseline_rne:
            count += 1
    return count


def average(values: Any) -> float | None:
    numbers = [value for value in (number_or_none(item) for item in values) if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean_number(value: Any) -> float | None:
    return number_or_none(value)


def fmt(value: Any) -> str:
    number = number_or_none(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def format_mtime(path: Path) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()
