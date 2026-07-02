"""Shared GT-experiment flow and audit helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .json_safety import sanitize_json_value


BASELINE_STAGE = "baseline"
GRID_STAGE = "grid"
FEEDBACK_STAGE = "feedback"
AMPLIFIER_STAGE = "amplifier"
FULL_FLOW_STAGES = (GRID_STAGE, FEEDBACK_STAGE, AMPLIFIER_STAGE)
AUDIT_STAGES = (BASELINE_STAGE, GRID_STAGE, FEEDBACK_STAGE, AMPLIFIER_STAGE)
STAGE_CALL_LIMITS = {
    BASELINE_STAGE: (1, 1),
    GRID_STAGE: (1, 1),
    FEEDBACK_STAGE: (1, 2),
    AMPLIFIER_STAGE: (1, 3),
}


def normalize_stage(prompt_type: Any) -> str:
    normalized = str(prompt_type or "").strip().lower()
    if normalized in {"feedback_crop_adaptive", "feedback_crop"}:
        return AMPLIFIER_STAGE
    return normalized


def summarize_stage_coverage(metric_records: list[dict[str, Any]]) -> dict[str, Any]:
    expected = set(AUDIT_STAGES)
    by_object: dict[str, dict[str, Any]] = {}
    for record in metric_records:
        processing_object = str(record.get("processing_object") or "__unknown__")
        stage = normalize_stage(record.get("stage") or record.get("prompt_type"))
        if not stage:
            continue
        item = by_object.setdefault(
            processing_object,
            {
                "processing_object": processing_object,
                "stages": {},
                "missing_required_stages": [],
                "missing_valid_full_flow_stages": [],
                "stage_call_violations": [],
            },
        )
        stages = item["stages"]
        stage_info = stages.setdefault(
            stage,
            {
                "call_count": 0,
                "round_count": 0,
                "rounds": [],
                "valid_call_count": 0,
                "valid_round_count": 0,
                "valid_rounds": [],
            },
        )
        is_modal_call = bool(str(record.get("call_id") or "").strip())
        if is_modal_call:
            stage_info["call_count"] += 1
        round_value = record.get("round")
        if is_modal_call and round_value not in stage_info["rounds"]:
            stage_info["rounds"].append(round_value)
            stage_info["round_count"] = len(stage_info["rounds"])
        if record.get("valid_prediction", True):
            if is_modal_call:
                stage_info["valid_call_count"] += 1
            if is_modal_call and round_value not in stage_info["valid_rounds"]:
                stage_info["valid_rounds"].append(round_value)
                stage_info["valid_round_count"] = len(stage_info["valid_rounds"])

    for item in by_object.values():
        present = set(item["stages"])
        item["missing_required_stages"] = sorted(expected - present)
        item["missing_valid_full_flow_stages"] = [
            stage
            for stage in FULL_FLOW_STAGES
            if item["stages"].get(stage, {}).get("valid_round_count", 0) == 0
        ]
        violations = []
        for stage, (minimum, maximum) in STAGE_CALL_LIMITS.items():
            round_count = item["stages"].get(stage, {}).get("round_count", 0)
            if round_count < minimum or round_count > maximum:
                violations.append(
                    {
                        "stage": stage,
                        "call_count": round_count,
                        "modal_call_count": item["stages"].get(stage, {}).get("call_count", 0),
                        "expected_min": minimum,
                        "expected_max": maximum,
                    }
                )
        item["stage_call_violations"] = violations
    return {
        "expected_full_flow_stages": list(FULL_FLOW_STAGES),
        "expected_audit_stages": list(AUDIT_STAGES),
        "stage_call_limits": {
            stage: {"min": minimum, "max": maximum}
            for stage, (minimum, maximum) in STAGE_CALL_LIMITS.items()
        },
        "object_count": len(by_object),
        "objects": list(by_object.values()),
        "missing_stage_objects": [
            item
            for item in by_object.values()
            if item.get("missing_required_stages")
        ],
        "missing_valid_full_flow_objects": [
            item
            for item in by_object.values()
            if item.get("missing_valid_full_flow_stages")
        ],
        "stage_call_violation_objects": [
            item
            for item in by_object.values()
            if item.get("stage_call_violations")
        ],
    }


def write_metric_artifacts(
    run_dir: Path,
    metrics: dict[str, Any],
    stage_coverage: dict[str, Any],
) -> dict[str, str]:
    metrics_json_path = run_dir / "gt_metrics.json"
    stage_json_path = run_dir / "stage_coverage.json"
    metrics_csv_path = run_dir / "gt_metric_records.csv"
    _write_json(metrics_json_path, metrics)
    _write_json(stage_json_path, stage_coverage)

    records = metrics.get("records")
    if isinstance(records, list) and records:
        columns = [
            "call_id",
            "chart_name",
            "processing_object",
            "object_category",
            "prompt_type",
            "stage",
            "round",
            "image_type",
            "image_path",
            "valid_prediction",
            "prediction_readable",
            "RE",
            "RNE",
        ]
        with metrics_csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    else:
        metrics_csv_path.write_text("", encoding="utf-8")

    return {
        "gt_metrics_json": str(metrics_json_path),
        "gt_metric_records_csv": str(metrics_csv_path),
        "stage_coverage_json": str(stage_json_path),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(sanitize_json_value(data), file, ensure_ascii=False, indent=2, allow_nan=False)
