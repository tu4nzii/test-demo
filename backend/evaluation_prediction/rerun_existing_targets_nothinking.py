"""Rerun previously executed GT experiment objects with the active model.

The script is intentionally API-based so it exercises the same backend path as
the frontend. It is resumable: each sample is recorded after every attempt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(os.getenv("CHART_GT_RESULTS_DIR", str(PROJECT_ROOT / "gt_runs"))).expanduser()
DEFAULT_TARGETS = RESULTS_ROOT / "rerun_targets_from_existing_results.json"
DEFAULT_REPORT = RESULTS_ROOT / "nothinking_rerun_report.json"
DEFAULT_CSV = RESULTS_ROOT / "nothinking_rerun_report.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    targets = read_targets(args.targets)
    if args.category:
        wanted = {item.strip().lower() for item in args.category if item.strip()}
        targets = [item for item in targets if str(item.get("category", "")).lower() in wanted]
    if args.limit > 0:
        targets = targets[: args.limit]

    report = read_report(args.report) if args.resume else {"runs": []}
    completed = {
        str(item.get("sample_id"))
        for item in report.get("runs", [])
        if item.get("passed") is True
    }

    print(f"targets={len(targets)} completed={len(completed)}")
    for index, target in enumerate(targets, start=1):
        sample_id = str(target.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if args.resume and sample_id in completed:
            print(f"[{index}/{len(targets)}] skip passed {label(target)}")
            continue

        best: dict[str, Any] | None = None
        for attempt in range(1, max(1, args.max_attempts) + 1):
            print(f"[{index}/{len(targets)}] attempt {attempt}: {label(target)} sample_id={sample_id}", flush=True)
            started = time.perf_counter()
            payload, error = call_run_api(args.base_url, sample_id, timeout=args.timeout)
            elapsed = round(time.perf_counter() - started, 3)
            row = summarize_payload(target, payload, error=error, attempt=attempt, elapsed_s=elapsed)
            append_run(report, row)
            write_report(args.report, report)
            write_csv(args.csv, report.get("runs", []))

            if best is None or score(row) < score(best):
                best = row
            print(
                "    status={status} passed={passed} baseline_RNE={baseline} final_RNE={final} run_dir={run_dir}".format(
                    status=row.get("status"),
                    passed=row.get("passed"),
                    baseline=fmt(row.get("baseline_avg_RNE")),
                    final=fmt(row.get("full_flow_final_avg_RNE")),
                    run_dir=row.get("run_dir") or "",
                ),
                flush=True,
            )
            if row.get("passed") is True:
                break

        if best is not None and best.get("passed") is not True:
            print(f"    needs_attention: {label(target)} best_final_RNE={fmt(best.get('full_flow_final_avg_RNE'))}")

    write_summary(args.report, report)
    write_csv(args.csv, report.get("runs", []))
    print_summary(report)


def read_targets(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Targets file must contain a list: {path}")
    return [item for item in data if isinstance(item, dict)]


def read_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"runs": []}
    return data if isinstance(data, dict) else {"runs": []}


def call_run_api(base_url: str, sample_id: str, *, timeout: int) -> tuple[dict[str, Any] | None, str | None]:
    query = urllib.parse.urlencode({"sample_id": sample_id})
    url = f"{base_url.rstrip('/')}/api/gt-experiment/run/?{query}"
    request = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return None, f"HTTP {exc.code}: {body[:500]}"
    except Exception as exc:
        return None, repr(exc)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, f"non-json response: {raw[:500]}"
    return payload if isinstance(payload, dict) else None, None


def summarize_payload(
    target: dict[str, Any],
    payload: dict[str, Any] | None,
    *,
    error: str | None,
    attempt: int,
    elapsed_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "category": target.get("category"),
        "chart_id": target.get("chart_id"),
        "sample_id": target.get("sample_id"),
        "attempt": attempt,
        "elapsed_s": elapsed_s,
    }
    if error or not isinstance(payload, dict):
        row.update({"status": "error", "error": error or "empty payload", "passed": False})
        return row

    metrics = payload.get("gt_metrics") if isinstance(payload.get("gt_metrics"), dict) else {}
    records = metrics.get("records") if isinstance(metrics.get("records"), list) else []
    stage = stage_metrics(records)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    final_rne = number_or_none(summary.get("full_flow_final_avg_RNE"))
    final_re = number_or_none(summary.get("full_flow_final_avg_RE"))
    baseline_rne = stage.get("baseline_avg_RNE")
    baseline_re = stage.get("baseline_avg_RE")
    passed = (
        payload.get("success") is True
        and final_rne is not None
        and baseline_rne is not None
        and (final_rne < baseline_rne or abs(final_rne - baseline_rne) <= 1e-12)
    )
    row.update(
        {
            "status": "ok" if payload.get("success") else "failed_payload",
            "success": payload.get("success"),
            "model_name": payload.get("model_name"),
            "run_dir": payload.get("run_dir") or payload.get("chart_run_dir"),
            "baseline_avg_RNE": baseline_rne,
            "baseline_avg_RE": baseline_re,
            "full_flow_final_avg_RNE": final_rne,
            "full_flow_final_avg_RE": final_re,
            "final_better_by_RNE": (
                baseline_rne - final_rne if baseline_rne is not None and final_rne is not None else None
            ),
            "final_better_by_RE": (
                baseline_re - final_re if baseline_re is not None and final_re is not None else None
            ),
            "passed": passed,
            "missing_valid_full_flow_object_count": summary.get("missing_valid_full_flow_object_count"),
            "stage_call_violation_object_count": summary.get("stage_call_violation_object_count"),
            "record_count": summary.get("record_count"),
            "full_flow_final_record_count": summary.get("full_flow_final_record_count"),
            "error": None,
        }
    )
    return row


def stage_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        stage = str(record.get("stage") or record.get("prompt_type") or "").strip()
        if stage:
            by_stage.setdefault(stage, []).append(record)
    return {
        "baseline_avg_RNE": average(item.get("RNE") for item in by_stage.get("baseline", [])),
        "baseline_avg_RE": average(item.get("RE") for item in by_stage.get("baseline", [])),
    }


def append_run(report: dict[str, Any], row: dict[str, Any]) -> None:
    report.setdefault("runs", []).append(row)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(path: Path, report: dict[str, Any]) -> None:
    runs = report.get("runs", [])
    latest = latest_by_sample(runs)
    report["summary"] = {
        "latest_run_count": len(latest),
        "passed_count": sum(1 for item in latest if item.get("passed") is True),
        "needs_attention_count": sum(1 for item in latest if item.get("passed") is not True),
        "by_category": dict(Counter(str(item.get("category")) for item in latest)),
    }
    report["needs_attention"] = [item for item in latest if item.get("passed") is not True]
    write_report(path, report)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "category",
        "chart_id",
        "sample_id",
        "attempt",
        "status",
        "success",
        "passed",
        "model_name",
        "baseline_avg_RNE",
        "full_flow_final_avg_RNE",
        "final_better_by_RNE",
        "baseline_avg_RE",
        "full_flow_final_avg_RE",
        "final_better_by_RE",
        "missing_valid_full_flow_object_count",
        "stage_call_violation_object_count",
        "record_count",
        "full_flow_final_record_count",
        "elapsed_s",
        "error",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latest_by_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id"))
        latest[sample_id] = row
    return list(latest.values())


def print_summary(report: dict[str, Any]) -> None:
    latest = latest_by_sample(report.get("runs", []))
    passed = [item for item in latest if item.get("passed") is True]
    failed = [item for item in latest if item.get("passed") is not True]
    print(f"latest={len(latest)} passed={len(passed)} needs_attention={len(failed)}")
    for item in failed[:50]:
        print(
            "needs_attention {category}/{chart_id} sample_id={sample_id} baseline_RNE={baseline} final_RNE={final} error={error}".format(
                category=item.get("category"),
                chart_id=item.get("chart_id"),
                sample_id=item.get("sample_id"),
                baseline=fmt(item.get("baseline_avg_RNE")),
                final=fmt(item.get("full_flow_final_avg_RNE")),
                error=item.get("error") or "",
            )
        )


def score(row: dict[str, Any]) -> float:
    final = number_or_none(row.get("full_flow_final_avg_RNE"))
    if final is None:
        return float("inf")
    return final


def label(target: dict[str, Any]) -> str:
    return f"{target.get('category')}/{target.get('chart_id')}"


def average(values: Any) -> float | None:
    numbers = [item for item in (number_or_none(value) for value in values) if item is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt(value: Any) -> str:
    number = number_or_none(value)
    return "" if number is None else f"{number:.6f}"


if __name__ == "__main__":
    main()
