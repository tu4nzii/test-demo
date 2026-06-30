from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from main import (  # noqa: E402
    dataset_evaluation_cache_path,
    evaluation_cache_usable,
    evaluate_processed_chart,
    iter_dataset_samples,
    load_json,
    process_chart_image,
    register_dataset_sample,
)


CHART_TYPES = ["rose", "radar", "v_bar", "h_bar", "line", "scatter", "bubble", "donut", "pie"]
FORBIDDEN_GENERATION_KEYS = {"data", "data_points", "ground_truth", "labels"}
OUT_DIR = BACKEND_ROOT / "evaluation" / "results" / "full_chain_smoke_latest"


def _cached_prediction_score(sample: dict[str, Any]) -> tuple[int, int, str]:
    cache_path = dataset_evaluation_cache_path(str(sample["sample_id"]))
    if not evaluation_cache_usable(cache_path, str(sample.get("chart_type") or "")):
        return (1, 999999, str(sample.get("name") or ""))
    try:
        data = load_json(cache_path)
        predictions = data.get("predictions") if isinstance(data.get("predictions"), list) else []
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        clean_runner = (
            bool(data.get("success"))
            and len(predictions) > 0
            and int(summary.get("chart_runs") or 0) > 0
            and not bool(summary.get("system_cv_fallback"))
            and not data.get("prediction_runner_error")
        )
        return (0 if clean_runner else 1, len(predictions) if predictions else 999999, str(sample.get("name") or ""))
    except Exception:
        return (1, 999999, str(sample.get("name") or ""))


def _sample_by_type(source: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {chart_type: [] for chart_type in CHART_TYPES}
    for sample in iter_dataset_samples(source):
        chart_type = str(sample.get("chart_type") or "")
        if chart_type in grouped:
            grouped[chart_type].append(sample)
    return {
        chart_type: sorted(samples, key=_cached_prediction_score)[0]
        for chart_type, samples in grouped.items()
        if samples
    }


def _json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for item in payload["records"]:
        rows.append(
            "| {chart_type} | {sample_name} | {status} | {prediction_count} | {chart_runs} | "
            "{system_cv_fallback} | {encrypted_exists} | {results_exists} | {elapsed_sec:.3f} | {error} |".format(
                chart_type=item.get("chart_type", ""),
                sample_name=item.get("sample_name", ""),
                status=item.get("status", ""),
                prediction_count=item.get("prediction_count", 0),
                chart_runs=item.get("chart_runs", 0),
                system_cv_fallback=item.get("system_cv_fallback", False),
                encrypted_exists=item.get("encrypted_exists", False),
                results_exists=item.get("results_exists", False),
                elapsed_sec=float(item.get("elapsed_sec") or 0),
                error=item.get("prediction_runner_error") or item.get("error") or "",
            )
        )

    text = "\n".join(
        [
            "# 全图表类型全链路可用性检查",
            "",
            f"生成时间：{payload['generated_at']}",
            f"数据来源：{payload['source']}",
            f"检查方式：每类 1 张，执行注册、网格/加密处理、3.评估预测；生成端 JSON 禁止包含 {', '.join(sorted(FORBIDDEN_GENERATION_KEYS))}。",
            "",
            f"总计：{payload['total']}，通过：{payload['passed']}，失败：{payload['failed']}",
            "",
            "| 类型 | 样例 | 状态 | 预测数 | chart_runs | CV fallback | 加密图 | 结果 JSON | 耗时(s) | 错误 |",
            "|---|---|---|---:|---:|---|---|---|---:|---|",
            *rows,
            "",
            "说明：`chart_runs > 0` 且 `CV fallback=False` 表示对应类型的 3.评估预测 runner 正常产生了系统预测结果。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


async def _check_one(chart_type: str, sample: dict[str, Any], fresh_evaluation: bool) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "chart_type": chart_type,
        "sample_id": sample.get("sample_id"),
        "sample_name": sample.get("name"),
        "sample_relative_path": sample.get("relative_path"),
        "source": sample.get("source"),
    }
    try:
        eval_cache_path = dataset_evaluation_cache_path(str(sample["sample_id"]))
        record["evaluation_cached_before"] = eval_cache_path.exists()
        if fresh_evaluation and eval_cache_path.exists():
            eval_cache_path.unlink()
            record["evaluation_cache_deleted"] = True

        chart_info = register_dataset_sample(str(sample["sample_id"]))
        record.update(
            {
                "chart_id": chart_info.get("chart_id"),
                "registered_chart_type": chart_info.get("chart_type"),
                "coordinate_system": chart_info.get("coordinate_system"),
                "registered": True,
            }
        )

        encrypted_path = Path(process_chart_image(chart_info, force=False))
        record["processed"] = True
        record["encrypted_image_path"] = str(encrypted_path)
        record["encrypted_exists"] = encrypted_path.exists()
        colored_path = chart_info.get("colored_image_path")
        record["colored_image_path"] = colored_path
        record["colored_exists"] = bool(colored_path and Path(colored_path).exists())

        results_path = Path(await evaluate_processed_chart(chart_info))
        results = _json_load(results_path)
        predictions = results.get("predictions") if isinstance(results.get("predictions"), list) else []
        summary = results.get("summary") if isinstance(results.get("summary"), dict) else {}
        processed_json = results.get("processed_json") if isinstance(results.get("processed_json"), dict) else {}
        forbidden_keys = sorted(FORBIDDEN_GENERATION_KEYS.intersection(processed_json.keys()))

        record.update(
            {
                "evaluated": True,
                "results_path": str(results_path),
                "results_exists": results_path.exists(),
                "success_flag": bool(results.get("success")),
                "prediction_count": len(predictions),
                "chart_runs": int(summary.get("chart_runs") or 0),
                "system_cv_fallback": bool(summary.get("system_cv_fallback")),
                "prediction_runner_error": results.get("prediction_runner_error"),
                "note": results.get("note"),
                "processed_json_forbidden_keys": forbidden_keys,
                "processed_json_keys": sorted(processed_json.keys()),
            }
        )
        passed = (
            record["registered"]
            and record["processed"]
            and record["encrypted_exists"]
            and record["evaluated"]
            and record["results_exists"]
            and record["success_flag"]
            and record["prediction_count"] > 0
            and record["chart_runs"] > 0
            and not record["system_cv_fallback"]
            and not record["prediction_runner_error"]
            and not forbidden_keys
        )
        record["status"] = "passed" if passed else "failed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
    record["elapsed_sec"] = round(time.perf_counter() - started, 3)
    return record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="realworld", choices=["realworld", "synthetic"])
    parser.add_argument("--fresh-evaluation", action="store_true")
    args = parser.parse_args()

    samples = _sample_by_type(args.source)
    records = []
    for chart_type in CHART_TYPES:
        sample = samples.get(chart_type)
        if sample is None:
            records.append({"chart_type": chart_type, "status": "failed", "error": "No sample found"})
            continue
        print(f"[smoke] checking {chart_type}: {sample.get('name')}")
        records.append(await _check_one(chart_type, sample, args.fresh_evaluation))

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": f"{args.source} dataset preview chain",
        "fresh_evaluation": bool(args.fresh_evaluation),
        "total": len(records),
        "passed": sum(1 for item in records if item.get("status") == "passed"),
        "failed": sum(1 for item in records if item.get("status") != "passed"),
        "records": records,
    }
    _json_dump(OUT_DIR / "summary.json", payload)
    _write_report(OUT_DIR / "report.md", payload)
    print(f"[smoke] wrote {OUT_DIR / 'summary.json'}")
    print(f"[smoke] wrote {OUT_DIR / 'report.md'}")
    if payload["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
