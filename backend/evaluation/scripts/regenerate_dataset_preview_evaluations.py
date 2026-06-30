from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

import main as backend_main  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate dataset-preview step-3 evaluation caches without touching grid caches."
    )
    parser.add_argument(
        "--source",
        choices=["all", "realworld", "synthetic"],
        default="all",
        help="Dataset source to evaluate.",
    )
    parser.add_argument(
        "--category",
        default="all",
        help="Optional category filter, e.g. v_bar, h_bar, line, scatter, bubble, pie, donut, radar, rose.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--resume", action="store_true", help="Skip samples with an existing evaluation cache.")
    parser.add_argument("--isolate", action="store_true", help="Run each sample in a child process.")
    parser.add_argument("--sample-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=backend_main.DATASET_PREVIEW_CACHE_DIR / "evaluation_cache_manifest.json",
    )
    parser.add_argument(
        "--mode",
        choices=["system", "model"],
        default="system",
        help=(
            "system writes a fast cache from current generated ticks/images without external model calls; "
            "model uses the normal step-3 prediction runner."
        ),
    )
    return parser.parse_args()


def iter_samples(source: str, category: str) -> list[dict[str, Any]]:
    sources = ["realworld", "synthetic"] if source == "all" else [source]
    samples: list[dict[str, Any]] = []
    category_filter = None if category in {"", "all"} else category
    for item_source in sources:
        samples.extend(list(backend_main.iter_dataset_samples(item_source, category_filter)))
    return samples


def build_system_evaluation(chart_info: dict[str, Any]) -> Path:
    output_dir = Path(chart_info.get("output_dir", backend_main.OUTPUT_DIR / chart_info["chart_type"]))
    image_stem = Path(chart_info["image_path"]).stem
    eval_json_path = next((path for path in backend_main.candidate_eval_json_paths(chart_info) if path.exists()), None)
    if eval_json_path is None:
        raise FileNotFoundError(f"No cached evaluation JSON found in {output_dir}")

    processed_payload = backend_main.load_json(eval_json_path)
    if not isinstance(processed_payload, dict):
        processed_payload = {}
    backend_main.merge_tick_sidecar(processed_payload, output_dir, image_stem)
    backend_main.strip_external_reference_data(
        processed_payload,
        preserve_data=chart_info["chart_type"] in {"pie", "donut"},
        preserve_series_color=True,
    )
    backend_main.ensure_series_color_from_colors(processed_payload)
    backend_main.write_json(eval_json_path, processed_payload)
    predictions = backend_main.system_cv_predictions(chart_info, eval_json_path)
    if not predictions:
        existing = processed_payload.get("predictions")
        predictions = existing if isinstance(existing, list) else []
    result = {
        "success": True,
        "mode": "prediction_extraction",
        "cache_mode": "system",
        "chart_id": chart_info["chart_id"],
        "chart_type": chart_info["chart_type"],
        "source_json": str(eval_json_path),
        "system_json": str(eval_json_path),
        "summary": {
            "object_count": len(predictions),
            "chart_runs": 0,
            "system_cv_fallback": True,
        },
        "predictions": predictions,
        "artifacts": [],
        "processed_json": processed_payload,
        "note": (
            "Dataset preview cache generated from current system ticks/images only. "
            "Grid cache was reused and not regenerated; dataset GT was not used."
        ),
    }
    dataset_sample = chart_info.get("dataset_sample") if isinstance(chart_info.get("dataset_sample"), dict) else {}
    sample_id = str(dataset_sample.get("sample_id") or "").strip()
    if not sample_id:
        raise ValueError("Missing dataset preview sample_id")
    result_path = backend_main.dataset_evaluation_cache_path(sample_id)
    backend_main.write_json(result_path, result)
    chart_info.update({"evaluated": True, "evaluation_results_path": str(result_path)})
    return result_path


async def evaluate_one_sample(sample: dict[str, Any], mode: str) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    chart_info = backend_main.register_dataset_sample(sample_id)
    if not chart_info.get("processed"):
        return {
            "status": "skipped_no_grid_cache",
            "reason": "dataset preview sample has no cached encrypted grid",
        }
    if mode == "model":
        result_path = await backend_main.evaluate_processed_chart(chart_info)
    else:
        result_path = build_system_evaluation(chart_info)
    payload = backend_main.normalize_result_payload(backend_main.load_json(result_path), result_path)
    return {
        "status": "success",
        "result_path": str(result_path),
        "object_count": payload.get("summary", {}).get("object_count"),
        "chart_runs": payload.get("summary", {}).get("chart_runs"),
        "system_cv_fallback": payload.get("summary", {}).get("system_cv_fallback"),
    }


def run_isolated_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        args.source,
        "--sample-id",
        str(sample["sample_id"]),
        "--mode",
        args.mode,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "failed",
            "error": f"timeout after {args.timeout_sec}s",
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "elapsed_sec": round(time.perf_counter() - started, 3),
        }
    result: dict[str, Any] = {
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        result.update({"status": "failed", "error": f"child exited with code {completed.returncode}"})
        return result
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        if isinstance(payload, dict):
            result.update(payload)
    except Exception as exc:
        result.update({"status": "failed", "error": f"could not parse child output: {exc}"})
    return result


async def run() -> int:
    args = parse_args()
    if args.sample_id:
        sample = backend_main.dataset_sample_by_id(args.sample_id)
        result = await evaluate_one_sample(sample, args.mode)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result.get("status") in {"success", "skipped_no_grid_cache"} else 1

    samples = iter_samples(args.source, args.category)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    manifest: dict[str, Any] = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "mode": "dataset_preview_evaluation_cache",
        "source": args.source,
        "category": args.category,
        "evaluation_mode": args.mode,
        "total": len(samples),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "records": [],
        "note": "Only step-3 evaluation prediction caches are generated. Existing grid/tick/image caches are not regenerated.",
    }

    def flush() -> None:
        write_json(args.manifest, manifest)

    print(f"Regenerate dataset preview evaluations: total={len(samples)}", flush=True)
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        eval_path = backend_main.dataset_evaluation_cache_path(sample_id)
        record: dict[str, Any] = {
            "index": index,
            "sample_id": sample_id,
            "source": sample.get("source"),
            "relative_path": sample.get("relative_path"),
            "chart_type": sample.get("chart_type"),
            "evaluation_cache": str(eval_path),
            "status": "pending",
        }
        try:
            if args.resume and eval_path.exists():
                record["status"] = "skipped_existing"
                manifest["skipped"] += 1
                print(f"[{index}/{len(samples)}] SKIP existing {sample_id} {sample.get('relative_path')}", flush=True)
            else:
                if eval_path.exists():
                    eval_path.unlink()
                if args.isolate:
                    result = run_isolated_sample(sample, args)
                else:
                    result = await evaluate_one_sample(sample, args.mode)
                if result.get("status") == "skipped_no_grid_cache":
                    record["status"] = "skipped_no_grid_cache"
                    record["reason"] = result.get("reason")
                    manifest["skipped"] += 1
                    print(f"[{index}/{len(samples)}] SKIP no grid {sample_id} {sample.get('relative_path')}", flush=True)
                elif result.get("status") == "success":
                    record.update(result)
                    manifest["success"] += 1
                    print(
                        f"[{index}/{len(samples)}] OK {sample_id} {sample.get('relative_path')} "
                        f"objects={record.get('object_count')}",
                        flush=True,
                    )
                else:
                    record.update(result)
                    manifest["failed"] += 1
                    print(f"[{index}/{len(samples)}] FAIL {sample_id} {sample.get('relative_path')}: {result.get('error')}", flush=True)
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            manifest["failed"] += 1
            print(f"[{index}/{len(samples)}] FAIL {sample_id} {sample.get('relative_path')}: {exc}", flush=True)
        finally:
            manifest["records"].append(record)
            flush()

    summary = {key: manifest[key] for key in ("run_id", "total", "success", "failed", "skipped")}
    write_json(args.manifest.with_name("evaluation_cache_summary.json"), summary)
    print(f"Done success={manifest['success']} failed={manifest['failed']} skipped={manifest['skipped']}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
