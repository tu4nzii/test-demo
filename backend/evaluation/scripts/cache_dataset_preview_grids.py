from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Generate dataset-preview grid caches.")
    parser.add_argument("--source", choices=["realworld", "synthetic"], default="synthetic")
    parser.add_argument("--category", action="append", default=[], help="Category to cache; can be repeated.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--isolate", action="store_true", help="Run each sample in a child process.")
    parser.add_argument("--sample-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=backend_main.DATASET_PREVIEW_CACHE_DIR / "grid_cache_manifest.json",
    )
    return parser.parse_args()


def iter_samples(source: str, categories: list[str]) -> list[dict[str, Any]]:
    if not categories:
        categories = ["all"]
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for category in categories:
        category_filter = None if category in {"", "all"} else category
        for sample in backend_main.iter_dataset_samples(source, category_filter):
            sample_id = str(sample["sample_id"])
            if sample_id in seen:
                continue
            seen.add(sample_id)
            samples.append(sample)
    return samples


def cache_is_ready(sample: dict[str, Any]) -> bool:
    chart_info = backend_main.register_dataset_sample(str(sample["sample_id"]))
    encrypted = chart_info.get("encrypted_image_path")
    return bool(chart_info.get("processed") and encrypted and Path(str(encrypted)).exists())


def cache_one_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    chart_info = backend_main.register_dataset_sample(sample_id)
    encrypted_path = backend_main.process_chart_image(chart_info, force=False)
    return {
        "status": "success",
        "encrypted_image_path": str(encrypted_path),
    }


def run_isolated_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        args.source,
        "--sample-id",
        str(sample["sample_id"]),
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


def main() -> int:
    args = parse_args()
    if args.sample_id:
        sample = backend_main.dataset_sample_by_id(args.sample_id)
        result = cache_one_sample(sample)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0 if result.get("status") == "success" else 1

    samples = iter_samples(args.source, args.category)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    manifest: dict[str, Any] = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "mode": "dataset_preview_grid_cache",
        "source": args.source,
        "categories": args.category or ["all"],
        "total": len(samples),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "records": [],
    }

    print(f"Dataset preview grid cache: total={len(samples)} source={args.source} categories={args.category or ['all']}", flush=True)
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])
        record: dict[str, Any] = {
            "index": index,
            "sample_id": sample_id,
            "source": sample.get("source"),
            "relative_path": sample.get("relative_path"),
            "chart_type": sample.get("chart_type"),
            "status": "pending",
        }
        try:
            if args.resume and cache_is_ready(sample):
                chart_info = backend_main.register_dataset_sample(sample_id)
                record.update(
                    {
                        "status": "skipped_existing",
                        "encrypted_image_path": chart_info.get("encrypted_image_path"),
                    }
                )
                manifest["skipped"] += 1
                print(f"[{index}/{len(samples)}] SKIP {sample.get('relative_path')}", flush=True)
            else:
                result = run_isolated_sample(sample, args) if args.isolate else cache_one_sample(sample)
                record.update(result)
                if result.get("status") == "success":
                    manifest["success"] += 1
                    print(f"[{index}/{len(samples)}] OK {sample.get('relative_path')}", flush=True)
                else:
                    manifest["failed"] += 1
                    print(f"[{index}/{len(samples)}] FAIL {sample.get('relative_path')}: {result.get('error')}", flush=True)
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            manifest["failed"] += 1
            print(f"[{index}/{len(samples)}] FAIL {sample.get('relative_path')}: {exc}", flush=True)
        finally:
            manifest["records"].append(record)
            write_json(args.manifest, manifest)

    summary = {key: manifest[key] for key in ("run_id", "total", "success", "failed", "skipped")}
    write_json(args.manifest.with_name("grid_cache_summary.json"), summary)
    print(f"Done success={manifest['success']} failed={manifest['failed']} skipped={manifest['skipped']}", flush=True)
    print(f"Manifest: {args.manifest}", flush=True)
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
