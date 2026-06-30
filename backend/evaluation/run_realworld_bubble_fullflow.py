"""Run backend full-flow processing for selected real-world bubble charts."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as backend_main  # noqa: E402
from evaluation_prediction.service import run_prediction_async  # noqa: E402


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def copy_if_exists(src: str | Path | None, dst_dir: Path) -> str | None:
    if not src:
        return None
    src_path = Path(src)
    if not src_path.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src_path.name
    shutil.copy2(src_path, dst)
    return str(dst)


async def run_one(
    *,
    number: str,
    run_id: str,
    batch_dir: Path,
    summary: dict[str, Any],
    manifest_path: Path,
) -> None:
    label = f"BubbleChart{number}"
    item_dir = batch_dir / label
    item_dir.mkdir(parents=True, exist_ok=True)

    image_src = ROOT / "backend" / "realworldcharts" / "bubble" / f"{label}.png"
    upload_image = ROOT / "backend" / "data" / "upload" / f"{run_id}_{label}_image.png"
    upload_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_src, upload_image)

    item: dict[str, Any] = {
        "label": label,
        "source_image": str(image_src),
        "uploaded_image": str(upload_image),
        "status": "started",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary["items"].append(item)
    write_json(manifest_path, summary)

    try:
        print(f"\n===== {label}: detect =====", flush=True)
        detection = backend_main.detect_chart_type(upload_image)
        chart_type = backend_main.normalize_chart_type(detection.get("type"))
        chart_id = f"realworld_bubble_{number}_{run_id}"
        chart_info = {
            "chart_id": chart_id,
            "chart_type": chart_type,
            "coordinate_system": backend_main.get_coordinate_system(chart_type).value,
            "confidence": detection.get("confidence", 0.5),
            "axis_repair": detection.get("axis_repair") or {},
            "image_path": str(upload_image),
            "json_path": None,
            "processed": False,
            "evaluated": False,
        }
        backend_main.charts_db[chart_id] = chart_info
        item.update(
            {
                "chart_id": chart_id,
                "chart_type": chart_type,
                "confidence": chart_info["confidence"],
                "axis_repair": chart_info["axis_repair"],
                "status": "detected",
            }
        )
        write_json(manifest_path, summary)

        print(f"===== {label}: process/encrypt =====", flush=True)
        encrypted_image_path = backend_main.process_chart_image(chart_info)
        eval_json_path = backend_main.resolve_eval_json(chart_info)
        output_dir = Path(chart_info["output_dir"])
        image_stem = upload_image.stem
        axes_json = output_dir / f"{chart_id}_axes.json"
        ticks_json = output_dir / f"{image_stem}_ticks.json"
        generated_json = output_dir / f"{image_stem}.json"

        copied_dir = item_dir / "processing_artifacts"
        copied = {
            "encrypted_grid": copy_if_exists(encrypted_image_path, copied_dir),
            "source_generated_json": copy_if_exists(generated_json, copied_dir),
            "ticks_json": copy_if_exists(ticks_json, copied_dir),
            "axes_json": copy_if_exists(axes_json, copied_dir),
        }
        processed_json = backend_main.processed_json_payload(eval_json_path, chart_type)
        write_json(item_dir / "processed_json_payload.json", processed_json)
        item.update(
            {
                "status": "processed",
                "encrypted_image_path": str(encrypted_image_path),
                "eval_json_path": str(eval_json_path),
                "output_dir": str(output_dir),
                "copied_processing_artifacts": copied,
                "x_ticks_count": len(processed_json.get("x_ticks", []))
                if isinstance(processed_json.get("x_ticks"), list)
                else 0,
                "y_ticks_count": len(processed_json.get("y_ticks", []))
                if isinstance(processed_json.get("y_ticks"), list)
                else 0,
                "colors_count": len(processed_json.get("colors", []))
                if isinstance(processed_json.get("colors"), list)
                else 0,
            }
        )
        write_json(manifest_path, summary)

        print(f"===== {label}: prediction =====", flush=True)
        prediction_runs = await run_prediction_async("bubble", eval_json_path, batch_size=1)
        prediction_summary_path = item_dir / "prediction_summary.json"
        write_json(prediction_summary_path, prediction_runs)
        prediction_result_dirs = [
            run.get("result_dir") for run in prediction_runs if isinstance(run, dict)
        ]
        for result_dir in prediction_result_dirs:
            if result_dir and Path(result_dir).exists():
                local_result_dir = item_dir / "prediction_results" / Path(result_dir).name
                if local_result_dir.exists():
                    shutil.rmtree(local_result_dir)
                shutil.copytree(result_dir, local_result_dir)

        item.update(
            {
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prediction_summary_path": str(prediction_summary_path),
                "prediction_result_dirs": prediction_result_dirs,
                "local_prediction_results_dir": str(item_dir / "prediction_results"),
            }
        )
        write_json(manifest_path, summary)
        print(f"===== {label}: done =====", flush=True)
    except Exception as exc:
        item.update(
            {
                "status": "failed",
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(manifest_path, summary)
        print(f"===== {label}: failed =====", flush=True)
        traceback.print_exc()


async def run_batch(args: argparse.Namespace) -> None:
    os.environ.setdefault("CHART_REPEAT_TIMES", str(args.repeat_times))
    batch_dir = Path(args.batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "manifest.json"
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chart_numbers": args.charts,
        "chart_repeat_times": os.environ.get("CHART_REPEAT_TIMES"),
        "batch_dir": str(batch_dir),
        "items": [],
    }
    write_json(manifest_path, summary)

    for number in args.charts:
        await run_one(
            number=str(number),
            run_id=args.run_id,
            batch_dir=batch_dir,
            summary=summary,
            manifest_path=manifest_path,
        )

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(manifest_path, summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts", nargs="+", default=["12", "14", "15", "16"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--repeat-times", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_batch(parse_args()))
