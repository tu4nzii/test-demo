import json
import os
import sys
import time
import traceback
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
GRID_DIR = BACKEND_DIR / "Grid_generation"
PROJECT_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(GRID_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from grid_generation import process_chart  # noqa: E402


def is_source_line_image(path: Path) -> bool:
    name = path.name.lower()
    ignored = ("grid", "with_grid", "temp", "encode", "marked")
    return path.suffix.lower() == ".png" and not any(part in name for part in ignored)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    input_dir = BACKEND_DIR / "charts" / "line"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = BACKEND_DIR / "evaluation" / "recheck_outputs" / f"line_full_test_{run_id}" / "line"
    result_path = BACKEND_DIR / "evaluation" / "results" / f"line_full_test_{run_id}.json"
    progress_path = BACKEND_DIR / "evaluation" / "results" / "line_full_test_latest_progress.json"

    images = sorted(path for path in input_dir.glob("*.png") if is_source_line_image(path))
    summary = {
        "run_id": run_id,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total": len(images),
        "completed": 0,
        "success": 0,
        "failed": 0,
        "items": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
    }
    write_json(progress_path, summary)

    for index, image_path in enumerate(images, start=1):
        chart_id = image_path.stem
        item = {
            "index": index,
            "chart_id": chart_id,
            "image_path": str(image_path),
            "success": False,
            "error": "",
            "ticks_json_path": str(output_dir / f"{chart_id}_ticks.json"),
            "grid_path": str(output_dir / f"{chart_id}_grid.png"),
            "with_grid_path": str(output_dir / f"{chart_id}_with_grid.png"),
        }
        started = time.time()
        try:
            result = process_chart(
                str(image_path),
                str(output_dir),
                chart_type_override="line",
                chart_id_override=chart_id,
            )
            if result:
                item.update(
                    {
                        "success": True,
                        "x_axis_type": result.get("x_axis_type"),
                        "y_axis_type": result.get("y_axis_type"),
                        "x_ticks": len(result.get("x_ticks", [])),
                        "y_ticks": len(result.get("y_ticks", [])),
                        "x_ticks_encrypted": len(result.get("x_ticks_encrypted", [])),
                        "y_ticks_encrypted": len(result.get("y_ticks_encrypted", [])),
                    }
                )
                summary["success"] += 1
            else:
                item["error"] = "process_chart returned None"
                summary["failed"] += 1
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["traceback"] = traceback.format_exc()
            summary["failed"] += 1

        item["elapsed_seconds"] = round(time.time() - started, 2)
        summary["completed"] = index
        summary["items"].append(item)
        write_json(progress_path, summary)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(progress_path, summary)
    write_json(result_path, summary)
    print(json.dumps({"result_path": str(result_path), **{k: summary[k] for k in ("total", "success", "failed")}}, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
