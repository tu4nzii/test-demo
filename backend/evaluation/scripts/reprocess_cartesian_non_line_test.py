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


CARTESIAN_TYPES = ["scatter", "bubble", "v_bar", "h_bar"]
IGNORED_NAME_PARTS = ("grid", "with_grid", "temp", "encode", "marked")
TYPE_SWITCH_MIN_ATTEMPTED = 50
TYPE_SWITCH_FAILED_LIMIT = 5


def is_source_image(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() == ".png" and not any(part in name for part in IGNORED_NAME_PARTS)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_images() -> list[dict]:
    items = []
    for chart_type in CARTESIAN_TYPES:
        input_dir = BACKEND_DIR / "charts" / chart_type
        images = sorted(path for path in input_dir.glob("*.png") if is_source_image(path))
        for image_path in images:
            items.append(
                {
                    "chart_type": chart_type,
                    "chart_id": image_path.stem,
                    "image_path": image_path,
                }
            )
    return items


def main() -> int:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_root = BACKEND_DIR / "evaluation" / "recheck_outputs" / "cartesian_non_line_full_test_latest"
    result_path = BACKEND_DIR / "evaluation" / "results" / "cartesian_non_line_full_test_latest.json"
    issue_path = BACKEND_DIR / "evaluation" / "results" / "cartesian_non_line_full_test_latest_issues.json"
    progress_path = BACKEND_DIR / "evaluation" / "results" / "cartesian_non_line_full_test_latest_progress.json"

    images = collect_images()
    by_type = {
        chart_type: {
            "total": 0,
            "completed": 0,
            "attempted": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "switched_early": False,
        }
        for chart_type in CARTESIAN_TYPES
    }
    for item in images:
        by_type[item["chart_type"]]["total"] += 1

    summary = {
        "run_id": run_id,
        "chart_types": CARTESIAN_TYPES,
        "input_root": str(BACKEND_DIR / "charts"),
        "output_root": str(output_root),
        "total": len(images),
        "by_type": by_type,
        "completed": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "type_switch_rule": {
            "min_attempted": TYPE_SWITCH_MIN_ATTEMPTED,
            "failed_limit": TYPE_SWITCH_FAILED_LIMIT,
            "description": "When a chart type has attempted at least min_attempted images and failed more than failed_limit, skip the remaining images in that type.",
        },
        "items": [],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
    }
    write_json(progress_path, summary)

    for index, source in enumerate(images, start=1):
        chart_type = source["chart_type"]
        type_stats = summary["by_type"][chart_type]
        chart_id = source["chart_id"]
        image_path = source["image_path"]
        output_dir = output_root / chart_type
        item = {
            "index": index,
            "chart_type": chart_type,
            "chart_id": chart_id,
            "image_path": str(image_path),
            "success": False,
            "error": "",
            "ticks_json_path": str(output_dir / f"{chart_id}_ticks.json"),
            "grid_path": str(output_dir / f"{chart_id}_grid.png"),
            "with_grid_path": str(output_dir / f"{chart_id}_with_grid.png"),
        }
        if (
            type_stats["attempted"] >= TYPE_SWITCH_MIN_ATTEMPTED
            and type_stats["failed"] > TYPE_SWITCH_FAILED_LIMIT
        ):
            item.update(
                {
                    "skipped": True,
                    "error": (
                        f"skipped because {chart_type} reached "
                        f"{type_stats['attempted']} attempted with {type_stats['failed']} failures"
                    ),
                    "elapsed_seconds": 0,
                }
            )
            type_stats["completed"] += 1
            type_stats["skipped"] += 1
            type_stats["switched_early"] = True
            summary["completed"] = index
            summary["skipped"] += 1
            summary["items"].append(item)
            write_json(progress_path, summary)
            continue

        started = time.time()
        type_stats["attempted"] += 1
        try:
            result = process_chart(
                str(image_path),
                str(output_dir),
                chart_type_override=chart_type,
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
                type_stats["success"] += 1
            else:
                item["error"] = "process_chart returned None"
                summary["failed"] += 1
                type_stats["failed"] += 1
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
            item["traceback"] = traceback.format_exc()
            summary["failed"] += 1
            type_stats["failed"] += 1

        item["elapsed_seconds"] = round(time.time() - started, 2)
        type_stats["completed"] += 1
        summary["completed"] = index
        summary["items"].append(item)
        write_json(progress_path, summary)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    issues = [item for item in summary["items"] if not item.get("success") and not item.get("skipped")]
    skipped = [item for item in summary["items"] if item.get("skipped")]
    issue_summary = {
        "run_id": run_id,
        "total": summary["total"],
        "attempted": summary["success"] + summary["failed"],
        "success": summary["success"],
        "failed": summary["failed"],
        "skipped": summary["skipped"],
        "issue_count": len(issues),
        "skipped_count": len(skipped),
        "by_type": summary["by_type"],
        "output_root": str(output_root),
        "issues": issues,
        "skipped_items": skipped,
    }
    write_json(progress_path, summary)
    write_json(result_path, summary)
    write_json(issue_path, issue_summary)
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "issue_path": str(issue_path),
                "output_root": str(output_root),
                "total": summary["total"],
                "success": summary["success"],
                "failed": summary["failed"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
