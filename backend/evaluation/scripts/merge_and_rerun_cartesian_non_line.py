import contextlib
import json
import logging
import os
import shutil
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


CHART_TYPES = ["scatter", "bubble", "v_bar", "h_bar"]
ORIGINAL_RESULT = BACKEND_DIR / "evaluation" / "results" / "cartesian_non_line_full_test_20260528_004823.json"
TODAY_RESULT_NAMES = [
    "scatter_issue_rerun_20260528_075830.json",
    "all_issue_rerun_20260528_080500.json",
    "bubble_issue_rerun_20260528_083412.json",
    "bar_issue_rerun_20260528_084437.json",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def item_key(item: dict) -> tuple[str, str]:
    return item.get("chart_type", ""), item.get("chart_id", "")


def existing_success_files(item: dict) -> bool:
    return all(
        Path(item.get(field, "")).exists()
        for field in ["ticks_json_path", "grid_path", "with_grid_path"]
    )


def copy_success_files(source_item: dict, output_root: Path) -> dict:
    chart_type = source_item["chart_type"]
    chart_id = source_item["chart_id"]
    output_dir = output_root / chart_type
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = dict(source_item)
    targets = {
        "ticks_json_path": output_dir / f"{chart_id}_ticks.json",
        "grid_path": output_dir / f"{chart_id}_grid.png",
        "with_grid_path": output_dir / f"{chart_id}_with_grid.png",
    }
    for field, target in targets.items():
        source = Path(source_item.get(field, ""))
        if source.exists() and source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied[field] = str(target)
    return copied


def load_success_overrides() -> dict[tuple[str, str], dict]:
    result_dir = BACKEND_DIR / "evaluation" / "results"
    best = {}

    original = read_json(ORIGINAL_RESULT)
    for item in original.get("items", []):
        if item.get("success") and existing_success_files(item):
            best[item_key(item)] = dict(item, source_result=str(ORIGINAL_RESULT))

    for name in TODAY_RESULT_NAMES:
        path = result_dir / name
        if not path.exists():
            continue
        data = read_json(path)
        for item in data.get("items", []):
            if item.get("success") and existing_success_files(item):
                best[item_key(item)] = dict(item, source_result=str(path))

    return best


def summarize_by_type(items: list[dict]) -> dict:
    by_type = {
        chart_type: {"total": 0, "success": 0, "failed": 0}
        for chart_type in CHART_TYPES
    }
    for item in items:
        stats = by_type.setdefault(item["chart_type"], {"total": 0, "success": 0, "failed": 0})
        stats["total"] += 1
        if item.get("success"):
            stats["success"] += 1
        else:
            stats["failed"] += 1
    return by_type


def main() -> int:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("grid_generation").setLevel(logging.WARNING)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_root = BACKEND_DIR / "evaluation" / "recheck_outputs" / f"cartesian_non_line_merged_rerun_{run_id}"
    result_path = BACKEND_DIR / "evaluation" / "results" / f"cartesian_non_line_merged_rerun_{run_id}.json"
    latest_path = BACKEND_DIR / "evaluation" / "results" / "cartesian_non_line_merged_rerun_latest.json"
    issue_path = BACKEND_DIR / "evaluation" / "results" / f"cartesian_non_line_merged_rerun_{run_id}_issues.json"
    latest_issue_path = BACKEND_DIR / "evaluation" / "results" / "cartesian_non_line_merged_rerun_latest_issues.json"
    log_path = BACKEND_DIR / "evaluation" / "results" / f"cartesian_non_line_merged_rerun_{run_id}.log"

    original = read_json(ORIGINAL_RESULT)
    source_items = original.get("items", [])
    best_success = load_success_overrides()

    merged_items = []
    pending = []
    copied_success = 0
    for source in source_items:
        key = item_key(source)
        if key in best_success:
            merged = copy_success_files(best_success[key], output_root)
            merged["merged_from"] = best_success[key].get("source_result")
            merged_items.append(merged)
            copied_success += 1
        else:
            pending_item = dict(source)
            pending_item["success"] = False
            pending_item["error"] = source.get("error") or source.get("previous_error") or "pending rerun"
            pending_item["merged_from"] = None
            pending.append(pending_item)
            merged_items.append(pending_item)

    summary = {
        "run_id": run_id,
        "source_original_result": str(ORIGINAL_RESULT),
        "today_result_names": TODAY_RESULT_NAMES,
        "output_root": str(output_root),
        "log_path": str(log_path),
        "total": len(source_items),
        "copied_success_before_rerun": copied_success,
        "pending_rerun_total": len(pending),
        "rerun_completed": 0,
        "rerun_success": 0,
        "rerun_failed": 0,
        "success": sum(1 for item in merged_items if item.get("success")),
        "failed": sum(1 for item in merged_items if not item.get("success")),
        "by_type": summarize_by_type(merged_items),
        "items": merged_items,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
    }
    write_json(result_path, summary)
    write_json(latest_path, summary)

    print(f"merged success copied: {copied_success}")
    print(f"pending rerun: {len(pending)}")
    print(f"output_root: {output_root}")

    index_by_key = {item_key(item): idx for idx, item in enumerate(merged_items)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        for rerun_index, item in enumerate(pending, start=1):
            chart_type = item["chart_type"]
            chart_id = item["chart_id"]
            image_path = item["image_path"]
            output_dir = output_root / chart_type
            print(f"[{rerun_index}/{len(pending)}] {chart_type}/{chart_id}", flush=True)
            log_file.write(f"\n===== [{rerun_index}/{len(pending)}] {chart_type}/{chart_id} =====\n")
            log_file.flush()

            started = time.time()
            rec = dict(item)
            rec.update(
                {
                    "success": False,
                    "error": "",
                    "ticks_json_path": str(output_dir / f"{chart_id}_ticks.json"),
                    "grid_path": str(output_dir / f"{chart_id}_grid.png"),
                    "with_grid_path": str(output_dir / f"{chart_id}_with_grid.png"),
                    "merged_from": "rerun",
                }
            )

            try:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    result = process_chart(
                        image_path,
                        str(output_dir),
                        chart_type_override=chart_type,
                        chart_id_override=chart_id,
                    )
                rec["elapsed_seconds"] = round(time.time() - started, 2)
                if result:
                    rec.update(
                        {
                            "success": True,
                            "x_axis_type": result.get("x_axis_type"),
                            "y_axis_type": result.get("y_axis_type"),
                            "x_ticks": len(result.get("x_ticks", [])),
                            "y_ticks": len(result.get("y_ticks", [])),
                            "x_pixels": result.get("x_pixels", []),
                            "y_pixels": result.get("y_pixels", []),
                            "x_ticks_encrypted": len(result.get("x_ticks_encrypted", [])),
                            "y_ticks_encrypted": len(result.get("y_ticks_encrypted", [])),
                        }
                    )
                    summary["rerun_success"] += 1
                    print(f"  ok ({rec['elapsed_seconds']}s)", flush=True)
                else:
                    rec["error"] = "process_chart returned None"
                    summary["rerun_failed"] += 1
                    print(f"  failed ({rec['elapsed_seconds']}s): {rec['error']}", flush=True)
            except Exception as exc:
                rec["elapsed_seconds"] = round(time.time() - started, 2)
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()
                summary["rerun_failed"] += 1
                log_file.write(rec["traceback"])
                print(f"  failed ({rec['elapsed_seconds']}s): {rec['error']}", flush=True)

            merged_items[index_by_key[item_key(item)]] = rec
            summary["rerun_completed"] = rerun_index
            summary["success"] = sum(1 for candidate in merged_items if candidate.get("success"))
            summary["failed"] = sum(1 for candidate in merged_items if not candidate.get("success"))
            summary["by_type"] = summarize_by_type(merged_items)
            summary["items"] = merged_items
            write_json(result_path, summary)
            write_json(latest_path, summary)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["success"] = sum(1 for item in merged_items if item.get("success"))
    summary["failed"] = sum(1 for item in merged_items if not item.get("success"))
    summary["by_type"] = summarize_by_type(merged_items)
    summary["items"] = merged_items
    write_json(result_path, summary)
    write_json(latest_path, summary)

    remaining = [item for item in merged_items if not item.get("success")]
    issue_doc = {
        "run_id": run_id,
        "source_result_path": str(result_path),
        "output_root": str(output_root),
        "remaining_issue_count": len(remaining),
        "remaining_issues": remaining,
    }
    write_json(issue_path, issue_doc)
    write_json(latest_issue_path, issue_doc)

    print(f"RESULT_PATH {result_path}")
    print(f"LATEST_PATH {latest_path}")
    print(f"ISSUES_PATH {issue_path}")
    print(f"OUTPUT_ROOT {output_root}")
    print(f"success {summary['success']} failed {summary['failed']}")
    print("by_type", json.dumps(summary["by_type"], ensure_ascii=False))
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
