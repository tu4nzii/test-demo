import argparse
import csv
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent
GRID_DIR = BACKEND_DIR / "Grid_generation"
RESULTS_DIR = BACKEND_DIR / "evaluation" / "results"
OUTPUT_ROOT = BACKEND_DIR / "evaluation" / "recheck_outputs" / "excelchart400k_grid"

if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from grid_generation import process_chart  # noqa: E402


SPLITS = ["train2019", "val2019", "test2019"]
SUPPORTED_DATASETS = ["bar", "line"]


def normalize_axis_type(axis_type):
    text = str(axis_type or "").lower()
    if "数值" in text or "numeric" in text:
        return "numeric"
    if "文字" in text or "文本" in text or "category" in text or "text" in text:
        return "text"
    return "unknown"


def expected_encrypted_count(original_count, axis_type):
    if original_count <= 0:
        return 0
    return original_count * 2 - 1 if normalize_axis_type(axis_type) == "numeric" else original_count


def safe_chart_id(dataset_name, split, index):
    return f"excel400k_{dataset_name}_{split}_{index:06d}"


def dataset_paths(dataset_root):
    root = Path(dataset_root)
    return {
        "bar": {
            "image_root": root / "bardata(1031)" / "bardata(1031)" / "bar" / "images",
            "annotation_root": root / "bardata(1031)" / "bardata(1031)" / "bar" / "annotations",
            "annotation_prefix": "instancesBar(1031)",
        },
        "line": {
            "image_root": root / "linedata(1028)" / "linedata(1028)" / "line" / "images",
            "annotation_root": root / "linedata(1028)" / "linedata(1028)" / "line" / "annotations",
            "annotation_prefix": "instancesLine(1023)",
        },
    }


def load_bar_orientation(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes_by_image = defaultdict(list)
    for annotation in data.get("annotations", []):
        bbox = annotation.get("bbox") or []
        if len(bbox) >= 4:
            boxes_by_image[annotation.get("image_id")].append(bbox)

    orientation_by_file = {}
    for image in data.get("images", []):
        boxes = boxes_by_image.get(image.get("id"), [])
        horizontal_votes = sum(1 for box in boxes if float(box[2]) > float(box[3]))
        chart_type = "h_bar" if boxes and horizontal_votes > len(boxes) / 2 else "v_bar"
        orientation_by_file[image.get("file_name")] = {
            "chart_type": chart_type,
            "image_id": image.get("id"),
        }
    return orientation_by_file


def collect_items(dataset_root, datasets, splits, limit_per_type_split=0):
    paths = dataset_paths(dataset_root)
    items = []
    inventory = {}

    for dataset_name in datasets:
        spec = paths[dataset_name]
        for split in splits:
            image_dir = spec["image_root"] / split
            image_paths = sorted(image_dir.glob("*.png"))
            inventory[f"{dataset_name}/{split}"] = len(image_paths)
            if limit_per_type_split > 0:
                image_paths = image_paths[:limit_per_type_split]

            orientation_by_file = {}
            if dataset_name == "bar":
                annotation_path = spec["annotation_root"] / f"{spec['annotation_prefix']}_{split}.json"
                orientation_by_file = load_bar_orientation(annotation_path)

            for index, image_path in enumerate(image_paths):
                if dataset_name == "line":
                    chart_type = "line"
                    image_id = ""
                else:
                    meta = orientation_by_file.get(image_path.name, {})
                    chart_type = meta.get("chart_type", "v_bar")
                    image_id = meta.get("image_id", "")
                items.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "index": index,
                        "image_id": image_id,
                        "chart_type": chart_type,
                        "image_path": str(image_path),
                        "chart_id": safe_chart_id(dataset_name, split, index),
                    }
                )
    return items, inventory


def read_done_rows(csv_path):
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            done.add(row.get("chart_id"))
    return done


def check_effect(result):
    if not result:
        return False, "process_failed"

    if not os.path.exists(result.get("encrypted_grid_path", "")):
        return False, "missing_encrypted_grid"
    if not os.path.exists(result.get("basic_grid_path", "")):
        return False, "missing_basic_grid"

    x_expected = expected_encrypted_count(len(result.get("x_ticks", [])), result.get("x_axis_type"))
    y_expected = expected_encrypted_count(len(result.get("y_ticks", [])), result.get("y_axis_type"))
    if len(result.get("x_ticks_encrypted", [])) != x_expected:
        return False, "x_encrypted_tick_count_mismatch"
    if len(result.get("y_ticks_encrypted", [])) != y_expected:
        return False, "y_encrypted_tick_count_mismatch"
    if len(result.get("x_pixels_encrypted", [])) != len(result.get("x_ticks_encrypted", [])):
        return False, "x_pixel_tick_count_mismatch"
    if len(result.get("y_pixels_encrypted", [])) != len(result.get("y_ticks_encrypted", [])):
        return False, "y_pixel_tick_count_mismatch"
    return True, ""


def cleanup_success_outputs(result):
    if not result:
        return
    for key in ("basic_grid_path", "encrypted_grid_path"):
        path = result.get(key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def row_from_result(item, result, ok, problem_type, error=""):
    return {
        "chart_id": item["chart_id"],
        "dataset": item["dataset"],
        "split": item["split"],
        "chart_type": item["chart_type"],
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "success": bool(result),
        "effect_ok": ok,
        "problem_type": problem_type,
        "x_axis_type": result.get("x_axis_type", "") if result else "",
        "y_axis_type": result.get("y_axis_type", "") if result else "",
        "x_ticks": len(result.get("x_ticks", [])) if result else 0,
        "y_ticks": len(result.get("y_ticks", [])) if result else 0,
        "x_ticks_encrypted": len(result.get("x_ticks_encrypted", [])) if result else 0,
        "y_ticks_encrypted": len(result.get("y_ticks_encrypted", [])) if result else 0,
        "basic_grid_path": result.get("basic_grid_path", "") if result else "",
        "encrypted_grid_path": result.get("encrypted_grid_path", "") if result else "",
        "ticks_json_path": str(Path(result.get("encrypted_grid_path", "")).with_name(f"{item['chart_id']}_ticks.json")) if result else "",
        "error": error,
    }


def write_markdown(summary, problems, md_path, csv_path, output_root):
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ExcelChart400k Grid Encryption Evaluation\n\n")
        f.write(f"- CSV: `{csv_path}`\n")
        f.write(f"- Output root: `{output_root}`\n")
        f.write("- Included datasets: `bardata` and `linedata` only; `pie` has no Cartesian grid, `cls` is classification/duplicate-style metadata.\n\n")
        f.write("## Summary\n\n")
        f.write("| chart_type | total | success | effect_ok | problems |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for chart_type, data in sorted(summary.items()):
            f.write(
                f"| {chart_type} | {data['total']} | {data['success']} | {data['effect_ok']} | "
                f"{data['problems']} |\n"
            )

        f.write("\n## Problems\n\n")
        f.write("| chart_id | chart_type | split | problem_type | output |\n")
        f.write("|---|---|---|---|---|\n")
        for row in problems[:500]:
            output = row.get("encrypted_grid_path") or ""
            f.write(f"| {row['chart_id']} | {row['chart_type']} | {row['split']} | {row['problem_type']} | `{output}` |\n")
        if len(problems) > 500:
            f.write(f"\nOnly the first 500 problems are listed here; see CSV for all {len(problems)} rows.\n")


def summarize_csv(csv_path):
    summary = defaultdict(lambda: {"total": 0, "success": 0, "effect_ok": 0, "problems": 0})
    problems = []
    if not csv_path.exists():
        return summary, problems
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            chart_type = row.get("chart_type", "unknown")
            summary[chart_type]["total"] += 1
            if row.get("success") == "True":
                summary[chart_type]["success"] += 1
            if row.get("effect_ok") == "True":
                summary[chart_type]["effect_ok"] += 1
            else:
                summary[chart_type]["problems"] += 1
                problems.append(row)
    return summary, problems


def main():
    parser = argparse.ArgumentParser(description="Run grid encryption effect evaluation on ExcelChart400k Cartesian subsets.")
    parser.add_argument("--dataset-root", default=r"F:\Dataset\ExcelChart400k")
    parser.add_argument("--datasets", nargs="+", default=SUPPORTED_DATASETS, choices=SUPPORTED_DATASETS)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--limit-per-type-split", type=int, default=0, help="0 means full selected splits.")
    parser.add_argument("--output-prefix", default="excelchart400k_grid_full")
    parser.add_argument("--keep-success-outputs", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / f"{args.output_prefix}.csv"
    md_path = RESULTS_DIR / f"{args.output_prefix}.md"
    log_path = RESULTS_DIR / f"{args.output_prefix}.log"

    items, inventory = collect_items(
        args.dataset_root,
        args.datasets,
        args.splits,
        limit_per_type_split=args.limit_per_type_split,
    )
    done = set() if args.no_resume else read_done_rows(csv_path)

    fieldnames = [
        "chart_id",
        "dataset",
        "split",
        "chart_type",
        "image_id",
        "image_path",
        "success",
        "effect_ok",
        "problem_type",
        "x_axis_type",
        "y_axis_type",
        "x_ticks",
        "y_ticks",
        "x_ticks_encrypted",
        "y_ticks_encrypted",
        "basic_grid_path",
        "encrypted_grid_path",
        "ticks_json_path",
        "error",
    ]
    csv_exists = csv_path.exists() and not args.no_resume
    mode = "a" if csv_exists else "w"

    with open(csv_path, mode, encoding="utf-8", newline="") as csv_file, open(log_path, "a", encoding="utf-8") as log_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not csv_exists:
            writer.writeheader()
        print(f"inventory={inventory}", file=log_file, flush=True)
        print(f"selected={len(items)} done={len(done)}", file=log_file, flush=True)

        for ordinal, item in enumerate(items, start=1):
            if item["chart_id"] in done:
                continue

            output_dir = OUTPUT_ROOT / item["chart_type"] / item["split"]
            output_dir.mkdir(parents=True, exist_ok=True)
            item["output_dir"] = str(output_dir)
            try:
                result = process_chart(
                    item["image_path"],
                    str(output_dir),
                    chart_type_override=item["chart_type"],
                    chart_id_override=item["chart_id"],
                )
                ok, problem_type = check_effect(result)
                if ok and not args.keep_success_outputs:
                    cleanup_success_outputs(result)
                row = row_from_result(item, result, ok, problem_type)
            except Exception as exc:
                problem_type = "exception"
                row = row_from_result(item, None, False, problem_type, error=repr(exc))
                print(f"[{ordinal}/{len(items)}] {item['chart_id']} exception: {exc}", file=log_file, flush=True)
                print(traceback.format_exc(), file=log_file, flush=True)

            writer.writerow(row)
            csv_file.flush()
            if ordinal % 10 == 0:
                print(f"[{ordinal}/{len(items)}] wrote {item['chart_id']} effect_ok={row['effect_ok']} problem={row['problem_type']}", flush=True)

    summary, problems = summarize_csv(csv_path)
    write_markdown(summary, problems, md_path, csv_path, OUTPUT_ROOT)
    print(json.dumps({"summary": summary, "problems": len(problems), "csv": str(csv_path), "md": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
