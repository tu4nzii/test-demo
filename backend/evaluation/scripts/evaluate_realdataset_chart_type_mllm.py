from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from type_detection.chart_type import ChartTypeDetector  # noqa: E402


DATASET_DIRS = {
    "vBar_21": "v_bar",
    "hBar_12": "h_bar",
    "Line_23": "line",
    "Scatter_9": "scatter",
    "Bubble_9": "bubble",
    "Pie_11": "pie",
    "Donut_13": "donut",
    "Radar_18": "radar",
    "Rose_6": "rose",
}


def build_ground_truth(dataset_root: Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    conflicts: dict[str, set[str]] = defaultdict(set)
    for dirname, chart_type in DATASET_DIRS.items():
        chart_root = dataset_root / dirname
        for path in chart_root.rglob("*.png"):
            if path.name in truth and truth[path.name] != chart_type:
                conflicts[path.name].update({truth[path.name], chart_type})
            truth[path.name] = chart_type
    if conflicts:
        compact = {name: sorted(types) for name, types in conflicts.items()}
        raise ValueError(f"Conflicting ground-truth labels: {compact}")
    return truth


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_for_orientation_family(chart_type: str) -> str:
    if chart_type in {"v_bar", "v_stacked_bar"}:
        return "v_bar"
    if chart_type in {"h_bar", "h_stacked_bar"}:
        return "h_bar"
    return chart_type


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row.get("correct"))
    family_correct = sum(1 for row in records if row.get("orientation_family_correct"))
    by_type: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        grouped[str(row["gt_type"])].append(row)
        confusion[str(row["gt_type"])][str(row["pred_type"])] += 1

    for chart_type in sorted(grouped):
        rows = grouped[chart_type]
        n = len(rows)
        type_correct = sum(1 for row in rows if row.get("correct"))
        type_family_correct = sum(1 for row in rows if row.get("orientation_family_correct"))
        confidences = [
            float(row["confidence"])
            for row in rows
            if isinstance(row.get("confidence"), (int, float))
        ]
        by_type[chart_type] = {
            "total": n,
            "correct": type_correct,
            "accuracy": type_correct / n if n else None,
            "orientation_family_correct": type_family_correct,
            "orientation_family_accuracy": type_family_correct / n if n else None,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "orientation_family_correct": family_correct,
        "orientation_family_accuracy": family_correct / total if total else None,
        "by_type": by_type,
        "confusion": {
            chart_type: dict(counter.most_common())
            for chart_type, counter in sorted(confusion.items())
        },
        "errors": [
            row for row in records if not row.get("correct")
        ],
    }


def classify_one(image_path: Path, gt_type: str) -> dict[str, Any]:
    detector = ChartTypeDetector()
    result = detector.detect_chart_type(str(image_path))
    pred_type = str(result.get("type", "unknown"))
    confidence = result.get("confidence")
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "gt_type": gt_type,
        "pred_type": pred_type,
        "confidence": confidence,
        "correct": pred_type == gt_type,
        "orientation_family_correct": (
            normalize_for_orientation_family(pred_type)
            == normalize_for_orientation_family(gt_type)
        ),
        "error": result.get("error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"F:\program\grid\Final-RealDataset"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "backend" / "evaluation" / "results" / "realdataset_mllm_chart_type_work.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    image_root = dataset_root / "ALL" / "charts"
    truth = build_ground_truth(dataset_root)
    images = sorted(image_root.glob("*.png"))
    if args.limit > 0:
        images = images[: args.limit]

    payload = load_json(args.output, {})
    records: list[dict[str, Any]] = [] if args.force else list(payload.get("records", []))
    done = {row.get("image_name") for row in records}
    detector = ChartTypeDetector()
    model_name = detector.model_name

    started = time.time()
    pending: list[tuple[int, Path, str]] = []
    for index, image_path in enumerate(images, start=1):
        if image_path.name in done:
            print(f"[{index}/{len(images)}] cached {image_path.name}", flush=True)
        else:
            gt_type = truth.get(image_path.name)
            if not gt_type:
                print(f"[{index}/{len(images)}] skipped without gt {image_path.name}", flush=True)
            else:
                pending.append((index, image_path, gt_type))

    def save_progress() -> None:
        write_json(
            args.output,
            {
                "dataset_root": str(dataset_root),
                "image_root": str(image_root),
                "model": model_name,
                "records": records,
                "summary": summarize(records),
                "elapsed_seconds": round(time.time() - started, 3),
            },
        )

    if args.workers <= 1:
        for index, image_path, gt_type in pending:
            record = classify_one(image_path, gt_type)
            records.append(record)
            save_progress()
            print(
                f"[{index}/{len(images)}] {image_path.name}: gt={gt_type} pred={record['pred_type']} "
                f"ok={record['correct']} conf={record['confidence']}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_meta = {
                executor.submit(classify_one, image_path, gt_type): (index, image_path, gt_type)
                for index, image_path, gt_type in pending
            }
            for future in as_completed(future_to_meta):
                index, image_path, gt_type = future_to_meta[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "gt_type": gt_type,
                        "pred_type": "error",
                        "confidence": 0,
                        "correct": False,
                        "orientation_family_correct": False,
                        "error": str(exc),
                    }
                records.append(record)
                save_progress()
                print(
                    f"[{index}/{len(images)}] {image_path.name}: gt={gt_type} pred={record['pred_type']} "
                    f"ok={record['correct']} conf={record['confidence']}",
                    flush=True,
                )

    final_payload = {
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "model": model_name,
        "records": records,
        "summary": summarize(records),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(args.output, final_payload)
    summary = final_payload["summary"]
    print(
        f"DONE total={summary['total']} correct={summary['correct']} "
        f"accuracy={summary['accuracy']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
