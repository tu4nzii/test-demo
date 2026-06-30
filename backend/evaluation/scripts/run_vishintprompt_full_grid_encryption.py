"""Run grid encryption for the imported VisHintPrompt datasets.

The script keeps a fixed output directory by default so repeated runs overwrite
or reuse the latest artifacts instead of creating timestamped folders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
from model_api_config import get_base_url, get_model_name, get_profile_name  # noqa: E402
from type_detection.chart_registry import get_coordinate_system, normalize_chart_type  # noqa: E402
from type_detection.chart_type import ChartTypeDetector  # noqa: E402


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
DATASET_CATEGORY_PREFIXES = (
    "Bubble_",
    "Donut_",
    "hBar_",
    "Line_",
    "Pie_",
    "Radar_",
    "Rose_",
    "Scatter_",
    "vBar_",
)


def safe_name(value: str, limit: int = 64) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")
    if not cleaned:
        cleaned = "item"
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: max(8, limit - 9)]}_{digest}"


def infer_chart_type(path: Path) -> str:
    text = path.as_posix().lower()
    name = path.stem.lower()
    if "bubble" in text:
        return "bubble"
    if "scatter" in text or "scatetr" in text:
        return "scatter"
    if "line" in text:
        return "line"
    if "donut" in text:
        return "donut"
    if "pie" in text:
        return "pie"
    if "radar" in text:
        return "radar"
    if "rose" in text or "nightingale" in name:
        return "rose"
    if "hbar" in text or "xbar" in text or "horizontal" in text:
        return "h_bar"
    if "vbar" in text or "bar" in text:
        return "v_bar"
    return "v_bar"


def iter_dataset_images(import_root: Path) -> list[tuple[str, Path, Path]]:
    datasets: list[tuple[str, Path, Path]] = []
    final_root = import_root / "Final-RealDataset"
    if final_root.exists():
        for group_dir in sorted(final_root.iterdir()):
            if not group_dir.is_dir() or group_dir.name == "ALL":
                continue
            if not any(group_dir.name.startswith(prefix) for prefix in DATASET_CATEGORY_PREFIXES):
                continue
            for pattern in ("charts", "chart"):
                chart_dir = group_dir / pattern
                if not chart_dir.exists():
                    continue
                for path in sorted(chart_dir.rglob("*")):
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                        datasets.append(("Final-RealDataset", final_root, path))

    synthetic_root = import_root / "Sy.Dataset"
    if synthetic_root.exists():
        seen: set[Path] = set()
        for group_dir in sorted(synthetic_root.iterdir()):
            if not group_dir.is_dir():
                continue
            if not any(group_dir.name.startswith(prefix) for prefix in DATASET_CATEGORY_PREFIXES):
                continue
            chart_dirs = [path for path in (group_dir / "charts", group_dir / "chart") if path.exists()]
            if not chart_dirs:
                chart_dirs = [group_dir]
            for charts_dir in chart_dirs:
                for path in sorted(charts_dir.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    datasets.append(("Sy.Dataset", synthetic_root, path))
    return datasets


def copy_if_exists(path: str | Path | None, dest: Path) -> str | None:
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)


def copy_generated_sidecars(output_dir: Path, image_stem: str, chart_id: str, dest_dir: Path) -> list[str]:
    copied: list[str] = []
    seen: set[Path] = set()
    for pattern in (f"{image_stem}*", f"{chart_id}*"):
        for src in sorted(output_dir.glob(pattern)):
            if not src.is_file() or src in seen:
                continue
            seen.add(src)
            dest = dest_dir / src.name
            try:
                shutil.copy2(src, dest)
                copied.append(str(dest))
            except OSError as exc:
                copied.append(f"SKIPPED:{src.name}:{exc}")
    return copied


def strip_source_json_refs(record: dict[str, Any]) -> dict[str, Any]:
    record.pop("source_json", None)
    copied = record.get("copied")
    if isinstance(copied, dict):
        copied.pop("source_json", None)
    return record


def load_existing_successes(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records = payload.get("records")
    if not isinstance(records, list):
        return {}
    successes = {}
    for record in records:
        if isinstance(record, dict) and record.get("status") == "success":
            key = str(record.get("dataset_relative") or "")
            copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
            encrypted = copied.get("encrypted_grid") or record.get("encrypted_grid_path")
            if key and encrypted and Path(str(encrypted)).exists():
                successes[key] = strip_source_json_refs(dict(record))
    return successes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full VisHintPrompt grid encryption.")
    parser.add_argument(
        "--import-root",
        default=str(BACKEND / "datasets" / "VisHintPrompt_datasets"),
        help="Imported VisHintPrompt dataset root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(BACKEND / "evaluation" / "recheck_outputs" / "vishintprompt_full_grid_encryption_latest"),
        help="Fixed output directory for artifacts and manifest.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for smoke tests.")
    parser.add_argument("--resume", action="store_true", help="Skip records already successful in manifest.")
    parser.add_argument("--shard-count", type=int, default=1, help="Split the dataset into N deterministic shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only this 0-based shard index.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import_root = Path(args.import_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    images = iter_dataset_images(import_root)
    full_total = len(images)
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must be in [0, shard-count)")
    if args.shard_count > 1:
        images = [
            item
            for item_index, item in enumerate(images)
            if item_index % args.shard_count == args.shard_index
        ]
    if args.limit and args.limit > 0:
        images = images[: args.limit]

    existing_successes = load_existing_successes(manifest_path) if args.resume else {}
    manifest: dict[str, Any] = {
        "run_id": time.strftime("%Y%m%d_%H%M%S"),
        "mode": "vishintprompt_full_grid_encryption",
        "import_root": str(import_root),
        "output_root": str(output_root),
        "model_profile": get_profile_name(),
        "model_name": get_model_name(),
        "base_url": get_base_url(),
        "full_total": full_total,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "total": len(images),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "records": [],
    }

    def write_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Start VisHintPrompt full grid encryption: {len(images)} images", flush=True)
    print(f"Input: {import_root}", flush=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Profile={get_profile_name()} model={get_model_name()} base={get_base_url()}", flush=True)
    detector = ChartTypeDetector()

    for index, (dataset_name, dataset_root, image_src) in enumerate(images, start=1):
        rel = image_src.relative_to(dataset_root)
        dataset_relative = f"{dataset_name}/{rel.as_posix()}"
        inferred_chart_type = normalize_chart_type(infer_chart_type(image_src))
        group = rel.parts[0] if len(rel.parts) > 1 else inferred_chart_type
        stem = safe_name(image_src.stem, 48)
        digest = hashlib.sha1(dataset_relative.encode("utf-8")).hexdigest()[:8]
        chart_id = f"vishint_{safe_name(dataset_name, 16)}_{safe_name(group, 20)}_{stem}_{digest}"
        item_dir = output_root / dataset_name / safe_name(group, 32) / f"{stem}_{digest}"
        artifact_dir = item_dir / "artifacts"
        upload_path = backend_main.UPLOAD_DIR / f"{chart_id}_image{image_src.suffix.lower()}"

        if dataset_relative in existing_successes:
            record = strip_source_json_refs(dict(existing_successes[dataset_relative]))
            record["status"] = "skipped_success_cache"
            manifest["records"].append(record)
            manifest["skipped"] += 1
            write_manifest()
            print(f"[{index}/{len(images)}] SKIP cached {dataset_relative}", flush=True)
            continue

        record: dict[str, Any] = {
            "index": index,
            "dataset": dataset_name,
            "dataset_relative": dataset_relative,
            "source_image": str(image_src),
            "chart_id": chart_id,
            "inferred_chart_type": inferred_chart_type,
            "group": group,
            "status": "pending",
        }

        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_src, upload_path)
            print(f"[{index}/{len(images)}] detect/process {dataset_relative}", flush=True)
            detection = detector.detect_chart_type(str(upload_path))
            chart_type = normalize_chart_type(detection["type"])
            axis_repair = detection.get("axis_repair") or {}
            record.update(
                {
                    "chart_type": chart_type,
                    "confidence": float(detection.get("confidence", 1.0)),
                    "axis_repair": axis_repair,
                    "series_items": axis_repair.get("series_items", {}).get("items", []),
                }
            )
            chart_info = {
                "chart_id": chart_id,
                "chart_type": chart_type,
                "coordinate_system": get_coordinate_system(chart_type).value,
                "confidence": float(detection.get("confidence", 1.0)),
                "axis_repair": axis_repair,
                "image_path": str(upload_path),
                "json_path": None,
                "processed": False,
                "evaluated": False,
                "output_dir": str(item_dir / "work"),
            }
            encrypted_path = backend_main.process_chart_image(chart_info, force=False)
            output_dir = Path(chart_info.get("output_dir") or item_dir / "work")
            copied = {
                "source_image": copy_if_exists(image_src, artifact_dir / f"source{image_src.suffix.lower()}"),
                "uploaded_image": copy_if_exists(upload_path, artifact_dir / f"upload{image_src.suffix.lower()}"),
                "encrypted_grid": copy_if_exists(encrypted_path, artifact_dir / "image_with_grid.png"),
            }
            tick_sidecar = output_dir / f"{upload_path.stem}_ticks.json"
            if tick_sidecar.exists():
                try:
                    ticks = json.loads(tick_sidecar.read_text(encoding="utf-8"))
                    copied["basic_grid"] = copy_if_exists(ticks.get("basic_grid_path"), artifact_dir / "image_grid.png")
                    copied["colored_grid"] = copy_if_exists(ticks.get("colored_grid_path"), artifact_dir / "image_with_grid_color.png")
                    copied["ticks_json"] = copy_if_exists(tick_sidecar, artifact_dir / "ticks.json")
                except Exception:
                    copied["ticks_json"] = copy_if_exists(tick_sidecar, artifact_dir / "ticks.json")
            sidecars = copy_generated_sidecars(output_dir, upload_path.stem, chart_id, artifact_dir)
            record.update(
                {
                    "status": "success",
                    "chart_type": chart_type,
                    "output_dir": str(output_dir),
                    "encrypted_grid_path": str(encrypted_path),
                    "copied": copied,
                    "sidecars": sidecars,
                }
            )
            manifest["success"] += 1
            print(f"[{index}/{len(images)}] OK {dataset_relative}", flush=True)
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            manifest["failed"] += 1
            print(f"[{index}/{len(images)}] FAIL {dataset_relative}: {exc}", flush=True)
        finally:
            manifest["records"].append(record)
            write_manifest()

    summary = {
        "run_id": manifest["run_id"],
        "total": manifest["total"],
        "success": manifest["success"],
        "failed": manifest["failed"],
        "skipped": manifest["skipped"],
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "model_profile": manifest["model_profile"],
        "model_name": manifest["model_name"],
        "base_url": manifest["base_url"],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done success={summary['success']} failed={summary['failed']} skipped={summary['skipped']}", flush=True)
    print(f"Summary: {output_root / 'summary.json'}", flush=True)
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
