"""Run backend grid/encrypted-grid generation for real-world chart images.

This intentionally stops after the processing/encryption stage. It does not run
evaluation_prediction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from model_api_config import get_base_url, get_model_name, get_profile_name  # noqa: E402
import main as backend_main  # noqa: E402
from type_detection.chart_registry import (  # noqa: E402
    DEFAULT_CHART_TYPE,
    get_coordinate_system,
    normalize_chart_type,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def copy_if_exists(path: str | Path | None, dest_dir: Path) -> str | None:
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    return str(dest)


def copy_generated_sidecars(output_dir: Path, image_stem: str, chart_id: str, dest_dir: Path) -> list[str]:
    copied: list[str] = []
    seen: set[Path] = set()
    for pattern in (f"{image_stem}*", f"{chart_id}*"):
        for src in output_dir.glob(pattern):
            if not src.is_file() or src in seen:
                continue
            seen.add(src)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            copied.append(str(dest))
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-world grid-only processing.")
    parser.add_argument(
        "--source-root",
        default=str(BACKEND / "realworldcharts"),
        help="Root containing real-world chart images.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output folder name under backend/evaluation/recheck_outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    source_root = Path(args.source_root)
    run_name = args.run_name or f"realworld_grid_{get_profile_name()}_{get_model_name()}_{run_id}"
    run_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_name)
    recheck_root = BACKEND / "evaluation" / "recheck_outputs" / run_name
    recheck_root.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    manifest = {
        "run_id": run_id,
        "mode": "grid_encryption_only",
        "source_root": str(source_root),
        "recheck_root": str(recheck_root),
        "model_profile": get_profile_name(),
        "model_name": get_model_name(),
        "base_url": get_base_url(),
        "total": len(images),
        "success": 0,
        "failed": 0,
        "records": [],
    }

    def write_manifest() -> None:
        (recheck_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Start realworld grid-only run: {len(images)} images", flush=True)
    print(
        f"Profile={get_profile_name()} model={get_model_name()} base={get_base_url()}",
        flush=True,
    )

    for index, image_src in enumerate(images, start=1):
        rel = image_src.relative_to(source_root)
        group = rel.parts[0] if len(rel.parts) > 1 else "root"
        safe_stem = image_src.stem.replace(" ", "_").replace(",", "_")
        chart_id = f"{get_profile_name()}_grid_{group}_{safe_stem}_{run_id}"
        item_dir = recheck_root / group / image_src.stem
        upload_path = backend_main.UPLOAD_DIR / f"{chart_id}_image{image_src.suffix.lower()}"
        record = {
            "index": index,
            "source": str(image_src),
            "relative": str(rel),
            "chart_id": chart_id,
            "group": group,
            "status": "pending",
        }

        try:
            shutil.copy2(image_src, upload_path)
            print(f"[{index}/{len(images)}] detect/process {rel}", flush=True)
            detection = backend_main.detect_chart_type(upload_path)
            chart_type = normalize_chart_type(detection.get("type", DEFAULT_CHART_TYPE))
            chart_info = {
                "chart_id": chart_id,
                "chart_type": chart_type,
                "coordinate_system": get_coordinate_system(chart_type).value,
                "confidence": detection.get("confidence", 0.0),
                "axis_repair": detection.get("axis_repair") or {},
                "image_path": str(upload_path),
                "json_path": None,
                "processed": False,
                "evaluated": False,
            }
            encrypted_path = backend_main.process_chart_image(chart_info)
            output_dir = Path(chart_info.get("output_dir") or backend_main.OUTPUT_DIR / chart_info["chart_type"])
            copied_dir = item_dir / "artifacts"
            copied = {
                "source_image": copy_if_exists(image_src, copied_dir),
                "uploaded_image": copy_if_exists(upload_path, copied_dir),
                "encrypted_grid": copy_if_exists(encrypted_path, copied_dir),
            }
            sidecars = copy_generated_sidecars(output_dir, upload_path.stem, chart_id, copied_dir)
            record.update(
                {
                    "status": "success",
                    "detected_type": chart_info["chart_type"],
                    "confidence": chart_info.get("confidence"),
                    "axis_repair": chart_info.get("axis_repair"),
                    "output_dir": str(output_dir),
                    "encrypted_grid_path": str(encrypted_path),
                    "copied": copied,
                    "sidecars": sidecars,
                }
            )
            manifest["success"] += 1
            print(f"[{index}/{len(images)}] OK {rel} -> {chart_info['chart_type']}", flush=True)
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            manifest["failed"] += 1
            print(f"[{index}/{len(images)}] FAIL {rel}: {exc}", flush=True)
        finally:
            manifest["records"].append(record)
            write_manifest()

    summary = {
        "run_id": run_id,
        "total": manifest["total"],
        "success": manifest["success"],
        "failed": manifest["failed"],
        "recheck_root": str(recheck_root),
        "model_profile": manifest["model_profile"],
        "model_name": manifest["model_name"],
        "base_url": manifest["base_url"],
    }
    (recheck_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done: success={manifest['success']} failed={manifest['failed']}", flush=True)
    print(f"Summary: {recheck_root / 'summary.json'}", flush=True)
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
