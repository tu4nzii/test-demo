"""Run backend grid/encrypted-grid generation for real-world chart images.

This intentionally stops after the processing/encryption stage. It does not run
evaluation_prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def safe_name(value: str, limit: int = 48) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = cleaned.strip("._") or "item"
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: max(8, limit - 9)]}_{digest}"


def short_artifact_name(src: Path, limit: int = 80) -> str:
    stem = safe_name(src.stem, max(16, limit - len(src.suffix)))
    return f"{stem}{src.suffix}"


def copy_if_exists(path: str | Path | None, dest_dir: Path, dest_name: str | None = None) -> str | None:
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (dest_name or short_artifact_name(src))
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
            dest = dest_dir / short_artifact_name(src, limit=52)
            try:
                shutil.copy2(src, dest)
            except OSError as exc:
                copied.append(f"SKIPPED:{src.name}:{exc}")
                continue
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
    parser.add_argument(
        "--include-dirs",
        default=None,
        help="Comma-separated first-level directory names to include.",
    )
    return parser.parse_args()


def write_contact_sheet(recheck_root: Path, manifest: dict) -> str | None:
    records = [record for record in manifest.get("records", []) if record.get("status") == "success"]
    if not records:
        return None

    thumb_w, thumb_h = 360, 280
    label_h = 38
    gap = 18
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    def fit(path: str | Path, width: int, height: int) -> Image.Image:
        img = Image.open(path).convert("RGB")
        img.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
        return canvas

    sheet_w = thumb_w * 2 + gap * 3
    sheet_h = (thumb_h + label_h + gap) * len(records) + gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    y = gap
    for record in records:
        copied = record.get("copied") or {}
        source = copied.get("source_image")
        encrypted = copied.get("encrypted_grid")
        if not source or not encrypted or not Path(source).exists() or not Path(encrypted).exists():
            continue
        x1 = gap
        x2 = gap * 2 + thumb_w
        sheet.paste(fit(source, thumb_w, thumb_h), (x1, y + label_h))
        sheet.paste(fit(encrypted, thumb_w, thumb_h), (x2, y + label_h))
        rel = Path(str(record.get("relative", record.get("source", "")))).name
        draw.text((x1, y), f"Original: {rel}", fill=(20, 20, 20), font=font)
        draw.text((x2, y), f"With grid: {record.get('detected_type')}", fill=(20, 20, 20), font=font)
        draw.rectangle([x1, y + label_h, x1 + thumb_w - 1, y + label_h + thumb_h - 1], outline=(210, 210, 210))
        draw.rectangle([x2, y + label_h, x2 + thumb_w - 1, y + label_h + thumb_h - 1], outline=(210, 210, 210))
        y += thumb_h + label_h + gap

    out = recheck_root / "contact_sheet_original_vs_with_grid.png"
    sheet.save(out)
    return str(out)


def main() -> int:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    source_root = Path(args.source_root).resolve()
    run_name = args.run_name or f"realworld_grid_{get_profile_name()}_{get_model_name()}_latest"
    run_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_name)
    recheck_root = BACKEND / "evaluation" / "recheck_outputs" / run_name
    recheck_root.mkdir(parents=True, exist_ok=True)

    include_dirs = None
    if args.include_dirs:
        include_dirs = {item.strip().lower() for item in args.include_dirs.split(",") if item.strip()}
    images = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if include_dirs:
            try:
                rel = path.relative_to(source_root)
            except ValueError:
                continue
            first = rel.parts[0].lower() if len(rel.parts) > 1 else source_root.name.lower()
            if first not in include_dirs:
                continue
        images.append(path.resolve())
    images = sorted(images)
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
        safe_stem = safe_name(image_src.stem, 40)
        chart_id = f"{get_profile_name()}_grid_{group}_{safe_stem}"
        item_dir = recheck_root / group / safe_stem
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
            chart_type = normalize_chart_type(detection.get("type"))
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
                "source_image": copy_if_exists(image_src, copied_dir, f"source{image_src.suffix.lower()}"),
                "uploaded_image": copy_if_exists(upload_path, copied_dir, f"upload{image_src.suffix.lower()}"),
                "encrypted_grid": copy_if_exists(encrypted_path, copied_dir, "image_with_grid.png"),
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
    contact_sheet = write_contact_sheet(recheck_root, manifest)
    if contact_sheet:
        summary["contact_sheet"] = contact_sheet
        (recheck_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Contact sheet: {contact_sheet}", flush=True)
    print(f"Done: success={manifest['success']} failed={manifest['failed']}", flush=True)
    print(f"Summary: {recheck_root / 'summary.json'}", flush=True)
    return 0 if manifest["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
