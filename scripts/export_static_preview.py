"""Export dataset preview cache as static frontend assets.

The generated files are consumed by the Vue app when no backend is available,
for example on GitHub Pages. The exporter reuses the backend preview registry
and copies only the files needed by the UI: original images, standard/color
grid previews, and cached evaluation JSON.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    DATASET_CATEGORY_PRIORITY,
    DATASET_SOURCE_ROOTS,
    dataset_category_options,
    dataset_evaluation_cache_path,
    evaluation_cache_usable,
    iter_dataset_samples,
    register_dataset_sample,
)


STATIC_PREVIEW_DIR = PROJECT_ROOT / "frontend" / "chart-demo-ui" / "public" / "static-preview"
SOURCE_LABELS = {
    "realworld": "Final-RealDataset",
    "synthetic": "Sy.Dataset",
}


def copy_asset(src: Any, dest: Path) -> str | None:
    if not src:
        return None
    src_path = Path(str(src))
    if not src_path.exists() or not src_path.is_file():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return dest.relative_to(STATIC_PREVIEW_DIR).as_posix()


def export_sample(sample: dict[str, Any]) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    chart_info = register_dataset_sample(sample_id)
    asset_dir = STATIC_PREVIEW_DIR / "assets" / sample_id

    original_src = Path(str(chart_info["image_path"]))
    original_ext = original_src.suffix.lower() or ".png"
    image_url = copy_asset(original_src, asset_dir / f"original{original_ext}")

    standard_grid_url = copy_asset(chart_info.get("encrypted_image_path"), asset_dir / "standard_grid.png")
    colored_grid_url = copy_asset(chart_info.get("colored_image_path"), asset_dir / "colored_grid.png")

    chart_type = str(chart_info.get("chart_type") or sample.get("chart_type") or "")
    evaluation_path = dataset_evaluation_cache_path(sample_id)
    results_url = None
    if evaluation_cache_usable(evaluation_path, chart_type):
        results_url = copy_asset(evaluation_path, asset_dir / "evaluation.json")

    exported = {
        "sample_id": sample_id,
        "source": sample.get("source"),
        "name": sample.get("name"),
        "filename": sample.get("filename"),
        "relative_path": sample.get("relative_path"),
        "category": sample.get("category"),
        "chart_type": chart_type,
        "coordinate_system": sample.get("coordinate_system"),
        "chart_id": chart_info.get("chart_id") or f"dataset_{sample_id}",
        "confidence": chart_info.get("confidence", 1.0),
        "image_url": image_url,
        "standard_grid_url": standard_grid_url,
        "encrypted_image_url": standard_grid_url,
        "colored_grid_url": colored_grid_url,
        "results_url": results_url,
        "cached": bool(standard_grid_url),
        "evaluation_cached": bool(results_url),
        "evaluated": bool(results_url),
        "static_preview": True,
    }
    return exported


def export_static_preview(clean: bool = True) -> dict[str, Any]:
    if clean and STATIC_PREVIEW_DIR.exists():
        shutil.rmtree(STATIC_PREVIEW_DIR)
    STATIC_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {},
    }
    total_samples = 0
    total_cached = 0
    total_evaluated = 0

    for source in DATASET_SOURCE_ROOTS:
        categories = dataset_category_options(source)
        samples = []
        for sample in iter_dataset_samples(source):
            exported = export_sample(sample)
            if not exported.get("image_url"):
                continue
            samples.append(exported)
            total_samples += 1
            total_cached += int(bool(exported.get("cached")))
            total_evaluated += int(bool(exported.get("evaluation_cached")))

        samples.sort(
            key=lambda item: (
                not item.get("cached"),
                DATASET_CATEGORY_PRIORITY.get(str(item.get("chart_type")), 99),
                str(item.get("name") or ""),
            )
        )
        manifest["sources"][source] = {
            "label": SOURCE_LABELS.get(source, source),
            "categories": categories,
            "samples": samples,
        }

    manifest["summary"] = {
        "samples": total_samples,
        "cached": total_cached,
        "evaluation_cached": total_evaluated,
    }
    manifest_path = STATIC_PREVIEW_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clean", action="store_true", help="keep existing static-preview files")
    args = parser.parse_args()
    manifest = export_static_preview(clean=not args.no_clean)
    summary = manifest.get("summary", {})
    print(
        "Exported static preview: "
        f"samples={summary.get('samples', 0)}, "
        f"cached={summary.get('cached', 0)}, "
        f"evaluation_cached={summary.get('evaluation_cached', 0)}"
    )
    print(f"Output: {STATIC_PREVIEW_DIR}")


if __name__ == "__main__":
    main()
