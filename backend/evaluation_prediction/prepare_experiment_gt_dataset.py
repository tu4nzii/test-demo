"""Prepare the external experiment GT dataset in a clean loader-friendly layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class DatasetSpec:
    source_dir: str
    output_source: str
    output_group: str
    chart_type: str
    flat: bool = False
    chart_subdir: str | None = None
    config_subdir: str | None = None
    image_subdir: str | None = None
    grid_suffix: str = "_grid.png"
    config_suffix: str = "_encrypted.json"
    synthetic_attributes: bool = False


SPECS = [
    DatasetSpec(
        source_dir="hbar_real_datasets",
        output_source="Final-RealDataset",
        output_group="hBar_GT",
        chart_type="h_bar",
        chart_subdir="chart",
        config_subdir="chart_configs",
    ),
    DatasetSpec(
        source_dir="vbar_real_datasets",
        output_source="Final-RealDataset",
        output_group="vBar_GT",
        chart_type="v_bar",
        chart_subdir="chart",
        config_subdir="chart_configs",
    ),
    DatasetSpec(
        source_dir="line_real_datasets",
        output_source="Final-RealDataset",
        output_group="Line_GT",
        chart_type="line",
        chart_subdir="chart",
        config_subdir="chart_configs",
    ),
    DatasetSpec(
        source_dir="scatter_bubble_real_datasets",
        output_source="Final-RealDataset",
        output_group="Scatter_GT",
        chart_type="scatter",
        chart_subdir="charts",
        config_subdir="chart_configs",
        image_subdir="scatter",
    ),
    DatasetSpec(
        source_dir="scatter_bubble_real_datasets",
        output_source="Final-RealDataset",
        output_group="Bubble_GT",
        chart_type="bubble",
        chart_subdir="charts",
        config_subdir="chart_configs",
        image_subdir="bubble",
    ),
    DatasetSpec(
        source_dir="Real_radarchart_withgrid",
        output_source="Final-RealDataset",
        output_group="Radar_GT",
        chart_type="radar",
        flat=True,
        grid_suffix="_gt_encrypt.png",
        config_suffix=".json",
    ),
    DatasetSpec(
        source_dir="Realrosechart_withgrid",
        output_source="Final-RealDataset",
        output_group="Rose_GT",
        chart_type="rose",
        flat=True,
        grid_suffix="_gt_encrypt.png",
        config_suffix=".json",
    ),
    DatasetSpec(
        source_dir="Syradarchart_eval_50_withgrid",
        output_source="Sy.Dataset",
        output_group="Radar_50_GT",
        chart_type="radar",
        flat=True,
        chart_subdir="eval_50",
        grid_suffix="+encode.jpg",
        config_suffix=".json",
        synthetic_attributes=True,
    ),
    DatasetSpec(
        source_dir="Sy_rosechart_eval_50_withgrid",
        output_source="Sy.Dataset",
        output_group="Rose_50_GT",
        chart_type="rose",
        flat=True,
        chart_subdir="eval_50",
        grid_suffix="+encode.jpg",
        config_suffix=".json",
        synthetic_attributes=True,
    ),
]


def safe_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    return text.strip(" ._") or "chart"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, Any], *, apply: bool) -> None:
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path, *, apply: bool) -> None:
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_config(
    *,
    source_config: Path,
    chart_type: str,
    chart_id: str,
    original_name: str,
    grid_name: str,
    source_root: Path,
) -> dict[str, Any]:
    data = read_json(source_config)
    data["chart_id"] = str(data.get("chart_id") or chart_id)
    data["chart_type"] = str(data.get("chart_type") or chart_type)
    data["image_paths"] = {
        "no_grid": f"chart/{original_name}",
        "with_grid": f"gt_grid/{grid_name}",
        "grid_with_grid": f"gt_grid/{grid_name}",
    }
    data["gt_grid_source"] = "provided"
    data["source_config_path"] = str(source_config.relative_to(source_root))
    return data


def iter_pairs(root: Path, spec: DatasetSpec) -> Iterable[tuple[str, Path, Path, Path]]:
    source_root = root / spec.source_dir
    if spec.flat:
        base = source_root / spec.chart_subdir if spec.chart_subdir else source_root
        for config_path in sorted(base.glob("*.json")):
            if config_path.name.endswith("_attributes.json"):
                continue
            stem = config_path.stem
            image_path = first_existing(base, [f"{stem}.png", f"{stem}.jpg", f"{stem}.jpeg"])
            grid_path = base / f"{stem}{spec.grid_suffix}"
            if image_path and grid_path.exists():
                yield stem, image_path, grid_path, config_path
        return

    image_base = source_root / str(spec.chart_subdir)
    config_base = source_root / str(spec.config_subdir)
    if spec.image_subdir:
        image_base = image_base / spec.image_subdir
        config_base = config_base / spec.image_subdir
    for image_path in sorted(image_base.iterdir() if image_base.exists() else []):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem.endswith("_grid") or image_path.stem.endswith("_with_grid") or "+encode" in image_path.stem:
            continue
        stem = image_path.stem
        grid_path = image_path.with_name(f"{stem}{spec.grid_suffix}")
        config_path = config_base / f"{stem}{spec.config_suffix}"
        if grid_path.exists() and config_path.exists():
            yield stem, image_path, grid_path, config_path


def first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def ensure_clean_output(path: Path, root: Path, *, apply: bool) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if root_resolved not in resolved.parents:
        raise RuntimeError(f"Refuse to clean outside dataset root: {path}")
    if apply and path.exists():
        shutil.rmtree(path)


def prepare_dataset(root: Path, output: Path, *, apply: bool, clean: bool) -> dict[str, Any]:
    if clean:
        ensure_clean_output(output, root, apply=apply)
    counts: dict[str, int] = {}
    missing: list[dict[str, str]] = []
    records: list[dict[str, str]] = []

    for spec in SPECS:
        key = f"{spec.output_source}/{spec.output_group}"
        group_count = 0
        source_root = root / spec.source_dir
        if not source_root.exists():
            missing.append({"group": key, "reason": "missing_source_dir", "path": str(source_root)})
            continue
        for stem, image_path, grid_path, config_path in iter_pairs(root, spec):
            original_name = image_path.name
            grid_name = grid_path.name
            config_name = f"{safe_name(stem)}_encrypted.json"
            out_group = output / spec.output_source / spec.output_group
            out_image = out_group / "chart" / original_name
            out_grid = out_group / "gt_grid" / grid_name
            out_config = out_group / "chart_configs" / config_name

            config = prepare_config(
                source_config=config_path,
                chart_type=spec.chart_type,
                chart_id=stem,
                original_name=original_name,
                grid_name=grid_name,
                source_root=root,
            )
            copy_file(image_path, out_image, apply=apply)
            copy_file(grid_path, out_grid, apply=apply)
            write_json(out_config, config, apply=apply)
            records.append(
                {
                    "source": spec.output_source,
                    "group": spec.output_group,
                    "chart_type": spec.chart_type,
                    "chart_id": str(config.get("chart_id") or stem),
                    "image": str(out_image.relative_to(output)),
                    "gt_grid": str(out_grid.relative_to(output)),
                    "config": str(out_config.relative_to(output)),
                }
            )
            group_count += 1
        counts[key] = group_count

    manifest = {
        "layout_version": "experiment_gt_dataset_clean_v1",
        "root": str(output),
        "records": records,
        "counts": counts,
        "missing": missing,
    }
    write_json(output / "manifest.json", manifest, apply=apply)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare experiment_gt_dataset into a clean GT experiment layout.")
    parser.add_argument("--root", type=Path, default=Path("experiment_gt_dataset"))
    parser.add_argument("--output", type=Path, default=Path("experiment_gt_dataset") / "organized")
    parser.add_argument("--apply", action="store_true", help="Actually copy files and write normalized JSON.")
    parser.add_argument("--clean", action="store_true", help="Clear output before writing. Only applies with --apply.")
    args = parser.parse_args()

    manifest = prepare_dataset(args.root.resolve(), args.output.resolve(), apply=args.apply, clean=args.clean)
    mode = "APPLY" if args.apply else "DRY"
    print(f"{mode} output={args.output.resolve()}")
    for key, count in sorted(manifest["counts"].items()):
        print(f"{key}: {count}")
    if manifest["missing"]:
        print("missing:")
        for item in manifest["missing"]:
            print(json.dumps(item, ensure_ascii=False))
    print(f"records={len(manifest['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
