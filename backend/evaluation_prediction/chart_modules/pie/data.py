"""Data loading for backend-generated pie prediction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import os

from ...common.chart_io import filter_chart_configs, load_json_configs, read_json
from ...common.gt_grid_renderer import render_gt_grid_image
from ...common.paths import ASSETS_ROOT


CONFIG_DIR = ASSETS_ROOT / "pie" / "chart_configs"


@dataclass(frozen=True)
class CircularTarget:
    chart_id: str
    point_name: str
    visual_name: str


def load_datasets(
    chart_ids: Iterable[str] | None = None,
    config_paths: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if config_paths:
        datasets = [_load_backend_generated_dataset(path, "pie") for path in config_paths]
        return filter_chart_configs(datasets, chart_ids)
    return filter_chart_configs(load_json_configs(CONFIG_DIR, recursive=True, exclude_emu=True), chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[CircularTarget]:
    return [
        CircularTarget(
            chart_id=str(dataset["chart_id"]),
            point_name=name,
            visual_name=name,
        )
        for name in _target_names(dataset)
    ]


def image_path(dataset: dict[str, Any], image_type: str) -> Path:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    if image_type == "grid_with_grid":
        value = image_paths.get("grid_with_grid") or image_paths.get("with_grid") or dataset.get("encrypted_grid_path")
    elif image_type == "with_grid":
        value = image_paths.get("with_grid") or image_paths.get("grid_with_grid") or dataset.get("encrypted_grid_path")
    else:
        value = image_paths.get("no_grid") or dataset.get("image_path")
    if not value:
        raise KeyError(f"Missing image path for {image_type!r} in {dataset.get('chart_id')!r}")
    return Path(value).resolve()


def _load_backend_generated_dataset(config_path: str | Path, chart_type: str) -> dict[str, Any]:
    path = Path(config_path).resolve()
    data = _read_json_dict(path)
    if path.stem.endswith("_axes"):
        image_json = path.with_name(f"{path.stem.removesuffix('_axes')}_image.json")
        data = _read_json_dict(image_json) | data

    axes = _read_json_dict(_sibling_axes_path(path))
    merged = dict(data)
    merged.update({key: value for key, value in axes.items() if key not in {"chart_id"}})
    _strip_reference_fields(merged)
    merged["chart_type"] = chart_type
    merged["chart_id"] = str(merged.get("chart_id") or path.stem.removesuffix("_image").removesuffix("_axes"))
    merged["image_paths"] = _image_paths(merged, path.parent, path)
    merged["target_labels"] = _label_map(merged)
    return merged


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _sibling_axes_path(path: Path) -> Path:
    stem = path.stem.removesuffix("_image").removesuffix("_axes")
    candidates = [
        path.with_name(f"{stem}_axes.json"),
        path.with_name(f"{stem}_image_axes.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _image_paths(dataset: dict[str, Any], base_dir: Path, config_path: Path) -> dict[str, str]:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    no_grid = image_paths.get("no_grid") or dataset.get("image_path")
    with_grid = (
        image_paths.get("grid_with_grid")
        or image_paths.get("with_grid")
        or dataset.get("encrypted_grid_path")
        or dataset.get("basic_grid_path")
    )
    if not with_grid or not _resolve_path(str(with_grid), base_dir).exists():
        rendered = render_gt_grid_image(config_path, dataset=dataset)
        if rendered is not None:
            with_grid = str(rendered)
    paths = {"no_grid": no_grid, "with_grid": with_grid, "grid_with_grid": with_grid}
    return {key: str(_resolve_path(value, base_dir)) for key, value in paths.items() if isinstance(value, str) and value}


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    dataset_root = base_dir.parent
    candidates = [
        (base_dir / path).resolve(),
        (dataset_root / path).resolve(),
        (dataset_root / "charts" / path.name).resolve(),
        (dataset_root / "chart" / path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _label_map(dataset: dict[str, Any]) -> dict[str, None]:
    names = []
    data_points = dataset.get("data_points")
    if isinstance(data_points, dict):
        for name in data_points:
            text = str(name).strip()
            if text and text not in names:
                names.append(text)
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict):
        for name in series_color:
            text = str(name).strip()
            if text and text not in names:
                names.append(text)
    colors = dataset.get("colors")
    if isinstance(colors, list):
        for item in colors:
            if isinstance(item, dict) and item.get("name"):
                name = str(item["name"]).strip()
                if name and name not in names:
                    names.append(name)
    return {name: None for name in names}


def _strip_reference_fields(dataset: dict[str, Any]) -> None:
    if os.getenv("CHART_EXPERIMENT_PRESERVE_GT", "").strip().lower() in {"1", "true", "yes"}:
        for key in ("reference_config_path", "reference_chart_id"):
            dataset.pop(key, None)
        return
    for key in ("data", "data_points", "ground_truth", "labels", "reference_config_path", "reference_chart_id"):
        dataset.pop(key, None)


def _target_names(dataset: dict[str, Any]) -> list[str]:
    labels = _label_map(dataset)
    return [name for name in labels if name and name.lower() not in {"series 1", "系列1"}]
