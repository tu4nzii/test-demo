"""Data loading for bubble-like point charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ...common.backend_dataset import load_backend_generated_datasets, resolve_image_path
from ...common.chart_io import filter_chart_configs, load_json_configs
from ...common.paths import ASSETS_ROOT, RESULTS_ROOT


@dataclass(frozen=True)
class PointChartConfig:
    chart_type: str
    result_dir_name: str
    mark_name: str = "circle"

    @property
    def asset_root(self) -> Path:
        return ASSETS_ROOT / self.chart_type

    @property
    def result_root(self) -> Path:
        return RESULTS_ROOT / self.chart_type


@dataclass(frozen=True)
class PointTarget:
    chart_id: str
    point_name: str
    visual_name: str
    gt_x: float | None
    gt_y: float | None


def _belongs_to_chart_type(dataset: dict[str, Any], chart_type: str) -> bool:
    chart_id = str(dataset.get("chart_id", ""))
    return chart_id.startswith(f"{chart_type}_")


def load_datasets(
    config: PointChartConfig,
    chart_ids: Iterable[str] | None = None,
    config_paths: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if config_paths:
        return load_backend_generated_datasets(config_paths, config.chart_type, chart_ids)
    configs = load_json_configs(config.asset_root / "chart_configs", recursive=True, exclude_emu=True)
    configs = [item for item in configs if _belongs_to_chart_type(item, config.chart_type)]
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[PointTarget]:
    targets: list[PointTarget] = []
    color_names = list(dataset.get("series_color", {}))
    data_points = dataset.get("data_points") if isinstance(dataset.get("data_points"), dict) else {}
    for index, (point_name, coords) in enumerate(data_points.items()):
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        visual_name = str(point_name)
        if visual_name not in dataset.get("series_color", {}) and index < len(color_names):
            visual_name = str(color_names[index])
        targets.append(
            PointTarget(
                chart_id=dataset["chart_id"],
                point_name=str(point_name),
                visual_name=visual_name,
                gt_x=_float_or_none(coords[0]),
                gt_y=_float_or_none(coords[1]),
            )
        )
    if targets:
        return targets

    for name in _target_names_from_generated_json(dataset):
        targets.append(
            PointTarget(
                chart_id=dataset["chart_id"],
                point_name=name,
                visual_name=name,
                gt_x=None,
                gt_y=None,
            )
        )
    return targets


def _target_names_from_generated_json(dataset: dict[str, Any]) -> list[str]:
    names: list[str] = []
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict):
        names.extend(str(name) for name in series_color if str(name).strip())

    colors = dataset.get("colors")
    if isinstance(colors, list):
        for index, item in enumerate(colors, start=1):
            if not isinstance(item, dict):
                continue
            name = item.get("name") or f"{dataset.get('chart_type', 'point')}_{index}"
            text = str(name).strip()
            if text and text not in names:
                names.append(text)

    return names


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def image_path(config: PointChartConfig, dataset: dict[str, Any], image_type: str) -> Path:
    path = resolve_image_path(dataset, image_type)
    if path.exists() or Path(dataset["image_paths"][image_type]).is_absolute():
        return path
    return config.asset_root / dataset["image_paths"][image_type]
