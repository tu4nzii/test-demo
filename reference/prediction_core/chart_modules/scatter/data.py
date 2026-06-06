"""Data loading for scatter charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from reference.prediction_core.chart_io import filter_chart_configs, load_json_configs
from reference.prediction_core.specs import PROJECT_ROOT


ASSETS_ROOT = PROJECT_ROOT / "prediction_core" / "assets"


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
        return self.asset_root / self.result_dir_name


@dataclass(frozen=True)
class PointTarget:
    chart_id: str
    point_name: str
    visual_name: str
    gt_x: float
    gt_y: float


def _belongs_to_chart_type(dataset: dict[str, Any], chart_type: str) -> bool:
    chart_id = str(dataset.get("chart_id", ""))
    return chart_id.startswith(f"{chart_type}_")


def load_datasets(config: PointChartConfig, chart_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    configs = load_json_configs(config.asset_root / "chart_configs", recursive=True, exclude_emu=True)
    configs = [item for item in configs if _belongs_to_chart_type(item, config.chart_type)]
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[PointTarget]:
    targets: list[PointTarget] = []
    color_names = list(dataset.get("series_color", {}))
    for index, (point_name, coords) in enumerate(dataset["data_points"].items()):
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
                gt_x=float(coords[0]),
                gt_y=float(coords[1]),
            )
        )
    return targets


def image_path(config: PointChartConfig, dataset: dict[str, Any], image_type: str) -> Path:
    return config.asset_root / dataset["image_paths"][image_type]
