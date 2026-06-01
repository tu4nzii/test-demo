"""Data loading and target enumeration for horizontal bar charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ...common.chart_io import filter_chart_configs, load_json_configs
from ...common.backend_dataset import (
    load_backend_generated_datasets,
    resolve_image_path,
)
from ...common.paths import ASSETS_ROOT


H_BAR_ROOT = ASSETS_ROOT / "h_bar"
CONFIG_DIR = H_BAR_ROOT / "chart_configs"


@dataclass(frozen=True)
class HBarTarget:
    chart_id: str
    point_name: str
    series_name: str
    y_label: str
    gt_x: float | None
    gt_y: str


def load_datasets(
    chart_ids: Iterable[str] | None = None,
    config_paths: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if config_paths:
        return load_backend_generated_datasets(config_paths, "h_bar", chart_ids)
    configs = load_json_configs(CONFIG_DIR)
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[HBarTarget]:
    targets: list[HBarTarget] = []
    data_points = dataset.get("data_points") if isinstance(dataset.get("data_points"), dict) else {}
    for series_name, sub_points in data_points.items():
        for y_label, x_value in sub_points.items():
            targets.append(
                HBarTarget(
                    chart_id=dataset["chart_id"],
                    point_name=f"{series_name}, {y_label}",
                    series_name=series_name,
                    y_label=y_label,
                    gt_x=float(x_value),
                    gt_y=str(y_label),
                )
            )
    if targets:
        return targets

    series_names = list(_series_names(dataset))
    y_ticks = dataset.get("y_ticks") if isinstance(dataset.get("y_ticks"), list) else []
    for series_name in series_names:
        for y_label in y_ticks:
            targets.append(
                HBarTarget(
                    chart_id=dataset["chart_id"],
                    point_name=f"{series_name}, {y_label}",
                    series_name=series_name,
                    y_label=str(y_label),
                    gt_x=None,
                    gt_y=str(y_label),
                )
            )
    return targets


def _series_names(dataset: dict[str, Any]) -> list[str]:
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict) and series_color:
        return [str(name) for name in series_color]
    colors = dataset.get("colors")
    if isinstance(colors, list) and colors:
        names = [
            str(item.get("name"))
            for item in colors
            if isinstance(item, dict) and item.get("name")
        ]
        if names:
            return names
    return ["series-0"]


def image_path(dataset: dict[str, Any], image_type: str) -> Path:
    path = resolve_image_path(dataset, image_type)
    if path.exists() or Path(dataset["image_paths"][image_type]).is_absolute():
        return path
    return H_BAR_ROOT / dataset["image_paths"][image_type]
