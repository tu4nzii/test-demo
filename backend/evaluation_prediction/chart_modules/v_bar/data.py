"""Data loading and target enumeration for vertical bar charts."""

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


V_BAR_ROOT = ASSETS_ROOT / "v_bar"
CONFIG_DIR = V_BAR_ROOT / "chart_configs"


@dataclass(frozen=True)
class VBarTarget:
    chart_id: str
    point_name: str
    series_name: str
    x_label: str
    gt_x: str
    gt_y: float | None


def load_datasets(
    chart_ids: Iterable[str] | None = None,
    config_paths: Iterable[str | Path] | None = None,
    chart_type: str = "v_bar",
) -> list[dict[str, Any]]:
    if config_paths:
        return load_backend_generated_datasets(config_paths, chart_type, chart_ids)
    root = ASSETS_ROOT / chart_type
    config_dir = root / "chart_configs" if (root / "chart_configs").exists() else root
    configs = load_json_configs(config_dir, recursive=True, exclude_emu=True)
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[VBarTarget]:
    targets: list[VBarTarget] = []
    data_points = dataset.get("data_points") if isinstance(dataset.get("data_points"), dict) else {}
    for series_name, points in data_points.items():
        for x_label, y_value in points.items():
            targets.append(
                VBarTarget(
                    chart_id=dataset["chart_id"],
                    point_name=f"{series_name}, {x_label}",
                    series_name=series_name,
                    x_label=str(x_label),
                    gt_x=str(x_label),
                    gt_y=float(y_value),
                )
            )
    if targets:
        return targets

    series_names = list(_series_names(dataset))
    x_ticks = dataset.get("x_ticks") if isinstance(dataset.get("x_ticks"), list) else []
    for series_name in series_names:
        for x_label in x_ticks:
            targets.append(
                VBarTarget(
                    chart_id=dataset["chart_id"],
                    point_name=f"{series_name}, {x_label}",
                    series_name=series_name,
                    x_label=str(x_label),
                    gt_x=str(x_label),
                    gt_y=None,
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
    return ["Series 1"]


def image_path(dataset: dict[str, Any], image_type: str) -> Path:
    path = resolve_image_path(dataset, image_type)
    if path.exists() or Path(dataset["image_paths"][image_type]).is_absolute():
        return path
    chart_type = str(dataset.get("chart_type") or "v_bar")
    return ASSETS_ROOT / chart_type / dataset["image_paths"][image_type]
