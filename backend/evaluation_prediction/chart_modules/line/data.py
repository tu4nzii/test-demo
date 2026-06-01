"""Data loading and target enumeration for line charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ...common.backend_dataset import (
    load_backend_generated_datasets,
    resolve_image_path,
)
from ...common.chart_io import filter_chart_configs, load_json_configs
from ...common.paths import ASSETS_ROOT


LINE_ROOT = ASSETS_ROOT / "line"
CONFIG_DIR = LINE_ROOT / "chart_configs"


@dataclass(frozen=True)
class LineTarget:
    chart_id: str
    point_name: str
    series_name: str
    x_label: str
    gt_x: str
    gt_y: float | None


def load_datasets(
    chart_ids: Iterable[str] | None = None,
    config_paths: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if config_paths:
        return load_backend_generated_datasets(config_paths, "line", chart_ids)
    configs = load_json_configs(CONFIG_DIR, recursive=False, exclude_emu=True)
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[LineTarget]:
    targets: list[LineTarget] = []
    data_points = dataset.get("data_points") if isinstance(dataset.get("data_points"), dict) else {}
    for series_name, points in data_points.items():
        if not isinstance(points, dict):
            continue
        for x_label, y_value in points.items():
            targets.append(
                LineTarget(
                    chart_id=dataset["chart_id"],
                    point_name=f"{series_name}, {x_label}",
                    series_name=str(series_name),
                    x_label=str(x_label),
                    gt_x=str(x_label),
                    gt_y=_float_or_none(y_value),
                )
            )
    if targets:
        return targets

    x_ticks = dataset.get("x_ticks") if isinstance(dataset.get("x_ticks"), list) else []
    for series_name in _series_names(dataset):
        for x_label in x_ticks:
            targets.append(
                LineTarget(
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
    return ["series-0"]


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def image_path(dataset: dict[str, Any], image_type: str) -> Path:
    path = resolve_image_path(dataset, image_type)
    if path.exists() or Path(dataset["image_paths"][image_type]).is_absolute():
        return path
    return LINE_ROOT / dataset["image_paths"][image_type]
