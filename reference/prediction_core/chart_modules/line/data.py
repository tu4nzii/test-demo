"""Data loading and target enumeration for line charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from reference.prediction_core.chart_io import filter_chart_configs, load_json_configs
from reference.prediction_core.specs import PROJECT_ROOT


LINE_ROOT = PROJECT_ROOT / "prediction_core" / "assets" / "line"
CONFIG_DIR = LINE_ROOT / "chart_configs"


@dataclass(frozen=True)
class LineTarget:
    chart_id: str
    point_name: str
    series_name: str
    x_label: str
    gt_x: str
    gt_y: float


def load_datasets(chart_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    configs = load_json_configs(CONFIG_DIR, recursive=False, exclude_emu=True)
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[LineTarget]:
    targets: list[LineTarget] = []
    for series_name, points in dataset["data_points"].items():
        for x_label, y_value in points.items():
            targets.append(
                LineTarget(
                    chart_id=dataset["chart_id"],
                    point_name=f"{series_name}, {x_label}",
                    series_name=series_name,
                    x_label=str(x_label),
                    gt_x=str(x_label),
                    gt_y=float(y_value),
                )
            )
    return targets


def image_path(dataset: dict[str, Any], image_type: str) -> Path:
    return LINE_ROOT / dataset["image_paths"][image_type]
