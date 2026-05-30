"""Data loading and target enumeration for horizontal bar charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from prediction_core.chart_io import filter_chart_configs, load_json_configs
from prediction_core.specs import PROJECT_ROOT


H_BAR_ROOT = PROJECT_ROOT / "prediction_core" / "assets" / "h_bar"
CONFIG_DIR = H_BAR_ROOT / "chart_configs"


@dataclass(frozen=True)
class HBarTarget:
    chart_id: str
    point_name: str
    series_name: str
    y_label: str
    gt_x: float
    gt_y: str


def load_datasets(chart_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    configs = load_json_configs(CONFIG_DIR)
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[HBarTarget]:
    targets: list[HBarTarget] = []
    for series_name, sub_points in dataset["data_points"].items():
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
    return targets


def image_path(dataset: dict[str, Any], image_type: str) -> Path:
    return H_BAR_ROOT / dataset["image_paths"][image_type]
