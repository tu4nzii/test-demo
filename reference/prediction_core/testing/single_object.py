"""Temporary single-object dataset trimming for end-to-end tests."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reference.prediction_core.chart_io import read_json, write_json
from reference.prediction_core.specs import ChartSpec


@dataclass(frozen=True)
class TrimResult:
    chart_type: str
    chart_id: str
    object_name: str
    data_path: Path


class TemporarySingleObjectDataset:
    """Backup a dataset, trim one chart to one object, then restore it."""

    def __init__(self, spec: ChartSpec):
        self.spec = spec
        self.backup_path = spec.data_path.with_name(spec.data_path.name + ".bak_test")
        self.result: TrimResult | None = None

    def __enter__(self) -> TrimResult:
        shutil.copy2(self.spec.data_path, self.backup_path)
        data = read_json(self.spec.data_path)
        trimmed, object_name = trim_data(data, self.spec)
        write_json(self.spec.data_path, trimmed)
        self.result = TrimResult(
            chart_type=self.spec.chart_type,
            chart_id=self.spec.sample_chart_id,
            object_name=object_name,
            data_path=self.spec.data_path,
        )
        return self.result

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.backup_path.exists():
            shutil.move(str(self.backup_path), str(self.spec.data_path))


def trim_data(data: Any, spec: ChartSpec) -> tuple[Any, str]:
    if spec.trim_strategy == "nested_series_first_point":
        return _trim_nested_series_first_point(data, spec)
    if spec.trim_strategy == "flat_first_point":
        return _trim_flat_first_point(data)
    if spec.trim_strategy == "rose_first_sector":
        return _trim_rose_first_sector(data, spec.sample_chart_id)
    if spec.trim_strategy == "radar_first_cell":
        return _trim_radar_first_cell(data, spec.sample_chart_id)
    raise ValueError(f"Unsupported trim strategy: {spec.trim_strategy}")


def _trim_nested_series_first_point(config: dict[str, Any], spec: ChartSpec) -> tuple[dict[str, Any], str]:
    series = next(iter(config["data_points"]))
    points = config["data_points"][series]
    point = next(iter(points))
    config["data_points"] = {series: {point: points[point]}}

    if "series_color" in config and series in config["series_color"]:
        config["series_color"] = {series: config["series_color"][series]}

    axis_key = "y" if spec.chart_type == "h_bar" else "x"
    ticks_key = f"{axis_key}_ticks"
    pixels_key = f"{axis_key}_pixels"
    if ticks_key in config and pixels_key in config and point in config[ticks_key]:
        idx = config[ticks_key].index(point)
        config[ticks_key] = [config[ticks_key][idx]]
        config[pixels_key] = [config[pixels_key][idx]]

    return config, f"{series} / {point}"


def _trim_flat_first_point(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    point = next(iter(config["data_points"]))
    for key in ("data_points", "data_points1", "series_color"):
        if key in config and isinstance(config[key], dict) and point in config[key]:
            config[key] = {point: config[key][point]}
    return config, point


def _find_chart(items: list[dict[str, Any]], chart_id: str) -> dict[str, Any]:
    for item in items:
        if item.get("chart_id") == chart_id:
            return item
    raise ValueError(f"Chart id not found in dataset: {chart_id}")


def _trim_rose_first_sector(items: list[dict[str, Any]], chart_id: str) -> tuple[list[dict[str, Any]], str]:
    chart = _find_chart(items, chart_id)
    point = next(iter(chart["data"]))
    for key in ("data", "data_points", "series_color"):
        if key in chart:
            chart[key] = {point: chart[key][point]}
    return items, point


def _trim_radar_first_cell(items: list[dict[str, Any]], chart_id: str) -> tuple[list[dict[str, Any]], str]:
    chart = _find_chart(items, chart_id)
    series = next(iter(chart["data"]))
    axis = next(iter(chart["data"][series]))
    for key in ("data", "data_points"):
        if key in chart:
            chart[key] = {series: {axis: chart[key][series][axis]}}
    if "series_color" in chart and series in chart["series_color"]:
        chart["series_color"] = {series: chart["series_color"][series]}
    chart["num_entities"] = 1
    return items, f"{series},{axis}"
