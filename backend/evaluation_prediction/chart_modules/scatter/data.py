"""Data loading for scatter-like point charts.

Scatter and bubble share the same point-target extraction logic: point labels
come from generated point objects or plot-area OCR labels, while legend/category
names are kept as optional metadata rather than target names.
"""

from __future__ import annotations

from ..bubble.data import PointChartConfig, PointTarget, image_path, iter_targets, load_datasets

__all__ = [
    "PointChartConfig",
    "PointTarget",
    "image_path",
    "iter_targets",
    "load_datasets",
]
