"""Data loading for pie charts."""

from __future__ import annotations

import os

from prediction_core.chart_io import load_json_configs


def load_chart_configs() -> list[dict]:
    def add_pie_subdir(config: dict, path, root: str) -> dict:
        subdir = os.path.relpath(path.parent, root)
        if subdir != ".":
            chart_id = config["chart_id"]
            config["image_paths"]["no_grid"] = f"charts/pie/{subdir}/pie_{chart_id}_no_grid.png"
            config["image_paths"]["with_grid"] = f"charts/pie/{subdir}/pie_{chart_id}_with_grid.png"
        return config

    return load_json_configs("chart_configs", recursive=True, transform=add_pie_subdir)
