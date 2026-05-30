"""Chart type specifications used by the modular runner and tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CoordinateSystem = Literal["cartesian", "polar"]
TrimStrategy = Literal[
    "nested_series_first_point",
    "flat_first_point",
    "rose_first_sector",
    "radar_first_cell",
]


@dataclass(frozen=True)
class ChartSpec:
    chart_type: str
    coordinate_system: CoordinateSystem
    script: Path
    sample_chart_id: str
    data_path: Path
    trim_strategy: TrimStrategy
    model_line: str
    note: str
    workdir_override: Path | None = None

    @property
    def workdir(self) -> Path:
        return self.workdir_override or self.script.parent

    @property
    def relative_script(self) -> Path:
        return self.script.relative_to(self.workdir)
