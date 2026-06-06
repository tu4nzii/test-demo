"""Runner for radar chart prediction."""

from __future__ import annotations

from .amplifier import async_crop_and_find
from .flow import main, process_single_task


async def run_experiment(batch_size: int | None = None, chart_ids: list[str] | None = None) -> None:
    await main(chart_ids=chart_ids)


__all__ = ["async_crop_and_find", "main", "process_single_task", "run_experiment"]
