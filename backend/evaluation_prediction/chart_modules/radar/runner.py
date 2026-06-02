"""Runner for backend-local radar value extraction."""

from __future__ import annotations

from pathlib import Path

from ..polar_value import run_polar_experiment


LEGACY_URLS = ["http://localhost:8508/v1/chat/completions"]


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
) -> list[dict]:
    return await run_polar_experiment(
        "radar",
        batch_size=batch_size,
        chart_ids=chart_ids,
        config_paths=config_paths,
        legacy_urls=LEGACY_URLS,
    )
