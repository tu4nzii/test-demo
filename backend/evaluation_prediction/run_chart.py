"""Unified CLI for backend-local chart value prediction."""

from __future__ import annotations

import argparse
import asyncio


async def _run(
    chart_type: str,
    batch_size: int | None,
    chart_ids: list[str] | None,
    config_paths: list[str] | None,
) -> None:
    if chart_type == "v_stacked_bar":
        chart_type = "v_bar"
    elif chart_type == "h_stacked_bar":
        chart_type = "h_bar"

    if chart_type == "v_bar":
        from .chart_modules.v_bar.runner import run_experiment

        await run_experiment(batch_size=batch_size, chart_ids=chart_ids, config_paths=config_paths, chart_type=chart_type)
        return
    elif chart_type == "h_bar":
        from .chart_modules.h_bar.runner import run_experiment

        await run_experiment(batch_size=batch_size, chart_ids=chart_ids, config_paths=config_paths, chart_type=chart_type)
        return
    elif chart_type == "line":
        from .chart_modules.line.runner import run_experiment
    elif chart_type == "scatter":
        from .chart_modules.scatter.data import PointChartConfig
        from .chart_modules.scatter.runner import run_experiment

        await run_experiment(
            PointChartConfig(chart_type="scatter", result_dir_name="scatter", mark_name="circle"),
            batch_size=batch_size,
            chart_ids=chart_ids,
            config_paths=config_paths,
        )
        return
    elif chart_type == "bubble":
        from .chart_modules.bubble.data import PointChartConfig
        from .chart_modules.bubble.runner import run_experiment

        await run_experiment(
            PointChartConfig(chart_type="bubble", result_dir_name="bubble", mark_name="bubble"),
            batch_size=batch_size,
            chart_ids=chart_ids,
            config_paths=config_paths,
        )
        return
    elif chart_type == "pie":
        from .chart_modules.pie.runner import run_experiment
    elif chart_type == "donut":
        from .chart_modules.donut.runner import run_experiment
    elif chart_type == "radar":
        from .chart_modules.radar.runner import run_experiment
    elif chart_type == "rose":
        from .chart_modules.rose.runner import run_experiment
    else:
        raise ValueError(f"Unsupported chart_type: {chart_type}")

    await run_experiment(batch_size=batch_size, chart_ids=chart_ids, config_paths=config_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backend-local chart value prediction.")
    parser.add_argument(
        "chart_type",
        choices=[
            "v_bar",
            "h_bar",
            "line",
            "scatter",
            "bubble",
            "pie",
            "donut",
            "radar",
            "rose",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    parser.add_argument(
        "--config-json",
        nargs="+",
        default=None,
        help="Backend-generated *_image.json or *_image_ticks.json files to use as prediction input.",
    )
    args = parser.parse_args()
    asyncio.run(_run(args.chart_type, args.batch_size, args.chart_ids, args.config_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
