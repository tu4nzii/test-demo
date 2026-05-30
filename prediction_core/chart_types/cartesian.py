"""Cartesian chart type specifications."""

from __future__ import annotations

from prediction_core.specs import PROJECT_ROOT, ChartSpec


CARTESIAN_SPECS: dict[str, ChartSpec] = {
    "v_bar": ChartSpec(
        chart_type="v_bar",
        coordinate_system="cartesian",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "v_bar" / "cli.py",
        sample_chart_id="v_bar_002",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "v_bar" / "chart_configs" / "v_bar" / "v_bar_002.json",
        trim_strategy="nested_series_first_point",
        model_line="latest 2026 vertical-bar flow with shared OpenAI-compatible config",
        note="Modular v_bar implementation using shared data/model/visual/evaluation modules.",
        workdir_override=PROJECT_ROOT,
    ),
    "h_bar": ChartSpec(
        chart_type="h_bar",
        coordinate_system="cartesian",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "h_bar" / "cli.py",
        sample_chart_id="h_bar_001",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "h_bar" / "chart_configs" / "h_bar_001.json",
        trim_strategy="nested_series_first_point",
        model_line="Pixtral flow with shared OpenAI-compatible config",
        note="Modular h_bar implementation using shared data/model/visual/evaluation modules.",
        workdir_override=PROJECT_ROOT,
    ),
    "scatter": ChartSpec(
        chart_type="scatter",
        coordinate_system="cartesian",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "scatter" / "cli.py",
        sample_chart_id="scatter_001",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "scatter" / "chart_configs" / "scatter_001.json",
        trim_strategy="flat_first_point",
        model_line="Pixtral2 scatter flow with shared OpenAI-compatible config",
        note="Modular scatter implementation with its own chart module.",
        workdir_override=PROJECT_ROOT,
    ),
    "bubble": ChartSpec(
        chart_type="bubble",
        coordinate_system="cartesian",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "bubble" / "cli.py",
        sample_chart_id="bubble_023",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "bubble" / "chart_configs" / "bubble_023.json",
        trim_strategy="flat_first_point",
        model_line="Pixtral bubble flow with shared OpenAI-compatible config",
        note="Modular bubble implementation with its own chart module.",
        workdir_override=PROJECT_ROOT,
    ),
    "line": ChartSpec(
        chart_type="line",
        coordinate_system="cartesian",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "line" / "cli.py",
        sample_chart_id="line_001",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "line" / "chart_configs" / "line_001.json",
        trim_strategy="nested_series_first_point",
        model_line="latest 2026 line flow, using shared OpenAI-compatible config",
        note="Modular line implementation using shared OpenAI-compatible config.",
        workdir_override=PROJECT_ROOT,
    ),
}
