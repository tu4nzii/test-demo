"""Polar chart type specifications."""

from __future__ import annotations

from prediction_core.specs import PROJECT_ROOT, ChartSpec


POLAR_SPECS: dict[str, ChartSpec] = {
    "pie": ChartSpec(
        chart_type="pie",
        coordinate_system="polar",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "pie" / "cli.py",
        sample_chart_id="001",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "pie" / "chart_configs" / "1-10_batch_7" / "pie_001.json",
        trim_strategy="flat_first_point",
        model_line="Pixtral pie flow with shared OpenAI-compatible config",
        note="Pie implementation exposed through chart_modules/pie.",
        workdir_override=PROJECT_ROOT,
    ),
    "donut": ChartSpec(
        chart_type="donut",
        coordinate_system="polar",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "donut" / "cli.py",
        sample_chart_id="donut_135",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "donut" / "chart_configs" / "donut_135.json",
        trim_strategy="flat_first_point",
        model_line="Pixtral donut flow with shared OpenAI-compatible config",
        note="Donut implementation exposed through chart_modules/donut.",
        workdir_override=PROJECT_ROOT,
    ),
    "rose": ChartSpec(
        chart_type="rose",
        coordinate_system="polar",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "rose" / "cli.py",
        sample_chart_id="rose_004",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "rose" / "evaluation_datasets_with_axes_rose.json",
        trim_strategy="rose_first_sector",
        model_line="Pixtral rose flow with shared OpenAI-compatible config",
        note="Rose implementation exposed through chart_modules/rose.",
        workdir_override=PROJECT_ROOT,
    ),
    "radar": ChartSpec(
        chart_type="radar",
        coordinate_system="polar",
        script=PROJECT_ROOT / "prediction_core" / "chart_modules" / "radar" / "cli.py",
        sample_chart_id="radar_009",
        data_path=PROJECT_ROOT / "prediction_core" / "assets" / "radar" / "evaluation_datasets_with_axes_radar.json",
        trim_strategy="radar_first_cell",
        model_line="Pixtral radar flow with shared OpenAI-compatible config",
        note="Radar implementation exposed through chart_modules/radar.",
        workdir_override=PROJECT_ROOT,
    ),
}
