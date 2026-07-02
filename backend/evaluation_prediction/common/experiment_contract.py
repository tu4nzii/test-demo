"""Static experiment contract for chart-type runners.

This module documents the required mechanisms in executable form so API code,
tests, and documentation can check the same source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from .experiment_flow import AMPLIFIER_STAGE, BASELINE_STAGE, FEEDBACK_STAGE, GRID_STAGE


@dataclass(frozen=True)
class ChartExperimentContract:
    chart_type: str
    runner_module: str
    prompt_module: str
    visual_module: str | None
    model_module: str | None
    stages: tuple[str, ...] = (BASELINE_STAGE, GRID_STAGE, FEEDBACK_STAGE, AMPLIFIER_STAGE)
    notes: str = ""


CONTRACTS: dict[str, ChartExperimentContract] = {
    "scatter": ChartExperimentContract(
        chart_type="scatter",
        runner_module="chart_modules.scatter.runner",
        prompt_module="chart_modules.scatter.prompts",
        visual_module="chart_modules.scatter.visual",
        model_module="chart_modules.scatter.model",
    ),
    "bubble": ChartExperimentContract(
        chart_type="bubble",
        runner_module="chart_modules.bubble.runner",
        prompt_module="chart_modules.bubble.prompts",
        visual_module="chart_modules.bubble.visual",
        model_module="chart_modules.bubble.model",
    ),
    "line": ChartExperimentContract(
        chart_type="line",
        runner_module="chart_modules.line.runner",
        prompt_module="chart_modules.line.prompts",
        visual_module="chart_modules.line.visual",
        model_module="chart_modules.line.model",
    ),
    "v_bar": ChartExperimentContract(
        chart_type="v_bar",
        runner_module="chart_modules.v_bar.runner",
        prompt_module="chart_modules.v_bar.prompts",
        visual_module="chart_modules.v_bar.visual",
        model_module="chart_modules.v_bar.model",
    ),
    "h_bar": ChartExperimentContract(
        chart_type="h_bar",
        runner_module="chart_modules.h_bar.runner",
        prompt_module="chart_modules.h_bar.prompts",
        visual_module="chart_modules.h_bar.visual",
        model_module="chart_modules.h_bar.model",
    ),
    "pie": ChartExperimentContract(
        chart_type="pie",
        runner_module="chart_modules.pie.runner",
        prompt_module="chart_modules.pie.prompts",
        visual_module="chart_modules.pie.visual",
        model_module="chart_modules.pie.model",
    ),
    "donut": ChartExperimentContract(
        chart_type="donut",
        runner_module="chart_modules.donut.runner",
        prompt_module="chart_modules.donut.prompts",
        visual_module="chart_modules.donut.visual",
        model_module="chart_modules.donut.model",
    ),
    "radar": ChartExperimentContract(
        chart_type="radar",
        runner_module="chart_modules.radar.runner",
        prompt_module="chart_modules.polar_value",
        visual_module="chart_modules.polar_visual",
        model_module="chart_modules.polar_value",
    ),
    "rose": ChartExperimentContract(
        chart_type="rose",
        runner_module="chart_modules.rose.runner",
        prompt_module="chart_modules.polar_value",
        visual_module="chart_modules.polar_visual",
        model_module="chart_modules.polar_value",
    ),
}


def missing_contract_stages(chart_type: str, observed_stages: set[str]) -> list[str]:
    contract = CONTRACTS[chart_type]
    return [stage for stage in contract.stages if stage not in observed_stages]
