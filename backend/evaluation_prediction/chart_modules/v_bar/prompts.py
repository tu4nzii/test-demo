"""Prompt builders for vertical bar value prediction."""

from __future__ import annotations

from typing import Any

from ..h_bar.prompts import generate_series_color_description


def split_item_name(item_name: str) -> tuple[str, str]:
    try:
        series_name, x_label = item_name.rsplit(",", 1)
    except ValueError:
        raise ValueError(
            f"item_name {item_name!r} must use the 'series_name, x_label' format for v_bar."
        ) from None
    return series_name.strip(), x_label.strip()


def _format_ticks(ticks: list[Any]) -> str:
    return ", ".join(str(item) for item in ticks)


def _format_visible_ticks(visible_ticks: list[Any] | None) -> str:
    visible_ticks = visible_ticks or []
    return ", ".join(str(round(float(y), 2)) for y in sorted(set(visible_ticks)))


def build_color_prompt(point_name: str, series_color: dict[str, str]) -> str:
    series_name, x_label = split_item_name(point_name)
    color_desc = generate_series_color_description(series_color)
    return (
        f"You are given a cropped vertical bar chart image for {point_name}.\n"
        f"**{color_desc}**.\n"
        f"Please check whether the cropped image contains the target vertical bar segment "
        f"for series \"{series_name}\" at x-axis category \"{x_label}\", and whether its "
        "top boundary/end edge is visible inside the crop. Do not return true if only the "
        "middle of the bar is visible but the top boundary is outside the crop.\n"
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )


def generate_prompt(
    item_name: str,
    prompt_type: str,
    x_ticks: list,
    y_ticks: list,
    series_color: dict[str, str],
    visible_ticks: list | None = None,
    pred_feedback: list | None = None,
    feedback_round: int = 0,
    current_round: int = 1,
) -> str:
    series_name, x_label = split_item_name(item_name)
    x_tick_str = _format_ticks(x_ticks)
    y_tick_str = _format_ticks(y_ticks)
    color_desc = generate_series_color_description(series_color)

    if prompt_type == "baseline":
        base_prompt = (
            "You are given a vertical bar chart image. "
            f"{color_desc}\n"
            f"Your task is to predict the y value, or height, of the bar labeled [{item_name}].\n"
            f"Locate the bar for series [{series_name}] at x-axis category [{x_label}], then estimate its top edge value."
        )
    elif prompt_type == "amplifier":
        visible_tick_str = _format_visible_ticks(visible_ticks)
        base_prompt = f"""
        You are given a cropped vertical bar chart image. Your task is to predict the y value for [{item_name}].
        The cropped image is centered around the x-axis category group **"{x_label}"**. In grouped or stacked bar charts, the target colored bar may appear left or right of the horizontal center within that category group; use the series color to select the correct bar.
        The left and right sides include a vertically drawn y-axis scale with tick values [{visible_tick_str}] and grid lines.
        Estimate the exact y value corresponding to the **top edge** of the colored bar.
        Use the color/legend alignment to verify the target series. {color_desc}
        Do not snap to the nearest tick; interpolate proportionally between adjacent grid lines.
        """
    else:
        base_prompt = f"""
        You are analyzing a vertical bar chart that contains reference grid lines.
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        {color_desc}
        Locate the bar for series [{series_name}] at x-axis category [{x_label}].
        Identify its top edge and estimate the corresponding y value by interpolating between the nearest horizontal grid lines.
        """

    if prompt_type == "feedback" and pred_feedback and current_round >= feedback_round:
        pred = pred_feedback[-1]
        base_prompt = f"""
        You are analyzing a vertical bar chart with reference grid lines.
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        {color_desc}

        The previous prediction for [{item_name}] is marked on the chart at x = "{pred[0]}", y = {pred[1]}.
        Compare that mark with the true top edge of the target bar, then refine the y value.
        """

    base_prompt += f"""
    Only respond in this JSON format:
    {{"datapoints": [{{"{item_name}": ["{x_label}", y]}}]}}
    """
    return base_prompt
