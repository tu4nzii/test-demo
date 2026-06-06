"""Prompt builders for horizontal bar value prediction."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_color_name_approx(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    min_dist = float("inf")
    closest_name = "unknown"
    for name, hex_code in mcolors.CSS4_COLORS.items():
        r2, g2, b2 = hex_to_rgb(hex_code)
        dist = (r - r2) ** 2 + (g - g2) ** 2 + (b - b2) ** 2
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def generate_series_color_description(series_color: dict[str, str]) -> str:
    lines = ["The chart uses specific colors for each series:"]
    for series, hex_val in series_color.items():
        rgb = hex_to_rgb(hex_val)
        color_name = get_color_name_approx(hex_val)
        lines.append(f'- "{series}" is colored **{hex_val}** (approx. {color_name}, RGB: {rgb}).')
    return "\n".join(lines)


def build_color_prompt(point_name: str, series_color: dict[str, str]) -> str:
    try:
        series_name, y_label = split_item_name(point_name)
    except Exception:
        series_name = point_name.split(",", 1)[0].strip()
        y_label = point_name.rsplit(",", 1)[-1].strip()
    color_desc = generate_series_color_description(series_color)
    return (
        f"You are given a cropped bar chart image for {point_name}.\n"
        f"**{color_desc}**.\n"
        f"Please check whether the cropped image contains the target horizontal bar segment "
        f"for series \"{series_name}\" at y-axis category \"{y_label}\", and whether its "
        "right boundary/end edge is visible inside the crop. Do not return true if only the "
        "middle of the bar is visible but the right boundary is outside the crop.\n"
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )


def _format_ticks(ticks: list[Any]) -> str:
    return ", ".join(str(item) for item in ticks)


def _format_visible_ticks(visible_ticks: list[Any] | None) -> str:
    visible_ticks = visible_ticks or []
    if visible_ticks and isinstance(visible_ticks[0], list):
        visible_ticks = visible_ticks[0]
    return ", ".join(str(round(float(x), 2)) for x in sorted(set(visible_ticks)))


def split_item_name(item_name: str) -> tuple[str, str]:
    try:
        series_name, y_label = item_name.rsplit(",", 1)
    except ValueError:
        raise ValueError(
            f"item_name {item_name!r} must use the 'series_name, y_label' format for h_bar."
        ) from None
    return series_name.strip(), y_label.strip()


def generate_prompt(
    item_name: str,
    prompt_type: str,
    x_ticks: list,
    y_ticks: list,
    series_color: dict[str, str],
    visible_ticks: list | None = None,
    axis_types: dict | None = None,
    pred_feedback: list | None = None,
    feedback_round: int = 0,
    current_round: int = 1,
) -> str:
    _, y_label = split_item_name(item_name)
    x_tick_str = _format_ticks(x_ticks)
    y_tick_str = _format_ticks(y_ticks)
    color_desc = generate_series_color_description(series_color)

    if prompt_type == "baseline":
        base_prompt = (
            "You are given a bar chart image. "
            f"{color_desc}\n"
            f"Your task is to predict the x coordinate for the segment labeled [{item_name}].\n"
            f"To identify the x coordinate, first locate the tick interval in which the right boundary "
            f"of the segment representing [{item_name}] falls."
        )
    elif prompt_type == "amplifier":
        visible_tick_str = _format_visible_ticks(visible_ticks)
        base_prompt = f"""
        You are given a chart image. Your task is to predict the x coordinate for the segment labeled [{item_name}].
        The cropped image is centered on the y-axis category group **"{y_label}"**. In grouped or stacked bar charts, the target colored segment may appear above or below the vertical center within that category group; use the series color to select the correct segment.
        The top and bottom sides include a **horizontally drawn x-axis**, with tick values [{visible_tick_str}] and grid lines.
        Your task is to estimate the **x coordinate** corresponding to the **right boundary** of the colored segment.
        The segment color indicates its series: use alignment between the legend and segment to verify the target. {color_desc}
        Instructions:
            - First, locate the x-axis tick interval in which the segment's right boundary falls.
            - Then, determine the relative position of the boundary within this interval. Use linear interpolation between the two tick values to estimate the precise x-axis value.
            - **Important:** Do not snap or round to the nearest tick; interpolate proportionally.
            - **Edge case:** If the segment cannot be visually detected even near the **minimum tick boundary**, output the **minimum tick value (e.g., 0)** as the coordinate.
        """
    else:
        base_prompt = f"""
        You are analyzing a bar chart that contains **reference grid lines**, where horizontal lines correspond to y-axis ticks, and vertical lines align with x-axis ticks.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        {color_desc}
        After locating the correct segment for [{item_name}], identify the position of its right edge by comparing it with the two nearest vertical grid lines on the X-axis.
        Use linear interpolation between these two ticks to estimate the accurate X-coordinate of [{item_name}].
        """

    if prompt_type == "feedback" and pred_feedback and current_round >= feedback_round:
        pred = pred_feedback[-1]
        x = f"{pred[0]:.2f}" if isinstance(pred[0], (int, float)) else f'"{pred[0]}"'
        y = f"{pred[1]:.2f}" if isinstance(pred[1], (int, float)) else f'"{pred[1]}"'
        base_prompt = f"""
        You are analyzing a bar chart with reference grid lines.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        {color_desc}

        The given chart shows your previous prediction for the x coordinate of [{item_name}], aligned with a red crosshair at (x = {x}, y = {y}).
        Compare this red crosshair to the true right boundary of [{item_name}]: determine whether it is too far left, too far right, or aligned correctly.
        After verifying, adjust the prediction proportionally to refine your estimate to get the most accurate result x coordinate of [{item_name}].
        """

    base_prompt += f"""
    Only respond in this JSON format:
    {{"datapoints": [{{"{item_name}": [x, "{y_label}"]}}]}}
    """
    return base_prompt
