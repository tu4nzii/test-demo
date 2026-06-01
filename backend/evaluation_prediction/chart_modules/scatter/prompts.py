"""Prompt builders for scatter charts."""

from __future__ import annotations

from typing import Any


def _format_ticks(ticks: list[Any]) -> str:
    return ", ".join(str(item) for item in ticks)


def _mark_phrase(mark_name: str) -> str:
    return "bubble" if mark_name == "bubble" else "circle"


def build_point_exists_prompt(point_name: str, mark_name: str, visual_name: str | None = None) -> str:
    mark = _mark_phrase(mark_name)
    target_text = f"[{point_name}]"
    if visual_name and visual_name != point_name:
        target_text += f", shown in the legend as [{visual_name}]"
    return (
        f"You are given a cropped chart image around the target {mark} {target_text}. "
        f"Please check if the {mark} corresponding to {target_text} is visible in this cropped region. "
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )


def generate_prompt(
    *,
    item_name: str,
    prompt_type: str,
    x_ticks: list[Any],
    y_ticks: list[Any],
    mark_name: str,
    visual_name: str | None = None,
    pred_feedback: tuple[Any, Any] | None = None,
) -> str:
    x_tick_str = _format_ticks(x_ticks)
    y_tick_str = _format_ticks(y_ticks)
    mark = _mark_phrase(mark_name)
    visual_hint = ""
    if visual_name and visual_name != item_name:
        visual_hint = f"The target id [{item_name}] is shown in the chart legend/label as [{visual_name}]."

    if prompt_type == "baseline":
        base_prompt = f"""
        You are given a chart image.
        Please extract the coordinates of the {mark} which represents [{item_name}].
        {visual_hint}
        Identify the graphical element that represents the target data item and extract its coordinates based on its visual center, not the location of any associated label or annotation.
        Return data coordinates on the chart axes, not pixel coordinates.
        """
    elif prompt_type in {"grid", "feedback", "feedback_crop_adaptive"}:
        base_prompt = f"""
        You are analyzing a chart that includes reference grid lines aligned with these axis ticks:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]

        Locate the visual center of the {mark} representing [{item_name}].
        {visual_hint}
        Estimate its (x, y) data coordinates by interpolating between adjacent grid lines.
        """
        if prompt_type == "feedback" and pred_feedback is not None:
            base_prompt += f"""
            A previous prediction is marked on the chart at approximately
            x = {pred_feedback[0]}, y = {pred_feedback[1]}.
            Compare that marker with the true center of [{item_name}] and refine the coordinates.
            """
        if prompt_type == "feedback_crop_adaptive":
            base_prompt += """
            This image is a cropped region around the target. Use the visible tick labels and grid lines in the crop.
            """
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    base_prompt += f"""
    Only respond in this JSON format:
    {{"datapoints": [{{"{item_name}": [x, y]}}]}}
    """
    return base_prompt
