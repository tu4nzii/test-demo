"""Prompt builders for scatter charts."""

from __future__ import annotations

from typing import Any


def _format_ticks(ticks: list[Any]) -> str:
    return ", ".join(str(item) for item in ticks)


def _format_tick_pixel_pairs(ticks: list[Any], pixels: list[Any] | None) -> str:
    if not pixels:
        return ""
    pairs = [f"{tick}->{pixel}px" for tick, pixel in zip(ticks, pixels)]
    return ", ".join(pairs)


def _mark_phrase(mark_name: str) -> str:
    return "bubble" if mark_name == "bubble" else "circle"


def _mark_reading_guidance(mark_name: str, item_name: str) -> str:
    if mark_name == "bubble":
        return (
            f"For bubble charts, the target [{item_name}] is the bubble body associated with that label or legend "
            "identity. Read the center of that bubble body. If another smaller mark overlaps inside or near the "
            "bubble, ignore it unless it is explicitly the target identity."
        )
    return (
        f"For scatter charts, read the center of the plotted circle representing [{item_name}], "
        "not nearby text or another point."
    )


def build_point_exists_prompt(point_name: str, mark_name: str, visual_name: str | None = None) -> str:
    mark = _mark_phrase(mark_name)
    mark_guidance = _mark_reading_guidance(mark_name, item_name)
    target_text = f"[{point_name}]"
    if visual_name and visual_name != point_name:
        target_text += f", shown in the legend as [{visual_name}]"
    return (
        f"You are given a cropped scatter/bubble chart image. Determine whether the plotted {mark} "
        f"corresponding specifically to {target_text} is visible in this crop. "
        f"Return true only if the target label/legend identity {target_text} is visible, or the visible mark can be "
        f"unambiguously matched to that identity. If the crop only shows other labels, partial unrelated marks, "
        f"axis ticks, grid labels, titles, or annotations, return false. "
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )


def generate_prompt(
    *,
    item_name: str,
    prompt_type: str,
    x_ticks: list[Any],
    y_ticks: list[Any],
    mark_name: str,
    x_pixels: list[Any] | None = None,
    y_pixels: list[Any] | None = None,
    visual_name: str | None = None,
    pred_feedback: tuple[Any, Any] | None = None,
    estimated_mark_diameter_px: float | None = None,
    mark_size_source: str | None = None,
) -> str:
    x_tick_str = _format_ticks(x_ticks)
    y_tick_str = _format_ticks(y_ticks)
    x_pixel_str = _format_tick_pixel_pairs(x_ticks, x_pixels)
    y_pixel_str = _format_tick_pixel_pairs(y_ticks, y_pixels)
    mark = _mark_phrase(mark_name)
    mark_guidance = _mark_reading_guidance(mark_name, item_name)
    json_contract = (
        f'CRITICAL OUTPUT RULE: Return only valid JSON and nothing else. '
        f'Do not write explanations, Markdown fences, or prose. '
        f'The complete response must be exactly one JSON object like this: '
        f'{{"readable": true, "datapoints": [{{"{item_name}": [x, y]}}]}}. '
        f'Use numeric x and y values.'
    )
    visual_hint = ""
    if visual_name and visual_name != item_name:
        visual_hint = f"The target id [{item_name}] is shown in the chart legend/label as [{visual_name}]."

    if prompt_type == "baseline":
        base_prompt = f"""
        {json_contract}

        You are given a chart image.
        Please extract the coordinates of the {mark} which represents [{item_name}].
        {visual_hint}
        {mark_guidance}
        Identify the graphical element that represents the target data item and extract its coordinates based on its visual center, not the location of any associated label or annotation.
        Return data coordinates on the chart axes, not pixel coordinates.
        """
    elif prompt_type in {"grid", "feedback", "feedback_crop_adaptive"}:
        base_prompt = f"""
        {json_contract}

        You are analyzing a chart that includes reference grid lines aligned with these axis ticks:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]

        Locate the visual center of the {mark} representing [{item_name}].
        {visual_hint}
        {mark_guidance}
        Estimate its (x, y) data coordinates by interpolating between adjacent grid lines.
        Read the center of the plotted {mark}, not the text label attached to it.
        """
        if prompt_type == "feedback" and pred_feedback is not None:
            base_prompt += f"""
            A previous prediction is marked on the chart at approximately
            x = {pred_feedback[0]}, y = {pred_feedback[1]}.
            Compare that marker with the true center of [{item_name}] and refine the coordinates.
            """
        if prompt_type == "feedback_crop_adaptive":
            size_hint = ""
            if estimated_mark_diameter_px is not None:
                size_hint = (
                    f"The target {mark} is estimated from the original image to be about "
                    f"{round(float(estimated_mark_diameter_px), 1)} px in diameter "
                    f"({mark_size_source or 'image estimate'}). Use this only to identify the correct visible mark "
                    "and judge readability; do not output the diameter."
                )
            base_prompt += f"""
            This image is a cropped region around the target. Use only the visible tick labels and grid lines
            inside this crop. The crop may include one or more nearby labels, so first identify the mark whose
            visible label/legend identity is [{item_name}], then interpolate the center of that mark.
            {size_hint}
            The red local grid labels are data-coordinate values from the original chart. Use the two nearest
            red vertical grid lines to interpolate x, and the two nearest red horizontal grid lines to interpolate y.
            The visual center of the plotted {mark} is the value to read; do not use the label position,
            the crop center, or any resized-image pixel coordinate.
            If the target mark is not visible or cannot be confidently identified in this crop, return
            {{"readable": false, "datapoints": [{{"{item_name}": [null, null]}}]}} using the exact same JSON shape.
            Return the original chart data coordinates indicated by the red tick labels and grid lines.
            Do not return crop pixel coordinates, resized-image pixel coordinates, or relative positions inside the crop.
            """
        else:
            base_prompt += """
            If uncertain, make the best estimate from the full chart and grid.
            """
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    return base_prompt
