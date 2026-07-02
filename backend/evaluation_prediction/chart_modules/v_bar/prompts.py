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


def _format_tick_pixel_pairs(ticks: list[Any], pixels: list[Any] | None) -> str:
    if not pixels:
        return ""
    return ", ".join(f"{tick}->{pixel}px" for tick, pixel in zip(ticks, pixels))


def _matched_tick_pixel(label: str, ticks: list[Any], pixels: list[Any] | None) -> Any:
    if not pixels:
        return None
    texts = [str(item).strip() for item in ticks]
    label_text = str(label).strip()
    if label_text in texts:
        return pixels[texts.index(label_text)]
    return None


def _format_visible_ticks(visible_ticks: list[Any] | None) -> str:
    visible_ticks = visible_ticks or []
    return ", ".join(str(round(float(y), 2)) for y in sorted(set(visible_ticks)))


def build_color_prompt(point_name: str, series_color: dict[str, str], chart_type: str = "v_bar") -> str:
    series_name, x_label = split_item_name(point_name)
    color_desc = generate_series_color_description(series_color)
    if str(chart_type or "").lower() == "v_stacked_bar":
        return (
            f"You are given a cropped vertical stacked bar chart image for {point_name}.\n"
            f"**{color_desc}**.\n"
            f"Please check whether the cropped image contains the target colored stack segment "
            f"for series \"{series_name}\" at x-axis category \"{x_label}\", and whether both "
            "its bottom and top boundaries are visible inside the crop. Do not return true if "
            "only the middle of the segment is visible.\n"
            "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
        )
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
    x_pixels: list | None = None,
    y_pixels: list | None = None,
    visible_ticks: list | None = None,
    pred_feedback: list | None = None,
    feedback_round: int = 0,
    current_round: int = 1,
    chart_type: str = "v_bar",
) -> str:
    series_name, x_label = split_item_name(item_name)
    x_tick_str = _format_ticks(x_ticks)
    y_tick_str = _format_ticks(y_ticks)
    x_pixel_str = _format_tick_pixel_pairs(x_ticks, x_pixels)
    y_pixel_str = _format_tick_pixel_pairs(y_ticks, y_pixels)
    color_desc = generate_series_color_description(series_color)
    series_names = [str(name) for name in series_color]
    series_order_desc = (
        "Within each grouped x-axis category, the visible bars follow this series order from left to right: "
        + ", ".join(f"[{name}]" for name in series_names)
        + "."
        if len(series_names) > 1
        else ""
    )
    target_series_position = ""
    if series_name in series_names and len(series_names) > 1:
        target_series_position = (
            f'The target series "{series_name}" is bar #{series_names.index(series_name) + 1} '
            f"from the left within category group \"{x_label}\"."
        )
    target_x_pixel = _matched_tick_pixel(x_label, x_ticks, x_pixels)
    target_x_hint = (
        f'The x-axis tick "{x_label}" is centered at approximately {target_x_pixel}px in the original image.'
        if target_x_pixel is not None
        else ""
    )
    is_stacked = str(chart_type or "").lower() == "v_stacked_bar"
    data_value_rule = f"""
        Response rule:
            - Return only valid JSON. Do not write reasoning, equations, Markdown, or explanatory text.
            - The JSON key must be exactly the requested target item. Do not return any nearby category, legend item, or other series.
        Output rule:
            - Return the y-axis DATA VALUE, never a pixel coordinate.
            - Use tick labels and tick-to-pixel pairs only for reasoning.
            - Never return a number from the pixel mapping list as the answer. Values such as [{y_pixel_str}] are image pixel positions, not data values.
            - Numeric labels drawn inside a colored bar are grid labels, not the bar's value. Ignore those labels as answers and read the bar's top edge.
            - The top side of the chart may crop or hide the largest tick labels. Still use the provided Y-axis tick list and GT Y tick-to-pixel mapping: a bar ending near the top plot boundary may correspond to a high value close to the maximum y tick, not merely the last clearly printed tick label.
            - For very tall bars, inspect the highest colored-background boundary and interpolate against the full axis range, including ticks that are listed in text even if their label is partially clipped in the image.
        """

    if prompt_type == "baseline":
        if is_stacked:
            base_prompt = (
                "You are given a vertical stacked bar chart image. "
                f"{color_desc}\n"
                f"Your task is to predict the y-axis value represented by the colored stack segment labeled [{item_name}].\n"
                f"Locate the segment for series [{series_name}] at x-axis category [{x_label}], estimate its bottom and top boundary values, then subtract bottom from top."
            )
        else:
            base_prompt = (
                "You are given a vertical bar chart image. "
                f"{color_desc}\n"
                f"Your task is to predict the y value, or height, of the bar labeled [{item_name}].\n"
                f"Locate the bar for series [{series_name}] at x-axis category [{x_label}], then estimate its top edge value."
            )
    elif prompt_type == "amplifier":
        visible_tick_str = _format_visible_ticks(visible_ticks)
        if is_stacked:
            base_prompt = f"""
        You are given a cropped vertical stacked bar chart image. Your task is to predict the y-axis value represented by [{item_name}].
        The cropped image is centered around the x-axis category group **"{x_label}"**; the target stacked segment may start above zero.
        The left and right sides include a vertically drawn y-axis scale with tick values [{visible_tick_str}] and grid lines.
        Use the color/legend alignment to verify the target series. {color_desc}
        If the target colored segment is not visible or cannot be confidently identified in this crop, return the same JSON shape with `"readable": false` and use `null` for y.
        Estimate the target segment value as top boundary value minus bottom boundary value.
        Do not return the cumulative top-edge value unless the segment starts at zero.
        """
        else:
            base_prompt = f"""
        You are given a cropped vertical bar chart image. Your task is to predict the y value for [{item_name}].
        The cropped image is centered around the x-axis category group **"{x_label}"**. In grouped or stacked bar charts, the target colored bar may appear left or right of the horizontal center within that category group; use the series color to select the correct bar.
        {series_order_desc}
        {target_series_position}
        {target_x_hint}
        The x-axis category label **"{x_label}"** identifies the category group center, not a y-axis value. Do not read a neighboring category group even if it has a taller or clearer bar.
        The crop includes a vertically drawn y-axis ruler. The left ruler has tick labels [{visible_tick_str}], and the red horizontal dashed lines are the corresponding grid lines.
        Estimate the exact y value corresponding to the **top edge** of the colored bar.
        Confirm that the visible bar belongs to the exact x-axis category **"{x_label}"** and series [{series_name}]. If the exact target category/series is not visible, return `"readable": false`.
        Use the color/legend alignment to verify the target series. {color_desc}
        {data_value_rule}
        If the target bar is not visible or cannot be confidently identified in this crop, return the same JSON shape with `"readable": false` and use `null` for y.
        In the zoom-in crop, the red horizontal guide labels are original chart y-axis values. Use the two nearest
        red guides that bracket the visible top edge and estimate the edge's fractional position between them.
        Do not snap to the nearest tick; interpolate proportionally between adjacent red grid lines.
        Only return a tick value when the colored bar's top edge is exactly on that red grid line.
        If the top edge is between two red grid lines, estimate the fraction of the gap. If it is just below the upper red grid line, return a value just below that upper tick, not the lower tick.
        Use the visible bar edge first and the red ruler second; labels printed elsewhere in the chart are not target values.
        """
    else:
        if is_stacked:
            base_prompt = f"""
        You are analyzing a vertical stacked bar chart that contains reference grid lines.
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        {color_desc}
        Locate the colored stack segment for series [{series_name}] at x-axis category [{x_label}].
        Estimate its own value by subtracting the y-axis value at its bottom boundary from the y-axis value at its top boundary.
        """
        else:
            base_prompt = f"""
        You are analyzing a vertical bar chart that contains reference grid lines.
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        {color_desc}
        {series_order_desc}
        {target_series_position}
        {target_x_hint}
        {data_value_rule}
        Locate the bar for series [{series_name}] at x-axis category [{x_label}].
        Treat the x tick [{x_label}] as the center of its category group. In a grouped chart, first isolate that category group, then select only the [{series_name}] bar by color; ignore bars from neighboring x categories.
        Identify its top edge and estimate the corresponding y value by interpolating between the nearest horizontal grid lines. Return the y-axis data value, not a pixel coordinate.
        """

    if prompt_type == "feedback" and pred_feedback and current_round >= feedback_round:
        pred = pred_feedback[-1]
        if is_stacked:
            base_prompt = f"""
        You are analyzing a vertical stacked bar chart with reference grid lines.
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        {color_desc}
        {series_order_desc}
        {target_series_position}
        {target_x_hint}

        The previous estimate for [{item_name}] was y = {pred[1]}. The red crosshair is only a scale reference for that estimate.
        Re-locate the target colored stack segment, compare its bottom and top boundaries against the grid, and refine the segment value as top minus bottom.
        """
        else:
            base_prompt = f"""
        You are analyzing a vertical bar chart with reference grid lines.
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        {color_desc}
        {series_order_desc}
        {target_series_position}
        {target_x_hint}
        {data_value_rule}

        The previous prediction for [{item_name}] is marked on the chart at x = "{pred[0]}", y = {pred[1]}.
        Only evaluate the exact target [{item_name}]. Ignore any neighboring category, legend item, or other bar even if it is clearer.
        If the red mark or crop falls on a neighboring category group, return to the category group centered at x tick [{x_label}] and the [{series_name}] colored bar.
        Compare that mark with the true top edge of the target bar, then refine the y value. Return the y-axis data value, not a pixel coordinate.
        """

    base_prompt += f"""
    Only respond in this JSON format:
    {{"readable": true, "datapoints": [{{"{item_name}": ["{x_label}", y]}}]}}
    The datapoints list must contain exactly one object and exactly the key "{item_name}".
    If the target is not readable in an amplifier crop, set `"readable": false` and use null for y.
    """
    return base_prompt
