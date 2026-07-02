"""Prompt builders for horizontal bar value prediction."""

from __future__ import annotations

import re
from typing import Any

import matplotlib.colors as mcolors


def normalize_hex_color(hex_color: Any) -> str:
    text = str(hex_color or "").strip()
    match = re.search(r"#[0-9a-fA-F]{6}", text)
    if match:
        return match.group(0)
    text = text.strip("'\"[](){} ")
    if re.fullmatch(r"[0-9a-fA-F]{6}", text):
        return f"#{text}"
    return "#000000"


def hex_to_rgb(hex_color: Any) -> tuple[int, int, int]:
    hex_color = normalize_hex_color(hex_color).lstrip("#")
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
        normalized_hex = normalize_hex_color(hex_val)
        rgb = hex_to_rgb(normalized_hex)
        color_name = get_color_name_approx(normalized_hex)
        lines.append(f'- "{series}" is colored **{normalized_hex}** (approx. {color_name}, RGB: {rgb}).')
    return "\n".join(lines)


def build_color_prompt(point_name: str, series_color: dict[str, str], chart_type: str = "h_bar") -> str:
    try:
        series_name, y_label = split_item_name(point_name)
    except Exception:
        series_name = point_name.split(",", 1)[0].strip()
        y_label = point_name.rsplit(",", 1)[-1].strip()
    color_desc = generate_series_color_description(series_color)
    if str(chart_type or "").lower() == "h_stacked_bar":
        return (
            f"You are given a cropped horizontal stacked bar chart image for {point_name}.\n"
            f"**{color_desc}**.\n"
            f"Please check whether the cropped image contains the target colored stack segment "
            f"for series \"{series_name}\" at y-axis category \"{y_label}\", and whether both "
            "its left and right boundaries are visible inside the crop. Do not return true if "
            "only the middle of the segment is visible.\n"
            "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
        )
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


def _format_tick_pixel_pairs(ticks: list[Any], pixels: list[Any] | None) -> str:
    if not pixels:
        return ""
    return ", ".join(f"{tick}->{pixel}px" for tick, pixel in zip(ticks, pixels))


def _format_visible_ticks(visible_ticks: list[Any] | None) -> str:
    visible_ticks = visible_ticks or []
    if visible_ticks and isinstance(visible_ticks[0], list):
        visible_ticks = visible_ticks[0]
    return ", ".join(str(round(float(x), 2)) for x in sorted(set(visible_ticks)))


def _latest_numeric_feedback(pred_feedback: list | None) -> float | None:
    if not pred_feedback:
        return None
    try:
        value = pred_feedback[-1][0]
        return float(value)
    except Exception:
        return None


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
    x_pixels: list | None = None,
    y_pixels: list | None = None,
    visible_ticks: list | None = None,
    axis_types: dict | None = None,
    pred_feedback: list | None = None,
    feedback_round: int = 0,
    current_round: int = 1,
    chart_type: str = "h_bar",
) -> str:
    series_name, y_label = split_item_name(item_name)
    x_tick_str = _format_ticks(x_ticks)
    y_tick_str = _format_ticks(y_ticks)
    x_pixel_str = _format_tick_pixel_pairs(x_ticks, x_pixels)
    y_pixel_str = _format_tick_pixel_pairs(y_ticks, y_pixels)
    color_desc = generate_series_color_description(series_color)
    is_stacked = str(chart_type or "").lower() == "h_stacked_bar"
    series_order = [str(name) for name in series_color.keys()]
    grouped_order_hint = ""
    if len(series_order) > 1 and series_name in series_order:
        ordinal = series_order.index(series_name) + 1
        visual_order = list(reversed(series_order))
        visual_ordinal = visual_order.index(series_name) + 1
        same_color = len({normalize_hex_color(value).lower() for value in series_color.values()}) == 1
        grouped_order_hint = (
            f'Within each y-axis category group, the grouped bars follow this top-to-bottom visual order: '
            f'{", ".join(visual_order)}. The legend/data order is {", ".join(series_order)} from bottom to top. '
            f'The target series "{series_name}" is bar #{visual_ordinal} from the top '
            f'and bar #{ordinal} from the bottom inside category "{y_label}".'
            f' The target category group is the complete group centered vertically in the crop; ignore partial bars '
            f'cut off near the top or bottom because they belong to neighboring categories.'
        )
        if same_color:
            grouped_order_hint += (
                " All series use the same color, so color alone cannot identify the target; use this vertical order."
            )
    latest_feedback_x = _latest_numeric_feedback(pred_feedback)
    signed_bar_hint = ""
    if latest_feedback_x is not None:
        if latest_feedback_x < 0:
            signed_bar_hint = (
                "The previous full-flow estimate is negative. For a negative horizontal bar, the bar extends leftward "
                "from the zero axis. Read the leftmost blue-white outer boundary, preserve the minus sign, and return "
                "a negative x-axis data value. Do not mirror it to a positive value."
            )
        elif latest_feedback_x > 0:
            signed_bar_hint = (
                "The previous full-flow estimate is positive. For a positive horizontal bar, read the rightmost "
                "blue-white outer boundary and return a positive x-axis data value."
            )
    single_series_hint = ""
    if series_name.strip().casefold() in {"none", "null", "single", "series 1", ""}:
        single_series_hint = (
            f'This is a single-series horizontal bar chart. The word "{series_name}" is an internal placeholder, '
            f'not a visible legend item. The target is the blue bar on y-axis category "{y_label}".'
        )
    data_value_rule = f"""
        Response rule:
            - Return only valid JSON. Do not write reasoning, equations, Markdown, or explanatory text.
            - The JSON key must be exactly "{item_name}". Do not return any other bar, nearby category, legend item, or series.
            - {single_series_hint}
        Output rule:
            - Return the x-axis DATA VALUE, never a pixel coordinate.
            - Values such as 519, 646, or 703 are pixel positions and are invalid as final answers.
            - Use tick labels and tick-to-pixel pairs only for reasoning. For example, a right edge near 646px corresponds to x=0.5, and near 703px corresponds to x=0.55.
            - Numeric labels drawn inside the colored bar are grid labels, not the bar's value. Ignore those labels as answers and read the bar's right edge.
            - The right side of the chart may crop or hide the largest tick labels. Still use the provided X-axis tick list and GT X tick-to-pixel mapping: a bar ending near the far right plot boundary may correspond to a high value close to the maximum x tick, not merely the last clearly printed tick label.
            - For very long bars, inspect the farthest blue-white boundary at the right end of the bar and interpolate against the full axis range, including ticks that are listed in text even if their label is partially clipped in the image.
            - {signed_bar_hint}
            - {grouped_order_hint}
        """

    if prompt_type == "baseline":
        if is_stacked:
            base_prompt = (
                "You are given a horizontal stacked bar chart image. "
                f"{color_desc}\n"
                f"Your task is to predict the x-axis value represented by the colored stack segment labeled [{item_name}].\n"
                "Locate both the left and right boundaries of this colored segment, then subtract the left boundary value from the right boundary value."
            )
        else:
            base_prompt = (
                "You are given a bar chart image. "
                f"{color_desc}\n"
                f"Your task is to predict the x coordinate for the segment labeled [{item_name}].\n"
                f"To identify the x coordinate, first locate the tick interval in which the right boundary "
                f"of the segment representing [{item_name}] falls."
            )
    elif prompt_type == "amplifier":
        visible_tick_str = _format_visible_ticks(visible_ticks)
        if is_stacked:
            base_prompt = f"""
        You are given a cropped horizontal stacked bar chart image. Your task is to predict the x-axis value represented by the colored segment labeled [{item_name}].
        The cropped image is centered on the y-axis category group **"{y_label}"**; the target stacked segment may start away from zero.
        The crop includes a **horizontally drawn x-axis/ruler**. The bottom ruler has tick labels [{visible_tick_str}], and the red vertical dashed lines are the corresponding grid lines.
        Use the series color to select the correct segment. {color_desc}
        Instructions:
            - Locate the target colored segment for [{item_name}].
            - If the target colored segment is not visible or cannot be confidently identified in this crop, return the same JSON shape with `"readable": false` and use `null` for x.
            - Estimate the x-axis value at the segment's left boundary and right boundary.
            - Return the segment value: right boundary value minus left boundary value.
            - Do not return the cumulative right-edge value unless the segment starts at zero.
        """
        else:
            base_prompt = f"""
        You are given a chart image. Your task is to predict the x coordinate for the segment labeled [{item_name}].
        The cropped image is centered on the y-axis category group **"{y_label}"**. In grouped or stacked bar charts, the target colored segment may appear above or below the vertical center within that category group; use the series color to select the correct segment.
        {grouped_order_hint}
        The top and bottom sides include a **horizontally drawn x-axis**, with tick values [{visible_tick_str}] and grid lines.
        Your task is to estimate the **x coordinate** corresponding to the colored segment's outer boundary away from the zero axis.
        The segment color indicates its series: use alignment between the legend and segment to verify the target. {color_desc}
        {data_value_rule}
        Instructions:
            - First, locate the x-axis tick interval in which the segment's value boundary falls.
            - Confirm that the visible bar belongs to the exact y-axis category **"{y_label}"** and the exact series in [{item_name}]. If the exact target category/series is not visible, return `"readable": false`.
            - For a positive value, read the rightmost vertical edge of the colored rectangle: the place where bar fill ends and white background begins.
            - For a negative value, read the leftmost vertical edge of the colored rectangle: the place where bar fill ends and white background begins. Preserve the minus sign from the red tick labels.
            - Do not use the zero-side edge, the middle of the bar, or an internal red/grid line as the final value.
            - The value boundary/end edge of the target bar must be visible. If the crop only shows the middle/body of the bar and the value boundary is outside the crop, return the same JSON shape with `"readable": false` and use `null` for x.
            - If the blue-white value boundary is visible at or very near a red vertical guide, and white background is visible immediately beyond that boundary, the target is readable. In that case estimate the value; do not return `"readable": false`.
            - A boundary close to a local crop frame or black ruler line is still readable when the blue fill ends and the adjacent white region is visible.
            - Then, determine the relative position of the boundary within this interval. Use linear interpolation between the two tick values to estimate the precise x-axis value.
            - In the zoom-in crop, the red vertical guide labels are original chart x-axis values. Use the two nearest red guides that bracket the visible value boundary and estimate the boundary's fractional position between them.
            - **Important:** Do not snap or round to the nearest tick; interpolate proportionally.
            - Only return a tick value when the blue-white boundary is exactly on that red grid line. If the boundary is between two red grid lines, estimate the fraction of the gap. For example, 60% of the way from 0.225 to 0.250 is about 0.240.
            - If the boundary is just before the next red grid line, return a value just below that next tick, not the previous tick.
            - Use the visible bar edge first and the red ruler second; labels printed elsewhere in the chart are not target values.
            - **Edge case:** If the segment cannot be visually detected even near the **minimum tick boundary**, return `"readable": false` so the next amplifier round can expand the ROI.
        """
    else:
        if is_stacked:
            base_prompt = f"""
        You are analyzing a horizontal stacked bar chart that contains reference grid lines.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        {color_desc}
        After locating the correct colored stack segment for [{item_name}], estimate its own value by subtracting the x-axis value at its left boundary from the x-axis value at its right boundary.
        """
        else:
            base_prompt = f"""
        You are analyzing a bar chart that contains **reference grid lines**, where horizontal lines correspond to y-axis ticks, and vertical lines align with x-axis ticks.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        {color_desc}
        {grouped_order_hint}
        {data_value_rule}
        After locating the correct segment for [{item_name}], identify the position of its right edge by comparing it with the two nearest vertical grid lines on the X-axis.
        The right edge means the rightmost blue-white boundary of the bar, not any red/grid line or label inside the blue rectangle.
        Use linear interpolation between these two ticks to estimate the accurate X-coordinate of [{item_name}].
        Do not return the pixel x-coordinate of the right edge; return the corresponding data value on the x-axis.
        """

    if prompt_type == "feedback" and pred_feedback and current_round >= feedback_round:
        pred = pred_feedback[-1]
        x = f"{pred[0]:.2f}" if isinstance(pred[0], (int, float)) else f'"{pred[0]}"'
        y = f"{pred[1]:.2f}" if isinstance(pred[1], (int, float)) else f'"{pred[1]}"'
        if is_stacked:
            base_prompt = f"""
        You are analyzing a horizontal stacked bar chart with reference grid lines.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        {color_desc}

        The previous estimate for the value of [{item_name}] was x = {x}. The red crosshair is only a scale reference for that estimate.
        Re-locate the target colored stack segment, compare its left and right boundaries against the grid, and refine the segment value as right minus left.
        """
        else:
            base_prompt = f"""
        You are analyzing a bar chart with reference grid lines.
        - Y-axis ticks: [{y_tick_str}]
        - X-axis ticks: [{x_tick_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        {color_desc}
        {data_value_rule}
        {grouped_order_hint}

        The given chart shows your previous prediction for the x coordinate of [{item_name}], aligned with a red crosshair at (x = {x}, y = {y}).
        Only evaluate the exact target [{item_name}]. Ignore any neighboring row, legend item, or other bar even if it is clearer.
        Compare this red crosshair to the true right boundary of [{item_name}]: determine whether it is too far left, too far right, or aligned correctly.
        The true boundary is the rightmost blue-white boundary of the target bar.
        After verifying, adjust the prediction proportionally to refine your estimate to get the most accurate result x coordinate of [{item_name}].
        Return the refined data value, not a pixel coordinate.
        """

    base_prompt += f"""
    Only respond in this JSON format:
    {{"readable": true, "datapoints": [{{"{item_name}": [x, "{y_label}"]}}]}}
    The datapoints list must contain exactly one object and exactly the key "{item_name}".
    If the target is not readable in an amplifier crop, set `"readable": false` and use null for x.
    """
    return base_prompt
