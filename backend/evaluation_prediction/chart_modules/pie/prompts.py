"""Prompt construction for pie-chart prediction flows."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union


def _color_context(item_name: str, series_color: dict[str, Any] | None) -> str:
    if not isinstance(series_color, dict) or not series_color:
        return ""
    lines = ["Label-color mapping from the chart configuration:"]
    for label, color in series_color.items():
        lines.append(f'- "{label}" uses color {color}.')
    target_color = series_color.get(item_name)
    if target_color:
        lines.append(
            f'Before reading angles, follow the text label/leader line for "{item_name}" and select the continuous sector with color {target_color}.'
        )
        lines.append(
            "If a red feedback arc or a cropped amplifier view mostly covers a different color, treat it as the complement or a wrong local crop; do not report that other colored wedge as the target."
        )
    return "\n".join(lines)

def generate_prompt(
        item_name: str,
        prompt_type: str,
        prev_angle: Optional[Union[float, Dict[str, float]]] = None,
        drawn_angles: Optional[list[int]] = None,
        angle_order_hint: Optional[str] = None,
        series_color: Optional[dict[str, Any]] = None,
) -> str:
    color_context = _color_context(item_name, series_color)
    printed_value_rule = f"""
        Important value-reading rule:
        - If numeric data labels are printed inside or near the pie sectors, read those numbers first.
        - This rule has priority over every angle-grid rule for the percentage field.
        - When labels for all sectors are visible, compute the percentage for "{item_name}" as:
          target numeric value / sum of all visible sector numeric values * 100.
        - Use angular span only for start_angle/end_angle, or for percentage only when explicit numeric sector values are not visible or cannot be read.
        - If printed numeric sector values are visible, do not estimate percentage from the sector angle or area.
        - Do not confuse a printed sector value such as 55 or 80 with a percentage unless the chart explicitly marks it as percent.
        """
    # —— baseline ——
    if prompt_type == "baseline":
        prompt = f"""
        You are analyzing a pie chart. It shows data proportions using circular sectors, which divide the circle into slices. The size of each sector, represented by its central angle and area, corresponds to its proportion of the whole.
        Your task is to estimate the **percentage** value for the pie chart sector labeled "{item_name}".
        {color_context}
        {printed_value_rule}
        Output *only*:
        {{"datapoints": [{{"{item_name}": percentage}}]}}
        """
        return prompt

    # —— amplifier ——
    elif prompt_type == "amplifier":
        # 上一轮预测说明
        if isinstance(prev_angle, dict) and "start_angle" in prev_angle and "end_angle" in prev_angle:
            try:
                sa = float(prev_angle["start_angle"])
                ea = float(prev_angle["end_angle"])
                prev_str = (
                    f'The previous prediction for "{item_name}" defined a clockwise sector from the start angle to the end angle, that is'
                    f'**start: {sa:.1f}°**, **end: {ea:.1f}°**, the entire arc is considered the candidate range for this item.'
                )
            except Exception:
                prev_str = ""
        else:
            prev_str = ""

        # 注入网格线信息
        if drawn_angles:
            ticks_str = ", ".join(f"{a}°" for a in drawn_angles)
            grid_str = f"To support angle estimation, the visible radial tick marks in the image are drawn clockwise at: {ticks_str}."
        else:
            grid_str = ""

        # ✅ angle_order_hint 注入
        if angle_order_hint:
            order_hint_str = f"⚠️⚠️Note: {angle_order_hint}"
        else:
            order_hint_str = ""

        return f"""
        You are analyzing a zoomed-in cropped sector view for "{item_name}" in the given pie chart.
        The crop is generated from the previous prediction with angular padding for tolerance adjustment.
        {grid_str}
        {prev_str}
        {order_hint_str}
        Your task is to refine the percentage and the start/end angles of the sector labeled "{item_name}".
        {color_context}
        {printed_value_rule}
        The local angle grid is used only for boundary refinement; it is not ground truth. You must correct the previous boundaries if the label/color evidence indicates a different sector.
        If the crop does not contain enough of the target-colored sector to identify both boundaries, return {{"readable": false, "datapoints": [{{"{item_name}": {{"percentage": null, "start_angle": null, "end_angle": null}}}}]}}.
        The only valid rule is: moving clockwise from start_angle to end_angle must cover the sector labeled "{item_name}" itself, not its complement. The sector may cross 0°, so start_angle may be greater than end_angle.
        Critical guardrail: do not report the small complementary wedge unless that wedge is truly the sector labeled "{item_name}". A labeled sector may be larger than 50% of the pie.
        Instructions:
        -Locate the sector labeled "{item_name}" and identify its marked color.
        - Adjust the start and end angles to match the exact color boundaries in the clockwise direction:
        • The start angle is the first boundary where the target color appears.
        !!! If the target color appears at the start of the cropped region, the visible start angle must be the start of the target arc.
        • The end angle is the next boundary where the target color disappears.
        !!! If no second boundary exists, the visible end angel must be the end of the target arc.
        - Also estimate the percentage share directly from the visible sector area. Use percentage as the primary value; use angles only as supporting evidence.
        Output *only* this JSON:
        {{"datapoints": [{{"{item_name}": {{"percentage": percentage, "start_angle": start_angle, "end_angle": end_angle}}}}]}}
        """

    elif prompt_type == "feedback":
        if isinstance(prev_angle, dict) and "start_angle" in prev_angle and "end_angle" in prev_angle:
            prev_str = (
                f"The model previously predicted **start: {prev_angle['start_angle']:.1f}°**, "
                f"**end: {prev_angle['end_angle']:.1f}°**"
            )
        else:
            prev_str = ""

        prompt = f"""
                    You are analyzing a pie chart. It shows data proportions using circular sectors.
                    To support estimation, this chart includes radial reference lines every 15° around the full circle.
                    Angle convention: 0° points upward at 12 o'clock, and angles increase clockwise.
                    The only valid rule is: the clockwise red arc between the two cross markers must cover the sector labeled "{item_name}" itself, not the complementary sector.
                    The sector may cross 0°, so start_angle may be greater than end_angle.
                    Critical guardrail: if the visible label/color of "{item_name}" occupies the larger part of the pie, return a percentage greater than 50; do not replace it with the smaller complementary slice.
                    {color_context}
                    {prev_str} for the segment that represents "{item_name}", with the red visual feedback marks the predicted sector boundaries.
                    {printed_value_rule}
                    A cross marker is drawn at the start and end angles, and a red arc connects them clockwise to indicate the predicted sector range.

                    Your task is to **refine the percentage plus both the start angle and the end angle** of the sector labeled **"{item_name}"** by:
                    1. You should first check if the red arc from the last prediction aligns with the true sector boundaries.
                    - If it does, keep the current order of the start and end angles. If the red arc instead corresponds to the complementary region of the true sector, this means the start and end angles were reversed, and you must swap their order.
                    2. Identify the color of the sector representing "{item_name}", most likely located within the clockwise arc defined by the previous prediction’s start and end angles.
                    3. Compare the red visual feedback lines from last prediction with the true boundaries of the sector:
                       - If the red lines align with the true boundaries, keep the predictions.
                       - If not, adjust the **start and/or end angle** by adding or subtracting a few degrees, to make the prediction align with the true boundaries.
                    4. Estimate the percentage share directly from the visible sector area. Use percentage as the primary value; use angles only as supporting evidence.
                    Output *only*:
                    {{"datapoints": [{{"{item_name}": {{"percentage": <float>, "start_angle": <float>, "end_angle": <float>}}}}]}}
                """

    else:  # grid

        prompt = f"""
                You are analyzing a pie chart. It shows data proportions using circular sectors.
                Your task is to estimate the percentage and the start/end angles of the sector labeled "{item_name}", such that the clockwise sector between them exactly corresponds to this labeled sector.
                {color_context}
                {printed_value_rule}
                To support angle estimation, the chart includes angular reference lines every 15° around the full circle.
                Angle convention: 0° points upward and angles increase clockwise.
                The sector may cross 0°, so start_angle may be greater than end_angle.
                The only valid rule is: moving clockwise from start_angle to end_angle must cover the sector labeled "{item_name}" itself, not its complement.
                Critical guardrail: first identify the text label and its colored region, then estimate that region's area share. Do not report the complement. A sector may exceed 50%.
                Instructions for accurately estimating the start and end angles (in degrees) of the sector labeled "{item_name}":
                Locate the sector labeled "{item_name}" on the outer ring.
                Identify the start angle (x) — the angular position, measured clockwise, where the colored sector first begins.
                Identify the end angle (y) — the angular position, measured clockwise, where the same sector finishes.
                Use the reference lines to estimate each boundary angle as accurately as possible:
                First, find the two nearest reference lines bracketing the boundary.
                Then interpolate the sector’s position between them to compute a precise angle.
                Also estimate the percentage share directly from the visible sector area. Use percentage as the primary value; use angles only as supporting evidence.
                Output *only*:
                {{"datapoints": [{{"{item_name}": {{"percentage": <float>, "start_angle": <float>, "end_angle": <float>}}}}]}}
                """

    return prompt.strip()
