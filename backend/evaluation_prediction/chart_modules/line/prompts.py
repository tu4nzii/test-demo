"""Prompt builders for line chart value prediction."""

from __future__ import annotations

from typing import Any

from ..h_bar.prompts import generate_series_color_description


def split_item_name(item_name: str) -> tuple[str, str]:
    try:
        series_name, x_label = item_name.rsplit(",", 1)
    except ValueError:
        raise ValueError(
            f"item_name {item_name!r} must use the 'series_name, x_label' format for line charts."
        ) from None
    return series_name.strip(), x_label.strip()


def _format_ticks(ticks: list[Any]) -> str:
    return ", ".join(str(item) for item in ticks)


def _format_tick_pixel_pairs(ticks: list[Any], pixels: list[Any] | None) -> str:
    if not pixels:
        return ""
    return ", ".join(f"{tick}->{pixel}px" for tick, pixel in zip(ticks, pixels))


def _format_visible_ticks(visible_ticks: list[Any] | None) -> str:
    visible_ticks = visible_ticks or []
    return ", ".join(str(round(float(y), 2)) for y in sorted(set(visible_ticks)))


def build_point_exists_prompt(point_name: str, series_color: dict[str, str]) -> str:
    series_name, x_label = split_item_name(point_name)
    color_desc = generate_series_color_description(series_color)
    return (
        f"You are given a cropped line chart image for [{point_name}].\n"
        f"{color_desc}\n"
        f"Please check if the line point for series [{series_name}] at x-axis category [{x_label}] is visible.\n"
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )


def generate_prompt(
    *,
    item_name: str,
    prompt_type: str,
    x_ticks: list[Any],
    y_ticks: list[Any],
    series_color: dict[str, str],
    x_pixels: list[Any] | None = None,
    y_pixels: list[Any] | None = None,
    visible_ticks: list[Any] | None = None,
    pred_feedback: tuple[Any, Any] | None = None,
) -> str:
    series_name, x_label = split_item_name(item_name)
    x_tick_str = _format_ticks(x_ticks)
    y_tick_str = _format_ticks(y_ticks)
    x_pixel_str = _format_tick_pixel_pairs(x_ticks, x_pixels)
    y_pixel_str = _format_tick_pixel_pairs(y_ticks, y_pixels)
    color_desc = generate_series_color_description(series_color)

    if prompt_type == "baseline":
        base_prompt = f"""
        You are given a line chart image.
        Locate the line representing series [{series_name}] and estimate the y value at x-axis category [{x_label}].
        Use the visual position of the line marker/intersection at that x category, not nearby labels.
        {color_desc}
        """
    elif prompt_type == "amplifier":
        visible_tick_str = _format_visible_ticks(visible_ticks)
        base_prompt = f"""
        You are given a cropped line chart image centered around [{item_name}].
        The cropped region includes the target x-axis category [{x_label}] and y-axis reference ticks [{visible_tick_str}].
        Locate the series [{series_name}] line point in this crop and estimate its y value by interpolation.
        The red horizontal guide labels are original chart y-axis values. Use the two nearest red horizontal
        guides around the visible line point and estimate the fractional position between them.
        The red vertical guide marks the target x category; read where the target-colored line intersects this guide.
        Do not output crop pixel coordinates, resized-image pixel coordinates, or the position of nearby text labels.
        If the target series point at [{x_label}] is not visible or cannot be confidently identified in this crop, return the same JSON shape with `"readable": false` and use `null` for y.
        {color_desc}
        """
    elif prompt_type in {"grid", "feedback"}:
        base_prompt = f"""
        You are analyzing a line chart with reference grid lines aligned with these axis ticks:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        - GT X tick-to-pixel mapping: [{x_pixel_str}]
        - GT Y tick-to-pixel mapping: [{y_pixel_str}]
        {color_desc}

        Locate the series [{series_name}] at x-axis category [{x_label}].
        Estimate the y value at that exact x position by interpolating between the nearest horizontal grid lines.
        """
        if prompt_type == "feedback" and pred_feedback is not None:
            base_prompt += f"""
            A previous prediction is marked on the chart at x = [{pred_feedback[0]}], y = {pred_feedback[1]}.
            Compare that marker with the actual line point and refine the y estimate.
            """
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    base_prompt += f"""
    Only respond in this JSON format:
    {{"readable": true, "datapoints": [{{"{item_name}": ["{x_label}", y]}}]}}
    If the target is not readable in an amplifier crop, set `"readable": false` and use null for y.
    """
    return base_prompt
