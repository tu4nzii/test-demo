"""Shared visual style for zoom-in/amplifier process images.

This is intentionally separate from gt_grid_renderer.py. GT grid images use the
publication grid style (#cccccc, 2/2 dash), while amplifier crops use red local
guides inherited from the original zoom-in experiments.
"""

from __future__ import annotations

from math import hypot
from typing import Any

from PIL import Image, ImageDraw

from .model_vision_registry import active_model_vision_profile, target_side_for_family


AMPLIFIER_GRID_COLOR = (255, 0, 0)
AMPLIFIER_GRID_COLOR_RGBA = (255, 0, 0, 255)
AMPLIFIER_GUIDE_COLOR_RGBA = (230, 35, 35, 255)
AMPLIFIER_GRID_WIDTH = 1
AMPLIFIER_BAR_DASH_LENGTH = 10
AMPLIFIER_BAR_DASH_GAP = 4
AMPLIFIER_POINT_DASH_LENGTH = 1
AMPLIFIER_POINT_DASH_GAP = 4
AMPLIFIER_TARGET_SIDE = active_model_vision_profile().target_side
AMPLIFIER_MAX_SIDE = active_model_vision_profile().max_side
AMPLIFIER_LABEL_PAD = active_model_vision_profile().label_pad


def amplifier_half_window(
    base_span_px: float,
    round_index: int,
    *,
    roi_scale: float = 1.0,
    min_px: int = 36,
    plot_span_px: float | None = None,
) -> int:
    """Return a prediction-centered crop half window for amplifier images.

    The schedule keeps enough surrounding context for MLLM visual encoders while
    still making later rounds more local. It intentionally avoids halving the
    crop every round, which often removes the nearest visible edge or tick pair.
    """
    schedule = active_model_vision_profile().half_window_schedule
    index = max(1, int(round_index)) - 1
    factor = schedule[min(index, len(schedule) - 1)]
    raw_half_window = float(base_span_px) * factor * max(1.0, float(roi_scale))
    if plot_span_px is not None and plot_span_px > 0:
        max_full_window = amplifier_source_window_limit(
            plot_span_px,
            round_index,
            roi_scale=roi_scale,
        )
        raw_half_window = min(raw_half_window, max_full_window / 2.0)
    return int(max(min_px, raw_half_window))


def amplifier_source_window_limit(
    plot_span_px: float,
    round_index: int,
    *,
    roi_scale: float = 1.0,
) -> int:
    """Maximum source-image ROI width/height before upsampling.

    Source ROI size and output image size are intentionally decoupled. The ROI
    should be local enough to remove distractors, then upsampled to the model's
    visual budget. ``roi_scale`` is used only after an unreadable amplifier
    round; it broadens context without returning to whole-chart crops.
    """
    fractions = (0.32, 0.24, 0.18)
    index = max(1, int(round_index)) - 1
    base_fraction = fractions[min(index, len(fractions) - 1)]
    scale = max(1.0, min(2.0, float(roi_scale)))
    fraction = min(0.55, base_fraction * scale)
    return int(max(1, float(plot_span_px) * fraction))


def amplifier_point_source_window(
    requested_size_px: float,
    round_index: int,
    *,
    x_plot_span_px: float | None,
    y_plot_span_px: float | None,
    roi_scale: float = 1.0,
    min_px: int = 60,
) -> int:
    spans = [float(value) for value in (x_plot_span_px, y_plot_span_px) if value and value > 0]
    size = float(requested_size_px)
    if spans:
        size = min(size, amplifier_source_window_limit(min(spans), round_index, roi_scale=roi_scale))
    return int(max(min_px, size))


def amplifier_tick_divisor(round_index: int) -> int:
    """Number of subdivisions per original tick interval for local rulers."""
    schedule = active_model_vision_profile().tick_divisors
    index = max(1, int(round_index)) - 1
    return schedule[min(index, len(schedule) - 1)]


def amplifier_output_side(crop_w: int, crop_h: int, *, family: str = "ruler") -> int:
    """Choose a square output side that keeps text readable for MLLMs."""
    crop_side = max(1, int(max(crop_w, crop_h)))
    profile = active_model_vision_profile()
    target_side = target_side_for_family(profile, family)
    if crop_side >= target_side:
        return min(profile.max_side, crop_side)
    return target_side


def amplifier_label_pad() -> int:
    return active_model_vision_profile().label_pad


def amplifier_target_side(*, family: str = "sector") -> int:
    profile = active_model_vision_profile()
    return target_side_for_family(profile, family)


def amplifier_max_side() -> int:
    return active_model_vision_profile().max_side


def amplifier_point_output_side(crop_size: int) -> int:
    profile = active_model_vision_profile()
    target_side = target_side_for_family(profile, "point")
    return min(max(target_side, int(crop_size)), profile.max_side)


def amplifier_point_grid_density() -> int:
    return active_model_vision_profile().point_grid_density


def amplifier_polar_window_factor(round_index: int) -> float:
    schedule = active_model_vision_profile().polar_window_factors
    index = max(1, int(round_index)) - 1
    return schedule[min(index, len(schedule) - 1)]


def amplifier_polar_min_window() -> int:
    return active_model_vision_profile().polar_min_window


def draw_amplifier_dashed_line(
    draw: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: Any = AMPLIFIER_GRID_COLOR,
    width: int = AMPLIFIER_GRID_WIDTH,
    dash_length: int = AMPLIFIER_BAR_DASH_LENGTH,
    gap_length: int = AMPLIFIER_BAR_DASH_GAP,
) -> None:
    x1, y1 = start
    x2, y2 = end
    total = hypot(x2 - x1, y2 - y1)
    if total <= 0:
        return
    step = max(1, dash_length + gap_length)
    for index in range(int(total // step) + 1):
        start_frac = (index * step) / total
        end_frac = min((index * step + dash_length) / total, 1)
        sx = x1 + (x2 - x1) * start_frac
        sy = y1 + (y2 - y1) * start_frac
        ex = x1 + (x2 - x1) * end_frac
        ey = y1 + (y2 - y1) * end_frac
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)


def draw_centered_label_box(
    draw: Any,
    text: str,
    center: tuple[float, float],
    *,
    font: Any,
    fill: Any = (0, 0, 0),
    background: Any = (255, 255, 255, 230),
    outline: Any | None = None,
    padding: tuple[int, int] = (4, 2),
) -> tuple[float, float, float, float]:
    """Draw a readable label whose box center is exactly ``center``.

    Amplifier labels are semantic tick/ruler annotations. Keeping the label box
    center on the corresponding grid line prevents the label from visually
    floating above/below/left/right of the reference it names.
    """
    cx, cy = float(center[0]), float(center[1])
    pad_x, pad_y = padding
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    left = cx - text_w / 2 - pad_x
    top = cy - text_h / 2 - pad_y
    right = cx + text_w / 2 + pad_x
    bottom = cy + text_h / 2 + pad_y
    draw.rectangle((left, top, right, bottom), fill=background, outline=outline)
    draw.text((cx - text_w / 2 - bbox[0], cy - text_h / 2 - bbox[1]), text, fill=fill, font=font)
    return left, top, right, bottom


def draw_rotated_centered_label_box(
    image: Image.Image,
    text: str,
    center: tuple[float, float],
    angle: float,
    *,
    font: Any,
    fill: Any = (0, 0, 0, 255),
    background: Any = (255, 255, 255, 230),
    outline: Any | None = None,
    padding: tuple[int, int] = (4, 2),
    resample: int = Image.Resampling.BICUBIC,
) -> tuple[int, int, int, int]:
    """Draw a rotated label box centered on a grid-line point."""
    scratch = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), text, font=font)
    text_w = max(1, bbox[2] - bbox[0])
    text_h = max(1, bbox[3] - bbox[1])
    pad_x, pad_y = padding
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2
    label_img = Image.new("RGBA", (box_w, box_h), background)
    label_draw = ImageDraw.Draw(label_img)
    if outline is not None:
        label_draw.rectangle((0, 0, box_w - 1, box_h - 1), outline=outline)
    label_draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=fill)
    rotated = label_img.rotate(angle, expand=True, resample=resample)
    x = int(round(float(center[0]) - rotated.width / 2))
    y = int(round(float(center[1]) - rotated.height / 2))
    image.paste(rotated, (x, y), rotated)
    return x, y, x + rotated.width, y + rotated.height
