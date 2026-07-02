"""Runtime knobs shared by chart prediction scripts."""

from __future__ import annotations

import os


def get_repeat_times(default: int = 3) -> int:
    if os.getenv("CHART_EXPERIMENT_MODE", "").strip().lower() in {"gt", "gt_grid", "true"}:
        return 1
    raw = os.getenv("CHART_REPEAT_TIMES", str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"CHART_REPEAT_TIMES must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError("CHART_REPEAT_TIMES must be >= 1")
    return value


def is_gt_experiment_mode() -> bool:
    return os.getenv("CHART_EXPERIMENT_MODE", "").strip().lower() in {"gt", "gt_grid", "true"}


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def get_bar_amplifier_rounds(default: int = 3) -> int:
    """Number of chained amplifier refinement rounds for h_bar/v_bar."""
    raw = os.getenv("CHART_BAR_AMPLIFIER_ROUNDS")
    if raw is not None:
        return _positive_int_from_env("CHART_BAR_AMPLIFIER_ROUNDS", default)
    return _positive_int_from_env("CHART_AMPLIFIER_ROUNDS", default)


def get_feedback_rounds(default: int = 2) -> int:
    return _positive_int_from_env("CHART_FEEDBACK_ROUNDS", default)


def get_amplifier_rounds(default: int = 3) -> int:
    return _positive_int_from_env("CHART_AMPLIFIER_ROUNDS", default)


def get_prompt_rounds(prompt_type: str, default_repeat_times: int) -> int:
    if not is_gt_experiment_mode():
        return default_repeat_times
    normalized = str(prompt_type or "").strip().lower()
    if normalized in {"baseline", "grid"}:
        return 1
    if normalized == "feedback":
        return get_feedback_rounds(2)
    if normalized in {"amplifier", "feedback_crop_adaptive", "feedback_crop"}:
        return get_amplifier_rounds(3)
    return default_repeat_times


def get_match_rne_tolerance(default: float = 0.01) -> float:
    raw = os.getenv("CHART_MATCH_RNE_TOLERANCE", str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"CHART_MATCH_RNE_TOLERANCE must be numeric, got {raw!r}") from None
    if value < 0:
        raise ValueError("CHART_MATCH_RNE_TOLERANCE must be >= 0")
    return value


def get_prediction_consistency_tolerance(default: float = 0.01) -> float:
    raw = os.getenv("CHART_PREDICTION_CONSISTENCY_TOLERANCE", str(default))
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"CHART_PREDICTION_CONSISTENCY_TOLERANCE must be numeric, got {raw!r}") from None
    if value < 0:
        raise ValueError("CHART_PREDICTION_CONSISTENCY_TOLERANCE must be >= 0")
    return value


def normalize_share_value(value) -> float | None:
    """Normalize circular values to a 0..1 share.

    The GT files and model outputs may use fractions, percentages, or degrees.
    """
    try:
        number = abs(float(value))
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    if number > 1.0:
        if number <= 100.0:
            return number / 100.0
        return number / 360.0
    return number


def circular_prediction_matches_gt(predicted, gt, *, tolerance: float | None = None) -> bool:
    pred_share = normalize_share_value(predicted)
    gt_share = normalize_share_value(gt)
    if pred_share is None or gt_share is None:
        return False
    limit = get_match_rne_tolerance() if tolerance is None else tolerance
    return abs(pred_share - gt_share) <= limit


def numeric_axis_span(ticks: list | None) -> float | None:
    values: list[float] = []
    for tick in ticks or []:
        try:
            values.append(float(tick))
        except (TypeError, ValueError):
            continue
    if len(values) < 2:
        return None
    span = max(values) - min(values)
    return span if span > 0 else None


def value_matches_gt(predicted, gt, axis_span: float | None, *, tolerance: float | None = None) -> bool:
    if gt is None or axis_span is None or axis_span <= 0:
        return False
    try:
        pred_value = float(predicted)
        gt_value = float(gt)
    except (TypeError, ValueError):
        return False
    return abs(pred_value - gt_value) / axis_span <= get_match_rne_tolerance() if tolerance is None else abs(pred_value - gt_value) / axis_span <= tolerance


def prediction_matches_gt(predicted, gt, ticks: list | None) -> bool:
    return value_matches_gt(predicted, gt, numeric_axis_span(ticks))


def value_consistent(current, reference, axis_span: float | None = None, *, tolerance: float | None = None) -> bool:
    """Return whether two model predictions are effectively unchanged.

    This intentionally compares prediction-to-prediction only. It must not use GT
    values because refinement stopping is part of the generation process.
    """
    try:
        current_value = float(current)
        reference_value = float(reference)
    except (TypeError, ValueError):
        return False
    if current_value != current_value or reference_value != reference_value:
        return False
    limit = get_prediction_consistency_tolerance() if tolerance is None else tolerance
    scale = axis_span if axis_span is not None and axis_span > 0 else max(abs(current_value), abs(reference_value), 1.0)
    return abs(current_value - reference_value) / scale <= limit


def point_prediction_consistent(
    current,
    reference,
    *,
    x_span: float | None = None,
    y_span: float | None = None,
    tolerance: float | None = None,
) -> bool:
    try:
        return value_consistent(current[0], reference[0], x_span, tolerance=tolerance) and value_consistent(
            current[1],
            reference[1],
            y_span,
            tolerance=tolerance,
        )
    except Exception:
        return False


def circular_prediction_consistent(current, reference, *, tolerance: float | None = None) -> bool:
    limit = get_prediction_consistency_tolerance() if tolerance is None else tolerance
    current_share = normalize_share_value(current)
    reference_share = normalize_share_value(reference)
    if current_share is not None and reference_share is not None:
        return abs(current_share - reference_share) <= limit
    return False
