"""Deterministic final selection inside the grid/feedback/amplifier flow."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .chart_io import ensure_dir
from .runtime import get_prediction_consistency_tolerance, numeric_axis_span, value_consistent


FULL_FLOW_STAGES = ("grid", "feedback", "amplifier")
POINT_FULL_FLOW_STAGES = ("grid", "feedback", "feedback_crop_adaptive", "amplifier")
AMPLIFIER_OUTLIER_TOLERANCE = 0.06
AMPLIFIER_LARGE_OUTLIER_TOLERANCE = 0.25
AMPLIFIER_CONTEXT_DRIFT_TOLERANCE = 0.02
AMPLIFIER_WEAK_EVIDENCE_DRIFT_TOLERANCE = 0.01
AMPLIFIER_UNSTABLE_DRIFT_TOLERANCE = 0.01
AMPLIFIER_CONFIRMATION_DRIFT_TOLERANCE = 0.004
FULL_VIEW_SELECTION_STABILITY_TOLERANCE = 0.01
PREDICTION_ABSOLUTE_STABILITY_TOLERANCE = 0.03


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def bar_prediction_value(record: dict[str, Any], *, axis: str) -> float | None:
    if str(record.get("prediction_readable", "")).strip().lower() == "false":
        return None
    return finite_float(record.get(f"pred_{axis}"))


def select_bar_full_flow_record(
    records: list[dict[str, Any]],
    *,
    axis: str,
    axis_ticks: list[Any] | None,
    use_unstable_amplifier_median: bool = True,
    prefer_grid_on_later_stage_drift: bool = False,
    trust_readable_amplifier_over_full_view: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select one final bar prediction without consulting GT or baseline.

    The rule favors the latest mechanism stage that produced readable evidence.
    Baseline is excluded, and GT is never consulted.  Amplifier is the terminal
    refinement stage, so a readable amplifier sequence should not be overwritten
    by an earlier grid/feedback agreement.  If amplifier does not yield a
    readable value, fall back to prediction-to-prediction consistency from
    grid/feedback.
    """
    axis_span = numeric_axis_span(axis_ticks)
    axis_bounds = _numeric_axis_bounds(axis_ticks)
    full_records = [record for record in records if record.get("prompt_type") in FULL_FLOW_STAGES]
    stage_records: dict[str, list[dict[str, Any]]] = {stage: [] for stage in FULL_FLOW_STAGES}
    for record in full_records:
        value = bar_prediction_value(record, axis=axis)
        if value is None or not _value_inside_axis_bounds(value, axis_bounds):
            continue
        stage_records[str(record.get("prompt_type"))].append(record)

    amp_records = stage_records["amplifier"]
    feedback_records = stage_records["feedback"]
    grid_records = stage_records["grid"]
    selected: dict[str, Any] | None = None
    reason = "no_readable_full_flow_prediction"
    selected_value: float | None = None

    amp_values = [
        value
        for record in amp_records
        for value in [bar_prediction_value(record, axis=axis)]
        if value is not None
    ]
    stable_amp_values = _stable_tail_values(amp_records, axis=axis, axis_span=axis_span)
    stable_feedback_values = _stable_tail_values(feedback_records, axis=axis, axis_span=axis_span)
    stable_feedback_value = median(stable_feedback_values)
    stable_grid_feedback_value = _stable_grid_feedback_value(
        grid_records,
        feedback_records,
        axis=axis,
        axis_span=axis_span,
    )
    prior_grid_feedback_value = _prior_grid_feedback_value(
        grid_records,
        feedback_records,
        axis=axis,
        axis_span=axis_span,
    )
    grid_reference_value = (
        _grid_reference_when_later_stages_drift(
            grid_records,
            feedback_records,
            amp_records,
            axis=axis,
            axis_span=axis_span,
        )
        if prefer_grid_on_later_stage_drift
        else None
    )

    if grid_reference_value is not None and stable_feedback_value is not None:
        selected = feedback_records[-1]
        selected_value = stable_feedback_value
        reason = "stable_feedback_later_stage_drift"
    elif grid_reference_value is not None:
        selected = grid_records[-1]
        selected_value = grid_reference_value
        reason = "grid_reference_later_stage_drift"
    elif stable_amp_values:
        selected = amp_records[-1]
        selected_value = median(stable_amp_values)
        reason = "stable_amplifier_tail"
    elif len(amp_values) >= 2 and use_unstable_amplifier_median:
        selected = amp_records[-1]
        selected_value = median(amp_values)
        reason = "amplifier_sequence_median"
        if (
            stable_grid_feedback_value is not None
            and selected_value is not None
            and not trust_readable_amplifier_over_full_view
            and _relative_distance(selected_value, stable_grid_feedback_value, axis_span) > AMPLIFIER_UNSTABLE_DRIFT_TOLERANCE
        ):
            selected = feedback_records[-1] if feedback_records else grid_records[-1]
            selected_value = stable_grid_feedback_value
            reason = "stable_grid_feedback_unstable_amplifier_outlier"
    elif len(amp_values) == 1:
        selected = amp_records[-1]
        selected_value = amp_values[0]
        reason = "latest_readable_amplifier"

    if (
        selected is not None
        and selected_value is not None
        and reason in {"stable_amplifier_tail", "amplifier_sequence_median", "latest_readable_amplifier"}
        and stable_grid_feedback_value is not None
        and not trust_readable_amplifier_over_full_view
    ):
        drift_from_full_view = _relative_distance(selected_value, stable_grid_feedback_value, axis_span)
        should_keep_full_view = False
        risk_reason = ""
        exact_full_view_value = _exact_grid_feedback_value(
            grid_records,
            feedback_records,
            axis=axis,
        )
        full_view_confirmation_value = stable_grid_feedback_value
        if full_view_confirmation_value is None:
            full_view_confirmation_value = stable_feedback_value
        exact_full_view_drift = (
            exact_full_view_value is not None
            and abs(selected_value - exact_full_view_value) > PREDICTION_ABSOLUTE_STABILITY_TOLERANCE
        )
        stable_feedback_drift = (
            full_view_confirmation_value is not None
            and _relative_distance(selected_value, full_view_confirmation_value, axis_span)
            > AMPLIFIER_CONFIRMATION_DRIFT_TOLERANCE
        )
        if exact_full_view_drift:
            should_keep_full_view = True
            risk_reason = "exact_grid_feedback_amplifier_drift"
            full_view_confirmation_value = exact_full_view_value
        elif stable_feedback_drift:
            should_keep_full_view = True
            risk_reason = "stable_full_view_amplifier_confirmation_drift"
        elif reason == "latest_readable_amplifier" and drift_from_full_view > AMPLIFIER_WEAK_EVIDENCE_DRIFT_TOLERANCE:
            should_keep_full_view = True
            risk_reason = "stable_grid_feedback_single_amplifier_drift"
        elif (
            _moves_away_from_zero(selected_value, stable_grid_feedback_value)
            and drift_from_full_view > AMPLIFIER_CONTEXT_DRIFT_TOLERANCE
        ):
            should_keep_full_view = True
            risk_reason = "stable_grid_feedback_amplifier_away_from_zero_drift"
        elif (
            not prefer_grid_on_later_stage_drift
            and _has_later_unreadable_amplifier(full_records, axis=axis)
            and reason != "stable_amplifier_tail"
        ):
            should_keep_full_view = True
            risk_reason = "stable_grid_feedback_after_unreadable_amplifier_drift"

        if should_keep_full_view:
            selected = feedback_records[-1] if feedback_records else grid_records[-1]
            selected_value = full_view_confirmation_value
            reason = risk_reason

    if (
        selected is not None
        and selected_value is not None
        and reason in {"stable_amplifier_tail", "amplifier_sequence_median", "latest_readable_amplifier"}
        and stable_grid_feedback_value is not None
        and not trust_readable_amplifier_over_full_view
        and _amplifier_is_large_outlier(selected_value, stable_grid_feedback_value, axis_span)
    ):
        selected = feedback_records[-1] if feedback_records else grid_records[-1]
        selected_value = stable_grid_feedback_value
        reason = "stable_grid_feedback_large_amplifier_outlier"
    elif (
        selected is not None
        and selected_value is not None
        and reason in {"stable_amplifier_tail", "amplifier_sequence_median", "latest_readable_amplifier"}
        and prior_grid_feedback_value is not None
        and not trust_readable_amplifier_over_full_view
        and _amplifier_is_large_outlier(selected_value, prior_grid_feedback_value, axis_span)
    ):
        selected = feedback_records[-1] if feedback_records else grid_records[-1]
        selected_value = prior_grid_feedback_value
        reason = "prior_grid_feedback_large_amplifier_outlier"
    elif (
        selected is not None
        and selected_value is not None
        and reason == "amplifier_sequence_median"
        and prior_grid_feedback_value is not None
        and not trust_readable_amplifier_over_full_view
        and not _value_between_full_view_range(selected_value, grid_records, feedback_records, axis=axis)
        and _relative_distance(selected_value, prior_grid_feedback_value, axis_span)
        > AMPLIFIER_UNSTABLE_DRIFT_TOLERANCE
    ):
        selected = feedback_records[-1] if feedback_records else grid_records[-1]
        selected_value = prior_grid_feedback_value
        reason = "prior_grid_feedback_unstable_amplifier_drift"

    if selected is None or selected_value is None:
        latest_amp = amp_records[-1] if amp_records else None
        latest_feedback = feedback_records[-1] if feedback_records else None
        latest_grid = grid_records[-1] if grid_records else None
        amp_value = bar_prediction_value(latest_amp, axis=axis) if latest_amp else None
        feedback_value = bar_prediction_value(latest_feedback, axis=axis) if latest_feedback else None
        grid_value = bar_prediction_value(latest_grid, axis=axis) if latest_grid else None

        references = [value for value in (feedback_value, grid_value) if value is not None]
        if amp_value is not None and any(value_consistent(amp_value, ref, axis_span) for ref in references):
            selected = latest_amp
            selected_value = amp_value
            reason = "amplifier_agrees_with_grid_or_feedback"
        elif (
            feedback_value is not None
            and grid_value is not None
            and value_consistent(feedback_value, grid_value, axis_span)
        ):
            selected = latest_feedback
            selected_value = median([feedback_value, grid_value])
            reason = "stable_grid_feedback"
        elif amp_records and not use_unstable_amplifier_median and prior_grid_feedback_value is not None:
            selected = latest_feedback or latest_grid
            selected_value = prior_grid_feedback_value
            reason = "prior_grid_feedback_unstable_amplifier_disabled"
        elif len(amp_records) >= 2 and use_unstable_amplifier_median and not references:
            selected = amp_records[-1]
            selected_value = median(
                [
                    bar_prediction_value(record, axis=axis)
                    for record in amp_records
                    if bar_prediction_value(record, axis=axis) is not None
                ]
            )
            reason = "amplifier_median_without_grid_feedback_reference"
        else:
            readable = [
                (record, value)
                for record in full_records
                for value in [bar_prediction_value(record, axis=axis)]
                if value is not None and _value_inside_axis_bounds(value, axis_bounds)
            ]
            values = [value for _, value in readable if value is not None]
            if values:
                selected_value = median(values)
                target = selected_value
                selected = min(
                    readable,
                    key=lambda item: abs(float(item[1]) - float(target)) if item[1] is not None else float("inf"),
                )[0]
                reason = "median_all_full_flow"

    if selected is None or selected_value is None:
        return None, {"reason": reason, "readable_full_flow_count": 0}

    output = dict(selected)
    output[f"pred_{axis}"] = selected_value
    _update_axis_metrics(output, axis=axis, selected_value=selected_value, axis_ticks=axis_ticks)
    output["prompt_type"] = "full_flow_final"
    output["image_type"] = "selected"
    output["run"] = "final"
    output["call_id"] = None
    output["selection_source_prompt_type"] = selected.get("prompt_type")
    output["selection_source_run"] = selected.get("run")
    output["selection_reason"] = reason
    return output, {
        "reason": reason,
        "source_prompt_type": selected.get("prompt_type"),
        "source_run": selected.get("run"),
        "readable_full_flow_count": sum(
            1 for record in full_records if bar_prediction_value(record, axis=axis) is not None
        ),
        "consistency_tolerance": get_prediction_consistency_tolerance(),
    }


def write_bar_full_flow_selection(
    *,
    records: list[dict[str, Any]],
    result_dir: Path,
    axis: str,
    axis_ticks: list[Any] | None,
    use_unstable_amplifier_median: bool = True,
    prefer_grid_on_later_stage_drift: bool = False,
    trust_readable_amplifier_over_full_view: bool = False,
) -> list[dict[str, Any]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_point.setdefault(str(record.get("point")), []).append(record)

    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for point, point_records in by_point.items():
        selected, audit = select_bar_full_flow_record(
            point_records,
            axis=axis,
            axis_ticks=axis_ticks,
            use_unstable_amplifier_median=use_unstable_amplifier_median,
            prefer_grid_on_later_stage_drift=prefer_grid_on_later_stage_drift,
            trust_readable_amplifier_over_full_view=trust_readable_amplifier_over_full_view,
        )
        audit_rows.append({"point": point, **audit})
        if selected is not None:
            selected_rows.append(selected)

    ensure_dir(result_dir)
    _write_rows(result_dir / "full_flow_final_predictions.csv", selected_rows)
    _write_rows(result_dir / "full_flow_final_selection_audit.csv", audit_rows)
    return selected_rows


def write_xy_full_flow_selection(
    *,
    records: list[dict[str, Any]],
    result_dir: Path,
    x_ticks: list[Any] | None,
    y_ticks: list[Any] | None,
) -> list[dict[str, Any]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        point = record.get("point") or record.get("point_name")
        by_point.setdefault(str(point), []).append(record)

    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    x_span = numeric_axis_span(x_ticks)
    y_span = numeric_axis_span(y_ticks)
    for point, point_records in by_point.items():
        selected, audit = _select_xy_record(point_records, x_span=x_span, y_span=y_span)
        audit_rows.append({"point": point, **audit})
        if selected is not None:
            selected_rows.append(selected)

    ensure_dir(result_dir)
    _write_rows(result_dir / "full_flow_final_predictions.csv", selected_rows)
    _write_rows(result_dir / "full_flow_final_selection_audit.csv", audit_rows)
    return selected_rows


def _stable_tail_values(
    records: list[dict[str, Any]],
    *,
    axis: str,
    axis_span: float | None,
) -> list[float]:
    if len(records) < 2:
        return []
    values = [bar_prediction_value(record, axis=axis) for record in records]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return []
    if value_consistent(values[-1], values[-2], axis_span):
        return values[-2:]
    return []


def _stable_grid_feedback_value(
    grid_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    *,
    axis: str,
    axis_span: float | None,
) -> float | None:
    if not grid_records or not feedback_records:
        return None
    grid_value = bar_prediction_value(grid_records[-1], axis=axis)
    feedback_value = bar_prediction_value(feedback_records[-1], axis=axis)
    if grid_value is None or feedback_value is None:
        return None
    if not _values_consistent_or_close(grid_value, feedback_value, axis_span):
        return None
    return median([grid_value, feedback_value])


def _exact_grid_feedback_value(
    grid_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    *,
    axis: str,
) -> float | None:
    if not grid_records or not feedback_records:
        return None
    grid_value = bar_prediction_value(grid_records[-1], axis=axis)
    feedback_value = bar_prediction_value(feedback_records[-1], axis=axis)
    if grid_value is None or feedback_value is None:
        return None
    if abs(grid_value - feedback_value) <= PREDICTION_ABSOLUTE_STABILITY_TOLERANCE:
        return median([grid_value, feedback_value])
    return None


def _prior_grid_feedback_value(
    grid_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    *,
    axis: str,
    axis_span: float | None,
) -> float | None:
    grid_value = bar_prediction_value(grid_records[-1], axis=axis) if grid_records else None
    if grid_value is None:
        values = [
            value
            for record in feedback_records
            for value in [bar_prediction_value(record, axis=axis)]
            if value is not None
        ]
        return median(values)
    values = [grid_value]
    for record in feedback_records:
        value = bar_prediction_value(record, axis=axis)
        if value is None:
            continue
        if not _amplifier_is_large_outlier(value, grid_value, axis_span):
            values.append(value)
    return median(values)


def _grid_reference_when_later_stages_drift(
    grid_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    amp_records: list[dict[str, Any]],
    *,
    axis: str,
    axis_span: float | None,
) -> float | None:
    """Keep the full-grid reading when later local evidence jumps to another region.

    This is only enabled by chart modules that are prone to grouped-object
    context loss.  It does not use GT or baseline: the trigger is a large
    disagreement between the first full-grid prediction and later feedback /
    amplifier predictions.
    """
    if not grid_records:
        return None
    grid_value = bar_prediction_value(grid_records[-1], axis=axis)
    if grid_value is None:
        return None

    later_records = feedback_records + amp_records
    later_values = [
        value
        for record in later_records
        for value in [bar_prediction_value(record, axis=axis)]
        if value is not None
    ]
    if not later_values:
        return None

    latest_later = later_values[-1]
    if _relative_distance(latest_later, grid_value, axis_span) <= AMPLIFIER_OUTLIER_TOLERANCE:
        return None

    if len(later_values) == 1:
        return grid_value

    later_reference = median(later_values[-2:])
    if later_reference is None:
        return None
    later_is_self_consistent = _values_consistent_or_close(later_values[-1], later_values[-2], axis_span)
    later_far_from_grid = _relative_distance(later_reference, grid_value, axis_span) > AMPLIFIER_OUTLIER_TOLERANCE
    return grid_value if later_is_self_consistent and later_far_from_grid else None


def _has_later_unreadable_amplifier(records: list[dict[str, Any]], *, axis: str) -> bool:
    saw_readable = False
    for record in records:
        if record.get("prompt_type") != "amplifier":
            continue
        if bar_prediction_value(record, axis=axis) is None:
            if saw_readable:
                return True
            continue
        saw_readable = True
    return False


def _moves_away_from_zero(current: float, reference: float) -> bool:
    if current == 0 or reference == 0:
        return abs(current) > abs(reference)
    if (current < 0 < reference) or (reference < 0 < current):
        return True
    return abs(current) > abs(reference)


def _relative_distance(current: float, reference: float, axis_span: float | None) -> float:
    scale = axis_span if axis_span is not None and axis_span > 0 else max(abs(current), abs(reference), 1.0)
    return abs(current - reference) / scale


def _amplifier_is_large_outlier(current: float, reference: float, axis_span: float | None) -> bool:
    if _relative_distance(current, reference, axis_span) > AMPLIFIER_LARGE_OUTLIER_TOLERANCE:
        return True
    if current == 0 or reference == 0:
        return False
    if (current < 0 < reference) or (reference < 0 < current):
        scale = axis_span if axis_span is not None and axis_span > 0 else max(abs(current), abs(reference), 1.0)
        return min(abs(current), abs(reference)) / scale > 0.02
    return False


def _values_consistent_or_close(current: float, reference: float, axis_span: float | None) -> bool:
    if value_consistent(current, reference, axis_span):
        return True
    if _relative_distance(current, reference, axis_span) <= max(
        get_prediction_consistency_tolerance(),
        FULL_VIEW_SELECTION_STABILITY_TOLERANCE,
    ):
        return True
    return abs(current - reference) <= PREDICTION_ABSOLUTE_STABILITY_TOLERANCE


def _value_between_full_view_range(
    value: float,
    grid_records: list[dict[str, Any]],
    feedback_records: list[dict[str, Any]],
    *,
    axis: str,
) -> bool:
    full_view_values = [
        full_value
        for record in (grid_records + feedback_records)
        for full_value in [bar_prediction_value(record, axis=axis)]
        if full_value is not None
    ]
    if len(full_view_values) < 2:
        return False
    low = min(full_view_values)
    high = max(full_view_values)
    padding = max((high - low) * 0.05, PREDICTION_ABSOLUTE_STABILITY_TOLERANCE)
    return (low - padding) <= value <= (high + padding)


def _numeric_axis_bounds(ticks: list[Any] | None) -> tuple[float, float] | None:
    values: list[float] = []
    for tick in ticks or []:
        try:
            values.append(float(tick))
        except (TypeError, ValueError):
            continue
    if len(values) < 2:
        return None
    low = min(values)
    high = max(values)
    span = high - low
    if span <= 0:
        return None
    diffs = sorted(
        abs(right - left)
        for left, right in zip(sorted(values), sorted(values)[1:])
        if abs(right - left) > 0
    )
    median_step = diffs[len(diffs) // 2] if diffs else 0.0
    pad = max(span * 0.25, median_step * 2.0)
    return low - pad, high + pad


def _value_inside_axis_bounds(value: float, bounds: tuple[float, float] | None) -> bool:
    if bounds is None:
        return True
    return bounds[0] <= value <= bounds[1]


def _xy_value(record: dict[str, Any]) -> tuple[float, float] | None:
    if str(record.get("prediction_readable", "")).strip().lower() == "false":
        return None
    x_value = finite_float(record.get("pred_x"))
    y_value = finite_float(record.get("pred_y"))
    if x_value is None or y_value is None:
        return None
    return x_value, y_value


def _select_xy_record(
    records: list[dict[str, Any]],
    *,
    x_span: float | None,
    y_span: float | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    full_records = [record for record in records if record.get("prompt_type") in POINT_FULL_FLOW_STAGES]
    readable = [(record, _xy_value(record)) for record in full_records]
    readable = [(record, value) for record, value in readable if value is not None]
    if not readable:
        return None, {"reason": "no_readable_full_flow_prediction", "readable_full_flow_count": 0}

    if len(readable) >= 3:
        selected_value = _xy_median([value for _, value in readable])
        selected = min(
            readable,
            key=lambda item: abs(item[1][0] - selected_value[0]) + abs(item[1][1] - selected_value[1]),
        )[0]
        output = dict(selected)
        output["pred_x"] = selected_value[0]
        output["pred_y"] = selected_value[1]
        output["prompt_type"] = "full_flow_final"
        output["image_type"] = "selected"
        output["run"] = "final"
        output["call_id"] = None
        output["selection_source_prompt_type"] = selected.get("prompt_type")
        output["selection_source_run"] = selected.get("run")
        output["selection_reason"] = "full_flow_consensus_median"
        return output, {
            "reason": "full_flow_consensus_median",
            "source_prompt_type": selected.get("prompt_type"),
            "source_run": selected.get("run"),
            "readable_full_flow_count": len(readable),
            "consistency_tolerance": get_prediction_consistency_tolerance(),
        }

    amp_records = [
        (record, value)
        for record, value in readable
        if record.get("prompt_type") in {"feedback_crop_adaptive", "amplifier"}
    ]
    feedback_records = [(record, value) for record, value in readable if record.get("prompt_type") == "feedback"]
    grid_records = [(record, value) for record, value in readable if record.get("prompt_type") == "grid"]

    selected: dict[str, Any]
    selected_value: tuple[float, float]
    reason: str
    if len(amp_records) >= 2 and _xy_consistent(amp_records[-1][1], amp_records[-2][1], x_span, y_span):
        selected = amp_records[-1][0]
        selected_value = _xy_median([amp_records[-2][1], amp_records[-1][1]])
        reason = "stable_amplifier_tail"
    elif len(amp_records) >= 2:
        selected = amp_records[-1][0]
        selected_value = _xy_median([value for _, value in amp_records])
        reason = "amplifier_sequence_median"
    elif len(amp_records) == 1:
        selected = amp_records[-1][0]
        selected_value = amp_records[-1][1]
        reason = "latest_readable_amplifier"
    else:
        latest_amp = amp_records[-1] if amp_records else None
        latest_feedback = feedback_records[-1] if feedback_records else None
        latest_grid = grid_records[-1] if grid_records else None
        references = [item[1] for item in (latest_feedback, latest_grid) if item is not None]
        if latest_amp is not None and any(_xy_consistent(latest_amp[1], ref, x_span, y_span) for ref in references):
            selected = latest_amp[0]
            selected_value = latest_amp[1]
            reason = "amplifier_agrees_with_grid_or_feedback"
        elif latest_feedback is not None and latest_grid is not None and _xy_consistent(
            latest_feedback[1],
            latest_grid[1],
            x_span,
            y_span,
        ):
            selected = latest_feedback[0]
            selected_value = _xy_median([latest_feedback[1], latest_grid[1]])
            reason = "stable_grid_feedback"
        elif len(amp_records) >= 2 and not references:
            selected = amp_records[-1][0]
            selected_value = _xy_median([value for _, value in amp_records])
            reason = "amplifier_median_without_grid_feedback_reference"
        else:
            selected_value = _xy_median([value for _, value in readable])
            selected = min(
                readable,
                key=lambda item: abs(item[1][0] - selected_value[0]) + abs(item[1][1] - selected_value[1]),
            )[0]
            reason = "median_all_full_flow"

    output = dict(selected)
    output["pred_x"] = selected_value[0]
    output["pred_y"] = selected_value[1]
    output["prompt_type"] = "full_flow_final"
    output["image_type"] = "selected"
    output["run"] = "final"
    output["call_id"] = None
    output["selection_source_prompt_type"] = selected.get("prompt_type")
    output["selection_source_run"] = selected.get("run")
    output["selection_reason"] = reason
    return output, {
        "reason": reason,
        "source_prompt_type": selected.get("prompt_type"),
        "source_run": selected.get("run"),
        "readable_full_flow_count": len(readable),
        "consistency_tolerance": get_prediction_consistency_tolerance(),
    }


def _xy_consistent(
    current: tuple[float, float],
    reference: tuple[float, float],
    x_span: float | None,
    y_span: float | None,
) -> bool:
    return value_consistent(current[0], reference[0], x_span) and value_consistent(current[1], reference[1], y_span)


def _xy_median(values: list[tuple[float, float]]) -> tuple[float, float]:
    return float(median([value[0] for value in values]) or 0.0), float(median([value[1] for value in values]) or 0.0)


def _update_axis_metrics(
    record: dict[str, Any],
    *,
    axis: str,
    selected_value: float,
    axis_ticks: list[Any] | None,
) -> None:
    gt_value = finite_float(record.get(f"gt_{axis}"))
    if gt_value is None:
        record["mae"] = None
        record[f"{axis}_re"] = None
        return
    absolute_error = abs(selected_value - gt_value)
    record["mae"] = round(absolute_error, 4)
    record[f"{axis}_re"] = None if gt_value == 0 else round(absolute_error / abs(gt_value), 4)
    span = numeric_axis_span(axis_ticks)
    if span:
        record[f"{axis}_rne"] = absolute_error / span


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
