"""Modular runner for vertical bar value prediction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor

from gemini_calls import get_last_modal_call_id, reset_modal_call_context, set_modal_call_context

from ...common.runtime import (
    get_bar_amplifier_rounds,
    get_prompt_rounds,
    get_repeat_times,
    numeric_axis_span,
    value_consistent,
)
from ...common.full_flow_selection import write_bar_full_flow_selection
from ...common.stacked_bar_geometry import stacked_segment_prior

from .data import VBarTarget, image_path, iter_targets, load_datasets
from .evaluation import compute_mae, compute_relative_error, save_results
from .geometry import category_pixel, category_span, numeric_pixel
from .model import VBarModelClient
from .prompts import generate_prompt
from .visual import chart_result_dir, crop_bar_window, draw_prediction_overlay


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("amplifier", "grid_with_grid"),
]

PREFERRED_PROMPTS = ["full_flow_final", "geometry", "amplifier", "feedback", "grid"]


def _valid_prediction(pred: tuple[Any, Any]) -> bool:
    try:
        float(pred[1])
        return True
    except Exception:
        return False


def _fallback_numeric_center(ticks: list[Any]) -> float | None:
    values: list[float] = []
    for tick in ticks:
        try:
            values.append(float(tick))
        except Exception:
            continue
    if not values:
        return None
    values.sort()
    return (values[0] + values[-1]) / 2


def _prediction_inside_y_axis(pred: tuple[Any, Any], y_ticks: list[Any]) -> bool:
    if not _valid_prediction(pred):
        return False
    values: list[float] = []
    for tick in y_ticks:
        try:
            values.append(float(tick))
        except Exception:
            continue
    if len(values) < 2:
        return True
    low = min(values)
    high = max(values)
    span = high - low
    pad = span * 0.1
    value = float(pred[1])
    return low - pad <= value <= high + pad


def _fallback_numeric_step(ticks: list[Any]) -> float:
    values: list[float] = []
    for tick in ticks:
        try:
            values.append(float(tick))
        except Exception:
            continue
    values = sorted(set(values))
    gaps = [abs(right - left) for left, right in zip(values, values[1:]) if right != left]
    if gaps:
        return sorted(gaps)[len(gaps) // 2]
    if len(values) >= 2:
        return abs(values[-1] - values[0]) / 4
    return 1.0


def _scan_offsets(max_attempts: int) -> list[int]:
    offsets = [0]
    for index in range(1, max_attempts):
        step = (index + 1) // 2
        offsets.append(-step if index % 2 == 1 else step)
    return offsets


def _series_rgb(dataset: dict[str, Any], series_name: str) -> tuple[int, int, int] | None:
    series_color = dataset.get("series_color")
    value = series_color.get(series_name) if isinstance(series_color, dict) else None
    if not isinstance(value, str) or not value.strip():
        if isinstance(series_color, dict) and len(series_color) == 1:
            value = next(iter(series_color.values()))
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return tuple(ImageColor.getrgb(value.strip())[:3])
    except Exception:
        return None


def _is_grouped_bar(dataset: dict[str, Any]) -> bool:
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict) and len(series_color) > 1:
        return True
    series_names = dataset.get("series_names")
    return isinstance(series_names, list) and len(series_names) > 1


def _has_repeated_grouped_bar_colors(dataset: dict[str, Any]) -> bool:
    series_color = dataset.get("series_color")
    if not isinstance(series_color, dict) or len(series_color) <= 1:
        return False
    colors = [str(value).strip().lower() for value in series_color.values() if str(value).strip()]
    return len(colors) != len(set(colors))


def _series_adjusted_category_band(
    *,
    dataset: dict[str, Any],
    target: VBarTarget,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    x_center = category_pixel(target.x_label, dataset["x_ticks"], dataset["x_pixels"])
    span = category_span(target.x_label, dataset["x_ticks"], dataset["x_pixels"], image_size)
    series_color = dataset.get("series_color")
    order = list(series_color.keys()) if isinstance(series_color, dict) else []
    if target.series_name in order and len(order) > 1:
        slot = max(8.0, float(span) / len(order))
        index = order.index(target.series_name)
        x_center = int(round(x_center - ((len(order) - 1) / 2.0 - index) * slot))
        return x_center, max(8, int(round(slot / 2.0 + 5)))
    return x_center, max(3, span // 2)


def _crop_has_target_color_and_edge(
    *,
    dataset: dict[str, Any],
    target: VBarTarget,
    visible_range: tuple[float, float],
) -> bool:
    rgb = _series_rgb(dataset, target.series_name)
    if rgb is None:
        return True
    try:
        source = Image.open(image_path(dataset, "no_grid")).convert("RGB")
        arr = np.asarray(source, dtype=np.int16)
    except Exception:
        return True
    try:
        x_center, half_x = _series_adjusted_category_band(
            dataset=dataset,
            target=target,
            image_size=source.size,
        )
        y0 = numeric_pixel(float(visible_range[0]), dataset["y_ticks"], dataset["y_pixels"])
        y1 = numeric_pixel(float(visible_range[1]), dataset["y_ticks"], dataset["y_pixels"])
    except Exception:
        return True
    top, bottom = sorted((int(y0), int(y1)))
    height, width = arr.shape[:2]
    left = max(0, int(x_center) - half_x)
    right = min(width, int(x_center) + half_x + 1)
    top = max(0, min(height - 1, top))
    bottom = max(0, min(height - 1, bottom))
    if bottom <= top or right <= left:
        return True

    band = arr[top : bottom + 1, left:right, :]
    target_rgb = np.array(rgb, dtype=np.int16)
    dist = np.sqrt(np.sum((band - target_rgb) ** 2, axis=2))
    mask = dist <= 42
    if int(np.count_nonzero(mask)) < 8:
        return False
    edge_height = min(4, mask.shape[0])
    top_edge_count = int(np.count_nonzero(mask[:edge_height, :]))
    edge_threshold = max(1, int(mask.shape[1] * 0.02))
    return top_edge_count < edge_threshold


async def _crop_until_bar_detected(
    *,
    client: VBarModelClient,
    dataset: dict[str, Any],
    target: VBarTarget,
    center_value: float,
    round_index: int,
    zoom_round_index: int | None = None,
    roi_scale: float = 1.0,
    max_attempts: int = 8,
    geometry_verified: bool = False,
    segment_pixel_span: int | None = None,
) -> tuple[Path, list[float], tuple[float, float]] | None:
    chart_type = str(dataset.get("chart_type") or "v_bar")
    value_span = _fallback_numeric_step(dataset["y_ticks"])

    scan_attempts = max(max_attempts, len(dataset.get("y_ticks", []) or []) * 2 + 1)
    for attempt_index, offset_units in enumerate(_scan_offsets(scan_attempts)):
        shifted_center = center_value + offset_units * value_span
        try:
            crop_path, visible_ticks, visible_range = crop_bar_window(
                chart_id=dataset["chart_id"],
                image_path=image_path(dataset, "no_grid"),
                point_name=target.point_name,
                x_label=target.x_label,
                center_value=shifted_center,
                x_ticks=dataset["x_ticks"],
                x_pixels=dataset["x_pixels"],
                y_ticks=dataset["y_ticks"],
                y_pixels=dataset["y_pixels"],
                round_index=round_index,
                zoom_round_index=zoom_round_index,
                roi_scale=roi_scale,
                attempt_index=attempt_index,
                pad_y=int(segment_pixel_span / 2 + 24) if segment_pixel_span else None,
                chart_type=chart_type,
                series_name=target.series_name,
                series_order=list(dataset.get("series_color", {}).keys())
                if isinstance(dataset.get("series_color"), dict)
                else None,
            )
        except Exception as exc:
            print(f"[v_bar runner] Amplifier crop attempt {attempt_index} failed: {exc}")
            continue

        span = abs(float(visible_range[1]) - float(visible_range[0]))
        if span > 0:
            value_span = span
        if not _crop_has_target_color_and_edge(dataset=dataset, target=target, visible_range=visible_range):
            print(
                f"[v_bar runner] amplifier crop attempt={attempt_index} "
                f"center={shifted_center:.4f} does not contain a visible target top edge; continue scanning."
            )
            continue
        print(
            f"[v_bar runner] amplifier crop attempt={attempt_index} "
            f"center={shifted_center:.4f} range={visible_range}"
        )
        return crop_path, visible_ticks, visible_range
    return None


def _record(
    *,
    dataset: dict[str, Any],
    target: VBarTarget,
    prompt_type: str,
    image_type: str,
    run: int,
    used_image_path: Path,
    pred: tuple[Any, Any],
    call_id: str | None = None,
) -> dict[str, Any]:
    pred_x, pred_y = pred
    prediction_readable = _valid_prediction(pred)
    mae = compute_mae(pred_y, target.gt_y)
    y_re = compute_relative_error(pred_y, target.gt_y)
    return {
        "chart_id": dataset["chart_id"],
        "call_id": call_id,
        "point": target.point_name,
        "series_name": target.series_name,
        "prompt_type": prompt_type,
        "image_type": image_type,
        "run": run,
        "image_path": str(used_image_path),
        "gt_x": target.gt_x,
        "gt_y": target.gt_y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "prediction_readable": prediction_readable,
        "mae": mae,
        "pixel_rel_x": -1,
        "pixel_rel_y": -1,
        "x_re": -1,
        "y_re": y_re,
    }


def _prediction_value(record: dict[str, Any]) -> float | None:
    if str(record.get("prediction_readable", "")).strip().lower() == "false":
        return None
    try:
        value = float(record["pred_y"])
    except Exception:
        return None
    return value


def _select_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if _prediction_value(record) is not None:
            by_point.setdefault(str(record["point"]), []).append(record)

    predictions: list[dict[str, Any]] = []
    for point, point_records in by_point.items():
        chosen = None
        for prompt_type in PREFERRED_PROMPTS:
            candidates = [record for record in point_records if record["prompt_type"] == prompt_type]
            if candidates:
                chosen = candidates[-1]
                break
        if chosen is None:
            continue
        predictions.append(
            {
                "id": point,
                "series_name": chosen.get("series_name"),
                "label": chosen.get("pred_x"),
                "axis": "y",
                "value": _prediction_value(chosen),
                "prompt_type": chosen.get("prompt_type"),
                "image_type": chosen.get("image_type"),
                "image_path": chosen.get("image_path"),
            }
        )
    return predictions


async def _run_target(
    *,
    client: VBarModelClient,
    dataset: dict[str, Any],
    target: VBarTarget,
    repeat_times: int,
    amplifier_rounds: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    history: list[tuple[Any, Any]] = []
    feedback_history: list[tuple[Any, Any]] = []
    feedback_pred: tuple[Any, Any] | None = None
    chart_type = str(dataset.get("chart_type") or "v_bar")
    segment_prior = stacked_segment_prior(
        dataset,
        series_name=target.series_name,
        category_label=target.x_label,
        orientation="v",
    )
    stacked_start_value = segment_prior.start_value if segment_prior is not None else None
    if segment_prior is not None:
        records.append(
            _record(
                dataset=dataset,
                target=target,
                prompt_type="geometry",
                image_type="no_grid",
                run=1,
                used_image_path=image_path(dataset, "no_grid"),
                pred=(target.gt_x, segment_prior.segment_value),
            )
        )

    for prompt_type, image_type in EXPERIMENT_TYPES:
        if prompt_type == "amplifier":
            center_value = segment_prior.center_value if segment_prior is not None else None
            if center_value is None:
                center_value = (
                    float(feedback_pred[1])
                    if feedback_pred and _valid_prediction(feedback_pred)
                    else _fallback_numeric_center(dataset["y_ticks"])
                )

            if center_value is None:
                print("[v_bar runner] Skip amplifier crop: no numeric center is available.")
                continue

            previous_amp_pred = feedback_pred
            valid_amp_count = 0
            expand_next_crop = False
            for amp_round in range(1, amplifier_rounds + 1):
                crop_zoom_round = max(1, valid_amp_count + 1)
                if expand_next_crop:
                    crop_zoom_round = max(1, valid_amp_count)
                crop_result = await _crop_until_bar_detected(
                    client=client,
                    dataset=dataset,
                    target=target,
                    center_value=float(center_value),
                    round_index=amp_round,
                    zoom_round_index=crop_zoom_round,
                    roi_scale=2.0 if expand_next_crop else 1.0,
                    geometry_verified=segment_prior is not None,
                    segment_pixel_span=(
                        abs(segment_prior.end_pixel - segment_prior.start_pixel)
                        if segment_prior is not None
                        else None
                    ),
                )
                if crop_result is None:
                    print(
                        f"[v_bar runner] Stop amplifier refinement at round {amp_round}: "
                        "no crop contains the target bar."
                    )
                    break

                used_image, visible_ticks, _ = crop_result
                prompt = generate_prompt(
                    item_name=target.point_name,
                    prompt_type=prompt_type,
                    x_ticks=dataset["x_ticks"],
                    y_ticks=dataset["y_ticks"],
                    series_color=dataset["series_color"],
                    x_pixels=dataset["x_pixels"],
                    y_pixels=dataset["y_pixels"],
                    visible_ticks=visible_ticks,
                    pred_feedback=history[-2:] if history else None,
                    feedback_round=2,
                    current_round=amp_round,
                    chart_type=chart_type,
                )

                print("\n==============================")
                print(
                    f"[v_bar] Amplifier Round {amp_round}/{amplifier_rounds} "
                    f"| Point: {target.point_name} | Image: {used_image}"
                )
                print("==============================\n")

                token = set_modal_call_context(
                    {
                        "chart_name": dataset.get("chart_id"),
                        "processing_object": target.point_name,
                        "object_category": target.series_name,
                        "gt": {"x": target.gt_x, "y": target.gt_y},
                        "stage": prompt_type,
                        "round": amp_round,
                        "image_path": str(used_image),
                    }
                )
                try:
                    pred = await client.predict_coords(prompt, used_image, target.point_name)
                    call_id = get_last_modal_call_id()
                finally:
                    reset_modal_call_context(token)
                if _valid_prediction(pred):
                    pred = (target.x_label, pred[1])
                records.append(
                    _record(
                        dataset=dataset,
                        target=target,
                        prompt_type=prompt_type,
                        image_type=image_type,
                        run=amp_round,
                        used_image_path=used_image,
                        pred=pred,
                        call_id=call_id,
                    )
                )
                if not _valid_prediction(pred):
                    expand_next_crop = True
                    print(
                        f"[v_bar] Amplifier round {amp_round} reported target not readable @ {target.point_name}: {pred}; "
                        "next amplifier round will expand ROI."
                    )
                    continue

                is_stable = (
                    previous_amp_pred is not None
                    and _valid_prediction(previous_amp_pred)
                    and value_consistent(
                        pred[1],
                        previous_amp_pred[1],
                        numeric_axis_span(dataset.get("y_ticks")),
                    )
                )
                history.append(pred)
                valid_amp_count += 1
                expand_next_crop = False
                center_value = segment_prior.center_value if segment_prior is not None else float(pred[1])
                previous_amp_pred = pred
                if is_stable:
                    print(f"[v_bar] Amplifier prediction stabilized at round {amp_round}; stop zoom-in refinement.")
                    break
                print(
                    f"[v_bar] Success amplifier {amp_round}/{amplifier_rounds} "
                    f"@ {target.point_name}; next center={center_value:.4f}"
                )
            continue

        prompt_rounds = get_prompt_rounds(prompt_type, repeat_times)
        for run_idx in range(1, prompt_rounds + 1):
            used_image = image_path(dataset, image_type)
            visible_ticks = None
            previous_pred = history[-1] if history else None

            if prompt_type == "feedback" and history and run_idx > 1:
                used_image = draw_prediction_overlay(
                    chart_id=dataset["chart_id"],
                    original_img_path=image_path(dataset, "grid_with_grid"),
                    pred_coords=history,
                    x_ticks=dataset["x_ticks"],
                    y_ticks=dataset["y_ticks"],
                    x_pixels=dataset["x_pixels"],
                    y_pixels=dataset["y_pixels"],
                    point_name=target.point_name,
                    draw_all_preds=False,
                    prompt_type=prompt_type,
                    image_type=image_type,
                    run_index=run_idx,
                    chart_type=chart_type,
                    stacked_start_value=stacked_start_value,
                )

            prompt = generate_prompt(
                item_name=target.point_name,
                prompt_type=prompt_type,
                x_ticks=dataset["x_ticks"],
                y_ticks=dataset["y_ticks"],
                series_color=dataset["series_color"],
                x_pixels=dataset["x_pixels"],
                y_pixels=dataset["y_pixels"],
                visible_ticks=visible_ticks,
                pred_feedback=history[-2:] if history else None,
                feedback_round=2,
                current_round=run_idx,
                chart_type=chart_type,
            )

            print("\n==============================")
            print(f"[v_bar] Round {run_idx} | Point: {target.point_name} | Type: {prompt_type} - {image_type}")
            print(f"[v_bar] Image: {used_image}")
            print("==============================\n")

            token = set_modal_call_context(
                {
                    "chart_name": dataset.get("chart_id"),
                    "processing_object": target.point_name,
                    "object_category": target.series_name,
                    "gt": {"x": target.gt_x, "y": target.gt_y},
                    "stage": prompt_type,
                    "round": run_idx,
                    "image_path": str(used_image),
                }
            )
            try:
                pred = await client.predict_coords(prompt, used_image, target.point_name)
                call_id = get_last_modal_call_id()
            finally:
                reset_modal_call_context(token)
            if _valid_prediction(pred):
                # Normalize x output to the requested category for stable records.
                pred = (target.x_label, pred[1])
            records.append(
                _record(
                    dataset=dataset,
                    target=target,
                    prompt_type=prompt_type,
                    image_type=image_type,
                    run=run_idx,
                    used_image_path=used_image,
                    pred=pred,
                    call_id=call_id,
                )
            )
            if _valid_prediction(pred):
                is_stable = (
                    prompt_type == "feedback"
                    and previous_pred is not None
                    and _valid_prediction(previous_pred)
                    and value_consistent(
                        pred[1],
                        previous_pred[1],
                        numeric_axis_span(dataset.get("y_ticks")),
                    )
                )
                if prompt_type != "baseline":
                    history.append(pred)
                if prompt_type == "feedback":
                    feedback_pred = pred
                    feedback_history.append(pred)
                    if is_stable:
                        print(f"[v_bar] Feedback prediction stabilized at round {run_idx}; stop feedback refinement.")
                        break
                print(f"[v_bar] Success {run_idx}/{repeat_times} [{prompt_type} - {image_type}] @ {target.point_name}")
            else:
                print(f"[v_bar] Invalid prediction [{prompt_type} - {image_type}] @ {target.point_name}: {pred}")

        if prompt_type == "feedback" and feedback_pred and _valid_prediction(feedback_pred):
            try:
                final_overlay_path = draw_prediction_overlay(
                    chart_id=dataset["chart_id"],
                    original_img_path=image_path(dataset, "grid_with_grid"),
                    pred_coords=feedback_history,
                    x_ticks=dataset["x_ticks"],
                    y_ticks=dataset["y_ticks"],
                    x_pixels=dataset["x_pixels"],
                    y_pixels=dataset["y_pixels"],
                    point_name=target.point_name,
                    draw_all_preds=True,
                    prompt_type=prompt_type,
                    image_type=image_type,
                    final_overlay=True,
                    chart_type=chart_type,
                    stacked_start_value=stacked_start_value,
                )
                print(f"[v_bar] Final feedback overlay saved: {final_overlay_path}")
            except Exception as exc:
                print(f"[v_bar] Final feedback overlay skipped for {target.point_name}: {exc}")

    return records


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
    chart_type: str = "v_bar",
) -> list[dict[str, Any]]:
    datasets = load_datasets(chart_ids, config_paths=config_paths, chart_type=chart_type)
    if not datasets:
        print("[v_bar] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times()
    amplifier_rounds = get_bar_amplifier_rounds()
    all_records: list[dict[str, Any]] = []
    async with VBarModelClient() as client:
        for start in range(0, len(datasets), batch_size or len(datasets)):
            batch = datasets[start : start + (batch_size or len(datasets))]
            tasks = [
                _run_target(
                    client=client,
                    dataset=dataset,
                    target=target,
                    repeat_times=repeat_times,
                    amplifier_rounds=amplifier_rounds,
                )
                for dataset in batch
                for target in iter_targets(dataset)
            ]
            for result in await asyncio.gather(*tasks):
                all_records.extend(result)

    if not all_records:
        print("[v_bar] No experiment records generated.")
        return []

    by_chart: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_chart.setdefault(record["chart_id"], []).append(record)
    dataset_by_chart = {str(dataset.get("chart_id")): dataset for dataset in datasets}

    summaries: list[dict[str, Any]] = []
    for chart_id, records in by_chart.items():
        result_dir = chart_result_dir(chart_id, chart_type)
        save_results(records, result_dir)
        dataset = dataset_by_chart.get(chart_id, {})
        repeated_grouped_colors = _has_repeated_grouped_bar_colors(dataset)
        final_records = write_bar_full_flow_selection(
            records=records,
            result_dir=result_dir,
            axis="y",
            axis_ticks=dataset.get("y_ticks"),
            use_unstable_amplifier_median=True,
            prefer_grid_on_later_stage_drift=not repeated_grouped_colors,
            trust_readable_amplifier_over_full_view=repeated_grouped_colors,
        )
        if not final_records:
            final_records = _select_final_input_records(records)
        print(f"[v_bar] Saved results for {chart_id}: {result_dir}")
        predictions = _select_predictions(final_records)
        summaries.append(
            {
                "chart_id": chart_id,
                "result_dir": str(result_dir),
                "record_count": len(records),
                "full_flow_final_count": len(final_records),
                "object_count": len(predictions),
                "predictions": predictions,
            }
        )
    return summaries


def _select_final_input_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("prompt_type") in {"amplifier", "feedback", "grid"}]
