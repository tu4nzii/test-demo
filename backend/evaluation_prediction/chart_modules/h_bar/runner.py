"""Modular runner for horizontal bar value prediction."""

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

from .data import HBarTarget, image_path, iter_targets, load_datasets
from .evaluation import compute_mae, compute_relative_error, save_results
from .geometry import category_pixel, category_span, numeric_pixel
from .model import HBarModelClient
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
        float(pred[0])
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


def _numeric_axis_bounds(ticks: list[Any]) -> tuple[float, float] | None:
    values: list[float] = []
    for tick in ticks:
        try:
            values.append(float(tick))
        except Exception:
            continue
    if len(values) < 2:
        return None
    return min(values), max(values)


def _value_from_axis_pixel(pixel_value: float, ticks: list[Any], pixels: list[Any]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for tick, pixel in zip(ticks, pixels):
        try:
            pairs.append((float(tick), float(pixel)))
        except Exception:
            continue
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda item: item[1])
    p_min, p_max = pairs[0][1], pairs[-1][1]
    v_min, v_max = pairs[0][0], pairs[-1][0]
    if p_max == p_min or not (min(p_min, p_max) <= pixel_value <= max(p_min, p_max)):
        return None
    return v_min + (pixel_value - p_min) * (v_max - v_min) / (p_max - p_min)


def _normalize_prediction_units(dataset: dict[str, Any], pred: tuple[Any, Any]) -> tuple[Any, Any]:
    if not _valid_prediction(pred):
        return pred
    try:
        pred_x = float(pred[0])
    except Exception:
        return pred
    bounds = _numeric_axis_bounds(dataset.get("x_ticks", []))
    if bounds is None:
        return pred
    low, high = bounds
    if low <= pred_x <= high:
        return pred
    converted = _value_from_axis_pixel(pred_x, dataset.get("x_ticks", []), dataset.get("x_pixels", []))
    if converted is None:
        return pred
    print(f"[h_bar runner] Converted pixel-like prediction x={pred_x:g} to data value {converted:.4g}.")
    return converted, pred[1]


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


def _is_single_series_bar(dataset: dict[str, Any], target: HBarTarget) -> bool:
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict):
        names = [str(name).strip().lower() for name in series_color if str(name).strip()]
        if len(names) <= 1:
            return True
        if str(target.series_name).strip().lower() in {"", "none", "series 1"} and len(names) == 1:
            return True
    series_names = dataset.get("series_names")
    if isinstance(series_names, list) and len(series_names) <= 1:
        return True
    return str(target.series_name).strip().lower() in {"", "none", "series 1"}


def _is_same_color_grouped_bar(dataset: dict[str, Any]) -> bool:
    series_color = dataset.get("series_color")
    if not isinstance(series_color, dict) or len(series_color) <= 1:
        return False
    colors = {str(value).strip().lower() for value in series_color.values() if str(value).strip()}
    return len(colors) == 1


def _has_repeated_grouped_bar_colors(dataset: dict[str, Any]) -> bool:
    series_color = dataset.get("series_color")
    if not isinstance(series_color, dict) or len(series_color) <= 1:
        return False
    colors = [str(value).strip().lower() for value in series_color.values() if str(value).strip()]
    return len(colors) != len(set(colors))


def _is_grouped_bar(dataset: dict[str, Any]) -> bool:
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict) and len(series_color) > 1:
        return True
    series_names = dataset.get("series_names")
    return isinstance(series_names, list) and len(series_names) > 1


def _series_adjusted_category_band(
    *,
    dataset: dict[str, Any],
    target: HBarTarget,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    y_center = category_pixel(target.y_label, dataset["y_ticks"], dataset["y_pixels"])
    span = category_span(target.y_label, dataset["y_ticks"], dataset["y_pixels"], image_size)
    series_color = dataset.get("series_color")
    order = list(series_color.keys()) if isinstance(series_color, dict) else []
    if target.series_name in order and len(order) > 1:
        slot = max(8.0, float(span) / len(order))
        index = order.index(target.series_name)
        y_center = int(round(y_center + ((len(order) - 1) / 2.0 - index) * slot))
        return y_center, max(12, int(round(slot / 2.0 + 7)))
    return y_center, max(3, span // 2)


def _crop_has_target_color_and_edge(
    *,
    dataset: dict[str, Any],
    target: HBarTarget,
    visible_range: tuple[float, float],
    center_value: float,
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
        y_center, half_y = _series_adjusted_category_band(dataset=dataset, target=target, image_size=source.size)
        x0 = numeric_pixel(float(visible_range[0]), dataset["x_ticks"], dataset["x_pixels"])
        x1 = numeric_pixel(float(visible_range[1]), dataset["x_ticks"], dataset["x_pixels"])
    except Exception:
        return True
    left, right = sorted((int(x0), int(x1)))
    height, width = arr.shape[:2]
    left = max(0, min(width - 1, left))
    right = max(0, min(width - 1, right))
    top = max(0, int(y_center) - half_y)
    bottom = min(height, int(y_center) + half_y + 1)
    if right <= left or bottom <= top:
        return True

    band = arr[top:bottom, left : right + 1, :]
    target_rgb = np.array(rgb, dtype=np.int16)
    dist = np.sqrt(np.sum((band - target_rgb) ** 2, axis=2))
    mask = dist <= 42
    if int(np.count_nonzero(mask)) < 8:
        return False
    edge_width = min(4, mask.shape[1])
    edge_slice = mask[:, :edge_width] if center_value < 0 else mask[:, -edge_width:]
    edge_count = int(np.count_nonzero(edge_slice))
    edge_threshold = max(1, int(mask.shape[0] * 0.02))
    return edge_count < edge_threshold


async def _crop_until_bar_detected(
    *,
    client: HBarModelClient,
    dataset: dict[str, Any],
    target: HBarTarget,
    center_value: float,
    round_index: int,
    zoom_round_index: int | None = None,
    roi_scale: float = 1.0,
    max_attempts: int = 8,
    geometry_verified: bool = False,
    segment_pixel_span: int | None = None,
) -> tuple[Path, list[float], tuple[float, float]] | None:
    chart_type = str(dataset.get("chart_type") or "h_bar")
    value_span = _fallback_numeric_step(dataset["x_ticks"])

    if _is_single_series_bar(dataset, target):
        # A single-series bar has one mark per category.  Horizontal scanning can
        # drift into another value interval with the same color, so keep the
        # seed fixed and only widen the ROI when the target edge is not visible.
        roi_candidates = [roi_scale, max(roi_scale * 1.5, 1.5), max(roi_scale * 2.25, 2.25)]
        candidates = [(attempt_index, 0, scale) for attempt_index, scale in enumerate(roi_candidates)]
    else:
        scan_attempts = max(max_attempts, len(dataset.get("x_ticks", []) or []) * 2 + 1)
        candidates = [
            (attempt_index, offset_units, roi_scale)
            for attempt_index, offset_units in enumerate(_scan_offsets(scan_attempts))
        ]

    for attempt_index, offset_units, candidate_roi_scale in candidates:
        shifted_center = center_value + offset_units * value_span
        try:
            crop_path, visible_ticks, visible_range = crop_bar_window(
                chart_id=dataset["chart_id"],
                image_path=image_path(dataset, "no_grid"),
                point_name=target.point_name,
                y_label=target.y_label,
                center_value=shifted_center,
                x_ticks=dataset["x_ticks"],
                x_pixels=dataset["x_pixels"],
                y_ticks=dataset["y_ticks"],
                y_pixels=dataset["y_pixels"],
                round_index=round_index,
                zoom_round_index=zoom_round_index,
                roi_scale=candidate_roi_scale,
                attempt_index=attempt_index,
                pad_x=int(segment_pixel_span / 2 + 24) if segment_pixel_span else None,
                chart_type=chart_type,
                series_name=target.series_name,
                series_order=list(dataset.get("series_color", {}).keys())
                if isinstance(dataset.get("series_color"), dict)
                else None,
            )
        except Exception as exc:
            print(f"[h_bar runner] Amplifier crop attempt {attempt_index} failed: {exc}")
            continue

        span = abs(float(visible_range[1]) - float(visible_range[0]))
        if span > 0:
            value_span = span
        if not _crop_has_target_color_and_edge(
            dataset=dataset,
            target=target,
            visible_range=visible_range,
            center_value=shifted_center,
        ):
            print(
                f"[h_bar runner] amplifier crop attempt={attempt_index} "
                f"center={shifted_center:.4f} roi_scale={candidate_roi_scale:.3f} "
                "does not contain a visible target value edge; continue refining crop."
            )
            continue
        print(
            f"[h_bar runner] amplifier crop attempt={attempt_index} "
            f"center={shifted_center:.4f} roi_scale={candidate_roi_scale:.3f} range={visible_range}"
        )
        return crop_path, visible_ticks, visible_range
    return None


def _record(
    *,
    dataset: dict[str, Any],
    target: HBarTarget,
    prompt_type: str,
    image_type: str,
    run: int,
    used_image_path: Path,
    pred: tuple[Any, Any],
    call_id: str | None = None,
) -> dict[str, Any]:
    pred_x, pred_y = pred
    prediction_readable = _valid_prediction(pred)
    mae = compute_mae(pred_x, target.gt_x)
    x_re = compute_relative_error(pred_x, target.gt_x)
    return {
        "chart_id": dataset["chart_id"],
        "call_id": call_id,
        "point": target.point_name,
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
        "x_re": x_re,
        "y_re": -1,
    }


def _prediction_value(record: dict[str, Any]) -> float | None:
    if str(record.get("prediction_readable", "")).strip().lower() == "false":
        return None
    try:
        value = float(record["pred_x"])
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
                "series_name": str(point).rsplit(",", 1)[0].strip() if "," in str(point) else "",
                "label": chosen.get("pred_y"),
                "axis": "x",
                "value": _prediction_value(chosen),
                "prompt_type": chosen.get("prompt_type"),
                "image_type": chosen.get("image_type"),
                "image_path": chosen.get("image_path"),
            }
        )
    return predictions


async def _run_target(
    *,
    client: HBarModelClient,
    dataset: dict[str, Any],
    target: HBarTarget,
    repeat_times: int,
    amplifier_rounds: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    history: list[tuple[Any, Any]] = []
    feedback_history: list[tuple[Any, Any]] = []
    feedback_pred: tuple[Any, Any] | None = None
    chart_type = str(dataset.get("chart_type") or "h_bar")
    segment_prior = stacked_segment_prior(
        dataset,
        series_name=target.series_name,
        category_label=target.y_label,
        orientation="h",
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
                pred=(segment_prior.segment_value, target.gt_y),
            )
        )

    for prompt_type, image_type in EXPERIMENT_TYPES:
        if prompt_type == "amplifier":
            center_value = segment_prior.center_value if segment_prior is not None else None
            if center_value is None:
                center_value = (
                    float(feedback_pred[0])
                    if feedback_pred and _valid_prediction(feedback_pred)
                    else _fallback_numeric_center(dataset["x_ticks"])
                )

            if center_value is None:
                print("[h_bar runner] Skip amplifier crop: no numeric center is available.")
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
                        f"[h_bar runner] Stop amplifier refinement at round {amp_round}: "
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
                    f"[h_bar] Amplifier Round {amp_round}/{amplifier_rounds} "
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
                pred = _normalize_prediction_units(dataset, pred)
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
                        f"[h_bar] Amplifier round {amp_round} reported target not readable @ {target.point_name}: {pred}; "
                        "next amplifier round will expand ROI."
                    )
                    continue

                is_stable = (
                    previous_amp_pred is not None
                    and _valid_prediction(previous_amp_pred)
                    and value_consistent(
                        pred[0],
                        previous_amp_pred[0],
                        numeric_axis_span(dataset.get("x_ticks")),
                    )
                )
                history.append(pred)
                valid_amp_count += 1
                expand_next_crop = False
                center_value = segment_prior.center_value if segment_prior is not None else float(pred[0])
                previous_amp_pred = pred
                if is_stable:
                    print(f"[h_bar] Amplifier prediction stabilized at round {amp_round}; stop zoom-in refinement.")
                    break
                print(
                    f"[h_bar] Success amplifier {amp_round}/{amplifier_rounds} "
                    f"@ {target.point_name}; next center={center_value:.4f}"
                )
            continue

        prompt_rounds = get_prompt_rounds(prompt_type, repeat_times)
        for run_idx in range(1, prompt_rounds + 1):
            used_image = image_path(dataset, image_type)
            visible_ticks = None
            previous_pred = history[-1] if history else None

            if prompt_type == "feedback" and history and run_idx > 1:
                try:
                    overlay_image = draw_prediction_overlay(
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
                    if overlay_image.exists():
                        used_image = overlay_image
                    else:
                        print(f"[h_bar runner] Feedback overlay was not created; use grid image: {overlay_image}")
                except Exception as exc:
                    print(f"[h_bar runner] Feedback overlay failed; use grid image instead: {exc}")

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
            print(f"[h_bar] Round {run_idx} | Point: {target.point_name} | Type: {prompt_type} - {image_type}")
            print(f"[h_bar] Image: {used_image}")
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
            pred = _normalize_prediction_units(dataset, pred)
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
                        pred[0],
                        previous_pred[0],
                        numeric_axis_span(dataset.get("x_ticks")),
                    )
                )
                if prompt_type != "baseline":
                    history.append(pred)
                if prompt_type == "feedback":
                    feedback_pred = pred
                    feedback_history.append(pred)
                    if is_stable:
                        print(f"[h_bar] Feedback prediction stabilized at round {run_idx}; stop feedback refinement.")
                        break
                print(f"[h_bar] Success {run_idx}/{repeat_times} [{prompt_type} - {image_type}] @ {target.point_name}")
            else:
                print(f"[h_bar] Invalid prediction [{prompt_type} - {image_type}] @ {target.point_name}: {pred}")

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
                print(f"[h_bar] Final feedback overlay saved: {final_overlay_path}")
            except Exception as exc:
                print(f"[h_bar] Final feedback overlay skipped for {target.point_name}: {exc}")

    return records


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
    chart_type: str = "h_bar",
) -> list[dict[str, Any]]:
    datasets = load_datasets(chart_ids, config_paths=config_paths, chart_type=chart_type)
    if not datasets:
        print("[h_bar] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times()
    amplifier_rounds = get_bar_amplifier_rounds()
    all_records: list[dict[str, Any]] = []
    async with HBarModelClient(legacy_urls=["http://localhost:8110/v1/chat/completions", "http://localhost:8111/v1/chat/completions"]) as client:
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
        print("[h_bar] No experiment records generated.")
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
            axis="x",
            axis_ticks=dataset.get("x_ticks"),
            use_unstable_amplifier_median=True,
            prefer_grid_on_later_stage_drift=_is_grouped_bar(dataset)
            and not repeated_grouped_colors,
            trust_readable_amplifier_over_full_view=repeated_grouped_colors,
        )
        if not final_records:
            final_records = _select_final_input_records(records)
        print(f"[h_bar] Saved results for {chart_id}: {result_dir}")
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
