"""Modular runner for line chart value prediction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageColor

from gemini_calls import get_last_modal_call_id, reset_modal_call_context, set_modal_call_context

from ...common.runtime import get_prompt_rounds, get_repeat_times, numeric_axis_span, value_consistent
from ...common.full_flow_selection import write_bar_full_flow_selection

from .data import LineTarget, image_path, iter_targets, load_datasets
from .evaluation import compute_mae, compute_relative_error, save_results
from .geometry import category_pixel, category_span, numeric_pixel
from .model import LineModelClient
from .prompts import generate_prompt
from .visual import chart_result_dir, crop_line_point_window, draw_prediction_overlay


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("amplifier", "grid_with_grid"),
]

PREFERRED_PROMPTS = ["full_flow_final", "amplifier", "feedback", "grid"]


def _valid_prediction(pred: tuple[Any, Any]) -> bool:
    try:
        float(pred[1])
        return True
    except Exception:
        return False


def _record(
    *,
    dataset: dict[str, Any],
    target: LineTarget,
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
        return float(record["pred_y"])
    except Exception:
        return None


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


def _fallback_center_value(dataset: dict[str, Any]) -> float:
    values = []
    for tick in dataset.get("y_ticks", []):
        try:
            values.append(float(tick))
        except Exception:
            continue
    if values:
        return (min(values) + max(values)) / 2
    return 0.0


def _numeric_step(ticks: list[Any]) -> float:
    values = sorted({float(tick) for tick in ticks if _is_number(tick)})
    gaps = [right - left for left, right in zip(values, values[1:]) if right > left]
    return gaps[len(gaps) // 2] if gaps else 1.0


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


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
        return None
    try:
        return tuple(ImageColor.getrgb(value.strip())[:3])
    except Exception:
        return None


def _line_crop_has_target_color(
    *,
    dataset: dict[str, Any],
    target: LineTarget,
    visible_range: tuple[float, float],
) -> bool:
    rgb = _series_rgb(dataset, target.series_name)
    if rgb is None:
        return True
    try:
        source = Image.open(image_path(dataset, "no_grid")).convert("RGB")
        arr = np.asarray(source, dtype=np.int16)
        center_x = category_pixel(target.x_label, dataset["x_ticks"], dataset["x_pixels"])
        half_x = max(6, category_span(target.x_label, dataset["x_ticks"], dataset["x_pixels"], source.size) // 2)
        y0 = numeric_pixel(float(visible_range[0]), dataset["y_ticks"], dataset["y_pixels"])
        y1 = numeric_pixel(float(visible_range[1]), dataset["y_ticks"], dataset["y_pixels"])
    except Exception:
        return True
    top, bottom = sorted((int(y0), int(y1)))
    height, width = arr.shape[:2]
    left = max(0, int(center_x) - half_x)
    right = min(width, int(center_x) + half_x + 1)
    top = max(0, min(height - 1, top))
    bottom = max(0, min(height - 1, bottom))
    if right <= left or bottom <= top:
        return True
    band = arr[top : bottom + 1, left:right, :]
    target_rgb = np.array(rgb, dtype=np.int16)
    dist = np.sqrt(np.sum((band - target_rgb) ** 2, axis=2))
    return int(np.count_nonzero(dist <= 42)) >= 6


async def _run_target(
    *,
    client: LineModelClient,
    dataset: dict[str, Any],
    target: LineTarget,
    repeat_times: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    history: list[tuple[Any, Any]] = []
    feedback_pred: tuple[Any, Any] | None = None

    for prompt_type, image_type in EXPERIMENT_TYPES:
        prompt_rounds = get_prompt_rounds(prompt_type, repeat_times)
        valid_amp_count = 0
        expand_next_crop = False
        for run_idx in range(1, prompt_rounds + 1):
            used_image = image_path(dataset, image_type)
            visible_ticks = None
            previous_pred = history[-1] if history else None

            if prompt_type == "feedback" and history:
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
                )

            if prompt_type == "amplifier":
                center_value = (
                    float(feedback_pred[1])
                    if feedback_pred and _valid_prediction(feedback_pred)
                    else _fallback_center_value(dataset)
                )
                crop_zoom_round = max(1, valid_amp_count + 1)
                if expand_next_crop:
                    crop_zoom_round = max(1, valid_amp_count)
                crop_result = None
                step = _numeric_step(dataset.get("y_ticks", []))
                scan_attempts = max(9, len(dataset.get("y_ticks", []) or []) * 2 + 1)
                for offset_index, offset_units in enumerate(_scan_offsets(scan_attempts)):
                    shifted_center = center_value + offset_units * step
                    candidate_image, candidate_ticks, visible_range = crop_line_point_window(
                        chart_id=dataset["chart_id"],
                        image_path=image_path(dataset, "no_grid"),
                        point_name=target.point_name,
                        x_label=target.x_label,
                        center_value=shifted_center,
                        x_ticks=dataset["x_ticks"],
                        x_pixels=dataset["x_pixels"],
                        y_ticks=dataset["y_ticks"],
                        y_pixels=dataset["y_pixels"],
                        round_index=run_idx,
                        zoom_round_index=crop_zoom_round,
                        roi_scale=2.0 if expand_next_crop else 1.0,
                    )
                    if not _line_crop_has_target_color(dataset=dataset, target=target, visible_range=visible_range):
                        print(
                            f"[line runner] amplifier crop attempt={offset_index} "
                            f"center={shifted_center:.4f} has no target-colored line/point; continue scanning."
                        )
                        continue
                    crop_result = (candidate_image, candidate_ticks, visible_range)
                    break
                if crop_result is None:
                    records.append(
                        _record(
                            dataset=dataset,
                            target=target,
                            prompt_type=prompt_type,
                            image_type=image_type,
                            run=run_idx,
                            used_image_path=used_image,
                            pred=(None, None),
                        )
                    )
                    expand_next_crop = True
                    continue
                used_image, visible_ticks, _ = crop_result
                print(f"[line runner] amplifier crop centered on previous prediction: {used_image}")

            prompt = generate_prompt(
                item_name=target.point_name,
                prompt_type=prompt_type,
                x_ticks=dataset["x_ticks"],
                y_ticks=dataset["y_ticks"],
                series_color=dataset["series_color"],
                x_pixels=dataset["x_pixels"],
                y_pixels=dataset["y_pixels"],
                visible_ticks=visible_ticks,
                pred_feedback=previous_pred,
            )

            print("\n==============================")
            print(f"[line] Round {run_idx} | Point: {target.point_name} | Type: {prompt_type} - {image_type}")
            print(f"[line] Image: {used_image}")
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
                    prompt_type in {"feedback", "amplifier"}
                    and previous_pred is not None
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
                elif prompt_type == "amplifier":
                    feedback_pred = pred
                    valid_amp_count += 1
                    expand_next_crop = False
                if is_stable:
                    print(f"[line] {prompt_type} prediction stabilized at round {run_idx}; stop refinement.")
                    break
                print(f"[line] Success {run_idx}/{repeat_times} [{prompt_type} - {image_type}] @ {target.point_name}")
            else:
                if prompt_type == "amplifier":
                    expand_next_crop = True
                    print(
                        f"[line] Amplifier round {run_idx} reported target not readable @ {target.point_name}: {pred}; "
                        "next amplifier round will expand ROI."
                    )
                    continue
                print(f"[line] Invalid prediction [{prompt_type} - {image_type}] @ {target.point_name}: {pred}")

    return records


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    datasets = load_datasets(chart_ids, config_paths=config_paths)
    if not datasets:
        print("[line] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times()
    all_records: list[dict[str, Any]] = []
    async with LineModelClient() as client:
        for start in range(0, len(datasets), batch_size or len(datasets)):
            batch = datasets[start : start + (batch_size or len(datasets))]
            tasks = [
                _run_target(client=client, dataset=dataset, target=target, repeat_times=repeat_times)
                for dataset in batch
                for target in iter_targets(dataset)
            ]
            for result in await asyncio.gather(*tasks):
                all_records.extend(result)

    if not all_records:
        print("[line] No experiment records generated.")
        return []

    by_chart: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_chart.setdefault(record["chart_id"], []).append(record)
    dataset_by_chart = {str(dataset.get("chart_id")): dataset for dataset in datasets}

    summaries: list[dict[str, Any]] = []
    for chart_id, records in by_chart.items():
        result_dir = chart_result_dir(chart_id)
        dataset = dataset_by_chart.get(chart_id, {})
        save_results(records, result_dir)
        final_records = write_bar_full_flow_selection(
            records=records,
            result_dir=result_dir,
            axis="y",
            axis_ticks=dataset.get("y_ticks"),
            use_unstable_amplifier_median=False,
        )
        if not final_records:
            final_records = [record for record in records if record.get("prompt_type") in {"amplifier", "feedback", "grid"}]
        print(f"[line] Saved results for {chart_id}: {result_dir}")
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
