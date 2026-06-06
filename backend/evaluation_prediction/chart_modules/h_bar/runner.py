"""Modular runner for horizontal bar value prediction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...common.runtime import get_bar_amplifier_rounds, get_repeat_times

from .data import HBarTarget, image_path, iter_targets, load_datasets
from .evaluation import compute_mae, compute_relative_error, save_results
from .model import HBarModelClient
from .prompts import build_color_prompt, generate_prompt
from .visual import chart_result_dir, crop_bar_window, draw_prediction_overlay


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("amplifier", "grid_with_grid"),
]

PREFERRED_PROMPTS = ["amplifier", "feedback", "grid", "baseline"]


def _valid_prediction(pred: tuple[Any, Any]) -> bool:
    try:
        float(pred[0])
        return pred != (-1, -1)
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


def _scan_offsets(max_attempts: int) -> list[int]:
    offsets = [0]
    for index in range(1, max_attempts):
        step = (index + 1) // 2
        offsets.append(-step if index % 2 == 1 else step)
    return offsets


async def _crop_until_bar_detected(
    *,
    client: HBarModelClient,
    dataset: dict[str, Any],
    target: HBarTarget,
    center_value: float,
    round_index: int,
    max_attempts: int = 8,
) -> tuple[Path, list[float], tuple[float, float]] | None:
    exists_prompt = build_color_prompt(target.point_name, dataset["series_color"])
    value_span = _fallback_numeric_step(dataset["x_ticks"])

    for attempt_index, offset_units in enumerate(_scan_offsets(max_attempts)):
        shifted_center = center_value + offset_units * value_span
        try:
            crop_path, visible_ticks, visible_range = crop_bar_window(
                chart_id=dataset["chart_id"],
                image_path=image_path(dataset, "grid_with_grid"),
                point_name=target.point_name,
                y_label=target.y_label,
                center_value=shifted_center,
                x_ticks=dataset["x_ticks"],
                x_pixels=dataset["x_pixels"],
                y_ticks=dataset["y_ticks"],
                y_pixels=dataset["y_pixels"],
                round_index=round_index,
                attempt_index=attempt_index,
            )
        except Exception as exc:
            print(f"[h_bar runner] Amplifier crop attempt {attempt_index} failed: {exc}")
            continue

        span = abs(float(visible_range[1]) - float(visible_range[0]))
        if span > 0:
            value_span = span
        exists = await client.check_exists(exists_prompt, crop_path)
        print(
            f"[h_bar runner] amplifier crop attempt={attempt_index} "
            f"center={shifted_center:.4f} range={visible_range} contains target={exists}"
        )
        if exists:
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
) -> dict[str, Any]:
    pred_x, pred_y = pred
    mae = compute_mae(pred_x, target.gt_x)
    x_re = compute_relative_error(pred_x, target.gt_x)
    return {
        "chart_id": dataset["chart_id"],
        "point": target.point_name,
        "prompt_type": prompt_type,
        "image_type": image_type,
        "run": run,
        "image_path": str(used_image_path),
        "gt_x": target.gt_x,
        "gt_y": target.gt_y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "mae": mae,
        "pixel_rel_x": -1,
        "pixel_rel_y": -1,
        "x_re": x_re,
        "y_re": -1,
    }


def _prediction_value(record: dict[str, Any]) -> float | None:
    try:
        return float(record["pred_x"])
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
            chosen = point_records[-1]
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
    feedback_pred: tuple[Any, Any] | None = None

    for prompt_type, image_type in EXPERIMENT_TYPES:
        if prompt_type == "amplifier":
            center_value = (
                float(feedback_pred[0])
                if feedback_pred and _valid_prediction(feedback_pred)
                else target.gt_x
            )
            if center_value is None:
                center_value = _fallback_numeric_center(dataset["x_ticks"])

            if center_value is None:
                print("[h_bar runner] Skip amplifier crop: no numeric center is available.")
                continue

            for amp_round in range(1, amplifier_rounds + 1):
                crop_result = await _crop_until_bar_detected(
                    client=client,
                    dataset=dataset,
                    target=target,
                    center_value=float(center_value),
                    round_index=amp_round,
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
                    visible_ticks=visible_ticks,
                    pred_feedback=history[-2:] if history else None,
                    feedback_round=2,
                    current_round=amp_round,
                )

                print("\n==============================")
                print(
                    f"[h_bar] Amplifier Round {amp_round}/{amplifier_rounds} "
                    f"| Point: {target.point_name} | Image: {used_image}"
                )
                print("==============================\n")

                pred = await client.predict_coords(prompt, used_image, target.point_name)
                records.append(
                    _record(
                        dataset=dataset,
                        target=target,
                        prompt_type=prompt_type,
                        image_type=image_type,
                        run=amp_round,
                        used_image_path=used_image,
                        pred=pred,
                    )
                )
                if not _valid_prediction(pred):
                    print(f"[h_bar] Invalid amplifier prediction round {amp_round} @ {target.point_name}: {pred}")
                    break

                history.append(pred)
                center_value = float(pred[0])
                print(
                    f"[h_bar] Success amplifier {amp_round}/{amplifier_rounds} "
                    f"@ {target.point_name}; next center={center_value:.4f}"
                )
            continue

        for run_idx in range(1, repeat_times + 1):
            used_image = image_path(dataset, image_type)
            visible_ticks = None

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
                )

            prompt = generate_prompt(
                item_name=target.point_name,
                prompt_type=prompt_type,
                x_ticks=dataset["x_ticks"],
                y_ticks=dataset["y_ticks"],
                series_color=dataset["series_color"],
                visible_ticks=visible_ticks,
                pred_feedback=history[-2:] if history else None,
                feedback_round=2,
                current_round=run_idx,
            )

            print("\n==============================")
            print(f"[h_bar] Round {run_idx} | Point: {target.point_name} | Type: {prompt_type} - {image_type}")
            print(f"[h_bar] Image: {used_image}")
            print("==============================\n")

            pred = await client.predict_coords(prompt, used_image, target.point_name)
            records.append(
                _record(
                    dataset=dataset,
                    target=target,
                    prompt_type=prompt_type,
                    image_type=image_type,
                    run=run_idx,
                    used_image_path=used_image,
                    pred=pred,
                )
            )
            if _valid_prediction(pred):
                history.append(pred)
                if prompt_type == "feedback":
                    feedback_pred = pred
                print(f"[h_bar] Success {run_idx}/{repeat_times} [{prompt_type} - {image_type}] @ {target.point_name}")
            else:
                print(f"[h_bar] Invalid prediction [{prompt_type} - {image_type}] @ {target.point_name}: {pred}")

        if prompt_type == "feedback" and feedback_pred and _valid_prediction(feedback_pred):
            final_overlay_path = draw_prediction_overlay(
                chart_id=dataset["chart_id"],
                original_img_path=image_path(dataset, "grid_with_grid"),
                pred_coords=history,
                x_ticks=dataset["x_ticks"],
                y_ticks=dataset["y_ticks"],
                x_pixels=dataset["x_pixels"],
                y_pixels=dataset["y_pixels"],
                point_name=target.point_name,
                draw_all_preds=True,
                prompt_type=prompt_type,
                image_type=image_type,
                run_index=repeat_times,
                final_overlay=True,
            )
            print(f"[h_bar] Final feedback overlay saved: {final_overlay_path}")

    return records


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    datasets = load_datasets(chart_ids, config_paths=config_paths)
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

    summaries: list[dict[str, Any]] = []
    for chart_id, records in by_chart.items():
        result_dir = chart_result_dir(chart_id)
        save_results(records, result_dir)
        print(f"[h_bar] Saved results for {chart_id}: {result_dir}")
        predictions = _select_predictions(records)
        summaries.append(
            {
                "chart_id": chart_id,
                "result_dir": str(result_dir),
                "record_count": len(records),
                "object_count": len(predictions),
                "predictions": predictions,
            }
        )
    return summaries
