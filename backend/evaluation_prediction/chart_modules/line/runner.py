"""Modular runner for line chart value prediction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...common.runtime import get_amplifier_rounds, get_repeat_times

from .data import LineTarget, image_path, iter_targets, load_datasets
from .evaluation import compute_mae, compute_relative_error, save_results
from .model import LineModelClient
from .prompts import build_point_exists_prompt, generate_prompt
from .visual import chart_result_dir, crop_line_point_window, draw_prediction_overlay


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("amplifier", "grid_with_grid"),
]

PREFERRED_PROMPTS = ["amplifier", "feedback", "grid", "baseline"]
AMPLIFIER_STABILITY_REL_THRESHOLD = 0.01

AMPLIFIER_REFINE_PARAMS = {
    1: {"half_ratio": 0.10, "zoom": 2, "grid_div": 1},
    2: {"half_ratio": 0.05, "zoom": 4, "grid_div": 2},
    3: {"half_ratio": 0.025, "zoom": 4, "grid_div": 2},
}


def _valid_prediction(pred: tuple[Any, Any]) -> bool:
    try:
        float(pred[1])
        return pred != ("", -1)
    except Exception:
        return False


def _numeric_range(ticks: list[Any]) -> float:
    values: list[float] = []
    for tick in ticks:
        try:
            values.append(float(tick))
        except Exception:
            continue
    return max(values) - min(values) if values else float("nan")


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


def _stability_ratio(prev_value: Any, curr_value: Any, value_range: float) -> float | None:
    try:
        prev = float(prev_value)
        curr = float(curr_value)
    except Exception:
        return None
    denom = max(abs(prev), abs(float(value_range)) * 0.01, 1e-6)
    return abs(curr - prev) / denom


def _strategy_repeat_times(prompt_type: str, repeat_times: int) -> int:
    if prompt_type in {"baseline", "grid"}:
        return 1
    if prompt_type == "feedback":
        return 2
    return repeat_times


def _scan_offsets(max_attempts: int) -> list[int]:
    offsets = [0]
    for index in range(1, max_attempts):
        step = (index + 1) // 2
        offsets.append(-step if index % 2 == 1 else step)
    return offsets


def _record(
    *,
    dataset: dict[str, Any],
    target: LineTarget,
    prompt_type: str,
    image_type: str,
    run: int,
    used_image_path: Path,
    pred: tuple[Any, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred_x, pred_y = pred
    mae = compute_mae(pred_y, target.gt_y)
    y_re = compute_relative_error(pred_y, target.gt_y)
    record = {
        "chart_id": dataset["chart_id"],
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
        "mae": mae,
        "pixel_rel_x": -1,
        "pixel_rel_y": -1,
        "x_re": -1,
        "y_re": y_re,
    }
    if extra:
        record.update(extra)
    return record


def _prediction_value(record: dict[str, Any]) -> float | None:
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
            chosen = point_records[-1]
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


async def _crop_until_point_detected(
    *,
    client: LineModelClient,
    dataset: dict[str, Any],
    target: LineTarget,
    center_value: float,
    round_index: int,
    crop_params: dict[str, Any] | None = None,
    max_attempts: int = 8,
) -> tuple[Path, list[float], tuple[float, float]] | None:
    exists_prompt = build_point_exists_prompt(target.point_name, dataset["series_color"])
    value_step = _fallback_numeric_step(dataset["y_ticks"])
    crop_params = crop_params or {}

    for attempt_index, offset_units in enumerate(_scan_offsets(max_attempts)):
        shifted_center = center_value + offset_units * value_step
        try:
            crop_path, visible_ticks, visible_range = crop_line_point_window(
                chart_id=dataset["chart_id"],
                image_path=image_path(dataset, "grid_with_grid"),
                point_name=target.point_name,
                x_label=target.x_label,
                center_value=shifted_center,
                x_ticks=dataset["x_ticks"],
                x_pixels=dataset["x_pixels"],
                y_ticks=dataset["y_ticks"],
                y_pixels=dataset["y_pixels"],
                round_index=round_index,
                attempt_index=attempt_index,
                half_ratio=crop_params.get("half_ratio"),
                zoom_factor=crop_params.get("zoom"),
                grid_div=crop_params.get("grid_div"),
            )
        except Exception as exc:
            print(f"[line runner] amplifier crop attempt {attempt_index} failed: {exc}")
            continue

        span = abs(float(visible_range[1]) - float(visible_range[0]))
        if span > 0:
            value_step = span
        exists = await client.check_exists(exists_prompt, crop_path)
        print(
            f"[line runner] amplifier crop attempt={attempt_index} "
            f"center={shifted_center:.4f} range={visible_range} contains target={exists}"
        )
        if exists:
            return crop_path, visible_ticks, visible_range
    return None


async def _run_target(
    *,
    client: LineModelClient,
    dataset: dict[str, Any],
    target: LineTarget,
    repeat_times: int,
    amplifier_rounds: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    history: list[tuple[Any, Any]] = []
    feedback_pred: tuple[Any, Any] | None = None
    grid_pred: tuple[Any, Any] | None = None

    for prompt_type, image_type in EXPERIMENT_TYPES:
        if prompt_type == "amplifier":
            center_value = (
                float(feedback_pred[1])
                if feedback_pred and _valid_prediction(feedback_pred)
                else (target.gt_y if target.gt_y is not None else _fallback_center_value(dataset))
            )
            last_amplifier_pred: tuple[Any, Any] | None = None
            last_amplifier_readable = True
            for amp_round in range(1, amplifier_rounds + 1):
                amplifier_params = AMPLIFIER_REFINE_PARAMS.get(amp_round, AMPLIFIER_REFINE_PARAMS[3])
                crop_result = await _crop_until_point_detected(
                    client=client,
                    dataset=dataset,
                    target=target,
                    center_value=float(center_value),
                    round_index=amp_round,
                    crop_params=amplifier_params,
                )
                if crop_result is None:
                    print(
                        f"[line runner] Stop amplifier refinement at round {amp_round}: "
                        "no crop contains the target point."
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
                    pred_feedback=history[-1] if history else None,
                )

                print("\n==============================")
                print(
                    f"[line] Amplifier Round {amp_round}/{amplifier_rounds} "
                    f"| Point: {target.point_name} | Image: {used_image}"
                )
                print("==============================\n")

                pred = await client.predict_coords(prompt, used_image, target.point_name)
                if _valid_prediction(pred):
                    pred = (target.x_label, pred[1])
                response_readable = bool(client.last_readable)
                valid_pred = _valid_prediction(pred) and response_readable
                amplifier_stability_ratio = None
                amplifier_stop_reason = ""
                if valid_pred and last_amplifier_pred is not None and last_amplifier_readable:
                    amplifier_stability_ratio = _stability_ratio(
                        last_amplifier_pred[1],
                        pred[1],
                        _numeric_range(dataset["y_ticks"]),
                    )
                    if (
                        amplifier_stability_ratio is not None
                        and amplifier_stability_ratio < AMPLIFIER_STABILITY_REL_THRESHOLD
                    ):
                        amplifier_stop_reason = "stable"
                records.append(
                    _record(
                        dataset=dataset,
                        target=target,
                        prompt_type=prompt_type,
                        image_type=image_type,
                        run=amp_round,
                        used_image_path=used_image,
                        pred=pred,
                        extra={
                            "readable": response_readable,
                            "amplifier_stop_reason": amplifier_stop_reason,
                            "amplifier_stability_ratio": amplifier_stability_ratio,
                        },
                    )
                )
                if not valid_pred:
                    print(f"[line] Invalid amplifier prediction round {amp_round} @ {target.point_name}: {pred}")
                    break

                history.append(pred)
                last_amplifier_pred = pred
                last_amplifier_readable = response_readable
                center_value = float(pred[1])
                print(
                    f"[line] Success amplifier {amp_round}/{amplifier_rounds} "
                    f"@ {target.point_name}; next center={center_value:.4f}"
                )
                if amplifier_stop_reason == "stable":
                    print(
                        f"[line] amplifier stable early stop @ {target.point_name}: "
                        f"ratio={float(amplifier_stability_ratio):.6f} < {AMPLIFIER_STABILITY_REL_THRESHOLD}"
                    )
                    break
            continue

        target_runs = _strategy_repeat_times(prompt_type, repeat_times)
        last_run_idx = 0
        for run_idx in range(1, target_runs + 1):
            last_run_idx = run_idx
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
                pred_feedback=history[-1] if history else None,
            )

            print("\n==============================")
            print(f"[line] Round {run_idx} | Point: {target.point_name} | Type: {prompt_type} - {image_type}")
            print(f"[line] Image: {used_image}")
            print("==============================\n")

            pred = await client.predict_coords(prompt, used_image, target.point_name)
            if _valid_prediction(pred):
                pred = (target.x_label, pred[1])
            feedback_stability_ratio = None
            feedback_stop_reason = ""
            if prompt_type == "feedback" and _valid_prediction(pred) and grid_pred and _valid_prediction(grid_pred):
                feedback_stability_ratio = _stability_ratio(
                    grid_pred[1],
                    pred[1],
                    _numeric_range(dataset["y_ticks"]),
                )
                if (
                    feedback_stability_ratio is not None
                    and feedback_stability_ratio < AMPLIFIER_STABILITY_REL_THRESHOLD
                ):
                    feedback_stop_reason = "stable_with_grid"
            records.append(
                _record(
                    dataset=dataset,
                    target=target,
                    prompt_type=prompt_type,
                    image_type=image_type,
                    run=run_idx,
                    used_image_path=used_image,
                    pred=pred,
                    extra={
                        "feedback_stop_reason": feedback_stop_reason if prompt_type == "feedback" else "",
                        "feedback_stability_ratio": (
                            feedback_stability_ratio if prompt_type == "feedback" else None
                        ),
                    },
                )
            )
            if _valid_prediction(pred):
                history.append(pred)
                if prompt_type == "grid":
                    grid_pred = pred
                if prompt_type == "feedback":
                    feedback_pred = pred
                print(f"[line] Success {run_idx}/{target_runs} [{prompt_type} - {image_type}] @ {target.point_name}")
                if prompt_type == "feedback" and feedback_stop_reason == "stable_with_grid":
                    print(
                        f"[line] feedback stable early stop @ {target.point_name}: "
                        f"ratio={float(feedback_stability_ratio):.6f} < {AMPLIFIER_STABILITY_REL_THRESHOLD}"
                    )
                    break
            else:
                print(f"[line] Invalid prediction [{prompt_type} - {image_type}] @ {target.point_name}: {pred}")

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
                run_index=last_run_idx or target_runs,
                final_overlay=True,
            )
            print(f"[line] Final feedback overlay saved: {final_overlay_path}")

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
    amplifier_rounds = get_amplifier_rounds()
    all_records: list[dict[str, Any]] = []
    async with LineModelClient() as client:
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
        print("[line] No experiment records generated.")
        return []

    by_chart: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_chart.setdefault(record["chart_id"], []).append(record)

    summaries: list[dict[str, Any]] = []
    for chart_id, records in by_chart.items():
        result_dir = chart_result_dir(chart_id)
        save_results(records, result_dir)
        print(f"[line] Saved results for {chart_id}: {result_dir}")
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
