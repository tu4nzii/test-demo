"""Modular runner for horizontal bar value prediction."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from prediction_core.runtime import get_repeat_times

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


def _valid_prediction(pred: tuple[Any, Any]) -> bool:
    try:
        float(pred[0])
        return pred != (-1, -1)
    except Exception:
        return False


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


async def _run_target(
    *,
    client: HBarModelClient,
    dataset: dict[str, Any],
    target: HBarTarget,
    repeat_times: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    history: list[tuple[Any, Any]] = []
    feedback_pred: tuple[Any, Any] | None = None

    for prompt_type, image_type in EXPERIMENT_TYPES:
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
                )

            if prompt_type == "amplifier":
                center_value = float(feedback_pred[0]) if feedback_pred and _valid_prediction(feedback_pred) else target.gt_x
                used_image, visible_ticks, _ = crop_bar_window(
                    chart_id=dataset["chart_id"],
                    image_path=image_path(dataset, "grid_with_grid"),
                    point_name=target.point_name,
                    y_label=target.y_label,
                    center_value=center_value,
                    x_ticks=dataset["x_ticks"],
                    x_pixels=dataset["x_pixels"],
                    y_ticks=dataset["y_ticks"],
                    y_pixels=dataset["y_pixels"],
                    round_index=run_idx,
                )
                exists_prompt = build_color_prompt(target.point_name, dataset["series_color"])
                exists = await client.check_exists(exists_prompt, used_image)
                print(f"[h_bar runner] amplifier crop contains target={exists}")

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

    return records


async def run_experiment(batch_size: int | None = None, chart_ids: list[str] | None = None) -> None:
    datasets = load_datasets(chart_ids)
    if not datasets:
        print("[h_bar] No matching chart configs. Nothing to run.")
        return

    repeat_times = get_repeat_times()
    all_records: list[dict[str, Any]] = []
    async with HBarModelClient(legacy_urls=["http://localhost:8110/v1/chat/completions", "http://localhost:8111/v1/chat/completions"]) as client:
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
        print("[h_bar] No experiment records generated.")
        return

    by_chart: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_chart.setdefault(record["chart_id"], []).append(record)

    for chart_id, records in by_chart.items():
        result_dir = chart_result_dir(chart_id)
        save_results(records, result_dir)
        print(f"[h_bar] Saved results for {chart_id}: {result_dir}")
