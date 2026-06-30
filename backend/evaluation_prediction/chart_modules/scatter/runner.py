"""Runner for scatter chart prediction."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from PIL import Image

from ...common.runtime import get_repeat_times

from .data import PointChartConfig, PointTarget, image_path, iter_targets, load_datasets
from .evaluation import compute_mae, compute_relative_error, save_results
from .geometry import compute_pixel_relative_error_xy, pixel_to_value
from .model import PointModelClient
from .prompts import build_point_exists_prompt, generate_prompt
from .visual import (
    chart_result_dir,
    crop_draw_ticks_resize,
    draw_prediction_overlay,
    generate_expanded_crop_with_grid_by_diameter,
)


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("feedback_crop_adaptive", "grid_with_grid"),
]

PREFERRED_PROMPTS = ["feedback_crop_adaptive", "feedback", "grid", "baseline"]
MAX_CROP_ATTEMPTS = 5
DEFAULT_MARK_DIAMETER = 20.0
MIN_MARK_DIAMETER = 4.0
MAX_MARK_DIAMETER = 220.0


def _valid_prediction(pred: tuple[Any, Any]) -> bool:
    try:
        float(pred[0])
        float(pred[1])
        return pred != (-1, -1)
    except Exception:
        return False


def _normalize_prediction(pred: tuple[Any, Any], dataset: dict[str, Any], used_image_path: Path) -> tuple[Any, Any]:
    if not _valid_prediction(pred):
        return pred
    pred_x, pred_y = float(pred[0]), float(pred[1])
    x_min, x_max = min(dataset["x_ticks"]), max(dataset["x_ticks"])
    y_min, y_max = min(dataset["y_ticks"]), max(dataset["y_ticks"])
    if x_min <= pred_x <= x_max and y_min <= pred_y <= y_max:
        return pred_x, pred_y

    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)
    # Real-world charts may place labels/marks slightly outside the detected
    # tick span. Treat near-axis values as data, not pixels.
    if (
        x_min - x_range * 0.6 <= pred_x <= x_max + x_range * 0.6
        and y_min - y_range * 0.6 <= pred_y <= y_max + y_range * 0.6
    ):
        return pred_x, pred_y

    try:
        width, height = Image.open(used_image_path).size
    except Exception:
        return pred
    looks_like_pixels = 0 <= pred_x <= width and 0 <= pred_y <= height
    if not looks_like_pixels:
        return pred

    data_x = pixel_to_value(pred_x, dataset["x_ticks"], dataset["x_pixels"])
    data_y = pixel_to_value(pred_y, dataset["y_ticks"], dataset["y_pixels"])
    print(f"[scatter runner] normalized pixel prediction ({pred_x}, {pred_y}) -> data ({data_x:.4f}, {data_y:.4f})")
    return round(data_x, 4), round(data_y, 4)


def _record(
    *,
    config: PointChartConfig,
    dataset: dict[str, Any],
    target: PointTarget,
    prompt_type: str,
    image_type: str,
    run: int,
    used_image_path: Path,
    pred: tuple[Any, Any],
) -> dict[str, Any]:
    gt = (target.gt_x, target.gt_y)
    pred_x, pred_y = pred
    x_re, y_re = compute_relative_error(pred, gt)
    x_abs_err = abs(float(pred_x) - target.gt_x) if _valid_prediction(pred) and target.gt_x is not None else None
    y_abs_err = abs(float(pred_y) - target.gt_y) if _valid_prediction(pred) and target.gt_y is not None else None
    x_range = max(dataset["x_ticks"]) - min(dataset["x_ticks"])
    y_range = max(dataset["y_ticks"]) - min(dataset["y_ticks"])
    x_err_over_range = x_abs_err / x_range if x_abs_err is not None and x_range else None
    y_err_over_range = y_abs_err / y_range if y_abs_err is not None and y_range else None
    xy_err_over_range = (
        (x_err_over_range + y_err_over_range) / 2
        if x_err_over_range is not None and y_err_over_range is not None
        else None
    )

    try:
        image_size = Image.open(image_path(config, dataset, "grid_with_grid")).size
        pixel_rel_x, pixel_rel_y = compute_pixel_relative_error_xy(
            (float(pred_x), float(pred_y)),
            gt,
            x_ticks=dataset["x_ticks"],
            y_ticks=dataset["y_ticks"],
            x_pixels=dataset["x_pixels"],
            y_pixels=dataset["y_pixels"],
            image_size=image_size,
        )
    except Exception:
        pixel_rel_x, pixel_rel_y = None, None

    return {
        "chart_id": dataset["chart_id"],
        "point_name": target.point_name,
        "category": target.category,
        "prompt_type": prompt_type,
        "image_type": image_type,
        "run": run,
        "image_path": str(used_image_path),
        "gt_x": target.gt_x,
        "gt_y": target.gt_y,
        "pred_x": pred_x,
        "pred_y": pred_y,
        "pixel_rel_x": pixel_rel_x,
        "pixel_rel_y": pixel_rel_y,
        "mae": compute_mae(pred, gt),
        "x_re": x_re,
        "y_re": y_re,
        "x_abs_err": x_abs_err,
        "y_abs_err": y_abs_err,
        "x_range": x_range,
        "y_range": y_range,
        "x_err_over_range": x_err_over_range,
        "y_err_over_range": y_err_over_range,
        "xy_err_over_range": xy_err_over_range,
    }


async def _run_target(
    *,
    config: PointChartConfig,
    client: PointModelClient,
    dataset: dict[str, Any],
    target: PointTarget,
    repeat_times: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    history: list[tuple[Any, Any]] = []
    feedback_history: list[tuple[Any, Any]] = []
    feedback_pred: tuple[Any, Any] | None = None

    for prompt_type, image_type in EXPERIMENT_TYPES:
        for run_idx in range(1, repeat_times + 1):
            used_image = image_path(config, dataset, image_type)
            local_x_ticks = dataset["x_ticks"]
            local_y_ticks = dataset["y_ticks"]
            pred_feedback = history[-1] if history else None

            if prompt_type == "feedback" and pred_feedback is not None:
                used_image = draw_prediction_overlay(
                    config=config,
                    chart_id=dataset["chart_id"],
                    original_img_path=image_path(config, dataset, "grid_with_grid"),
                    pred_coords=[pred_feedback],
                    x_ticks=dataset["x_ticks"],
                    y_ticks=dataset["y_ticks"],
                    x_pixels=dataset["x_pixels"],
                    y_pixels=dataset["y_pixels"],
                    point_name=target.point_name,
                    run_index=run_idx,
                    prompt_type=prompt_type,
                    image_type=image_type,
                )

            if prompt_type == "feedback_crop_adaptive":
                if not feedback_pred or not _valid_prediction(feedback_pred):
                    fallback_record = _record(
                        config=config,
                        dataset=dataset,
                        target=target,
                        prompt_type=prompt_type,
                        image_type=image_type,
                        run=run_idx,
                        used_image_path=used_image,
                        pred=(-1, -1),
                    )
                    records.append(fallback_record)
                    print(f"[{config.chart_type} runner] skip adaptive crop without valid feedback @ {target.point_name}")
                    continue
                exists_prompt = build_point_exists_prompt(target.point_name, config.mark_name, target.visual_name)
                crop_result = await _try_generate_crop_until_point_detected(
                    config=config,
                    client=client,
                    chart_id=dataset["chart_id"],
                    image_path=image_path(config, dataset, "no_grid"),
                    point_name=target.point_name,
                    mark_name=config.mark_name,
                    visual_name=target.visual_name,
                    pred_coord=(float(feedback_pred[0]), float(feedback_pred[1])),
                    x_ticks=dataset["x_ticks"],
                    y_ticks=dataset["y_ticks"],
                    x_pixels=dataset["x_pixels"],
                    y_pixels=dataset["y_pixels"],
                    feedback_round=run_idx,
                    judge_prompt=exists_prompt,
                )
                if crop_result is None:
                    records.append(
                        _record(
                            config=config,
                            dataset=dataset,
                            target=target,
                            prompt_type=prompt_type,
                            image_type=image_type,
                            run=run_idx,
                            used_image_path=used_image,
                            pred=(-1, -1),
                        )
                    )
                    print(f"[{config.chart_type} runner] adaptive crop failed to include target @ {target.point_name}")
                    continue
                used_image, local_x_ticks, local_y_ticks, _, _, _, _, _ = crop_result

            prompt = generate_prompt(
                item_name=target.point_name,
                prompt_type=prompt_type,
                x_ticks=local_x_ticks,
                y_ticks=local_y_ticks,
                mark_name=config.mark_name,
                visual_name=target.visual_name,
                pred_feedback=pred_feedback,
            )

            print("\n==============================")
            print(f"[{config.chart_type}] Round {run_idx} | Point: {target.point_name} | Type: {prompt_type} - {image_type}")
            print(f"[{config.chart_type}] Image: {used_image}")
            print("==============================\n")

            pred = await client.predict_coords(prompt, used_image, target.point_name)
            pred = _normalize_prediction(pred, dataset, used_image)
            records.append(
                _record(
                    config=config,
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
                    feedback_history.append(pred)
                    draw_prediction_overlay(
                        config=config,
                        chart_id=dataset["chart_id"],
                        original_img_path=image_path(config, dataset, "grid_with_grid"),
                        pred_coords=feedback_history,
                        x_ticks=dataset["x_ticks"],
                        y_ticks=dataset["y_ticks"],
                        x_pixels=dataset["x_pixels"],
                        y_pixels=dataset["y_pixels"],
                        point_name=target.point_name,
                        draw_all_preds=True,
                        prompt_type=prompt_type,
                        image_type=image_type,
                    )
                print(f"[{config.chart_type}] Success {run_idx}/{repeat_times} [{prompt_type} - {image_type}] @ {target.point_name}")
            else:
                print(f"[{config.chart_type}] Invalid prediction [{prompt_type} - {image_type}] @ {target.point_name}: {pred}")

    return records


async def _estimate_diameter_via_llm(
    *,
    client: PointModelClient,
    image_path: Path,
    point_name: str,
    mark_name: str,
    visual_name: str,
) -> float:
    mark = "bubble" if mark_name == "bubble" else "circle"
    target = f"{point_name}"
    if visual_name and visual_name != point_name:
        target += f" / {visual_name}"
    prompt = (
        f"Estimate the visible diameter in pixels of the {mark} corresponding to [{target}] in this chart image. "
        f"Return only a JSON object like {{\"diameter\": 24}}. "
        f"If the exact object is hard to identify, estimate the typical diameter of that target mark."
    )
    content = await client.call_text(prompt, image_path, f"diameter:{point_name}")
    candidates: list[float] = []
    for match in re.findall(r"\d+(?:\.\d+)?", str(content or "")):
        try:
            value = float(match)
        except Exception:
            continue
        if MIN_MARK_DIAMETER <= value <= MAX_MARK_DIAMETER:
            candidates.append(value)
    if candidates:
        return candidates[0]
    return DEFAULT_MARK_DIAMETER


async def _try_generate_crop_until_point_detected(
    *,
    config: PointChartConfig,
    client: PointModelClient,
    chart_id: str,
    image_path: Path,
    point_name: str,
    mark_name: str,
    visual_name: str,
    pred_coord: tuple[float, float],
    x_ticks: list[float],
    y_ticks: list[float],
    x_pixels: list[int],
    y_pixels: list[int],
    feedback_round: int,
    judge_prompt: str,
    init_crop_size: int = 120,
    max_attempts: int = MAX_CROP_ATTEMPTS,
) -> tuple[Path, list[float], list[float], list[float], list[float], Any, int, int] | None:
    crop_size = init_crop_size
    diameter = await _estimate_diameter_via_llm(
        client=client,
        image_path=image_path,
        point_name=point_name,
        mark_name=mark_name,
        visual_name=visual_name,
    )

    for attempt in range(max_attempts):
        if attempt == 0:
            result = generate_expanded_crop_with_grid_by_diameter(
                config=config,
                chart_id=chart_id,
                image_path=image_path,
                point_name=f"{point_name}_rt{attempt}",
                pred_coord=pred_coord,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
                x_pixels=x_pixels,
                y_pixels=y_pixels,
                diameter=diameter,
                feedback_round=feedback_round,
                base_crop_size=crop_size,
                resize_to=(224, 224),
            )
        else:
            output_side = min(max(224, crop_size), 1024)
            result = crop_draw_ticks_resize(
                config=config,
                chart_id=chart_id,
                image_path=image_path,
                point_name=f"{point_name}_rt{attempt}",
                pred_coord=pred_coord,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
                x_pixels=x_pixels,
                y_pixels=y_pixels,
                feedback_round=feedback_round,
                window_size=crop_size,
                output_size=(output_side, output_side),
                font_size=8 if crop_size <= 180 else 10,
                x_grid_density=1,
                y_grid_density=1,
            )

        crop_path = result[0]
        exists = await client.check_exists(judge_prompt, crop_path)
        print(
            f"[{config.chart_type} runner] adaptive crop attempt={attempt + 1}/{max_attempts} "
            f"size={crop_size} contains target={exists} @ {point_name}"
        )
        if exists:
            return result
        crop_size *= 2
    return None


def _prediction_pair(record: dict[str, Any]) -> tuple[float, float] | None:
    try:
        pair = float(record["pred_x"]), float(record["pred_y"])
    except Exception:
        return None
    return pair if _valid_prediction(pair) else None


def _select_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if _prediction_pair(record) is not None:
            by_point.setdefault(str(record["point_name"]), []).append(record)

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
        pair = _prediction_pair(chosen)
        if pair is None:
            continue
        predictions.append(
            {
                "id": point,
                "series_name": chosen.get("category", ""),
                "category": chosen.get("category", ""),
                "label": point,
                "axis": "xy",
                "value": {"x": pair[0], "y": pair[1]},
                "x": pair[0],
                "y": pair[1],
                "prompt_type": chosen.get("prompt_type"),
                "image_type": chosen.get("image_type"),
                "image_path": chosen.get("image_path"),
            }
        )
    return predictions


def _fallback_center(dataset: dict[str, Any], target: PointTarget) -> tuple[float, float]:
    if target.gt_x is not None and target.gt_y is not None:
        return target.gt_x, target.gt_y
    return _axis_midpoint(dataset.get("x_ticks", [])), _axis_midpoint(dataset.get("y_ticks", []))


def _axis_midpoint(ticks: Any) -> float:
    values = []
    for tick in ticks if isinstance(ticks, list) else []:
        try:
            values.append(float(tick))
        except Exception:
            continue
    if not values:
        return 0.0
    return (min(values) + max(values)) / 2


async def run_experiment(
    config: PointChartConfig,
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    datasets = load_datasets(config, chart_ids, config_paths=config_paths)
    if not datasets:
        print(f"[{config.chart_type}] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times()
    all_records: list[dict[str, Any]] = []
    async with PointModelClient() as client:
        for start in range(0, len(datasets), batch_size or len(datasets)):
            batch = datasets[start : start + (batch_size or len(datasets))]
            tasks = [
                _run_target(config=config, client=client, dataset=dataset, target=target, repeat_times=repeat_times)
                for dataset in batch
                for target in iter_targets(dataset)
            ]
            for result in await asyncio.gather(*tasks):
                all_records.extend(result)

    by_chart: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        by_chart.setdefault(record["chart_id"], []).append(record)

    summaries: list[dict[str, Any]] = []
    for chart_id, records in by_chart.items():
        result_dir = chart_result_dir(config, chart_id)
        save_results(records, result_dir)
        print(f"[{config.chart_type}] Saved results for {chart_id}: {result_dir}")
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
