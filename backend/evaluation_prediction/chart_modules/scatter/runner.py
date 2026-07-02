"""Runner for scatter chart prediction."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from PIL import Image

from gemini_calls import get_last_modal_call_id, reset_modal_call_context, set_modal_call_context

from ...common.chart_io import safe_filename
from ...common.runtime import get_prompt_rounds, get_repeat_times, numeric_axis_span, point_prediction_consistent
from ...common.full_flow_selection import write_xy_full_flow_selection
from ...common.amplifier_style import (
    amplifier_point_grid_density,
    amplifier_point_output_side,
    amplifier_point_source_window,
)

from .data import PointChartConfig, PointTarget, image_path, iter_targets, load_datasets
from .evaluation import compute_mae, compute_relative_error, save_results
from .geometry import compute_pixel_relative_error_xy, pixel_to_value
from .mark_size import MarkDiameterEstimate, crop_size_from_mark_diameter, estimate_mark_diameter
from .model import PointModelClient
from .prompts import generate_prompt
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

PREFERRED_PROMPTS = ["full_flow_final", "feedback_crop_adaptive", "feedback", "grid"]
MAX_CROP_ATTEMPTS = 5
DEFAULT_MARK_DIAMETER = 20.0
MIN_MARK_DIAMETER = 4.0
MAX_MARK_DIAMETER = 220.0


def _experiment_types() -> list[tuple[str, str]]:
    raw = os.getenv("CHART_POINT_EXPERIMENT_TYPES", "").strip()
    if not raw:
        return EXPERIMENT_TYPES
    enabled = {item.strip() for item in raw.split(",") if item.strip()}
    return [item for item in EXPERIMENT_TYPES if item[0] in enabled]


def _valid_prediction(pred: tuple[Any, Any]) -> bool:
    try:
        float(pred[0])
        float(pred[1])
        return True
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
    call_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt = (target.gt_x, target.gt_y)
    pred_x, pred_y = pred
    prediction_readable = _valid_prediction(pred)
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

    record = {
        "chart_id": dataset["chart_id"],
        "call_id": call_id,
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
        "prediction_readable": prediction_readable,
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
    if extra:
        record.update(extra)
    return record


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

    for prompt_type, image_type in _experiment_types():
        prompt_rounds = get_prompt_rounds(prompt_type, repeat_times)
        valid_amp_count = 0
        expand_next_crop = False
        for run_idx in range(1, prompt_rounds + 1):
            used_image = image_path(config, dataset, image_type)
            local_x_ticks = dataset["x_ticks"]
            local_y_ticks = dataset["y_ticks"]
            local_x_pixels = dataset["x_pixels"]
            local_y_pixels = dataset["y_pixels"]
            pred_feedback = history[-1] if history else None
            record_extra: dict[str, Any] = {}
            estimated_mark: MarkDiameterEstimate | None = None

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
                        pred=(None, None),
                        extra={"skip_reason": "missing_feedback_prediction"},
                    )
                    records.append(fallback_record)
                    print(f"[{config.chart_type} runner] skip adaptive crop without valid feedback @ {target.point_name}")
                    continue
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
                    init_crop_size=int(max(60, 120 / (2 ** valid_amp_count)) * (2.0 if expand_next_crop else 1.0)),
                    expand_crop=expand_next_crop,
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
                            pred=(None, None),
                            extra={"skip_reason": "adaptive_crop_failed"},
                        )
                    )
                    print(f"[{config.chart_type} runner] adaptive crop failed to include target @ {target.point_name}")
                    continue
                (
                    used_image,
                    local_x_ticks,
                    local_y_ticks,
                    local_x_pixels,
                    local_y_pixels,
                    _,
                    _,
                    _,
                    record_extra,
                ) = crop_result
                if "estimated_mark_diameter_px" in record_extra:
                    estimated_mark = MarkDiameterEstimate(
                        diameter_px=float(record_extra["estimated_mark_diameter_px"]),
                        source=str(record_extra.get("mark_size_source") or "unknown"),
                        confidence=float(record_extra.get("mark_size_confidence") or 0.0),
                    )

            prompt = generate_prompt(
                item_name=target.point_name,
                prompt_type=prompt_type,
                x_ticks=local_x_ticks,
                y_ticks=local_y_ticks,
                mark_name=config.mark_name,
                x_pixels=local_x_pixels,
                y_pixels=local_y_pixels,
                visual_name=target.visual_name,
                pred_feedback=pred_feedback,
                estimated_mark_diameter_px=(
                    estimated_mark.diameter_px if estimated_mark is not None else None
                ),
                mark_size_source=estimated_mark.source if estimated_mark is not None else None,
            )

            print("\n==============================")
            print(f"[{config.chart_type}] Round {run_idx} | Point: {target.point_name} | Type: {prompt_type} - {image_type}")
            print(f"[{config.chart_type}] Image: {used_image}")
            print("==============================\n")

            token = set_modal_call_context(
                {
                    "chart_name": dataset.get("chart_id"),
                    "processing_object": target.point_name,
                    "object_category": target.category,
                    "gt": {"x": target.gt_x, "y": target.gt_y},
                    "stage": "amplifier" if prompt_type == "feedback_crop_adaptive" else prompt_type,
                    "round": run_idx,
                    "image_path": str(used_image),
                    "stage_support": record_extra or None,
                }
            )
            try:
                pred = await client.predict_coords(prompt, used_image, target.point_name)
                call_id = get_last_modal_call_id()
            finally:
                reset_modal_call_context(token)
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
                    call_id=call_id,
                    extra=record_extra,
                )
            )
            if _valid_prediction(pred):
                if prompt_type != "baseline":
                    history.append(pred)
                if prompt_type == "feedback":
                    feedback_pred = pred
                    feedback_history.append(pred)
                    overlay_path = draw_prediction_overlay(
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
                    records[-1]["image_path"] = str(overlay_path)
                if (
                    prompt_type in {"feedback", "feedback_crop_adaptive"}
                    and pred_feedback is not None
                    and point_prediction_consistent(
                        pred,
                        pred_feedback,
                        x_span=numeric_axis_span(dataset.get("x_ticks")),
                        y_span=numeric_axis_span(dataset.get("y_ticks")),
                    )
                ):
                    stage_name = "amplifier" if prompt_type == "feedback_crop_adaptive" else prompt_type
                    print(f"[{config.chart_type}] {stage_name} prediction stabilized at round {run_idx}; stop refinement.")
                    break
                if prompt_type == "feedback_crop_adaptive":
                    valid_amp_count += 1
                    expand_next_crop = False
                print(f"[{config.chart_type}] Success {run_idx}/{repeat_times} [{prompt_type} - {image_type}] @ {target.point_name}")
            else:
                if prompt_type == "feedback_crop_adaptive":
                    expand_next_crop = True
                    print(
                        f"[{config.chart_type}] amplifier round {run_idx} reported target not readable @ {target.point_name}: {pred}; "
                        "next amplifier round will expand ROI."
                    )
                    continue
                print(f"[{config.chart_type}] Invalid prediction [{prompt_type} - {image_type}] @ {target.point_name}: {pred}")

    return records


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
    init_crop_size: int = 120,
    expand_crop: bool = False,
    max_attempts: int = MAX_CROP_ATTEMPTS,
) -> tuple[Path, list[float], list[float], list[float], list[float], Any, int, int, dict[str, Any]] | None:
    estimated_mark = estimate_mark_diameter(
        image_path=image_path,
        pred_coord=pred_coord,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        mark_name=mark_name,
        default_diameter=DEFAULT_MARK_DIAMETER,
        min_diameter=MIN_MARK_DIAMETER,
        max_diameter=MAX_MARK_DIAMETER,
    )
    crop_size = crop_size_from_mark_diameter(
        base_crop_size=init_crop_size,
        estimate=estimated_mark,
        mark_name=mark_name,
        expand=expand_crop,
    )
    crop_size = amplifier_point_source_window(
        crop_size,
        feedback_round,
        x_plot_span_px=_axis_pixel_span(x_pixels),
        y_plot_span_px=_axis_pixel_span(y_pixels),
        roi_scale=2.0 if expand_crop else 1.0,
        min_px=60,
    )
    diameter = estimated_mark.diameter_px
    crop_meta = estimated_mark.as_record_fields()
    crop_meta.update(
        {
            "adaptive_crop_size_px": crop_size,
            "adaptive_crop_expanded": bool(expand_crop),
        }
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
                resize_to=(amplifier_point_output_side(crop_size), amplifier_point_output_side(crop_size)),
            )
        else:
            output_side = amplifier_point_output_side(crop_size)
            grid_density = amplifier_point_grid_density()
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
                font_size=18 if output_side >= 768 else 14,
                x_grid_density=grid_density,
                y_grid_density=grid_density,
            )

        crop_path = result[0]
        print(
            f"[{config.chart_type} runner] adaptive crop attempt={attempt + 1}/{max_attempts} "
            f"size={crop_size} diameter={diameter:.2f} source={estimated_mark.source} centered on feedback @ {point_name}"
        )
        return (*result, crop_meta)
    return None


def _axis_pixel_span(pixels: list[int | float]) -> float | None:
    values = [float(value) for value in pixels if value is not None]
    if len(values) < 2:
        return None
    return max(values) - min(values)


def _prediction_pair(record: dict[str, Any]) -> tuple[float, float] | None:
    if str(record.get("prediction_readable", "")).strip().lower() == "false":
        return None
    try:
        pair = float(record["pred_x"]), float(record["pred_y"])
    except Exception:
        return None
    return pair


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
            continue
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


def _ensure_existing_image_path(record: dict[str, Any], config: PointChartConfig) -> None:
    raw_path = record.get("image_path")
    if raw_path and Path(str(raw_path)).exists():
        return
    chart_id = str(record.get("chart_id") or "")
    point_name = safe_filename(str(record.get("point_name") or record.get("id") or ""))
    prompt_type = safe_filename(str(record.get("selection_source_prompt_type") or record.get("prompt_type") or ""))
    image_type = safe_filename(str(record.get("image_type") or ""))
    if not chart_id or not point_name:
        return
    search_dir = chart_result_dir(config, chart_id) / "tempy"
    if not search_dir.exists():
        return
    patterns = [
        f"*{point_name}*{prompt_type}*{image_type}*.png",
        f"*{point_name}*.png",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(search_dir.glob(pattern))
    candidates = [path for path in candidates if path.exists()]
    if candidates:
        record["image_path"] = str(max(candidates, key=lambda path: path.stat().st_mtime))


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
    dataset_by_chart = {str(dataset.get("chart_id")): dataset for dataset in datasets}

    summaries: list[dict[str, Any]] = []
    for chart_id, records in by_chart.items():
        for record in records:
            _ensure_existing_image_path(record, config)
        result_dir = chart_result_dir(config, chart_id)
        save_results(records, result_dir)
        final_records = write_xy_full_flow_selection(
            records=records,
            result_dir=result_dir,
            x_ticks=dataset_by_chart.get(chart_id, {}).get("x_ticks"),
            y_ticks=dataset_by_chart.get(chart_id, {}).get("y_ticks"),
        )
        if not final_records:
            final_records = [
                record
                for record in records
                if record.get("prompt_type") in {"feedback_crop_adaptive", "feedback", "grid"}
            ]
        print(f"[{config.chart_type}] Saved results for {chart_id}: {result_dir}")
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
