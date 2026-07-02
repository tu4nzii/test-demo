"""Runner for backend-local donut value extraction."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from aiohttp import ClientTimeout
from PIL import Image, ImageDraw

from gemini_calls import FAILURE_TEXT, chat_with_gemini, get_last_modal_call_id, reset_modal_call_context, set_modal_call_context

from ...common.chart_io import ensure_dir, image_to_data_url
from ...common.paths import RESULTS_ROOT
from ...common.runtime import (
    get_amplifier_rounds,
    get_feedback_rounds,
    get_repeat_times,
)

from ..circular_artifacts import save_circular_artifacts
from ..circular_flow import angle_prediction_stable, crop_circular_amplifier_image, draw_circular_grid_image
from ..circular_model_config import get_model_name
from ..circular_predictions import complete_circular_predictions, normalize_circular_prediction_shares, system_label_order
from .data import CircularTarget, image_path, iter_targets, load_datasets
from .model import call_llm_once
from .prompts import generate_prompt
from .visual import draw_angle_feedback


EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "with_grid"),
]
PREFERRED_PROMPTS = ["amplifier", "feedback", "grid", "whole_chart"]
CHART_TYPE = "donut"
CIRCULAR_STABLE_SHARE_TOLERANCE = 2.0 / 360.0


async def _run_target(dataset: dict[str, Any], target: CircularTarget, repeat_times: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for prompt_type, image_type in EXPERIMENT_TYPES:
        for run_idx in range(1, repeat_times + 1):
            used_image = image_path(dataset, image_type)
            prompt = generate_prompt(
                target.point_name,
                prompt_type,
                dataset.get("theta_ticks") if prompt_type == "baseline" else None,
                series_color=dataset.get("series_color"),
            )
            print(f"[donut] Round {run_idx} | Segment: {target.point_name} | {prompt_type} - {image_type}")
            token = set_modal_call_context(
                _modal_context(dataset, target.point_name, prompt_type, run_idx, used_image)
            )
            try:
                raw_pred = await call_llm_once(prompt=prompt, image_path=str(used_image), item_name=target.point_name)
                call_id = get_last_modal_call_id()
            finally:
                reset_modal_call_context(token)
            prediction = _prediction_from_raw(raw_pred)
            records.append(
                {
                    "chart_id": dataset["chart_id"],
                    "call_id": call_id,
                    "point": target.point_name,
                    "prompt_type": prompt_type,
                    "image_type": image_type,
                    "run": run_idx,
                    "image_path": str(used_image),
                    "pred": prediction.get("value"),
                    "pred_pct": prediction.get("value"),
                    "percentage": prediction.get("percentage"),
                    "start_angle": prediction.get("start_angle"),
                    "end_angle": prediction.get("end_angle"),
                    "raw_prediction": json.dumps(raw_pred, ensure_ascii=False),
                }
            )
    return records


async def _run_baseline_target(dataset: dict[str, Any], target: CircularTarget) -> list[dict[str, Any]]:
    used_image = image_path(dataset, "no_grid")
    prompt = generate_prompt(target.point_name, "baseline", series_color=dataset.get("series_color"))
    print(f"[donut] Baseline | Segment: {target.point_name}")
    token = set_modal_call_context(
        _modal_context(dataset, target.point_name, "baseline", 1, used_image)
    )
    try:
        raw_pred = await call_llm_once(prompt=prompt, image_path=str(used_image), item_name=target.point_name)
        call_id = get_last_modal_call_id()
    finally:
        reset_modal_call_context(token)
    prediction = _prediction_from_raw(raw_pred)
    return [
        _record_from_prediction(
            dataset,
            target.point_name,
            "baseline",
            "no_grid",
            1,
            used_image,
            prediction,
            raw_pred,
            call_id=call_id,
        )
    ]


async def _run_refined_target(
    dataset: dict[str, Any],
    target: CircularTarget,
    result_dir: Path,
    initial_prediction: dict[str, Any] | None,
    repeat_times: int,
) -> list[dict[str, Any]]:
    """Run the reference-style donut chain for one system-generated label."""

    records: list[dict[str, Any]] = []
    chart_id = str(dataset["chart_id"])
    if initial_prediction is not None:
        records.append(
            _record_from_prediction(
                dataset,
                target.point_name,
                "whole_chart",
                "with_grid",
                1,
                initial_prediction.get("image_path"),
                initial_prediction,
                initial_prediction,
            )
        )

    try:
        with_grid_image = image_path(dataset, "with_grid")
        no_grid_image = image_path(dataset, "no_grid")
    except Exception as exc:
        print(f"[donut] Missing image path for {target.point_name}: {exc}")
        return records

    center, inner_radius, outer_radius = _donut_geometry(dataset, no_grid_image)
    grid_image = draw_circular_grid_image(
        source=no_grid_image,
        output_path=result_dir / "grid_img" / f"{_safe_name(target.point_name)}_grid15.png",
        center=center,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        interval_deg=15,
    )
    _add_target_color_swatch(grid_image, dataset, target.point_name)

    prompt = generate_prompt(
        item_name=target.point_name,
        prompt_type="grid",
        prev_angle=None,
        series_color=dataset.get("series_color"),
    )
    print(f"[donut] {chart_id} | {target.point_name} | grid round 1")
    token = set_modal_call_context(
        _modal_context(dataset, target.point_name, "grid", 1, grid_image)
    )
    try:
        raw_pred = await call_llm_once(prompt=prompt, image_path=str(grid_image), item_name=target.point_name)
        call_id = get_last_modal_call_id()
    finally:
        reset_modal_call_context(token)
    grid_prediction = _prediction_from_raw(raw_pred)
    records.append(
        _record_from_prediction(
            dataset,
            target.point_name,
            "grid",
            "with_grid",
            1,
            grid_image,
            grid_prediction,
            raw_pred,
            call_id=call_id,
        )
    )
    last_pred = _angles_from_prediction(grid_prediction) or _angles_from_prediction(initial_prediction)
    last_prediction = grid_prediction
    if not _has_angles(last_pred):
        return records

    for feedback_round in range(1, get_feedback_rounds(2) + 1):
        try:
            feedback_image = draw_angle_feedback(
                image_path=str(grid_image),
                angle_deg=[float(last_pred["start_angle"]), float(last_pred["end_angle"])],
                output_path=str(result_dir / f"{_safe_name(target.point_name)}_feedback_round{feedback_round}.png"),
                circle_center=center,
                inner_radius=outer_radius,
            )
        except Exception as exc:
            print(f"[donut] Feedback image failed for {target.point_name}: {exc}")
            break

        previous_prediction = last_prediction
        prompt = generate_prompt(
            item_name=target.point_name,
            prompt_type="feedback",
            prev_angle=last_pred,
            series_color=dataset.get("series_color"),
        )
        print(f"[donut] {chart_id} | {target.point_name} | feedback round {feedback_round}")
        token = set_modal_call_context(
            _modal_context(dataset, target.point_name, "feedback", feedback_round, feedback_image)
        )
        try:
            raw_pred = await call_llm_once(prompt=prompt, image_path=feedback_image, item_name=target.point_name)
            call_id = get_last_modal_call_id()
        finally:
            reset_modal_call_context(token)
        prediction = _prediction_from_raw(raw_pred)
        records.append(
            _record_from_prediction(
                dataset,
                target.point_name,
                "feedback",
                "feedback",
                feedback_round,
                feedback_image,
                prediction,
                raw_pred,
                call_id=call_id,
            )
        )
        if angle_prediction_stable(prediction, previous_prediction):
            print(f"[donut] {chart_id} | {target.point_name} | feedback stabilized within 2 deg; enter amplifier.")
            if _has_angles(prediction):
                last_pred = _angles_from_prediction(prediction)
                last_prediction = prediction
            break
        if _has_angles(prediction):
            last_pred = _angles_from_prediction(prediction)
            last_prediction = prediction
        else:
            break

    if not _has_angles(last_pred):
        return records

    amp_pred = dict(last_pred)
    previous_amp_prediction = last_prediction
    for amp_round in range(1, get_amplifier_rounds(3) + 1):
        try:
            amp_image, drawn_angles = crop_circular_amplifier_image(
                source=no_grid_image,
                output_path=result_dir / "amplifier_img" / f"{_safe_name(target.point_name)}_amp{amp_round}.png",
                center=center,
                inner_radius=inner_radius,
                outer_radius=outer_radius,
                start_angle=float(amp_pred["start_angle"]),
                end_angle=float(amp_pred["end_angle"]),
                round_index=amp_round,
            )
            _add_target_color_swatch(amp_image, dataset, target.point_name)
        except Exception as exc:
            print(f"[donut] Amplifier crop failed for {target.point_name}, round {amp_round}: {exc}")
            break

        prompt = generate_prompt(
            item_name=target.point_name,
            prompt_type="amplifier",
            prev_angle=amp_pred,
            drawn_angles=drawn_angles,
            angle_order_hint=None,
            series_color=dataset.get("series_color"),
        )
        print(f"[donut] {chart_id} | {target.point_name} | amplifier round {amp_round}")
        token = set_modal_call_context(
            _modal_context(dataset, target.point_name, "amplifier", amp_round, amp_image)
        )
        try:
            raw_pred = await call_llm_once(prompt=prompt, image_path=str(amp_image), item_name=target.point_name)
            call_id = get_last_modal_call_id()
        finally:
            reset_modal_call_context(token)
        prediction = _prediction_from_raw(raw_pred)
        records.append(
            _record_from_prediction(
                dataset,
                target.point_name,
                "amplifier",
                "no_grid",
                amp_round,
                amp_image,
                prediction,
                raw_pred,
                call_id=call_id,
            )
        )
        is_stable = angle_prediction_stable(prediction, previous_amp_prediction)
        if not _has_angles(prediction):
            break
        amp_pred = {
            "start_angle": float(prediction["start_angle"]),
            "end_angle": float(prediction["end_angle"]),
        }
        previous_amp_prediction = prediction
        if is_stable:
            print(f"[donut] {chart_id} | {target.point_name} | amplifier prediction stabilized within 2 deg; stop refinement.")
            break

    return records


def _prediction_from_raw(raw_pred: Any) -> dict[str, float | None]:
    if isinstance(raw_pred, dict):
        explicit = _number_or_none(raw_pred.get("percentage") if "percentage" in raw_pred else raw_pred.get("value"))
        if explicit is not None:
            value = explicit / 100.0 if abs(explicit) > 1.0 else explicit
            return {
                "value": value,
                "percentage": value * 100.0,
                "start_angle": _number_or_none(raw_pred.get("start_angle")),
                "end_angle": _number_or_none(raw_pred.get("end_angle")),
            }
    if isinstance(raw_pred, dict) and "start_angle" in raw_pred and "end_angle" in raw_pred:
        try:
            start = float(raw_pred["start_angle"])
            end = float(raw_pred["end_angle"])
            value = ((end - start + 360.0) % 360.0) / 360.0
            return {
                "value": value,
                "percentage": value * 100.0,
                "start_angle": start,
                "end_angle": end,
            }
        except Exception:
            return {"value": None, "percentage": None, "start_angle": None, "end_angle": None}
    try:
        number = float(raw_pred)
    except Exception:
        return {"value": None, "percentage": None, "start_angle": None, "end_angle": None}
    value = number / 100.0 if abs(number) > 1.0 else number
    return {"value": value, "percentage": value * 100.0, "start_angle": None, "end_angle": None}


def _record_from_prediction(
    dataset: dict[str, Any],
    point: str,
    prompt_type: str,
    image_type: str,
    run: int,
    used_image: Any,
    prediction: dict[str, Any],
    raw_pred: Any,
    call_id: str | None = None,
) -> dict[str, Any]:
    value = prediction.get("value") if isinstance(prediction, dict) else None
    return {
        "chart_id": dataset["chart_id"],
        "call_id": call_id,
        "point": point,
        "prompt_type": prompt_type,
        "image_type": image_type,
        "run": run,
        "image_path": str(used_image) if used_image else None,
        "pred": value,
        "pred_pct": value,
        "percentage": prediction.get("percentage") if isinstance(prediction, dict) else None,
        "start_angle": prediction.get("start_angle") if isinstance(prediction, dict) else None,
        "end_angle": prediction.get("end_angle") if isinstance(prediction, dict) else None,
        "raw_prediction": json.dumps(raw_pred, ensure_ascii=False),
    }


def _has_angles(prediction: Any) -> bool:
    if not isinstance(prediction, dict):
        return False
    return _number_or_none(prediction.get("start_angle")) is not None and _number_or_none(prediction.get("end_angle")) is not None


def _angles_from_prediction(prediction: dict[str, Any] | None) -> dict[str, float] | None:
    if not _has_angles(prediction):
        return None
    return {
        "start_angle": float(prediction["start_angle"]),
        "end_angle": float(prediction["end_angle"]),
    }


def _prediction_value(record: dict[str, Any]) -> float | None:
    try:
        value = float(record["pred_pct"])
        return value if value == value else None
    except Exception:
        return None


def _select_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if _prediction_value(record) is not None:
            by_point.setdefault(str(record["point"]), []).append(record)

    predictions: list[dict[str, Any]] = []
    for point, point_records in by_point.items():
        chosen, value, reason = _select_circular_record(point_records)
        if chosen is None:
            continue
        if value is None:
            continue
        predictions.append(
            {
                "id": point,
                "series_name": "",
                "label": point,
                "axis": "theta",
                "value": value,
                "percentage": value * 100.0,
                "start_angle": chosen.get("start_angle"),
                "end_angle": chosen.get("end_angle"),
                "prompt_type": chosen.get("prompt_type"),
                "image_type": chosen.get("image_type"),
                "image_path": chosen.get("image_path"),
                "extraction_source": reason,
            }
        )
    return predictions


def _select_circular_record(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float | None, str]:
    full_records = [record for record in records if record.get("prompt_type") in {"grid", "feedback", "amplifier"}]
    readable = [(record, _prediction_value(record)) for record in full_records]
    readable = [(record, value) for record, value in readable if value is not None]
    if not readable:
        return None, None, "no_readable_full_flow_prediction"

    amp = [(record, value) for record, value in readable if record.get("prompt_type") == "amplifier"]
    feedback = [(record, value) for record, value in readable if record.get("prompt_type") == "feedback"]
    grid = [(record, value) for record, value in readable if record.get("prompt_type") == "grid"]
    stable_ref = _stable_circular_grid_feedback(grid, feedback)
    grid_reference = _grid_reference_after_feedback_drift(grid, feedback, amp)

    if len(amp) >= 2 and _circular_close(amp[-1][1], amp[-2][1]):
        value = _median_float([amp[-2][1], amp[-1][1]])
        if grid_reference is not None and abs(value - grid_reference[1]) > 0.06:
            return grid_reference[0], grid_reference[1], "grid_reference_after_feedback_amplifier_drift"
        if stable_ref is not None and abs(value - stable_ref) > 0.08:
            return feedback[-1][0], stable_ref, "stable_grid_feedback_amplifier_outlier"
        return amp[-1][0], value, "stable_amplifier_tail"
    if grid_reference is not None:
        return grid_reference[0], grid_reference[1], "grid_reference_after_feedback_drift"
    if amp:
        amp_value = amp[-1][1]
        refs = [value for _record, value in (feedback[-1:] + grid[-1:])]
        if any(_circular_close(amp_value, ref) for ref in refs):
            return amp[-1][0], amp_value, "amplifier_agrees_with_grid_or_feedback"
    if len(amp) >= 2 and feedback:
        amp_values = [value for _record, value in amp]
        if max(amp_values) - min(amp_values) > 0.12:
            return feedback[-1][0], feedback[-1][1], "unstable_amplifier_use_latest_feedback"
    if stable_ref is not None:
        return feedback[-1][0], stable_ref, "stable_grid_feedback"

    values = [value for _record, value in readable]
    value = _median_float(values)
    chosen = min(readable, key=lambda item: abs(item[1] - value))[0]
    return chosen, value, "median_all_full_flow"


def _stable_circular_grid_feedback(
    grid: list[tuple[dict[str, Any], float]],
    feedback: list[tuple[dict[str, Any], float]],
) -> float | None:
    if not grid or not feedback:
        return None
    grid_value = grid[-1][1]
    feedback_value = feedback[-1][1]
    if not _circular_close(grid_value, feedback_value):
        return None
    return _median_float([grid_value, feedback_value])


def _grid_reference_after_feedback_drift(
    grid: list[tuple[dict[str, Any], float]],
    feedback: list[tuple[dict[str, Any], float]],
    amp: list[tuple[dict[str, Any], float]],
) -> tuple[dict[str, Any], float] | None:
    if not grid or not feedback:
        return None
    grid_value = grid[-1][1]
    feedback_values = [value for _record, value in feedback]
    if not feedback_values or abs(feedback_values[-1] - grid_value) <= 0.08:
        return None
    later_values = feedback_values + [value for _record, value in amp]
    far_later = [value for value in later_values if abs(value - grid_value) > 0.08]
    if len(far_later) >= max(2, len(later_values) - 1):
        return grid[-1][0], grid_value
    return None


def _circular_close(current: float, reference: float) -> bool:
    return abs(current - reference) <= CIRCULAR_STABLE_SHARE_TOLERANCE


def _median_float(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


async def _extract_all_segments(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    used_image = image_path(dataset, "with_grid")
    labels = system_label_order(dataset)
    label_text = ", ".join(json.dumps(label, ensure_ascii=False) for label in labels)
    label_rule = (
        f"Use exactly these labels and return one datapoint for each of them, in this order: [{label_text}]."
        if labels
        else "Use visible legend or slice labels when available."
    )
    prompt = """
You are analyzing a donut chart with angular reference lines.
Identify every visible ring sector and estimate its share of the whole chart.

Return only JSON in this exact shape:
{{"datapoints":[{{"name":"A","percentage":20,"start_angle":0,"end_angle":72}}]}}

Rules:
1. {label_rule}
2. If labels are not visible, name sectors Segment 1, Segment 2, ... in clockwise order.
3. percentage is a number from 0 to 100.
4. Angles are optional; include them only when you can estimate them from the grid.
""".format(label_rule=label_rule)
    timeout = ClientTimeout(total=300)
    token = set_modal_call_context(
        _modal_context(dataset, "__whole_chart__", "whole_chart", 1, used_image)
    )
    try:
        text = await chat_with_gemini(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(used_image)}},
                    ],
                }
            ],
            model=get_model_name(),
            max_tokens=4096,
            temperature=0.0,
            timeout=timeout,
        )
    finally:
        reset_modal_call_context(token)
    if text == FAILURE_TEXT:
        return []
    parsed = _parse_json_object(text)
    items = parsed.get("datapoints") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []
    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("name") or item.get("label") or f"Segment {index}").strip()
        value = _percentage_to_value(item.get("percentage") if "percentage" in item else item.get(label))
        if value is None:
            continue
        predictions.append(
            {
                "id": label,
                "series_name": "",
                "label": label,
                "axis": "theta",
                "value": value,
                "percentage": value * 100.0,
                "start_angle": _number_or_none(item.get("start_angle")),
                "end_angle": _number_or_none(item.get("end_angle")),
                "prompt_type": "whole_chart",
                "image_type": "with_grid",
                "image_path": str(used_image),
            }
        )
    return predictions


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```json", "", text.strip())
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _percentage_to_value(raw: Any) -> float | None:
    number = _number_or_none(raw)
    if number is None:
        return None
    return number / 100.0 if abs(number) > 1.0 else number


def _number_or_none(raw: Any) -> float | None:
    try:
        value = float(str(raw).strip().rstrip("%"))
        return value if value == value else None
    except Exception:
        return None


def _target_gt(dataset: dict[str, Any], label: str) -> Any:
    data_points = dataset.get("data_points")
    if isinstance(data_points, dict):
        return data_points.get(label)
    return None


def _prediction_consistent(current: dict[str, Any] | None, reference: dict[str, Any] | None) -> bool:
    return angle_prediction_stable(current, reference)


def _modal_context(
    dataset: dict[str, Any],
    processing_object: str,
    stage: str,
    round_index: int,
    image_path: Any,
) -> dict[str, Any]:
    return {
        "chart_name": dataset.get("chart_id"),
        "processing_object": processing_object,
        "object_category": "",
        "gt": dataset.get("data_points") if processing_object == "__whole_chart__" else _target_gt(dataset, processing_object),
        "stage": stage,
        "round": round_index,
        "image_path": str(image_path),
    }


def _donut_geometry(dataset: dict[str, Any], source_image: Path) -> tuple[tuple[int, int], int, int]:
    with Image.open(source_image) as img:
        width, height = img.size

    raw_center = dataset.get("center")
    if isinstance(raw_center, dict):
        center = (int(raw_center.get("x", width // 2)), int(raw_center.get("y", height // 2)))
    elif isinstance(raw_center, (list, tuple)) and len(raw_center) >= 2:
        center = (int(raw_center[0]), int(raw_center[1]))
    else:
        center = (width // 2, height // 2)

    raw_radius = dataset.get("r_pixels") or dataset.get("radius")
    if isinstance(raw_radius, (list, tuple)):
        radius_values = [_number_or_none(value) for value in raw_radius]
        valid_radii = [int(value) for value in radius_values if value is not None and value > 0]
        outer_radius = max(valid_radii) if valid_radii else int(min(width, height) * 0.35)
        inner_radius = min(valid_radii) if len(valid_radii) >= 2 else int(outer_radius * 0.45)
    else:
        outer_radius = int(_number_or_none(raw_radius) or min(width, height) * 0.35)
        inner_radius = int(_number_or_none(dataset.get("inner_radius")) or _number_or_none(dataset.get("hole_radius")) or outer_radius * 0.45)

    inner_radius = max(1, min(inner_radius, outer_radius - 1))
    outer_radius = max(inner_radius + 1, outer_radius)
    return center, inner_radius, outer_radius


def _add_target_color_swatch(path: Path, dataset: dict[str, Any], point_name: str) -> None:
    series_color = dataset.get("series_color")
    if not isinstance(series_color, dict):
        return
    color = series_color.get(point_name)
    if not color:
        return
    try:
        with Image.open(path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            box_w = min(152, max(96, width // 2))
            box_h = 30
            x0 = 6
            y0 = max(6, height - box_h - 6)
            draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill="white", outline="black", width=1)
            draw.rectangle((x0 + 5, y0 + 5, x0 + 29, y0 + 25), fill=str(color), outline="black", width=1)
            draw.text((x0 + 36, y0 + 8), f"target: {point_name}", fill="black")
            image.save(path)
    except Exception as exc:
        print(f"[donut] Target color swatch skipped for {point_name}: {exc}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "segment"


def _chart_result_dir(chart_id: str) -> Path:
    return ensure_dir(RESULTS_ROOT / CHART_TYPE / chart_id)


def _save_records(records: list[dict[str, Any]], result_dir: Path) -> None:
    ensure_dir(result_dir)
    if not records:
        return
    pd.DataFrame(records).to_csv(result_dir / "experiment_results.csv", index=False, encoding="utf-8")


async def run_experiment(
    batch_size: int | None = None,
    chart_ids: list[str] | None = None,
    config_paths: list[str | Path] | None = None,
) -> list[dict[str, Any]]:
    datasets = load_datasets(chart_ids, config_paths=config_paths)
    if not datasets:
        print("[donut] No matching chart configs. Nothing to run.")
        return []

    repeat_times = get_repeat_times()
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        result_dir = _chart_result_dir(str(dataset["chart_id"]))
        targets = iter_targets(dataset)
        records: list[dict[str, Any]] = []
        if targets:
            for target in targets:
                records.extend(await _run_baseline_target(dataset, target))
                refined_records = await _run_refined_target(
                    dataset=dataset,
                    target=target,
                    result_dir=result_dir,
                    initial_prediction=None,
                    repeat_times=repeat_times,
                )
                records.extend(refined_records)

            predictions = _select_predictions(records)
            predictions = complete_circular_predictions(dataset, CHART_TYPE, predictions)
            predictions = normalize_circular_prediction_shares(predictions)
        else:
            predictions = await _extract_all_segments(dataset)
            predictions = complete_circular_predictions(dataset, CHART_TYPE, predictions)
            predictions = normalize_circular_prediction_shares(predictions)
            records = [
                {
                    "chart_id": dataset["chart_id"],
                    "point": item["label"],
                    "prompt_type": "whole_chart",
                    "image_type": "with_grid",
                    "run": 1,
                    "image_path": item.get("image_path"),
                    "pred": item.get("value"),
                    "pred_pct": item.get("value"),
                    "percentage": item.get("percentage"),
                    "start_angle": item.get("start_angle"),
                    "end_angle": item.get("end_angle"),
                    "raw_prediction": "",
                }
                for item in predictions
            ]
        records = save_circular_artifacts(
            dataset=dataset,
            chart_type=CHART_TYPE,
            result_dir=result_dir,
            records=records,
            predictions=predictions,
        )
        summaries.append(
            {
                "chart_id": dataset["chart_id"],
                "result_dir": str(result_dir),
                "record_count": len(records),
                "object_count": len(predictions),
                "predictions": predictions,
            }
        )
    return summaries
