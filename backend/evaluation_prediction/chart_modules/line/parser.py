"""Model-output parsing for line chart value prediction."""

from __future__ import annotations

import json
from typing import Any

from ...common.json_utils import safe_json_loads as core_safe_json_loads

from .prompts import split_item_name


FAILED_COORDS = ("", -1)


def safe_json_loads(text: str) -> Any | None:
    return core_safe_json_loads(text)


def _maybe_float(value: Any) -> Any:
    try:
        return float(value) if isinstance(value, (str, int, float)) else value
    except Exception:
        return value


def _pair(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            value = value[0]
        if len(value) >= 2:
            return (str(value[0]), _maybe_float(value[1]))
    if isinstance(value, (int, float, str)):
        number = _maybe_float(value)
        if isinstance(number, float):
            return ("", number)
    return None


def _key_matches(key: Any, point_name: str) -> bool:
    return isinstance(key, str) and point_name.strip().lower() in key.strip().lower()


def _extract_series_payload(data: dict[str, Any], point_name: str) -> tuple[Any, Any] | None:
    series_name, x_label = split_item_name(point_name)
    datapoints = data.get("datapoints", data)
    if not isinstance(datapoints, dict):
        return None
    payload = datapoints.get(series_name)
    if isinstance(payload, dict):
        xs = payload.get("x")
        ys = payload.get("y")
        if isinstance(xs, list) and isinstance(ys, list):
            for x, y in zip(xs, ys):
                if str(x).strip() == x_label:
                    return (str(x), _maybe_float(y))
        if x_label in payload:
            return (x_label, _maybe_float(payload[x_label]))
    return None


def _extract_mapping(data: dict[str, Any], point_name: str) -> tuple[Any, Any] | None:
    if "response" in data:
        response = data["response"]
        if isinstance(response, str):
            parsed = safe_json_loads(response)
            if parsed is not None:
                return _extract(parsed, point_name)
        elif isinstance(response, dict):
            return _extract(response, point_name)

    direct_series = _extract_series_payload(data, point_name)
    if direct_series is not None:
        return direct_series

    datapoints = data.get("datapoints", data)
    if isinstance(datapoints, dict):
        for key, value in datapoints.items():
            if _key_matches(key, point_name):
                pair = _pair(value)
                if pair is not None:
                    return pair
    elif isinstance(datapoints, list):
        return _extract_sequence(datapoints, point_name)
    return None


def _extract_sequence(data: list[Any] | tuple[Any, ...], point_name: str) -> tuple[Any, Any] | None:
    direct_pair = _pair(data)
    if direct_pair is not None and "datapoints" not in str(data):
        return direct_pair
    for item in data:
        if isinstance(item, dict):
            if len(item) == 1:
                key = next(iter(item))
                if _key_matches(key, point_name):
                    pair = _pair(item[key])
                    if pair is not None:
                        return pair
            nested = _extract_mapping(item, point_name)
            if nested is not None:
                return nested
    return None


def _extract(data: Any, point_name: str) -> tuple[Any, Any] | None:
    if isinstance(data, dict):
        return _extract_mapping(data, point_name)
    if isinstance(data, (list, tuple)):
        return _extract_sequence(data, point_name)
    return None


def extract_coords(coords_json: Any, point_name: str) -> tuple[Any, Any]:
    if coords_json is None:
        print("[line parser] Empty JSON payload.")
        return FAILED_COORDS
    print(
        "[line parser] input="
        f"{type(coords_json).__name__} {json.dumps(coords_json, ensure_ascii=False)[:200]}"
    )
    result = _extract(coords_json, point_name)
    if result is None:
        print(f"[line parser] No matching coordinates found for {point_name!r}")
        return FAILED_COORDS
    return result
