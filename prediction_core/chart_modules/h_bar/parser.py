"""Model-output parsing for horizontal bar value prediction."""

from __future__ import annotations

import json
import re
from typing import Any

from prediction_core.json_utils import safe_json_loads as core_safe_json_loads


FAILED_COORDS = (-1, -1)


def safe_json_loads(text: str) -> Any | None:
    return core_safe_json_loads(text)


def _maybe_float(value: Any) -> Any:
    try:
        return float(value) if isinstance(value, (str, int, float)) else value
    except Exception:
        return value


def _coords_from_pair(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            value = value[0]
        if len(value) >= 2:
            return (_maybe_float(value[0]), value[1])
    return None


def _key_matches(key: Any, point_name: str) -> bool:
    return isinstance(key, str) and point_name.strip().lower() in key.strip().lower()


def _extract_from_mapping(data: dict[str, Any], point_name: str) -> tuple[Any, Any] | None:
    if "response" in data:
        response = data["response"]
        if isinstance(response, str):
            parsed = safe_json_loads(response)
            if parsed is not None:
                return _extract(parsed, point_name)
        elif isinstance(response, dict):
            return _extract(response, point_name)

    if "x" in data and "y" in data:
        return (_maybe_float(data["x"]), data["y"])

    datapoints = data.get("datapoints", data)
    if isinstance(datapoints, dict):
        for key, value in datapoints.items():
            if _key_matches(key, point_name):
                pair = _coords_from_pair(value)
                if pair is not None:
                    return pair
    elif isinstance(datapoints, list):
        return _extract(datapoints, point_name)
    elif isinstance(datapoints, (int, float)):
        return (float(datapoints), "")
    elif isinstance(datapoints, str) and re.match(r"^[\d.]+$", datapoints):
        return (float(datapoints), "")

    return None


def _extract_from_sequence(data: list[Any] | tuple[Any, ...], point_name: str) -> tuple[Any, Any] | None:
    direct_pair = _coords_from_pair(data)
    if direct_pair is not None and "datapoints" not in str(data):
        return direct_pair

    for item in data:
        if isinstance(item, dict):
            if "x" in item and "y" in item:
                return (_maybe_float(item["x"]), item["y"])
            if len(item) == 1:
                key = next(iter(item))
                if _key_matches(key, point_name):
                    pair = _coords_from_pair(item[key])
                    if pair is not None:
                        return pair
            nested = _extract_from_mapping(item, point_name)
            if nested is not None:
                return nested
    return None


def _extract(data: Any, point_name: str) -> tuple[Any, Any] | None:
    if isinstance(data, dict):
        return _extract_from_mapping(data, point_name)
    if isinstance(data, (list, tuple)):
        return _extract_from_sequence(data, point_name)
    return None


def extract_coords(coords_json: Any, point_name: str) -> tuple[Any, Any]:
    if coords_json is None:
        print("[h_bar parser] Empty JSON payload.")
        return FAILED_COORDS

    print(
        "[h_bar parser] input="
        f"{type(coords_json).__name__} {json.dumps(coords_json, ensure_ascii=False)[:200]}"
    )
    result = _extract(coords_json, point_name)
    if result is None:
        print(
            "[h_bar parser] No matching coordinates found "
            f"for point_name={point_name!r}, payload_type={type(coords_json).__name__}"
        )
        return FAILED_COORDS
    return result
