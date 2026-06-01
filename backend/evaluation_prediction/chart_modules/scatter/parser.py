"""Model-output parsing for scatter charts."""

from __future__ import annotations

import json
from typing import Any

from ...common.json_utils import safe_json_loads as core_safe_json_loads


FAILED_COORDS = (-1, -1)


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
            return (_maybe_float(value[0]), _maybe_float(value[1]))
    return None


def _key_matches(key: Any, point_name: str) -> bool:
    return isinstance(key, str) and key.strip().lower() == point_name.strip().lower()


def _extract_mapping(data: dict[str, Any], point_name: str) -> tuple[Any, Any] | None:
    if "response" in data:
        response = data["response"]
        if isinstance(response, str):
            parsed = safe_json_loads(response)
            if parsed is not None:
                return _extract(parsed, point_name)
        elif isinstance(response, dict):
            return _extract(response, point_name)

    datapoints = data.get("datapoints", data)
    if isinstance(datapoints, dict):
        for key, value in datapoints.items():
            if _key_matches(key, point_name):
                pair = _pair(value)
                if pair is not None:
                    return pair
    elif isinstance(datapoints, list):
        return _extract_sequence(datapoints, point_name)

    if data.get("label") == point_name and "point" in data:
        return _pair(data["point"])
    return None


def _extract_sequence(data: list[Any] | tuple[Any, ...], point_name: str) -> tuple[Any, Any] | None:
    direct_pair = _pair(data)
    if direct_pair is not None and len(data) == 2 and not isinstance(data[0], dict):
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
        print("[scatter parser] Empty JSON payload.")
        return FAILED_COORDS
    print("[scatter parser] input=" f"{type(coords_json).__name__} {json.dumps(coords_json, ensure_ascii=False)[:200]}")
    result = _extract(coords_json, point_name)
    if result is None:
        print(f"[scatter parser] No matching coordinates found for {point_name!r}")
        return FAILED_COORDS
    return result
