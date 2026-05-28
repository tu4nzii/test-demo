from __future__ import annotations

from math import isfinite
from statistics import mean
from typing import Iterable, List, Optional, Sequence


def to_float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value.strip().rstrip("%"))
        except ValueError:
            return None
        return number if isfinite(number) else None
    return None


def absolute_error(predicted: object, expected: object) -> Optional[float]:
    pred = to_float(predicted)
    gt = to_float(expected)
    if pred is None or gt is None:
        return None
    return abs(pred - gt)


def relative_error(predicted: object, expected: object) -> Optional[float]:
    abs_err = absolute_error(predicted, expected)
    gt = to_float(expected)
    if abs_err is None or gt is None or gt == 0:
        return None
    return abs_err / abs(gt)


def vector_mae(predicted: Sequence[object], expected: Sequence[object]) -> Optional[float]:
    if len(predicted) != len(expected):
        return None

    errors: List[float] = []
    for pred_item, gt_item in zip(predicted, expected):
        err = absolute_error(pred_item, gt_item)
        if err is None:
            return None
        errors.append(err)
    return mean(errors) if errors else None


def vector_relative_error(predicted: Sequence[object], expected: Sequence[object]) -> Optional[float]:
    if len(predicted) != len(expected):
        return None

    errors: List[float] = []
    for pred_item, gt_item in zip(predicted, expected):
        err = relative_error(pred_item, gt_item)
        if err is not None:
            errors.append(err)
    return mean(errors) if errors else None


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None and isfinite(value)]
    return mean(numeric) if numeric else None


def round_metric(value: Optional[float], digits: int = 6) -> Optional[float]:
    return round(value, digits) if value is not None and isfinite(value) else None
