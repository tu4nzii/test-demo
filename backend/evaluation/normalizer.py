from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class DataRecord:
    key: Tuple[str, ...]
    value: Any

    @property
    def id(self) -> str:
        return " / ".join(self.key)


def first_mapping(data: Mapping[str, Any], keys: Iterable[str]) -> Optional[Mapping[str, Any]]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def get_ground_truth(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return first_mapping(data, ("ground_truth", "data_points", "data")) or {}


def get_predictions(data: Mapping[str, Any]) -> Mapping[str, Any]:
    return first_mapping(
        data,
        (
            "predictions",
            "prediction",
            "predicted_data",
            "prediction_data",
            "extracted_data",
            "estimated_data",
        ),
    ) or {}


def flatten_data_points(value: Any, prefix: Tuple[str, ...] = ()) -> List[DataRecord]:
    if isinstance(value, Mapping):
        records: List[DataRecord] = []
        for key, nested_value in value.items():
            records.extend(flatten_data_points(nested_value, prefix + (str(key),)))
        return records

    if isinstance(value, list):
        if _is_scalar_vector(value):
            return [DataRecord(prefix, value)]

        records = []
        for index, nested_value in enumerate(value):
            records.extend(flatten_data_points(nested_value, prefix + (str(index),)))
        return records

    return [DataRecord(prefix, value)]


def records_by_id(records: Iterable[DataRecord]) -> Dict[str, DataRecord]:
    return {record.id: record for record in records}


def _is_scalar_vector(value: List[Any]) -> bool:
    return all(not isinstance(item, (Mapping, list)) for item in value)
