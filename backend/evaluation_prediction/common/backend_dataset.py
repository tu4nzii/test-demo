"""Adapters for backend-generated chart JSON used by prediction flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import unicodedata

from PIL import Image

from .chart_io import read_json
from .paths import ASSETS_ROOT


def load_backend_generated_dataset(config_path: str | Path, chart_type: str) -> dict[str, Any]:
    """Load the JSON emitted by backend encryption and merge its tick sidecar.

    The backend writes the user-facing chart payload to ``*_image.json`` and the
    grid/tick metadata to ``*_image_ticks.json``. Prediction needs both.
    """
    path = Path(config_path).resolve()
    base = _read_json_dict(path)
    ticks = _read_json_dict(_sibling_ticks_path(path))
    if path.stem.endswith("_ticks"):
        base = _read_json_dict(_sibling_chart_path(path))
        ticks = _read_json_dict(path)

    merged = dict(base)
    merged.update(ticks)
    merged["chart_type"] = chart_type
    merged["chart_id"] = str(base.get("chart_id") or ticks.get("chart_id") or path.stem)
    _strip_external_reference_data(merged)
    # Backend upload prediction must not consume GT/reference data. Targets are
    # derived from the current system output: ticks plus extracted colors/labels.
    merged["data_points"] = {}
    merged["series_color"] = _series_color(merged)
    merged["image_paths"] = _image_paths(merged, path.parent)

    _prefer_encrypted_numeric_axis(merged, chart_type)
    _prefer_data_categories(merged, chart_type)
    return merged


def load_backend_generated_datasets(
    config_paths: Iterable[str | Path],
    chart_type: str,
    chart_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    wanted = set(chart_ids or [])
    datasets = [load_backend_generated_dataset(path, chart_type) for path in config_paths]
    if wanted:
        datasets = [dataset for dataset in datasets if dataset.get("chart_id") in wanted]
    return datasets


def resolve_image_path(dataset: dict[str, Any], image_type: str) -> Path:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}
    value = image_paths.get(image_type)
    if value is None and image_type == "grid_with_grid":
        value = image_paths.get("with_grid") or dataset.get("encrypted_grid_path")
    if value is None and image_type == "with_grid":
        value = image_paths.get("grid_with_grid") or dataset.get("encrypted_grid_path")
    if value is None and image_type == "no_grid":
        value = dataset.get("image_path")
    if not value:
        raise KeyError(f"Missing image path for {image_type!r} in {dataset.get('chart_id')!r}")
    return Path(value).resolve()


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def _sibling_ticks_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_ticks.json")


def _sibling_chart_path(path: Path) -> Path:
    stem = path.stem.removesuffix("_ticks")
    return path.with_name(f"{stem}.json")


def _data_points(dataset: dict[str, Any]) -> dict[str, Any]:
    for key in ("data_points", "data", "ground_truth"):
        value = dataset.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _strip_external_reference_data(dataset: dict[str, Any]) -> None:
    dataset.pop("reference_config_path", None)
    dataset.pop("reference_chart_id", None)
    for key in ("data", "data_points", "ground_truth", "series_color", "labels"):
        dataset.pop(key, None)


def _series_color(dataset: dict[str, Any]) -> dict[str, str]:
    colors = dataset.get("colors")
    if not isinstance(colors, list):
        return {}

    result: dict[str, str] = {}
    for index, item in enumerate(colors):
        if isinstance(item, dict) and item.get("name") and item.get("color"):
            name = _normalize_series_name(item.get("name"), index)
            result[name] = str(item["color"])
    return result


def _normalize_series_name(name: Any, index: int) -> str:
    text = str(name or "").strip()
    lowered = text.lower()
    if (
        not text
        or lowered in {"none", "null", "nan", "series-0"}
        or _looks_like_mojibake(text)
    ):
        return f"Series {index + 1}"
    return text


def _looks_like_mojibake(text: str) -> bool:
    if "\ufffd" in text:
        return True
    if any(unicodedata.category(char) == "Co" for char in text):
        return True
    non_ascii = sum(1 for char in text if ord(char) > 127)
    question_marks = text.count("?")
    return question_marks > 0 and non_ascii >= max(1, len(text) // 3)


def _image_paths(dataset: dict[str, Any], base_dir: Path) -> dict[str, str]:
    image_paths = dataset.get("image_paths") if isinstance(dataset.get("image_paths"), dict) else {}

    no_grid = image_paths.get("no_grid") or dataset.get("image_path")
    with_grid = (
        image_paths.get("grid_with_grid")
        or image_paths.get("with_grid")
        or dataset.get("encrypted_grid_path")
        or dataset.get("basic_grid_path")
    )
    basic_grid = image_paths.get("with_grid") or dataset.get("basic_grid_path") or with_grid

    paths = {
        "no_grid": no_grid,
        "with_grid": basic_grid,
        "grid_with_grid": with_grid,
    }
    return {
        key: str(_resolve_path(value, base_dir))
        for key, value in paths.items()
        if isinstance(value, str) and value
    }


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _prefer_encrypted_numeric_axis(dataset: dict[str, Any], chart_type: str) -> None:
    if chart_type == "h_bar":
        _copy_if_present(dataset, "x_ticks_encrypted", "x_ticks")
        _copy_if_present(dataset, "x_pixels_encrypted", "x_pixels")
    elif chart_type in {"v_bar", "line"}:
        _copy_if_present(dataset, "y_ticks_encrypted", "y_ticks")
        _copy_if_present(dataset, "y_pixels_encrypted", "y_pixels")
    elif chart_type in {"scatter", "bubble"}:
        _copy_if_present(dataset, "x_ticks_encrypted", "x_ticks")
        _copy_if_present(dataset, "x_pixels_encrypted", "x_pixels")
        _copy_if_present(dataset, "y_ticks_encrypted", "y_ticks")
        _copy_if_present(dataset, "y_pixels_encrypted", "y_pixels")


def _copy_if_present(dataset: dict[str, Any], source: str, target: str) -> None:
    value = dataset.get(source)
    if isinstance(value, list) and value:
        dataset[target] = value


def _prefer_data_categories(dataset: dict[str, Any], chart_type: str) -> None:
    labels = _category_labels(dataset)
    if not labels:
        return
    if chart_type in {"v_bar", "line"} and _same_length(dataset.get("x_pixels"), labels):
        dataset["x_ticks"] = labels
    elif chart_type == "h_bar" and _same_length(dataset.get("y_pixels"), labels):
        dataset["y_ticks"] = labels
    elif chart_type in {"v_bar", "line"}:
        inferred = _infer_category_pixels(dataset, "x", len(labels))
        if inferred:
            dataset["x_ticks"] = labels
            dataset["x_pixels"] = inferred
    elif chart_type == "h_bar":
        inferred = _infer_category_pixels(dataset, "y", len(labels))
        if inferred:
            dataset["y_ticks"] = labels
            dataset["y_pixels"] = inferred


def _category_labels(dataset: dict[str, Any]) -> list[str]:
    data_points = _data_points(dataset)
    labels: list[str] = []
    for points in data_points.values():
        if isinstance(points, dict):
            for label in points:
                text = str(label)
                if text not in labels:
                    labels.append(text)
    return labels


def _same_length(value: Any, labels: list[str]) -> bool:
    return isinstance(value, list) and len(value) == len(labels)


def _infer_category_pixels(dataset: dict[str, Any], axis: str, count: int) -> list[int]:
    if count <= 0:
        return []

    key = f"{axis}_pixels"
    existing = [int(value) for value in dataset.get(key, []) if isinstance(value, (int, float))]
    if len(existing) >= 2 and abs(max(existing) - min(existing)) >= max(12, count * 4):
        start, end = min(existing), max(existing)
    else:
        start, end = _fallback_plot_span(dataset, axis)

    if end <= start:
        return []
    if count == 1:
        return [round((start + end) / 2)]

    step = (end - start) / count
    centers = [round(start + step * (index + 0.5)) for index in range(count)]
    if axis == "y":
        centers = list(reversed(centers))
    return centers


def _fallback_plot_span(dataset: dict[str, Any], axis: str) -> tuple[int, int]:
    size = _image_size(dataset)
    if size is None:
        return 0, 0
    width, height = size
    if axis == "x":
        return round(width * 0.12), round(width * 0.95)
    return round(height * 0.12), round(height * 0.9)


def _image_size(dataset: dict[str, Any]) -> tuple[int, int] | None:
    for key in ("grid_with_grid", "with_grid", "no_grid"):
        try:
            path = resolve_image_path(dataset, key)
            if path.exists():
                with Image.open(path) as img:
                    return img.size
        except Exception:
            continue
    return None
