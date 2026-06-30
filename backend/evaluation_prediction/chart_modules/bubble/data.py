"""Data loading for bubble-like point charts."""

from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ...common.backend_dataset import load_backend_generated_datasets, resolve_image_path
from ...common.chart_io import filter_chart_configs, load_json_configs
from ...common.paths import ASSETS_ROOT, RESULTS_ROOT


@dataclass(frozen=True)
class PointChartConfig:
    chart_type: str
    result_dir_name: str
    mark_name: str = "circle"

    @property
    def asset_root(self) -> Path:
        return ASSETS_ROOT / self.chart_type

    @property
    def result_root(self) -> Path:
        return RESULTS_ROOT / self.chart_type


@dataclass(frozen=True)
class PointTarget:
    chart_id: str
    point_name: str
    visual_name: str
    category: str
    gt_x: float | None
    gt_y: float | None


def _belongs_to_chart_type(dataset: dict[str, Any], chart_type: str) -> bool:
    chart_id = str(dataset.get("chart_id", ""))
    return chart_id.startswith(f"{chart_type}_")


def load_datasets(
    config: PointChartConfig,
    chart_ids: Iterable[str] | None = None,
    config_paths: Iterable[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if config_paths:
        return load_backend_generated_datasets(config_paths, config.chart_type, chart_ids)
    configs = load_json_configs(config.asset_root / "chart_configs", recursive=True, exclude_emu=True)
    configs = [item for item in configs if _belongs_to_chart_type(item, config.chart_type)]
    return filter_chart_configs(configs, chart_ids)


def iter_targets(dataset: dict[str, Any]) -> list[PointTarget]:
    targets: list[PointTarget] = []
    color_names = list(dataset.get("series_color", {}))
    data_points = dataset.get("data_points") if isinstance(dataset.get("data_points"), dict) else {}
    for index, (point_name, coords) in enumerate(data_points.items()):
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        visual_name = str(point_name)
        if visual_name not in dataset.get("series_color", {}) and index < len(color_names):
            visual_name = str(color_names[index])
        targets.append(
            PointTarget(
                chart_id=dataset["chart_id"],
                point_name=str(point_name),
                visual_name=visual_name,
                category=_category_for_point(dataset, visual_name),
                gt_x=_float_or_none(coords[0]),
                gt_y=_float_or_none(coords[1]),
            )
        )
    if targets:
        return targets

    for name in _target_names_from_generated_json(dataset):
        targets.append(
            PointTarget(
                chart_id=dataset["chart_id"],
                point_name=name["name"],
                visual_name=name["name"],
                category=name.get("category", ""),
                gt_x=None,
                gt_y=None,
            )
        )
    return targets


def _target_names_from_generated_json(dataset: dict[str, Any]) -> list[dict[str, str]]:
    explicit = _explicit_point_labels(dataset)
    if explicit:
        return explicit
    ocr_targets = _ocr_point_labels(dataset)
    if ocr_targets:
        return ocr_targets
    return []


def _explicit_point_labels(dataset: dict[str, Any]) -> list[dict[str, str]]:
    value = dataset.get("point_labels") or dataset.get("point_items") or dataset.get("points")
    if not isinstance(value, list):
        return []
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            name = item.strip()
            category = ""
        elif isinstance(item, dict):
            name = str(item.get("name") or item.get("label") or item.get("id") or "").strip()
            category = str(item.get("category") or item.get("series_name") or item.get("group") or "").strip()
        else:
            continue
        key = _normalize_name_key(name)
        if name and key not in seen:
            seen.add(key)
            targets.append({"name": name, "category": category})
    return targets


def _ocr_point_labels(dataset: dict[str, Any]) -> list[dict[str, str]]:
    ocr_path = _ocr_axis_path(dataset)
    if ocr_path is None or not ocr_path.exists():
        return []
    try:
        with ocr_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    bounds = _plot_bounds(dataset)
    if bounds is None:
        return []
    left, top, right, bottom = bounds
    candidates: list[dict[str, Any]] = []
    image = _open_no_grid_image(dataset)
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _clean_point_label(item.get("text"))
        if not text or not _looks_like_point_label(text):
            continue
        if item.get("role") in {"x_axis", "y_axis"} or item.get("label_kind") == "axis_label":
            continue
        center = _center(item)
        if center is None:
            continue
        x, y = center
        if not (left <= x <= right and top <= y <= bottom):
            continue
        score = _float_or_none(item.get("score"))
        if score is not None and score < 0.65:
            continue
        candidates.append({"text": text, "item": item, "center": center, "box": _box_bounds(item)})

    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in _group_ocr_label_candidates(candidates):
        text = _clean_point_label(" ".join(str(item["text"]) for item in group))
        if not text or not _looks_like_point_label(text):
            continue
        text = _canonical_point_label(dataset, text)
        representative = _merge_candidate_items(group)
        key = _normalize_name_key(text)
        if key in seen:
            continue
        seen.add(key)
        category = _category_near_label(dataset, representative, image)
        targets.append({"name": text, "category": category})
    return targets


def _ocr_axis_path(dataset: dict[str, Any]) -> Path | None:
    reconstruction = dataset.get("enhanced_grid_reconstruction")
    outputs = reconstruction.get("outputs") if isinstance(reconstruction, dict) else None
    value = outputs.get("ocr_axis_path") if isinstance(outputs, dict) else None
    if value:
        return Path(str(value))
    return None


def _plot_bounds(dataset: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        x_pixels = [float(value) for value in dataset.get("x_pixels", [])]
        y_pixels = [float(value) for value in dataset.get("y_pixels", [])]
    except Exception:
        return None
    if not x_pixels or not y_pixels:
        return None
    return min(x_pixels), min(y_pixels), max(x_pixels), max(y_pixels)


def _clean_point_label(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.strip("·•:;,. ")


def _looks_like_point_label(text: str) -> bool:
    if len(text) < 2 or len(text) > 32:
        return False
    if "=" in text or ":" in text:
        return False
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", text):
        return False
    if not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return False
    # Long prose inside the plot area is usually an annotation, not a point id.
    if len(text.split()) > 4:
        return False
    return True


def _center(item: dict[str, Any]) -> tuple[float, float] | None:
    value = item.get("center")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _float_or_none(value[0])
        y = _float_or_none(value[1])
        if x is not None and y is not None:
            return x, y
    box = item.get("box")
    if isinstance(box, list) and box:
        xs: list[float] = []
        ys: list[float] = []
        for point in box:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x = _float_or_none(point[0])
                y = _float_or_none(point[1])
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
        if xs and ys:
            return sum(xs) / len(xs), sum(ys) / len(ys)
    return None


def _legend_names(dataset: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict):
        names.update(_normalize_name_key(name) for name in series_color)
    colors = dataset.get("colors")
    if isinstance(colors, list):
        for item in colors:
            if isinstance(item, dict) and item.get("name"):
                names.add(_normalize_name_key(item.get("name")))
    return {name for name in names if name}


def _canonical_point_label(dataset: dict[str, Any], text: str) -> str:
    normalized = _normalize_name_key(text)
    best_name = text
    best_score = 0.0
    for name in _series_or_color_names(dataset):
        candidate = _normalize_name_key(name)
        if not candidate:
            continue
        score = SequenceMatcher(None, normalized, candidate).ratio()
        if score > best_score:
            best_score = score
            best_name = name
    return best_name if best_score >= 0.62 else text


def _series_or_color_names(dataset: dict[str, Any]) -> list[str]:
    names: list[str] = []
    series_color = dataset.get("series_color")
    if isinstance(series_color, dict):
        names.extend(str(name).strip() for name in series_color if str(name).strip())
    colors = dataset.get("colors")
    if isinstance(colors, list):
        for item in colors:
            if isinstance(item, dict) and item.get("name"):
                text = str(item.get("name")).strip()
                if text:
                    names.append(text)
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = _normalize_name_key(name)
        if key and key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def _category_for_point(dataset: dict[str, Any], point_name: str) -> str:
    key = _normalize_name_key(point_name)
    for name in _legend_names(dataset):
        if key == name:
            return point_name
    return ""


def _category_near_label(dataset: dict[str, Any], item: dict[str, Any], image: Image.Image | None) -> str:
    if image is None:
        return ""
    legend_colors = _legend_colors(dataset)
    if not legend_colors:
        return ""
    box = _box_bounds(item)
    if box is None:
        center = _center(item)
        if center is None:
            return ""
        x, y = center
        box = (x - 12, y - 12, x + 12, y + 12)
    left, top, right, bottom = box
    sample_box = (
        max(0, int(left - 18)),
        max(0, int(top - 18)),
        min(image.width, int(right + 18)),
        min(image.height, int(bottom + 18)),
    )
    pixels = image.crop(sample_box).convert("RGB").getdata()
    best_name = ""
    best_score = float("inf")
    for pixel in pixels:
        if _is_neutral_pixel(pixel):
            continue
        for name, color in legend_colors.items():
            distance = _rgb_distance(pixel, color)
            if distance < best_score:
                best_score = distance
                best_name = name
    return best_name if best_score <= 90 else ""


def _box_bounds(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = item.get("box")
    if not isinstance(box, list) or not box:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x = _float_or_none(point[0])
            y = _float_or_none(point[1])
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _group_ocr_label_candidates(candidates: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: (item["center"][1], item["center"][0])):
        placed = False
        for group in groups:
            if any(_same_local_label(candidate, existing) for existing in group):
                group.append(candidate)
                placed = True
                break
        if not placed:
            groups.append([candidate])

    changed = True
    while changed:
        changed = False
        merged: list[list[dict[str, Any]]] = []
        for group in groups:
            target = None
            for existing in merged:
                if any(_same_local_label(left, right) for left in group for right in existing):
                    target = existing
                    break
            if target is None:
                merged.append(group)
            else:
                target.extend(group)
                changed = True
        groups = merged

    for group in groups:
        group.sort(key=lambda item: (item["center"][1], item["center"][0]))
    return groups


def _same_local_label(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_box = left.get("box")
    right_box = right.get("box")
    if left_box is None or right_box is None:
        return False
    l0, t0, l1, t1 = left_box
    r0, u0, r1, u1 = right_box
    horizontal_overlap = min(l1, r1) - max(l0, r0)
    vertical_overlap = min(t1, u1) - max(t0, u0)
    min_width = max(1.0, min(l1 - l0, r1 - r0))
    min_height = max(1.0, min(t1 - t0, u1 - u0))
    x_gap = max(0.0, max(l0, r0) - min(l1, r1))
    y_gap = max(0.0, max(t0, u0) - min(t1, u1))

    same_stacked_label = horizontal_overlap >= min_width * 0.35 and y_gap <= 14
    same_text_line = vertical_overlap >= min_height * 0.35 and x_gap <= 18
    return same_stacked_label or same_text_line


def _merge_candidate_items(group: list[dict[str, Any]]) -> dict[str, Any]:
    boxes = [item.get("box") for item in group if item.get("box") is not None]
    if not boxes:
        return group[0]["item"]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return {
        "text": " ".join(str(item["text"]) for item in group),
        "box": [[left, top], [right, top], [right, bottom], [left, bottom]],
        "center": [(left + right) / 2, (top + bottom) / 2],
    }


def _open_no_grid_image(dataset: dict[str, Any]) -> Image.Image | None:
    try:
        path = resolve_image_path(dataset, "no_grid")
        if path.exists():
            return Image.open(path).convert("RGB")
    except Exception:
        return None
    return None


def _has_nearby_point_mark(
    item: dict[str, Any],
    image: Image.Image,
    bounds: tuple[float, float, float, float],
) -> bool:
    box = _box_bounds(item)
    if box is None:
        center = _center(item)
        if center is None:
            return False
        x, y = center
        box = (x - 8, y - 8, x + 8, y + 8)
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = bounds
    margin = 38
    sample_left = max(int(plot_left), int(left - margin))
    sample_top = max(int(plot_top), int(top - margin))
    sample_right = min(int(plot_right), int(right + margin))
    sample_bottom = min(int(plot_bottom), int(bottom + margin))
    if sample_right <= sample_left or sample_bottom <= sample_top:
        return False

    label_left = int(left) - 2
    label_top = int(top) - 2
    label_right = int(right) + 2
    label_bottom = int(bottom) + 2
    colored_pixels = 0
    dark_pixels = 0
    for y in range(sample_top, sample_bottom):
        for x in range(sample_left, sample_right):
            if label_left <= x <= label_right and label_top <= y <= label_bottom:
                continue
            r, g, b = image.getpixel((x, y))[:3]
            if max(r, g, b) > 248:
                continue
            chroma = max(r, g, b) - min(r, g, b)
            if chroma >= 28 and max(r, g, b) >= 80:
                colored_pixels += 1
            elif max(r, g, b) < 80:
                dark_pixels += 1
            if colored_pixels >= 24 or dark_pixels >= 36:
                return True
    return False


def _legend_colors(dataset: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    source = dataset.get("series_color")
    if isinstance(source, dict):
        for name, value in source.items():
            rgb = _parse_color(value)
            if rgb is not None:
                colors[str(name)] = rgb
    if not _legend_colors_reliable(colors):
        return {}
    return colors


def _legend_colors_reliable(colors: dict[str, tuple[int, int, int]]) -> bool:
    values = list(colors.values())
    if len(values) <= 1:
        return bool(values)
    min_distance = min(
        _rgb_distance(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )
    return min_distance >= 50


def _parse_color(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, list) and value:
        value = value[0]
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    return None


def _is_neutral_pixel(pixel: tuple[int, int, int]) -> bool:
    r, g, b = pixel
    if max(pixel) > 245 or min(pixel) < 25:
        return True
    return max(abs(r - g), abs(r - b), abs(g - b)) < 18


def _rgb_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _normalize_name_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def image_path(config: PointChartConfig, dataset: dict[str, Any], image_type: str) -> Path:
    path = resolve_image_path(dataset, image_type)
    if path.exists() or Path(dataset["image_paths"][image_type]).is_absolute():
        return path
    return config.asset_root / dataset["image_paths"][image_type]
