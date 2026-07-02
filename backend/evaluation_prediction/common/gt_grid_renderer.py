"""Render GT grid-with-grid images from dataset JSON metadata."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


MODULE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = BACKEND_ROOT.parent
DEFAULT_OUTPUT_ROOT = MODULE_ROOT / "results" / "gt_rendered_grids"
GRID_STYLE_VERSION = "grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1"
GRID_LINE_COLOR = (204, 204, 204, 255)
GRID_LINE_WIDTH = 1
GRID_DASH_LENGTH = 2
GRID_DASH_GAP = 2
LABEL_TEXT_COLOR = (0, 0, 0, 255)
LABEL_BACKGROUND_COLOR = (255, 255, 255, 238)
LABEL_PADDING_X = 3
LABEL_PADDING_Y = 2


def render_gt_grid_image(
    config_path: str | Path,
    *,
    dataset: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
    force: bool = False,
) -> Path | None:
    """Create a deterministic GT grid image from ticks/pixels in the config."""
    path = Path(config_path).resolve()
    data = dict(dataset or _read_json(path))
    source = _resolve_no_grid_path(data, path.parent)
    if source is None or not source.exists():
        return None

    output_dir = Path(output_root or DEFAULT_OUTPUT_ROOT) / _safe_name(path.parent.parent.name) / _safe_name(path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_name(path.stem)}_gt_grid_{GRID_STYLE_VERSION}.png"
    if output_path.exists() and not force:
        return output_path

    with Image.open(source).convert("RGBA") as base:
        image = base.copy()
    draw = ImageDraw.Draw(image, "RGBA")

    if _has_cartesian_grid(data):
        _draw_cartesian_grid(draw, data, image.size, image)
    elif _has_polar_grid(data):
        _draw_polar_grid(draw, data, image.size)
    else:
        return None

    image.convert("RGB").save(output_path)
    return output_path


def existing_or_rendered_grid_path(
    config_path: str | Path,
    *,
    dataset: dict[str, Any] | None = None,
    candidate: Any = None,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return an existing grid path or render one from GT metadata."""
    if candidate:
        resolved = _resolve_path(candidate, Path(base_dir) if base_dir else Path(config_path).resolve().parent)
        if resolved.exists():
            return resolved
    return render_gt_grid_image(config_path, dataset=dataset)


def _draw_cartesian_grid(
    draw: ImageDraw.ImageDraw,
    data: dict[str, Any],
    size: tuple[int, int],
    source_image: Image.Image,
) -> None:
    width, height = size
    x_ticks, x_pixels, x_label_ticks, x_label_pixels = _axis_ticks_pixels_and_labels(data, "x")
    y_ticks, y_pixels, y_label_ticks, y_label_pixels = _axis_ticks_pixels_and_labels(data, "y")
    if not x_pixels or not y_pixels:
        return

    for x in x_pixels:
        _draw_dashed_line(draw, (x, 0), (x, height - 1))
    for y in y_pixels:
        _draw_dashed_line(draw, (0, y), (width - 1, y))

    if x_label_ticks or y_label_ticks:
        font = _load_font(_label_font_size(size, x_label_ticks, y_label_ticks))
        x_label_y = _infer_x_label_center_y(source_image, x_pixels, y_pixels, font) if x_label_ticks else None
        y_label_anchor = _infer_y_label_anchor(source_image, x_pixels, y_pixels, font) if y_label_ticks else None
        _draw_x_tick_labels(draw, x_label_ticks, x_label_pixels, y_pixels, size, font, fixed_center_y=x_label_y)
        _draw_y_tick_labels(draw, y_label_ticks, y_label_pixels, x_pixels, size, font, fixed_anchor=y_label_anchor)


def _draw_polar_grid(draw: ImageDraw.ImageDraw, data: dict[str, Any], size: tuple[int, int]) -> None:
    width, height = size
    center = _center(data, size)
    radii, radius_label_ticks = _polar_radii_and_labels(data, size)
    if not radii:
        return
    radius = max(radii)
    angles, angle_labels = _polar_angles_and_labels(data)
    if not angles:
        angles = list(range(0, 360, 15))
        angle_labels = []

    cx, cy = center

    for r in radii:
        _draw_dashed_ellipse(draw, center, r)
    for angle in angles:
        theta = math.radians(angle - 90.0)
        edge_radius = _ray_to_image_edge_radius(center, theta, size)
        end = (cx + edge_radius * math.cos(theta), cy + edge_radius * math.sin(theta))
        _draw_dashed_line(draw, center, end)
    if radius_label_ticks or angle_labels:
        font = _load_font(max(8, min(13, round(min(size) * 0.018))))
        _draw_polar_labels(draw, center, radii, radius_label_ticks, angles, angle_labels, size, font)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}


def _resolve_no_grid_path(data: dict[str, Any], base_dir: Path) -> Path | None:
    image_paths = data.get("image_paths") if isinstance(data.get("image_paths"), dict) else {}
    for value in (
        image_paths.get("no_grid"),
        data.get("image_path"),
        image_paths.get("grid_with_grid"),
        image_paths.get("with_grid"),
    ):
        if isinstance(value, (str, Path)) and str(value):
            path = _resolve_path(value, base_dir)
            if path.exists():
                return path
    return None


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    dataset_root = base_dir.parent
    candidates = [
        (base_dir / path).resolve(),
        (dataset_root / path).resolve(),
        (dataset_root / "charts" / path.name).resolve(),
        (dataset_root / "chart" / path.name).resolve(),
        (BACKEND_ROOT / path).resolve(),
        (PROJECT_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _has_cartesian_grid(data: dict[str, Any]) -> bool:
    x_pixels = _numeric_list(data.get("x_pixels_encrypted")) or _numeric_list(data.get("x_pixels"))
    y_pixels = _numeric_list(data.get("y_pixels_encrypted")) or _numeric_list(data.get("y_pixels"))
    return bool(x_pixels and y_pixels)


def _has_polar_grid(data: dict[str, Any]) -> bool:
    return bool(data.get("center") or data.get("r_pixels") or data.get("radius") or data.get("theta_angles"))


def _numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        number = _number_or_none(item)
        if number is not None:
            result.append(number)
    return result


def _axis_ticks_pixels_and_labels(data: dict[str, Any], axis: str) -> tuple[list[Any], list[float], list[Any], list[float]]:
    encrypted_ticks = data.get(f"{axis}_ticks_encrypted")
    pixels = _numeric_list(data.get(f"{axis}_pixels_encrypted")) or _numeric_list(data.get(f"{axis}_pixels"))
    if isinstance(encrypted_ticks, list) and encrypted_ticks and pixels:
        ticks = list(encrypted_ticks[: len(pixels)])
        label_ticks, label_pixels = _encrypted_midpoint_tick_labels(ticks, pixels)
        if not label_ticks and len(encrypted_ticks) < len(pixels):
            midpoint_indices = _midpoint_indices_from_pixels(pixels)
            label_ticks = list(encrypted_ticks[: len(midpoint_indices)])
            label_pixels = [pixels[index] for index in midpoint_indices[: len(label_ticks)]]
        return ticks, pixels, label_ticks, label_pixels
    ticks = data.get(f"{axis}_ticks") if isinstance(data.get(f"{axis}_ticks"), list) else []
    ticks = list(ticks[: len(pixels)])
    label_ticks, label_pixels = _encrypted_midpoint_tick_labels(ticks, pixels)
    return ticks, pixels, label_ticks, label_pixels


def _encrypted_midpoint_tick_labels(ticks: list[Any], pixels: list[float]) -> tuple[list[Any], list[float]]:
    """Return labels for inserted encrypted ticks.

    The GT grid encryption inserts one value between every pair of original
    neighboring ticks. Therefore final tick sequences look like:
    original, encrypted-midpoint, original, encrypted-midpoint, original...
    Endpoints and even-index ticks are original; odd-index ticks are encrypted.
    """
    limit = min(len(ticks), len(pixels))
    if limit < 3:
        return [], []
    ticks = ticks[:limit]
    pixels = pixels[:limit]
    midpoint_indices = _midpoint_indices_from_ticks_and_pixels(ticks, pixels)
    return [ticks[index] for index in midpoint_indices], [pixels[index] for index in midpoint_indices]


def _midpoint_indices_from_ticks_and_pixels(ticks: list[Any], pixels: list[float]) -> list[int]:
    numeric_ticks = [_number_or_none(tick) for tick in ticks]
    numeric_axis = sum(value is not None for value in numeric_ticks) == len(ticks)
    if not numeric_axis:
        return []
    result: list[int] = []
    for index in range(1, len(ticks) - 1, 2):
        tick_value = numeric_ticks[index]
        prev_tick = numeric_ticks[index - 1]
        next_tick = numeric_ticks[index + 1]
        if tick_value is None or prev_tick is None or next_tick is None:
            continue
        tick_ok = _near_midpoint(tick_value, prev_tick, next_tick)
        pixel_ok = _near_midpoint(float(pixels[index]), float(pixels[index - 1]), float(pixels[index + 1]), tolerance=1.5)
        if tick_ok and pixel_ok:
            result.append(index)
    return result


def _midpoint_indices_from_pixels(pixels: list[float]) -> list[int]:
    return [
        index
        for index in range(1, len(pixels) - 1, 2)
        if _near_midpoint(float(pixels[index]), float(pixels[index - 1]), float(pixels[index + 1]), tolerance=1.5)
    ]


def _near_midpoint(value: float, left: float, right: float, *, tolerance: float | None = None) -> bool:
    midpoint = (float(left) + float(right)) / 2.0
    if tolerance is None:
        tolerance = max(1e-6, abs(float(right) - float(left)) * 0.02)
    return abs(float(value) - midpoint) <= tolerance


def _span(values: list[float], limit: int) -> tuple[float, float]:
    low, high = min(values), max(values)
    pad = 0
    return max(0, low - pad), min(limit - 1, high + pad)


def _label_font_size(
    size: tuple[int, int],
    x_ticks: list[Any],
    y_ticks: list[Any],
) -> int:
    width, height = size
    max_labels = max(len(x_ticks), len(y_ticks), 1)
    base = min(width, height)
    by_image = base * 0.017
    by_density = width / max(max_labels, 1) * 0.18
    return int(max(8, min(14, round(min(by_image, by_density)))))


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except Exception:
        return len(text) * 7, 10


def _format_label(value: Any, references: list[Any]) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        number = float(value)
    except Exception:
        return str(value)
    decimals = 0
    for item in references:
        text = str(item)
        if "." in text:
            decimals = max(decimals, len(text.split(".", 1)[1].rstrip("0")))
    if decimals > 0:
        text = f"{number:.{min(decimals, 6)}f}".rstrip("0").rstrip(".")
    else:
        text = f"{number:.6g}"
    return text


def _draw_label_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: tuple[float, float],
    size: tuple[int, int],
    font: ImageFont.ImageFont,
    *,
    anchor: str = "center",
) -> None:
    if not text:
        return
    width, height = size
    text_w, text_h = _text_bbox(draw, text, font)
    if anchor == "left":
        x = int(round(center[0]))
    elif anchor == "right":
        x = int(round(center[0] - text_w))
    else:
        x = int(round(center[0] - text_w / 2))
    y = int(round(center[1] - text_h / 2))
    x = max(1, min(width - text_w - LABEL_PADDING_X * 2 - 1, x))
    y = max(1, min(height - text_h - LABEL_PADDING_Y * 2 - 1, y))
    box = (
        x - LABEL_PADDING_X,
        y - LABEL_PADDING_Y,
        x + text_w + LABEL_PADDING_X,
        y + text_h + LABEL_PADDING_Y,
    )
    draw.rectangle(box, fill=LABEL_BACKGROUND_COLOR)
    draw.text((x, y), text, fill=LABEL_TEXT_COLOR, font=font)


def _draw_x_tick_labels(
    draw: ImageDraw.ImageDraw,
    ticks: list[Any],
    pixels: list[float],
    y_pixels: list[float],
    size: tuple[int, int],
    font: ImageFont.ImageFont,
    *,
    fixed_center_y: float | None = None,
) -> None:
    if not ticks or not pixels:
        return
    _width, height = size
    plot_bottom = max(y_pixels) if y_pixels else height - 1
    plot_top = min(y_pixels) if y_pixels else 0
    sample_text = max((_format_label(tick, ticks) for tick in ticks), key=len, default="")
    _tw, text_h = _text_bbox(draw, sample_text, font)
    gap = max(6, text_h // 2 + 3)
    if fixed_center_y is not None:
        label_y = fixed_center_y
    elif plot_bottom + text_h + gap + LABEL_PADDING_Y * 2 < height:
        label_y = plot_bottom + gap + text_h / 2
    elif plot_bottom > height * 0.55:
        label_y = plot_bottom - gap - text_h / 2
    else:
        label_y = min(height - text_h, plot_top + gap + text_h / 2)
    for tick, pixel in zip(ticks, pixels):
        _draw_label_box(draw, _format_label(tick, ticks), (pixel, label_y), size, font)


def _draw_y_tick_labels(
    draw: ImageDraw.ImageDraw,
    ticks: list[Any],
    pixels: list[float],
    x_pixels: list[float],
    size: tuple[int, int],
    font: ImageFont.ImageFont,
    *,
    fixed_anchor: tuple[float, str] | None = None,
) -> None:
    if not ticks or not pixels:
        return
    width, _height = size
    plot_left = min(x_pixels) if x_pixels else 0
    plot_right = max(x_pixels) if x_pixels else width - 1
    labels = [_format_label(tick, ticks) for tick in ticks]
    max_w = max((_text_bbox(draw, label, font)[0] for label in labels), default=0)
    gap = max(6, max_w // 8 + 5)
    left_space = plot_left
    right_space = width - plot_right
    if fixed_anchor is not None:
        label_x, anchor = fixed_anchor
    elif left_space >= max_w + gap + LABEL_PADDING_X * 2:
        label_x = plot_left - gap
        anchor = "right"
    elif right_space >= max_w + gap + LABEL_PADDING_X * 2:
        label_x = plot_right + gap
        anchor = "left"
    elif plot_left > width * 0.5:
        label_x = plot_left + gap
        anchor = "left"
    else:
        label_x = plot_left - gap
        anchor = "right"
    for label, pixel in zip(labels, pixels):
        _draw_label_box(draw, label, (label_x, pixel), size, font, anchor=anchor)


def _infer_x_label_center_y(
    image: Image.Image,
    x_pixels: list[float],
    y_pixels: list[float],
    font: ImageFont.ImageFont,
) -> float | None:
    if not y_pixels:
        return None
    width, height = image.size
    plot_top = int(max(0, min(y_pixels)))
    plot_bottom = int(min(height - 1, max(y_pixels)))
    _tw, text_h = _text_bbox(ImageDraw.Draw(image), "0", font)
    min_band = max(12, text_h + LABEL_PADDING_Y * 2)

    below = (plot_bottom + 2, min(height - 1, plot_bottom + max(min_band * 4, round(height * 0.18))))
    above = (max(0, plot_top - max(min_band * 4, round(height * 0.18))), plot_top - 2)
    for band in (below, above):
        center = _dominant_dark_row_center(image, band[0], band[1], x_pixels=x_pixels)
        if center is not None:
            return center
    return None


def _infer_y_label_anchor(
    image: Image.Image,
    x_pixels: list[float],
    y_pixels: list[float],
    font: ImageFont.ImageFont,
) -> tuple[float, str] | None:
    if not x_pixels:
        return None
    width, height = image.size
    plot_left = int(max(0, min(x_pixels)))
    plot_right = int(min(width - 1, max(x_pixels)))
    text_w, _th = _text_bbox(ImageDraw.Draw(image), "0000", font)
    min_band = max(18, text_w + LABEL_PADDING_X * 2)

    left_band = (max(0, plot_left - max(min_band * 3, round(width * 0.18))), plot_left - 2)
    right_band = (plot_right + 2, min(width - 1, plot_right + max(min_band * 3, round(width * 0.18))))
    left_center = _dominant_dark_col_center(image, left_band[0], left_band[1], y_pixels=y_pixels)
    right_center = _dominant_dark_col_center(image, right_band[0], right_band[1], y_pixels=y_pixels)

    if left_center is not None and right_center is not None:
        if abs(left_center - plot_left) <= abs(right_center - plot_right):
            return left_center + text_w / 2, "right"
        return right_center - text_w / 2, "left"
    if left_center is not None:
        return left_center + text_w / 2, "right"
    if right_center is not None:
        return right_center - text_w / 2, "left"
    return None


def _dominant_dark_row_center(
    image: Image.Image,
    y0: int,
    y1: int,
    *,
    x_pixels: list[float],
) -> float | None:
    if y1 <= y0:
        return None
    gray = image.convert("L")
    width, height = gray.size
    plot_left = int(max(0, min(x_pixels))) if x_pixels else 0
    plot_right = int(min(width - 1, max(x_pixels))) if x_pixels else width - 1
    x0 = max(0, plot_left - round(width * 0.08))
    x1 = min(width - 1, plot_right + round(width * 0.08))
    row_scores: list[tuple[int, int]] = []
    for y in range(max(0, y0), min(height - 1, y1) + 1):
        score = 0
        for x in range(x0, x1 + 1):
            if gray.getpixel((x, y)) < 128:
                score += 1
        row_scores.append((y, score))
    return _weighted_center_from_scores(row_scores, min_score=max(4, (x1 - x0) // 100))


def _dominant_dark_col_center(
    image: Image.Image,
    x0: int,
    x1: int,
    *,
    y_pixels: list[float],
) -> float | None:
    if x1 <= x0:
        return None
    gray = image.convert("L")
    width, height = gray.size
    plot_top = int(max(0, min(y_pixels))) if y_pixels else 0
    plot_bottom = int(min(height - 1, max(y_pixels))) if y_pixels else height - 1
    y0 = max(0, plot_top - round(height * 0.08))
    y1 = min(height - 1, plot_bottom + round(height * 0.08))
    col_scores: list[tuple[int, int]] = []
    for x in range(max(0, x0), min(width - 1, x1) + 1):
        score = 0
        for y in range(y0, y1 + 1):
            if gray.getpixel((x, y)) < 128:
                score += 1
        col_scores.append((x, score))
    return _weighted_center_from_scores(col_scores, min_score=max(4, (y1 - y0) // 100))


def _weighted_center_from_scores(scores: list[tuple[int, int]], *, min_score: int) -> float | None:
    if not scores:
        return None
    best = max(score for _pos, score in scores)
    if best < min_score:
        return None
    threshold = max(min_score, int(best * 0.55))
    selected = [(pos, score) for pos, score in scores if score >= threshold]
    if not selected:
        return None
    total = sum(score for _pos, score in selected)
    if total <= 0:
        return None
    return sum(pos * score for pos, score in selected) / total


def _center(data: dict[str, Any], size: tuple[int, int]) -> tuple[float, float]:
    width, height = size
    raw = data.get("center")
    if isinstance(raw, dict):
        return float(raw.get("x", width / 2)), float(raw.get("y", height / 2))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return float(raw[0]), float(raw[1])
    return width / 2, height / 2


def _polar_radii_and_labels(data: dict[str, Any], size: tuple[int, int]) -> tuple[list[float], list[Any]]:
    encrypted_ticks = data.get("r_ticks_encrypted")
    encrypted_radii = sorted(_numeric_list(data.get("r_pixels_encrypted")) or _numeric_list(data.get("r_pixels")))
    if isinstance(encrypted_ticks, list) and encrypted_ticks and encrypted_radii:
        ticks = list(encrypted_ticks[: len(encrypted_radii)])
        labels, label_radii = _encrypted_midpoint_tick_labels(ticks, encrypted_radii)
        return encrypted_radii, _labels_aligned_to_positions(encrypted_radii, label_radii, labels)

    raw = data.get("r_pixels") or data.get("radius")
    if isinstance(raw, list):
        values = _numeric_list(raw)
        if values:
            radii = sorted(values)
            ticks = data.get("r_ticks") if isinstance(data.get("r_ticks"), list) else []
            labels, label_radii = _encrypted_midpoint_tick_labels(list(ticks[: len(radii)]), radii)
            return radii, _labels_aligned_to_positions(radii, label_radii, labels)
    number = _number_or_none(raw)
    if number is not None and number > 0:
        return [number], []
    return [min(size) * 0.35], []


def _labels_aligned_to_positions(positions: list[float], label_positions: list[float], labels: list[Any]) -> list[Any]:
    aligned: list[Any] = [""] * len(positions)
    for label, label_position in zip(labels, label_positions):
        try:
            index = min(range(len(positions)), key=lambda item: abs(float(positions[item]) - float(label_position)))
        except ValueError:
            continue
        aligned[index] = label
    return aligned


def _polar_angles_and_labels(data: dict[str, Any]) -> tuple[list[float], list[str]]:
    encrypted_angles = _numeric_list(data.get("theta_angles_encrypted") or data.get("axes_angles_encrypted"))
    if not encrypted_angles:
        encrypted_angles = _numeric_list(data.get("theta_angles") or data.get("axes_angles"))
    encrypted_ticks = data.get("theta_ticks_encrypted") or data.get("axis_labels_encrypted") or data.get("axes_labels_encrypted")
    if encrypted_angles:
        labels = [str(item) for item in encrypted_ticks[: len(encrypted_angles)]] if isinstance(encrypted_ticks, list) else []
        if labels:
            return encrypted_angles, labels

    for key in ("theta_angles", "axes_angles"):
        values = _numeric_list(data.get(key))
        if values:
            return values, []
    axis_labels = data.get("axis_labels")
    if isinstance(axis_labels, dict):
        values = [_number_or_none(key) for key in axis_labels]
        return [value for value in values if value is not None], []
    theta_ticks = data.get("theta_ticks")
    if isinstance(theta_ticks, list) and theta_ticks:
        return [360.0 * index / len(theta_ticks) for index in range(len(theta_ticks))], []
    return [], []


def _ray_to_image_edge_radius(
    center: tuple[float, float],
    theta: float,
    size: tuple[int, int],
) -> float:
    width, height = size
    cx, cy = center
    dx, dy = math.cos(theta), math.sin(theta)
    candidates: list[float] = []
    if abs(dx) > 1e-9:
        candidates.extend([(0 - cx) / dx, (width - 1 - cx) / dx])
    if abs(dy) > 1e-9:
        candidates.extend([(0 - cy) / dy, (height - 1 - cy) / dy])
    positive = [value for value in candidates if value > 0]
    return min(positive) if positive else min(size) * 0.5


def _draw_polar_labels(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radii: list[float],
    radius_labels: list[Any],
    angles: list[float],
    angle_labels: list[str],
    size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    width, height = size
    cx, cy = center
    max_radius = max(radii) if radii else min(size) * 0.35
    for angle, label in zip(angles, angle_labels):
        theta = math.radians(angle - 90.0)
        edge_radius = min(_ray_to_image_edge_radius(center, theta, size), max_radius + min(size) * 0.08)
        x = cx + edge_radius * math.cos(theta)
        y = cy + edge_radius * math.sin(theta)
        _draw_label_box(draw, str(label), (x, y), size, font)

    label_x_limit = width - 4
    for radius, tick in zip(radii, radius_labels):
        if tick in ("", None):
            continue
        text = _format_label(tick, radius_labels)
        label_x = min(label_x_limit, cx + radius + 8)
        label_y = cy
        anchor = "left"
        if label_x > width - 40:
            label_x = max(4, cx + radius - 8)
            anchor = "right"
        _draw_label_box(draw, text, (label_x, label_y), size, font, anchor=anchor)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int] = GRID_LINE_COLOR,
    width: int = GRID_LINE_WIDTH,
    dash_length: int = GRID_DASH_LENGTH,
    dash_gap: int = GRID_DASH_GAP,
) -> None:
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return

    cycle = max(1.0, float(dash_length + dash_gap))
    dash = max(1.0, float(dash_length))
    ux, uy = dx / length, dy / length
    offset = 0.0
    while offset <= length:
        segment_end = min(offset + dash, length)
        sx, sy = x1 + ux * offset, y1 + uy * offset
        ex, ey = x1 + ux * segment_end, y1 + uy * segment_end
        draw.line(
            [(round(sx), round(sy)), (round(ex), round(ey))],
            fill=fill,
            width=width,
        )
        offset += cycle


def _draw_dashed_ellipse(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    radius: float,
    *,
    fill: tuple[int, int, int, int] = GRID_LINE_COLOR,
    width: int = GRID_LINE_WIDTH,
    dash_length: int = GRID_DASH_LENGTH,
    dash_gap: int = GRID_DASH_GAP,
) -> None:
    if radius <= 0:
        return
    cx, cy = float(center[0]), float(center[1])
    circumference = 2.0 * math.pi * float(radius)
    cycle = max(1.0, float(dash_length + dash_gap))
    dash = max(1.0, float(dash_length))
    offset = 0.0
    while offset < circumference:
        end_offset = min(offset + dash, circumference)
        start_angle = offset / circumference * 360.0
        end_angle = end_offset / circumference * 360.0
        draw.arc(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            start=start_angle,
            end=end_angle,
            fill=fill,
            width=width,
        )
        offset += cycle


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(str(value).strip().rstrip("%").rstrip("°"))
        return number if number == number else None
    except Exception:
        return None


def _safe_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "")).strip("_") or "grid"
