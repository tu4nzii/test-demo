from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

import cv2
import numpy as np

from grid_geometry import grid_positions_and_bounds

def mask_panel(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)

def overlay_masks(image: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    h_pixels = horizontal > 0
    v_pixels = vertical > 0
    both = h_pixels & v_pixels

    h_color = np.array([0, 190, 255], dtype=np.uint8)
    v_color = np.array([255, 0, 190], dtype=np.uint8)
    both_color = np.array([0, 210, 80], dtype=np.uint8)

    overlay[h_pixels] = (0.35 * overlay[h_pixels] + 0.65 * h_color).astype(np.uint8)
    overlay[v_pixels] = (0.35 * overlay[v_pixels] + 0.65 * v_color).astype(np.uint8)
    overlay[both] = (0.25 * overlay[both] + 0.75 * both_color).astype(np.uint8)
    return overlay

def overlay_grid(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    pixels = grid > 0
    color = np.array([0, 160, 255], dtype=np.uint8)
    overlay[pixels] = (0.35 * overlay[pixels] + 0.65 * color).astype(np.uint8)
    return overlay

def binding_source_color(source: str) -> tuple[int, int, int]:
    colors = {
        "ocr+mllm": (0, 150, 40),
        "ocr": (220, 110, 0),
        "mllm": (165, 45, 185),
        "none": (140, 140, 140),
    }
    return colors.get(source, colors["none"])

def draw_label_box(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    anchor: str = "left",
    font_scale: float = 0.46,
    draw_border: bool = True,
    text_color: tuple[int, int, int] | None = None,
) -> None:
    if not text:
        return
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    if anchor == "center":
        x -= text_w // 2
    elif anchor == "right":
        x -= text_w
    x = max(2, min(w - text_w - 6, x))
    y = max(text_h + 6, min(h - baseline - 4, y))
    pad_x = 4
    pad_y = 3
    top_left = (x - pad_x, y - text_h - pad_y)
    bottom_right = (x + text_w + pad_x, y + baseline + pad_y)
    cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1, cv2.LINE_AA)
    if draw_border:
        cv2.rectangle(image, top_left, bottom_right, color, 1, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, font_scale, text_color or color, thickness, cv2.LINE_AA)

def ocr_box_points(ocr_tick: object) -> np.ndarray | None:
    if not isinstance(ocr_tick, dict):
        return None
    box = ocr_tick.get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        points = np.array([[float(point[0]), float(point[1])] for point in box], dtype=np.int32)
    except (TypeError, ValueError, IndexError):
        return None
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
        return None
    return points

def ocr_box_float_points(ocr_tick: object) -> np.ndarray | None:
    if not isinstance(ocr_tick, dict):
        return None
    box = ocr_tick.get("box")
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        points = np.array([[float(point[0]), float(point[1])] for point in box], dtype=np.float32)
    except (TypeError, ValueError, IndexError):
        return None
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
        return None
    return points

def ocr_center_point(ocr_tick: object) -> tuple[int, int] | None:
    if not isinstance(ocr_tick, dict):
        return None
    center = ocr_tick.get("center")
    if not isinstance(center, (list, tuple)) or len(center) < 2:
        x_value = ocr_tick.get("x")
        y_value = ocr_tick.get("y")
        if x_value is None or y_value is None:
            return None
        center = [x_value, y_value]
    try:
        return int(round(float(center[0]))), int(round(float(center[1])))
    except (TypeError, ValueError):
        return None

def sampled_text_color(image: np.ndarray, points: np.ndarray) -> tuple[int, int, int]:
    h, w = image.shape[:2]
    x0 = max(0, int(np.floor(float(points[:, 0].min()))) - 2)
    x1 = min(w - 1, int(np.ceil(float(points[:, 0].max()))) + 2)
    y0 = max(0, int(np.floor(float(points[:, 1].min()))) - 2)
    y1 = min(h - 1, int(np.ceil(float(points[:, 1].max()))) + 2)
    if x1 <= x0 or y1 <= y0:
        return (80, 80, 80)
    crop = image[y0 : y1 + 1, x0 : x1 + 1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = crop[gray < 210]
    if len(dark) == 0:
        return (80, 80, 80)
    color = np.median(dark.reshape(-1, 3), axis=0)
    return tuple(int(max(25, min(180, value))) for value in color)

def ocr_box_text_geometry(points: np.ndarray) -> tuple[tuple[float, float], float, float, float] | None:
    if points.shape[0] < 4:
        return None
    edges: list[tuple[float, np.ndarray]] = []
    for index in range(points.shape[0]):
        delta = points[(index + 1) % points.shape[0]] - points[index]
        length = float(np.linalg.norm(delta))
        if length > 0:
            edges.append((length, delta))
    if not edges:
        return None
    long_length, long_delta = max(edges, key=lambda item: item[0])
    short_lengths = [length for length, _ in edges if length < long_length * 0.75]
    short_length = float(np.median(short_lengths)) if short_lengths else max(8.0, long_length * 0.35)
    angle = float(np.degrees(np.arctan2(float(long_delta[1]), float(long_delta[0]))))
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    center = (float(points[:, 0].mean()), float(points[:, 1].mean()))
    return center, angle, max(8.0, long_length), max(6.0, short_length)

def draw_text_like_ocr_box(
    target: np.ndarray,
    source_image: np.ndarray,
    text: str,
    ocr_tick: object,
    *,
    fallback_color: tuple[int, int, int],
    draw_box: bool = True,
    text_color: tuple[int, int, int] | None = None,
    alpha_scale: float = 0.82,
    fill_background: bool = False,
) -> bool:
    if not text or not isinstance(ocr_tick, dict):
        return False
    points = ocr_box_float_points(ocr_tick)
    if points is None:
        return False
    geometry = ocr_box_text_geometry(points)
    if geometry is None:
        return False
    (cx, cy), angle, box_width, box_height = geometry
    h, w = target.shape[:2]
    if not (0 <= cx < w and 0 <= cy < h):
        return False

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    base_scale = 1.0
    (base_w, base_h), baseline = cv2.getTextSize(text, font, base_scale, thickness)
    if base_w <= 0 or base_h <= 0:
        return False
    scale = min(box_width * 0.92 / base_w, box_height * 0.78 / max(1, base_h))
    scale = max(0.24, min(0.72, float(scale)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    if fill_background:
        cv2.fillConvexPoly(target, points.astype(np.int32), (255, 255, 255), lineType=cv2.LINE_AA)

    text_layer = np.zeros_like(target)
    mask = np.zeros((h, w), dtype=np.uint8)
    origin = (int(round(cx - text_w / 2)), int(round(cy + text_h / 2)))
    sampled_color = text_color or (
        (80, 80, 80) if ocr_tick.get("mllm_pseudo_box") else sampled_text_color(source_image, points)
    )
    cv2.putText(text_layer, text, origin, font, scale, sampled_color, thickness, cv2.LINE_AA)
    cv2.putText(mask, text, origin, font, scale, 255, thickness, cv2.LINE_AA)
    # atan2 above measures the OCR box direction in image coordinates where y grows downward;
    # OpenCV's rotation matrix uses the opposite sign for the visual text rotation.
    matrix = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    rotated_text = cv2.warpAffine(text_layer, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rotated_mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    pixels = rotated_mask > 0
    if not np.any(pixels):
        return False
    alpha = (rotated_mask[pixels].astype(np.float32) / 255.0 * float(alpha_scale))[:, None]
    target[pixels] = (target[pixels].astype(np.float32) * (1.0 - alpha) + rotated_text[pixels].astype(np.float32) * alpha).astype(np.uint8)
    if draw_box:
        cv2.polylines(target, [points.astype(np.int32)], isClosed=True, color=fallback_color, thickness=1, lineType=cv2.LINE_AA)
    return True

def binding_geometry_color(binding: dict[str, object]) -> tuple[int, int, int]:
    ocr_tick = binding.get("ocr")
    if isinstance(ocr_tick, dict) and ocr_tick.get("mllm_pseudo_box"):
        return (0, 170, 255)
    if binding.get("source") == "mllm" and isinstance(ocr_tick, dict):
        return (0, 95, 255)
    return (255, 185, 60)

def binding_geometry_caption(binding: dict[str, object]) -> str:
    return ""

def draw_binding_ocr_geometry(
    image: np.ndarray,
    binding: dict[str, object],
    axis_key: str,
    grid_position: int,
) -> None:
    ocr_tick = binding.get("ocr")
    if not isinstance(ocr_tick, dict):
        return
    color = binding_geometry_color(binding)
    box = ocr_box_points(ocr_tick)
    center = ocr_center_point(ocr_tick)
    emphasized = bool(ocr_tick.get("mllm_pseudo_box")) or (
        binding.get("source") == "mllm" and str(ocr_tick.get("text", "") or "").strip() != str(binding.get("label", "") or "").strip()
    )
    if box is not None:
        cv2.polylines(image, [box], isClosed=True, color=color, thickness=2 if emphasized else 1, lineType=cv2.LINE_AA)
    if center is not None:
        cx, cy = center
        h, w = image.shape[:2]
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        cv2.circle(image, (cx, cy), 3, color, -1, cv2.LINE_AA)
        if axis_key == "x_axis":
            cv2.line(image, (cx, cy), (max(0, min(w - 1, grid_position)), cy), color, 1, cv2.LINE_AA)
        else:
            cv2.line(image, (cx, cy), (cx, max(0, min(h - 1, grid_position))), color, 1, cv2.LINE_AA)
        caption = binding_geometry_caption(binding)
        if caption:
            draw_label_box(image, caption, (cx + 5, cy - 6), color, anchor="left", font_scale=0.36)

def draw_grid_label_overlay(
    image: np.ndarray,
    grid_horizontal: np.ndarray,
    grid_vertical: np.ndarray,
    grid_label_bindings: dict[str, object],
) -> np.ndarray:
    overlay = overlay_grid(image, cv2.bitwise_or(grid_horizontal, grid_vertical))
    h, w = image.shape[:2]
    x_axis = grid_label_bindings.get("x_axis", {}) if isinstance(grid_label_bindings, dict) else {}
    y_axis = grid_label_bindings.get("y_axis", {}) if isinstance(grid_label_bindings, dict) else {}
    if not isinstance(x_axis, dict):
        x_axis = {}
    if not isinstance(y_axis, dict):
        y_axis = {}

    x_bounds = x_axis.get("bounds")
    y_bounds = y_axis.get("bounds")
    y0, y1 = (0, h - 1)
    x0, x1 = (0, w - 1)
    if isinstance(x_bounds, list) and len(x_bounds) >= 2:
        y0, y1 = int(round(float(x_bounds[0]))), int(round(float(x_bounds[1])))
    if isinstance(y_bounds, list) and len(y_bounds) >= 2:
        x0, x1 = int(round(float(y_bounds[0]))), int(round(float(y_bounds[1])))
    y0, y1 = max(0, min(h - 1, y0)), max(0, min(h - 1, y1))
    x0, x1 = max(0, min(w - 1, x0)), max(0, min(w - 1, x1))
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 < x0:
        x0, x1 = x1, x0

    x_bindings = x_axis.get("tick_bindings", [])
    if isinstance(x_bindings, list):
        for index, binding in enumerate(x_bindings):
            if not isinstance(binding, dict):
                continue
            if binding.get("display_suppressed"):
                continue
            label = str(binding.get("label", "") or "").strip()
            if not label:
                continue
            try:
                x = int(round(float(binding.get("position", 0.0))))
            except (TypeError, ValueError):
                continue
            source = str(binding.get("source", "none") or "none")
            color = binding_source_color(source)
            draw_binding_ocr_geometry(overlay, binding, "x_axis", x)
            drawn_like_original = draw_text_like_ocr_box(
                overlay,
                image,
                label,
                binding.get("ocr"),
                fallback_color=color,
            )
            if not drawn_like_original:
                tick_y = y1 + 18 + (index % 2) * 18
                tick_y = min(h - 6, max(18, tick_y))
                cv2.line(overlay, (x, min(h - 1, y1)), (x, tick_y - 11), color, 1, cv2.LINE_AA)
                draw_label_box(overlay, label, (x, tick_y), color, anchor="center")

    y_bindings = y_axis.get("tick_bindings", [])
    if isinstance(y_bindings, list):
        for binding in y_bindings:
            if not isinstance(binding, dict):
                continue
            if binding.get("display_suppressed"):
                continue
            label = str(binding.get("label", "") or "").strip()
            if not label:
                continue
            try:
                y = int(round(float(binding.get("position", 0.0))))
            except (TypeError, ValueError):
                continue
            source = str(binding.get("source", "none") or "none")
            color = binding_source_color(source)
            draw_binding_ocr_geometry(overlay, binding, "y_axis", y)
            drawn_like_original = draw_text_like_ocr_box(
                overlay,
                image,
                label,
                binding.get("ocr"),
                fallback_color=color,
            )
            if not drawn_like_original:
                label_x = x0 - 8
                cv2.line(overlay, (max(0, x0), y), (max(0, label_x - 4), y), color, 1, cv2.LINE_AA)
                draw_label_box(overlay, label, (label_x, y + 5), color, anchor="right")

    legend_items = [("ocr+mllm", "OCR+MLLM"), ("ocr", "OCR"), ("mllm", "MLLM")]
    geometry_legend = [((255, 185, 60), "OCR geom"), ((0, 170, 255), "pseudo"), ((0, 95, 255), "corrected")]
    legend_w = 132
    legend_h = 112
    legend_x = max(8, w - legend_w - 8)
    legend_y = 20
    cv2.rectangle(
        overlay,
        (legend_x - 5, legend_y - 15),
        (min(w - 2, legend_x + legend_w), min(h - 2, legend_y + legend_h - 15)),
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        overlay,
        (legend_x - 5, legend_y - 15),
        (min(w - 2, legend_x + legend_w), min(h - 2, legend_y + legend_h - 15)),
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    for source, text in legend_items:
        color = binding_source_color(source)
        cv2.rectangle(overlay, (legend_x, legend_y - 9), (legend_x + 12, legend_y + 3), color, -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            text,
            (legend_x + 18, legend_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        legend_y += 17
    for color, text in geometry_legend:
        cv2.rectangle(overlay, (legend_x, legend_y - 9), (legend_x + 12, legend_y + 3), color, 1, cv2.LINE_AA)
        cv2.circle(overlay, (legend_x + 6, legend_y - 3), 2, color, -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            text,
            (legend_x + 18, legend_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        legend_y += 17
    return overlay

def draw_json_summary_panel(
    image_shape: tuple[int, int],
    title: str,
    payload: dict[str, object],
) -> np.ndarray:
    h, w = image_shape
    panel = np.full((h, w, 3), 255, dtype=np.uint8)
    lines = [title]
    error = payload.get("error")
    if error:
        lines.append(str(error)[:80])
    for key in ("chart_type", "grid_intent", "horizontal_source", "vertical_source"):
        if key in payload:
            lines.append(f"{key}: {payload.get(key)}")
    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key, value in summary.items():
            lines.append(f"{key}: {value}")
    for axis_key in ("x_axis", "y_axis"):
        axis = payload.get(axis_key)
        if isinstance(axis, dict):
            axis_type = axis.get("type") or axis.get("kind") or "unknown"
            confidence = axis.get("confidence", "")
            count = axis.get("count", "")
            line_count = axis.get("grid_line_count", "")
            if line_count != "":
                lines.append(f"{axis_key}: lines={line_count} ocr={axis.get('ocr_tick_count', '')} mllm={axis.get('mllm_tick_count', '')}")
            else:
                lines.append(f"{axis_key}: {axis_type} c={confidence} n={count}")
            label = axis.get("axis_label")
            if isinstance(label, dict):
                label_text = str(label.get("text", "") or "")
                label_source = str(label.get("source", "") or "")
                label_confidence = label.get("confidence", "")
                if label_text:
                    lines.append(f"  label: {label_text[:34]}")
                    lines.append(f"  label src: {label_source} c={label_confidence}")
            tick_bindings = axis.get("tick_bindings")
            if isinstance(tick_bindings, list):
                for binding in tick_bindings[:8]:
                    if not isinstance(binding, dict):
                        continue
                    text = str(binding.get("label", "") or "")
                    source = str(binding.get("source", "") or "")
                    pos = binding.get("position", "")
                    if text:
                        lines.append(f"  {pos}: {text[:22]} ({source})")
    y = 24
    for line in lines[: max(1, (h - 10) // 18)]:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
        y += 18
    return panel

def draw_ocr_overlay(image: np.ndarray, ocr_items: list[dict[str, object]], error: str | None) -> np.ndarray:
    overlay = image.copy()
    colors = {
        "x_axis": (30, 170, 30),
        "y_axis": (210, 70, 210),
        "other": (150, 150, 150),
    }
    for item in ocr_items:
        box = np.array(item["box"], dtype=np.int32)
        role = str(item.get("role", "other"))
        raw_role = str(item.get("raw_role", role))
        color = colors.get(role, colors["other"])
        cv2.polylines(overlay, [box], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        x = int(box[:, 0].min())
        y = max(12, int(box[:, 1].min()) - 4)
        role_text = f"{raw_role}->{role}" if raw_role != role else role
        label = f"{role_text}:{item['text']}"
        cv2.putText(overlay, label[:36], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    if error:
        cv2.putText(
            overlay,
            error[:90],
            (10, max(24, image.shape[0] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 210),
            1,
            cv2.LINE_AA,
        )
    return overlay

def draw_items_overlay(
    image: np.ndarray,
    items: list[dict[str, object]],
    title: str,
    *,
    show_source: bool = True,
) -> np.ndarray:
    overlay = image.copy()
    colors = {
        "x_axis": (40, 170, 40),
        "y_axis": (210, 70, 210),
        "other": (150, 150, 150),
    }
    kind_colors = {
        "axis_label": (0, 130, 255),
        "other": (150, 150, 150),
    }
    for item in items:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        points = np.array(box, dtype=np.int32)
        role = str(item.get("role", "other"))
        label_kind = str(item.get("label_kind", "") or "")
        color = kind_colors.get(label_kind, colors.get(role, colors["other"]))
        if item.get("mllm_pseudo_box"):
            color = (20, 140, 230)
        thickness = 2 if label_kind == "axis_label" or item.get("mllm_pseudo_box") else 1
        cv2.polylines(overlay, [points], True, color, thickness, cv2.LINE_AA)
        if item.get("bbox_refined"):
            marker_x = int(points[:, 0].min())
            marker_y = int(points[:, 1].min())
            cv2.circle(overlay, (marker_x, marker_y), 3, color, -1, cv2.LINE_AA)
        x = int(points[:, 0].min())
        y = max(14, int(points[:, 1].min()) - 4)
        prefix = ""
        if item.get("split_from_merged"):
            prefix += "split:"
        if item.get("bbox_refined"):
            prefix += "refined:"
        if item.get("mllm_pseudo_box"):
            prefix += "pseudo:"
        if label_kind == "axis_label":
            prefix += "axis-title:"
        elif label_kind == "tick_label":
            prefix += "tick:"
        source = f"/{item.get('role_source', '')}" if show_source and item.get("role_source") else ""
        label = f"{prefix}{role}{source}:{item.get('text', '')}"
        cv2.putText(overlay, label[:60], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(overlay, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    return overlay

def make_ocr_summary_panel(
    image_shape: tuple[int, int],
    ocr_items: list[dict[str, object]],
    error: str | None,
) -> np.ndarray:
    h, w = image_shape
    panel = np.full((h, w, 3), 255, dtype=np.uint8)
    lines = ["OCR axis summary"]
    if error:
        lines.append(error)
    for role in ("x_axis", "y_axis", "other"):
        values = [item for item in ocr_items if item.get("role") == role]
        lines.append(f"{role}: {len(values)}")
        for item in values[:12]:
            text = str(item["text"])
            score = float(item.get("score", 0.0))
            cx, cy = item.get("center", [0.0, 0.0])
            lines.append(f"  {text[:24]}  {score:.2f} @({cx:.0f},{cy:.0f})")
    corrections = [item for item in ocr_items if item.get("raw_role") and item.get("raw_role") != item.get("role")]
    if corrections:
        lines.append(f"mllm role fixes: {len(corrections)}")
        for item in corrections[:8]:
            lines.append(f"  {item.get('raw_role')}->{item.get('role')}: {str(item.get('text', ''))[:22]}")

    y = 24
    for line in lines[: max(1, (h - 10) // 18)]:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)
        y += 18
    return panel

def make_ocr_label_lab_preview(
    image: np.ndarray,
    raw_items: list[dict[str, object]],
    split_items: list[dict[str, object]],
    bbox_refined_items: list[dict[str, object]],
    role_refined_items: list[dict[str, object]],
    pseudo_items: list[dict[str, object]],
    canonical_items: list[dict[str, object]],
    mllm_result: dict[str, object],
    metrics: dict[str, object],
    panel_width: int,
) -> np.ndarray:
    panels = [
        fit_panel(image, "original", panel_width),
        fit_panel(draw_items_overlay(image, raw_items, "raw PaddleOCR", show_source=False), "raw OCR boxes", panel_width),
        fit_panel(draw_items_overlay(image, split_items, "MLLM-guided split", show_source=False), "MLLM split boxes", panel_width),
        fit_panel(draw_items_overlay(image, bbox_refined_items, "projection-refined split boxes", show_source=False), "refined text boxes", panel_width),
        fit_panel(draw_items_overlay(image, role_refined_items, "MLLM role refined", show_source=True), "role refined labels", panel_width),
        fit_panel(draw_items_overlay(image, pseudo_items, "MLLM missing-label completion", show_source=True), "pseudo label boxes", panel_width),
        fit_panel(draw_items_overlay(image, canonical_items, "MLLM canonical text + OCR geometry", show_source=True), "canonical labels", panel_width),
        fit_panel(draw_json_summary_panel(image.shape[:2], "MLLM axis labels", mllm_result), "MLLM labels", panel_width),
        fit_panel(draw_json_summary_panel(image.shape[:2], "OCR label lab metrics", metrics), "metrics", panel_width),
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    cols = 3
    blank = np.full((max_h, panel_width, 3), 255, dtype=np.uint8)
    rows = []
    for index in range(0, len(padded), cols):
        row = padded[index : index + cols]
        while len(row) < cols:
            row.append(blank.copy())
        rows.append(np.hstack(row))
    return np.vstack(rows)

def make_grid_layer_summary_panel(
    image_shape: tuple[int, int],
    metadata: dict[str, object],
) -> np.ndarray:
    h, w = image_shape
    priority = metadata.get("priority_arbitration", {})
    if not isinstance(priority, dict):
        priority = {}
    panel = np.full((h, w, 3), 255, dtype=np.uint8)
    lines = [
        "Grid layer summary",
        f"horizontal: {metadata.get('horizontal_source', 'none')}",
        f"vertical: {metadata.get('vertical_source', 'none')}",
        f"pre-priority h: {metadata.get('pre_priority_horizontal_source', metadata.get('horizontal_source', 'none'))}",
        f"pre-priority v: {metadata.get('pre_priority_vertical_source', metadata.get('vertical_source', 'none'))}",
        f"direct h lines: {metadata.get('direct_horizontal_count', 0)}",
        f"direct v lines: {metadata.get('direct_vertical_count', 0)}",
        f"tick h lines: {metadata.get('tick_horizontal_count', 0)}",
        f"tick v lines: {metadata.get('tick_vertical_count', 0)}",
        f"guide h lines: {metadata.get('semantic_guide', {}).get('horizontal_count', 0) if isinstance(metadata.get('semantic_guide'), dict) else 0}",
        f"guide v lines: {metadata.get('semantic_guide', {}).get('vertical_count', 0) if isinstance(metadata.get('semantic_guide'), dict) else 0}",
        f"final h lines: {metadata.get('final_horizontal_count', 0)}",
        f"final v lines: {metadata.get('final_vertical_count', 0)}",
        "",
        "Priority arbitration:",
        f"x->vertical: {priority.get('x_axis_vertical_grid_choice', 'not_run')} ({priority.get('x_axis_reason', '')})",
        f"y->horizontal: {priority.get('y_axis_horizontal_grid_choice', 'not_run')} ({priority.get('y_axis_reason', '')})",
        f"mllm judge used: {priority.get('mllm_used', False)}",
        "",
        "Priority candidates:",
        "1 combined mask, 2 tick supplement, 3 semantic guide",
    ]
    y = 24
    for line in lines[: max(1, (h - 10) // 18)]:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        y += 18
    return panel

def fit_panel(image: np.ndarray, label: str, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = width / w
    resized = cv2.resize(image, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    label_h = 30
    panel = np.full((resized.shape[0] + label_h, width, 3), 255, dtype=np.uint8)
    panel[label_h:, :, :] = resized
    cv2.putText(
        panel,
        label,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (35, 35, 35),
        1,
        cv2.LINE_AA,
    )
    return panel

def make_preview(
    image: np.ndarray,
    candidate: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    direct_grid: np.ndarray,
    tick_grid: np.ndarray,
    semantic_guide_grid: np.ndarray,
    grid: np.ndarray,
    grid_horizontal: np.ndarray,
    grid_vertical: np.ndarray,
    grid_meta: dict[str, object],
    ocr_items: list[dict[str, object]],
    ocr_error: str | None,
    ocr_axis_evidence: dict[str, object],
    mllm_result: dict[str, object],
    fused_axis_evidence: dict[str, object],
    grid_label_bindings: dict[str, object],
    panel_width: int,
) -> np.ndarray:
    combined = cv2.bitwise_or(horizontal, vertical)
    overlay = overlay_masks(image, horizontal, vertical)
    grid_preview = overlay_grid(image, grid)
    label_overlay = draw_grid_label_overlay(image, grid_horizontal, grid_vertical, grid_label_bindings)
    layer_summary = make_grid_layer_summary_panel(image.shape[:2], grid_meta)
    ocr_overlay = draw_ocr_overlay(image, ocr_items, ocr_error)
    ocr_summary = make_ocr_summary_panel(image.shape[:2], ocr_items, ocr_error)
    ocr_semantic = draw_json_summary_panel(image.shape[:2], "OCR semantic evidence", ocr_axis_evidence)
    mllm_semantic = draw_json_summary_panel(image.shape[:2], "MLLM semantic evidence", mllm_result)
    fused_semantic = draw_json_summary_panel(image.shape[:2], "Fused axis evidence", fused_axis_evidence)
    binding_semantic = draw_json_summary_panel(image.shape[:2], "Grid-label bindings", grid_label_bindings)
    panels = [
        fit_panel(image, "original", panel_width),
        fit_panel(mask_panel(candidate), "neutral candidate", panel_width),
        fit_panel(mask_panel(horizontal), "horizontal filter", panel_width),
        fit_panel(mask_panel(vertical), "vertical filter", panel_width),
        fit_panel(mask_panel(combined), "combined mask", panel_width),
        fit_panel(overlay, "overlay", panel_width),
        fit_panel(mask_panel(direct_grid), "direct grid (combined)", panel_width),
        fit_panel(mask_panel(tick_grid), "tick supplement", panel_width),
        fit_panel(mask_panel(semantic_guide_grid), "semantic guide (label midline)", panel_width),
        fit_panel(mask_panel(grid), "hierarchical grid", panel_width),
        fit_panel(grid_preview, "hierarchical overlay", panel_width),
        fit_panel(label_overlay, "label binding overlay", panel_width),
        fit_panel(layer_summary, "grid layer summary", panel_width),
        fit_panel(ocr_overlay, "ocr overlay", panel_width),
        fit_panel(ocr_summary, "ocr axis summary", panel_width),
        fit_panel(ocr_semantic, "ocr semantic", panel_width),
        fit_panel(mllm_semantic, "mllm semantic", panel_width),
        fit_panel(fused_semantic, "fused axis semantic", panel_width),
        fit_panel(binding_semantic, "grid label bindings", panel_width),
    ]

    max_h = max(panel.shape[0] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.full((max_h - panel.shape[0], panel.shape[1], 3), 255, dtype=np.uint8)
            panel = np.vstack([panel, pad])
        padded.append(panel)
    columns = 4
    blank = np.full((max_h, panel_width, 3), 255, dtype=np.uint8)
    rows = []
    for index in range(0, len(padded), columns):
        row_panels = padded[index : index + columns]
        while len(row_panels) < columns:
            row_panels.append(blank.copy())
        rows.append(np.hstack(row_panels))
    return np.vstack(rows)
