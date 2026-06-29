import base64
import json
import os
import re

import cv2
import numpy as np

from gemini_calls import FAILURE_TEXT, chat_with_gemini_sync
from model_api_config import get_model_name
from type_detection.chart_registry import (
    SUPPORTED_CHART_TYPES,
    format_chart_options,
    format_supported_types,
)


class ChartTypeDetector:
    """Detect chart type and optional missing-axis repair hints with an MLLM."""

    TYPE_ALIASES = {
        "scatterplot": "scatter",
        "scatter_plot": "scatter",
        "scatter plot": "scatter",
        "bubblechart": "bubble",
        "bubble_chart": "bubble",
        "bubble chart": "bubble",
        "barchart": "v_bar",
        "bar_chart": "v_bar",
        "bar chart": "v_bar",
        "vertical_bar": "v_bar",
        "vertical bar": "v_bar",
        "v_stacked_bar": "v_stacked_bar",
        "vertical_stacked_bar": "v_stacked_bar",
        "vertical stacked bar": "v_stacked_bar",
        "stacked_bar": "v_stacked_bar",
        "stacked bar": "v_stacked_bar",
        "horizontal_bar": "h_bar",
        "horizontal bar": "h_bar",
        "h_stacked_bar": "h_stacked_bar",
        "horizontal_stacked_bar": "h_stacked_bar",
        "horizontal stacked bar": "h_stacked_bar",
        "linechart": "line",
        "line_chart": "line",
        "line chart": "line",
        "piechart": "pie",
        "pie_chart": "pie",
        "pie chart": "pie",
        "donutchart": "donut",
        "donut_chart": "donut",
        "donut chart": "donut",
        "radarchart": "radar",
        "radar_chart": "radar",
        "radar chart": "radar",
        "rosechart": "rose",
        "rose_chart": "rose",
        "rose chart": "rose",
    }

    def __init__(self):
        self.model_name = get_model_name()
        self.supported_types = SUPPORTED_CHART_TYPES

    def normalize_chart_type_strict(self, value):
        raw = str(value or "").strip().lower()
        normalized = raw.replace("-", "_")
        normalized = self.TYPE_ALIASES.get(normalized, self.TYPE_ALIASES.get(raw, normalized))
        if normalized in self.supported_types:
            return normalized
        raise ValueError(f"Unsupported or missing chart type from model: {value!r}")

    def extract_json_response(self, content: str):
        text = content or ""
        decoder = json.JSONDecoder()
        candidates = []
        for fence in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
            candidates.append(fence.strip())
        candidates.append(text.strip())

        last_error = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except Exception as error:
                last_error = error

            for match in re.finditer(r"\{", candidate):
                try:
                    parsed, _ = decoder.raw_decode(candidate[match.start():])
                    if isinstance(parsed, dict):
                        return parsed
                except Exception as error:
                    last_error = error
                    continue

        if last_error:
            print(f"[Error] JSON parse failed: {last_error}")
        return None

    def normalize_detection_result(self, result):
        if not isinstance(result, dict):
            return result
        normalized = dict(result)
        if "type" not in normalized:
            for key in ("chart_type", "chartType", "type_name", "detected_type"):
                if key in normalized:
                    normalized["type"] = normalized[key]
                    break
        if "confidence" not in normalized:
            for key in ("score", "confidence_score", "probability"):
                if key in normalized:
                    normalized["confidence"] = normalized[key]
                    break
        return normalized

    def image_to_base64(self, image_path):
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image file: {image_path}")

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # cv2.imencode expects BGR input. Do not convert to RGB here, otherwise
        # the image sent to the MLLM has red/blue channels swapped.
        success, encoded_image = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Image encoding failed")

        image_data = np.ascontiguousarray(encoded_image)
        return base64.b64encode(image_data).decode("utf-8")

    def normalize_axis_repair(self, result):
        """Normalize optional MLLM axis-missing hints.

        The repair path is deliberately opt-in. If the model omits this object
        or is unsure, all flags remain false so normal charts keep the existing
        encryption behavior.
        """
        repair = result.get("axis_repair") if isinstance(result, dict) else None
        if not isinstance(repair, dict):
            repair = {}

        missing_axes = result.get("missing_axes") if isinstance(result, dict) else None
        if isinstance(missing_axes, dict):
            repair = {**missing_axes, **repair}

        def as_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "y", "missing"}
            return False

        def as_choice(value, choices, default="unknown"):
            text = str(value or "").strip().lower()
            return text if text in choices else default

        def normalize_tick_list(values, limit=40):
            if not isinstance(values, list):
                return []
            normalized = []
            for value in values[:limit]:
                if isinstance(value, (str, int, float)):
                    text = str(value).strip()
                    if text:
                        normalized.append(text)
            return normalized

        def numeric_tick_values(values):
            numeric = []
            for value in values:
                text = str(value).replace(",", "").strip()
                match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
                if not match:
                    return None
                try:
                    numeric.append(float(match.group(0)))
                except ValueError:
                    return None
            return numeric if len(numeric) == len(values) else None

        def encrypted_tick_prior(values, axis_type):
            if axis_type not in {"numeric", "unknown"}:
                return list(values)
            numeric = numeric_tick_values(values)
            if not numeric or len(numeric) < 2:
                return list(values)
            encrypted = []
            for index, value in enumerate(numeric):
                encrypted.append(value)
                if index < len(numeric) - 1:
                    mid_value = (value + numeric[index + 1]) / 2
                    encrypted.append(round(float(f"{mid_value:.12f}"), 10))
            return encrypted

        try:
            confidence = float(repair.get("confidence", result.get("axis_repair_confidence", 0)))
        except (TypeError, ValueError):
            confidence = 0.0

        radar_grid = result.get("radar_grid") if isinstance(result, dict) else None
        if not isinstance(radar_grid, dict):
            radar_grid = repair.get("radar_grid") if isinstance(repair, dict) else None
        if not isinstance(radar_grid, dict):
            radar_grid = {}
        radar_grid_shape = str(
            radar_grid.get("shape", repair.get("radar_grid_shape", "unknown"))
        ).strip().lower()
        if radar_grid_shape not in {"polygon", "circular", "unknown"}:
            radar_grid_shape = "unknown"
        try:
            radar_grid_confidence = float(
                radar_grid.get("confidence", repair.get("radar_grid_confidence", 0))
            )
        except (TypeError, ValueError):
            radar_grid_confidence = 0.0

        axis_tick_labels = result.get("axis_tick_labels") if isinstance(result, dict) else None
        if not isinstance(axis_tick_labels, dict):
            axis_tick_labels = repair.get("axis_tick_labels") if isinstance(repair, dict) else None
        if not isinstance(axis_tick_labels, dict):
            axis_tick_labels = {}
        x_axis_tick_type = as_choice(
            axis_tick_labels.get("x_axis_type", repair.get("x_axis_role", result.get("x_axis_role"))),
            {"numeric", "category", "date", "unknown"},
        )
        y_axis_tick_type = as_choice(
            axis_tick_labels.get("y_axis_type", repair.get("y_axis_role", result.get("y_axis_role"))),
            {"numeric", "category", "date", "unknown"},
        )
        x_upload_ticks = normalize_tick_list(axis_tick_labels.get("x_ticks"))
        y_upload_ticks = normalize_tick_list(axis_tick_labels.get("y_ticks"))
        series_items = self.normalize_series_items(
            result.get("series_items") or repair.get("series_items") or result
        )

        return {
            "x_axis_missing": as_bool(repair.get("x_axis_missing", repair.get("x", False))),
            "y_axis_missing": as_bool(repair.get("y_axis_missing", repair.get("y", False))),
            "x_ticks_missing": as_bool(repair.get("x_ticks_missing", False)),
            "y_ticks_missing": as_bool(repair.get("y_ticks_missing", False)),
            "x_axis_role": as_choice(
                repair.get("x_axis_role", result.get("x_axis_role")),
                {"numeric", "category", "date", "unknown"},
            ),
            "y_axis_role": as_choice(
                repair.get("y_axis_role", result.get("y_axis_role")),
                {"numeric", "category", "date", "unknown"},
            ),
            "x_axis_position": as_choice(
                repair.get("x_axis_position", result.get("x_axis_position")),
                {"bottom", "top", "middle", "none", "unknown"},
            ),
            "y_axis_position": as_choice(
                repair.get("y_axis_position", result.get("y_axis_position")),
                {"left", "right", "middle", "both", "none", "unknown"},
            ),
            "plot_area_style": as_choice(
                repair.get("plot_area_style", result.get("plot_area_style")),
                {"explicit_axes", "weak_axes", "grid_only", "no_axes", "unknown"},
            ),
            "has_background_grid": as_bool(
                repair.get("has_background_grid", result.get("has_background_grid", False))
            ),
            "x_tick_recovery_from_grid": as_bool(
                repair.get("x_tick_recovery_from_grid", result.get("x_tick_recovery_from_grid", False))
            ),
            "y_tick_recovery_from_grid": as_bool(
                repair.get("y_tick_recovery_from_grid", result.get("y_tick_recovery_from_grid", False))
            ),
            "bar_layout": as_choice(
                repair.get("bar_layout", result.get("bar_layout")),
                {"single", "grouped", "stacked", "dense", "unknown"},
            ),
            "bar_orientation": as_choice(
                repair.get("bar_orientation", result.get("bar_orientation")),
                {"vertical", "horizontal", "unknown"},
            ),
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(repair.get("reason", "") or ""),
            "radar_grid_shape": radar_grid_shape,
            "radar_grid_confidence": max(0.0, min(1.0, radar_grid_confidence)),
            "axis_tick_labels": {
                "x_ticks": x_upload_ticks,
                "y_ticks": y_upload_ticks,
                "x_ticks_encrypted": encrypted_tick_prior(x_upload_ticks, x_axis_tick_type),
                "y_ticks_encrypted": encrypted_tick_prior(y_upload_ticks, y_axis_tick_type),
                "x_axis_type": x_axis_tick_type,
                "y_axis_type": y_axis_tick_type,
                "source": "upload_detection",
            },
            "series_items": series_items,
        }

    def apply_bar_layout_type_hint(self, chart_type, axis_repair):
        if not isinstance(axis_repair, dict) or axis_repair.get("bar_layout") != "stacked":
            return chart_type
        orientation = axis_repair.get("bar_orientation")
        if chart_type == "h_bar" or orientation == "horizontal":
            return "h_stacked_bar"
        if chart_type == "v_bar" or orientation == "vertical":
            return "v_stacked_bar"
        return chart_type

    def normalize_series_items(self, value):
        """Normalize visible legend/point color hints returned by upload MLLM."""
        if not isinstance(value, dict):
            return {"items": [], "source": "upload_detection", "kind": "unknown"}

        raw_items = value.get("items")
        if not isinstance(raw_items, list):
            raw_items = value.get("legend_items")
        if not isinstance(raw_items, list):
            raw_items = value.get("point_items")
        if not isinstance(raw_items, list):
            raw_items = value.get("colors")
        if not isinstance(raw_items, list) and isinstance(value.get("series_color"), dict):
            raw_items = [
                {"name": name, "color": color}
                for name, color in value["series_color"].items()
            ]
        if not isinstance(raw_items, list):
            raw_items = []

        items = []
        seen = set()
        for raw_item in raw_items[:80]:
            if not isinstance(raw_item, dict):
                continue
            name = str(raw_item.get("name") or raw_item.get("label") or "").strip()
            if not name:
                continue
            color = raw_item.get("color")
            if isinstance(color, str):
                color = color.strip()
                if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                    color = None
            else:
                color = None
            key = (name.casefold(), color or "")
            if key in seen:
                continue
            seen.add(key)
            items.append({"name": name, "color": color})

        return {
            "items": items,
            "source": "upload_detection",
            "kind": str(value.get("kind") or "unknown").strip().lower() or "unknown",
        }

    def _component_hex_color(self, image, labels, component_id):
        pixels = image[labels == component_id]
        if pixels.size == 0:
            return None
        bgr = np.median(pixels, axis=0).astype(int)
        b, g, r = [max(0, min(255, int(value))) for value in bgr[:3]]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _legend_swatch_candidates(self, image):
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 80)).astype("uint8") * 255
        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for component_id in range(1, component_count):
            x, y, w, h, area = stats[component_id]
            if area < 8 or area > 900:
                continue
            if w < 2 or h < 2 or w > 90 or h > 50:
                continue
            cx, cy = centroids[component_id]
            color = self._component_hex_color(image, labels, component_id)
            if not color:
                continue
            candidates.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "area": float(area),
                    "cx": float(cx),
                    "cy": float(cy),
                    "color": color,
                    "image_width": float(width),
                    "image_height": float(height),
                }
            )
        return candidates

    def _aligned_legend_group(self, candidates, expected_count, orientation):
        if len(candidates) < expected_count:
            return None
        align_key = "cx" if orientation == "vertical" else "cy"
        order_key = "cy" if orientation == "vertical" else "cx"
        best_group = None
        best_score = None
        for anchor in candidates:
            tolerance = 28.0 if orientation == "vertical" else 18.0
            group = [
                item
                for item in candidates
                if abs(float(item[align_key]) - float(anchor[align_key])) <= tolerance
            ]
            if len(group) < expected_count:
                continue
            group = sorted(group, key=lambda item: item[order_key])
            if len(group) > expected_count:
                windows = [group[index : index + expected_count] for index in range(len(group) - expected_count + 1)]
            else:
                windows = [group]
            for window in windows:
                aligned = np.array([item[align_key] for item in window], dtype=np.float32)
                ordered = np.array([item[order_key] for item in window], dtype=np.float32)
                spacing = np.diff(ordered)
                if spacing.size and float(np.min(spacing)) < 3.0:
                    continue
                align_std = float(np.std(aligned))
                span = float(ordered[-1] - ordered[0]) if len(ordered) > 1 else 0.0
                area_std = float(np.std(np.array([item["area"] for item in window], dtype=np.float32)))
                score = (align_std, area_std, span)
                if best_score is None or score < best_score:
                    best_score = score
                    best_group = window
        return best_group

    def sample_legend_swatch_colors(self, image_path, expected_count):
        """Sample visible legend swatches locally instead of asking the MLLM to guess hex colors."""
        if expected_count < 2:
            return []
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return []
        height, width = image.shape[:2]
        candidates = self._legend_swatch_candidates(image)
        if not candidates:
            return []

        regions = [
            (
                "right",
                "vertical",
                [item for item in candidates if item["cx"] >= width * 0.60],
            ),
            (
                "left",
                "vertical",
                [item for item in candidates if item["cx"] <= width * 0.40],
            ),
            (
                "top",
                "horizontal",
                [item for item in candidates if item["cy"] <= height * 0.30],
            ),
            (
                "bottom",
                "horizontal",
                [item for item in candidates if item["cy"] >= height * 0.70],
            ),
        ]
        best = None
        best_score = None
        for region_name, orientation, region_candidates in regions:
            group = self._aligned_legend_group(region_candidates, expected_count, orientation)
            if not group:
                continue
            overflow = max(0, len(region_candidates) - expected_count)
            align_key = "cx" if orientation == "vertical" else "cy"
            align_std = float(np.std(np.array([item[align_key] for item in group], dtype=np.float32)))
            edge_priority = 0 if region_name in {"right", "left"} else 1
            score = (overflow, align_std, edge_priority)
            if best_score is None or score < best_score:
                best_score = score
                best = (orientation, group)
        if not best:
            return []
        orientation, group = best
        order_key = "cy" if orientation == "vertical" else "cx"
        return [item["color"] for item in sorted(group, key=lambda item: item[order_key])]

    def refine_legend_series_colors_from_image(self, image_path, axis_repair):
        series_items = axis_repair.get("series_items") if isinstance(axis_repair, dict) else None
        if not isinstance(series_items, dict):
            return
        if str(series_items.get("kind") or "").strip().lower() != "legend":
            return
        items = series_items.get("items")
        if not isinstance(items, list) or len(items) < 2:
            return
        sampled_colors = self.sample_legend_swatch_colors(image_path, len(items))
        if len(sampled_colors) != len(items):
            return
        for item, sampled_color in zip(items, sampled_colors):
            if isinstance(item, dict):
                item["color"] = sampled_color
        series_items["source"] = "upload_detection+local_legend_swatch"
        series_items["color_source"] = "local_legend_swatch"

    def save_detection_debug(self, image_path, prompt, content, parsed):
        debug_dir = os.getenv("CHART_TYPE_DEBUG_DIR")
        if not debug_dir:
            return
        try:
            import time
            from pathlib import Path

            path = Path(debug_dir)
            path.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            debug_path = path / f"{stem}_{int(time.time())}.json"
            debug_path.write_text(
                json.dumps(
                    {
                        "image_path": str(image_path),
                        "model": self.model_name,
                        "prompt": prompt,
                        "raw_response": content,
                        "parsed_response": parsed,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as error:
            print(f"[Warning] Failed to save chart-type debug payload: {error}")

    def default_axis_repair(self, reason=""):
        return {
            "x_axis_missing": False,
            "y_axis_missing": False,
            "x_ticks_missing": False,
            "y_ticks_missing": False,
            "x_axis_role": "unknown",
            "y_axis_role": "unknown",
            "x_axis_position": "unknown",
            "y_axis_position": "unknown",
            "plot_area_style": "unknown",
            "has_background_grid": False,
            "x_tick_recovery_from_grid": False,
            "y_tick_recovery_from_grid": False,
            "bar_layout": "unknown",
            "bar_orientation": "unknown",
            "confidence": 0.0,
            "reason": reason,
            "radar_grid_shape": "unknown",
            "radar_grid_confidence": 0.0,
            "axis_tick_labels": {
                "x_ticks": [],
                "y_ticks": [],
                "x_axis_type": "unknown",
                "y_axis_type": "unknown",
                "source": "fallback",
            },
            "series_items": {"items": [], "source": "fallback", "kind": "unknown"},
        }

    def detect_chart_type(self, image_path):
        try:
            base64_image = self.image_to_base64(image_path)

            prompt = f"""
Analyze this chart image.

Supported chart types:
{format_chart_options()}

Return strict JSON only:
{{
  "type": "<chart_type>",
  "confidence": <number between 0 and 1>,
  "axis_repair": {{
    "x_axis_missing": <true_or_false>,
    "y_axis_missing": <true_or_false>,
    "x_ticks_missing": <true_or_false>,
    "y_ticks_missing": <true_or_false>,
    "x_axis_role": "numeric" | "category" | "date" | "unknown",
    "y_axis_role": "numeric" | "category" | "date" | "unknown",
    "x_axis_position": "bottom" | "top" | "middle" | "none" | "unknown",
    "y_axis_position": "left" | "right" | "middle" | "both" | "none" | "unknown",
    "plot_area_style": "explicit_axes" | "weak_axes" | "grid_only" | "no_axes" | "unknown",
    "has_background_grid": <true_or_false>,
    "x_tick_recovery_from_grid": <true_or_false>,
    "y_tick_recovery_from_grid": <true_or_false>,
    "bar_layout": "single" | "grouped" | "stacked" | "dense" | "unknown",
    "bar_orientation": "vertical" | "horizontal" | "unknown",
    "confidence": <number between 0 and 1>,
    "reason": "<short reason>"
  }},
  "axis_tick_labels": {{
    "x_axis_type": "numeric" | "category" | "date" | "unknown",
    "y_axis_type": "numeric" | "category" | "date" | "unknown",
    "x_ticks": ["visible x-axis tick labels in visual order"],
    "y_ticks": ["visible y-axis tick labels in visual order"]
  }},
  "radar_grid": {{
    "shape": "polygon" | "circular" | "unknown",
    "confidence": <number between 0 and 1>,
    "reason": "<short reason>"
  }},
  "series_items": {{
    "kind": "legend" | "point_labels" | "single_series" | "none" | "unknown",
    "items": [
      {{"name": "<visible legend/series/category/point label>", "color": "#RRGGBB or null"}}
    ]
  }}
}}

Rules:
- "type" must be one of {format_supported_types()}.
- Use "pie" for a circular chart divided into slices that represent parts of
  a whole and has no center hole. Use "donut" for the same part-of-whole chart
  when it has a center hole or ring shape.
- Use "rose" only for radial bar / nightingale / polar area charts where
  values are encoded by radial length or radius, usually with a polar value
  scale. Do not classify ordinary pie/donut/progress-ring charts as rose just
  because they contain circular sectors or rings.
- Mark an axis or tick marks as missing only when the visible chart clearly
  lacks that axis line or the short tick marks.
- Be conservative: for normal charts with visible axes/ticks, set every
  axis_repair flag to false.
- Missing tick labels alone are not a repair signal here.
- plot_area_style describes the visible plotting frame: use "explicit_axes"
  for clear x/y axis strokes, "weak_axes" for faint/partial axes, "grid_only"
  when the chart relies on background gridlines instead of axis/tick strokes,
  and "no_axes" only when no axis or grid structure is visible.
- x_axis_position and y_axis_position describe where the visible axis/tick
  labels are located in the plot. Use "right" when the numeric Y-axis labels
  are on the right side, and "middle" when the axis crosses through the data
  region rather than sitting on the plot boundary.
- Set x_tick_recovery_from_grid or y_tick_recovery_from_grid to true only when
  tick marks are absent or unreliable but background grid lines align with the
  tick labels and can be used to recover tick positions.
- For bar charts, x_axis_role/y_axis_role should reflect category vs numeric
  roles, and bar_layout should distinguish single, grouped, stacked, or dense
  bars. Do not use bar_orientation to override the chart type; it is diagnostic.
- axis_tick_labels is a conservative upload-time prior. Copy only labels that
  are visible in the image. Preserve units and signs if shown. For numeric
  axes, include the numeric tick labels in the same visual order as the axis
  labels. Do not invent missing labels.
- series_items is also a conservative upload-time extraction. It must describe
  data-mark colors, not grid/axis/text/background colors.
- Treat series_items as a focused color-binding task inside this response:
  read the color from visible pixels of the mark/swatch/line/marker/slice,
  not from a default chart palette or a semantic color name.
- If a legend is visible, return each legend label with the color of the
  immediately adjacent legend swatch, line, or marker. Preserve the exact
  name-color pairing and legend order.
- For line charts, match each legend text to the colored line sample beside it;
  do not approximate all colors as shades of one palette unless the visible
  legend samples really look that way.
- Do not substitute a named/default palette. If the exact legend swatch color is
  uncertain, use null instead of guessing a familiar palette color.
- For pie/donut charts, pair legend labels or direct labels with the
  corresponding slice/ring fill color. Do not omit colors merely because the
  chart type is non-Cartesian.
- For single-color charts with no visible legend, return kind "single_series"
  with one item. Use the visible measure/series/title text when readable, or
  "Series 1" only when no meaningful series name is visible. Do not enumerate
  axis/category labels as separate series when all marks share one color.
- For bar charts with no visible legend but visible category labels, return one
  item per visible bar/category label only when bar colors differ by category
  or group. Use the fill color sampled from that bar, including gradients.
- For scatter or bubble charts with labels printed next to marks and no legend,
  return those point labels with their marker colors when visible.
- Only return "Series 1" when there is one plotted series and no readable
  legend/category/point label can be attached to the mark color.
- Use null for color only when the color is genuinely ambiguous. Do not infer
  hidden labels.
- For blue/gray/single-color marks, output the actual sampled hex color, not a
  canonical color such as #0000FF or #808080 unless the pixels really match it.
- Do not return future ticks, axis tick labels, or category labels as
  series_items unless they are directly tied to distinct colored marks.
- For radar charts only, set radar_grid.shape to "polygon" when the grid/rings
  are straight-edged polygons, "circular" when the rings are circular, otherwise
  "unknown". For non-radar charts, use "unknown".
"""

            content = chat_with_gemini_sync(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                model=self.model_name,
                max_tokens=int(os.getenv("CHART_TYPE_MAX_TOKENS", "4096")),
                temperature=0,
                response_format={"type": "json_object"},
            )
            if not content or content == FAILURE_TEXT:
                raise Exception("API request failed")

            parsed_result = self.extract_json_response(content)
            result = self.normalize_detection_result(parsed_result)
            self.save_detection_debug(image_path, prompt, content, result)
            if result is None or "type" not in result or "confidence" not in result:
                raise ValueError("LLM result does not match required format")

            chart_type = self.normalize_chart_type_strict(result["type"])

            confidence = result["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                raise ValueError(f"Invalid confidence value: {confidence}")

            axis_repair = self.normalize_axis_repair(result)
            chart_type = self.apply_bar_layout_type_hint(chart_type, axis_repair)
            self.refine_legend_series_colors_from_image(image_path, axis_repair)

            print(f"[Success] Chart type detected: {chart_type}, confidence={confidence}")
            return {
                "type": chart_type,
                "confidence": float(confidence),
                "axis_repair": axis_repair,
            }

        except Exception as error:
            print(f"[Error] Chart type detection failed: {error}")
            raise


def detect_chart(image_path):
    detector = ChartTypeDetector()
    return detector.detect_chart_type(image_path)


if __name__ == "__main__":
    test_image = "./data/upload/radar_000.png"
    result = detect_chart(test_image)
    print(result)
