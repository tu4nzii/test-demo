import base64
import json
import re

import cv2
import numpy as np
import requests

from model_api_config import get_chat_completion_url, get_headers, get_model_name
from type_detection.chart_registry import (
    DEFAULT_CHART_TYPE,
    SUPPORTED_CHART_TYPES,
    format_chart_options,
    format_supported_types,
    normalize_chart_type,
)


class ChartTypeDetector:
    """Detect chart type and optional missing-axis repair hints with an MLLM."""

    def __init__(self):
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        self.supported_types = SUPPORTED_CHART_TYPES

    def extract_json_response(self, content: str):
        try:
            match = re.search(r"(\{[\s\S]*\})", content or "")
            if not match:
                return None
            return json.loads(match.group(1))
        except Exception as error:
            print(f"[Error] JSON parse failed: {error}")
            return None

    def image_to_base64(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image file: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

        success, encoded_image = cv2.imencode(".jpg", image_rgb)
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

        return {
            "x_axis_missing": as_bool(repair.get("x_axis_missing", repair.get("x", False))),
            "y_axis_missing": as_bool(repair.get("y_axis_missing", repair.get("y", False))),
            "x_ticks_missing": as_bool(repair.get("x_ticks_missing", False)),
            "y_ticks_missing": as_bool(repair.get("y_ticks_missing", False)),
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(repair.get("reason", "") or ""),
            "radar_grid_shape": radar_grid_shape,
            "radar_grid_confidence": max(0.0, min(1.0, radar_grid_confidence)),
        }

    def default_axis_repair(self, reason=""):
        return {
            "x_axis_missing": False,
            "y_axis_missing": False,
            "x_ticks_missing": False,
            "y_ticks_missing": False,
            "confidence": 0.0,
            "reason": reason,
            "radar_grid_shape": "unknown",
            "radar_grid_confidence": 0.0,
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
    "confidence": <number between 0 and 1>,
    "reason": "<short reason>"
  }},
  "radar_grid": {{
    "shape": "polygon" | "circular" | "unknown",
    "confidence": <number between 0 and 1>,
    "reason": "<short reason>"
  }}
}}

Rules:
- "type" must be one of {format_supported_types()}.
- Mark an axis or tick marks as missing only when the visible chart clearly
  lacks that axis line or the short tick marks.
- Be conservative: for normal charts with visible axes/ticks, set every
  axis_repair flag to false.
- Missing tick labels alone are not a repair signal here.
- For radar charts only, set radar_grid.shape to "polygon" when the grid/rings
  are straight-edged polygons, "circular" when the rings are circular, otherwise
  "unknown". For non-radar charts, use "unknown".
"""

            payload = {
                "model": self.model_name,
                "messages": [
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
                "max_tokens": 900,
            }

            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code}, {response.text}")

            response_data = response.json()
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise Exception("Invalid API response format")

            content = response_data["choices"][0]["message"]["content"]
            result = self.extract_json_response(content)
            if result is None or "type" not in result or "confidence" not in result:
                raise ValueError("LLM result does not match required format")

            chart_type = normalize_chart_type(result["type"])
            if chart_type not in self.supported_types:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            confidence = result["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                raise ValueError(f"Invalid confidence value: {confidence}")

            print(f"[Success] Chart type detected: {chart_type}, confidence={confidence}")
            return {
                "type": chart_type,
                "confidence": float(confidence),
                "axis_repair": self.normalize_axis_repair(result),
            }

        except Exception as error:
            print(f"[Error] Chart type detection failed: {error}")
            return {
                "type": DEFAULT_CHART_TYPE,
                "confidence": 0.5,
                "axis_repair": self.default_axis_repair("type detection fallback"),
                "error": str(error),
            }


def detect_chart(image_path):
    detector = ChartTypeDetector()
    return detector.detect_chart_type(image_path)


if __name__ == "__main__":
    test_image = "./data/upload/radar_000.png"
    result = detect_chart(test_image)
    print(result)
