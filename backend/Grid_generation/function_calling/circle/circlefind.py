import base64
import json
import os
import re
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
backend_root = os.path.abspath(os.path.join(current_dir, "../../.."))
for path in (project_root, backend_root):
    if path not in sys.path:
        sys.path.insert(0, path)

from model_api_config import get_chat_completion_url, get_headers, get_model_name  # noqa: E402


def find_center_by_hough_transform(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: cannot read image {image_path}")
        return image, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    detected_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=20,
        param2=30,
        minRadius=10,
        maxRadius=300,
    )

    if detected_circles is None:
        print("Hough transform failed: no circle detected.")
        return image, None, None

    first_circle = np.uint16(np.around(detected_circles))[0, 0]
    cx, cy, r = int(first_circle[0]), int(first_circle[1]), int(first_circle[2])
    r = round(r + 0.5)
    center_coordinates = (cx, cy)
    cv2.circle(image, center_coordinates, r, (0, 255, 0), 2)
    cv2.circle(image, center_coordinates, 2, (0, 0, 255), -1)
    cv2.putText(
        image,
        f"Center: ({cx}, {cy})",
        (cx + 15, cy + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )
    print(f"Hough circle center/radius: {center_coordinates}, {r}")
    return image, center_coordinates, r


def _encode_cv_image(cv_image) -> str:
    ok, encoded = cv2.imencode(".png", cv_image)
    if not ok:
        raise ValueError("Failed to encode cv image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _extract_response_text(result: dict) -> str:
    if "choices" in result and result["choices"]:
        message = result["choices"][0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content)

    if "candidates" in result and result["candidates"]:
        parts = result["candidates"][0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))

    return ""


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def get_radial_chart_scale_info_from_vlm(cv_image, center_x, center_y, hough_radius_px):
    print("--- Calling unified MLLM for radial scale info ---")

    prompt = f"""
This is a polar/radar-style chart image. A green circle has been drawn around a
detected concentric guide. Its approximate center is ({center_x}, {center_y})
in pixels and its pixel radius is {hough_radius_px}.

Please analyze the chart and return only strict JSON:
{{
  "hough_circle_value": <the tick value represented by the green circle>,
  "polar_center_value": <the value at the polar center, use 0 if the center is 0>,
  "max_tick_value": <the maximum visible radial tick value>
}}

Use null for any value that cannot be identified.
"""

    payload = {
        "model": os.getenv("CIRCLE_LLM_MODEL") or os.getenv("MLLM_MODEL") or get_model_name(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{_encode_cv_image(cv_image)}",
                        },
                    },
                ],
            }
        ],
        "temperature": float(os.getenv("CIRCLE_LLM_TEMPERATURE", "0")),
    }

    try:
        response = requests.post(
            get_chat_completion_url(),
            headers=get_headers(),
            json=payload,
            timeout=int(os.getenv("CIRCLE_LLM_TIMEOUT_SECONDS", "180")),
        )
        response.raise_for_status()
        vlm_raw_text = _extract_response_text(response.json()).strip()
        print(f"Radial scale raw response:\n{vlm_raw_text}")

        vlm_data = _parse_json_object(vlm_raw_text)
        return_info = {
            "hough_circle_value": vlm_data.get("hough_circle_value"),
            "polar_center_value": vlm_data.get("polar_center_value"),
            "max_tick_value": vlm_data.get("max_tick_value"),
        }
        print(f"Parsed radial scale info: {return_info}")
        return return_info
    except Exception as e:
        print(f"Unified MLLM radial scale request failed: {e}")
        return {
            "hough_circle_value": None,
            "polar_center_value": None,
            "max_tick_value": None,
        }


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def encrypt_radial_chart_with_tick(image_path, interval):
    initial_image, center_coords, hough_radius_px = find_center_by_hough_transform(image_path)

    if center_coords is None or hough_radius_px is None:
        print(f"Could not detect a circle in {image_path}.")
        return None

    center_x, center_y = center_coords
    scale_info = get_radial_chart_scale_info_from_vlm(
        initial_image, center_x, center_y, hough_radius_px
    )

    hough_circle_value = _as_float(scale_info.get("hough_circle_value"))
    chart_origin_value = _as_float(scale_info.get("polar_center_value"))
    max_tick_value = _as_float(scale_info.get("max_tick_value"))

    if None in [hough_circle_value, chart_origin_value, max_tick_value]:
        print(f"Incomplete scale info; cannot encrypt {image_path}.")
        return None

    if (hough_circle_value - chart_origin_value) == 0:
        print("Invalid scale info: hough value equals center value.")
        return None

    pixels_per_value = float(hough_radius_px) / (hough_circle_value - chart_origin_value)
    print(f"Pixels per tick unit: {pixels_per_value}")

    current_image = initial_image.copy()
    current_tick = chart_origin_value
    while current_tick <= max_tick_value:
        encrypted_radius_px = round((current_tick - chart_origin_value) * pixels_per_value + 0.5)
        cv2.circle(current_image, center_coords, encrypted_radius_px, (255, 0, 255), 1)
        text_x = center_coords[0] + encrypted_radius_px + 10
        text_y = center_coords[1]
        cv2.putText(
            current_image,
            str(current_tick),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            1,
        )
        print(f"Added encrypted radial tick {current_tick}: radius {encrypted_radius_px}px")
        current_tick += interval

    return current_image


if __name__ == "__main__":
    img_dir = "./data/basic_images/circle"
    out_dir = "./data/out_encrypted_gemini_multiple"
    os.makedirs(out_dir, exist_ok=True)

    fname = "circle2.png"
    img_path = os.path.join(img_dir, fname)
    interval = 2.5
    encrypted_image = encrypt_radial_chart_with_tick(img_path, interval)
    out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_encrypted_intervals.jpg")

    if encrypted_image is not None:
        cv2.imwrite(out_path, encrypted_image)
        print(f"Saved encrypted result to {out_path}")
    else:
        print(f"Failed to process {fname}.")
