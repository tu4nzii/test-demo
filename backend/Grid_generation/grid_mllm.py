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

def sanitize_error(error: Exception | str) -> str:
    text = str(error)
    return re.sub(r"(key=|Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1<redacted>", text)[:300]

def extract_json_object(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None

def mllm_prompt(ocr_axis_evidence: dict[str, object] | None = None, grid_geometry_evidence: dict[str, object] | None = None) -> str:
    return (
        "You are independently reading chart axis text from the image only. "
        "Do not assume any OCR result; no OCR text is provided. Return JSON only, no markdown. "
        "Read the x-axis and y-axis tick labels, axis titles, axis type, other visible non-axis text, and recommended grid intent. "
        "Tick labels must be separate items in visual order; do not merge multiple ticks into one string. "
        "Do not omit intermediate visible ticks in a regular sequence. Use empty strings or empty arrays when absent. "
        "Schema: {"
        "\"chart_type\":\"line|bar|scatter|bubble|area|unknown\","
        "\"grid_intent\":\"existing|reconstruct_from_ticks|avoid|unknown\","
        "\"x_axis\":{\"type\":\"numeric|time|category|mixed|unknown\",\"confidence\":0-1,"
        "\"axis_label\":{\"text\":\"axis title or empty string\",\"confidence\":0-1},"
        "\"tick_labels\":[\"separate visible tick labels in visual order\"],"
        "\"tick_order\":\"left_to_right|right_to_left|unknown\"},"
        "\"y_axis\":{\"type\":\"numeric|time|category|mixed|unknown\",\"confidence\":0-1,"
        "\"axis_label\":{\"text\":\"axis title or empty string\",\"confidence\":0-1},"
        "\"tick_labels\":[\"separate visible tick labels in visual order\"],"
        "\"tick_order\":\"top_to_bottom|bottom_to_top|unknown\"},"
        "\"other_texts\":[\"visible non-axis text such as title, legend, annotations; empty if none\"],"
        "\"recommended_grid\":{\"horizontal\":\"existing|reconstruct|avoid|unknown\",\"vertical\":\"existing|reconstruct|avoid|unknown\"},"
        "\"reason\":\"short\"}."
    )

def image_to_base64_png(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image for MLLM")
    return base64.b64encode(encoded.tobytes()).decode("ascii")

def post_json(url: str, payload: dict[str, object], headers: dict[str, str], timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {"raw": parsed}

def run_mllm_axis_extraction(
    image: np.ndarray,
    ocr_axis_evidence: dict[str, object],
    grid_geometry_evidence: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    if not args.mllm:
        return {"enabled": False, "error": "MLLM disabled"}

    api_key = os.environ.get(args.mllm_api_key_env, "")
    if not api_key:
        return {"enabled": True, "error": f"Missing API key env: {args.mllm_api_key_env}"}

    endpoint = args.mllm_endpoint or os.environ.get("MLLM_ENDPOINT", "")
    model = args.mllm_model
    prompt = mllm_prompt(ocr_axis_evidence, grid_geometry_evidence)
    image_b64 = image_to_base64_png(image)
    headers = {"Content-Type": "application/json"}

    try:
        if endpoint and "/chat/completions" in endpoint:
            headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            response = post_json(endpoint, payload, headers, args.mllm_timeout)
            text = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        else:
            if not endpoint:
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            sep = "&" if "?" in endpoint else "?"
            url = f"{endpoint}{sep}key={api_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
            }
            response = post_json(url, payload, headers, args.mllm_timeout)
            candidates = response.get("candidates", [])
            parts = []
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
            text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"enabled": True, "model": model, "error": sanitize_error(exc)}

    parsed = extract_json_object(text)
    if parsed is None:
        return {"enabled": True, "model": model, "error": "MLLM returned non-JSON", "raw": text[:1000]}
    parsed["enabled"] = True
    parsed["model"] = model
    parsed["error"] = None
    return parsed

def grid_arbitration_prompt(candidate_summary: dict[str, object]) -> str:
    compact = json.dumps(candidate_summary, ensure_ascii=False)[:5000]
    return (
        "You are the final arbiter choosing the best reconstructed chart grid from visual crops. "
        "The image contains four rows: original crops, then combined_mask, tick_supplement, and semantic_guide. "
        "Columns are x-axis crop and y-axis crop. Each candidate crop overlays grid lines and bound labels. "
        "Choose x_axis_vertical_grid_choice by judging the x-axis crop, and choose y_axis_horizontal_grid_choice by judging the y-axis crop. "
        "For bar charts or category axes, grid lines should align with category label midlines and bar centers; semantic_guide is specifically designed for this. "
        "When an axis is category/bar-position based and semantic_guide is valid and visually comparable to tick_supplement, choose semantic_guide for that axis unless it is visibly wrong. "
        "This is especially important for horizontal bar charts: y_axis_horizontal_grid_choice should favor semantic_guide when it better represents the centers of the bars/categories. "
        "For numeric axes, prefer candidates whose lines pass through the visible tick positions and bound labels. "
        "Prefer visible chart grid/tick evidence when it is well aligned, but do not penalize semantic_guide merely for being label-midline based. "
        "Ignore candidates that have many extra lines, missing obvious tick positions, shifted grid lines, or labels bound to wrong grid lines. "
        "The numeric summary includes rule scores and invalid reasons; use it as supporting evidence, but make the final choice from the visual crop and binding quality. "
        "If the summary contains a position_tie_analysis or tie_break_review, the image is a focused local comparison for tied candidates. "
        "In that focused crop, candidate marker colors are listed in tie_break_review.legend; cyan marks the trusted OCR/label center when available. Choose the candidate whose local grid line best matches the visible tick/label position, and prefer the candidate closer to the cyan trusted label center unless visible tick evidence clearly contradicts it. "
        "Do not choose candidates marked invalid unless every valid candidate is visibly worse. "
        "Return JSON only. Schema: {"
        "\"x_axis_vertical_grid_choice\":\"combined_mask|tick_supplement|semantic_guide\","
        "\"y_axis_horizontal_grid_choice\":\"combined_mask|tick_supplement|semantic_guide\","
        "\"confidence\":0-1,"
        "\"reason\":\"short\"}. "
        f"Candidate numeric summary: {compact}"
    )

def run_mllm_grid_arbitration(
    review_image: np.ndarray,
    candidate_summary: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    if not args.mllm:
        return {"enabled": False, "error": "MLLM disabled"}
    api_key = os.environ.get(args.mllm_api_key_env, "")
    if not api_key:
        return {"enabled": True, "error": f"Missing API key env: {args.mllm_api_key_env}"}

    endpoint = args.mllm_endpoint or os.environ.get("MLLM_ENDPOINT", "")
    model = args.mllm_model
    prompt = grid_arbitration_prompt(candidate_summary)
    image_b64 = image_to_base64_png(review_image)
    headers = {"Content-Type": "application/json"}
    try:
        if endpoint and "/chat/completions" in endpoint:
            headers["Authorization"] = f"Bearer {api_key}"
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        ],
                    }
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            response = post_json(endpoint, payload, headers, args.mllm_timeout)
            text = str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        else:
            if not endpoint:
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            sep = "&" if "?" in endpoint else "?"
            url = f"{endpoint}{sep}key={api_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                        ],
                    }
                ],
                "generationConfig": {"temperature": 0, "response_mime_type": "application/json"},
            }
            response = post_json(url, payload, headers, args.mllm_timeout)
            candidates = response.get("candidates", [])
            parts = []
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
            text = "\n".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"enabled": True, "model": model, "error": sanitize_error(exc)}

    parsed = extract_json_object(text)
    if parsed is None:
        return {"enabled": True, "model": model, "error": "MLLM returned non-JSON", "raw": text[:1000]}
    parsed.pop("ocr_check_fields", None)
    for axis_key in ("x_axis", "y_axis"):
        axis = parsed.get(axis_key)
        if isinstance(axis, dict):
            axis.pop("merged_or_crowded_labels", None)
    parsed["enabled"] = True
    parsed["model"] = model
    parsed["error"] = None
    return parsed
