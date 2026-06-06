"""Robust JSON extraction for model responses."""

from __future__ import annotations

import json
import re
from typing import Any


def clean_json_text(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?", "", str(text).strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(r"[\x00-\x1f]+", "", cleaned)
    cleaned = cleaned.replace("`", "").strip()
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = re.sub(r"\[x\s*,\s*y\]", "[-1, -1]", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'"?x"?\s*,\s*"?y"?', '"-1", "-1"', cleaned, flags=re.IGNORECASE)
    return cleaned


def balance_json_text(text: str) -> str:
    balanced = text
    diff_c = balanced.count("{") - balanced.count("}")
    diff_s = balanced.count("[") - balanced.count("]")
    if diff_s > 0:
        balanced += "]" * diff_s
    if diff_c > 0:
        balanced += "}" * diff_c
    return balanced


def extract_json_candidate(text: str) -> str | None:
    """Extract the first plausible JSON object/array from raw model text."""
    if not text:
        return None

    blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if blocks:
        return blocks[0].strip()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

    object_match = re.search(r"(\{[\s\S]*\})", text)
    if object_match:
        return object_match.group(1)
    array_match = re.search(r"(\[[\s\S]*\])", text)
    if array_match:
        return array_match.group(1)
    return None


def safe_json_loads(text: str, *, allow_substring: bool = True) -> Any | None:
    """Parse slightly malformed JSON returned by multimodal models."""
    cleaned = balance_json_text(clean_json_text(text))
    try:
        return json.loads(cleaned)
    except Exception:
        if not allow_substring:
            return None

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(start_char)
        end = cleaned.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(balance_json_text(cleaned[start : end + 1]))
            except Exception:
                pass
    return None


def parse_model_json(text: str) -> Any | None:
    candidate = extract_json_candidate(text)
    if candidate is None:
        return None
    return safe_json_loads(candidate)


def unwrap_openai_content(response: Any) -> str:
    """Return assistant content from OpenAI-compatible or legacy wrapper JSON."""
    parsed = safe_json_loads(response) if isinstance(response, str) else response
    if isinstance(parsed, dict):
        if "choices" in parsed:
            try:
                content = parsed["choices"][0]["message"]["content"]
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            except Exception:
                pass
        if "response" in parsed:
            content = parsed["response"]
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)


def extract_json_response(content: str) -> Any | None:
    """Compatibility helper used by polar scripts."""
    return parse_model_json(content)


def validate_xy(coords: object) -> bool:
    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        return False
    return all(isinstance(value, (int, float)) or value is None for value in coords)
