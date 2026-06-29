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

MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
)

def looks_like_date_label(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    compact = re.sub(r"\s+", " ", value).casefold()
    if re.fullmatch(r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?", compact):
        return True
    if re.search(rf"\b(?:{MONTH_PATTERN})\.?\s+\d{{1,2}}[,.]?\s+\d{{4}}\b", compact):
        return True
    if re.search(rf"\b\d{{1,2}}\s+(?:{MONTH_PATTERN})\.?\s+\d{{4}}\b", compact):
        return True
    return False

def parse_numeric_label(text: str) -> float | None:
    if looks_like_date_label(text):
        return None
    cleaned = text.strip().replace(",", "")
    cleaned = re.sub(r"(?<=\d)['’](?=\d{3}\b)", "", cleaned)
    if re.fullmatch(r"\d{4}[-/]\d{1,2}(?:[-/]\d{1,2})?", cleaned):
        return None
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if match is None:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    suffix = cleaned[match.end() :].strip().lower()
    if suffix.startswith("%"):
        return value
    if suffix.startswith(("km", "kg", "kph", "khz", "kwh")):
        return value
    if suffix in {"k", "k.", "k)"} or suffix.startswith("k "):
        return value * 1_000.0
    if suffix.startswith("m"):
        return value * 1_000_000.0
    if suffix.startswith("b"):
        return value * 1_000_000_000.0
    return value

def regularity_score(positions: list[float]) -> float:
    if len(positions) < 3:
        return 0.0
    diffs = np.diff(np.array(sorted(positions), dtype=np.float32))
    median = float(np.median(diffs))
    if median <= 0:
        return 0.0
    return float(np.median(np.abs(diffs - median)) / median)

def is_valid_tick_series(
    positions: list[float],
    image_extent: int,
    *,
    min_span_frac: float,
    max_irregularity: float = 0.45,
) -> bool:
    if len(positions) < 2:
        return False
    ordered = sorted(positions)
    span = ordered[-1] - ordered[0]
    if span < max(35.0, image_extent * min_span_frac):
        return False
    if len(ordered) >= 4 and regularity_score(ordered) > max_irregularity:
        return False
    return True

def circular_residual(values: np.ndarray, offset: float, step: float) -> np.ndarray:
    return np.abs(((values - offset + step / 2) % step) - step / 2)
