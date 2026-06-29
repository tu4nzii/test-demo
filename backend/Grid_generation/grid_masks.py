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

def make_line_masks(
    image: np.ndarray,
    *,
    sat_max: int,
    white_cutoff: int,
    min_gray: int,
    contrast_min: int,
    include_dark: bool,
    dark_cutoff: int,
    min_line_frac: float,
    gap_frac: float,
    max_thickness_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    blur_size = max(15, int(round(min(h, w) * 0.035)))
    if blur_size % 2 == 0:
        blur_size += 1
    local_background = cv2.medianBlur(gray, blur_size)
    local_contrast = cv2.absdiff(gray, local_background)

    neutral_line = (
        (sat <= sat_max)
        & (gray >= min_gray)
        & (gray <= white_cutoff)
        & (local_contrast >= contrast_min)
    )
    if include_dark:
        neutral_line |= gray <= dark_cutoff
    candidate = np.where(neutral_line, 255, 0).astype(np.uint8)

    h_min_len = max(15, int(round(w * min_line_frac)))
    v_min_len = max(15, int(round(h * min_line_frac)))
    h_gap = max(3, int(round(w * gap_frac)))
    v_gap = max(3, int(round(h * gap_frac)))
    max_thickness = max(3, int(round(min(h, w) * max_thickness_frac)))

    h_connected = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_gap, 1)),
    )
    v_connected = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_gap)),
    )

    horizontal = cv2.morphologyEx(
        h_connected,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (h_min_len, 1)),
    )
    vertical = cv2.morphologyEx(
        v_connected,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_min_len)),
    )

    horizontal = filter_line_components(
        horizontal,
        orientation="horizontal",
        min_length=h_min_len,
        max_thickness=max_thickness,
    )
    vertical = filter_line_components(
        vertical,
        orientation="vertical",
        min_length=v_min_len,
        max_thickness=max_thickness,
    )

    horizontal = cv2.dilate(horizontal, np.ones((1, 3), np.uint8), iterations=1)
    vertical = cv2.dilate(vertical, np.ones((3, 1), np.uint8), iterations=1)
    return candidate, horizontal, vertical

def filter_line_components(
    mask: np.ndarray,
    *,
    orientation: str,
    min_length: int,
    max_thickness: int,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, count):
        x, y, width, height, _ = stats[label]
        if orientation == "horizontal":
            keep = width >= min_length and height <= max_thickness
        else:
            keep = height >= min_length and width <= max_thickness
        if keep:
            filtered[labels == label] = 255
    return filtered
