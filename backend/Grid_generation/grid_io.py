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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GENERATED_DIR_NAMES = {
    "__pycache__",
    "ocr_label_lab",
}

def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image

def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(_windows_long_path(path))

def write_json_file(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(_windows_long_path(path), "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

def _windows_long_path(path: Path) -> str:
    text = str(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    absolute = str(path.resolve())
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")
    return "\\\\?\\" + absolute

def discover_images(root: Path, output_root: Path) -> list[Path]:
    images: list[Path] = []
    output_root = output_root.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_parts = path.relative_to(root).parts[:-1]
        if any(part.startswith("grid_") or part in GENERATED_DIR_NAMES for part in relative_parts):
            continue
        try:
            path.resolve().relative_to(output_root)
            continue
        except ValueError:
            images.append(path)
    return sorted(images)

def evenly_sample(paths: list[Path], sample_count: int | None) -> list[Path]:
    if sample_count is None or sample_count <= 0 or sample_count >= len(paths):
        return paths
    if sample_count == 1:
        return [paths[0]]
    indexes = np.linspace(0, len(paths) - 1, sample_count).round().astype(int)
    return [paths[int(i)] for i in indexes]
