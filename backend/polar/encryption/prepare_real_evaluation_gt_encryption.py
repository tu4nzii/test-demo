"""Prepare GT-encrypted real radar/rose charts for downstream evaluation.

The generated grids intentionally bypass runtime circle/OCR detection and use
the trusted GT metadata already selected by the axis-prior evaluation:
``center``, ``r_ticks``, ``r_pixels``, ``theta_ticks`` and ``theta_angles``.
This keeps the final value-evaluation dataset aligned with the coordinate-prior
numbers that are reported in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT_DIR = Path(__file__).resolve().parents[3]
POLAR_DATA = ROOT_DIR / "backend" / "data" / "polar"
REAL_RADAR_DIR = ROOT_DIR / "backend" / "real" / "RadarChart-18 & RoseChart-6" / "RadarChart-18-final"
REAL_ROSE_DIR = ROOT_DIR / "backend" / "real" / "RadarChart-18 & RoseChart-6" / "RoseChart-6"
RADAR_EVAL_CSV = POLAR_DATA / "output" / "radar_grid_eval" / "radar_grid_eval_real_gt-nearest.csv"
ROSE_EVAL_CSV = POLAR_DATA / "output" / "rose_grid_eval" / "rose_grid_eval_real_corrected_gt-nearest.csv"
DEFAULT_OUTPUT_DIR = POLAR_DATA / "real_evaluation_data"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def numbers(values: Any) -> list[float]:
    if isinstance(values, str):
        values = json.loads(values)
    return [float(value) for value in values]


def center_xy(data: dict[str, Any]) -> tuple[int, int]:
    raw = data.get("center") or data.get("pred_coords")
    if isinstance(raw, dict):
        return int(round(float(raw["x"]))), int(round(float(raw["y"])))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(round(float(raw[0]))), int(round(float(raw[1])))
    raise ValueError("missing center")


def font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def fit_argument(r_ticks: list[float], r_pixels: list[float]) -> dict[str, float | str]:
    if len(r_ticks) != len(r_pixels) or len(r_ticks) < 2:
        raise ValueError("r_ticks and r_pixels must have the same length >= 2")
    x = np.asarray(r_ticks, dtype=float)
    y = np.asarray(r_pixels, dtype=float)
    a, b = np.polyfit(x, y, 1)
    return {"a": float(a), "b": float(b), "source": "gt_linear_fit"}


def gt_pairs(r_ticks: list[float], r_pixels: list[float]) -> list[tuple[float, float, bool]]:
    pairs = sorted(zip(r_ticks, r_pixels), key=lambda item: item[1])
    if pairs and pairs[0][0] > 0 and pairs[0][1] > 0:
        pairs = [(0.0, 0.0), *pairs]

    output: list[tuple[float, float, bool]] = []
    for tick, radius in pairs:
        if radius > 0:
            output.append((tick, radius, True))
    for (tick_a, radius_a), (tick_b, radius_b) in zip(pairs, pairs[1:]):
        if radius_b <= radius_a:
            continue
        output.append(((tick_a + tick_b) / 2.0, (radius_a + radius_b) / 2.0, False))

    deduped: list[tuple[float, float, bool]] = []
    for tick, radius, original in sorted(output, key=lambda item: item[1]):
        if deduped and abs(radius - deduped[-1][1]) < 1e-6:
            if original and not deduped[-1][2]:
                deduped[-1] = (tick, radius, original)
            continue
        deduped.append((tick, radius, original))
    return deduped


def draw_dashed_circle(
    image: np.ndarray,
    center: tuple[int, int],
    radius: float,
    color: tuple[int, int, int] = (128, 128, 128),
    thickness: int = 1,
) -> None:
    circumference = max(1, int(2 * math.pi * radius))
    dash_length = 2
    gap_length = 3
    cx, cy = center
    for step in range(0, circumference, dash_length + gap_length):
        angle_start = 2 * math.pi * step / circumference
        angle_end = 2 * math.pi * (step + dash_length) / circumference
        x1 = int(round(cx + radius * math.cos(angle_start)))
        y1 = int(round(cy + radius * math.sin(angle_start)))
        x2 = int(round(cx + radius * math.cos(angle_end)))
        y2 = int(round(cy + radius * math.sin(angle_end)))
        cv2.line(image, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


def draw_tick_labels(
    image: np.ndarray,
    center: tuple[int, int],
    pairs: list[tuple[float, float, bool]],
    chart_type: str,
) -> np.ndarray:
    h, w = image.shape[:2]
    cx, cy = center
    min_spacing = min(
        [abs(pairs[i + 1][1] - pairs[i][1]) for i in range(len(pairs) - 1)]
        or [min(h, w) * 0.05]
    )
    font_size = max(7, min(16, int(round(min(min_spacing * 0.55, min(h, w) * 0.018)))))
    label_offset = max(2, int(round(font_size * 0.35)))
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    pil_font = font(font_size)
    fill = (0, 0, 0)

    for tick, radius, _ in pairs:
        if radius <= 0:
            continue
        text = format_tick(tick)
        bbox = draw.textbbox((0, 0), text, font=pil_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        positions = [
            (cx - tw / 2, cy - radius - th / 2),
            (cx + radius + label_offset, cy - th / 2),
        ]
        if chart_type == "radar":
            positions.extend(
                [
                    (cx - tw / 2, cy + radius + label_offset),
                    (cx - radius - tw - label_offset, cy - th / 2),
                ]
            )

        for x, y in positions:
            draw.text((x, y), text, font=pil_font, fill=fill)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def resolve_image(data: dict[str, Any], json_path: Path, source_dir: Path) -> Path:
    direct = data.get("image")
    if direct:
        candidate = Path(str(direct))
        if candidate.exists():
            return candidate

    image_paths = data.get("image_paths") or {}
    for key in ("no_grid", "with_grid"):
        rel = image_paths.get(key)
        if not rel:
            continue
        for candidate in (json_path.parent / rel, source_dir / rel):
            if candidate.exists():
                return candidate

    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found for {json_path}")


def draw_gt_encryption(
    image_path: Path,
    data: dict[str, Any],
    out_png: Path,
    chart_type: str,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"image unreadable: {image_path}")

    cx, cy = center_xy(data)
    r_ticks = numbers(data.get("r_ticks", []))
    r_pixels = numbers(data.get("r_pixels", []))
    if len(r_ticks) != len(r_pixels) or len(r_ticks) < 2:
        raise ValueError("missing or invalid r_ticks/r_pixels")

    pairs = gt_pairs(r_ticks, r_pixels)
    result = image.copy()
    for _, radius, is_original in pairs:
        if is_original:
            continue
        draw_dashed_circle(result, (cx, cy), radius)

    result = draw_tick_labels(result, (cx, cy), pairs, chart_type)
    cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), result)

    return {
        "center": [cx, cy],
        "r_ticks": r_ticks,
        "r_pixels": r_pixels,
        "argument": fit_argument(r_ticks, r_pixels),
        "encrypted_r_ticks": [tick for tick, _, _ in pairs],
        "encrypted_r_pixels": [radius for _, radius, _ in pairs],
        "encrypted_ring_ticks": [tick for tick, _, original in pairs if not original],
        "encrypted_ring_pixels": [radius for _, radius, original in pairs if not original],
    }


def axis_labels(data: dict[str, Any]) -> dict[str, str]:
    labels = [str(value) for value in data.get("theta_ticks", [])]
    angles = numbers(data.get("theta_angles", data.get("axes_angles", [])))
    return {label: format_tick(angle) for label, angle in zip(labels, angles)}


def kept_rows(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("fallback") == "False":
                rows.append(row)
    return rows


def write_dataset_item(
    source_json: Path,
    source_image: Path,
    out_dir: Path,
    chart_type: str,
    output_stem: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = read_json(source_json)

    image_ext = source_image.suffix or ".png"
    no_grid_name = f"{output_stem}{image_ext}"
    with_grid_name = f"{output_stem}_gt_encrypt{image_ext}"
    out_image = out_dir / no_grid_name
    out_grid = out_dir / with_grid_name
    shutil.copy2(source_image, out_image)

    gt_meta = draw_gt_encryption(source_image, data, out_grid, chart_type)
    updated = dict(data)
    updated["chart_id"] = updated.get("chart_id") or output_stem
    updated["chart_type"] = chart_type
    updated["center"] = gt_meta["center"]
    updated["pred_coords"] = gt_meta["center"]
    updated["r_ticks"] = gt_meta["r_ticks"]
    updated["r_pixels"] = gt_meta["r_pixels"]
    updated["argument"] = gt_meta["argument"]
    updated["axis_labels"] = axis_labels(updated)
    updated["image_paths"] = {
        **(updated.get("image_paths") or {}),
        "no_grid": no_grid_name,
        "with_grid": with_grid_name,
        "grid_with_grid": with_grid_name,
    }
    updated["image"] = no_grid_name
    updated["json"] = f"{output_stem}.json"
    updated["output"] = with_grid_name
    updated["source_files"] = {
        "json": str(source_json),
        "image": str(source_image),
    }
    if chart_type == "rose":
        updated["axes_angles"] = numbers(updated.get("theta_angles", []))
    if chart_type == "radar" and "data" not in updated and "data_points" in updated:
        updated["data"] = updated["data_points"]
    updated["encryption"] = {
        "source": "groundtruth_metadata",
        "rule": "double_density_labels_midpoint_rings",
        **{key: value for key, value in gt_meta.items() if key.startswith("encrypted_")},
    }

    out_json = out_dir / f"{output_stem}.json"
    write_json(out_json, updated)
    return {
        "chart_id": updated["chart_id"],
        "chart_type": chart_type,
        "source_json": str(source_json),
        "source_image": str(source_image),
        "output_json": str(out_json),
        "output_image": str(out_image),
        "output_grid": str(out_grid),
        "center": gt_meta["center"],
        "r_ticks": gt_meta["r_ticks"],
        "r_pixels": gt_meta["r_pixels"],
    }


def prepare_radar(output_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in kept_rows(RADAR_EVAL_CSV):
        source_json = Path(row["json_path"])
        source_image = Path(row["image_path"])
        stem = source_json.stem
        results.append(write_dataset_item(source_json, source_image, output_root / "radar", "radar", stem))
    return results


def prepare_rose(output_root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in kept_rows(ROSE_EVAL_CSV):
        chart_id = row["chart_id"]
        source_json = REAL_ROSE_DIR / f"{chart_id}_gt_encrypt.json"
        data = read_json(source_json)
        source_image = resolve_image(data, source_json, REAL_ROSE_DIR)
        results.append(write_dataset_item(source_json, source_image, output_root / "rose", "rose", chart_id))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GT-encrypted real radar/rose evaluation data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chart-type", choices=["all", "radar", "rose"], default="all")
    args = parser.parse_args()

    summary: dict[str, Any] = {"output_dir": str(args.output_dir), "radar": [], "rose": []}
    if args.chart_type in {"all", "radar"}:
        summary["radar"] = prepare_radar(args.output_dir)
    if args.chart_type in {"all", "rose"}:
        summary["rose"] = prepare_rose(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "gt_encryption_summary.json"
    write_json(summary_path, summary)
    print(f"[summary] {summary_path}")
    print(f"[radar] {len(summary['radar'])} charts")
    print(f"[rose] {len(summary['rose'])} charts")


if __name__ == "__main__":
    main()
