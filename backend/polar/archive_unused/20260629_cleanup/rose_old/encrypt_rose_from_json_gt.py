"""Draw rose-chart encrypted grids directly from JSON ground-truth metadata.

This debug helper bypasses circle/radius detection.  It is intended for
checking whether real-chart JSON ``center`` / ``r_pixels`` are visually
consistent with the source image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = Path(__file__).resolve().parents[2]

REAL_ROSE_DIR = (
    BACKEND_DIR
    / "real"
    / "RadarChart-18 & RoseChart-6"
    / "RoseChart-6"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "output" / "rose_gt_encrypt"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _center(data: dict[str, Any]) -> tuple[int, int]:
    raw = data.get("center") or data.get("pred_coords")
    if isinstance(raw, dict):
        return int(round(float(raw["x"]))), int(round(float(raw["y"])))
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return int(round(float(raw[0]))), int(round(float(raw[1])))
    raise ValueError("missing center")


def _numbers(values: Any) -> list[float]:
    if isinstance(values, str):
        values = json.loads(values)
    return [float(value) for value in values]


def _format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def _font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_gt_grid(image_path: Path, json_path: Path, output_dir: Path) -> dict[str, Any]:
    data = _read_json(json_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"image unreadable: {image_path}")

    cx, cy = _center(data)
    r_ticks = _numbers(data.get("r_ticks", []))
    r_pixels = _numbers(data.get("r_pixels", []))
    theta_angles = _numbers(data.get("theta_angles", []))
    theta_ticks = [str(v) for v in data.get("theta_ticks", [])]

    positive_pairs = [
        (tick, radius)
        for tick, radius in zip(r_ticks, r_pixels)
        if abs(radius) > 1e-9
    ]
    if not positive_pairs:
        raise ValueError("no positive r_pixels")

    output = image.copy()
    h, w = output.shape[:2]
    outer_radius = max(radius for _, radius in positive_pairs)
    positive_pairs = sorted(positive_pairs, key=lambda item: item[1])
    draw_pairs: list[tuple[float, float, bool]] = []
    all_pairs = sorted(zip(r_ticks, r_pixels), key=lambda item: item[1])
    for (tick_a, radius_a), (tick_b, radius_b) in zip(all_pairs, all_pairs[1:]):
        if radius_b <= radius_a:
            continue
        midpoint_radius = (radius_a + radius_b) / 2.0
        midpoint_tick = (tick_a + tick_b) / 2.0
        if midpoint_radius > 0:
            draw_pairs.append((midpoint_tick, midpoint_radius, False))
        if radius_b > 0:
            draw_pairs.append((tick_b, radius_b, True))

    # De-duplicate possible repeated midpoint/original entries while keeping
    # exact JSON radii authoritative.
    deduped: list[tuple[float, float, bool]] = []
    for tick, radius, is_original in sorted(draw_pairs, key=lambda item: item[1]):
        if deduped and abs(radius - deduped[-1][1]) < 1e-6:
            if is_original and not deduped[-1][2]:
                deduped[-1] = (tick, radius, is_original)
            continue
        deduped.append((tick, radius, is_original))
    draw_pairs = deduped

    # Draw double-density encrypted rings in light gray.  Original tick rings
    # are slightly darker so the JSON radii can be visually inspected.
    for _, radius, original_tick_ring in draw_pairs:
        color = (95, 95, 95) if original_tick_ring else (155, 155, 155)
        cv2.circle(output, (cx, cy), int(round(radius)), color, 1, cv2.LINE_AA)

    # Draw all axes if present.
    for angle, label in zip(theta_angles, theta_ticks):
        rad = math.radians(angle)
        x2 = int(round(cx + outer_radius * math.cos(rad)))
        y2 = int(round(cy - outer_radius * math.sin(rad)))
        cv2.line(output, (cx, cy), (x2, y2), (115, 115, 115), 1, cv2.LINE_AA)

    cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # Put ring labels on the 0-degree ray and the 90-degree ray.  The labels
    # use all double-density ticks, while rings remain visible at half steps.
    pil_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font_size = max(8, min(18, int(round(min(h, w) * 0.022))))
    font = _font(font_size)
    label_fill = (0, 0, 0)

    for tick_value, radius, _ in draw_pairs:
        text = _format_tick(tick_value)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x_right = cx + radius - tw / 2
        y_right = cy - th - 2
        x_top = cx - tw / 2
        y_top = cy - radius - th / 2
        draw.text((x_right, y_right), text, font=font, fill=label_fill)
        draw.text((x_top, y_top), text, font=font, fill=label_fill)

    output = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{json_path.stem}_gt_encrypt.png"
    cv2.imwrite(str(out_path), output)

    meta = {
        "chart_id": data.get("chart_id", json_path.stem),
        "image": str(image_path),
        "json": str(json_path),
        "output": str(out_path),
        "center": [cx, cy],
        "r_ticks": r_ticks,
        "r_pixels": r_pixels,
        "theta_ticks": theta_ticks,
        "theta_angles": theta_angles,
    }
    meta_path = output_dir / f"{json_path.stem}_gt_encrypt.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def resolve_image(json_path: Path, data: dict[str, Any]) -> Path:
    direct_image = data.get("image")
    if direct_image:
        candidate = Path(str(direct_image))
        if candidate.exists():
            return candidate

    image_paths = data.get("image_paths") or {}
    rel = image_paths.get("with_grid") or image_paths.get("no_grid")
    if rel:
        candidates = [
            json_path.parent / rel,
            json_path.parent.parent / rel,
            REAL_ROSE_DIR.parent / rel,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"image not found for {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw rose encrypted grid from JSON GT metadata.")
    parser.add_argument(
        "--charts",
        nargs="+",
        default=["Rose1", "RoseDiagramExample2", "plotivy-nightingale-rose-chart"],
        help="Chart stems under the real rose directory.",
    )
    parser.add_argument("--json-dir", type=Path, default=REAL_ROSE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    results = []
    for chart in args.charts:
        json_path = args.json_dir / f"{chart}.json"
        data = _read_json(json_path)
        image_path = resolve_image(json_path, data)
        meta = draw_gt_grid(image_path, json_path, args.output_dir)
        print(f"[ok] {chart}: {meta['output']}")
        print(f"     center={meta['center']} r_pixels={meta['r_pixels']}")
        results.append(meta)

    summary_path = args.output_dir / "gt_encrypt_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
