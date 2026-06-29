"""Probe legend/series color extraction prompts on a few cached samples.

This helper is for offline diagnosis only. It never sends GT JSON to the
model; GT is loaded after the model response only to score and display the
comparison.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from gemini_calls import FAILURE_TEXT, chat_with_gemini_sync  # noqa: E402
from model_api_config import get_model_name  # noqa: E402
from evaluation.scripts.evaluate_vishintprompt_latest_metrics import (  # noqa: E402
    artifact_payload,
    flatten_colors,
    normalize_text,
    source_config_path,
)
from evaluation.scripts.diagnose_legend_color_binding import (  # noqa: E402
    color_distance,
    diagnose_item,
)


TARGET_STATUSES = (
    "name_exists_color_far",
    "likely_swapped_binding",
    "missing_pred_colors",
    "name_missing_color_far",
    "name_missing_color_close",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def image_to_base64(image_path: Path) -> str:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image file: {image_path}")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {image_path}")
    return base64.b64encode(np.ascontiguousarray(encoded)).decode("utf-8")


def extract_json_response(content: str) -> dict[str, Any] | None:
    text = content or ""
    decoder = json.JSONDecoder()
    candidates = []
    import re

    for fence in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE):
        candidates.append(fence.strip())
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start() :])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                continue
    return None


def normalize_probe_colors(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        series = value.get("series_items")
        if isinstance(series, dict):
            raw_items = series.get("items")
    if not isinstance(raw_items, list):
        return flatten_colors(value.get("series_color")) or flatten_colors(value.get("colors"))
    result: dict[str, str] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        color = str(item.get("color") or "").strip()
        if not name or not color.startswith("#") or len(color) != 7:
            continue
        result[name] = color
    return result


def build_color_prompt() -> str:
    return """
You are extracting the visible mapping between chart data labels and mark colors.

Return strict JSON only:
{
  "series_items": {
    "kind": "legend" | "direct_labels" | "single_series" | "none" | "unknown",
    "items": [
      {
        "name": "<exact visible legend/category/series label>",
        "color": "#RRGGBB or null",
        "evidence": "<legend swatch / line sample / bar fill / slice fill / marker>"
      }
    ]
  },
  "notes": "<short note>"
}

Rules:
- Do not infer from a default palette, chart library, or semantic color names.
- Read the color from pixels adjacent to the label: legend swatch, colored line
  sample, marker, bar fill, pie/donut slice, or labeled sector.
- Preserve the visual order of the legend or directly labeled marks.
- If there is a visible legend, every item must pair the legend text with the
  immediately adjacent swatch/line/marker color, not with the plotted order.
- If there is no visible legend but bars/slices/points are directly labeled,
  return each visible label with its own mark color.
- If a single-color chart has only one series name, return that series name and
  the actual plotted mark color.
- Use null only when the color cannot be visually tied to that label.
- Output hex colors as sampled visual colors, not named colors and not broad
  approximations.
""".strip()


def call_probe_model(image_path: Path) -> dict[str, Any]:
    prompt = build_color_prompt()
    content = chat_with_gemini_sync(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_to_base64(image_path)}"},
                    },
                ],
            }
        ],
        model=get_model_name(),
        max_tokens=2048,
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = None if not content or content == FAILURE_TEXT else extract_json_response(content)
    return {
        "model": get_model_name(),
        "prompt": prompt,
        "raw_response": content,
        "parsed_response": parsed,
        "colors": normalize_probe_colors(parsed),
    }


def color_accuracy(gt_colors: dict[str, str], pred_colors: dict[str, str], threshold: float) -> tuple[int, int]:
    total = len(gt_colors)
    correct = 0
    for gt_name, gt_color in gt_colors.items():
        diagnosis = diagnose_item(gt_name, gt_color, pred_colors, threshold=threshold)
        if diagnosis.get("status") == "correct":
            correct += 1
    return correct, total


def pick_records(diagnosis: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    picked: list[dict[str, Any]] = []
    used_samples: set[str] = set()
    records = diagnosis.get("records", [])
    for target in TARGET_STATUSES:
        for row in records:
            sample = str(row.get("dataset_relative") or "")
            if sample in used_samples:
                continue
            statuses = {str(item.get("status")) for item in row.get("items", []) if isinstance(item, dict)}
            if target in statuses:
                picked.append(row)
                used_samples.add(sample)
                break
            if len(picked) >= limit:
                return picked
    if len(picked) < limit:
        for row in records:
            sample = str(row.get("dataset_relative") or "")
            if sample not in used_samples:
                picked.append(row)
                used_samples.add(sample)
            if len(picked) >= limit:
                break
    return picked[:limit]


def manifest_by_relative(batch_root: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(batch_root / "manifest.json")
    return {
        str(record.get("dataset_relative")): record
        for record in manifest.get("records", [])
        if isinstance(record, dict)
    }


def copy_if_exists(src: Any, dst: Path) -> str | None:
    if not src:
        return None
    path = Path(str(src))
    if not path.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)
    return dst.name


def format_pct(correct_total: tuple[int, int]) -> str:
    correct, total = correct_total
    return "-" if total == 0 else f"{correct}/{total} ({correct / total * 100:.2f}%)"


def color_table(gt_colors: dict[str, str], old_colors: dict[str, str], new_colors: dict[str, str], threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gt_name, gt_color in gt_colors.items():
        old_diag = diagnose_item(gt_name, gt_color, old_colors, threshold=threshold)
        new_diag = diagnose_item(gt_name, gt_color, new_colors, threshold=threshold)
        rows.append(
            {
                "name": gt_name,
                "gt_color": gt_color,
                "old_status": old_diag.get("status"),
                "old_color": old_diag.get("pred_color")
                or ((old_diag.get("best_color_match") or [None, None, None])[1]),
                "old_distance": old_diag.get("distance")
                or ((old_diag.get("best_color_match") or [None, None, None])[2]),
                "new_status": new_diag.get("status"),
                "new_color": new_diag.get("pred_color")
                or ((new_diag.get("best_color_match") or [None, None, None])[1]),
                "new_distance": new_diag.get("distance")
                or ((new_diag.get("best_color_match") or [None, None, None])[2]),
            }
        )
    return rows


def format_distance(value: Any) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.1f}"
    return "-"


def format_item_status(value: Any) -> str:
    status_map = {
        "correct": "正确",
        "missing_pred_colors": "预测缺失颜色",
        "name_exists_invalid_color": "名称匹配但颜色无效",
        "name_exists_color_far": "名称匹配但颜色偏差大",
        "likely_swapped_binding": "疑似颜色绑定错位",
        "name_missing_color_close": "名称不匹配但颜色接近",
        "name_missing_color_far": "名称不匹配且颜色偏差大",
    }
    return status_map.get(str(value or ""), str(value or "-"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("backend/datasets/VisHintPrompt_datasets"))
    parser.add_argument("--batch-root", type=Path, default=Path("backend/evaluation/recheck_outputs/vishintprompt_full_grid_encryption_latest"))
    parser.add_argument("--diagnosis", type=Path, default=Path("backend/evaluation/results/legend_color_prompt_probe/legend_color_diagnosis.json"))
    parser.add_argument("--output", type=Path, default=Path("backend/evaluation/results/legend_color_prompt_probe/samples"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--color-threshold", type=float, default=45.0)
    parser.add_argument("--force", action="store_true", help="Call the model again even if sample JSON exists.")
    args = parser.parse_args()

    diagnosis = read_json(args.diagnosis)
    manifest = manifest_by_relative(args.batch_root)
    picked = pick_records(diagnosis, args.limit)
    args.output.mkdir(parents=True, exist_ok=True)

    sample_reports = []
    for index, diag_row in enumerate(picked, start=1):
        dataset_relative = str(diag_row.get("dataset_relative"))
        record = manifest.get(dataset_relative)
        if not record:
            continue
        sample_dir = args.output / f"{index:02d}_{Path(dataset_relative).stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
        source_image = Path(str(record.get("source_image") or copied.get("source_image") or copied.get("uploaded_image")))
        source_name = copy_if_exists(source_image, sample_dir / "source.png")
        grid_name = copy_if_exists(copied.get("encrypted_grid"), sample_dir / "image_with_grid.png")
        color_grid_name = copy_if_exists(copied.get("colored_grid"), sample_dir / "image_with_grid_color.png")

        gt_path = source_config_path(args.dataset_root, dataset_relative)
        gt = read_json(gt_path) if gt_path and gt_path.exists() else {}
        pred = artifact_payload(record)
        gt_colors = flatten_colors(gt.get("series_color"))
        old_colors = flatten_colors(pred.get("series_color")) or flatten_colors(pred.get("colors"))
        payload_path = sample_dir / "legend_color_probe.json"
        if payload_path.exists() and not args.force:
            payload = read_json(payload_path)
        else:
            new_probe = call_probe_model(source_image)
            new_colors = new_probe.get("colors", {})
            table = color_table(gt_colors, old_colors, new_colors, args.color_threshold)
            payload = {
                "dataset_relative": dataset_relative,
                "record_status": record.get("status"),
                "chart_type": record.get("chart_type"),
                "source_image": str(source_image),
                "gt_config": str(gt_path) if gt_path else None,
                "current_cached_colors": old_colors,
                "probe_colors": new_colors,
                "current_accuracy": {
                    "text": format_pct(color_accuracy(gt_colors, old_colors, args.color_threshold)),
                },
                "probe_accuracy": {
                    "text": format_pct(color_accuracy(gt_colors, new_colors, args.color_threshold)),
                },
                "comparison": table,
                "probe_model_io": new_probe,
            }
            write_json(payload_path, payload)
        sample_reports.append((sample_dir, payload, source_name, grid_name, color_grid_name))

    status_counts = diagnosis.get("summary", {}).get("status_counts", {})
    total_items = sum(int(value) for value in status_counts.values())
    strict_correct = int(status_counts.get("correct", 0))
    close_or_swapped = strict_correct + int(status_counts.get("name_missing_color_close", 0)) + int(
        status_counts.get("likely_swapped_binding", 0)
    )
    lines = [
        "# 图例颜色提示词抽样诊断",
        "",
        "- 模型输入只包含图像和颜色抽取提示词，不包含 GT。",
        "- GT 只在模型返回后用于离线对照打分。",
        f"- 颜色距离阈值：{args.color_threshold}px/RGB Euclidean。",
        "",
        "## 全量诊断概览",
        "",
        f"- 严格名称+颜色正确：{strict_correct}/{total_items} ({strict_correct / total_items * 100:.2f}%)"
        if total_items
        else "- 严格名称+颜色正确：-",
        f"- 颜色接近或疑似绑定错位：{close_or_swapped}/{total_items} ({close_or_swapped / total_items * 100:.2f}%)"
        if total_items
        else "- 颜色接近或疑似绑定错位：-",
        "",
        "| 诊断类型 | 数量 |",
        "| --- | ---: |",
    ]
    for status, count in sorted(status_counts.items(), key=lambda item: (-int(item[1]), item[0])):
        lines.append(f"| {format_item_status(status)} | {count} |")
    lines.append("")
    for sample_dir, payload, source_name, grid_name, color_grid_name in sample_reports:
        rel_dir = sample_dir.relative_to(args.output)
        lines.extend(
            [
                f"## {payload['dataset_relative']}",
                "",
                f"- 当前缓存颜色准确率：{payload['current_accuracy']['text']}",
                f"- 专用提示词颜色准确率：{payload['probe_accuracy']['text']}",
                f"- 图表类型：`{payload.get('chart_type')}`",
                "",
            ]
        )
        if source_name:
            lines.append(f"![原图]({rel_dir.as_posix()}/{source_name})")
        if grid_name:
            lines.append(f"![灰色网格]({rel_dir.as_posix()}/{grid_name})")
        if color_grid_name:
            lines.append(f"![彩色网格]({rel_dir.as_posix()}/{color_grid_name})")
        lines.extend(
            [
                "",
                "| 名称 | GT颜色 | 当前缓存颜色/状态/距离 | 专用提示词颜色/状态/距离 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in payload["comparison"]:
            lines.append(
                "| {name} | `{gt}` | `{old}` / {old_status} / {old_dist} | `{new}` / {new_status} / {new_dist} |".format(
                    name=str(item["name"]).replace("|", "/"),
                    gt=item["gt_color"],
                    old=item.get("old_color") or "-",
                    old_status=format_item_status(item.get("old_status")),
                    old_dist=format_distance(item.get("old_distance")),
                    new=item.get("new_color") or "-",
                    new_status=format_item_status(item.get("new_status")),
                    new_dist=format_distance(item.get("new_distance")),
                )
            )
        lines.extend(["", f"- 完整 JSON：`{rel_dir.as_posix()}/legend_color_probe.json`", ""])

    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
