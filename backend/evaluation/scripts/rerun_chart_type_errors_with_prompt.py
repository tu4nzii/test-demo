"""Rerun chart-type detection for misclassified samples with a focused prompt.

This is an offline probe. The model receives only the image and prompt; GT type
is used after the response to score the probe.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from collections import Counter
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
from type_detection.chart_registry import format_supported_types  # noqa: E402
from type_detection.chart_type import ChartTypeDetector  # noqa: E402


SUPPORTED = {
    "rose",
    "radar",
    "v_bar",
    "h_bar",
    "line",
    "scatter",
    "bubble",
    "donut",
    "pie",
}


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


def normalize_type(value: Any) -> str:
    detector = ChartTypeDetector()
    try:
        return detector.normalize_chart_type_strict(value)
    except Exception:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "bar": "v_bar",
            "vertical_bar": "v_bar",
            "horizontal_bar": "h_bar",
            "stacked_bar": "v_bar",
            "vertical_stacked_bar": "v_bar",
            "horizontal_stacked_bar": "h_bar",
            "v_stacked_bar": "v_bar",
            "h_stacked_bar": "h_bar",
            "scatterplot": "scatter",
            "scatter_plot": "scatter",
            "bubble_chart": "bubble",
            "pie_chart": "pie",
            "donut_chart": "donut",
            "radar_chart": "radar",
            "rose_chart": "rose",
        }
        text = aliases.get(text, text)
        return text if text in SUPPORTED else "unknown"


def build_prompt() -> str:
    return f"""
You are a strict chart-type classifier. Return JSON only.

Allowed chart types: {format_supported_types()}

Return:
{{
  "type": "<one allowed chart type>",
  "confidence": <0 to 1>,
  "evidence": {{
    "mark_orientation": "vertical" | "horizontal" | "radial" | "point" | "line" | "unknown",
    "marker_size_role": "uniform" | "varies_as_data" | "unknown",
    "polar_shape": "pie" | "donut" | "rose" | "radar" | "none" | "unknown"
  }},
  "reason": "<short visual reason>"
}}

Decision rules:
- Use v_bar when bars are vertical rectangles whose value is encoded by height,
  extending upward/downward from a horizontal baseline.
- Use h_bar when bars are horizontal rectangles whose value is encoded by
  length, extending left/right from a vertical baseline.
- Stacked bar is not an allowed output class. If bars are stacked, still output
  v_bar for vertical stacked bars or h_bar for horizontal stacked bars. Grouped
  side-by-side bars are also v_bar or h_bar according to orientation.
- Use scatter when the marks are points with roughly uniform size and position
  encodes the data values.
- Use bubble when the marks are circles and circle area/radius visibly encodes
  an additional variable. If marker sizes vary substantially across points or
  there is a size/diameter legend, prefer bubble over scatter.
- A scatterplot with slightly anti-aliased or perspective-varying dots is still
  scatter if marker size is not a data channel.
- Use line when data points are connected by lines as the primary mark.
- Use pie for a filled circular part-of-whole chart with no center hole.
- Use donut for a circular part-of-whole chart with a center hole/ring.
- Use rose only for radial bars or Nightingale/polar area charts where radial
  length encodes magnitude. Do not call ordinary pie/donut charts rose.
- Use radar for multiple radial axes/spokes with polygon/circular grid and
  connected values around the axes.
- Do not infer from the filename, dataset path, or title wording alone; classify
  by visible chart geometry.
- If uncertain between two types, choose the type supported by the dominant
  data marks, not by decorative legends or annotations.
""".strip()


def call_model(image_path: Path) -> dict[str, Any]:
    prompt = build_prompt()
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
        max_tokens=1024,
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = None if not content or content == FAILURE_TEXT else extract_json_response(content)
    pred_type = normalize_type(parsed.get("type") if isinstance(parsed, dict) else None)
    return {
        "model": get_model_name(),
        "prompt": prompt,
        "raw_response": content,
        "parsed_response": parsed,
        "pred_type": pred_type,
    }


def source_image_for_record(record: dict[str, Any]) -> Path | None:
    source = record.get("source_image")
    if source and Path(str(source)).exists():
        return Path(str(source))
    copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
    for key in ("source_image", "uploaded_image"):
        path = copied.get(key)
        if path and Path(str(path)).exists():
            return Path(str(path))
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("backend/evaluation/results/vishintprompt_full_latest_report/chart_type_errors/chart_type_errors.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/evaluation/results/vishintprompt_full_latest_report/chart_type_errors_prompt_rerun"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = read_json(args.errors)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(payload.get("records", []), start=1):
        if not isinstance(record, dict):
            continue
        dataset_relative = str(record.get("dataset_relative") or "")
        gt_type = str(record.get("gt_type") or "")
        old_pred_type = str(record.get("pred_type") or "")
        source_image = source_image_for_record(record.get("record") if isinstance(record.get("record"), dict) else record)
        sample_json = args.output / f"{index:02d}_{Path(dataset_relative).stem}_rerun.json"
        if sample_json.exists() and not args.force:
            model_result = read_json(sample_json).get("model_result", {})
        elif source_image is not None:
            model_result = call_model(source_image)
            write_json(
                sample_json,
                {
                    "dataset_relative": dataset_relative,
                    "gt_type": gt_type,
                    "old_pred_type": old_pred_type,
                    "source_image": str(source_image),
                    "model_result": model_result,
                },
            )
        else:
            model_result = {"pred_type": "unknown", "error": "source_image_missing"}

        new_pred_type = str(model_result.get("pred_type") or "unknown")
        rows.append(
            {
                "index": index,
                "dataset": record.get("dataset"),
                "dataset_relative": dataset_relative,
                "gt_type": gt_type,
                "old_pred_type": old_pred_type,
                "new_pred_type": new_pred_type,
                "old_correct": old_pred_type == gt_type,
                "new_correct": new_pred_type == gt_type,
                "source_image": str(source_image) if source_image else None,
                "sample_json": sample_json.name,
                "reason": (
                    model_result.get("parsed_response", {}).get("reason")
                    if isinstance(model_result.get("parsed_response"), dict)
                    else ""
                ),
            }
        )

    total = len(rows)
    old_correct = sum(1 for row in rows if row["old_correct"])
    new_correct = sum(1 for row in rows if row["new_correct"])
    confusion = Counter((row["gt_type"], row["new_pred_type"]) for row in rows if not row["new_correct"])

    write_json(
        args.output / "rerun_summary.json",
        {
            "total": total,
            "old_correct": old_correct,
            "new_correct": new_correct,
            "new_accuracy": new_correct / total if total else None,
            "remaining_confusion": [
                {"gt_type": gt, "pred_type": pred, "count": count}
                for (gt, pred), count in confusion.most_common()
            ],
            "records": rows,
        },
    )

    lines = [
        "# 图表分类错误样本提示词重跑",
        "",
        "- 模型输入只包含图像和优化后的分类提示词，不包含 GT。",
        "- GT 只在模型返回后用于离线对比。",
        f"- 重跑样本数：{total}",
        f"- 原错误集中旧结果正确数：{old_correct}/{total}",
        f"- 优化提示词重跑正确数：{new_correct}/{total} ({new_correct / total * 100 if total else 0:.2f}%)",
        "",
        "## 剩余混淆",
        "",
        "| GT类型 | 新预测类型 | 数量 |",
        "| --- | --- | ---: |",
    ]
    for (gt_type, pred_type), count in confusion.most_common():
        lines.append(f"| `{gt_type}` | `{pred_type}` | {count} |")
    if not confusion:
        lines.append("| - | - | 0 |")
    lines.extend(
        [
            "",
            "## 样本明细",
            "",
            "| 序号 | 样本 | GT | 旧预测 | 新预测 | 是否修正 | 新判断理由 | 原始输出 |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        fixed = "是" if row["new_correct"] else "否"
        reason = str(row.get("reason") or "-").replace("|", "/")
        lines.append(
            f"| {row['index']} | `{row['dataset_relative']}` | `{row['gt_type']}` | "
            f"`{row['old_pred_type']}` | `{row['new_pred_type']}` | {fixed} | {reason} | "
            f"`{row['sample_json']}` |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
