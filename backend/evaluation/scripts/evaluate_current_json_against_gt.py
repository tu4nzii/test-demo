import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent
GRID_DIR = BACKEND_DIR / "Grid_generation"

sys.path.insert(0, str(GRID_DIR))

from function_calling.axis.detect_lines import detect_candidate_lines  # noqa: E402
from function_calling.axis.infer_axes import infer_axes_from_lines  # noqa: E402
from function_calling.axis.merge_lines import merge_similar_lines  # noqa: E402


CURRENT_RESULT_FILES = [
    BACKEND_DIR / "evaluation" / "results" / "line_full_test_20260527_234736.json",
    BACKEND_DIR / "evaluation" / "results" / "cartesian_non_line_merged_rerun_latest.json",
]
CHART_TYPES = ["line", "scatter", "bubble", "v_bar", "h_bar", "radar", "rose", "pie", "donut"]
EVALUABLE_CARTESIAN_TYPES = {"line", "scatter", "bubble", "v_bar", "h_bar"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: list[float]) -> Optional[float]:
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.fmean(values) if values else None


def safe_median(values: list[float]) -> Optional[float]:
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.median(values) if values else None


def percentile(values: list[float], q: float) -> Optional[float]:
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(math.floor(index))
    high = int(math.ceil(index))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)


def normalize_label(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def as_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def numeric_key(value: Any) -> Optional[str]:
    number = as_float(value)
    if number is None:
        return None
    return f"{number:.8g}"


def is_numeric_axis(ticks: list[Any]) -> bool:
    return bool(ticks) and all(as_float(tick) is not None for tick in ticks)


def axis_span(pixels: list[Any]) -> Optional[float]:
    numeric = [as_float(pixel) for pixel in pixels]
    numeric = [pixel for pixel in numeric if pixel is not None]
    if len(numeric) < 2:
        return None
    span = max(numeric) - min(numeric)
    return span if span > 0 else None


def preferred_ticks(data: dict[str, Any], axis: str) -> tuple[list[Any], list[Any]]:
    encrypted_ticks = data.get(f"{axis}_ticks_encrypted")
    encrypted_pixels = data.get(f"{axis}_pixels_encrypted")
    if (
        isinstance(encrypted_ticks, list)
        and isinstance(encrypted_pixels, list)
        and len(encrypted_ticks) == len(encrypted_pixels)
        and encrypted_ticks
    ):
        return encrypted_ticks, encrypted_pixels
    ticks = data.get(f"{axis}_ticks", [])
    pixels = data.get(f"{axis}_pixels", [])
    return ticks if isinstance(ticks, list) else [], pixels if isinstance(pixels, list) else []


def semantic_axis_kind(chart_type: str, axis: str, ticks: list[Any]) -> str:
    if chart_type == "v_bar" and axis == "x":
        return "categorical"
    if chart_type == "h_bar" and axis == "y":
        return "categorical"
    if chart_type == "line" and axis == "x" and any(isinstance(tick, str) for tick in ticks):
        return "categorical"
    return "numeric" if is_numeric_axis(ticks) else "categorical"


def tick_map(ticks: list[Any], pixels: list[Any], axis_kind: Optional[str] = None) -> tuple[dict[str, float], str]:
    axis_kind = axis_kind or ("numeric" if is_numeric_axis(ticks) else "categorical")
    mapping: dict[str, float] = {}
    for tick, pixel in zip(ticks, pixels):
        numeric_pixel = as_float(pixel)
        if numeric_pixel is None:
            continue
        key = numeric_key(tick) if axis_kind == "numeric" else normalize_label(tick)
        if key is not None and key not in mapping:
            mapping[key] = numeric_pixel
    return mapping, axis_kind


def tick_position_metrics(gt: dict[str, Any], pred: dict[str, Any], axis: str, chart_type: str) -> dict[str, Any]:
    gt_ticks, gt_pixels = preferred_ticks(gt, axis)
    pred_ticks, pred_pixels = preferred_ticks(pred, axis)
    gt_kind = semantic_axis_kind(chart_type, axis, gt_ticks)
    pred_kind = semantic_axis_kind(chart_type, axis, pred_ticks)
    gt_map, gt_kind = tick_map(gt_ticks, gt_pixels, gt_kind)
    pred_map, pred_kind = tick_map(pred_ticks, pred_pixels, pred_kind)
    span = axis_span(gt_pixels)
    errors = []
    for key, gt_pixel in gt_map.items():
        if key in pred_map and span:
            errors.append(abs(pred_map[key] - gt_pixel) / span)
    matched = len(errors)
    total = len(gt_map)
    return {
        f"{axis}_axis_kind_gt": gt_kind,
        f"{axis}_axis_kind_pred": pred_kind,
        f"{axis}_tick_total_gt": total,
        f"{axis}_tick_matched": matched,
        f"{axis}_tick_coverage": matched / total if total else None,
        f"{axis}_tick_norm_mae": safe_mean(errors),
        f"{axis}_tick_norm_median": safe_median(errors),
        f"{axis}_tick_norm_p95": percentile(errors, 0.95),
    }


def fit_numeric_transform(ticks: list[Any], pixels: list[Any]) -> Optional[tuple[float, float]]:
    pairs = []
    for tick, pixel in zip(ticks, pixels):
        x = as_float(tick)
        y = as_float(pixel)
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    values = np.array([pair[0] for pair in pairs], dtype=float)
    coords = np.array([pair[1] for pair in pairs], dtype=float)
    if np.ptp(values) == 0:
        return None
    try:
        slope, intercept = np.polyfit(values, coords, 1)
    except Exception:
        return None
    return float(slope), float(intercept)


def axis_transform(data: dict[str, Any], axis: str, chart_type: str) -> dict[str, Any]:
    ticks, pixels = preferred_ticks(data, axis)
    mapping, kind = tick_map(ticks, pixels, semantic_axis_kind(chart_type, axis, ticks))
    transform: dict[str, Any] = {"kind": kind, "mapping": mapping, "span": axis_span(pixels)}
    if kind == "numeric":
        transform["numeric"] = fit_numeric_transform(ticks, pixels)
    return transform


def apply_transform(transform: dict[str, Any], value: Any) -> Optional[float]:
    if transform["kind"] == "numeric":
        number = as_float(value)
        params = transform.get("numeric")
        if number is None or params is None:
            return None
        slope, intercept = params
        return slope * number + intercept
    return transform["mapping"].get(normalize_label(value))


def flatten_points(chart_type: str, data_points: Any) -> list[tuple[Any, Any]]:
    points: list[tuple[Any, Any]] = []
    if not isinstance(data_points, dict):
        return points

    if chart_type in {"scatter", "bubble"}:
        for value in data_points.values():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                points.append((value[0], value[1]))
        return points

    if chart_type == "line":
        for series in data_points.values():
            if isinstance(series, dict):
                for x_value, y_value in series.items():
                    points.append((x_value, y_value))
        return points

    if chart_type == "v_bar":
        for series in data_points.values():
            if isinstance(series, dict):
                for category, value in series.items():
                    points.append((category, value))
        return points

    if chart_type == "h_bar":
        for series in data_points.values():
            if isinstance(series, dict):
                for category, value in series.items():
                    points.append((value, category))
        return points

    return points


def transform_mse(gt: dict[str, Any], pred: dict[str, Any], chart_type: str) -> dict[str, Any]:
    gt_x = axis_transform(gt, "x", chart_type)
    gt_y = axis_transform(gt, "y", chart_type)
    pred_x = axis_transform(pred, "x", chart_type)
    pred_y = axis_transform(pred, "y", chart_type)
    x_span = gt_x.get("span")
    y_span = gt_y.get("span")
    squared_errors = []
    normalized_squared_errors = []
    for x_value, y_value in flatten_points(chart_type, gt.get("data_points")):
        gt_px = apply_transform(gt_x, x_value)
        gt_py = apply_transform(gt_y, y_value)
        pred_px = apply_transform(pred_x, x_value)
        pred_py = apply_transform(pred_y, y_value)
        if None in {gt_px, gt_py, pred_px, pred_py}:
            continue
        dx = float(pred_px - gt_px)
        dy = float(pred_py - gt_py)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            continue
        squared_errors.append(dx * dx + dy * dy)
        if x_span and y_span:
            normalized_squared_errors.append((dx / x_span) ** 2 + (dy / y_span) ** 2)
    return {
        "coord_points": len(squared_errors),
        "coord_pixel_mse": safe_mean(squared_errors),
        "coord_norm_mse": safe_mean(normalized_squared_errors),
        "coord_pixel_rmse": math.sqrt(safe_mean(squared_errors)) if squared_errors else None,
        "coord_norm_rmse": math.sqrt(safe_mean(normalized_squared_errors)) if normalized_squared_errors else None,
    }


def line_angle_deg(line: list[int]) -> Optional[float]:
    if not line or len(line) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in line[:4]]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def angle_distance(angle: Optional[float], expected: float) -> Optional[float]:
    if angle is None:
        return None
    diff = abs((angle - expected + 90) % 180 - 90)
    return diff


def infer_current_axis_errors(image_path: Path) -> dict[str, Any]:
    try:
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"x_axis_angle_error_deg": None, "y_axis_angle_error_deg": None}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        raw_lines = detect_candidate_lines(
            gray,
            canny_threshold1=30,
            canny_threshold2=100,
            hough_threshold=15,
            min_length=15,
            max_gap=15,
        )
        merged = merge_similar_lines(raw_lines)
        x_axis, y_axis, _ = infer_axes_from_lines(merged, (w, h), gray)
        return {
            "x_axis_angle_error_deg": angle_distance(line_angle_deg(x_axis), 0.0),
            "y_axis_angle_error_deg": angle_distance(line_angle_deg(y_axis), 90.0),
            "axis_angle_mean_error_deg": safe_mean(
                [
                    value
                    for value in [
                        angle_distance(line_angle_deg(x_axis), 0.0),
                        angle_distance(line_angle_deg(y_axis), 90.0),
                    ]
                    if value is not None
                ]
            ),
        }
    except Exception:
        return {"x_axis_angle_error_deg": None, "y_axis_angle_error_deg": None, "axis_angle_mean_error_deg": None}


def chart_type_from_id(chart_id: str, fallback: Optional[str] = None) -> str:
    if fallback:
        return fallback
    for chart_type in CHART_TYPES:
        if chart_id.startswith(f"{chart_type}_"):
            return chart_type
    return "unknown"


def load_current_items() -> dict[tuple[str, str], dict[str, Any]]:
    items: dict[tuple[str, str], dict[str, Any]] = {}
    for result_path in CURRENT_RESULT_FILES:
        if not result_path.exists():
            continue
        data = read_json(result_path)
        for item in data.get("items", []):
            chart_id = item.get("chart_id")
            chart_type = chart_type_from_id(chart_id or "", item.get("chart_type"))
            if not chart_id or chart_type == "unknown":
                continue
            item = dict(item)
            item["chart_type"] = chart_type
            items[(chart_type, chart_id)] = item
    return items


def load_gt_index() -> dict[tuple[str, str], Path]:
    index = {}
    for chart_type in CHART_TYPES:
        chart_dir = BACKEND_DIR / "charts" / chart_type
        if not chart_dir.exists():
            continue
        for path in chart_dir.glob("*.json"):
            index[(chart_type, path.stem)] = path
    return index


def evaluate_item(chart_type: str, chart_id: str, gt_path: Path, current_item: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "chart_type": chart_type,
        "chart_id": chart_id,
        "gt_json_path": str(gt_path),
        "current_json_path": current_item.get("ticks_json_path"),
        "success": bool(current_item.get("success")),
        "error": current_item.get("error", ""),
    }
    if not current_item.get("success"):
        return record
    current_path = Path(current_item.get("ticks_json_path", ""))
    if not current_path.exists():
        record["success"] = False
        record["error"] = "current ticks json missing"
        return record
    gt = read_json(gt_path)
    pred = read_json(current_path)
    record.update(tick_position_metrics(gt, pred, "x", chart_type))
    record.update(tick_position_metrics(gt, pred, "y", chart_type))
    record.update(transform_mse(gt, pred, chart_type))
    image_path = Path(current_item.get("image_path") or BACKEND_DIR / "charts" / chart_type / f"{chart_id}.png")
    record.update(infer_current_axis_errors(image_path))
    return record


def aggregate(records: list[dict[str, Any]], gt_counts: dict[str, int]) -> dict[str, Any]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["chart_type"]].append(record)

    summaries = {}
    for chart_type in CHART_TYPES:
        rows = grouped.get(chart_type, [])
        evaluated = [row for row in rows if row.get("success")]
        summaries[chart_type] = {
            "gt_total": gt_counts.get(chart_type, 0),
            "current_total": len(rows),
            "evaluated": len(evaluated),
            "failed_or_missing": gt_counts.get(chart_type, 0) - len(evaluated),
        }
        for metric in [
            "axis_angle_mean_error_deg",
            "x_tick_norm_mae",
            "y_tick_norm_mae",
            "x_tick_coverage",
            "y_tick_coverage",
            "coord_pixel_mse",
            "coord_norm_mse",
            "coord_pixel_rmse",
            "coord_norm_rmse",
        ]:
            values = [
                row.get(metric)
                for row in evaluated
                if isinstance(row.get(metric), (int, float)) and math.isfinite(row.get(metric))
            ]
            summaries[chart_type][f"{metric}_mean"] = safe_mean(values)
            summaries[chart_type][f"{metric}_median"] = safe_median(values)
            summaries[chart_type][f"{metric}_p95"] = percentile(values, 0.95)
    return summaries


def html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_bar(value: Optional[float], max_value: float, invert: bool = False) -> str:
    if value is None or max_value <= 0:
        return '<span class="muted">N/A</span>'
    pct = max(0.0, min(100.0, value / max_value * 100))
    cls = "good" if invert else "bad"
    return f'<div class="bar"><span class="{cls}" style="width:{pct:.1f}%"></span></div><span>{fmt(value)}</span>'


def render_html(payload: dict[str, Any]) -> str:
    summaries = payload["summary_by_type"]
    rows = []
    for chart_type in CHART_TYPES:
        summary = summaries[chart_type]
        rows.append(
            "<tr>"
            f"<td><strong>{html_escape(chart_type)}</strong></td>"
            f"<td>{summary['gt_total']}</td>"
            f"<td>{summary['evaluated']}</td>"
            f"<td>{summary['failed_or_missing']}</td>"
            f"<td>{render_bar(summary.get('axis_angle_mean_error_deg_mean'), 5)}</td>"
            f"<td>{fmt(summary.get('x_tick_norm_mae_mean'), 5)} / {fmt(summary.get('x_tick_norm_mae_median'), 5)}</td>"
            f"<td>{fmt(summary.get('y_tick_norm_mae_mean'), 5)} / {fmt(summary.get('y_tick_norm_mae_median'), 5)}</td>"
            f"<td>{fmt(summary.get('coord_norm_mse_mean'), 6)} / {fmt(summary.get('coord_norm_mse_median'), 6)}</td>"
            f"<td>{fmt(summary.get('coord_pixel_rmse_mean'), 2)} / {fmt(summary.get('coord_pixel_rmse_median'), 2)}</td>"
            "</tr>"
        )

    worst = sorted(
        [row for row in payload["records"] if row.get("success")],
        key=lambda row: (
            row.get("coord_norm_mse") is None,
            -(row.get("coord_norm_mse") or -1),
        ),
    )[:40]
    worst_rows = []
    for row in worst:
        worst_rows.append(
            "<tr>"
            f"<td>{html_escape(row['chart_type'])}</td>"
            f"<td>{html_escape(row['chart_id'])}</td>"
            f"<td>{fmt(row.get('axis_angle_mean_error_deg'), 3)}</td>"
            f"<td>{fmt(row.get('x_tick_norm_mae'), 5)}</td>"
            f"<td>{fmt(row.get('y_tick_norm_mae'), 5)}</td>"
            f"<td>{fmt(row.get('coord_norm_mse'), 6)}</td>"
            f"<td>{fmt(row.get('coord_pixel_rmse'), 2)}</td>"
            f"<td>{fmt(row.get('x_tick_coverage'), 3)} / {fmt(row.get('y_tick_coverage'), 3)}</td>"
            "</tr>"
        )

    issue_rows = []
    for row in payload["records"]:
        if not row.get("success"):
            issue_rows.append(
                "<tr>"
                f"<td>{html_escape(row.get('chart_type'))}</td>"
                f"<td>{html_escape(row.get('chart_id'))}</td>"
                f"<td>{html_escape(row.get('error') or 'missing current recognition json')}</td>"
                "</tr>"
            )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Chart Recognition Evaluation</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; background: #f7f8fa; }}
    h1, h2 {{ margin: 0 0 14px; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; margin-top: 28px; }}
    .meta {{ color: #56616f; margin-bottom: 18px; line-height: 1.6; }}
    .panel {{ background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 9px 8px; text-align: left; vertical-align: middle; }}
    th {{ background: #eef2f6; color: #263445; position: sticky; top: 0; }}
    .bar {{ display:inline-block; width: 100px; height: 8px; background:#e6e9ef; border-radius: 4px; overflow:hidden; margin-right:8px; vertical-align:middle; }}
    .bar span {{ display:block; height:100%; }}
    .bad {{ background:#d95f59; }}
    .good {{ background:#2f9e73; }}
    .muted {{ color:#7b8794; }}
    .note {{ line-height:1.7; color:#4b5563; }}
    code {{ background:#eef2f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>图表识别 JSON 评估报告</h1>
  <div class="meta">
    生成时间：{html_escape(payload['generated_at'])}<br>
    真值目录：<code>{html_escape(payload['gt_root'])}</code><br>
    当前识别结果：<code>{html_escape(', '.join(payload['current_result_files']))}</code>
  </div>

  <div class="panel note">
    指标说明：轴线角度误差使用当前轴线检测流程从图片重新推断轴线，与水平/垂直理论方向比较；tick 位置误差按真值轴跨度归一化后计算匹配 tick 的 MAE；
    坐标变换 MSE 使用真值数据点分别经过真值 JSON 和当前识别 JSON 建立的坐标变换映射到像素空间后比较。当前没有识别 JSON 的极坐标/饼环图类型保留在表中，但不参与均值计算。
    表格中含斜杠的单元格格式为 <code>mean / median</code>，用于区分整体水平和少数异常值的影响。
  </div>

  <div class="panel">
    <h2>按图表类型汇总</h2>
    <table>
      <thead>
        <tr>
          <th>类型</th><th>真值样本</th><th>已评估</th><th>缺失/失败</th>
          <th>轴角误差均值(°)</th><th>X tick归一化MAE</th><th>Y tick归一化MAE</th>
          <th>坐标变换Norm MSE</th><th>坐标变换Pixel RMSE</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2>坐标变换误差最高样本</h2>
    <table>
      <thead><tr><th>类型</th><th>图表</th><th>轴角误差(°)</th><th>X tick MAE</th><th>Y tick MAE</th><th>Norm MSE</th><th>Pixel RMSE</th><th>Tick覆盖 X/Y</th></tr></thead>
      <tbody>{''.join(worst_rows)}</tbody>
    </table>
  </div>

  <div class="panel">
    <h2>缺失或失败样本</h2>
    <table>
      <thead><tr><th>类型</th><th>图表</th><th>原因</th></tr></thead>
      <tbody>{''.join(issue_rows) if issue_rows else '<tr><td colspan="3">无</td></tr>'}</tbody>
    </table>
  </div>
</body>
</html>"""


def main() -> int:
    gt_index = load_gt_index()
    current_items = load_current_items()
    gt_counts = defaultdict(int)
    for chart_type, _ in gt_index:
        gt_counts[chart_type] += 1

    records = []
    for (chart_type, chart_id), gt_path in sorted(gt_index.items()):
        current_item = current_items.get((chart_type, chart_id))
        if current_item is None:
            records.append(
                {
                    "chart_type": chart_type,
                    "chart_id": chart_id,
                    "gt_json_path": str(gt_path),
                    "current_json_path": None,
                    "success": False,
                    "error": "missing current recognition json",
                }
            )
            continue
        if chart_type not in EVALUABLE_CARTESIAN_TYPES:
            records.append(
                {
                    "chart_type": chart_type,
                    "chart_id": chart_id,
                    "gt_json_path": str(gt_path),
                    "current_json_path": current_item.get("ticks_json_path"),
                    "success": False,
                    "error": "non-cartesian metrics not implemented for current output schema",
                }
            )
            continue
        records.append(evaluate_item(chart_type, chart_id, gt_path, current_item))

    payload = {
        "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gt_root": str(BACKEND_DIR / "charts"),
        "current_result_files": [str(path) for path in CURRENT_RESULT_FILES],
        "summary_by_type": aggregate(records, dict(gt_counts)),
        "records": records,
    }

    out_dir = BACKEND_DIR / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "current_json_metric_report.json"
    html_path = out_dir / "current_json_metric_report.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")
    print(f"JSON_REPORT={json_path}")
    print(f"HTML_REPORT={html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
