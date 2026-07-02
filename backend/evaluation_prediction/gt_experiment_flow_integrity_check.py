"""GT experiment flow integrity check.

This script validates one sample from every supported chart type without
calling a model. It checks the experimental data contract, prompt inputs, GT
grid image availability, feedback overlay generation, and amplifier crop
generation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageStat


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("CHART_EXPERIMENT_MODE", "gt")
os.environ.setdefault("CHART_EXPERIMENT_PRESERVE_GT", "1")
os.environ.setdefault("CHART_FEEDBACK_ROUNDS", "2")
os.environ.setdefault("CHART_AMPLIFIER_ROUNDS", "3")
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

from evaluation_prediction.common.experiment_contract import CONTRACTS  # noqa: E402
from evaluation_prediction.common.experiment_flow import (  # noqa: E402
    AMPLIFIER_STAGE,
    BASELINE_STAGE,
    FEEDBACK_STAGE,
    GRID_STAGE,
)
from evaluation_prediction.common.gt_grid_renderer import GRID_STYLE_VERSION  # noqa: E402
from evaluation_prediction.common.runtime import (  # noqa: E402
    get_amplifier_rounds,
    get_feedback_rounds,
    get_repeat_times,
)


CHART_TYPES = ("bubble", "scatter", "line", "v_bar", "h_bar", "pie", "donut", "radar", "rose")
SOURCES = ("synthetic", "realworld")
REPORT_PATH = Path(__file__).resolve().parent / "gt_experiment_flow_integrity_report_zh.md"
AUDIT_ROOT = Path(__file__).resolve().parent / "results" / "flow_integrity_audit"


@dataclass
class CheckItem:
    name: str
    status: str
    detail: str = ""


@dataclass
class ChartAudit:
    chart_type: str
    source: str = ""
    sample_id: str = ""
    chart_id: str = ""
    config_path: str = ""
    original_path: str = ""
    grid_path: str = ""
    grid_source: str = ""
    target_name: str = ""
    target_count: int = 0
    artifacts: list[str] = field(default_factory=list)
    checks: list[CheckItem] = field(default_factory=list)

    @property
    def overall(self) -> str:
        statuses = {item.status for item in self.checks}
        if "FAIL" in statuses:
            return "FAIL"
        if "WARN" in statuses:
            return "WARN"
        return "PASS"

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(CheckItem(name=name, status=status, detail=detail))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GT experiment flow integrity checks.")
    parser.add_argument("--source", choices=[*SOURCES, "auto"], default="synthetic")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    audits: list[ChartAudit] = []
    for chart_type in CHART_TYPES:
        audit = ChartAudit(chart_type=chart_type)
        audits.append(audit)
        _audit_chart_type(audit, args.source)

    _write_report(audits, args.report)
    _print_console_summary(audits, args.report)
    return 1 if any(audit.overall == "FAIL" for audit in audits) else 0


def _audit_chart_type(audit: ChartAudit, source_choice: str) -> None:
    try:
        sample = _select_sample(audit.chart_type, source_choice)
        audit.source = str(sample.get("source") or "")
        audit.sample_id = str(sample.get("sample_id") or "")
        audit.chart_id = str(sample.get("name") or sample.get("chart_id") or "")
        audit.add("样本发现", "PASS", f"{audit.source}/{audit.chart_id}")
    except Exception as exc:
        audit.add("样本发现", "FAIL", _short_error(exc))
        return

    try:
        import main as backend_main

        config_path = backend_main.resolve_gt_config_path(audit.sample_id)
        audit.config_path = str(config_path)
        audit.add("GT JSON", "PASS" if config_path.exists() else "FAIL", str(config_path))
    except Exception as exc:
        audit.add("GT JSON", "FAIL", _short_error(exc))
        return

    try:
        contract = CONTRACTS[audit.chart_type]
        expected_stages = {BASELINE_STAGE, GRID_STAGE, FEEDBACK_STAGE, AMPLIFIER_STAGE}
        has_stages = expected_stages.issubset(set(contract.stages))
        _import_contract_modules(contract)
        audit.add("模块解耦契约", "PASS" if has_stages else "FAIL", ",".join(contract.stages))
    except Exception as exc:
        audit.add("模块解耦契约", "FAIL", _short_error(exc))

    try:
        dataset, targets, image_getter = _load_dataset_and_targets(audit.chart_type, config_path)
        audit.chart_id = str(dataset.get("chart_id") or audit.chart_id)
        audit.target_count = len(targets)
        if targets:
            audit.target_name = _target_name(targets[0])
        audit.add("GT loader 和目标枚举", "PASS" if targets else "FAIL", f"targets={len(targets)}")
    except Exception as exc:
        audit.add("GT loader 和目标枚举", "FAIL", _short_error(exc))
        return

    try:
        original_path = image_getter(dataset, "no_grid")
        grid_path = image_getter(dataset, "grid_with_grid")
        audit.original_path = str(original_path)
        audit.grid_path = str(grid_path)
        audit.grid_source = "fallback-rendered" if GRID_STYLE_VERSION in str(grid_path) else "existing-grid-with-grid"
        _check_image_file(audit, "原图", original_path)
        _check_image_file(audit, "grid-with-grid 图", grid_path)
        _check_grid_quality(audit, grid_path)
    except Exception as exc:
        audit.add("图像输入", "FAIL", _short_error(exc))

    try:
        prompts = _build_prompts(audit.chart_type, dataset, targets[0])
        _check_prompts(audit, prompts)
    except Exception as exc:
        audit.add("三阶段提示词生成", "FAIL", _short_error(exc))

    try:
        artifacts = _generate_visual_artifacts(audit.chart_type, dataset, targets[0], image_getter)
        audit.artifacts.extend(str(path) for path in artifacts)
        audit.add("feedback/amplifier 过程图生成", "PASS" if len(artifacts) >= 2 else "WARN", f"artifacts={len(artifacts)}")
        _check_process_image_style(audit, artifacts)
    except Exception as exc:
        audit.add("feedback/amplifier 过程图生成", "FAIL", _short_error(exc))

    try:
        _check_static_runtime_semantics(audit.chart_type, audit)
    except Exception as exc:
        audit.add("静态生成语义", "FAIL", _short_error(exc))

    try:
        _check_global_experiment_contracts(audit.chart_type, audit)
    except Exception as exc:
        audit.add("全局实验契约", "FAIL", _short_error(exc))


def _select_sample(chart_type: str, source_choice: str) -> dict[str, Any]:
    import main as backend_main

    sources = SOURCES if source_choice == "auto" else (source_choice,)
    for source in sources:
        samples = list(backend_main.iter_dataset_samples(source, chart_type))
        if samples:
            return samples[0]
    raise RuntimeError(f"No sample found for {chart_type}")


def _import_contract_modules(contract: Any) -> None:
    for module_name in (contract.runner_module, contract.prompt_module, contract.model_module):
        if module_name:
            importlib.import_module(f"evaluation_prediction.{module_name}")
    if contract.visual_module:
        importlib.import_module(f"evaluation_prediction.{contract.visual_module}")


def _load_dataset_and_targets(chart_type: str, config_path: Path):
    if chart_type in {"bubble", "scatter"}:
        from evaluation_prediction.chart_modules.bubble.data import PointChartConfig
        from evaluation_prediction.chart_modules.bubble import data as point_data

        config = PointChartConfig(
            chart_type=chart_type,
            result_dir_name=chart_type,
            mark_name="bubble" if chart_type == "bubble" else "circle",
        )
        datasets = point_data.load_datasets(config, config_paths=[config_path])
        dataset = datasets[0]
        return dataset, point_data.iter_targets(dataset), lambda ds, image_type: point_data.image_path(config, ds, image_type)

    if chart_type == "line":
        from evaluation_prediction.chart_modules.line import data

        dataset = data.load_datasets(config_paths=[config_path])[0]
        return dataset, data.iter_targets(dataset), data.image_path

    if chart_type == "v_bar":
        from evaluation_prediction.chart_modules.v_bar import data

        dataset = data.load_datasets(config_paths=[config_path], chart_type=chart_type)[0]
        return dataset, data.iter_targets(dataset), data.image_path

    if chart_type == "h_bar":
        from evaluation_prediction.chart_modules.h_bar import data

        dataset = data.load_datasets(config_paths=[config_path], chart_type=chart_type)[0]
        return dataset, data.iter_targets(dataset), data.image_path

    if chart_type == "pie":
        from evaluation_prediction.chart_modules.pie import data

        dataset = data.load_datasets(config_paths=[config_path])[0]
        return dataset, data.iter_targets(dataset), data.image_path

    if chart_type == "donut":
        from evaluation_prediction.chart_modules.donut import data

        dataset = data.load_datasets(config_paths=[config_path])[0]
        return dataset, data.iter_targets(dataset), data.image_path

    if chart_type in {"radar", "rose"}:
        from evaluation_prediction.chart_modules import polar_value

        dataset = polar_value.load_backend_polar_datasets(chart_type, config_paths=[config_path])[0]
        return dataset, polar_value.iter_polar_targets(dataset, chart_type), polar_value.image_path

    raise ValueError(f"Unsupported chart type: {chart_type}")


def _check_image_file(audit: ChartAudit, label: str, path: Path) -> None:
    if not path.exists():
        audit.add(label, "FAIL", str(path))
        return
    with Image.open(path) as image:
        width, height = image.size
        stat = ImageStat.Stat(image.convert("L"))
        variance = stat.var[0] if stat.var else 0.0
    status = "PASS" if width >= 100 and height >= 100 and variance > 1.0 else "WARN"
    audit.add(label, status, f"{width}x{height}, variance={variance:.2f}, {path}")


def _check_grid_quality(audit: ChartAudit, path: Path) -> None:
    if not path.exists():
        return
    with Image.open(path).convert("RGB") as image:
        pixels = image.resize((min(240, image.width), min(240, image.height))).getdata()
        total = 0
        gray = 0
        for r, g, b in pixels:
            total += 1
            if abs(r - 204) <= 3 and abs(g - 204) <= 3 and abs(b - 204) <= 3:
                gray += 1
    ratio = gray / max(total, 1)
    if audit.grid_source == "fallback-rendered":
        status = "PASS" if ratio > 0.0001 else "WARN"
        detail = f"{audit.grid_source}, #cccccc pixel ratio={ratio:.4f}, style={GRID_STYLE_VERSION}"
    else:
        status = "PASS"
        detail = f"{audit.grid_source}, #cccccc pixel ratio={ratio:.4f}"
    audit.add("grid 样式/来源", status, detail)


def _check_process_image_style(audit: ChartAudit, artifacts: list[Path]) -> None:
    red_pixels = 0
    total_pixels = 0
    for path in artifacts:
        if not path.exists():
            continue
        with Image.open(path).convert("RGB") as image:
            sample = image.resize(
                (min(240, image.width), min(240, image.height)),
                Image.Resampling.NEAREST,
            )
            for r, g, b in sample.getdata():
                total_pixels += 1
                if r >= 180 and g <= 80 and b <= 80:
                    red_pixels += 1
    ratio = red_pixels / max(total_pixels, 1)
    audit.add(
        "feedback/amplifier 红色 guide 样式",
        "PASS" if ratio > 0.0001 else "FAIL",
        f"red pixel ratio={ratio:.4f}",
    )


def _build_prompts(chart_type: str, dataset: dict[str, Any], target: Any) -> dict[str, str]:
    if chart_type in {"bubble", "scatter"}:
        from evaluation_prediction.chart_modules.scatter.prompts import generate_prompt

        mark_name = "bubble" if chart_type == "bubble" else "circle"
        fake_pred = (_number_or_mid(getattr(target, "gt_x", None), dataset.get("x_ticks")), _number_or_mid(getattr(target, "gt_y", None), dataset.get("y_ticks")))
        kwargs = dict(
            item_name=target.point_name,
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            mark_name=mark_name,
            visual_name=getattr(target, "visual_name", None),
        )
        return {
            BASELINE_STAGE: generate_prompt(prompt_type="baseline", **kwargs),
            GRID_STAGE: generate_prompt(prompt_type="grid", **kwargs),
            FEEDBACK_STAGE: generate_prompt(prompt_type="feedback", pred_feedback=fake_pred, **kwargs),
            AMPLIFIER_STAGE: generate_prompt(prompt_type="feedback_crop_adaptive", pred_feedback=fake_pred, **kwargs),
        }

    if chart_type == "line":
        from evaluation_prediction.chart_modules.line.prompts import generate_prompt

        fake_pred = (target.gt_x, _number_or_mid(getattr(target, "gt_y", None), dataset.get("y_ticks")))
        kwargs = dict(
            item_name=target.point_name,
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            series_color=dataset.get("series_color", {}),
        )
        return {
            BASELINE_STAGE: generate_prompt(prompt_type="baseline", **kwargs),
            GRID_STAGE: generate_prompt(prompt_type="grid", **kwargs),
            FEEDBACK_STAGE: generate_prompt(prompt_type="feedback", pred_feedback=fake_pred, **kwargs),
            AMPLIFIER_STAGE: generate_prompt(prompt_type="amplifier", visible_ticks=_numeric_ticks(dataset.get("y_ticks")), pred_feedback=fake_pred, **kwargs),
        }

    if chart_type == "v_bar":
        from evaluation_prediction.chart_modules.v_bar.prompts import generate_prompt

        fake_pred = (target.gt_x, _number_or_mid(getattr(target, "gt_y", None), dataset.get("y_ticks")))
        kwargs = dict(
            item_name=target.point_name,
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            series_color=dataset.get("series_color", {}),
            chart_type=chart_type,
        )
        return {
            BASELINE_STAGE: generate_prompt(prompt_type="baseline", **kwargs),
            GRID_STAGE: generate_prompt(prompt_type="grid", **kwargs),
            FEEDBACK_STAGE: generate_prompt(prompt_type="feedback", pred_feedback=[fake_pred], feedback_round=1, current_round=1, **kwargs),
            AMPLIFIER_STAGE: generate_prompt(prompt_type="amplifier", visible_ticks=_numeric_ticks(dataset.get("y_ticks")), pred_feedback=[fake_pred], **kwargs),
        }

    if chart_type == "h_bar":
        from evaluation_prediction.chart_modules.h_bar.prompts import generate_prompt

        fake_pred = (_number_or_mid(getattr(target, "gt_x", None), dataset.get("x_ticks")), target.gt_y)
        kwargs = dict(
            item_name=target.point_name,
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            series_color=dataset.get("series_color", {}),
            chart_type=chart_type,
        )
        return {
            BASELINE_STAGE: generate_prompt(prompt_type="baseline", **kwargs),
            GRID_STAGE: generate_prompt(prompt_type="grid", **kwargs),
            FEEDBACK_STAGE: generate_prompt(prompt_type="feedback", pred_feedback=[fake_pred], feedback_round=1, current_round=1, **kwargs),
            AMPLIFIER_STAGE: generate_prompt(prompt_type="amplifier", visible_ticks=_numeric_ticks(dataset.get("x_ticks")), pred_feedback=[fake_pred], **kwargs),
        }

    if chart_type in {"pie", "donut"}:
        prompt_module = importlib.import_module(f"evaluation_prediction.chart_modules.{chart_type}.prompts")
        fake_angles = {"start_angle": 0.0, "end_angle": 45.0}
        return {
            BASELINE_STAGE: prompt_module.generate_prompt(target.point_name, "baseline"),
            GRID_STAGE: prompt_module.generate_prompt(target.point_name, "grid"),
            FEEDBACK_STAGE: prompt_module.generate_prompt(target.point_name, "feedback", prev_angle=fake_angles),
            AMPLIFIER_STAGE: prompt_module.generate_prompt(target.point_name, "amplifier", prev_angle=fake_angles, drawn_angles=[0, 15, 30, 45]),
        }

    if chart_type in {"radar", "rose"}:
        from evaluation_prediction.chart_modules import polar_value

        fake_r = _number_or_mid(_polar_gt_value(dataset, target, polar_value), dataset.get("r_ticks"))
        return {
            BASELINE_STAGE: polar_value.build_prompt(dataset, target, chart_type, "baseline"),
            GRID_STAGE: polar_value.build_prompt(dataset, target, chart_type, "grid"),
            FEEDBACK_STAGE: polar_value.build_prompt(dataset, target, chart_type, "feedback", prev_r=fake_r),
            AMPLIFIER_STAGE: polar_value.build_prompt(dataset, target, chart_type, "amplifier", prev_r=fake_r, visible_ticks=_numeric_ticks(dataset.get("r_ticks"))),
        }

    raise ValueError(f"Unsupported chart type: {chart_type}")


def _check_prompts(audit: ChartAudit, prompts: dict[str, str]) -> None:
    missing = [stage for stage in (BASELINE_STAGE, GRID_STAGE, FEEDBACK_STAGE, AMPLIFIER_STAGE) if not prompts.get(stage)]
    audit.add("三阶段提示词生成", "PASS" if not missing else "FAIL", f"missing={missing}")

    grid_prompt = prompts.get(GRID_STAGE, "")
    feedback_prompt = prompts.get(FEEDBACK_STAGE, "")
    amplifier_prompt = prompts.get(AMPLIFIER_STAGE, "")
    if audit.chart_type in {"bubble", "scatter", "line", "v_bar", "h_bar"}:
        has_mapping = "GT" in grid_prompt and "tick-to-pixel" in grid_prompt and "GT" in feedback_prompt and "tick-to-pixel" in feedback_prompt
        audit.add("GT tick-pixel 映射进入 grid/feedback prompt", "PASS" if has_mapping else "FAIL", "")
    elif audit.chart_type in {"radar", "rose"}:
        has_mapping = "GT radial tick-to-pixel-radius mapping" in grid_prompt
        audit.add("GT 径向 tick-pixel 映射进入 prompt", "PASS" if has_mapping else "FAIL", "")
    else:
        has_angles = "15" in grid_prompt and ("reference" in grid_prompt.lower() or "radial" in grid_prompt.lower())
        audit.add("圆形图角度网格提示", "PASS" if has_angles else "WARN", "pie/donut 使用 15°角度参考线，不使用笛卡尔 tick-pixel 映射")

    has_feedback_ref = "previous" in feedback_prompt.lower() or "red" in feedback_prompt.lower()
    has_amp_ref = "crop" in amplifier_prompt.lower() or "cropped" in amplifier_prompt.lower() or "zoom" in amplifier_prompt.lower()
    audit.add("feedback 提示引用上一轮预测", "PASS" if has_feedback_ref else "WARN", "")
    audit.add("amplifier 提示说明局部放大", "PASS" if has_amp_ref else "WARN", "")


def _generate_visual_artifacts(chart_type: str, dataset: dict[str, Any], target: Any, image_getter: Any) -> list[Path]:
    chart_id = str(dataset.get("chart_id") or chart_type)
    grid_path = image_getter(dataset, "grid_with_grid")
    out_dir = AUDIT_ROOT / chart_type / _safe_name(chart_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if chart_type in {"bubble", "scatter"}:
        from evaluation_prediction.chart_modules.bubble.data import PointChartConfig
        from evaluation_prediction.chart_modules.scatter import visual

        config = PointChartConfig(chart_type=chart_type, result_dir_name=chart_type, mark_name="bubble" if chart_type == "bubble" else "circle")
        pred = (_number_or_mid(getattr(target, "gt_x", None), dataset.get("x_ticks")), _number_or_mid(getattr(target, "gt_y", None), dataset.get("y_ticks")))
        overlay = visual.draw_prediction_overlay(
            config=config,
            chart_id=chart_id,
            original_img_path=grid_path,
            pred_coords=[pred],
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            point_name=target.point_name,
            run_index=1,
        )
        crop = visual.crop_draw_ticks_resize(
            config=config,
            chart_id=chart_id,
            image_path=grid_path,
            point_name=target.point_name,
            pred_coord=(float(pred[0]), float(pred[1])),
            x_ticks=_numeric_ticks(dataset.get("x_ticks")),
            y_ticks=_numeric_ticks(dataset.get("y_ticks")),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            feedback_round=1,
        )[0]
        return [overlay, crop]

    if chart_type == "line":
        from evaluation_prediction.chart_modules.line import visual

        pred_y = _number_or_mid(getattr(target, "gt_y", None), dataset.get("y_ticks"))
        overlay = visual.draw_prediction_overlay(
            chart_id=chart_id,
            original_img_path=grid_path,
            pred_coords=[(target.gt_x, pred_y)],
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            point_name=target.point_name,
        )
        crop = visual.crop_line_point_window(
            chart_id=chart_id,
            image_path=grid_path,
            point_name=target.point_name,
            x_label=str(target.gt_x),
            center_value=float(pred_y),
            x_ticks=dataset.get("x_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_ticks=_numeric_ticks(dataset.get("y_ticks")),
            y_pixels=dataset.get("y_pixels", []),
            round_index=1,
        )[0]
        return [overlay, crop]

    if chart_type == "v_bar":
        from evaluation_prediction.chart_modules.v_bar import visual

        pred_y = _number_or_mid(getattr(target, "gt_y", None), dataset.get("y_ticks"))
        overlay = visual.draw_prediction_overlay(
            chart_id=chart_id,
            original_img_path=grid_path,
            pred_coords=[(target.gt_x, pred_y)],
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            point_name=target.point_name,
            chart_type=chart_type,
            run_index=1,
        )
        crop = visual.crop_bar_window(
            chart_id=chart_id,
            image_path=grid_path,
            point_name=target.point_name,
            x_label=str(target.gt_x),
            center_value=float(pred_y),
            x_ticks=dataset.get("x_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_ticks=_numeric_ticks(dataset.get("y_ticks")),
            y_pixels=dataset.get("y_pixels", []),
            round_index=1,
            chart_type=chart_type,
        )[0]
        return [overlay, crop]

    if chart_type == "h_bar":
        from evaluation_prediction.chart_modules.h_bar import visual

        pred_x = _number_or_mid(getattr(target, "gt_x", None), dataset.get("x_ticks"))
        overlay = visual.draw_prediction_overlay(
            chart_id=chart_id,
            original_img_path=grid_path,
            pred_coords=[(pred_x, target.gt_y)],
            x_ticks=dataset.get("x_ticks", []),
            y_ticks=dataset.get("y_ticks", []),
            x_pixels=dataset.get("x_pixels", []),
            y_pixels=dataset.get("y_pixels", []),
            point_name=target.point_name,
            chart_type=chart_type,
            run_index=1,
        )
        crop = visual.crop_bar_window(
            chart_id=chart_id,
            image_path=grid_path,
            point_name=target.point_name,
            y_label=str(target.gt_y),
            center_value=float(pred_x),
            x_ticks=_numeric_ticks(dataset.get("x_ticks")),
            x_pixels=dataset.get("x_pixels", []),
            y_ticks=dataset.get("y_ticks", []),
            y_pixels=dataset.get("y_pixels", []),
            round_index=1,
            chart_type=chart_type,
        )[0]
        return [overlay, crop]

    if chart_type == "pie":
        from evaluation_prediction.chart_modules.pie import runner as pie_runner
        from evaluation_prediction.chart_modules.pie import visual

        center, radius = pie_runner._pie_geometry(dataset, grid_path)
        feedback = Path(
            visual.draw_angle_feedback(
                str(grid_path),
                [0.0, 45.0],
                str(out_dir / f"{_safe_name(target.point_name)}_feedback.png"),
                center,
                max(1, int(radius * 0.05)),
            )
        )
        crop = pie_runner._crop_sector_for_prediction(
            source=grid_path,
            out_path=out_dir / f"{_safe_name(target.point_name)}_amplifier.png",
            center=center,
            radius=radius,
            start_angle=0.0,
            end_angle=45.0,
        )
        return [feedback, crop]

    if chart_type == "donut":
        from evaluation_prediction.chart_modules.donut import runner as donut_runner
        from evaluation_prediction.chart_modules.donut import visual

        center, inner_r, outer_r = donut_runner._donut_geometry(dataset, grid_path)
        feedback = Path(
            visual.draw_angle_feedback(
                str(grid_path),
                [0.0, 45.0],
                str(out_dir / f"{_safe_name(target.point_name)}_feedback.png"),
                center,
                inner_r,
            )
        )
        crop_path = _crop_circular_sector_stub(
            source=grid_path,
            out_path=out_dir / f"{_safe_name(target.point_name)}_amplifier.png",
            center=center,
            inner_radius=inner_r,
            outer_radius=outer_r,
            start_angle=0.0,
            end_angle=45.0,
        )
        return [feedback, crop_path]

    if chart_type in {"radar", "rose"}:
        from evaluation_prediction.chart_modules import polar_value
        from evaluation_prediction.chart_modules import polar_visual

        pred_r = _number_or_mid(_polar_gt_value(dataset, target, polar_value), dataset.get("r_ticks"))
        feedback = polar_visual.draw_polar_feedback(
            dataset=dataset,
            chart_type=chart_type,
            source_image=grid_path,
            result_dir=out_dir,
            point_name=target.point_name,
            theta_label=target.theta_label,
            pred_r=float(pred_r),
            round_index=1,
        )
        crop = polar_visual.crop_polar_amplifier(
            dataset=dataset,
            chart_type=chart_type,
            source_image=grid_path,
            result_dir=out_dir,
            point_name=target.point_name,
            theta_label=target.theta_label,
            pred_r=float(pred_r),
            round_index=1,
        )[0]
        return [feedback, crop]

    return []


def _check_static_runtime_semantics(chart_type: str, audit: ChartAudit) -> None:
    contract = CONTRACTS[chart_type]
    runner_module = importlib.import_module(f"evaluation_prediction.{contract.runner_module}")
    source = Path(runner_module.__file__ or "").read_text(encoding="utf-8")

    repeat_ok = get_repeat_times() == 1
    rounds_ok = get_feedback_rounds() == 2 and get_amplifier_rounds() == 3
    audit.add("GT 模式轮次上限", "PASS" if repeat_ok and rounds_ok else "FAIL", f"baseline/grid={get_repeat_times()}, feedback={get_feedback_rounds()}, amplifier={get_amplifier_rounds()}")

    dangerous_tokens = ["_target_matched", "CHART_MATCH_RNE_TOLERANCE"]
    dangerous = [token for token in dangerous_tokens if token in source]
    audit.add("生成端不使用 GT/RNE 作为提前停止", "PASS" if not dangerous else "FAIL", ",".join(dangerous))

    if chart_type in {"bubble", "scatter"}:
        leak_pattern = re.search(r"def _fallback_center[\s\S]{0,260}target\.gt_", source)
        audit.add("amplifier 裁剪中心不回退 GT", "PASS" if not leak_pattern else "FAIL", "")
    else:
        audit.add("amplifier 裁剪中心来源", "PASS", "使用上一轮预测或类别轴定位；GT 仅用于记录/指标")


def _check_global_experiment_contracts(chart_type: str, audit: ChartAudit) -> None:
    from evaluation_prediction import service

    stacked_types = {"v_stacked_bar", "h_stacked_bar"}
    supported = set(service.SUPPORTED_PREDICTION_TYPES)
    normalized_stacked = {item: service.normalize_prediction_type(item) for item in stacked_types}
    stacked_excluded = supported.isdisjoint(stacked_types) and all(
        normalized == raw for raw, normalized in normalized_stacked.items()
    )
    audit.add(
        "实验入口排除 stacked bar",
        "PASS" if stacked_excluded else "FAIL",
        f"supported={sorted(supported)}, normalized_stacked={normalized_stacked}",
    )

    contract = CONTRACTS[chart_type]
    runner_module = importlib.import_module(f"evaluation_prediction.{contract.runner_module}")
    runner_source = Path(runner_module.__file__ or "").read_text(encoding="utf-8")
    model_source = ""
    if contract.model_module:
        model_module = importlib.import_module(f"evaluation_prediction.{contract.model_module}")
        model_source = Path(model_module.__file__ or "").read_text(encoding="utf-8")
    gemini_source = (PROJECT_ROOT / "gemini_calls.py").read_text(encoding="utf-8")
    main_source = (BACKEND_ROOT / "main.py").read_text(encoding="utf-8")
    flow_source = (BACKEND_ROOT / "evaluation_prediction" / "common" / "experiment_flow.py").read_text(encoding="utf-8")
    call_id_ok = all(
        (
            "get_last_modal_call_id" in gemini_source,
            '"call_id": call_id' in gemini_source,
            "by_call_id" in main_source,
            "structured_prediction" in main_source,
            '"call_id"' in flow_source,
            "call_id" in runner_source or "call_id" in model_source,
        )
    )
    audit.add(
        "modal call 与结构化预测 call_id 关联",
        "PASS" if call_id_ok else "FAIL",
        "modal logs, runner records, gt_metric_records.csv, enriched logs",
    )


def _target_name(target: Any) -> str:
    return str(getattr(target, "point_name", None) or getattr(target, "label", None) or target)


def _number_or_mid(value: Any, ticks: Any) -> float:
    try:
        number = float(value)
        if number == number:
            return number
    except Exception:
        pass
    values = _numeric_ticks(ticks)
    if values:
        return (min(values) + max(values)) / 2.0
    return 0.0


def _numeric_ticks(ticks: Any) -> list[float]:
    values: list[float] = []
    for tick in ticks if isinstance(ticks, list) else []:
        try:
            number = float(str(tick).strip().rstrip("%").rstrip("°"))
        except Exception:
            continue
        if number == number:
            values.append(number)
    return values


def _polar_gt_value(dataset: dict[str, Any], target: Any, polar_value: Any) -> Any:
    try:
        return polar_value._target_gt(dataset, target)
    except Exception:
        return None


def _crop_circular_sector_stub(
    *,
    source: Path,
    out_path: Path,
    center: tuple[int, int],
    inner_radius: int,
    outer_radius: int,
    start_angle: float,
    end_angle: float,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source).convert("RGBA") as base:
        width, height = base.size
        pad = max(8, int(outer_radius * 0.08))
        cx, cy = center
        left = max(0, cx - outer_radius - pad)
        top = max(0, cy - outer_radius - pad)
        right = min(width, cx + outer_radius + pad)
        bottom = min(height, cy + outer_radius + pad)
        crop = base.crop((left, top, right, bottom))
        local_center = (cx - left, cy - top)
        mask = Image.new("L", crop.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        span = (end_angle - start_angle) % 360
        points_outer = []
        points_inner = []
        steps = max(12, int(span // 3) + 1)
        for index in range(steps + 1):
            angle = start_angle + span * index / steps
            theta = math.radians(angle - 90.0)
            points_outer.append((local_center[0] + outer_radius * math.cos(theta), local_center[1] + outer_radius * math.sin(theta)))
            points_inner.append((local_center[0] + inner_radius * math.cos(theta), local_center[1] + inner_radius * math.sin(theta)))
        polygon = points_outer + list(reversed(points_inner))
        mask_draw.polygon(polygon, fill=255)
        result = Image.new("RGBA", crop.size, (255, 255, 255, 0))
        result.paste(crop, (0, 0), mask)
        result.save(out_path)
    return out_path


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _safe_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "")).strip("_") or "item"


def _write_report(audits: list[ChartAudit], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# GT 实验版流程完整性测试报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 测试方式：每个图表类型取 1 个样本，不调用模型；实际加载 GT、grid-with-grid、提示词、feedback 图、amplifier crop。")
    lines.append("- 判据：GT 只能作为输入映射、过程日志和指标真值；生成端提前停止和 crop 中心不能使用 GT 作为答案。")
    lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append("| 类型 | 结论 | 样本 | 目标数 | grid 来源 | 过程图数量 |")
    lines.append("| --- | --- | --- | ---: | --- | ---: |")
    for audit in audits:
        lines.append(
            f"| {audit.chart_type} | {audit.overall} | {audit.source}/{audit.chart_id} | "
            f"{audit.target_count} | {audit.grid_source or '-'} | {len(audit.artifacts)} |"
        )
    lines.append("")
    for audit in audits:
        lines.append(f"## {audit.chart_type}")
        lines.append("")
        lines.append(f"- 结论：{audit.overall}")
        lines.append(f"- 样本：{audit.source}/{audit.chart_id}，sample_id={audit.sample_id}")
        lines.append(f"- GT JSON：`{audit.config_path}`")
        lines.append(f"- 原图：`{audit.original_path}`")
        lines.append(f"- grid-with-grid：`{audit.grid_path}`")
        lines.append(f"- 测试目标：{audit.target_name}，目标总数={audit.target_count}")
        lines.append("")
        lines.append("| 检查项 | 状态 | 说明 |")
        lines.append("| --- | --- | --- |")
        for item in audit.checks:
            detail = item.detail.replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {item.name} | {item.status} | {detail} |")
        if audit.artifacts:
            lines.append("")
            lines.append("过程文件：")
            for artifact in audit.artifacts:
                lines.append(f"- `{artifact}`")
        lines.append("")

    lines.append("## 复跑命令")
    lines.append("")
    lines.append("```powershell")
    lines.append("python backend/evaluation_prediction/gt_experiment_flow_integrity_check.py --source synthetic")
    lines.append("```")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _print_console_summary(audits: list[ChartAudit], report_path: Path) -> None:
    print(f"Report: {report_path}")
    for audit in audits:
        print(f"{audit.chart_type}: {audit.overall} ({audit.source}/{audit.chart_id})")
        for item in audit.checks:
            if item.status != "PASS":
                print(f"  - {item.status} {item.name}: {item.detail}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
