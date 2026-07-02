import asyncio
import io
import json
import os
import shutil
import sys
import uuid
import csv
import hashlib
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


def configure_windows_encoding() -> None:
    if sys.platform != "win32":
        return

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"


configure_windows_encoding()

BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from type_detection.chart_processor import ChartProcessorFactory  # noqa: E402
from type_detection.chart_registry import (  # noqa: E402
    DEFAULT_CHART_TYPE,
    SUPPORTED_CHART_TYPES,
    get_coordinate_system,
    normalize_chart_type,
)
from type_detection.chart_type import ChartTypeDetector  # noqa: E402
from evaluation_prediction.service import SUPPORTED_PREDICTION_TYPES, run_prediction_async  # noqa: E402
from evaluation_prediction.common.experiment_flow import (  # noqa: E402
    normalize_stage,
    summarize_stage_coverage,
    write_metric_artifacts,
)
from evaluation_prediction.common.experiment_contract import CONTRACTS  # noqa: E402
from evaluation_prediction.common.gt_grid_renderer import render_gt_grid_image  # noqa: E402
from evaluation_prediction.common.json_safety import sanitize_json_value  # noqa: E402
from evaluation_prediction.common.model_config import get_model_name  # noqa: E402
from evaluation_prediction.common.model_vision_registry import (  # noqa: E402
    active_model_vision_profile,
    estimate_image_tokens,
    profile_for_model,
)
from demo_radar.color import RadarColorMatcher  # noqa: E402


app = FastAPI(title="Chart Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GT_EXPERIMENT_RUN_LOCK = asyncio.Lock()

UPLOAD_DIR = BACKEND_DIR / "data" / "upload"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
RESULTS_DIR = BACKEND_DIR / "data" / "results"
OUTPUT_DIR = BACKEND_DIR / "data" / "output"
DATASET_PREVIEW_CACHE_DIR = BACKEND_DIR / "data" / "dataset_preview_cache"

for directory in [UPLOAD_DIR, PROCESSED_DIR, RESULTS_DIR, OUTPUT_DIR, DATASET_PREVIEW_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

charts_db: Dict[str, Dict[str, Any]] = {}


def safe_error_message(error: Exception) -> str:
    try:
        return str(error).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        return "Unexpected server error"


def safe_path_fragment(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[\\/:*?"<>|]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("._ ") or "chart"


def model_path_alias(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    aliases = {
        "gemini-2.5-flash-nothinking": "g25fnt",
        "gemini-2.5-flash-lite": "g25fl",
        "gemini-2.5-flash": "g25f",
        "gpt-4o": "gpt4o",
        "gpt-4.1": "gpt41",
        "claude-haiku-4.5": "haiku45",
        "intern vl3-78b": "internvl3_78b",
        "pixtral-12b-2409": "pixtral12b2409",
    }
    if normalized in aliases:
        return aliases[normalized]
    compact = safe_path_fragment(normalized).replace("-", "")
    return compact[:32] or "model"


def ensure_legacy_flow_enabled() -> None:
    enabled = os.getenv("CHART_EXPERIMENT_ENABLE_LEGACY", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        raise HTTPException(
            status_code=410,
            detail="Legacy upload/process/cache flow is disabled in the GT experiment branch.",
        )


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {"data": data}


def write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(sanitize_json_value(data), file, ensure_ascii=False, indent=indent, allow_nan=False)


def save_upload_file(upload: UploadFile, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)


def detect_chart_type(image_path: Path) -> Dict[str, Any]:
    try:
        detection = ChartTypeDetector().detect_chart_type(str(image_path))
        if not isinstance(detection, dict):
            raise ValueError("Chart type detector returned an empty result")
        chart_type = detection.get("type")
        if chart_type not in SUPPORTED_CHART_TYPES:
            raise ValueError(f"Unsupported or missing chart type from model: {chart_type!r}")
        return detection
    except Exception as error:
        message = safe_error_message(error)
        print(f"Chart type detection failed: {message}")
        raise HTTPException(status_code=400, detail=f"Chart type detection failed: {message}")


def infer_bar_orientation_from_image(image_path: Path) -> Optional[str]:
    """Infer bar orientation from colored data marks, without using GT."""
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 35, 35]), np.array([179, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = max(40, width * height * 0.00025)
        vertical_score = 0.0
        horizontal_score = 0.0
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = box_width * box_height
            if area < min_area:
                continue
            if x > width * 0.82:
                continue
            if y < height * 0.08:
                continue
            if box_height >= max(10, box_width * 1.15):
                vertical_score += area
            if box_width >= max(10, box_height * 1.35):
                horizontal_score += area

        if vertical_score > horizontal_score * 1.6 and vertical_score > 0:
            return "v_bar"
        if horizontal_score > vertical_score * 1.6 and horizontal_score > 0:
            return "h_bar"
    except Exception as error:
        print(f"Bar geometry orientation inference failed: {safe_error_message(error)}")
    return None


DATASET_SOURCE_ROOTS = {
    "realworld": BACKEND_DIR / "datasets" / "VisHintPrompt_datasets" / "Final-RealDataset",
    "synthetic": BACKEND_DIR / "datasets" / "VisHintPrompt_datasets" / "Sy.Dataset",
    "experiment_gt_real": PROJECT_ROOT / "experiment_gt_dataset" / "organized" / "Final-RealDataset",
    "experiment_gt_synthetic": PROJECT_ROOT / "experiment_gt_dataset" / "organized" / "Sy.Dataset",
}
DATASET_PREVIEW_BATCH_OUTPUT_DIR = (
    BACKEND_DIR / "evaluation" / "recheck_outputs" / "vishintprompt_full_grid_encryption_latest"
)
DATASET_PREVIEW_LABEL_STYLE_VERSION = "ocr_box_position_white_bg_axis_all_or_none_v1"
DATASET_CATEGORY_LABELS = {
    "bubble": "Bubble",
    "scatter": "Scatter",
    "line": "Line",
    "v_bar": "Vertical Bar",
    "h_bar": "Horizontal Bar",
    "pie": "Pie",
    "donut": "Donut",
    "radar": "Radar",
    "rose": "Rose",
}
DATASET_CATEGORY_PREFIXES = (
    "Bubble_",
    "Donut_",
    "hBar_",
    "Line_",
    "Pie_",
    "Radar_",
    "Rose_",
    "Scatter_",
    "vBar_",
)
DATASET_CATEGORY_PRIORITY = {
    "bubble": 0,
    "scatter": 1,
    "line": 2,
    "v_bar": 3,
    "h_bar": 4,
    "pie": 5,
    "donut": 6,
    "radar": 7,
    "rose": 8,
}
DATASET_PREVIEW_MANIFEST_CACHE: Dict[str, Any] = {"mtime": None, "records": {}}
LEGACY_GT_EXPERIMENT_RESULTS_DIR = BACKEND_DIR / "evaluation_prediction" / "results" / "gt_experiments"
GT_EXPERIMENT_RESULTS_DIR = PROJECT_ROOT / "gt_runs"
GT_EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def dataset_file_id(source: str, image_path: Path) -> str:
    root = DATASET_SOURCE_ROOTS[source]
    rel_path = image_path.resolve().relative_to(root.resolve()).as_posix()
    return hashlib.sha1(f"{source}:{rel_path}".encode("utf-8")).hexdigest()[:16]


def dataset_source_kind(source: str) -> str:
    return "synthetic" if source in {"synthetic", "experiment_gt_synthetic"} else "realworld"


def dataset_manifest_relative(source: str, relative_path: str) -> str:
    dataset_name = "Sy.Dataset" if dataset_source_kind(source) == "synthetic" else "Final-RealDataset"
    return f"{dataset_name}/{relative_path.replace(os.sep, '/')}"


def batch_cached_dataset_records() -> Dict[str, Dict[str, Any]]:
    manifest_path = DATASET_PREVIEW_BATCH_OUTPUT_DIR / "manifest.json"
    if not manifest_path.exists():
        DATASET_PREVIEW_MANIFEST_CACHE["mtime"] = None
        DATASET_PREVIEW_MANIFEST_CACHE["records"] = {}
        return {}
    mtime = manifest_path.stat().st_mtime
    if DATASET_PREVIEW_MANIFEST_CACHE.get("mtime") == mtime:
        return DATASET_PREVIEW_MANIFEST_CACHE.get("records", {})
    try:
        manifest = load_json(manifest_path)
    except Exception:
        return {}
    records = manifest.get("records")
    if not isinstance(records, list):
        return {}
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = record.get("dataset_relative")
        if not isinstance(key, str) or not key:
            continue
        if record.get("status") not in {"success", "skipped_success_cache", "recovered_from_grid_reference"}:
            continue
        copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
        encrypted = copied.get("encrypted_grid")
        if encrypted and Path(str(encrypted)).exists():
            indexed[key] = record
            if key.startswith("Final-RealDataset/ALL/charts/"):
                filename = Path(key).name
                realworld_root = DATASET_SOURCE_ROOTS["realworld"]
                for group_dir in sorted(realworld_root.iterdir() if realworld_root.exists() else []):
                    if not group_dir.is_dir() or group_dir.name == "ALL":
                        continue
                    if not any(group_dir.name.startswith(prefix) for prefix in DATASET_CATEGORY_PREFIXES):
                        continue
                    for pattern in ("charts", "chart"):
                        candidate = group_dir / pattern / filename
                        if candidate.exists():
                            rel = candidate.relative_to(realworld_root).as_posix()
                            indexed[f"Final-RealDataset/{rel}"] = record
    DATASET_PREVIEW_MANIFEST_CACHE["mtime"] = mtime
    DATASET_PREVIEW_MANIFEST_CACHE["records"] = indexed
    return indexed


def batch_cached_dataset_record(source: str, relative_path: str) -> Optional[Dict[str, Any]]:
    target = dataset_manifest_relative(source, relative_path)
    records = batch_cached_dataset_records()
    record = records.get(target)
    if record is not None:
        return record
    if source == "synthetic" and "Scatter_50/" in target:
        return records.get(target.replace("Scatter_50/", "Scatetr_50/"))
    return None


def dataset_evaluation_cache_path(sample_id: str) -> Path:
    return DATASET_PREVIEW_CACHE_DIR / sample_id / f"{sample_id}_evaluation.json"


def evaluation_cache_usable(path: Path, chart_type: Optional[str] = None) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False

    cache_chart_type = str(chart_type or payload.get("chart_type") or "").strip().lower()
    prediction_error = str(payload.get("prediction_runner_error") or "")
    if "unexpected keyword argument 'chart_type'" in prediction_error:
        return False
    if cache_chart_type in {"pie", "donut", "radar", "rose"}:
        processed_json = payload.get("processed_json")
        if isinstance(processed_json, dict):
            if any(key in processed_json for key in ("data", "data_points", "ground_truth", "labels")):
                return False
        predictions = payload.get("predictions")
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        chart_runs = int(summary.get("chart_runs") or 0)
        if not isinstance(predictions, list) or (not predictions and chart_runs == 0):
            return False
    return True


def seed_preview_cache_from_batch_record(record: Dict[str, Any], output_dir: Path, image_stem: str) -> tuple[Optional[str], Optional[str]]:
    copied = record.get("copied") if isinstance(record.get("copied"), dict) else {}
    encrypted_src = copied.get("encrypted_grid")
    colored_src = copied.get("colored_grid")
    basic_src = copied.get("basic_grid")
    ticks_src = copied.get("ticks_json")
    encrypted_dest = output_dir / f"{image_stem}_with_grid.png"
    colored_dest = output_dir / f"{image_stem}_with_grid_color.png"
    basic_dest = output_dir / f"{image_stem}_grid.png"
    ticks_dest = output_dir / f"{image_stem}_ticks.json"
    chart_json_dest = output_dir / f"{image_stem}.json"
    copied_encrypted = copy_existing_file(encrypted_src, encrypted_dest)
    copied_colored = copy_existing_file(colored_src, colored_dest)
    copy_existing_file(basic_src, basic_dest)
    copied_ticks = copy_existing_file(ticks_src, ticks_dest)
    sidecars = record.get("sidecars") if isinstance(record.get("sidecars"), list) else []
    for item in sidecars:
        item_path = Path(str(item))
        if item_path.name.endswith("_image.json") and item_path.exists():
            copy_existing_file(item_path, chart_json_dest)
            break
    if copied_ticks:
        try:
            ticks_data = load_json(ticks_dest)
            if copied_encrypted:
                ticks_data["encrypted_grid_path"] = copied_encrypted
            if copied_colored:
                ticks_data["colored_grid_path"] = copied_colored
            if basic_dest.exists():
                ticks_data["basic_grid_path"] = str(basic_dest)
            ticks_data["encrypted_label_style_version"] = DATASET_PREVIEW_LABEL_STYLE_VERSION
            write_json(ticks_dest, ticks_data)
        except Exception:
            pass
    return copied_encrypted, copied_colored


def preview_tick_sidecar_current(tick_sidecar: Path) -> bool:
    if not tick_sidecar.exists():
        return False
    try:
        ticks_data = load_json(tick_sidecar)
    except Exception:
        return False
    return ticks_data.get("encrypted_label_style_version") == DATASET_PREVIEW_LABEL_STYLE_VERSION


def preview_encrypted_grid_path(cache_dir: Path, image_stem: str) -> Optional[Path]:
    candidates = [
        cache_dir / f"{image_stem}_with_grid.png",
        cache_dir / f"{image_stem}_encode.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for pattern in ("*_with_grid.png", "*_encode.png"):
        match = next(cache_dir.glob(pattern), None)
        if match is not None:
            return match
    return None


def copy_existing_file(src: Any, dest: Path) -> Optional[str]:
    if not src:
        return None
    src_path = Path(str(src))
    if not src_path.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return str(dest)


def infer_dataset_chart_type(image_path: Path) -> str:
    text = image_path.as_posix().lower()
    name = image_path.stem.lower()
    if "bubble" in text:
        return "bubble"
    if "scatter" in text:
        return "scatter"
    if "line" in text:
        return "line"
    if "donut" in text:
        return "donut"
    if "pie" in text:
        return "pie"
    if "radar" in text:
        return "radar"
    if "rose" in text:
        return "rose"
    if "stacked" in text and ("xbar" in text or "hbar" in text or "horizontal" in text):
        return "h_bar"
    if "stacked" in text and "bar" in text:
        return "v_bar"
    if "xbar" in text or "hbar" in text or "horizontal" in text:
        return "h_bar"
    if "bar" in text:
        return "v_bar"
    return DEFAULT_CHART_TYPE


def dataset_chart_category(image_path: Path) -> str:
    return normalize_chart_type(infer_dataset_chart_type(image_path))


def dataset_image_paths(source: str = "realworld", category: Optional[str] = None) -> list[Path]:
    source = source if source in DATASET_SOURCE_ROOTS else "realworld"
    source_kind = dataset_source_kind(source)
    root = DATASET_SOURCE_ROOTS[source]
    if not root.exists():
        return []
    image_paths: list[Path] = []
    if source_kind == "realworld":
        seen: set[Path] = set()
        for group_dir in sorted(root.iterdir()):
            if not group_dir.is_dir() or group_dir.name == "ALL":
                continue
            if not any(group_dir.name.startswith(prefix) for prefix in DATASET_CATEGORY_PREFIXES):
                continue
            group_category = dataset_chart_category(group_dir)
            if category and category != "all" and group_category != category:
                continue
            for pattern in ("charts", "chart"):
                chart_dir = group_dir / pattern
                if not chart_dir.exists():
                    continue
                for path in chart_dir.rglob("*"):
                    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    image_paths.append(path)
    elif source_kind == "synthetic":
        seen: set[Path] = set()
        for group_dir in sorted(root.iterdir()):
            if not group_dir.is_dir():
                continue
            if not any(group_dir.name.startswith(prefix) for prefix in DATASET_CATEGORY_PREFIXES):
                continue
            group_category = dataset_chart_category(group_dir)
            if category and category != "all" and group_category != category:
                continue
            chart_dirs = [path for path in (group_dir / "charts", group_dir / "chart") if path.exists()]
            if not chart_dirs:
                chart_dirs = [group_dir]
            for chart_dir in chart_dirs:
                for path in sorted(chart_dir.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                        continue
                    resolved = path.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    image_paths.append(path)
    if not image_paths:
        image_paths = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    if category and category != "all":
        image_paths = [path for path in image_paths if dataset_chart_category(path) == category]
    return sorted(image_paths)


def dataset_category_options(source: str = "realworld") -> list[Dict[str, Any]]:
    source = source if source in DATASET_SOURCE_ROOTS else "realworld"
    counts: Dict[str, int] = {}
    for image_path in dataset_image_paths(source):
        category = dataset_chart_category(image_path)
        counts[category] = counts.get(category, 0) + 1
    return [
        {
            "value": category,
            "label": DATASET_CATEGORY_LABELS.get(category, category),
            "count": count,
        }
        for category, count in sorted(
            counts.items(),
            key=lambda item: (DATASET_CATEGORY_PRIORITY.get(item[0], 99), item[0]),
        )
    ]


def gt_experiment_category_options(source: str = "realworld") -> list[Dict[str, Any]]:
    return [
        item
        for item in dataset_category_options(source)
        if item["value"] in SUPPORTED_PREDICTION_TYPES
    ]


def iter_dataset_samples(source: str = "realworld", category: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    source = source if source in DATASET_SOURCE_ROOTS else "realworld"
    root = DATASET_SOURCE_ROOTS[source]
    image_paths = dataset_image_paths(source, category)
    image_paths = sorted(image_paths)
    samples = []
    for image_path in image_paths:
        sample_id = dataset_file_id(source, image_path)
        chart_type = dataset_chart_category(image_path)
        rel_path = image_path.resolve().relative_to(root.resolve()).as_posix()
        cache_dir = DATASET_PREVIEW_CACHE_DIR / sample_id / "output"
        tick_sidecar = cache_dir / f"{sample_id}_original_ticks.json"
        encrypted_preview = preview_encrypted_grid_path(cache_dir, f"{sample_id}_original")
        preview_cached = bool(
            encrypted_preview
            and (
                preview_tick_sidecar_current(tick_sidecar)
                or encrypted_preview.name.endswith("_encode.png")
            )
        )
        batch_cached = batch_cached_dataset_record(source, rel_path) is not None
        cached = preview_cached or batch_cached
        evaluation_cached = dataset_evaluation_cache_path(sample_id).exists()
        samples.append(
            {
                "sample_id": sample_id,
                "source": source,
                "name": image_path.stem,
                "filename": image_path.name,
                "relative_path": rel_path,
                "category": chart_type,
                "chart_type": chart_type,
                "coordinate_system": get_coordinate_system(chart_type).value,
                "image_path": str(image_path),
                "image_url": f"/api/dataset-preview/image/{sample_id}",
                "cached": cached,
                "evaluation_cached": evaluation_cached,
            }
        )
    samples.sort(key=lambda sample: (not sample["cached"], DATASET_CATEGORY_PRIORITY.get(sample["chart_type"], 99), sample["name"]))
    return samples


def dataset_sample_by_id(sample_id: str) -> Dict[str, Any]:
    for source in DATASET_SOURCE_ROOTS:
        for sample in iter_dataset_samples(source):
            if sample["sample_id"] == sample_id:
                return sample
    raise HTTPException(status_code=404, detail="Dataset sample not found")


def dataset_group_dir(sample: Dict[str, Any]) -> Path:
    image_path = Path(sample["image_path"])
    if image_path.parent.name.lower() in {"chart", "charts"}:
        return image_path.parent.parent
    return image_path.parent


def gt_config_path_for_sample(sample: Dict[str, Any]) -> Optional[Path]:
    stem = Path(sample["image_path"]).stem
    stem_variants = [stem]
    for suffix in ("_no_grid", "_grid", "_with_grid", "_grid_with_grid"):
        if stem.endswith(suffix):
            stem_variants.append(stem[: -len(suffix)])

    organized_gt = organized_gt_config_path_for_sample(sample, stem_variants)
    if organized_gt is not None:
        return organized_gt

    group_dir = dataset_group_dir(sample)
    config_dirs = [
        group_dir,
        group_dir / "chart_config",
        group_dir / "chart_configs",
        group_dir / "configs",
    ]
    candidates: list[Path] = []
    for config_dir in config_dirs:
        for variant in dict.fromkeys(stem_variants):
            candidates.extend(
                [
                    config_dir / f"{variant}_encrypted.json",
                    config_dir / f"{variant}.json",
                    config_dir / f"{variant}_image.json",
                    config_dir / f"{variant}_axes.json",
                ]
            )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for config_dir in config_dirs:
        if config_dir.exists():
            fuzzy = []
            for variant in dict.fromkeys(stem_variants):
                fuzzy.extend(sorted(config_dir.glob(f"{variant}*.json")))
            if fuzzy:
                return fuzzy[0]
    return None


def organized_gt_config_path_for_sample(sample: Dict[str, Any], stem_variants: list[str]) -> Optional[Path]:
    """Prefer the curated GT experiment dataset over legacy preview sidecars.

    Old dataset sample ids may still point at the original release dataset, but
    the experiment branch should run against experiment_gt_dataset/organized
    whenever a same-name GT config exists there.
    """
    source = str(sample.get("source") or "")
    source_roots = (
        ["experiment_gt_synthetic", "experiment_gt_real"]
        if dataset_source_kind(source) == "synthetic"
        else ["experiment_gt_real", "experiment_gt_synthetic"]
    )
    chart_type = str(sample.get("chart_type") or sample.get("category") or "").strip().lower()
    candidates: list[Path] = []
    for source_key in source_roots:
        root = DATASET_SOURCE_ROOTS.get(source_key)
        if root is None or not root.exists():
            continue
        for variant in dict.fromkeys(stem_variants):
            candidates.extend(root.rglob(f"{variant}_encrypted.json"))
            candidates.extend(root.rglob(f"{variant}.json"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    if chart_type:
        typed = [path for path in existing if dataset_chart_category(path.parent.parent) == chart_type]
        if typed:
            return sorted(typed, key=lambda path: len(path.parts))[0]
    return sorted(existing, key=lambda path: len(path.parts))[0]


def resolve_gt_config_path(sample_id: str) -> Path:
    sample = dataset_sample_by_id(sample_id)
    config_path = gt_config_path_for_sample(sample)
    if config_path is None:
        raise HTTPException(status_code=404, detail="GT config JSON not found for this sample")
    return config_path


def resolve_gt_image_path(config_path: Path, image_type: str) -> Path:
    data = load_json(config_path)
    image_paths = data.get("image_paths") if isinstance(data.get("image_paths"), dict) else {}
    if image_type in {"grid", "with_grid", "grid_with_grid"}:
        value = image_paths.get("grid_with_grid") or image_paths.get("with_grid")
    else:
        value = image_paths.get("no_grid") or data.get("image_path")
    if not value:
        raise HTTPException(status_code=404, detail=f"GT image path not found: {image_type}")
    path = Path(str(value))
    if path.is_absolute():
        return path
    for base in (config_path.parent, config_path.parent.parent, BACKEND_DIR, PROJECT_ROOT):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    if image_type in {"grid", "with_grid", "grid_with_grid"}:
        rendered = render_gt_grid_image(config_path, dataset=data)
        if rendered is not None and rendered.exists():
            return rendered
    return (config_path.parent.parent / path).resolve()


def enrich_gt_experiment_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    config_path = gt_config_path_for_sample(sample)
    gt_grid_path = None
    if config_path is not None:
        try:
            gt_grid_path = resolve_gt_image_path(config_path, "grid_with_grid")
        except Exception:
            gt_grid_path = None
    return {
        **sample,
        "cached": False,
        "evaluation_cached": False,
        "gt_config_available": config_path is not None,
        "gt_grid_available": bool(gt_grid_path and gt_grid_path.exists()),
        "gt_config_path": str(config_path) if config_path else None,
        "gt_grid_url": f"/api/gt-experiment/image/{sample['sample_id']}/grid" if gt_grid_path else None,
        "image_url": f"/api/gt-experiment/image/{sample['sample_id']}/original",
    }


@contextmanager
def temporary_env(values: Dict[str, str]):
    old_values = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _numeric(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _axis_range(ticks: Any) -> Optional[float]:
    if not isinstance(ticks, list):
        return None
    values = [_numeric(item) for item in ticks]
    values = [item for item in values if item is not None]
    if len(values) < 2:
        return None
    span = max(values) - min(values)
    return span if span > 0 else None


def _normalize_key(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _gt_lookup_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    lookup: Dict[str, Any] = {}

    def add(key: Any, value: Any) -> None:
        normalized = _normalize_key(key)
        if normalized:
            lookup[normalized] = value

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_text = str(key)
                combined = f"{prefix}, {key_text}" if prefix else key_text
                if isinstance(nested, dict):
                    walk(combined, nested)
                else:
                    add(combined, nested)
                    add(key_text, nested)
        elif prefix:
            add(prefix, value)

    for key in ("data_points", "ground_truth", "data", "labels"):
        value = config.get(key)
        if isinstance(value, dict):
            walk("", value)
    return lookup


def _lookup_gt_value(config: Dict[str, Any], row: Dict[str, Any]) -> Any:
    lookup = _gt_lookup_from_config(config)
    candidates: list[Any] = [
        row.get("point"),
        row.get("point_name"),
        row.get("id"),
        row.get("label"),
        row.get("theta_label"),
    ]
    series_name = row.get("series_name") or row.get("category")
    for label_key in ("point", "point_name", "label", "theta_label"):
        label = row.get(label_key)
        if series_name and label:
            candidates.append(f"{series_name}, {label}")
    for candidate in candidates:
        normalized = _normalize_key(candidate)
        if normalized in lookup:
            return lookup[normalized]
    return None


def _normalize_share(value: Any) -> Optional[float]:
    number = _numeric(value)
    if number is None:
        return None
    number = abs(number)
    if number > 1.0:
        if number <= 100.0:
            return number / 100.0
        return number / 360.0
    return number


def _first_numeric(*values: Any) -> Optional[float]:
    for value in values:
        number = _numeric(value)
        if number is not None:
            return number
    return None


def _error_metrics(pred: Any, gt: Any, axis_span: Optional[float]) -> Dict[str, Optional[float]]:
    pred_number = _numeric(pred)
    gt_number = _numeric(gt)
    if pred_number is None or gt_number is None:
        return {"absolute_error": None, "RE": None, "RNE": None}
    absolute_error = abs(pred_number - gt_number)
    re = absolute_error / max(abs(gt_number), 1e-12)
    rne = absolute_error / axis_span if axis_span else None
    return {
        "absolute_error": absolute_error,
        "RE": re,
        "RNE": rne,
    }


def _is_invalid_prediction_sentinel(*values: Any) -> bool:
    normalized = [str(value).strip() for value in values if value is not None]
    if not normalized:
        return False
    return any(value.lower() in {"", "none", "null", "nan"} for value in normalized)


def _is_false_like(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "not_readable", "unreadable"}


def _row_has_invalid_prediction(row: Dict[str, Any]) -> bool:
    if _is_false_like(row.get("prediction_readable")):
        return True
    pred_x = row.get("pred_x")
    pred_y = row.get("pred_y")
    if _is_invalid_prediction_sentinel(pred_x, pred_y):
        return True
    if _is_invalid_prediction_sentinel(row.get("pred"), row.get("pred_pct"), row.get("pred_r")):
        return True
    return False


def _read_csv_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _row_metric_record(
    *,
    config: Dict[str, Any],
    config_path: Path,
    row: Dict[str, Any],
    x_span: Optional[float],
    y_span: Optional[float],
    r_span: Optional[float],
) -> Dict[str, Any]:
    prompt_type = str(row.get("prompt_type") or "")
    stage = normalize_stage(prompt_type)
    gt_lookup = _lookup_gt_value(config, row)
    valid_prediction = not _row_has_invalid_prediction(row)

    x_metrics = (
        _error_metrics(row.get("pred_x"), row.get("gt_x"), x_span)
        if valid_prediction
        else {"absolute_error": None, "RE": None, "RNE": None}
    )
    y_metrics = (
        _error_metrics(row.get("pred_y"), row.get("gt_y"), y_span)
        if valid_prediction
        else {"absolute_error": None, "RE": None, "RNE": None}
    )

    pred_share = None
    if valid_prediction:
        pred_share = _normalize_share(row.get("pred_pct"))
        if pred_share is None:
            pred_share = _normalize_share(row.get("pred"))
        if pred_share is None:
            pred_share = _normalize_share(row.get("percentage"))
    gt_share = _normalize_share(row.get("gt_pct"))
    if gt_share is None:
        gt_share = _normalize_share(row.get("gt"))
    if gt_share is None:
        gt_share = _normalize_share(gt_lookup)
    share_metrics = _error_metrics(pred_share, gt_share, 1.0)

    pred_r = _first_numeric(row.get("pred_r"), row.get("r"), row.get("value")) if valid_prediction else None
    gt_r = _first_numeric(row.get("gt_r"), gt_lookup)
    radial_metrics = _error_metrics(pred_r, gt_r, r_span)

    row_readable = row.get("prediction_readable")
    prediction_readable = (
        valid_prediction
        if row_readable is None or str(row_readable).strip() == ""
        else not _is_false_like(row_readable)
    )

    metric_candidates = [x_metrics, y_metrics, share_metrics, radial_metrics]
    available_re = [item["RE"] for item in metric_candidates if item.get("RE") is not None]
    available_rne = [item["RNE"] for item in metric_candidates if item.get("RNE") is not None]

    prediction_payload = {
        "x": row.get("pred_x"),
        "y": row.get("pred_y"),
        "share": pred_share,
        "r": pred_r,
        "raw_pred": row.get("pred"),
        "percentage": row.get("percentage"),
        "start_angle": row.get("start_angle"),
        "end_angle": row.get("end_angle"),
    }
    gt_payload = {
        "x": row.get("gt_x"),
        "y": row.get("gt_y"),
        "share": gt_share,
        "r": gt_r,
        "value": gt_lookup,
    }
    return {
        "call_id": row.get("call_id"),
        "chart_name": config.get("chart_id") or config_path.stem,
        "processing_object": row.get("point") or row.get("point_name") or row.get("label") or row.get("id"),
        "object_category": row.get("category") or row.get("series_name"),
        "prompt_type": prompt_type,
        "stage": stage,
        "round": row.get("run"),
        "image_type": row.get("image_type"),
        "image_path": row.get("image_path"),
        "gt": gt_payload,
        "prediction": prediction_payload,
        "valid_prediction": valid_prediction,
        "prediction_readable": prediction_readable,
        "x": x_metrics,
        "y": y_metrics,
        "share": share_metrics,
        "radial": radial_metrics,
        "RE": sum(available_re) / len(available_re) if available_re else None,
        "RNE": sum(available_rne) / len(available_rne) if available_rne else None,
    }


def summarize_gt_prediction_records(
    *,
    config_path: Path,
    prediction_results: list[Dict[str, Any]],
    result_csv_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    config = load_json(config_path)
    x_span = _axis_range(config.get("x_ticks"))
    y_span = _axis_range(config.get("y_ticks"))
    r_span = _axis_range(config.get("r_ticks"))
    records: list[Dict[str, Any]] = []
    for chart_result in prediction_results:
        result_dir = chart_result.get("result_dir")
        if not isinstance(result_dir, str):
            continue
        result_path = Path(result_dir)
        if result_csv_names is not None:
            csv_candidates = [result_path / name for name in result_csv_names]
        else:
            primary_csv = result_path / "experiment_results.csv"
            csv_candidates = (
                [primary_csv]
                if primary_csv.exists()
                else [
                    result_path / "full_results_with_xre.csv",
                    result_path / "full_results_with_yre.csv",
                    result_path / "full_results_with_rre.csv",
                ]
            )
        seen: set[str] = set()
        for csv_path in csv_candidates:
            if not csv_path.exists() or str(csv_path) in seen:
                continue
            seen.add(str(csv_path))
            for row in _read_csv_rows(csv_path):
                records.append(
                    _row_metric_record(
                        config=config,
                        config_path=config_path,
                        row=row,
                        x_span=x_span,
                        y_span=y_span,
                        r_span=r_span,
                    )
                )
    re_values = [record["RE"] for record in records if record.get("RE") is not None]
    rne_values = [record["RNE"] for record in records if record.get("RNE") is not None]
    return {
        "record_count": len(records),
        "avg_RE": sum(re_values) / len(re_values) if re_values else None,
        "avg_RNE": sum(rne_values) / len(rne_values) if rne_values else None,
        "records": records,
    }


def summarize_gt_selected_predictions(
    *,
    config_path: Path,
    prediction_results: list[Dict[str, Any]],
) -> Dict[str, Any]:
    config = load_json(config_path)
    x_span = _axis_range(config.get("x_ticks"))
    y_span = _axis_range(config.get("y_ticks"))
    r_span = _axis_range(config.get("r_ticks"))
    records: list[Dict[str, Any]] = []
    for chart_result in prediction_results:
        for prediction in chart_result.get("predictions", []):
            if not isinstance(prediction, dict):
                continue
            value = prediction.get("value")
            row: Dict[str, Any] = {
                "point": prediction.get("id") or prediction.get("label"),
                "label": prediction.get("label") or prediction.get("id"),
                "series_name": prediction.get("series_name"),
                "category": prediction.get("category"),
                "prompt_type": prediction.get("prompt_type") or "selected_prediction",
                "image_type": prediction.get("image_type") or "selected",
                "run": "final",
                "image_path": prediction.get("image_path"),
                "prediction_readable": True,
            }
            if isinstance(value, dict):
                row["pred_x"] = value.get("x")
                row["pred_y"] = value.get("y")
            elif prediction.get("axis") == "xy":
                row["pred_x"] = prediction.get("x")
                row["pred_y"] = prediction.get("y")
            elif prediction.get("axis") in {"x", "y"}:
                row[f"pred_{prediction.get('axis')}"] = value
            elif prediction.get("axis") in {"theta", "share", "percentage"}:
                row["pred_pct"] = prediction.get("percentage", value)
                row["pred"] = value
            elif prediction.get("axis") in {"r", "radial"}:
                row["pred_r"] = value
            else:
                row["pred"] = value
            row.update(
                {
                    "percentage": prediction.get("percentage"),
                    "start_angle": prediction.get("start_angle"),
                    "end_angle": prediction.get("end_angle"),
                }
            )
            records.append(
                _row_metric_record(
                    config=config,
                    config_path=config_path,
                    row=row,
                    x_span=x_span,
                    y_span=y_span,
                    r_span=r_span,
                )
            )
    re_values = [record["RE"] for record in records if record.get("RE") is not None]
    rne_values = [record["RNE"] for record in records if record.get("RNE") is not None]
    return {
        "record_count": len(records),
        "avg_RE": sum(re_values) / len(re_values) if re_values else None,
        "avg_RNE": sum(rne_values) / len(rne_values) if rne_values else None,
        "records": records,
    }


def _jsonl_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(sanitize_json_value(row), ensure_ascii=False, allow_nan=False) + "\n")


def _normalized_path_key(value: Any) -> str:
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve()).casefold()
    except Exception:
        return str(value).casefold()


def _modal_metric_key(value: Dict[str, Any], *, include_path: bool) -> tuple[str, str, str, str, str]:
    path_value = value.get("image_path") if include_path else ""
    return (
        _normalize_key(value.get("chart_name")),
        _normalize_key(value.get("processing_object")),
        _normalize_key(value.get("stage")),
        str(value.get("round") or ""),
        _normalized_path_key(path_value),
    )


def enrich_modal_call_logs(
    *,
    modal_log_path: Path,
    modal_full_log_path: Path,
    metric_records: list[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    by_call_id: Dict[str, Dict[str, Any]] = {}
    by_full_key: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    by_weak_key: Dict[tuple[str, str, str, str, str], Dict[str, Any]] = {}
    for record in metric_records:
        call_id = str(record.get("call_id") or "").strip()
        if call_id:
            by_call_id.setdefault(call_id, record)
        full_key = _modal_metric_key(record, include_path=True)
        weak_key = _modal_metric_key(record, include_path=False)
        by_full_key.setdefault(full_key, record)
        by_weak_key.setdefault(weak_key, record)

    def enrich_one(path: Path) -> Optional[Path]:
        rows = _jsonl_rows(path)
        if not rows:
            return None
        enriched_rows: list[Dict[str, Any]] = []
        for row in rows:
            call_id = str(row.get("call_id") or "").strip()
            full_key = _modal_metric_key(row, include_path=True)
            weak_key = _modal_metric_key(row, include_path=False)
            record = by_call_id.get(call_id) if call_id else None
            record = record or by_full_key.get(full_key) or by_weak_key.get(weak_key)
            if record:
                row = {
                    **row,
                    "prompt_type": record.get("prompt_type") or row.get("prompt_type") or record.get("stage") or row.get("stage"),
                    "stage": record.get("stage") or row.get("stage") or record.get("prompt_type") or row.get("prompt_type"),
                    "round": record.get("round") or row.get("round"),
                    "valid_prediction": record.get("valid_prediction"),
                    "prediction_readable": record.get("prediction_readable"),
                    "duration_ms": row.get("duration_ms") or row.get("request_duration_ms"),
                    "structured_gt": record.get("gt"),
                    "structured_prediction": record.get("prediction"),
                    "metrics": {
                        "RE": record.get("RE"),
                        "RNE": record.get("RNE"),
                        "x": record.get("x"),
                        "y": record.get("y"),
                        "share": record.get("share"),
                        "radial": record.get("radial"),
                    },
                }
            enriched_rows.append(row)
        enriched_path = path.with_name(f"{path.stem}_enriched{path.suffix}")
        _write_jsonl(enriched_path, enriched_rows)
        return enriched_path

    summary_path = enrich_one(modal_log_path)
    full_path = enrich_one(modal_full_log_path)
    return {
        "modal_call_log_enriched": str(summary_path) if summary_path else None,
        "modal_full_log_enriched": str(full_path) if full_path else None,
    }


def _json_cell(value: Any) -> str:
    if value in (None, ""):
        return ""
    return json.dumps(sanitize_json_value(value), ensure_ascii=False, allow_nan=False)


def _nested_value(mapping: Any, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def write_modal_call_records_csv(enriched_modal_log_path: Optional[str], output_path: Path) -> Optional[str]:
    if not enriched_modal_log_path:
        return None
    rows = _jsonl_rows(Path(enriched_modal_log_path))
    if not rows:
        return None

    columns = [
        "call_id",
        "chart_name",
        "processing_object",
        "object_category",
        "stage",
        "prompt_type",
        "round",
        "gt",
        "gt_x",
        "gt_y",
        "gt_share",
        "gt_r",
        "gt_value",
        "prediction",
        "pred_x",
        "pred_y",
        "pred_share",
        "pred_r",
        "pred_value",
        "pred_percentage",
        "pred_start_angle",
        "pred_end_angle",
        "valid_prediction",
        "prediction_readable",
        "attempts",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "text_prompt_tokens",
        "single_request_duration_ms",
        "request_duration_ms",
        "success",
        "model",
        "image_path",
        "image_width",
        "image_height",
        "estimated_image_tokens",
        "vision_profile",
        "raw_prediction",
        "RE",
        "RNE",
        "x_RE",
        "x_RNE",
        "y_RE",
        "y_RNE",
        "share_RE",
        "share_RNE",
        "radial_RE",
        "radial_RNE",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            gt = row.get("structured_gt") if isinstance(row.get("structured_gt"), dict) else row.get("gt")
            prediction = (
                row.get("structured_prediction")
                if isinstance(row.get("structured_prediction"), dict)
                else row.get("prediction")
            )
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            image_width, image_height = _image_dimensions(row.get("image_path"))
            model_name = str(row.get("model") or "")
            vision_profile = profile_for_model(model_name)
            estimated_tokens = (
                estimate_image_tokens(
                    image_width,
                    image_height,
                    profile=vision_profile,
                )
                if image_width and image_height
                else None
            )
            writer.writerow(
                {
                    "call_id": row.get("call_id"),
                    "chart_name": row.get("chart_name"),
                    "processing_object": row.get("processing_object"),
                    "object_category": row.get("object_category"),
                    "stage": row.get("stage") or row.get("prompt_type"),
                    "prompt_type": row.get("prompt_type") or row.get("stage"),
                    "round": row.get("round"),
                    "gt": _json_cell(gt),
                    "gt_x": _nested_value(gt, "x"),
                    "gt_y": _nested_value(gt, "y"),
                    "gt_share": _nested_value(gt, "share"),
                    "gt_r": _nested_value(gt, "r"),
                    "gt_value": _nested_value(gt, "value"),
                    "prediction": _json_cell(prediction),
                    "pred_x": _nested_value(prediction, "x"),
                    "pred_y": _nested_value(prediction, "y"),
                    "pred_share": _nested_value(prediction, "share"),
                    "pred_r": _nested_value(prediction, "r"),
                    "pred_value": _nested_value(prediction, "value"),
                    "pred_percentage": _nested_value(prediction, "percentage"),
                    "pred_start_angle": _nested_value(prediction, "start_angle"),
                    "pred_end_angle": _nested_value(prediction, "end_angle"),
                    "valid_prediction": row.get("valid_prediction"),
                    "prediction_readable": row.get("prediction_readable"),
                    "attempts": row.get("attempts"),
                    "input_tokens": row.get("input_tokens"),
                    "output_tokens": row.get("output_tokens"),
                    "total_tokens": row.get("total_tokens"),
                    "text_prompt_tokens": row.get("text_prompt_tokens"),
                    "single_request_duration_ms": row.get("duration_ms") or row.get("request_duration_ms"),
                    "request_duration_ms": row.get("request_duration_ms") or row.get("duration_ms"),
                    "success": row.get("success"),
                    "model": row.get("model"),
                    "image_path": row.get("image_path"),
                    "image_width": image_width,
                    "image_height": image_height,
                    "estimated_image_tokens": estimated_tokens,
                    "vision_profile": vision_profile.key,
                    "raw_prediction": row.get("raw_prediction"),
                    "RE": metrics.get("RE"),
                    "RNE": metrics.get("RNE"),
                    "x_RE": _nested_value(metrics, "x", "RE"),
                    "x_RNE": _nested_value(metrics, "x", "RNE"),
                    "y_RE": _nested_value(metrics, "y", "RE"),
                    "y_RNE": _nested_value(metrics, "y", "RNE"),
                    "share_RE": _nested_value(metrics, "share", "RE"),
                    "share_RNE": _nested_value(metrics, "share", "RNE"),
                    "radial_RE": _nested_value(metrics, "radial", "RE"),
                    "radial_RNE": _nested_value(metrics, "radial", "RNE"),
                }
            )
    return str(output_path)


def _image_dimensions(path_value: Any) -> tuple[int | None, int | None]:
    if not path_value:
        return None, None
    try:
        from PIL import Image

        path = Path(str(path_value))
        if not path.exists():
            return None, None
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def register_dataset_sample(sample_id: str) -> Dict[str, Any]:
    sample = dataset_sample_by_id(sample_id)
    source_path = Path(sample["image_path"])
    cache_dir = DATASET_PREVIEW_CACHE_DIR / sample_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_image_path = cache_dir / f"{sample_id}_original{source_path.suffix.lower() or '.png'}"
    if not cached_image_path.exists() or cached_image_path.stat().st_mtime < source_path.stat().st_mtime:
        shutil.copy2(source_path, cached_image_path)

    output_dir = cache_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_id = f"dataset_{sample_id}"
    chart_type = sample["chart_type"]
    tick_sidecar = output_dir / f"{cached_image_path.stem}_ticks.json"
    encrypted_image_path = None
    colored_image_path = None
    batch_record = batch_cached_dataset_record(sample["source"], sample["relative_path"])
    if batch_record is not None and not preview_tick_sidecar_current(tick_sidecar):
        encrypted_image_path, colored_image_path = seed_preview_cache_from_batch_record(
            batch_record,
            output_dir,
            cached_image_path.stem,
        )
    if tick_sidecar.exists():
        try:
            ticks_data = load_json(tick_sidecar)
            encrypted_image_path = ticks_data.get("encrypted_grid_path")
            colored_image_path = ticks_data.get("colored_grid_path")
        except Exception:
            encrypted_image_path = None
            colored_image_path = None
    seeded_encrypted = output_dir / f"{cached_image_path.stem}_with_grid.png"
    if seeded_encrypted.exists():
        encrypted_image_path = str(seeded_encrypted)
    polar_encrypted = output_dir / f"{cached_image_path.stem}_encode.png"
    if polar_encrypted.exists():
        encrypted_image_path = str(polar_encrypted)
    seeded_colored = output_dir / f"{cached_image_path.stem}_with_grid_color.png"
    if seeded_colored.exists():
        colored_image_path = str(seeded_colored)

    processed = bool(encrypted_image_path and Path(encrypted_image_path).exists())
    chart_info = {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "coordinate_system": sample["coordinate_system"],
        "confidence": 1.0,
        "axis_repair": {},
        "image_path": str(cached_image_path),
        "json_path": None,
        "processed": processed,
        "evaluated": False,
        "dataset_preview": True,
        "dataset_sample": {key: value for key, value in sample.items() if key != "image_path"},
        "output_dir": str(output_dir),
    }
    if processed:
        chart_info["encrypted_image_path"] = encrypted_image_path
        chart_info["colored_image_path"] = colored_image_path
    evaluation_path = dataset_evaluation_cache_path(sample_id)
    if evaluation_cache_usable(evaluation_path, chart_type):
        chart_info["evaluated"] = True
        chart_info["evaluation_results_path"] = str(evaluation_path)
    charts_db[chart_id] = chart_info
    return chart_info


def register_chart(image_file: UploadFile, json_file: Optional[UploadFile] = None) -> Dict[str, Any]:
    chart_id = str(uuid.uuid4())
    image_suffix = Path(image_file.filename or "").suffix or ".png"
    image_path = UPLOAD_DIR / f"{chart_id}_image{image_suffix}"
    json_path = UPLOAD_DIR / f"{chart_id}_data.json" if json_file else None

    save_upload_file(image_file, image_path)
    if json_file and json_path:
        save_upload_file(json_file, json_path)

    detection = detect_chart_type(image_path)
    chart_type = detection["type"]
    confidence = detection.get("confidence", 0.5)
    axis_repair = detection.get("axis_repair") or {}
    coordinate_system = get_coordinate_system(chart_type).value

    chart_info = {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "coordinate_system": coordinate_system,
        "confidence": confidence,
        "axis_repair": axis_repair,
        "image_path": str(image_path),
        "json_path": str(json_path) if json_path else None,
        "processed": False,
        "evaluated": False,
    }
    charts_db[chart_id] = chart_info
    return chart_info


def get_chart(chart_id: str) -> Dict[str, Any]:
    chart_info = charts_db.get(chart_id)
    if not chart_info:
        raise HTTPException(status_code=404, detail="Chart not found")
    return chart_info


def get_chart_output_dir(chart_type: str) -> Path:
    output_dir = OUTPUT_DIR / chart_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def image_response_url(path: Union[str, Path]) -> str:
    image_path = Path(path)
    version = ""
    try:
        version = f"?v={image_path.stat().st_mtime_ns}"
    except OSError:
        pass
    return f"/api/images/{image_path.name}{version}"


def result_response_url(path: Union[str, Path]) -> str:
    result_path = Path(path)
    version = ""
    try:
        version = f"?v={result_path.stat().st_mtime_ns}"
    except OSError:
        pass
    return f"/api/results/{result_path.name}{version}"


PREFERRED_EXTRACTION_PROMPTS = ("geometry", "amplifier", "feedback", "grid")


def bar_base_type(chart_type: Any) -> str:
    text = str(chart_type or "").lower()
    if text in {"h_bar", "h_stacked_bar"}:
        return "h_bar"
    if text in {"v_bar", "v_stacked_bar"}:
        return "v_bar"
    return text


def is_bar_type(chart_type: Any) -> bool:
    return bar_base_type(chart_type) in {"h_bar", "v_bar"}


def strip_external_reference_data(
    data: Dict[str, Any],
    *,
    preserve_data: bool = False,
    preserve_series_color: bool = False,
) -> None:
    """Remove explicit GT/reference fields while keeping system-extracted metadata."""
    data.pop("reference_config_path", None)
    data.pop("reference_chart_id", None)

    keys = ["data_points", "ground_truth", "labels"]
    if not preserve_data:
        keys.append("data")
    for key in keys:
        data.pop(key, None)


def ensure_series_color_from_colors(data: Dict[str, Any]) -> None:
    current = data.get("series_color")
    if isinstance(current, dict) and current:
        return
    colors = data.get("colors")
    if not isinstance(colors, list):
        return
    series_color: Dict[str, str] = {}
    for index, item in enumerate(colors):
        if not isinstance(item, dict) or not item.get("color"):
            continue
        name = str(item.get("name") or f"Series {index + 1}").strip() or f"Series {index + 1}"
        series_color[name] = str(item["color"])
    if series_color:
        data["series_color"] = series_color


def processed_json_payload(eval_json_path: Union[str, Path], chart_type: Optional[str] = None) -> Dict[str, Any]:
    path = Path(eval_json_path)
    data = load_json(path)
    if not path.stem.endswith("_ticks"):
        merge_tick_sidecar(data, path.parent, path.stem)
    strip_external_reference_data(
        data,
        preserve_data=False,
        preserve_series_color=True,
    )
    ensure_series_color_from_colors(data)
    return data


def generated_json_path(chart_info: Dict[str, Any], output_dir: Path) -> Path:
    return output_dir / f"{Path(chart_info['image_path']).stem}.json"


def ensure_radar_series_color(data: Dict[str, Any], chart_info: Dict[str, Any], output_dir: Path) -> None:
    if chart_info.get("chart_type") != "radar":
        return
    current = data.get("series_color")
    if isinstance(current, dict) and current:
        return

    try:
        matcher = RadarColorMatcher()
        matcher.output_dir = str(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = matcher.extract_legend_series_colors(chart_info["image_path"], use_auto_crop=True)
    except Exception as error:
        print(f"Radar series-color extraction skipped: {safe_error_message(error)}")
        return

    if not isinstance(result, dict):
        return
    series_color = result.get("series_color") or result.get("entity_colors")
    if isinstance(series_color, dict) and series_color:
        data["series_color"] = {str(name): str(color) for name, color in series_color.items() if name and color}
        data["colors"] = [
            {"name": str(name), "color": str(color)}
            for name, color in data["series_color"].items()
        ]
        if result.get("legend_path"):
            data["legend_path"] = str(Path(result["legend_path"]).absolute())
        print(f"Radar series_color extracted from image: {data['series_color']}")


def enrich_generated_json(
    chart_info: Dict[str, Any],
    output_dir: Path,
    encrypted_image_path: Union[str, Path],
) -> Path:
    json_path = generated_json_path(chart_info, output_dir)
    generated_data = load_json(json_path) if json_path.exists() else {}
    merge_tick_sidecar(generated_data, output_dir, Path(chart_info["image_path"]).stem)
    strip_external_reference_data(
        generated_data,
        preserve_data=False,
        preserve_series_color=True,
    )
    ensure_radar_series_color(generated_data, chart_info, output_dir)
    ensure_series_color_from_colors(generated_data)
    existing_image_paths = generated_data.get("image_paths") if isinstance(generated_data.get("image_paths"), dict) else {}

    generated_data.update(
        {
            "chart_id": chart_info["chart_id"],
            "chart_type": chart_info["chart_type"],
            "coordinate_system": chart_info["coordinate_system"],
            "image_paths": {
                "no_grid": str(Path(chart_info["image_path"]).absolute()),
                "with_grid": str(Path(encrypted_image_path).absolute()),
                **{
                    key: value
                    for key, value in existing_image_paths.items()
                    if key in {"grid_with_grid_color", "with_grid_color", "colored_grid"}
                },
            },
        }
    )

    write_json(json_path, generated_data)
    return json_path


def merge_tick_sidecar(data: Dict[str, Any], output_dir: Path, image_stem: str) -> None:
    ticks_json_path = output_dir / f"{image_stem}_ticks.json"
    if not ticks_json_path.exists():
        tick_candidates = sorted(
            output_dir.glob("*_ticks.json"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        ticks_json_path = tick_candidates[0] if tick_candidates else ticks_json_path
    if not ticks_json_path.exists():
        return

    ticks_data = load_json(ticks_json_path)
    for key, value in ticks_data.items():
        if key == "chart_id":
            continue
        data[key] = value

    image_paths = data.setdefault("image_paths", {})
    if isinstance(image_paths, dict):
        encrypted_grid_path = ticks_data.get("encrypted_grid_path")
        colored_grid_path = ticks_data.get("colored_grid_path")
        basic_grid_path = ticks_data.get("basic_grid_path")
        if encrypted_grid_path:
            image_paths["grid_with_grid"] = str(Path(encrypted_grid_path).absolute())
            image_paths["with_grid"] = str(Path(encrypted_grid_path).absolute())
        if colored_grid_path:
            image_paths["grid_with_grid_color"] = str(Path(colored_grid_path).absolute())
            image_paths["with_grid_color"] = str(Path(colored_grid_path).absolute())
            image_paths["colored_grid"] = str(Path(colored_grid_path).absolute())
        if basic_grid_path:
            image_paths["basic_grid"] = str(Path(basic_grid_path).absolute())


def axis_data_from_generated_sidecar(chart_info: Dict[str, Any], output_dir: Path) -> Optional[Dict[str, Any]]:
    image_stem = Path(chart_info["image_path"]).stem
    candidates = [
        output_dir / f"{image_stem}_ticks.json",
        output_dir / f"{chart_info['chart_id']}_ticks.json",
    ]
    for sidecar_path in candidates:
        if not sidecar_path.exists():
            continue
        try:
            ticks_data = load_json(sidecar_path)
        except Exception as error:
            print(f"Could not read generated tick sidecar {sidecar_path}: {safe_error_message(error)}")
            continue

        axis_data = {
            "x_ticks": ticks_data.get("x_ticks", []),
            "y_ticks": ticks_data.get("y_ticks", []),
            "x_axis_type": ticks_data.get("x_axis_type", "numeric"),
            "y_axis_type": ticks_data.get("y_axis_type", "numeric"),
            "axis_source": "generated_tick_sidecar",
            "tick_sidecar_path": str(sidecar_path),
            "generation_cache_disabled": bool(ticks_data.get("generation_cache_disabled")),
        }
        if ticks_data.get("x_axis") is not None:
            axis_data["x_axis"] = ticks_data.get("x_axis")
        if ticks_data.get("y_axis") is not None:
            axis_data["y_axis"] = ticks_data.get("y_axis")
        return axis_data
    return None


def save_axis_data(chart_info: Dict[str, Any], output_dir: Path) -> None:
    image_stem = Path(chart_info["image_path"]).stem
    if chart_info["chart_type"] == "radar":
        from polar.radar.demo_axis_find_radar import RadarChartAxisFinder

        finder = RadarChartAxisFinder()
        finder.output_dir = str(output_dir)
        finder.axes_output_dir = str(output_dir)
        axis_data = finder.process_single_image(
            chart_info["image_path"],
            output_json_path=str(output_dir / f"{image_stem}_axes.json"),
        )
        if axis_data and chart_info["chart_id"] != image_stem:
            write_json(output_dir / f"{chart_info['chart_id']}_axes.json", axis_data, indent=4)
        return

    if chart_info["chart_type"] == "rose":
        from polar.rose.demo_axis_find_rose import RoseChartAxisFinder

        finder = RoseChartAxisFinder()
        finder.output_dir = str(output_dir)
        finder.axes_output_dir = str(output_dir)
        axis_data = finder.process_single_image(
            chart_info["image_path"],
            output_json_path=str(output_dir / f"{image_stem}_axes.json"),
        )
        if axis_data and chart_info["chart_id"] != image_stem:
            write_json(output_dir / f"{chart_info['chart_id']}_axes.json", axis_data, indent=4)
        return

    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    axis_data = axis_data_from_generated_sidecar(chart_info, output_dir)
    if axis_data:
        write_json(output_dir / f"{chart_info['chart_id']}_axes.json", axis_data, indent=4)
        return

    try:
        axis_data = processor.find_axis(
            chart_info["image_path"],
            axis_repair_hint=chart_info.get("axis_repair"),
            disable_cache=not chart_info.get("dataset_preview"),
        )
    except Exception as error:
        print(f"Axis detection failed: {safe_error_message(error)}")
        return

    if axis_data:
        write_json(output_dir / f"{chart_info['chart_id']}_axes.json", axis_data, indent=4)


def load_generated_chart_metadata(chart_info: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    image_stem = Path(chart_info["image_path"]).stem
    merged: Dict[str, Any] = {}
    for path in (output_dir / f"{image_stem}.json", output_dir / f"{image_stem}_ticks.json"):
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except Exception as error:
            print(f"Could not read generated bar metadata {path}: {safe_error_message(error)}")
            continue
        if isinstance(data, dict):
            merged.update(data)
    return merged


def infer_bar_type_from_axis_metadata(data: Dict[str, Any]) -> Optional[str]:
    axis_repair = data.get("axis_repair") if isinstance(data.get("axis_repair"), dict) else {}
    hint = axis_repair.get("hint") if isinstance(axis_repair.get("hint"), dict) else {}
    reason = str(hint.get("reason") or data.get("reason") or "").lower()
    if "horizontal bar" in reason or "horizontal bars" in reason:
        return "h_bar"
    if "vertical bar" in reason or "vertical bars" in reason:
        return "v_bar"

    x_role = axis_role(data.get("x_axis_type"))
    y_role = axis_role(data.get("y_axis_type"))
    if x_role == "numeric" and y_role == "text":
        return "h_bar"
    if x_role == "text" and y_role == "numeric":
        return "v_bar"
    return None


def axis_role(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if any(marker in text for marker in ("numeric", "number", "value", "quant", "数值")):
        return "numeric"
    if any(marker in text for marker in ("text", "category", "categorical", "label", "文字", "类别", "离散")):
        return "text"
    return None


def process_chart_image(chart_info: Dict[str, Any], force: bool = False) -> str:
    if not force and chart_info.get("processed") and chart_info.get("encrypted_image_path"):
        return chart_info["encrypted_image_path"]

    original_type = chart_info["chart_type"]
    output_dir = Path(chart_info.get("output_dir") or get_chart_output_dir(original_type))
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = ChartProcessorFactory.create_processor(original_type)
    encrypted_image_path = processor.encode_image(
        chart_info["image_path"],
        str(output_dir),
        axis_repair_hint=chart_info.get("axis_repair"),
        disable_cache=not chart_info.get("dataset_preview"),
    )

    if not encrypted_image_path:
        raise HTTPException(status_code=500, detail="Chart processing failed")

    save_axis_data(chart_info, output_dir)
    enrich_generated_json(chart_info, output_dir, encrypted_image_path)
    tick_sidecar = output_dir / f"{Path(chart_info['image_path']).stem}_ticks.json"
    colored_image_path = None
    if tick_sidecar.exists():
        try:
            colored_image_path = load_json(tick_sidecar).get("colored_grid_path")
        except Exception:
            colored_image_path = None

    chart_info.update(
        {
            "processed": True,
            "encrypted_image_path": encrypted_image_path,
            "colored_image_path": colored_image_path,
            "output_dir": str(output_dir),
        }
    )
    return encrypted_image_path


def candidate_eval_json_paths(chart_info: Dict[str, Any]) -> Iterable[Path]:
    output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
    image_stem = Path(chart_info["image_path"]).stem
    yield output_dir / f"{image_stem}.json"
    yield output_dir / f"{image_stem}_ticks.json"
    yield output_dir / f"{chart_info['chart_id']}.json"


def resolve_eval_json(chart_info: Dict[str, Any]) -> Path:
    for path in candidate_eval_json_paths(chart_info):
        if path.exists():
            output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
            data = load_json(path)
            merge_tick_sidecar(data, output_dir, Path(chart_info["image_path"]).stem)
            strip_external_reference_data(
                data,
                preserve_data=False,
                preserve_series_color=True,
            )
            ensure_radar_series_color(data, chart_info, output_dir)
            ensure_series_color_from_colors(data)
            write_json(path, data)
            return path

    output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    processed_data = processor.process_data(
        chart_info["chart_id"],
        chart_info["image_path"],
        None,
        str(output_dir),
        disable_cache=not chart_info.get("dataset_preview"),
    )

    if processed_data is None:
        available_files = [file.name for file in output_dir.glob("*.json")]
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate evaluation data. Available JSON files: {available_files}",
        )

    fallback_path = output_dir / f"{chart_info['chart_id']}.json"
    if isinstance(processed_data, dict):
        strip_external_reference_data(processed_data, preserve_data=False, preserve_series_color=True)
        ensure_radar_series_color(processed_data, chart_info, output_dir)
        ensure_series_color_from_colors(processed_data)
    write_json(fallback_path, processed_data)
    return fallback_path


def build_extraction_placeholder(chart_info: Dict[str, Any], eval_json_path: Path) -> Dict[str, Any]:
    data = processed_json_payload(eval_json_path, chart_info["chart_type"])
    x_ticks = data.get("x_ticks", [])
    y_ticks = data.get("y_ticks", [])
    r_ticks = data.get("r_ticks", [])
    theta_ticks = data.get("theta_ticks", [])
    predictions = data.get("predictions") if isinstance(data.get("predictions"), list) else []
    used_system_cv_fallback = False
    if not predictions:
        predictions = system_cv_predictions(chart_info, eval_json_path)
        used_system_cv_fallback = bool(predictions)

    return {
        "success": True,
        "mode": "prediction_extraction",
        "chart_id": chart_info["chart_id"],
        "chart_type": chart_info["chart_type"],
        "source_json": str(eval_json_path),
        "system_json": str(eval_json_path),
        "summary": {
            "object_count": len(predictions),
            "chart_runs": 0,
            "system_cv_fallback": used_system_cv_fallback,
        },
        "predictions": predictions,
        "processed_json": data,
        "quality": {
            "x_ticks_count": len(x_ticks) if isinstance(x_ticks, list) else 0,
            "y_ticks_count": len(y_ticks) if isinstance(y_ticks, list) else 0,
            "r_ticks_count": len(r_ticks) if isinstance(r_ticks, list) else 0,
            "theta_ticks_count": len(theta_ticks) if isinstance(theta_ticks, list) else 0,
            "colors_count": (
                len(data.get("colors", []))
                if isinstance(data.get("colors"), list)
                else len(data.get("series_color", {})) if isinstance(data.get("series_color"), dict) else 0
            ),
            "has_basic_grid": bool(data.get("basic_grid_path")),
            "has_encrypted_grid": bool(data.get("encrypted_grid_path") or data.get("image_paths", {}).get("with_grid")),
        },
        "note": "Value extraction runner is not wired for this chart type yet; no ground-truth error evaluation was run.",
    }


def normalize_result_payload(result: Dict[str, Any], result_path: Optional[Path] = None) -> Dict[str, Any]:
    if result.get("mode") != "bar_prediction_evaluation":
        system_json = result.get("system_json") or result.get("source_json")
        if system_json and "system_json" not in result:
            result["system_json"] = system_json
        if (
            "processed_json" not in result
            and isinstance(system_json, str)
            and Path(system_json).exists()
        ):
            result["processed_json"] = processed_json_payload(system_json, result.get("chart_type"))
        return result

    chart_type = str(result.get("chart_type", ""))
    legacy_runs = result.get("prediction_results")
    predictions = extract_predictions_from_legacy_runs(legacy_runs, chart_type)
    normalized = {
        "success": result.get("success", True),
        "mode": "prediction_extraction",
        "chart_id": result.get("chart_id"),
        "chart_type": chart_type,
        "system_json": result.get("system_json") or result.get("source_json"),
        "summary": {
            "object_count": len(predictions),
            "chart_runs": len(legacy_runs) if isinstance(legacy_runs, list) else 0,
        },
        "predictions": predictions,
        "artifacts": legacy_runs if isinstance(legacy_runs, list) else [],
        "legacy_mode": result.get("mode"),
    }
    system_json = normalized.get("system_json")
    if isinstance(system_json, str) and Path(system_json).exists():
        normalized["processed_json"] = processed_json_payload(system_json, chart_type)
    if result_path is not None:
        normalized["result_path"] = str(result_path)
    return normalized


def extract_predictions_from_legacy_runs(legacy_runs: Any, chart_type: str) -> list[Dict[str, Any]]:
    if not isinstance(legacy_runs, list):
        return []

    predictions: list[Dict[str, Any]] = []
    for run in legacy_runs:
        if not isinstance(run, dict):
            continue
        result_dir = run.get("result_dir")
        if not isinstance(result_dir, str):
            continue
        predictions.extend(extract_predictions_from_result_dir(Path(result_dir), chart_type))
    return predictions


def extract_predictions_from_result_dir(result_dir: Path, chart_type: str) -> list[Dict[str, Any]]:
    csv_path = preferred_legacy_csv(result_dir, chart_type)
    if csv_path is None:
        return []

    by_point: Dict[str, list[Dict[str, str]]] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                value = legacy_prediction_value(row, chart_type)
                if value is None:
                    continue
                point = row.get("point") or row.get("point_name")
                if point:
                    by_point.setdefault(point, []).append(row)
    except OSError:
        return []

    predictions: list[Dict[str, Any]] = []
    for point, rows in by_point.items():
        chosen = choose_prediction_row(rows)
        value = legacy_prediction_value(chosen, chart_type)
        if value is None:
            continue
        predictions.append(
            {
                "id": point,
                "series_name": point.rsplit(",", 1)[0].strip() if "," in point else "",
                "label": chosen.get("pred_y") if bar_base_type(chart_type) == "h_bar" else chosen.get("pred_x"),
                "axis": "x" if bar_base_type(chart_type) == "h_bar" else "y",
                "value": value,
                "prompt_type": chosen.get("prompt_type"),
                "image_type": chosen.get("image_type"),
                "image_path": chosen.get("image_path"),
            }
        )
    return predictions


def preferred_legacy_csv(result_dir: Path, chart_type: str) -> Optional[Path]:
    names = (
        ["full_results_with_xre.csv", "experiment_results.csv"]
        if bar_base_type(chart_type) == "h_bar"
        else ["full_results_with_yre.csv", "experiment_results.csv"]
    )
    for name in names:
        path = result_dir / name
        if path.exists():
            return path
    return None


def choose_prediction_row(rows: list[Dict[str, str]]) -> Dict[str, str]:
    for prompt_type in PREFERRED_EXTRACTION_PROMPTS:
        candidates = [row for row in rows if row.get("prompt_type") == prompt_type]
        if candidates:
            return candidates[-1]
    return rows[-1]


def legacy_prediction_value(row: Dict[str, str], chart_type: str) -> Optional[float]:
    key = "pred_x" if bar_base_type(chart_type) == "h_bar" else "pred_y"
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def numeric_tick_pairs(ticks: Any, pixels: Any) -> list[tuple[float, float]]:
    if not isinstance(ticks, list) or not isinstance(pixels, list):
        return []
    pairs: list[tuple[float, float]] = []
    for tick, pixel in zip(ticks, pixels):
        try:
            pairs.append((float(tick), float(pixel)))
        except (TypeError, ValueError):
            continue
    return pairs


def pixel_to_data_value(pixel: float, ticks: Any, pixels: Any) -> Optional[float]:
    pairs = numeric_tick_pairs(ticks, pixels)
    if len(pairs) < 2:
        return None
    pairs = sorted(pairs, key=lambda item: item[1])
    try:
        return float(np.interp(float(pixel), [item[1] for item in pairs], [item[0] for item in pairs]))
    except Exception:
        return None


def nearest_label(pixel: float, ticks: Any, pixels: Any) -> str:
    if not isinstance(ticks, list) or not isinstance(pixels, list):
        return ""
    pairs: list[tuple[str, float]] = []
    for tick, tick_pixel in zip(ticks, pixels):
        try:
            pairs.append((str(tick), float(tick_pixel)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return ""
    return min(pairs, key=lambda item: abs(item[1] - float(pixel)))[0]


def plot_bounds_from_ticks(data: Dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    x_pixels = [float(value) for value in data.get("x_pixels", []) if isinstance(value, (int, float))]
    y_pixels = [float(value) for value in data.get("y_pixels", []) if isinstance(value, (int, float))]
    x0 = int(max(0, min(x_pixels) if x_pixels else width * 0.08))
    x1 = int(min(width - 1, max(x_pixels) if x_pixels else width * 0.95))
    y0 = int(max(0, min(y_pixels) if y_pixels else height * 0.08))
    y1 = int(min(height - 1, max(y_pixels) if y_pixels else height * 0.92))
    if x1 <= x0:
        x0, x1 = int(width * 0.08), int(width * 0.95)
    if y1 <= y0:
        y0, y1 = int(height * 0.08), int(height * 0.92)
    return max(0, x0 - 4), max(0, y0 - 4), min(width - 1, x1 + 4), min(height - 1, y1 + 4)


def foreground_mask(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    roi = image[y0 : y1 + 1, x0 : x1 + 1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    colored = (hsv[:, :, 1] > 35) & (hsv[:, :, 2] < 248) & (gray < 245)
    dark = gray < 130
    mask = (colored | dark).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    full = np.zeros(image.shape[:2], dtype=np.uint8)
    full[y0 : y1 + 1, x0 : x1 + 1] = mask
    return full


def point_foreground_mask(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    roi = image[y0 : y1 + 1, x0 : x1 + 1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    colored = hsv[:, :, 1] > 30
    dark = (gray < 95) & (hsv[:, :, 1] > 18)
    mask = (colored | dark).astype(np.uint8) * 255
    full = np.zeros(image.shape[:2], dtype=np.uint8)
    full[y0 : y1 + 1, x0 : x1 + 1] = mask
    return full


def cv_point_predictions(data: Dict[str, Any], image_path: Path, chart_type: str) -> list[Dict[str, Any]]:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]
    bounds = plot_bounds_from_ticks(data, width, height)
    mask = point_foreground_mask(image, bounds)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    predictions: list[Dict[str, Any]] = []
    min_area = 4 if chart_type == "scatter" else 18
    max_area = width * height * (0.012 if chart_type == "scatter" else 0.08)
    max_side_ratio = 0.08 if chart_type == "scatter" else 0.22
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if w > width * max_side_ratio or h > height * max_side_ratio:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > (3.0 if chart_type == "scatter" else 4.5):
            continue
        cx, cy = centroids[label]
        x_value = pixel_to_data_value(cx, data.get("x_ticks"), data.get("x_pixels"))
        y_value = pixel_to_data_value(cy, data.get("y_ticks"), data.get("y_pixels"))
        if x_value is None or y_value is None:
            continue
        index = len(predictions) + 1
        predictions.append(
            {
                "id": f"{chart_type}_{index}",
                "label": f"{chart_type}_{index}",
                "axis": "xy",
                "x": round(x_value, 4),
                "y": round(y_value, 4),
                "value": {"x": round(x_value, 4), "y": round(y_value, 4)},
                "pixel": {"x": round(float(cx), 2), "y": round(float(cy), 2)},
                "area": int(area),
                "source": "system_cv_fallback",
            }
        )
    if not predictions:
        mask = foreground_mask(image, bounds)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < 6 or area > max_area or w > width * max_side_ratio or h > height * max_side_ratio:
                continue
            cx, cy = centroids[label]
            x_value = pixel_to_data_value(cx, data.get("x_ticks"), data.get("x_pixels"))
            y_value = pixel_to_data_value(cy, data.get("y_ticks"), data.get("y_pixels"))
            if x_value is None or y_value is None:
                continue
            index = len(predictions) + 1
            predictions.append(
                {
                    "id": f"{chart_type}_{index}",
                    "label": f"{chart_type}_{index}",
                    "axis": "xy",
                    "x": round(x_value, 4),
                    "y": round(y_value, 4),
                    "value": {"x": round(x_value, 4), "y": round(y_value, 4)},
                    "pixel": {"x": round(float(cx), 2), "y": round(float(cy), 2)},
                    "area": int(area),
                    "source": "system_cv_fallback",
                }
            )
    predictions.sort(key=lambda item: (item["pixel"]["x"], item["pixel"]["y"]))
    return predictions[:200]


def cv_bar_predictions(data: Dict[str, Any], image_path: Path, chart_type: str) -> list[Dict[str, Any]]:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]
    mask = foreground_mask(image, plot_bounds_from_ticks(data, width, height))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    base = bar_base_type(chart_type)
    x_numeric = len(numeric_tick_pairs(data.get("x_ticks"), data.get("x_pixels"))) >= 2
    y_numeric = len(numeric_tick_pairs(data.get("y_ticks"), data.get("y_pixels"))) >= 2
    if x_numeric and not y_numeric:
        base = "h_bar"
    elif y_numeric and not x_numeric:
        base = "v_bar"
    predictions: list[Dict[str, Any]] = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 40:
            continue
        if base == "h_bar":
            if w < 8 or h < 4:
                continue
            value = pixel_to_data_value(x + w, data.get("x_ticks"), data.get("x_pixels"))
            label_text = nearest_label(y + h / 2, data.get("y_ticks"), data.get("y_pixels"))
            axis = "x"
        else:
            if h < 8 or w < 4:
                continue
            value = pixel_to_data_value(y, data.get("y_ticks"), data.get("y_pixels"))
            label_text = nearest_label(x + w / 2, data.get("x_ticks"), data.get("x_pixels"))
            axis = "y"
        if value is None:
            continue
        predictions.append(
            {
                "id": f"bar_{len(predictions) + 1}",
                "label": label_text,
                "axis": axis,
                "value": round(value, 4),
                "pixel": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "source": "system_cv_fallback",
            }
        )
    predictions.sort(key=lambda item: (item["label"], item["pixel"]["x"], item["pixel"]["y"]))
    return predictions[:200]


def cv_line_predictions(data: Dict[str, Any], image_path: Path) -> list[Dict[str, Any]]:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return []
    height, width = image.shape[:2]
    mask = foreground_mask(image, plot_bounds_from_ticks(data, width, height))
    predictions: list[Dict[str, Any]] = []
    x_ticks = data.get("x_ticks") if isinstance(data.get("x_ticks"), list) else []
    x_pixels = data.get("x_pixels") if isinstance(data.get("x_pixels"), list) else []
    for tick, pixel in zip(x_ticks, x_pixels):
        try:
            x = int(round(float(pixel)))
        except (TypeError, ValueError):
            continue
        column = mask[:, max(0, x - 2) : min(width, x + 3)]
        ys = np.where(column > 0)[0]
        if len(ys) == 0:
            continue
        y = float(np.median(ys))
        y_value = pixel_to_data_value(y, data.get("y_ticks"), data.get("y_pixels"))
        if y_value is None:
            continue
        predictions.append(
            {
                "id": f"line_{len(predictions) + 1}",
                "label": str(tick),
                "axis": "y",
                "value": round(y_value, 4),
                "x": tick,
                "pixel": {"x": x, "y": round(y, 2)},
                "source": "system_cv_fallback",
            }
        )
    return predictions


def system_cv_predictions(chart_info: Dict[str, Any], eval_json_path: Path) -> list[Dict[str, Any]]:
    data = processed_json_payload(eval_json_path, chart_info["chart_type"])
    image_path = Path(chart_info["image_path"])
    chart_type = str(chart_info["chart_type"])
    try:
        if chart_type in {"scatter", "bubble"}:
            return cv_point_predictions(data, image_path, chart_type)
        if is_bar_type(chart_type):
            return cv_bar_predictions(data, image_path, chart_type)
        if chart_type == "line":
            return cv_line_predictions(data, image_path)
    except Exception as error:
        print(f"System CV prediction fallback failed: {safe_error_message(error)}")
    return []


def prediction_has_concrete_value(prediction: Dict[str, Any]) -> bool:
    value = prediction.get("value")
    if isinstance(value, dict):
        return any(prediction_has_concrete_value({"value": item}) for item in value.values())
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(numeric)


async def evaluate_processed_chart(chart_info: Dict[str, Any]) -> Path:
    if not chart_info.get("processed"):
        raise HTTPException(status_code=400, detail="Please process the chart first")

    dataset_sample = chart_info.get("dataset_sample") if isinstance(chart_info.get("dataset_sample"), dict) else {}
    dataset_eval_path = None
    if chart_info.get("dataset_preview") and dataset_sample.get("sample_id"):
        dataset_eval_path = dataset_evaluation_cache_path(str(dataset_sample["sample_id"]))
        if evaluation_cache_usable(dataset_eval_path, chart_info["chart_type"]):
            chart_info.update({"evaluated": True, "evaluation_results_path": str(dataset_eval_path)})
            return dataset_eval_path

    if (
        chart_info["chart_type"] not in SUPPORTED_PREDICTION_TYPES
        and chart_info.get("evaluated")
        and chart_info.get("evaluation_results_path")
    ):
        return Path(chart_info["evaluation_results_path"])

    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    eval_json_path = resolve_eval_json(chart_info)

    if chart_info["chart_type"] in SUPPORTED_PREDICTION_TYPES:
        prediction_results: list[Dict[str, Any]] = []
        prediction_error: Optional[str] = None
        processed_payload = processed_json_payload(eval_json_path, chart_info["chart_type"])
        try:
            prediction_results = await run_prediction_async(chart_info["chart_type"], eval_json_path)
        except Exception as error:
            prediction_error = safe_error_message(error)
        predictions = [
            prediction
            for chart_result in prediction_results
            for prediction in chart_result.get("predictions", [])
            if isinstance(prediction, dict) and prediction_has_concrete_value(prediction)
        ]
        used_system_cv_fallback = False
        if not predictions:
            predictions = system_cv_predictions(chart_info, eval_json_path)
            used_system_cv_fallback = bool(predictions)
        evaluation_results = {
            "success": True,
            "mode": "prediction_extraction",
            "chart_id": chart_info["chart_id"],
            "chart_type": chart_info["chart_type"],
            "source_json": str(eval_json_path),
            "system_json": str(eval_json_path),
            "summary": {
                "object_count": len(predictions),
                "chart_runs": len(prediction_results),
                "system_cv_fallback": used_system_cv_fallback,
            },
            "predictions": predictions,
            "artifacts": prediction_results,
            "processed_json": processed_payload,
        }
        if prediction_error:
            evaluation_results["prediction_runner_error"] = prediction_error
        if used_system_cv_fallback:
            evaluation_results["note"] = (
                "Prediction runner had no usable system-generated targets/results; "
                "returned CV fallback predictions from generated ticks and image pixels. "
                "Ground-truth dataset JSON is intentionally not used."
            )
        elif not prediction_results:
            evaluation_results["note"] = (
                "Prediction runner received no system-generated targets. "
                "Ground-truth dataset JSON is intentionally not used."
            )
    else:
        evaluation_results = build_extraction_placeholder(chart_info, eval_json_path)

    results_path = dataset_eval_path or (RESULTS_DIR / f"{chart_info['chart_id']}_evaluation.json")
    processor.save_evaluation_results(evaluation_results, str(results_path))

    chart_info.update({"evaluated": True, "evaluation_results_path": str(results_path)})
    return results_path


def find_file(filename: str, roots: Iterable[Path]) -> Optional[Path]:
    for root in roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
        if root.exists() and root in {
            OUTPUT_DIR,
            DATASET_PREVIEW_CACHE_DIR,
            GT_EXPERIMENT_RESULTS_DIR,
            LEGACY_GT_EXPERIMENT_RESULTS_DIR,
        }:
            for nested_candidate in root.rglob(filename):
                if nested_candidate.is_file():
                    return nested_candidate
    return None


@app.get("/")
async def root():
    return {"message": "Chart Analysis API", "version": "1.0.0"}


@app.post("/api/upload/")
async def upload_files(
    file: UploadFile = File(..., description="Chart image file"),
):
    ensure_legacy_flow_enabled()
    try:
        chart_info = register_chart(file)
        return {
            "chart_id": chart_info["chart_id"],
            "chart_type": chart_info["chart_type"],
            "coordinate_system": chart_info["coordinate_system"],
            "confidence": chart_info["confidence"],
            "axis_repair": chart_info.get("axis_repair", {}),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Upload failed: {safe_error_message(error)}")


@app.get("/api/gt-experiment/categories/")
async def list_gt_experiment_categories(
    source: str = Query("realworld", description="Dataset source: realworld or synthetic"),
):
    try:
        normalized_source = source if source in DATASET_SOURCE_ROOTS else "realworld"
        return {"source": normalized_source, "categories": gt_experiment_category_options(normalized_source)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"List GT experiment categories failed: {safe_error_message(error)}")


@app.get("/api/gt-experiment/samples/")
async def list_gt_experiment_samples(
    source: str = Query("realworld", description="Dataset source: realworld or synthetic"),
    category: Optional[str] = Query(None, description="Chart category to load lazily"),
    limit: int = Query(200, ge=1, le=500),
):
    try:
        normalized_source = source if source in DATASET_SOURCE_ROOTS else "realworld"
        categories = gt_experiment_category_options(normalized_source)
        allowed_categories = {item["value"] for item in categories}
        normalized_category = category if category in allowed_categories else (categories[0]["value"] if categories else None)
        samples = [
            enrich_gt_experiment_sample(sample)
            for sample in list(iter_dataset_samples(normalized_source, normalized_category))[:limit]
        ]
        return {
            "source": normalized_source,
            "category": normalized_category,
            "categories": categories,
            "samples": samples,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"List GT experiment samples failed: {safe_error_message(error)}")


@app.get("/api/gt-experiment/image/{sample_id}/{image_type}")
async def get_gt_experiment_image(sample_id: str, image_type: str):
    config_path = resolve_gt_config_path(sample_id)
    resolved_type = "grid_with_grid" if image_type in {"grid", "with_grid", "grid_with_grid"} else "no_grid"
    image_path = resolve_gt_image_path(config_path, resolved_type)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="GT experiment image not found")
    return FileResponse(str(image_path))


@app.post("/api/gt-experiment/select/")
async def select_gt_experiment_sample(sample_id: str = Query(..., description="Dataset sample ID")):
    try:
        sample = enrich_gt_experiment_sample(dataset_sample_by_id(sample_id))
        if sample["chart_type"] not in SUPPORTED_PREDICTION_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported GT experiment chart type: {sample['chart_type']}")
        config_path = resolve_gt_config_path(sample_id)
        chart_id = f"gt_{sample_id}"
        chart_info = {
            "chart_id": chart_id,
            "chart_type": sample["chart_type"],
            "coordinate_system": sample["coordinate_system"],
            "confidence": 1.0,
            "image_path": str(resolve_gt_image_path(config_path, "no_grid")),
            "gt_grid_image_path": str(resolve_gt_image_path(config_path, "grid_with_grid")),
            "gt_config_path": str(config_path),
            "processed": True,
            "dataset_preview": True,
            "dataset_sample": sample,
        }
        charts_db[chart_id] = chart_info
        return {
            "chart_id": chart_id,
            "chart_type": chart_info["chart_type"],
            "coordinate_system": chart_info["coordinate_system"],
            "confidence": chart_info["confidence"],
            "dataset_sample": sample,
            "original_image_url": f"/api/gt-experiment/image/{sample_id}/original",
            "standard_grid_url": f"/api/gt-experiment/image/{sample_id}/grid",
            "gt_config_path": str(config_path),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Select GT experiment sample failed: {safe_error_message(error)}")


@app.post("/api/gt-experiment/run/")
async def run_gt_experiment(
    sample_id: str = Query(..., description="Dataset sample ID"),
):
    async with GT_EXPERIMENT_RUN_LOCK:
        return await _run_gt_experiment_locked(sample_id)


async def _run_gt_experiment_locked(sample_id: str):
    """Run one GT experiment with process-global output state isolated by the caller lock."""
    try:
        return await _run_gt_experiment_unlocked(sample_id)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Run GT experiment failed: {safe_error_message(error)}")


async def _run_gt_experiment_unlocked(sample_id: str):
    try:
        sample = dataset_sample_by_id(sample_id)
        config_path = resolve_gt_config_path(sample_id)
        expected_chart_id = str(load_json(config_path).get("chart_id") or Path(config_path).stem)
        chart_type = sample["chart_type"]
        if chart_type not in SUPPORTED_PREDICTION_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported GT experiment chart type: {chart_type}")
        chart_name = sample.get("name") or Path(config_path).stem
        category_dir = GT_EXPERIMENT_RESULTS_DIR / safe_path_fragment(chart_type)
        chart_dir = category_dir / safe_path_fragment(chart_name)
        model_name = get_model_name()
        run_id = f"{model_path_alias(model_name)}__{safe_path_fragment(Path(config_path).stem)}"
        run_dir = chart_dir / run_id
        process_files_dir = run_dir / "pf"
        if run_dir.exists():
            chart_dir_resolved = chart_dir.resolve()
            run_dir_resolved = run_dir.resolve()
            if chart_dir_resolved not in run_dir_resolved.parents:
                raise HTTPException(status_code=500, detail="Refuse to clear GT experiment directory outside chart folder")
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        process_files_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, run_dir / "gt_config.json")
        modal_log_path = run_dir / "modal_calls.jsonl"
        modal_full_log_path = run_dir / "modal_calls_full.jsonl"

        with temporary_env(
            {
                "CHART_EXPERIMENT_MODE": "gt",
                "CHART_EXPERIMENT_PRESERVE_GT": "1",
                "CHART_FEEDBACK_ROUNDS": "2",
                "CHART_AMPLIFIER_ROUNDS": "3",
                "CHART_BAR_AMPLIFIER_ROUNDS": "3",
                "CHART_PREDICTION_CONSISTENCY_TOLERANCE": "0.0025",
                "EVALUATION_PREDICTION_RESULTS_ROOT": str(process_files_dir),
                "GT_MODAL_CALL_LOG_PATH": str(modal_log_path),
                "GT_MODAL_FULL_LOG_PATH": str(modal_full_log_path),
            }
        ):
            prediction_results = await run_prediction_async(chart_type, config_path)

        unexpected_chart_ids = sorted(
            {
                str(chart_result.get("chart_id"))
                for chart_result in prediction_results
                if isinstance(chart_result, dict)
                and chart_result.get("chart_id") is not None
                and str(chart_result.get("chart_id")) != expected_chart_id
            }
        )
        if unexpected_chart_ids:
            write_json(
                run_dir / "contamination_guard.json",
                {
                    "expected_chart_id": expected_chart_id,
                    "unexpected_chart_ids": unexpected_chart_ids,
                    "reason": "Prediction results contained chart IDs outside the selected GT config.",
                },
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    "GT experiment output contamination guard triggered: "
                    f"expected {expected_chart_id}, got {unexpected_chart_ids}"
                ),
            )

        metrics = summarize_gt_prediction_records(
            config_path=config_path,
            prediction_results=prediction_results,
        )
        final_metrics = summarize_gt_prediction_records(
            config_path=config_path,
            prediction_results=prediction_results,
            result_csv_names=["full_flow_final_predictions.csv"],
        )
        if final_metrics["record_count"] == 0:
            final_metrics = summarize_gt_selected_predictions(
                config_path=config_path,
                prediction_results=prediction_results,
            )
        stage_coverage = summarize_stage_coverage(metrics["records"])
        metric_artifacts = write_metric_artifacts(run_dir, metrics, stage_coverage)
        enriched_logs = enrich_modal_call_logs(
            modal_log_path=modal_log_path,
            modal_full_log_path=modal_full_log_path,
            metric_records=metrics["records"],
        )
        modal_call_records_csv = write_modal_call_records_csv(
            enriched_logs.get("modal_call_log_enriched"),
            run_dir / "modal_call_records.csv",
        )
        payload = {
            "success": True,
            "mode": "gt_experiment_prediction",
            "chart_id": sample.get("name"),
            "chart_type": chart_type,
            "model_name": model_name,
            "model_vision_profile": active_model_vision_profile().__dict__,
            "experiment_contract": (
                CONTRACTS[chart_type].__dict__
                if chart_type in CONTRACTS
                else None
            ),
            "sample_id": sample_id,
            "gt_config_path": str(config_path),
            "category_dir": str(category_dir),
            "chart_dir": str(chart_dir),
            "chart_run_dir": str(run_dir),
            "process_files_dir": str(process_files_dir),
            "run_dir": str(run_dir),
            "modal_call_log": str(modal_log_path),
            "modal_full_log": str(modal_full_log_path),
            **enriched_logs,
            "modal_call_records_csv": modal_call_records_csv,
            **metric_artifacts,
            "summary": {
                "chart_runs": len(prediction_results),
                "record_count": metrics["record_count"],
                "avg_RE": metrics["avg_RE"],
                "avg_RNE": metrics["avg_RNE"],
                "full_flow_final_avg_RE": final_metrics["avg_RE"],
                "full_flow_final_avg_RNE": final_metrics["avg_RNE"],
                "full_flow_final_record_count": final_metrics["record_count"],
                "grid_rounds": 1,
                "feedback_rounds": 2,
                "amplifier_rounds": 3,
                "missing_stage_object_count": len(stage_coverage.get("missing_stage_objects", [])),
                "missing_valid_full_flow_object_count": len(stage_coverage.get("missing_valid_full_flow_objects", [])),
                "stage_call_violation_object_count": len(stage_coverage.get("stage_call_violation_objects", [])),
            },
            "prediction_results": prediction_results,
            "gt_metrics": metrics,
            "full_flow_final_metrics": final_metrics,
            "stage_coverage": stage_coverage,
            "predictions": [
                prediction
                for chart_result in prediction_results
                for prediction in chart_result.get("predictions", [])
                if isinstance(prediction, dict)
            ],
            "processed_json": load_json(config_path),
        }
        result_path = run_dir / f"{run_id}_gt_experiment_result.json"
        write_json(result_path, payload)
        return sanitize_json_value({"results_url": result_response_url(result_path), "result_path": str(result_path), **payload})
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Run GT experiment failed: {safe_error_message(error)}")

@app.get("/api/dataset-preview/samples/")
async def list_dataset_preview_samples(
    source: str = Query("realworld", description="Dataset source: realworld or synthetic"),
    category: Optional[str] = Query(None, description="Chart category to load lazily"),
    limit: int = Query(36, ge=1, le=200),
):
    ensure_legacy_flow_enabled()
    try:
        normalized_source = source if source in DATASET_SOURCE_ROOTS else "realworld"
        categories = dataset_category_options(normalized_source)
        allowed_categories = {item["value"] for item in categories}
        normalized_category = category if category in allowed_categories else (categories[0]["value"] if categories else None)
        samples = list(iter_dataset_samples(normalized_source, normalized_category))[:limit]
        return {
            "source": normalized_source,
            "category": normalized_category,
            "categories": categories,
            "samples": samples,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"List dataset samples failed: {safe_error_message(error)}")


@app.get("/api/dataset-preview/categories/")
async def list_dataset_preview_categories(
    source: str = Query("realworld", description="Dataset source: realworld or synthetic"),
):
    ensure_legacy_flow_enabled()
    try:
        normalized_source = source if source in DATASET_SOURCE_ROOTS else "realworld"
        return {"source": normalized_source, "categories": dataset_category_options(normalized_source)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"List dataset categories failed: {safe_error_message(error)}")


@app.get("/api/dataset-preview/image/{sample_id}")
async def get_dataset_preview_image(sample_id: str):
    ensure_legacy_flow_enabled()
    sample = dataset_sample_by_id(sample_id)
    image_path = Path(sample["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Dataset sample image not found")
    return FileResponse(str(image_path))


@app.post("/api/dataset-preview/select/")
async def select_dataset_preview_sample(sample_id: str = Query(..., description="Dataset sample ID")):
    ensure_legacy_flow_enabled()
    try:
        chart_info = register_dataset_sample(sample_id)
        response = {
            "chart_id": chart_info["chart_id"],
            "chart_type": chart_info["chart_type"],
            "coordinate_system": chart_info["coordinate_system"],
            "confidence": chart_info["confidence"],
            "axis_repair": chart_info.get("axis_repair", {}),
            "dataset_sample": chart_info.get("dataset_sample", {}),
            "original_image_url": image_response_url(chart_info["image_path"]),
            "processed": chart_info.get("processed", False),
            "cached": chart_info.get("processed", False),
        }
        if chart_info.get("processed") and chart_info.get("encrypted_image_path"):
            response["encrypted_image_url"] = image_response_url(chart_info["encrypted_image_path"])
            response["standard_grid_url"] = image_response_url(chart_info["encrypted_image_path"])
        colored_image_path = chart_info.get("colored_image_path")
        if colored_image_path and Path(colored_image_path).exists():
            response["colored_grid_url"] = image_response_url(colored_image_path)
        if chart_info.get("evaluated") and chart_info.get("evaluation_results_path"):
            response["evaluated"] = True
            response["results_url"] = result_response_url(chart_info["evaluation_results_path"])
        return response
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Select dataset sample failed: {safe_error_message(error)}")


@app.post("/api/process/")
async def process_chart(chart_id: str = Query(..., description="Chart ID")):
    ensure_legacy_flow_enabled()
    try:
        chart_info = get_chart(chart_id)
        encrypted_image_path = process_chart_image(chart_info, force=not chart_info.get("dataset_preview"))
        colored_image_path = chart_info.get("colored_image_path")
        if not colored_image_path:
            output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
            tick_sidecar = output_dir / f"{Path(chart_info['image_path']).stem}_ticks.json"
            if tick_sidecar.exists():
                colored_image_path = load_json(tick_sidecar).get("colored_grid_path")
        response = {
            "encrypted_image_url": image_response_url(encrypted_image_path),
            "standard_grid_url": image_response_url(encrypted_image_path),
        }
        if colored_image_path and Path(colored_image_path).exists():
            response["colored_grid_url"] = image_response_url(colored_image_path)
        return response
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Process failed: {safe_error_message(error)}")


@app.post("/api/evaluate/")
async def evaluate_chart(chart_id: str = Query(..., description="Chart ID")):
    ensure_legacy_flow_enabled()
    try:
        results_path = await evaluate_processed_chart(get_chart(chart_id))
        return {"results_url": result_response_url(results_path)}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Evaluate failed: {safe_error_message(error)}")


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    roots = [PROCESSED_DIR, DATASET_PREVIEW_CACHE_DIR]
    roots.extend(path for path in OUTPUT_DIR.iterdir() if path.is_dir())
    roots.append(UPLOAD_DIR)

    image_path = find_file(filename, roots)
    if image_path:
        return FileResponse(
            str(image_path),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/results/{filename}")
async def get_results(filename: str):
    results_path = RESULTS_DIR / filename
    if not results_path.exists():
        nested_path = find_file(
            filename,
            [DATASET_PREVIEW_CACHE_DIR, GT_EXPERIMENT_RESULTS_DIR, LEGACY_GT_EXPERIMENT_RESULTS_DIR],
        )
        if nested_path:
            results_path = nested_path
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    payload = normalize_result_payload(load_json(results_path), results_path)
    return JSONResponse(
        content=payload,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
