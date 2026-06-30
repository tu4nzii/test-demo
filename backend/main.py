import io
import json
import os
import shutil
import sys
import uuid
import csv
import hashlib
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
from demo_radar.color import RadarColorMatcher  # noqa: E402


app = FastAPI(title="Chart Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {"data": data}


def write_json(path: Path, data: Dict[str, Any], indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


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


def dataset_file_id(source: str, image_path: Path) -> str:
    root = DATASET_SOURCE_ROOTS[source]
    rel_path = image_path.resolve().relative_to(root.resolve()).as_posix()
    return hashlib.sha1(f"{source}:{rel_path}".encode("utf-8")).hexdigest()[:16]


def dataset_manifest_relative(source: str, relative_path: str) -> str:
    dataset_name = "Sy.Dataset" if source == "synthetic" else "Final-RealDataset"
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
    root = DATASET_SOURCE_ROOTS[source]
    if not root.exists():
        return []
    image_paths: list[Path] = []
    if source == "realworld":
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
    elif source == "synthetic":
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


PREFERRED_EXTRACTION_PROMPTS = ("geometry", "amplifier", "feedback", "grid", "baseline")


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
    return np.isfinite(numeric) and numeric != -1.0


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
        if root.exists() and root in {OUTPUT_DIR, DATASET_PREVIEW_CACHE_DIR}:
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


@app.get("/api/dataset-preview/samples/")
async def list_dataset_preview_samples(
    source: str = Query("realworld", description="Dataset source: realworld or synthetic"),
    category: Optional[str] = Query(None, description="Chart category to load lazily"),
    limit: int = Query(36, ge=1, le=200),
):
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
    try:
        normalized_source = source if source in DATASET_SOURCE_ROOTS else "realworld"
        return {"source": normalized_source, "categories": dataset_category_options(normalized_source)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"List dataset categories failed: {safe_error_message(error)}")


@app.get("/api/dataset-preview/image/{sample_id}")
async def get_dataset_preview_image(sample_id: str):
    sample = dataset_sample_by_id(sample_id)
    image_path = Path(sample["image_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Dataset sample image not found")
    return FileResponse(str(image_path))


@app.post("/api/dataset-preview/select/")
async def select_dataset_preview_sample(sample_id: str = Query(..., description="Dataset sample ID")):
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
        nested_path = find_file(filename, [DATASET_PREVIEW_CACHE_DIR])
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
