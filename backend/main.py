import io
import json
import os
import shutil
import sys
import uuid
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

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
sys.path.insert(0, str(BACKEND_DIR))

from type_detection.chart_processor import ChartProcessorFactory  # noqa: E402
from type_detection.chart_registry import DEFAULT_CHART_TYPE, get_coordinate_system, normalize_chart_type  # noqa: E402
from type_detection.chart_type import ChartTypeDetector  # noqa: E402
from evaluation_prediction.service import SUPPORTED_PREDICTION_TYPES, run_prediction_async  # noqa: E402


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

for directory in [UPLOAD_DIR, PROCESSED_DIR, RESULTS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

charts_db: Dict[str, Dict[str, Any]] = {}


def safe_error_message(error: Exception) -> str:
    try:
        return str(error).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    except Exception:
        return "Unexpected server error"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
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
        detected_type = normalize_chart_type(detection.get("type", DEFAULT_CHART_TYPE))
        if detected_type in {"h_bar", "v_bar"}:
            geometry_type = infer_bar_orientation_from_image(image_path)
            if geometry_type and geometry_type != detected_type:
                detection["type"] = geometry_type
                detection["geometry_type_override"] = {
                    "from": detected_type,
                    "to": geometry_type,
                    "reason": "colored bar geometry orientation",
                }
        return detection
    except Exception as error:
        print(f"Chart type detection failed, using fallback: {safe_error_message(error)}")
        return {
            "type": DEFAULT_CHART_TYPE,
            "confidence": 0.5,
            "axis_repair": {
                "x_axis_missing": False,
                "y_axis_missing": False,
                "x_ticks_missing": False,
                "y_ticks_missing": False,
                "confidence": 0.0,
                "reason": "type detection fallback",
            },
            "error": safe_error_message(error),
        }


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


def register_chart(image_file: UploadFile, json_file: Optional[UploadFile] = None) -> Dict[str, Any]:
    chart_id = str(uuid.uuid4())
    image_suffix = Path(image_file.filename or "").suffix or ".png"
    image_path = UPLOAD_DIR / f"{chart_id}_image{image_suffix}"
    json_path = UPLOAD_DIR / f"{chart_id}_data.json" if json_file else None

    save_upload_file(image_file, image_path)
    if json_file and json_path:
        save_upload_file(json_file, json_path)

    detection = detect_chart_type(image_path)
    chart_type = normalize_chart_type(detection.get("type", DEFAULT_CHART_TYPE))
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
    return f"/api/images/{Path(path).name}"


def result_response_url(path: Union[str, Path]) -> str:
    return f"/api/results/{Path(path).name}"


PREFERRED_EXTRACTION_PROMPTS = ("amplifier", "feedback", "grid", "baseline")


def strip_external_reference_data(
    data: Dict[str, Any],
    *,
    preserve_data: bool = False,
    preserve_series_color: bool = False,
) -> None:
    """Remove ground-truth/reference fields from user-upload processing data."""
    data.pop("reference_config_path", None)
    data.pop("reference_chart_id", None)

    keys = ["data_points", "ground_truth", "labels"]
    if not preserve_series_color:
        keys.append("series_color")
    if not preserve_data:
        keys.append("data")
    for key in keys:
        data.pop(key, None)


def processed_json_payload(eval_json_path: Union[str, Path], chart_type: Optional[str] = None) -> Dict[str, Any]:
    path = Path(eval_json_path)
    data = load_json(path)
    if not path.stem.endswith("_ticks"):
        merge_tick_sidecar(data, path.parent, path.stem)
    strip_external_reference_data(
        data,
        preserve_data=chart_type in {"pie", "donut"},
        preserve_series_color=chart_type in {"radar", "rose"},
    )
    return data


def generated_json_path(chart_info: Dict[str, Any], output_dir: Path) -> Path:
    return output_dir / f"{Path(chart_info['image_path']).stem}.json"


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
        preserve_data=chart_info["chart_type"] in {"pie", "donut"},
        preserve_series_color=chart_info["chart_type"] in {"radar", "rose"},
    )

    generated_data.update(
        {
            "chart_id": chart_info["chart_id"],
            "chart_type": chart_info["chart_type"],
            "coordinate_system": chart_info["coordinate_system"],
            "image_paths": {
                "no_grid": str(Path(chart_info["image_path"]).absolute()),
                "with_grid": str(Path(encrypted_image_path).absolute()),
            },
        }
    )

    write_json(json_path, generated_data)
    return json_path


def merge_tick_sidecar(data: Dict[str, Any], output_dir: Path, image_stem: str) -> None:
    ticks_json_path = output_dir / f"{image_stem}_ticks.json"
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
        basic_grid_path = ticks_data.get("basic_grid_path")
        if encrypted_grid_path:
            image_paths["grid_with_grid"] = str(Path(encrypted_grid_path).absolute())
            image_paths["with_grid"] = str(Path(encrypted_grid_path).absolute())
        if basic_grid_path:
            image_paths["basic_grid"] = str(Path(basic_grid_path).absolute())


def save_axis_data(chart_info: Dict[str, Any], output_dir: Path) -> None:
    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    try:
        axis_data = processor.find_axis(chart_info["image_path"], axis_repair_hint=chart_info.get("axis_repair"))
    except Exception as error:
        print(f"Axis detection failed: {safe_error_message(error)}")
        return

    if axis_data:
        write_json(output_dir / f"{chart_info['chart_id']}_axes.json", axis_data, indent=4)


def process_chart_image(chart_info: Dict[str, Any]) -> str:
    if chart_info.get("processed") and chart_info.get("encrypted_image_path"):
        return chart_info["encrypted_image_path"]

    output_dir = get_chart_output_dir(chart_info["chart_type"])
    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    encrypted_image_path = processor.encode_image(
        chart_info["image_path"],
        str(output_dir),
        axis_repair_hint=chart_info.get("axis_repair"),
    )

    if not encrypted_image_path and chart_info["chart_type"] in {"h_bar", "v_bar"}:
        alternate_type = "v_bar" if chart_info["chart_type"] == "h_bar" else "h_bar"
        alternate_output_dir = get_chart_output_dir(alternate_type)
        alternate_processor = ChartProcessorFactory.create_processor(alternate_type)
        alternate_image_path = alternate_processor.encode_image(
            chart_info["image_path"],
            str(alternate_output_dir),
            axis_repair_hint=chart_info.get("axis_repair"),
        )
        if alternate_image_path:
            print(f"Bar processing fallback succeeded: {chart_info['chart_type']} -> {alternate_type}")
            chart_info["chart_type"] = alternate_type
            chart_info["coordinate_system"] = get_coordinate_system(alternate_type).value
            output_dir = alternate_output_dir
            processor = alternate_processor
            encrypted_image_path = alternate_image_path

    if not encrypted_image_path:
        raise HTTPException(status_code=500, detail="Chart processing failed")

    save_axis_data(chart_info, output_dir)
    enrich_generated_json(chart_info, output_dir, encrypted_image_path)

    chart_info.update(
        {
            "processed": True,
            "encrypted_image_path": encrypted_image_path,
            "output_dir": str(output_dir),
        }
    )
    return encrypted_image_path


def candidate_eval_json_paths(chart_info: Dict[str, Any]) -> Iterable[Path]:
    output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
    image_stem = Path(chart_info["image_path"]).stem
    yield output_dir / f"{image_stem}.json"
    yield output_dir / f"{chart_info['chart_id']}.json"


def resolve_eval_json(chart_info: Dict[str, Any]) -> Path:
    for path in candidate_eval_json_paths(chart_info):
        if path.exists():
            output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
            data = load_json(path)
            merge_tick_sidecar(data, output_dir, Path(chart_info["image_path"]).stem)
            strip_external_reference_data(
                data,
                preserve_data=chart_info["chart_type"] in {"pie", "donut"},
                preserve_series_color=chart_info["chart_type"] in {"radar", "rose"},
            )
            write_json(path, data)
            return path

    output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    processed_data = processor.process_data(
        chart_info["chart_id"],
        chart_info["image_path"],
        None,
        str(output_dir),
    )

    if processed_data is None:
        available_files = [file.name for file in output_dir.glob("*.json")]
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate evaluation data. Available JSON files: {available_files}",
        )

    fallback_path = output_dir / f"{chart_info['chart_id']}.json"
    write_json(fallback_path, processed_data)
    return fallback_path


def build_extraction_placeholder(chart_info: Dict[str, Any], eval_json_path: Path) -> Dict[str, Any]:
    data = processed_json_payload(eval_json_path, chart_info["chart_type"])
    x_ticks = data.get("x_ticks", [])
    y_ticks = data.get("y_ticks", [])
    r_ticks = data.get("r_ticks", [])
    theta_ticks = data.get("theta_ticks", [])
    predictions = data.get("predictions") if isinstance(data.get("predictions"), list) else []

    return {
        "success": True,
        "mode": "prediction_extraction",
        "chart_id": chart_info["chart_id"],
        "chart_type": chart_info["chart_type"],
        "source_json": str(eval_json_path),
        "summary": {
            "object_count": len(predictions),
            "chart_runs": 0,
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
        source_json = result.get("source_json")
        if (
            "processed_json" not in result
            and isinstance(source_json, str)
            and Path(source_json).exists()
        ):
            result["processed_json"] = processed_json_payload(source_json, result.get("chart_type"))
        return result

    chart_type = str(result.get("chart_type", ""))
    legacy_runs = result.get("prediction_results")
    predictions = extract_predictions_from_legacy_runs(legacy_runs, chart_type)
    normalized = {
        "success": result.get("success", True),
        "mode": "prediction_extraction",
        "chart_id": result.get("chart_id"),
        "chart_type": chart_type,
        "source_json": result.get("source_json"),
        "summary": {
            "object_count": len(predictions),
            "chart_runs": len(legacy_runs) if isinstance(legacy_runs, list) else 0,
        },
        "predictions": predictions,
        "artifacts": legacy_runs if isinstance(legacy_runs, list) else [],
        "legacy_mode": result.get("mode"),
    }
    source_json = result.get("source_json")
    if isinstance(source_json, str) and Path(source_json).exists():
        normalized["processed_json"] = processed_json_payload(source_json, chart_type)
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
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
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
                "label": chosen.get("pred_y") if chart_type == "h_bar" else chosen.get("pred_x"),
                "axis": "x" if chart_type == "h_bar" else "y",
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
        if chart_type == "h_bar"
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
    key = "pred_x" if chart_type == "h_bar" else "pred_y"
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


async def evaluate_processed_chart(chart_info: Dict[str, Any]) -> Path:
    if not chart_info.get("processed"):
        raise HTTPException(status_code=400, detail="Please process the chart first")

    if (
        chart_info["chart_type"] not in SUPPORTED_PREDICTION_TYPES
        and chart_info.get("evaluated")
        and chart_info.get("evaluation_results_path")
    ):
        return Path(chart_info["evaluation_results_path"])

    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    eval_json_path = resolve_eval_json(chart_info)

    if chart_info["chart_type"] in SUPPORTED_PREDICTION_TYPES:
        prediction_results = await run_prediction_async(chart_info["chart_type"], eval_json_path)
        predictions = [
            prediction
            for chart_result in prediction_results
            for prediction in chart_result.get("predictions", [])
        ]
        evaluation_results = {
            "success": True,
            "mode": "prediction_extraction",
            "chart_id": chart_info["chart_id"],
            "chart_type": chart_info["chart_type"],
            "source_json": str(eval_json_path),
            "summary": {
                "object_count": len(predictions),
                "chart_runs": len(prediction_results),
            },
            "predictions": predictions,
            "artifacts": prediction_results,
            "processed_json": processed_json_payload(eval_json_path, chart_info["chart_type"]),
        }
    else:
        evaluation_results = build_extraction_placeholder(chart_info, eval_json_path)

    results_path = RESULTS_DIR / f"{chart_info['chart_id']}_evaluation.json"
    processor.save_evaluation_results(evaluation_results, str(results_path))

    chart_info.update({"evaluated": True, "evaluation_results_path": str(results_path)})
    return results_path


def find_file(filename: str, roots: Iterable[Path]) -> Optional[Path]:
    for root in roots:
        candidate = root / filename
        if candidate.exists():
            return candidate
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


@app.post("/api/process/")
async def process_chart(chart_id: str = Query(..., description="Chart ID")):
    try:
        encrypted_image_path = process_chart_image(get_chart(chart_id))
        return {"encrypted_image_url": image_response_url(encrypted_image_path)}
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
    roots = [PROCESSED_DIR]
    roots.extend(path for path in OUTPUT_DIR.iterdir() if path.is_dir())
    roots.append(UPLOAD_DIR)

    image_path = find_file(filename, roots)
    if image_path:
        return FileResponse(str(image_path))

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/results/{filename}")
async def get_results(filename: str):
    results_path = RESULTS_DIR / filename
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    payload = normalize_result_payload(load_json(results_path), results_path)
    return JSONResponse(content=payload)


if __name__ == "__main__":
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
