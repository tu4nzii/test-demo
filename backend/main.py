import io
import json
import os
import shutil
import sys
import uuid
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
        return ChartTypeDetector().detect_chart_type(str(image_path))
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


def extract_original_data(original_json: Dict[str, Any]) -> Any:
    if "data_points" in original_json:
        return original_json["data_points"]
    if "data" in original_json:
        return original_json["data"]
    return None


def generated_json_path(chart_info: Dict[str, Any], output_dir: Path) -> Path:
    return output_dir / f"{Path(chart_info['image_path']).stem}.json"


def enrich_generated_json(
    chart_info: Dict[str, Any],
    output_dir: Path,
    encrypted_image_path: Union[str, Path],
) -> Path:
    json_path = generated_json_path(chart_info, output_dir)
    generated_data = load_json(json_path) if json_path.exists() else {}
    source_json_path = chart_info.get("json_path")
    original_data = None
    if source_json_path:
        source_path = Path(source_json_path)
        if source_path.exists():
            original_data = extract_original_data(load_json(source_path))

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
    if original_data is not None:
        generated_data["data"] = original_data

    write_json(json_path, generated_data)
    return json_path


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
            return path

    output_dir = Path(chart_info.get("output_dir", OUTPUT_DIR / chart_info["chart_type"]))
    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    processed_data = processor.process_data(
        chart_info["chart_id"],
        chart_info["image_path"],
        chart_info.get("json_path"),
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


def evaluate_processed_chart(chart_info: Dict[str, Any]) -> Path:
    if not chart_info.get("processed"):
        raise HTTPException(status_code=400, detail="Please process the chart first")

    if chart_info.get("evaluated") and chart_info.get("evaluation_results_path"):
        return Path(chart_info["evaluation_results_path"])

    processor = ChartProcessorFactory.create_processor(chart_info["chart_type"])
    evaluation_results = processor.evaluate(str(resolve_eval_json(chart_info)))

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
    json_data: Optional[UploadFile] = File(None, description="Optional chart JSON data file"),
):
    try:
        chart_info = register_chart(file, json_data)
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
        results_path = evaluate_processed_chart(get_chart(chart_id))
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

    return JSONResponse(content=load_json(results_path))


if __name__ == "__main__":
    port = int(os.environ.get("BACKEND_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
