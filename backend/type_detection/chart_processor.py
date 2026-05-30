import json
import os
import shutil
from typing import Any, Dict, Optional, Protocol, Type

from demo_radar.demo_axis_find_radar import RadarChartAxisFinder
from demo_radar.demo_radar_circle_find import RadarChartEncoder
from demo_rose.demo_axis_find_rose import RoseChartAxisFinder
from demo_rose.demo_rose_circle_find import RoseChartEncoder
from evaluation import evaluate_chart_data
from Grid_generation.circular_angle_grid import process_circular_angle_chart
from Grid_generation.grid_generation import process_chart
from type_detection.chart_registry import (
    CARTESIAN_CHART_TYPES,
    CoordinateSystem,
    SUPPORTED_CHART_TYPES,
    get_coordinate_system,
    normalize_chart_type,
)


class ChartProcessor(Protocol):
    def encode_image(self, image_path: str, output_dir: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Optional[str]:
        ...

    def find_axis(self, image_path: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    def process_data(self, chart_id: str, image_path: str, json_path: Optional[str], output_dir: str) -> Optional[Dict[str, Any]]:
        ...

    def evaluate(self, eval_data_path: str) -> Dict[str, Any]:
        ...

    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        ...


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {"data": data}


def dump_json(path: str, data: Dict[str, Any], indent: int = 4) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


def extract_data_points(original_data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(original_data.get("data_points"), dict):
        return original_data["data_points"]
    if isinstance(original_data.get("data"), dict):
        return original_data["data"]
    return {}


class BaseChartProcessor:
    chart_type = "chart"
    coordinate_system = "unknown"

    def __init__(self, chart_type: Optional[str] = None):
        if chart_type:
            self.chart_type = chart_type
            self.coordinate_system = get_coordinate_system(chart_type).value

    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        dump_json(output_path, results)

    def evaluate(self, eval_data_path: str) -> Dict[str, Any]:
        return evaluate_chart_data(load_json(eval_data_path))


class PolarChartProcessor(BaseChartProcessor):
    coordinate_system = CoordinateSystem.POLAR.value
    encoder_cls: Type[Any]
    axis_finder_cls: Type[Any]

    def encode_image(self, image_path: str, output_dir: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Optional[str]:
        return self.encoder_cls().process_single_image(image_path, output_dir)

    def find_axis(self, image_path: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.axis_finder_cls().process_single_image(image_path)

    def process_data(self, chart_id: str, image_path: str, json_path: Optional[str], output_dir: str) -> Optional[Dict[str, Any]]:
        target_json_path = os.path.join(output_dir, f"{chart_id}.json")
        if not os.path.exists(target_json_path):
            return None

        chart_data = load_json(target_json_path)
        chart_data["chart_id"] = chart_id
        chart_data["chart_type"] = self.chart_type
        chart_data["coordinate_system"] = self.coordinate_system
        if json_path and os.path.exists(json_path):
            chart_data["data"] = extract_data_points(load_json(json_path))
        return chart_data


class RoseChartProcessor(PolarChartProcessor):
    chart_type = "rose"
    encoder_cls = RoseChartEncoder
    axis_finder_cls = RoseChartAxisFinder


class RadarChartProcessor(PolarChartProcessor):
    chart_type = "radar"
    encoder_cls = RadarChartEncoder
    axis_finder_cls = RadarChartAxisFinder


class CircularAngleChartProcessor(BaseChartProcessor):
    coordinate_system = CoordinateSystem.POLAR.value

    def __init__(self, chart_type: Optional[str] = None):
        super().__init__(chart_type)
        self.coordinate_system = CoordinateSystem.POLAR.value

    def encode_image(self, image_path: str, output_dir: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Optional[str]:
        result = process_circular_angle_chart(image_path, output_dir, self.chart_type)
        return result.get("encrypted_grid_path")

    def find_axis(self, image_path: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        temp_output = os.path.join(os.path.dirname(image_path), "temp_output")
        os.makedirs(temp_output, exist_ok=True)
        try:
            result = process_circular_angle_chart(image_path, temp_output, self.chart_type)
            return {
                "center": result.get("center", []),
                "r_pixels": result.get("r_pixels"),
                "theta_ticks": result.get("theta_ticks", []),
                "theta_angles": result.get("theta_angles", []),
                "r_ticks": result.get("r_ticks", []),
            }
        finally:
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)

    def process_data(self, chart_id: str, image_path: str, json_path: Optional[str], output_dir: str) -> Optional[Dict[str, Any]]:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        target_json_path = os.path.join(output_dir, f"{image_stem}.json")
        if os.path.exists(target_json_path):
            chart_data = load_json(target_json_path)
        else:
            chart_data = process_circular_angle_chart(image_path, output_dir, self.chart_type)

        chart_data["chart_id"] = chart_id
        chart_data["chart_type"] = self.chart_type
        chart_data["coordinate_system"] = self.coordinate_system
        if json_path and os.path.exists(json_path):
            chart_data["data"] = extract_data_points(load_json(json_path))
        return chart_data


class PieChartProcessor(CircularAngleChartProcessor):
    chart_type = "pie"


class DonutChartProcessor(CircularAngleChartProcessor):
    chart_type = "donut"


class CartesianChartProcessor(BaseChartProcessor):
    chart_type = CoordinateSystem.CARTESIAN.value
    coordinate_system = CoordinateSystem.CARTESIAN.value

    def encode_image(self, image_path: str, output_dir: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Optional[str]:
        result = process_chart(
            image_path,
            output_dir,
            chart_type_override=self.chart_type,
            axis_repair_hint=axis_repair_hint,
        )
        if result:
            return result.get("encrypted_grid_path")
        return None

    def find_axis(self, image_path: str, axis_repair_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        temp_output = os.path.join(os.path.dirname(image_path), "temp_output")
        os.makedirs(temp_output, exist_ok=True)

        try:
            result = process_chart(
                image_path,
                temp_output,
                chart_type_override=self.chart_type,
                axis_repair_hint=axis_repair_hint,
            )
            if not result:
                return {}

            return {
                "x_ticks": result.get("x_ticks", []),
                "y_ticks": result.get("y_ticks", []),
                "x_axis_type": result.get("x_axis_type", "numeric"),
                "y_axis_type": result.get("y_axis_type", "numeric"),
            }
        finally:
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)

    def process_data(self, chart_id: str, image_path: str, json_path: Optional[str], output_dir: str) -> Optional[Dict[str, Any]]:
        result = process_chart(
            image_path,
            output_dir,
            chart_type_override=self.chart_type,
        )
        if not result:
            return None

        chart_data = {
            "chart_id": chart_id,
            "chart_type": self.chart_type,
            "coordinate_system": self.coordinate_system,
            "x_ticks": result.get("x_ticks", []),
            "y_ticks": result.get("y_ticks", []),
            "x_axis_type": result.get("x_axis_type", "numeric"),
            "y_axis_type": result.get("y_axis_type", "numeric"),
            "colors": result.get("colors", []),
            "image_path": result.get("image_path", ""),
            "basic_grid_path": result.get("basic_grid_path", ""),
            "encrypted_grid_path": result.get("encrypted_grid_path", ""),
        }

        if json_path and os.path.exists(json_path):
            chart_data["data"] = extract_data_points(load_json(json_path))

        return chart_data


class ChartProcessorFactory:
    _processors = {
        "rose": RoseChartProcessor,
        "radar": RadarChartProcessor,
        "pie": PieChartProcessor,
        "donut": DonutChartProcessor,
    }

    @classmethod
    def create_processor(cls, chart_type: str) -> ChartProcessor:
        normalized_type = normalize_chart_type(chart_type)
        processor_cls = cls._processors.get(normalized_type)
        if processor_cls is not None:
            return processor_cls(normalized_type)

        if normalized_type in CARTESIAN_CHART_TYPES:
            return CartesianChartProcessor(normalized_type)

        return CartesianChartProcessor(normalized_type)
