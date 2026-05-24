import json
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Protocol, Type

from demo_radar.demo_axis_find_radar import RadarChartAxisFinder
from demo_radar.demo_evaluation_radar import RadarChartEvaluator
from demo_radar.demo_radar_circle_find import RadarChartEncoder
from demo_rose.demo_axis_find_rose import RoseChartAxisFinder
from demo_rose.demo_evaluation_rose import RoseChartEvaluator
from demo_rose.demo_rose_circle_find import RoseChartEncoder
from Grid_generation.grid_generation import process_chart


SUPPORTED_CHART_TYPES = [
    "rose",
    "radar",
    "v_bar",
    "h_bar",
    "line",
    "scatter",
    "bubble",
    "donut",
    "pie",
]
CARTESIAN_CHART_TYPES = {"v_bar", "h_bar", "line", "scatter", "bubble", "donut", "pie"}


class ChartProcessor(Protocol):
    def encode_image(self, image_path: str, output_dir: str) -> Optional[str]:
        ...

    def find_axis(self, image_path: str) -> Dict[str, Any]:
        ...

    def process_data(self, chart_id: str, image_path: str, json_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
        ...

    def evaluate(self, eval_data_path: str) -> Dict[str, Any]:
        ...

    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        ...


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
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

    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        dump_json(output_path, results)


class PolarChartProcessor(BaseChartProcessor):
    encoder_cls: Type[Any]
    axis_finder_cls: Type[Any]
    evaluator_cls: Type[Any]

    def encode_image(self, image_path: str, output_dir: str) -> Optional[str]:
        return self.encoder_cls().process_single_image(image_path, output_dir)

    def find_axis(self, image_path: str) -> Dict[str, Any]:
        return self.axis_finder_cls().process_single_image(image_path)

    def process_data(self, chart_id: str, image_path: str, json_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
        target_json_path = os.path.join(output_dir, f"{chart_id}.json")
        if not os.path.exists(target_json_path):
            return None

        chart_data = load_json(target_json_path)
        chart_data["chart_id"] = chart_id
        chart_data["chart_type"] = self.chart_type
        chart_data["data"] = extract_data_points(load_json(json_path))
        return chart_data

    def evaluate(self, eval_data_path: str) -> Dict[str, Any]:
        evaluator = self.evaluator_cls()
        evaluator.process_single_image(eval_data_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as temp_file:
            temp_path = temp_file.name

        try:
            evaluator.save_results(temp_path)
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                return {"error": "Evaluation produced no results"}

            content = open(temp_path, "r", encoding="utf-8").read().strip()
            if not content:
                return {"error": "Evaluation result is empty"}
            return json.loads(content)
        except json.JSONDecodeError as error:
            return {"error": f"Evaluation JSON parse failed: {error}"}
        except Exception as error:
            return {"error": f"Evaluation failed: {error}"}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


class RoseChartProcessor(PolarChartProcessor):
    chart_type = "rose"
    encoder_cls = RoseChartEncoder
    axis_finder_cls = RoseChartAxisFinder
    evaluator_cls = RoseChartEvaluator


class RadarChartProcessor(PolarChartProcessor):
    chart_type = "radar"
    encoder_cls = RadarChartEncoder
    axis_finder_cls = RadarChartAxisFinder
    evaluator_cls = RadarChartEvaluator


class CartesianChartProcessor(BaseChartProcessor):
    chart_type = "cartesian"

    def encode_image(self, image_path: str, output_dir: str) -> Optional[str]:
        result = process_chart(image_path, output_dir)
        if result:
            return result.get("encrypted_grid_path")
        return None

    def find_axis(self, image_path: str) -> Dict[str, Any]:
        temp_output = os.path.join(os.path.dirname(image_path), "temp_output")
        os.makedirs(temp_output, exist_ok=True)

        try:
            result = process_chart(image_path, temp_output)
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

    def process_data(self, chart_id: str, image_path: str, json_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
        result = process_chart(image_path, output_dir)
        if not result:
            return None

        chart_data = {
            "chart_id": chart_id,
            "chart_type": self.chart_type,
            "x_ticks": result.get("x_ticks", []),
            "y_ticks": result.get("y_ticks", []),
            "x_axis_type": result.get("x_axis_type", "numeric"),
            "y_axis_type": result.get("y_axis_type", "numeric"),
            "colors": result.get("colors", []),
            "image_path": result.get("image_path", ""),
            "basic_grid_path": result.get("basic_grid_path", ""),
            "encrypted_grid_path": result.get("encrypted_grid_path", ""),
        }

        if os.path.exists(json_path):
            chart_data["data"] = extract_data_points(load_json(json_path))

        return chart_data

    def evaluate(self, eval_data_path: str) -> Dict[str, Any]:
        temp_output = os.path.join(os.path.dirname(eval_data_path), "temp_eval")
        os.makedirs(temp_output, exist_ok=True)

        try:
            result = process_chart(eval_data_path, temp_output)
            if result:
                return {
                    "chart_id": result.get("chart_id", ""),
                    "x_ticks_count": len(result.get("x_ticks", [])),
                    "y_ticks_count": len(result.get("y_ticks", [])),
                    "colors_count": len(result.get("colors", [])),
                    "success": True,
                }
        finally:
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)

        return {"error": "Evaluation failed"}


class ChartProcessorFactory:
    _processors = {
        "rose": RoseChartProcessor,
        "radar": RadarChartProcessor,
    }

    @classmethod
    def create_processor(cls, chart_type: str) -> ChartProcessor:
        if chart_type in CARTESIAN_CHART_TYPES:
            return CartesianChartProcessor()

        processor_cls = cls._processors.get(chart_type, CartesianChartProcessor)
        return processor_cls()
