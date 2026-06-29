import json
import os
from typing import Optional, Tuple

from model_api_config import get_chat_completion_url, get_headers, get_model_name


class RoseChartEvaluator:
    """Legacy rose evaluator compatibility wrapper.

    The active system evaluation flow lives under backend/evaluation_prediction.
    This module is kept importable for older scripts, but it does not run the
    deprecated ground-truth style evaluation pipeline.
    """

    def __init__(self):
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.llm_model = get_model_name()
        self.feedback_image_dir = "./data/feedback"
        self.amplifier_image_dir = "./data/amplifier/rose"
        self.results_by_image = {}
        self._create_directories()

    def _create_directories(self) -> None:
        os.makedirs(self.feedback_image_dir, exist_ok=True)
        os.makedirs(self.amplifier_image_dir, exist_ok=True)

    def load_dataset(self, json_path: str) -> dict:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Operation failed: {e}")
            return {}

    def extract_json_response(self, content: str) -> Optional[dict]:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start < 0 or end <= start:
                return None
            return json.loads(content[start:end])
        except Exception as e:
            print(f"JSON parse failed: {e}")
            return None

    def generate_prompt(self, item_name: str, prompt_type: str, dataset: dict, tick: float = 0) -> str:
        chart_type = dataset.get("chart_type", "rose")
        return f"""
You are analyzing a {chart_type} chart.
Estimate the value for target item "{item_name}" using visible marks and reference lines.
Prompt mode: {prompt_type}
Previous estimate, if any: {tick}

Return strict JSON only:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}
""".strip()

    def call_llm_response(
        self,
        prompt: str,
        image_path: str,
        item_name: str,
        dataset: dict,
    ) -> Tuple[Optional[float], Optional[float]]:
        print("Legacy rose evaluator is disabled; use backend/evaluation_prediction instead.")
        return (None, None)

    def process_single_image(self, json_path: str) -> None:
        dataset = self.load_dataset(json_path)
        if not dataset:
            print("Dataset is empty; skipping evaluation")
            return
        chart_id = dataset.get("chart_id", "unknown")
        print(f"Start processing chart: {chart_id}")
        self.results_by_image[chart_id] = {
            "chart_type": dataset.get("chart_type", "rose"),
            "data": {},
        }

    def save_results(self, output_path: Optional[str] = None) -> None:
        if not self.results_by_image:
            print("No results to save")
            return
        if output_path is None:
            first_chart_id = next(iter(self.results_by_image.keys()), "")
            chart_type = self.results_by_image.get(first_chart_id, {}).get("chart_type", "unknown")
            output_path = f"coordinates_by_image_{chart_type}_{self.llm_model}.json"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results_by_image, f, ensure_ascii=False, indent=4)
            print(f"Output saved to: {output_path}")
        except Exception as e:
            print(f"Operation failed: {e}")


if __name__ == "__main__":
    evaluator = RoseChartEvaluator()
    json_file_path = "./sample_outputs/rose/result/chart_1761120786_evalution_datasets.json"
    evaluator.process_single_image(json_file_path)
    evaluator.save_results()
