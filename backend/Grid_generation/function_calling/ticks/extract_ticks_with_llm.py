# extract_ticks_with_llm.py
import base64
import json
import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
backend_root = os.path.abspath(os.path.join(current_dir, "../../.."))
grid_root = os.path.abspath(os.path.join(current_dir, "../.."))
function_calling_root = os.path.abspath(os.path.join(current_dir, ".."))
for path in (project_root, backend_root, grid_root, function_calling_root):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import DEBUG_OUTPUT_DIRS, IMG_PATHS  # noqa: E402
from gemini_calls import FAILURE_TEXT, chat_with_gemini_sync  # noqa: E402
from model_api_config import get_model_name  # noqa: E402


LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("TICK_LLM_TIMEOUT_SECONDS", "180"))


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_tick_extraction_prompt():
    return (
        "You are given a chart image. Please extract all X-axis and Y-axis tick values "
        "as they appear on the axis. Respond only in this JSON format:\n"
        "{\n"
        '  "x_ticks": [tick1, tick2, ...],\n'
        '  "y_ticks": [tick1, tick2, ...],\n'
        '  "x_pixels": [],\n'
        '  "y_pixels": []\n'
        "}"
    )


def call_gemini_for_ticks(image_path):
    image_base64 = encode_image_to_base64(image_path)
    prompt = build_tick_extraction_prompt()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                },
            ],
        }
    ]

    try:
        text = chat_with_gemini_sync(
            messages,
            model=os.getenv("TICK_LLM_MODEL") or os.getenv("MLLM_MODEL") or get_model_name(),
            temperature=float(os.getenv("TICK_LLM_TEMPERATURE", "0")),
            timeout_seconds=LLM_REQUEST_TIMEOUT_SECONDS,
        )
        if text == FAILURE_TEXT:
            raise ValueError("Model API request failed.")
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        json_data = json.loads(match.group(0))
        json_data["x_pixels"] = []
        json_data["y_pixels"] = []
        return json_data
    except Exception as e:
        print(f"Failed to parse tick LLM response: {e}")
        return {
            "x_ticks": [],
            "y_ticks": [],
            "x_pixels": [],
            "y_pixels": [],
        }


def extract_ticks_main():
    from config import CLEAR_OUTPUT_BEFORE_RUN

    output_dir = DEBUG_OUTPUT_DIRS["extract_ticks_with_llm"]
    if CLEAR_OUTPUT_BEFORE_RUN.get("extract_ticks_with_llm", False):
        if os.path.exists(output_dir):
            import shutil

            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for name, path in IMG_PATHS.items():
        print(f"[Info] Extracting tick labels from: {name}")
        result = call_gemini_for_ticks(path)
        out_path = os.path.join(output_dir, f"{name}_ticks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"[OK] Tick extraction result saved to: {out_path}")


if __name__ == "__main__":
    extract_ticks_main()
