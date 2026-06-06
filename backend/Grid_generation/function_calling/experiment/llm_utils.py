import aiohttp
import base64
import json
import os
import re
import sys
from typing import Tuple, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
backend_root = os.path.abspath(os.path.join(current_dir, "../../.."))
for path in (project_root, backend_root):
    if path not in sys.path:
        sys.path.insert(0, path)

from model_api_config import get_chat_completion_url, get_headers, get_model_name  # noqa: E402


LLM_REQUEST_TIMEOUT_SECONDS = int(os.getenv("EXPERIMENT_LLM_TIMEOUT_SECONDS", "180"))


def _extract_response_text(result: dict) -> str:
    if "choices" in result and result["choices"]:
        message = result["choices"][0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return str(content)

    if "candidates" in result and result["candidates"]:
        parts = result["candidates"][0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))

    return ""


async def call_llm_response(
    prompt: str,
    image_path: str,
    point_name: str,
    task: str = "default",
) -> Union[Tuple[float, float], str]:
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": os.getenv("EXPERIMENT_LLM_MODEL") or os.getenv("MLLM_MODEL") or get_model_name(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "temperature": float(os.getenv("EXPERIMENT_LLM_TEMPERATURE", "0")),
    }

    timeout = aiohttp.ClientTimeout(total=LLM_REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(
                get_chat_completion_url(),
                headers=get_headers(),
                json=payload,
            ) as response:
                text = await response.text()
                if response.status != 200:
                    print(f"LLM HTTP {response.status}: {text[:200]}")
                    return (-1, -1) if task != "diameter_estimation" else "-1"
                result = json.loads(text)
        except Exception as e:
            print(f"LLM request/decode error: {e}")
            return (-1, -1) if task != "diameter_estimation" else "-1"

    content = _extract_response_text(result)
    print(f"LLM Response:\n{content}")

    if task == "diameter_estimation":
        return content.strip()

    try:
        json_code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
        if json_code_blocks:
            json_str = json_code_blocks[0].strip()
        else:
            json_str = next(
                s
                for s in content.splitlines()
                if s.strip().startswith("{") or s.strip().startswith("[")
            )

        coords_json = json.loads(json_str)

        if isinstance(coords_json, dict) and "datapoints" in coords_json:
            dp = coords_json["datapoints"]
            if isinstance(dp, list):
                for item in dp:
                    if isinstance(item, dict) and point_name in item:
                        return tuple(item[point_name])
        elif isinstance(coords_json, list):
            for item in coords_json:
                if item.get("label") == point_name:
                    return tuple(item["point"])

    except Exception as e:
        print(f"JSON parse error: {e}\nRaw content:\n{content}")

    return (-1, -1)


async def estimate_diameter_via_llm(image_path: str, point_name: str) -> float:
    prompt = f"""
You are analyzing a chart that contains a circular visual mark.
Your task is to estimate the diameter in pixels of the circle [{point_name}] in the image.
Only respond with a single number representing the estimated diameter in pixels, like this:
65
"""
    print(f"\nPrompt for Diameter Estimation:\n{prompt.strip()}")
    response_text = await call_llm_response(
        prompt, image_path, point_name, task="diameter_estimation"
    )

    matches = re.findall(r"\d+\.?\d*", str(response_text))
    if matches:
        diameter = float(matches[0])
        print(f"Estimated diameter: {diameter:.2f} px")
        return diameter

    print("Could not extract diameter, using default 20.")
    return 20.0


if __name__ == "__main__":
    import asyncio

    async def test():
        img_path = "test_image.png"
        pt_name = "C3"
        prompt = "What are the coordinates of C3 in this image?"
        result = await call_llm_response(prompt, img_path, pt_name)
        print("Predicted coordinates:", result)

    asyncio.run(test())
