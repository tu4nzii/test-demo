import aiohttp
import base64
import json
import re
from typing import Union, Tuple

api_key = "<your_api_key_here>"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}

async def call_llm_response(prompt: str, image_path: str, point_name: str, task: str = "default") -> Union[Tuple[float, float], str]:
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64_image
                        }
                    },
                    {"text": prompt}
                ]
            }
        ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            try:
                result = await response.json()
            except Exception as e:
                print(f"❌ JSON decode error: {e}")
                return (-1, -1) if task != "diameter_estimation" else "-1"

            if "candidates" not in result:
                print("❌ No 'candidates' in result")
                return (-1, -1) if task != "diameter_estimation" else "-1"

            content = result["candidates"][0]["content"]["parts"][0]["text"]
            print(f"📤 LLM Response:\n{content}")

            if task == "diameter_estimation":
                return content.strip()

            try:
                json_code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
                if json_code_blocks:
                    json_str = json_code_blocks[0].strip()
                else:
                    json_str = next(s for s in content.splitlines() if s.strip().startswith("{") or s.strip().startswith("["))

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
                print(f"❌ JSON parse error: {e}\nRaw content:\n{content}")

            return (-1, -1)

async def estimate_diameter_via_llm(image_path: str, point_name: str) -> float:
    prompt = f"""
You are analyzing a chart that contains a circular visual mark.
Your task is to estimate the **diameter (in pixels)** of the circle [{point_name}] in the image.
Only respond with a single number representing the estimated diameter in pixels, like this:
65
"""
    print(f"\n📤 Prompt for Diameter Estimation:\n{prompt.strip()}")
    response_text = await call_llm_response(prompt, image_path, point_name, task="diameter_estimation")

    matches = re.findall(r"\d+\.?\d*", str(response_text))
    if matches:
        diameter = float(matches[0])
        print(f"📏 Estimated diameter: {diameter:.2f} px")
        return diameter
    else:
        print("⚠️ Could not extract diameter, using default 20.")
        return 20.0


if __name__ == "__main__":
    import asyncio

    async def test():
        img_path = "test_image.png"  # ✅ 替换为实际图像路径
        pt_name = "C3"
        prompt = "What are the coordinates of C3 in this image?"
        result = await call_llm_response(prompt, img_path, pt_name)
        print("✅ Predicted coordinates:", result)

    asyncio.run(test())
