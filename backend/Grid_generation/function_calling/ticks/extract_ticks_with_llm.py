# extract_ticks_with_llm.py
import os
import json
import base64
import sys
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import requests
from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, GEMINI_API_KEY, GEMINI_MODEL

# 构造 Gemini 请求 URL
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_tick_extraction_prompt():
    return (
        "You are given a chart image. Please extract all X-axis and Y-axis tick values "
        "as they appear on the axis. Respond in this JSON format:\n"
        "{\n"
        "  \"x_ticks\": [tick1, tick2, ...],\n"
        "  \"y_ticks\": [tick1, tick2, ...],\n"
        "  \"x_pixels\": [],\n"
        "  \"y_pixels\": []\n"
        "}"
    )

def call_gemini_for_ticks(image_path):
    image_base64 = encode_image_to_base64(image_path)
    prompt = build_tick_extraction_prompt()

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_base64
                    }
                },
                {
                    "text": prompt
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0
        }
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload)

    try:
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        # 用正则提取第一个合法JSON块
        match = re.search(r'{[\s\S]*}', text)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        json_data = json.loads(match.group(0))
        json_data["x_pixels"] = []
        json_data["y_pixels"] = []
        return json_data
    except Exception as e:
        print(f"❌ Failed to parse Gemini response: {e}")
        return {
            "x_ticks": [],
            "y_ticks": [],
            "x_pixels": [],
            "y_pixels": []
        }

# 🧪 模块测试入口
def extract_ticks_main():
    from utils.image_io import load_image, save_image
    from config import CLEAR_OUTPUT_BEFORE_RUN

    output_dir = DEBUG_OUTPUT_DIRS['extract_ticks_with_llm']
    if CLEAR_OUTPUT_BEFORE_RUN.get('extract_ticks_with_llm', False):
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for name, path in IMG_PATHS.items():
        print(f"[Info] 识别图像刻度: {name}")
        result = call_gemini_for_ticks(path)
        out_path = os.path.join(output_dir, f"{name}_ticks.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4)
        print(f"[✔] Tick 值提取结果保存至: {out_path}")

if __name__ == '__main__':
    extract_ticks_main()
