# -*- coding: utf-8 -*-
"""JSON parsing helpers for model outputs."""

import json
import os
import re
import sys
import asyncio
import aiohttp
from aiohttp import ClientTimeout

from model_api_config import get_chat_completion_url, get_headers, get_model_name


GEMINI_URL = get_chat_completion_url()
GEMINI_HEADERS = get_headers()
GEMINI_MODEL = get_model_name()


def print(*values, sep=" ", end="\n"):
    text = sep.join(str(value) for value in values)
    try:
        sys.stdout.write(text + end)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + end)


def normalize_value(data):
    """Recursively normalize numeric values while preserving label text."""
    if isinstance(data, dict):
        return {k: normalize_value(v) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) >= 2:
            return [normalize_value(data[0]), data[1]]
        else:
            return [normalize_value(item) for item in data]
    elif isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, str):
        if "%" in data:
            try:
                return float(data.replace("%", "")) / 100
            except ValueError:
                return data
        try:
            return float(data)
        except ValueError:
            return data
    else:
        return data


def extract_json_with_regex(txt):
    """Extract and lightly repair a JSON object or array from model text."""
    try:
        json_str = re.sub(r'\\(?!(["\\\/bfnrt]|u[0-9a-fA-F]{4}))', '', txt)
        
        codeblock_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
        if codeblock_match:
            json_str = codeblock_match.group(1).strip()
        
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = json_str[first_brace:last_brace+1]
        else:
            first_bracket = json_str.find('[')
            last_bracket = json_str.rfind(']')
            
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_candidate = json_str[first_bracket:last_bracket+1]
            else:
                json_objects = re.findall(r"(\{.*\}|\[.*\])", json_str, re.DOTALL)
                if json_objects:
                    json_candidate = max(json_objects, key=len)
                else:
                    raise ValueError("No JSON object or array found in output")
        
        json_str = json_candidate.strip()
        
        json_str = re.sub(r'```json|```', '', json_str)
        
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")
        
        if open_brackets > close_brackets:
            json_str += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            json_str += "}" * (open_braces - close_braces)
        
        json_str = re.sub(r',\s*\]', r']', json_str)
        json_str = re.sub(r',\s*\}', r'}', json_str)

        try:
            parsed = json.loads(json_str)
            return normalize_value(parsed)
        except Exception:
            print("[Warning] Regex JSON parsing found a complex format; falling back to model repair.")
            return None
    except Exception as e:
        print(f"[Warning] Regex JSON parsing failed: {e}")
        return None


async def call_gemini_for_json_fix(original_output, prompt):
    """Ask the configured model to repair malformed JSON."""
    fix_prompt = f"""
    你是专业的 JSON 格式修复工具。请修复下面文本中的 JSON 格式错误，
    确保它可以被 JSON.parse 正确解析。
    只返回修复后的 JSON 字符串，不要添加解释、Markdown 或其他内容。

    原始文本：
    ```
    {original_output}
    ```

    修复后的 JSON：
    """
    
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional JSON formatter. Fix the JSON format errors in the given text."},
            {"role": "user", "content": fix_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        timeout = ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GEMINI_URL, json=payload, headers=GEMINI_HEADERS) as response:
                if response.status != 200:
                    print(f"[Error] Model API request failed with status: {response.status}")
                    return None
                
                response_data = await response.json()
                
                if not response_data or "choices" not in response_data or not response_data["choices"]:
                    print(f"[Error] Model API returned an unexpected response: {response_data}")
                    return None
                
                fixed_content = response_data["choices"][0]["message"]["content"].strip()
                
                fixed_json = extract_json_with_regex(fixed_content)
                if fixed_json:
                    return fixed_json
                else:
                    try:
                        parsed = json.loads(fixed_content)
                        return normalize_value(parsed)
                    except Exception:
                        print(f"[Error] Repaired model output is still not valid JSON: {fixed_content}")
                        return None
    except asyncio.TimeoutError:
        print("[Error] Model API request timed out.")
        return None
    except json.JSONDecodeError as e:
        print(f"[Error] Model API response is not valid JSON: {e}")
        return None
    except Exception as e:
        print(f"[Error] JSON repair failed: {e}")
        return None


async def parse_model_output(original_output, prompt):
    """Parse model output as JSON, using model repair only when needed."""
    parsed = extract_json_with_regex(original_output)
    if parsed:
        print("Regex JSON parsing succeeded")
        return parsed
    
    print("[Info] Regex JSON parsing failed; trying model-based repair.")
    fixed_parsed = await call_gemini_for_json_fix(original_output, prompt)
    if fixed_parsed:
        print("[Info] Model-based JSON repair succeeded.")
        return fixed_parsed
    
    print("[Error] All JSON parsing methods failed.")
    return None


async def process_parsed_json(parsed_json, chart_type):
    """Adapt parsed JSON to the shape expected by downstream chart modules."""
    if not parsed_json:
        return None
    
    if "exists" in parsed_json and parsed_json["exists"] is False:
        return {
            "exists": False,
            "datapoints": []
        }
    
    if chart_type == "scatter":
        if "datapoints" in parsed_json:
            datapoints = parsed_json["datapoints"]
            if isinstance(datapoints, list):
                return {
                    "exists": True,
                    "datapoints": datapoints
                }
            else:
                print("[Error] datapoints is not a list.")
                return None
        else:
            print("[Error] Missing datapoints field.")
            return None
    elif chart_type == "pie":
        if "datapoints" in parsed_json:
            datapoints = parsed_json["datapoints"]
            if isinstance(datapoints, list):
                return {
                    "exists": True,
                    "datapoints": datapoints
                }
            else:
                print("[Error] datapoints is not a list.")
                return None
        else:
            print("[Error] Missing datapoints field.")
            return None
    else:
        print(f"[Error] Unsupported chart type: {chart_type}")
        return None
