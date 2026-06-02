# -*- coding: utf-8 -*-
"""
独立的JSON解析模块，用于处理模型输出的JSON格式解析（直角坐标系专用）
"""

import json
import os
import re
import sys
import asyncio
import aiohttp
from aiohttp import ClientTimeout

# Gemini API配置
GEMINI_API_KEY = os.getenv("CHART_API_KEY") or os.getenv("OPENAI_API_KEY", "")
GEMINI_URL = os.getenv("CHART_BASE_URL", "https://api.vveai.com/v1").rstrip("/") + "/chat/completions"
GEMINI_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {GEMINI_API_KEY}"}


def print(*values, sep=" ", end="\n"):
    text = sep.join(str(value) for value in values)
    try:
        sys.stdout.write(text + end)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace") + end)


def normalize_value(data):
    """
    递归将数据中的数值转换为float类型
    识别并转换百分比字符串，避免将年份等字符串转换为数值
    """
    if isinstance(data, dict):
        return {k: normalize_value(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 只转换列表中的第一个元素（假设是数值），第二个元素保持原样
        if len(data) >= 2:
            return [normalize_value(data[0]), data[1]]
        else:
            return [normalize_value(item) for item in data]
    elif isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, str):
        # 识别并转换百分比字符串
        if "%" in data:
            try:
                # 提取百分比值并转换为float
                return float(data.replace("%", "")) / 100
            except ValueError:
                # 如果转换失败，保持原样
                return data
        # 识别并转换数值字符串
        try:
            return float(data)
        except ValueError:
            # 如果转换失败，保持原样
            return data
    else:
        return data


def extract_json_with_regex(txt):
    """
    使用正则表达式提取JSON字符串
    优先使用代码块解析，只进行最基本的修复
    对于复杂格式直接返回None，让Gemini来处理
    """
    try:
        # 1. 预处理：移除无效的转义字符
        # 保留有效的JSON转义字符
        json_str = re.sub(r'\\(?!(["\\\/bfnrt]|u[0-9a-fA-F]{4}))', '', txt)
        
        # 2. 如果有代码块 ```json ... ```，先提取里面的
        codeblock_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
        if codeblock_match:
            json_str = codeblock_match.group(1).strip()
        
        # 3. 查找最外层的完整JSON结构
        # 优先查找以{开头到最后一个}结尾的结构
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = json_str[first_brace:last_brace+1]
        else:
            # 否则查找以[开头到最后一个]结尾的结构
            first_bracket = json_str.find('[')
            last_bracket = json_str.rfind(']')
            
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_candidate = json_str[first_bracket:last_bracket+1]
            else:
                # 如果都没找到，尝试查找任何可能的JSON结构
                json_objects = re.findall(r"(\{.*\}|\[.*\])", json_str, re.DOTALL)
                if json_objects:
                    json_candidate = max(json_objects, key=len)
                else:
                    raise ValueError("No JSON object or array found in output")
        
        json_str = json_candidate.strip()
        
        # ----------- 最基本的JSON修复 ----------- #
        # 1. 移除代码块标记
        json_str = re.sub(r'```json|```', '', json_str)
        
        # 2. 修复括号不匹配
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")
        
        if open_brackets > close_brackets:
            json_str += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            json_str += "}" * (open_braces - close_braces)
        
        # 3. 移除多余的逗号
        json_str = re.sub(r',\s*\]', r']', json_str)
        json_str = re.sub(r',\s*\}', r'}', json_str)
        # ----------------------------------- #

        # 尝试直接解析，如果失败则让Gemini来处理
        try:
            parsed = json.loads(json_str)
            parsed = normalize_value(parsed)   # ✅ 统一 float 化
            return parsed
        except:
            # 对于复杂格式，直接返回None让Gemini处理
            print("⚠️ 正则式解析遇到复杂格式，交给Gemini处理")
            return None
    except Exception as e:
        print(f"⚠️ 正则式解析失败: {e}")
        # 解析失败后直接返回None，让Gemini来处理
        return None


async def call_gemini_for_json_fix(original_output, prompt):
    """
    调用Gemini模型来修复JSON格式
    """
    # 构造修复请求的提示词
    fix_prompt = f"""
    你是一个专业的JSON格式修复工具。请修复以下文本中的JSON格式错误，确保它可以被JSON.parse正确解析。
    请只返回修复后的JSON字符串，不要添加任何解释或其他内容。

    原始文本：
    ```
    {original_output}
    ```

    修复后的JSON：
    """
    
    # 构造API请求的payload
    payload = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "system", "content": "You are a professional JSON formatter. Fix the JSON format errors in the given text."},
            {"role": "user", "content": fix_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        # 设置超时时间为5分钟
        timeout = ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GEMINI_URL, json=payload, headers=GEMINI_HEADERS) as response:
                if response.status != 200:
                    print(f"❌ Gemini API请求失败，状态码: {response.status}")
                    return None
                
                response_data = await response.json()
                
                # 检查响应格式
                if not response_data or "choices" not in response_data or not response_data["choices"]:
                    print(f"❌ Gemini API返回格式错误: {response_data}")
                    return None
                
                # 提取修复后的内容
                fixed_content = response_data["choices"][0]["message"]["content"].strip()
                
                # 尝试提取JSON部分
                fixed_json = extract_json_with_regex(fixed_content)
                if fixed_json:
                    return fixed_json
                else:
                    # 再次尝试直接解析
                    try:
                        parsed = json.loads(fixed_content)
                        return normalize_value(parsed)
                    except:
                        print(f"❌ Gemini修复后的内容仍无法解析: {fixed_content}")
                        return None
    except asyncio.TimeoutError:
        print(f"❌ Gemini API请求超时")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Gemini API返回内容不是有效的JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ Gemini修复过程中发生错误: {e}")
        return None


async def parse_model_output(original_output, prompt):
    """
    解析模型输出的JSON格式
    优先使用正则表达式提取，失败时调用Gemini API修复
    """
    # 1. 首先尝试使用正则表达式提取和修复
    parsed = extract_json_with_regex(original_output)
    if parsed:
        print("✅ 正则式解析成功")
        return parsed
    
    # 2. 如果正则表达式解析失败，调用Gemini API修复
    print("🔄 正则式解析失败，尝试使用Gemini修复")
    fixed_parsed = await call_gemini_for_json_fix(original_output, prompt)
    if fixed_parsed:
        print("✅ Gemini修复成功")
        return fixed_parsed
    
    # 3. 所有方法都失败
    print("❌ 所有解析方法均失败")
    return None


async def process_parsed_json(parsed_json, chart_type):
    """
    处理解析后的JSON数据，根据图表类型进行适配
    """
    if not parsed_json:
        return None
    
    # 检查是否包含exists字段
    if "exists" in parsed_json and parsed_json["exists"] is False:
        return {
            "exists": False,
            "datapoints": []
        }
    
    # 根据图表类型处理数据
    if chart_type == "scatter":
        # 直角坐标系数据处理
        if "datapoints" in parsed_json:
            datapoints = parsed_json["datapoints"]
            # 确保datapoints是列表格式
            if isinstance(datapoints, list):
                return {
                    "exists": True,
                    "datapoints": datapoints
                }
            else:
                print("❌ datapoints不是列表格式")
                return None
        else:
            print("❌ 缺少datapoints字段")
            return None
    elif chart_type == "pie":
        # 饼图数据处理
        if "datapoints" in parsed_json:
            datapoints = parsed_json["datapoints"]
            # 确保datapoints是列表格式
            if isinstance(datapoints, list):
                return {
                    "exists": True,
                    "datapoints": datapoints
                }
            else:
                print("❌ datapoints不是列表格式")
                return None
        else:
            print("❌ 缺少datapoints字段")
            return None
    else:
        print(f"❌ 不支持的图表类型: {chart_type}")
        return None
