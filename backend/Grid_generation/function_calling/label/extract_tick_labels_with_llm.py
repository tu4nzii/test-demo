# -*- coding: utf-8 -*-
"""
使用Gemini API提取图表刻度标签
支持轴类型判断和刻度值识别
"""

import os
import sys
import base64
import re
import json
import hashlib
import requests
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import numpy as np

# 导入gemini调用模块（gemini_calls.py在项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从 f:\program\test-demo\backend\Grid_generation\function_calling\label 向上3级到达项目根目录
project_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# API配置
API_URL = "https://api.vveai.com/v1/chat/completions"
API_KEYS = [
    "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    "sk-2nzrUYD0JWLFzopWF477111f78E746AbAcA9Ed8534C3A481",
    "sk-CiD5WVUNIkBeXDgYB46b90C06aD24636BcEaBaFa993970C4",
    "sk-WvF4fU10VeOkfFMq579610Fc01E8496d827d0d3e04C44d0a",
    "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
]
key_index = 0

def rotate_key():
    """切换到下一个 key"""
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"🔑 已切换至新的 API Key [{key_index + 1}/{len(API_KEYS)}]")

def chat_with_gemini(messages: list) -> str:
    """与Gemini进行对话（同步版本）"""
    payload = {
        "model": "gemini-2.5-pro",
        "messages": messages,
        "temperature": 0.7
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_index]}"
    }
    
    for attempt in range(1, 4):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=180)
            
            if response.status_code == 429:
                print(f"🚫 请求频率超限，切换 Key 重试...")
                rotate_key()
                continue
            
            if response.status_code != 200:
                print(f"⚠️ HTTP {response.status_code}: {response.text[:200]}")
                continue
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                print(f"⚠️ 响应格式错误: {result}")
                
        except Exception as e:
            print(f"❌ 第 {attempt} 次尝试失败: {e}")
            continue
    
    print("❌ 所有尝试均失败")
    return "抱歉，我暂时无法回应您的请求。"


def get_cache_file_path(image_path: str, cache_dir: str) -> str:
    """
    根据图像路径生成缓存文件路径
    
    Args:
        image_path: 图像文件路径
        cache_dir: 缓存目录
    
    Returns:
        缓存文件路径
    """
    # 使用图像路径的hash作为文件名
    image_hash = hashlib.md5(image_path.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f"{image_hash}.json")
    return cache_file


def load_llm_cache(cache_file: str) -> Optional[Dict]:
    """
    从缓存文件加载LLM识别结果
    
    Args:
        cache_file: 缓存文件路径
    
    Returns:
        识别结果字典，如果文件不存在则返回None
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        print(f"[Warning] 读取缓存文件失败: {e}")
        return None


def save_llm_cache(cache_file: str, result: Dict, image_path: str, x_response: str = "", y_response: str = "") -> None:
    """
    保存LLM识别结果到缓存文件
    
    Args:
        cache_file: 缓存文件路径
        result: 识别结果字典
        image_path: 原始图像路径
        x_response: X轴LLM原始响应（可选）
        y_response: Y轴LLM原始响应（可选）
    """
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        cache_data = {
            "image_path": image_path,
            "x_ticks": result.get("x_ticks", []),
            "y_ticks": result.get("y_ticks", []),
            "x_axis_type": result.get("x_axis_type", "未知"),
            "y_axis_type": result.get("y_axis_type", "未知"),
            "x_llm_response": x_response,  # 保存原始响应以便调试
            "y_llm_response": y_response,
            "cached_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"[Info] LLM识别结果已缓存: {cache_file}")
    except Exception as e:
        print(f"[Warning] 保存缓存文件失败: {e}")


def encode_image_to_base64(image_path: str) -> str:
    """
    将图像编码为base64字符串
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        base64编码的字符串
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_tick_extraction_prompt(direction: str = 'x') -> str:
    """
    构造提示词，增加轴类型判断和顺序要求
    
    Args:
        direction: 'x' 表示X轴，'y' 表示Y轴
    
    Returns:
        提示词字符串
    """
    # X轴从左到右，Y轴从下到上
    if direction.lower() == 'x':
        direction_text = "X轴"
        order_text = "从左到右"
    else:
        direction_text = "Y轴"
        order_text = "从下到上"
    
    prompt = f"""请识别图片中{direction_text}上的所有刻度值和轴类型。

**步骤1: 轴类型判断**
请首先判断该轴是"数值轴"还是"文字轴"：
- 数值轴：刻度为纯数字，如1, 2, 3.5, -4等
- 文字轴：刻度为文字、字母或年份、月份等非数值信息，如'Jan', 'Feb', '2020', 'A', 'B', 'Company A'等

**步骤2: 刻度值识别**
请识别{direction_text}上的所有刻度值，严格按照{order_text}的顺序识别和返回。

**输出格式示例：**

如果是数值轴：
```
轴类型:数值轴
刻度值:
```
数字1
数字2
...
```
```

如果是文字轴：
```
轴类型:文字轴
刻度值:
```
文字1
文字2
...
```
```

**重要要求：**
1. 刻度值必须按照{order_text}的顺序列出，不要颠倒顺序。
2. 对于长文本标签(如公司名称、地名、产品名称等)，请完整识别整个标签文本，不要截断。
3. 每个刻度值必须在一行内完整显示，不要使用换行符(\n)或任何其他特殊字符分隔。
4. 如果标签文本很长，请保持原样完整返回，不要添加换行符或分段。
5. 不要包含任何其他文字说明，只返回轴类型和刻度值。"""
    return prompt


def extract_axis_ticks_with_llm(image_path: str, direction: str = 'x') -> Dict:
    """
    使用LLM识别指定轴的刻度值和类型（同步版本）
    
    Args:
        image_path: 图像文件路径
        direction: 'x' 表示X轴，'y' 表示Y轴
    
    Returns:
        包含axis_type、ticks和raw_response的字典，格式: {"axis_type": "数值轴"或"文字轴", "ticks": [...], "raw_response": "..."}
    """
    try:
        # 读取图像并编码为base64
        image_base64 = encode_image_to_base64(image_path)
        
        # 构建提示词
        prompt = build_tick_extraction_prompt(direction)
        
        # 构建包含图像的消息
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的图表分析专家，擅长识别图表中的坐标轴和刻度标签。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        # 调用Gemini API（同步版本）
        response = chat_with_gemini(messages)
        
        # 解析响应
        result = parse_llm_response(response, direction)
        result["raw_response"] = response  # 保存原始响应
        return result
        
    except Exception as e:
        print(f"[Error] LLM识别{direction}轴刻度失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "axis_type": "未知",
            "ticks": [],
            "raw_response": ""
        }


def parse_llm_response(response_text: str, direction: str) -> Dict:
    """
    解析LLM响应，提取轴类型和刻度值
    
    Args:
        response_text: LLM返回的文本
        direction: 轴方向（用于错误提示）
    
    Returns:
        包含axis_type和ticks的字典
    """
    result = {
        "axis_type": "未知",
        "ticks": []
    }
    
    try:
        # 提取轴类型
        axis_type_match = re.search(r'轴类型[：:]\s*(数值轴|文字轴)', response_text)
        if axis_type_match:
            result["axis_type"] = axis_type_match.group(1)
        
        # 提取刻度值部分
        # 查找刻度值区域（在"刻度值:"之后）
        ticks_section_match = re.search(r'刻度值[：:]\s*\n?```\s*\n?(.*?)\n?```', response_text, re.DOTALL)
        if ticks_section_match:
            ticks_text = ticks_section_match.group(1).strip()
        else:
            # 如果没有找到代码块，尝试查找"刻度值:"之后的所有行
            ticks_section_match = re.search(r'刻度值[：:]\s*\n(.*?)(?=\n\n|\Z)', response_text, re.DOTALL)
            if ticks_section_match:
                ticks_text = ticks_section_match.group(1).strip()
            else:
                # 如果还是找不到，尝试提取所有非空行
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                # 找到"刻度值:"之后的行
                start_idx = -1
                for i, line in enumerate(lines):
                    if '刻度值' in line:
                        start_idx = i + 1
                        break
                if start_idx > 0:
                    ticks_text = '\n'.join(lines[start_idx:])
                else:
                    ticks_text = ""
        
        # 解析刻度值列表
        if ticks_text:
            ticks = []
            for line in ticks_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('```'):
                    # 尝试转换数值
                    try:
                        # 如果是数值，转换为float
                        tick_value = float(line)
                        ticks.append(tick_value)
                    except ValueError:
                        # 如果是文字，保持原样
                        ticks.append(line)
            
            result["ticks"] = ticks
        
        # 如果没有找到刻度值，尝试从整个响应中提取
        if not result["ticks"]:
            # 尝试提取所有可能的数值或文字
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not any(keyword in line for keyword in ['轴类型', '刻度值', '```', '---']):
                    try:
                        tick_value = float(line)
                        result["ticks"].append(tick_value)
                    except ValueError:
                        if len(line) > 0:
                            result["ticks"].append(line)
        
    except Exception as e:
        print(f"[Warning] 解析LLM响应失败: {e}")
        print(f"[Debug] 响应内容: {response_text[:500]}")
    
    return result


def extract_tick_labels_with_llm(image_path: str, cache_dir: Optional[str] = None) -> Dict:
    """
    使用LLM识别图表的X轴和Y轴刻度标签（支持缓存，同步版本）
    
    Args:
        image_path: 图像文件路径
        cache_dir: 缓存目录（如果为None，则不使用缓存）
    
    Returns:
        包含x_ticks和y_ticks的字典，格式: {"x_ticks": [...], "y_ticks": [...]}
    """
    # 检查缓存
    if cache_dir:
        cache_file = get_cache_file_path(image_path, cache_dir)
        cached_result = load_llm_cache(cache_file)
        if cached_result:
            print(f"[Info] 从缓存加载LLM识别结果: {cache_file}")
            return {
                "x_ticks": cached_result.get("x_ticks", []),
                "y_ticks": cached_result.get("y_ticks", []),
                "x_axis_type": cached_result.get("x_axis_type", "未知"),
                "y_axis_type": cached_result.get("y_axis_type", "未知")
            }
    
    print(f"[Info] 开始使用LLM识别刻度标签: {image_path}")
    
    # 分别识别X轴和Y轴
    x_result = extract_axis_ticks_with_llm(image_path, direction='x')
    y_result = extract_axis_ticks_with_llm(image_path, direction='y')
    
    # 获取原始响应（如果可用）
    x_response = x_result.get("raw_response", "")
    y_response = y_result.get("raw_response", "")
    
    result = {
        "x_ticks": x_result.get("ticks", []),
        "y_ticks": y_result.get("ticks", []),
        "x_axis_type": x_result.get("axis_type", "未知"),
        "y_axis_type": y_result.get("axis_type", "未知")
    }
    
    print(f"[Info] X轴识别结果: 类型={result['x_axis_type']}, 刻度数={len(result['x_ticks'])}")
    print(f"[Info] Y轴识别结果: 类型={result['y_axis_type']}, 刻度数={len(result['y_ticks'])}")
    
    # 保存到缓存
    if cache_dir:
        cache_file = get_cache_file_path(image_path, cache_dir)
        save_llm_cache(cache_file, result, image_path, x_response, y_response)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试LLM刻度标签提取')
    parser.add_argument('--image', type=str, required=True, help='图像文件路径')
    args = parser.parse_args()
    
    result = extract_tick_labels_with_llm(args.image)
    print("\n识别结果:")
    print(f"X轴类型: {result['x_axis_type']}")
    print(f"X轴刻度: {result['x_ticks']}")
    print(f"Y轴类型: {result['y_axis_type']}")
    print(f"Y轴刻度: {result['y_ticks']}")
