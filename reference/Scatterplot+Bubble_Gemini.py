# Experiment Scheduler for Grid-Based Prompt Evaluation (Three Prompt-Image Settings)

import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import base64
import asyncio
import aiohttp
import sys
from datetime import datetime

from PIL import Image, ImageDraw
import numpy as np
from math import hypot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_LOG_FILE = None


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_console_log(log_path: str | None = None) -> str:
    global _LOG_FILE
    if _LOG_FILE is not None:
        return _LOG_FILE.name

    if log_path is None:
        log_dir = os.path.join(SCRIPT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"Scatterplot+Bubble_Gemini_{timestamp}.log")
    else:
        log_path = resolve_local_path(log_path)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    _LOG_FILE = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _LOG_FILE)
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _LOG_FILE)
    print(f"📝 控制台日志保存至: {log_path}")
    return log_path


def close_console_log() -> None:
    global _LOG_FILE
    if _LOG_FILE is None:
        return
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
    _LOG_FILE.close()
    _LOG_FILE = None


def resolve_local_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def normalize_image_paths(dataset: dict) -> dict:
    image_paths = dataset.get("image_paths") or {}
    fallback = image_paths.get("no_grid") or image_paths.get("with_grid") or image_paths.get("grid_with_grid")
    if fallback:
        for key in ("no_grid", "with_grid", "grid_with_grid"):
            image_paths.setdefault(key, fallback)
    dataset["image_paths"] = image_paths
    return dataset


def load_local_env(env_path: str = ".env") -> None:
    env_file = resolve_local_path(env_path)
    if not os.path.isfile(env_file):
        return

    with open(env_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()


# === 实验配置 ===
REPEAT_TIMES = 3
MAX_ATTEMPTS = 5  # 每个点最多尝试5次来获得3次成功预测
EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("feedback_crop_adaptive", "grid_with_grid"),  # 启用基于feedback最后一轮预测的裁剪实验（最后一轮）

]

# ========== 配置项 ========== #
import aiohttp, asyncio, json, re, base64, threading, random
from typing import Tuple, Union
from aiohttp import ClientTimeout

# Gemini API 配置。不要把密钥硬编码进脚本，运行前设置 GEMINI_API_KEY 或 GEMINI_API_KEYS。
GEMINI_API_URL = os.getenv("GEMINI_API_URL", "https://api.vveai.com/v1/chat/completions")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "2048"))
DEFAULT_MARK_DIAMETER = float(os.getenv("DEFAULT_MARK_DIAMETER", "20"))
MIN_MARK_DIAMETER = float(os.getenv("MIN_MARK_DIAMETER", "5"))
MAX_MARK_DIAMETER = float(os.getenv("MAX_MARK_DIAMETER", "160"))
API_KEYS = [
    key.strip()
    for key in re.split(r"[,\n;]+", os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY", ""))
    if key.strip()
]
API_URLS = [GEMINI_API_URL]


# API 与密钥轮换索引
api_index = 0
key_index = 0
api_lock = threading.Lock()  # 线程锁，确保多线程环境下的安全性
JSON_ONLY_SYSTEM_PROMPT = (
    "You are a chart coordinate extraction engine. "
    "Return only valid JSON. Do not use Markdown. Do not explain. "
    "Do not include any text before or after the JSON."
)


def normalize_predicted_point(value, point_name: str) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        print(f"⚠️ [{point_name}] 坐标格式无效：{value}")
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        print(f"⚠️ [{point_name}] 坐标不是有效数字：{value}")
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        print(f"⚠️ [{point_name}] 坐标不是有限数字：{value}")
        return None
    if x == -1 and y == -1:
        return None
    return (x, y)

# 获取下一个API URL（轮询策略）
def get_next_api_url():
    global api_index
    with api_lock:
        url = API_URLS[api_index]
        api_index = (api_index + 1) % len(API_URLS)
        return url


def get_headers() -> dict:
    if not API_KEYS:
        raise RuntimeError("未配置 Gemini API Key：请设置环境变量 GEMINI_API_KEY 或 GEMINI_API_KEYS")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_index]}",
    }


def rotate_key() -> None:
    global key_index
    if not API_KEYS:
        return
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"🔑 已切换至新的 API Key [{key_index + 1}/{len(API_KEYS)}]")

# ===== 全局配置 =====
BASE_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=30, sock_connect=30, sock_read=240)
SEM_LIMIT = 5
sem = asyncio.Semaphore(SEM_LIMIT)  # 并发限制
MAX_RETRIES = 3
_session: aiohttp.ClientSession | None = None


# ======= Gemini 模型调用函数 =======


# ======= 主函数 =======
async def call_llm_response(prompt: str, image_path: str, point_name: str, task: str = "default") -> Union[Tuple[float, float], str]:
    """
    调用 Gemini API
    使用 OpenAI 兼容的 JSON 格式上传图片
    """
    global _session

    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)

    img_size_kb = os.path.getsize(image_path) / 1024

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 读取并编码图片为 base64
                with open(image_path, "rb") as f:
                    image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # 构建 OpenAI 兼容的请求体
                payload = {
                    "model": GEMINI_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": JSON_ONLY_SYSTEM_PROMPT
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
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": GEMINI_MAX_TOKENS,
                    "max_completion_tokens": GEMINI_MAX_TOKENS,
                    "temperature": 0,
                    "response_format": {"type": "json_object"}
                }
                
                # 轮询获取下一个API URL
                current_url = get_next_api_url()
                print(f"🔄 [{point_name}] 使用Gemini API端点: {current_url}")
                
                # 发送请求
                async with _session.post(
                    current_url,
                    headers=get_headers(),
                    json=payload
                ) as response:
                    text_buf = None
                    for phase in range(1, 6):
                        try:
                            text_buf = await asyncio.wait_for(response.text(), timeout=30)
                            break
                        except asyncio.TimeoutError:
                            print(f"💤 [{point_name}] 阶段 {phase}/5 超时等待中…")
                            if phase == 5:
                                raise asyncio.TimeoutError
                            await asyncio.sleep(1.5 * phase)

                    if not text_buf:
                        print(f"⚠️ [{point_name}] 阶段性超时，重试。")
                        continue

                    if response.status == 429:
                        print(f"🚫 [{point_name}] 请求频率超限，切换 Key 重试...")
                        rotate_key()
                        await asyncio.sleep(3)
                        continue

                    if response.status != 200:
                        print(f"⚠️ [{point_name}] HTTP {response.status}: {text_buf[:200]}")
                        await asyncio.sleep(2 ** attempt)
                        continue

                    # 解析模型返回
                    text = text_buf.strip()

                    # ① 优先尝试结构化解析
                    def safe_json_loads(s: str):
                        original = s
                        s = re.sub(r"^```(?:json)?", "", s)
                        s = re.sub(r"```$", "", s)
                        s = re.sub(r"[\x00-\x1f]+", "", s)
                        s = s.replace("`", "").strip()
                        s = re.sub(r',\s*([}\]])', r'\1', s)
                        # 处理模型返回的[x, y]占位符
                        s = re.sub(r'\[x\s*,\s*y\]', '[-1, -1]', s, flags=re.IGNORECASE)
                        # 处理可能的单独x或y占位符
                        s = re.sub(r'"?x"?\s*,\s*"?y"?', '"-1", "-1"', s, flags=re.IGNORECASE)
                        diff_c, diff_s = s.count("{") - s.count("}"), s.count("[") - s.count("]")
                        if diff_c > 0: s += "}" * diff_c
                        if diff_s > 0: s += "]" * diff_s
                        try:
                            return json.loads(s)
                        except Exception:
                            start, end = s.find("{"), s.rfind("}")
                            if start != -1 and end != -1:
                                try:
                                    return json.loads(s[start:end + 1])
                                except Exception:
                                    pass
                        print(f"⚠️ JSON 修复失败: {original}")
                        return None

                    result = safe_json_loads(text)
                    content = None
                    
                    # 解析响应内容
                    if result and "choices" in result:
                        content = result["choices"][0]["message"]["content"]
                        
                        # 处理模型返回的嵌套JSON格式
                        if isinstance(content, str):
                            # 尝试解析content为JSON
                            parsed_content = safe_json_loads(content)
                            if parsed_content:
                                # 如果解析后的内容包含response字段，进一步处理
                                if isinstance(parsed_content, dict) and "response" in parsed_content:
                                    response_content = parsed_content["response"]
                                    # 如果response是字符串，再次解析
                                    if isinstance(response_content, str):
                                        final_content = safe_json_loads(response_content)
                                        if final_content:
                                            content = json.dumps(final_content)
                                        else:
                                            content = response_content
                                    else:
                                        content = json.dumps(response_content)
                                else:
                                    content = json.dumps(parsed_content)
                    elif isinstance(result, dict):
                        # 处理模型返回的嵌套JSON格式
                        if "response" in result:
                            # 如果response字段是字符串形式的JSON，尝试解析
                            response_content = result["response"]
                            parsed_response = safe_json_loads(response_content)
                            if parsed_response:
                                content = json.dumps(parsed_response)
                            else:
                                content = response_content
                        # 有时直接返回 {"datapoints": {...}}
                        elif "datapoints" in result:
                            content = json.dumps(result)
                        else:
                            content = json.dumps(result)
                    else:
                        content = text

                    print(f"🧠 [{point_name}] 模型原始返回内容：\n{content}")
                    
                    if task == "diameter_estimation":
                        return content.strip()

                    # ② 坐标提取
                    try:
                        json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
                        if json_blocks:
                            json_str = json_blocks[0].strip()
                        else:
                            json_str = next(
                                s for s in content.splitlines()
                                if s.strip().startswith("{") or s.strip().startswith("[")
                            )
                    except StopIteration:
                        print(f"⚠️ [{point_name}] 未检测到 JSON 结构：{content}")
                        return (-1, -1)
                    
                    coords_json = safe_json_loads(json_str)
                    if not coords_json:
                        print(f"⚠️ [{point_name}] 无法解析 JSON：{content}")
                        return (-1, -1)
                    
                    # --- 坐标匹配逻辑 ---
                    if isinstance(coords_json, list):
                        for item in coords_json:
                            if isinstance(item, dict) and item.get("label") == point_name:
                                point = normalize_predicted_point(item.get("point"), point_name)
                                return point if point is not None else (-1, -1)
                    
                    elif isinstance(coords_json, dict) and "datapoints" in coords_json:
                        dp = coords_json["datapoints"]
                        if isinstance(dp, list):
                            for item in dp:
                                if isinstance(item, dict) and point_name in item:
                                    point = normalize_predicted_point(item.get(point_name), point_name)
                                    return point if point is not None else (-1, -1)
                        elif isinstance(dp, dict):
                            if point_name in dp:
                                point = normalize_predicted_point(dp.get(point_name), point_name)
                                return point if point is not None else (-1, -1)
                    
                    print(f"⚠️ [{point_name}] 未找到 {point_name}，解析后的 JSON：")
                    print(json.dumps(coords_json, indent=2))
                    return (-1, -1)

            except asyncio.TimeoutError:
                print(f"⏳ [{point_name}] 超时（第 {attempt}/{MAX_RETRIES} 次）")
                await asyncio.sleep(3 * attempt)
                continue

            except aiohttp.ClientConnectionError as e:
                print(f"🌐 [{point_name}] 网络异常：{e} → session 重建")
                if _session and not _session.closed:
                    await _session.close()
                _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)
                await asyncio.sleep(5)
                continue

            except asyncio.exceptions.CancelledError:
                print(f"⏸️ [{point_name}] 请求被取消，重试。")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                print(f"❌ [{point_name}] 未知错误：{type(e).__name__} - {e}")
                await asyncio.sleep(2)
                continue

        print(f"❌ [{point_name}] 连续 {MAX_RETRIES} 次失败，放弃。")
        return (-1, -1)


# ====== 重试封装 ======
async def call_llm_with_retry(prompt: str, image_path: str, point_name: str, task: str = "default") -> Union[
    Tuple[float, float], str]:
    """重试包装，兼容旧逻辑"""
    for _ in range(MAX_ATTEMPTS):
        coords = await call_llm_response(prompt, image_path, point_name, task)
        if coords != (-1, -1) and coords != "-1":
            return coords
        await asyncio.sleep(6)
    return None


# ====== 点存在性检测 ======
async def call_llm_point_existence(prompt: str, image_path: str) -> bool:
    """调用 LLM 判断裁剪图像中是否存在目标散点"""
    global _session

    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # 读取并编码图片为 base64
                with open(image_path, "rb") as f:
                    image_data = f.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # 构建 OpenAI 兼容的请求体
                payload = {
                    "model": GEMINI_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": JSON_ONLY_SYSTEM_PROMPT
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
                                        "url": f"data:image/png;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": GEMINI_MAX_TOKENS,
                    "max_completion_tokens": GEMINI_MAX_TOKENS,
                    "temperature": 0,
                    "response_format": {"type": "json_object"}
                }
                
                # 获取下一个API URL
                current_url = get_next_api_url()
                print(f"🔄 使用Gemini API端点: {current_url}")
                
                # 发送请求
                async with _session.post(
                    current_url,
                    headers=get_headers(),
                    json=payload
                ) as response:
                    text = await asyncio.wait_for(response.text(), timeout=60)
                    if response.status == 429:
                        print("🚫 存在性请求频率超限，切换 Key 重试...")
                        rotate_key()
                        await asyncio.sleep(3)
                        continue

                    if response.status != 200:
                        print(f"⚠️ 存在性请求 HTTP {response.status}: {text[:200]}")
                        await asyncio.sleep(2)
                        continue

                    # 解析响应
                    def safe_json_loads(s: str):
                        original = s
                        s = re.sub(r"^```(?:json)?", "", s)
                        s = re.sub(r"```$", "", s)
                        s = re.sub(r"[\x00-\x1f]+", "", s)
                        s = s.replace("`", "").strip()
                        s = re.sub(r',\s*([}\]])', r'\1', s)
                        # 处理模型返回的[x, y]占位符
                        s = re.sub(r'\[x\s*,\s*y\]', '[-1, -1]', s, flags=re.IGNORECASE)
                        # 处理可能的单独x或y占位符
                        s = re.sub(r'"?x"?\s*,\s*"?y"?', '"-1", "-1"', s, flags=re.IGNORECASE)
                        diff_c, diff_s = s.count("{") - s.count("}"), s.count("[") - s.count("]")
                        if diff_c > 0: s += "}" * diff_c
                        if diff_s > 0: s += "]" * diff_s
                        try:
                            return json.loads(s)
                        except Exception:
                            start, end = s.find("{"), s.rfind("}")
                            if start != -1 and end != -1:
                                try:
                                    return json.loads(s[start:end + 1])
                                except Exception:
                                    pass
                        print(f"⚠️ JSON 修复失败: {original}")
                        return None

                    result = safe_json_loads(text)
                    content = None
                    
                    # 解析响应内容
                    if result and "choices" in result:
                        content = result["choices"][0]["message"]["content"]
                        
                        # 处理模型返回的嵌套JSON格式
                        if isinstance(content, str):
                            # 尝试解析content为JSON
                            parsed_content = safe_json_loads(content)
                            if parsed_content:
                                # 如果解析后的内容包含response字段，进一步处理
                                if isinstance(parsed_content, dict) and "response" in parsed_content:
                                    response_content = parsed_content["response"]
                                    # 如果response是字符串，再次解析
                                    if isinstance(response_content, str):
                                        final_content = safe_json_loads(response_content)
                                        if final_content:
                                            content = json.dumps(final_content)
                                        else:
                                            content = response_content
                                    else:
                                        content = json.dumps(response_content)
                                else:
                                    content = json.dumps(parsed_content)
                    elif isinstance(result, dict):
                        # 处理模型返回的嵌套JSON格式
                        if "response" in result:
                            # 如果response字段是字符串形式的JSON，尝试解析
                            response_content = result["response"]
                            parsed_response = safe_json_loads(response_content)
                            if parsed_response:
                                content = json.dumps(parsed_response)
                            else:
                                content = response_content
                        else:
                            content = json.dumps(result)
                    else:
                        content = text
                    
                    if not content:
                        print(f"⚠️ 未获取到响应内容")
                        continue
                    
                    content = content.strip().lower()
                    
                    # 优先 JSON
                    try:
                        json_str = next(s for s in content.splitlines() if s.strip().startswith("{"))
                        parsed = json.loads(json_str)
                        return parsed.get("exists", False)
                    except Exception:
                        pass
                    
                    # 回退关键词判断
                    return "yes" in content and "no" not in content

            except asyncio.TimeoutError:
                print(f"⏳ 存在性检测超时（第 {attempt}/{MAX_RETRIES} 次）")
                await asyncio.sleep(3 * attempt)
                continue

            except aiohttp.ClientConnectionError as e:
                print(f"🌐 网络异常：{e} → session 重建")
                if _session and not _session.closed:
                    await _session.close()
                _session = aiohttp.ClientSession(timeout=BASE_TIMEOUT)
                await asyncio.sleep(5)
                continue

            except asyncio.exceptions.CancelledError:
                print(f"⏸️ 存在性检测请求被取消，重试。")
                await asyncio.sleep(2)
                continue
            except Exception as e:
                print(f"❌ 未知错误：{type(e).__name__} - {e}")
                await asyncio.sleep(2)
                continue

    return False


# async def call_llm_response(prompt: str, image_path: str, point_name: str, task: str = "default") -> Union[Tuple[float, float], str]:
#     # 图像读取与编码
#     with open(image_path, "rb") as img_file:
#         base64_image = base64.b64encode(img_file.read()).decode("utf-8")
#
#     # 构造 OpenAI payload
#     payload = {
#         "model": "gemini-2.5-flash-lite", # gemini-2.0-flash / gpt-4o / qwen-vl-max
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#                     {"type": "text", "text": prompt}
#                 ]
#             }
#         ],
#         "temperature": 0
#     }
#
#     # 发送请求
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, headers=headers, json=payload) as response:
#             try:
#                 result = await response.json()
#                 if "error" in result and result["error"].get("code") == 429:
#                     print(f"❌ 请求频率超限，错误信息: {result['error']['message']}")
#                     return (-1, -1) if task != "diameter_estimation" else "-1"
#             except Exception as e:
#                 print(f"❌ JSON 解析失败：{e}")
#                 print("Raw response:", await response.text())
#                 return (-1, -1) if task != "diameter_estimation" else "-1"
#
#             # 检查 response 是否包含 choices
#             if "choices" not in result:
#                 print("❌ API 返回不包含 'choices' 字段：")
#                 print(json.dumps(result, indent=2))
#                 return (-1, -1) if task != "diameter_estimation" else "-1"
#
#             # 提取文本内容
#             content = result["choices"][0]["message"]["content"]
#             print(f"🧠 模型原始返回内容：\n{content}")
#
#             if task == "diameter_estimation":
#                 return content.strip()
#
#             # 默认坐标解析逻辑
#             try:
#                 # 尝试提取 JSON 代码块
#                 json_code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
#                 if json_code_blocks:
#                     json_str = json_code_blocks[0].strip()
#                 else:
#                     # 否则提取第一个可能是 json 的行
#                     json_str = next(
#                         s for s in content.splitlines()
#                         if s.strip().startswith("{") or s.strip().startswith("[")
#                     )
#
#                 coords_json = json.loads(json_str)
#
#                 # 情形1：list 类型，含 label 和 point
#                 if isinstance(coords_json, list):
#                     for item in coords_json:
#                         if item.get("label") == point_name:
#                             return tuple(item["point"])
#
#                 # 情形2：dict 类型，含 datapoints
#                 elif isinstance(coords_json, dict):
#                     if "datapoints" in coords_json:
#                         dp = coords_json["datapoints"]
#                         if isinstance(dp, list):
#                             for item in dp:
#                                 if isinstance(item, dict) and point_name in item:
#                                     return tuple(item[point_name])
#                         elif isinstance(dp, dict):
#                             if point_name in dp:
#                                 return tuple(dp[point_name])
#
#                 print(f"⚠️ 未能找到 {point_name}，解析后的 JSON：")
#                 print(json.dumps(coords_json, indent=2))
#
#             except Exception as e:
#                 print(f"⚠️ JSON 解析错误：{e}")
#                 print("模型原始返回内容：\n", content)
#
#             return (-1, -1) if task != "diameter_estimation" else "-1"
#
# # 重试函数
# async def call_llm_with_retry(prompt: str, image_path: str, point_name: str, task: str = "default") -> Union[Tuple[float, float], str]:
#     for attempt in range(MAX_ATTEMPTS):
#         coords = await call_llm_response(prompt, image_path, point_name, task)
#         if coords != (-1, -1) and coords != "-1":
#             return coords
#         print("❌ 请求失败，等待并重试...")
#         await asyncio.sleep(7)
#     print(f"⚠️ {point_name} 无法成功预测，已尝试 {MAX_ATTEMPTS} 次")
#     return (-1, -1) if task != "diameter_estimation" else "-1"
#
#
# async def call_llm_point_existence(prompt: str, image_path: str) -> bool:
#     """调用 LLM 判断裁剪图像中是否存在目标散点"""
#     with open(image_path, "rb") as img_file:
#         base64_image = base64.b64encode(img_file.read()).decode("utf-8")
#
#     payload = {
#         "model": "gemini-2.5-flash-lite",  # 或 gemini-2.0-flash
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
#                     {"type": "text", "text": prompt}
#                 ]
#             }
#         ],
#         "temperature": 0
#     }
#
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, headers=headers, json=payload) as response:
#             try:
#                 result = await response.json()
#                 content = result["choices"][0]["message"]["content"].strip().lower()
#
#                 # JSON 解析优先
#                 try:
#                     json_str = next(s for s in content.splitlines() if s.strip().startswith("{"))
#                     parsed = json.loads(json_str)
#                     return parsed.get("exists", False)
#                 except Exception:
#                     pass
#
#                 # 回退到关键词
#                 return "yes" in content and "no" not in content
#
#             except Exception as e:
#                 print(f"❌ LLM 判断异常：{e}")
#                 return False


# 新增函数：自动加载 charts 文件夹下的文件配置
def load_chart_configs(config_dir: str = "chart_configs"):
    config_dir = resolve_local_path(config_dir)
    chart_configs = []
    if not os.path.isdir(config_dir):
        print(f"⚠️ 配置目录不存在，跳过批量配置加载: {config_dir}")
        return chart_configs

    for dirpath, _, filenames in os.walk(config_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(dirpath, filename)
            with open(filepath, "r", encoding="utf-8-sig") as f:
                config = json.load(f)
                chart_configs.append(normalize_image_paths(config))
    return chart_configs


def build_dataset_from_inputs(
        config_path: str,
        chart_path: str,
        chart_id: str | None = None,
        no_grid_chart_path: str | None = None,
        with_grid_chart_path: str | None = None,
        grid_with_grid_chart_path: str | None = None
) -> dict:
    """从单个图表 config 和外部图像路径构造可直接运行的 dataset。"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config 文件不存在: {config_path}")
    if not os.path.isfile(chart_path):
        raise FileNotFoundError(f"chart 图像不存在: {chart_path}")

    with open(config_path, "r", encoding="utf-8-sig") as f:
        dataset = json.load(f)

    required_keys = ["data_points", "x_ticks", "y_ticks", "x_pixels", "y_pixels"]
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise ValueError(f"config 缺少必要字段: {missing}")

    resolved_chart_path = os.path.abspath(chart_path)
    image_paths = {
        "no_grid": os.path.abspath(no_grid_chart_path or resolved_chart_path),
        "with_grid": os.path.abspath(with_grid_chart_path or resolved_chart_path),
        "grid_with_grid": os.path.abspath(grid_with_grid_chart_path or resolved_chart_path),
    }

    for image_type, image_file in image_paths.items():
        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"{image_type} 图像不存在: {image_file}")

    dataset["chart_id"] = chart_id or dataset.get("chart_id") or os.path.splitext(os.path.basename(config_path))[0]
    dataset["image_paths"] = image_paths
    return normalize_image_paths(dataset)


# === 模拟图表数据与标注点 ===
# 新增批量处理配置，支持多数据集配置
DATASET_CONFIGS = load_chart_configs()


#  新增辅助函数：绘制虚线
def draw_dashed_line(draw, start, end, dash_length=5, gap_length=5, fill="gray", width=0.5):
    x1, y1 = start
    x2, y2 = end
    total_length = hypot(x2 - x1, y2 - y1)
    dash_gap = dash_length + gap_length
    num_dashes = int(total_length // dash_gap)

    for i in range(num_dashes + 1):
        start_frac = (i * dash_gap) / total_length
        end_frac = min((i * dash_gap + dash_length) / total_length, 1)
        sx = x1 + (x2 - x1) * start_frac
        sy = y1 + (y2 - y1) * start_frac
        ex = x1 + (x2 - x1) * end_frac
        ey = y1 + (y2 - y1) * end_frac
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)


def draw_crosshair_on_resized_image(
        img_path,  # 可以是 str 路径，或 Image.Image 对象
        coords: list,
        output_path: str,
        color: str = "red",
        length: int = 18,
        thickness: int = 3
):
    """
    在图像上绘制多个十字线。支持传入 PIL.Image 或路径。
    """
    if isinstance(img_path, str):
        img = Image.open(img_path).convert("RGB")
    else:
        img = img_path  # 已是 Image.Image 对象

    draw = ImageDraw.Draw(img)
    for (x, y) in coords:
        draw.line([(x - length // 2, y), (x + length // 2, y)], fill=color, width=thickness)
        draw.line([(x, y - length // 2), (x, y + length // 2)], fill=color, width=thickness)

    img.save(output_path)


def convert_data_coord_to_resized_crop_pixel(
        data_coord, x_ticks, y_ticks, x_pixels, y_pixels,
        crop_origin, crop_size, resize_size
):
    """
    将数据坐标 → 原图像素 → 裁剪图相对像素 → resize 后像素
    参数：
        - data_coord: (x_val, y_val)
        - x_ticks, y_ticks: 原图坐标轴数值
        - x_pixels, y_pixels: 原图 tick 对应像素位置
        - crop_origin: (left, upper) 裁剪区域起点
        - crop_size: (width, height) 裁剪窗口大小
        - resize_size: (w, h) 裁剪图最终resize到的尺寸（默认224x224）
    返回值：
        - resized_crop_pixel_x, resized_crop_pixel_y
    """
    from numpy import interp

    x_val, y_val = data_coord
    left, upper = crop_origin
    crop_w, crop_h = crop_size
    resize_w, resize_h = resize_size

    # Step 1: 原图坐标 → 原图像素
    x_pix = interp(x_val, x_ticks, x_pixels)
    y_pix = interp(y_val, y_ticks, y_pixels)

    # Step 2: 裁剪图中相对像素
    rel_x = x_pix - left
    rel_y = y_pix - upper

    # Step 3: 缩放比例调整
    scale_x = resize_w / crop_w
    scale_y = resize_h / crop_h
    return rel_x * scale_x, rel_y * scale_y


from typing import Optional


from typing import Tuple, List, Optional

async def try_generate_crop_until_point_detected(
        image_path: str,
        pred_coord: Tuple[float, float],
        x_ticks: List[float],
        y_ticks: List[float],
        x_pixels: List[float],
        y_pixels: List[float],
        chart_id: str,
        point_name: str,
        feedback_round: int,
        judge_prompt: str,
        init_crop_size: int = 120,
        max_attempts: int = 5,
        resize_to: Tuple[int, int] = (224, 224),
        max_output_side: int = 1024,   # ✅ 重试裁剪时输出图像的边长上限，防止过大
) -> Optional[Tuple]:
    """
    【重试裁剪专用】

    前提：正常轮已经用 generate_expanded_crop_with_grid_by_diameter 裁过，
          但 LLM 判断该区域内没有目标点，此时才调用本函数。

    逻辑：
    - 第一次尝试（attempt == 0）：
      仍然调用 generate_expanded_crop_with_grid_by_diameter，保持与正常轮一致，
      方便代码复用，也给一个“较精细”的重试版本。
    - 后续尝试（attempt >= 1）：
      使用 crop_draw_ticks_resize，窗口大小 crop_size 逐轮翻倍。
      但输出尺寸不再强制固定为 224×224，而是随 crop_size 一起变大，
      避免 800×800 的区域被粗暴压缩到 224×224 造成严重模糊。

    - 每次生成裁剪图后调用 call_llm_point_existence 判断是否包含目标点；
      若检测到点，则立即返回该轮的裁剪结果；
      若连续 max_attempts 轮仍未检测到点，则返回 None。

    注意：
    - 重试轮不再要求遵守“按散点直径绘制网格”的法则，只需清晰显示更大范围，
      让 LLM 更有机会看到目标点及其周围结构。
    """

    crop_size = init_crop_size

    for attempt in range(max_attempts):

        # ---------- 本轮输出尺寸策略 ----------
        # 第 0 轮：沿用原始 resize_to（通常是 224×224），行为与正常轮一致
        # 后续轮：当 crop_size 明显大于 224 时，输出尺寸随 crop_size 变大，
        #         并限制在 max_output_side 以内，避免图像过大。
        if attempt == 0:
            cur_resize_to = resize_to
        else:
            if crop_size <= max(resize_to):
                # 小窗口：保持 224×224 或用户指定的默认尺寸
                cur_resize_to = resize_to
            else:
                # 大窗口：输出尺寸跟随窗口变大，但限制上限
                side = min(crop_size, max_output_side)
                cur_resize_to = (side, side)

        print(
            f"[try_generate_crop] attempt={attempt}, "
            f"crop_size={crop_size}px, output_size={cur_resize_to}"
        )

        # ---------- 第一次尝试：仍用 canvas 版本 ----------
        if attempt == 0:
            res = await generate_expanded_crop_with_grid_by_diameter(
                image_path=image_path,
                pred_coord=pred_coord,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
                x_pixels=x_pixels,
                y_pixels=y_pixels,
                chart_id=chart_id,
                point_name=f"{point_name}_rt{attempt}",
                feedback_round=feedback_round,
                base_crop_size=crop_size,
                resize_to=cur_resize_to,
            )
        else:
            # ---------- 后续重试：普通裁剪 + 强制网格 ----------
            res = crop_draw_ticks_resize(
                image_path=image_path,
                pred_coord=pred_coord,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
                x_pixels=x_pixels,
                y_pixels=y_pixels,
                chart_id=chart_id,
                point_name=f"{point_name}_rt{attempt}",
                feedback_round=feedback_round,
                window_size=crop_size,
                output_size=cur_resize_to,
                return_ticks=True,
                x_grid_density=1,  # 强制绘制网格（所有 tick）
                y_grid_density=1,
            )

        out_path = res[0]  # 裁剪图路径（两种裁剪函数的第一个返回值都应为路径）

        # ---------- 调用 LLM 判断裁剪内是否包含目标点 ----------
        exists = await call_llm_point_existence(judge_prompt, out_path)
        if exists:
            print(f"✅ 重试裁剪第 {attempt + 1} 次检测到点: {point_name}")
            return res

        print(
            f"⚠️ 重试裁剪第 {attempt + 1} 次未检测到点，"
            f"下次将裁剪窗口扩大至 {crop_size * 2}px"
        )
        crop_size *= 2

    print(f"❌ 重试裁剪超过最大次数 ({max_attempts}) 仍未检测到点: {point_name}")
    return None



#  主函数
def crop_draw_ticks_resize(
        image_path: str,
        pred_coord: tuple,
        x_ticks: list,
        y_ticks: list,
        x_pixels: list,
        y_pixels: list,
        chart_id: str,
        point_name: str,
        feedback_round: int,
        window_size: int = 120,
        output_size: tuple = (224, 224),
        font_size: int = 12,
        x_grid_density: int = 0,
        y_grid_density: int = 0,
        crosshair_length: int = 18,
        crosshair_thickness: int = 3,
        return_ticks: bool = False,
        dash_style: str = "dot"
) -> tuple:
    from PIL import ImageFont, ImageDraw, Image
    import numpy as np
    from math import hypot
    import os

    # 根据样式设定 dash 长度与间距
    if dash_style == "short":
        dash_length, gap_length = 2, 2
    elif dash_style == "long":
        dash_length, gap_length = 12, 6
    elif dash_style == "dot":
        dash_length, gap_length = 1, 4
    else:
        dash_length, gap_length = 5, 5

    # 虚线画线工具
    def draw_dashed_line(draw, start, end, fill="gray", width=1):
        x1, y1 = start
        x2, y2 = end
        total_length = hypot(x2 - x1, y2 - y1)
        dash_gap = dash_length + gap_length
        num_dashes = int(total_length // dash_gap)

        for i in range(num_dashes + 1):
            start_frac = (i * dash_gap) / total_length
            end_frac = min((i * dash_gap + dash_length) / total_length, 1)
            sx = x1 + (x2 - x1) * start_frac
            sy = y1 + (y2 - y1) * start_frac
            ex = x1 + (x2 - x1) * end_frac
            ey = y1 + (y2 - y1) * end_frac
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

    x_mapper = lambda v: np.interp(v, x_ticks, x_pixels)
    y_mapper = lambda v: np.interp(v, y_ticks, y_pixels)
    cx, cy = int(x_mapper(pred_coord[0])), int(y_mapper(pred_coord[1]))

    left = max(cx - window_size // 2, 0)
    upper = max(cy - window_size // 2, 0)
    right = left + window_size
    lower = upper + window_size

    img = Image.open(image_path).convert("RGB")
    cropped = img.crop((left, upper, right, lower))
    draw = ImageDraw.Draw(cropped)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    new_x_ticks, new_y_ticks, new_x_pixels, new_y_pixels = [], [], [], []
    img_w, img_h = cropped.width, cropped.height

    def format_tick_value(value):
        rounded_value = round(value, 2)
        if abs(rounded_value - round(rounded_value)) < 1e-6:
            return f"{int(round(rounded_value))}"
        else:
            return f"{rounded_value:.2f}"

    def draw_tick_text(px, py, text, horizontal=True):
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if horizontal:
            # 横轴 → 改成竖排显示
            txt_img = Image.new("RGBA", (text_w * 2, text_h * 2), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((text_w, text_h), text, fill="red", font=font, anchor="mm")

            rotated = txt_img.rotate(90, expand=1, resample=Image.BICUBIC)  # 旋转90度
            rw, rh = rotated.size

            # 贴在底部，稍微留一点空隙
            cropped.paste(rotated, (int(px - rw / 2), img_h - rh - 2), rotated)

        else:
            # 纵轴 → 保持水平写
            draw.text((5, py - text_h / 2), text, fill="red", font=font)

    # ---- X方向 ----
    # 原始 tick 永远画线
    for val, pix in zip(x_ticks, x_pixels):
        if left <= pix <= right:
            rel_px = pix - left
            draw_dashed_line(draw, (rel_px, 0), (rel_px, img_h), fill="gray", width=1)
            draw_tick_text(rel_px, 0, format_tick_value(val), horizontal=True)
            new_x_ticks.append(val)
            new_x_pixels.append(rel_px)

    # 插值 tick
    if x_grid_density > 0:
        for i in range(len(x_ticks) - 1):
            val1, val2 = x_ticks[i], x_ticks[i + 1]
            pix1, pix2 = x_pixels[i], x_pixels[i + 1]
            for j in range(1, x_grid_density + 1):
                interp_val = val1 + (val2 - val1) * j / (x_grid_density + 1)
                interp_pix = pix1 + (pix2 - pix1) * j / (x_grid_density + 1)
                if left <= interp_pix <= right:
                    rel_px = interp_pix - left
                    draw_dashed_line(draw, (rel_px, 0), (rel_px, img_h), fill="lightgray", width=1)
                    draw_tick_text(rel_px, 0, format_tick_value(interp_val), horizontal=True)
                    new_x_ticks.append(interp_val)
                    new_x_pixels.append(rel_px)

    # ---- Y方向 ----
    # 原始 tick 永远画线
    for val, pix in zip(y_ticks, y_pixels):
        if upper <= pix <= lower:
            rel_py = pix - upper
            draw_dashed_line(draw, (0, rel_py), (img_w, rel_py), fill="gray", width=1)
            draw_tick_text(0, rel_py, format_tick_value(val), horizontal=False)
            new_y_ticks.append(val)
            new_y_pixels.append(rel_py)

    # 插值 tick
    if y_grid_density > 0:
        for i in range(len(y_ticks) - 1):
            val1, val2 = y_ticks[i], y_ticks[i + 1]
            pix1, pix2 = y_pixels[i], y_pixels[i + 1]
            for j in range(1, y_grid_density + 1):
                interp_val = val1 + (val2 - val1) * j / (y_grid_density + 1)
                interp_pix = pix1 + (pix2 - pix1) * j / (y_grid_density + 1)
                if upper <= interp_pix <= lower:
                    rel_py = interp_pix - upper
                    draw_dashed_line(draw, (0, rel_py), (img_w, rel_py), fill="lightgray", width=1)
                    draw_tick_text(0, rel_py, format_tick_value(interp_val), horizontal=False)
                    new_y_ticks.append(interp_val)
                    new_y_pixels.append(rel_py)

    resized = cropped.resize(output_size, Image.LANCZOS)
    temp_dir = os.path.join("temp", chart_id)
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, f"cropped_{point_name}_{feedback_round}.png")
    resized.save(save_path)

    def pixel_to_cropped_coords(x_px, y_px):
        return x_px - left, y_px - upper

    return (
        save_path,
        new_x_ticks, new_y_ticks,
        new_x_pixels, new_y_pixels,
        pixel_to_cropped_coords,
        left, upper
    )


def generate_overlayed_image_multi_with_mapping(
        original_img_path: str,
        pred_coords: list,
        x_ticks: list,
        y_ticks: list,
        x_pixels: list,
        y_pixels: list,
        output_path: str,
        feedback_round: int = 1,
        draw_all: bool = False  # ✅ 控制是否绘制所有轮次（保留参数但当前行为只绘制最新）
):
    """
    在图上绘制预测点（用于反馈引导）：
    - 默认只绘制最近一轮的预测点
    - 十字线短、加粗、无文本标签
    """
    from numpy import interp

    def build_axis_mapping(tick_values, tick_pixels):
        return lambda v: interp(v, tick_values, tick_pixels)

    x_mapper = build_axis_mapping(x_ticks, x_pixels)
    y_mapper = build_axis_mapping(y_ticks, y_pixels)

    img = Image.open(original_img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = ["red", "purple", "orange", "green", "blue"]

    # 只绘制最后一个预测点（或所有点，仅当 draw_all=True）
    coords_to_draw = pred_coords if draw_all else ([pred_coords[-1]] if pred_coords else [])

    for idx, coord in enumerate(coords_to_draw):
        try:
            x_val = float(coord[0])
            y_val = float(coord[1])
        except Exception as e:
            print(f"⚠️ 坐标格式异常: {coord}，跳过")
            continue

        x_pixel = int(x_mapper(x_val))
        y_pixel = int(y_mapper(y_val))

        # 越界检查
        if not (0 <= x_pixel < img.width and 0 <= y_pixel < img.height):
            print(f"⚠️ 跳过越界点：({x_val:.1f}, {y_val:.1f}) → ({x_pixel}, {y_pixel})")
            continue

        color = colors[min(idx, len(colors) - 1)]

        # 绘图参数
        # line_len = 8       # ✅ 十字线长度
        # line_width = 2     # ✅ 加粗线宽度
        # line_len = 12  # ✅ 十字线长度
        # line_width = 4  # ✅ 加粗线宽度
        line_len = 9  # ✅ 十字线长度
        line_width = 3  # ✅ 加粗线宽度

        # ✅ 绘制加粗的局部网格十字线（无文字标签）
        draw.line((x_pixel - line_len, y_pixel, x_pixel + line_len, y_pixel), fill=color, width=line_width)
        draw.line((x_pixel, y_pixel - line_len, x_pixel, y_pixel + line_len), fill=color, width=line_width)

        print(f"✅ 绘制点: ({x_val:.1f}, {y_val:.1f}) → ({x_pixel}, {y_pixel})")

    img.save(output_path)
    print(f"🖼️ 多轮反馈图已保存至: {output_path}")


# def generate_prompt(
#         item_name: str,
#         prompt_type: str,
#         x_ticks: list,
#         y_ticks: list,
#         pred_feedback: list = None,
#         feedback_round: int = 0,
#         current_round: int = 1
# ) -> str:
#     x_tick_str = ", ".join(str(x) for x in x_ticks)
#     y_tick_str = ", ".join(str(y) for y in y_ticks)
#
#     print(f"🎯 generate_prompt 收到 x_ticks: {x_ticks}")
#     print(f"🎯 generate_prompt 收到 y_ticks: {y_ticks}")
#
#     # === 1. Baseline Prompt ===
#     if prompt_type == "baseline":
#         return f'''
#         You are given a chart image.
#         Please extract the coordinates of the circle which represents [{item_name}].
#         Identify the graphical element that represents the target data item and extract its coordinates based on its visual center, not the location of any associated label or annotation.
#         Only respond in this JSON format:
#          {{"datapoints": [{{"{item_name}": [x, y]}}]}}
#          '''
#     # === 2. Grid Prompt ===
#     elif prompt_type == "grid":
#         return f'''
#         You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
#         - X-axis ticks: [{x_tick_str}]
#         - Y-axis ticks: [{y_tick_str}]
#         These grid lines divide the chart into rectangular cells aligned with axis ticks.
#         The center of each visual mark (e.g., a circle) falls into a grid cell.
#         The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
#         Your task is to accurately extract the coordinates of the circle center representing [{item_name}] by:
#         - Locating the center of the circle representing the data item;
#         - Identifying its position between tick grid lines and using interpolation to estimate the (x, y) values of the circle center representing [{item_name}].
#
#         Only respond in this JSON format:
#         {{"datapoints": [{{"{item_name}": [x, y]}}]}}
#         '''
#
#     # === 3. Grid + Feedback Prompt ===
#     elif prompt_type == "feedback":
#         base_prompt = f'''
#         You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
#         - X-axis ticks: [{x_tick_str}]
#         - Y-axis ticks: [{y_tick_str}]
#         These grid lines divide the chart into rectangular cells aligned with axis ticks.
#         The center of each visual mark (e.g., a circle) falls into a grid cell.
#         The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
#         Your task is to accurately extract the coordinates of the circle center representing [{item_name}] by:
#         - Locating the center of the circle representing the data item;
#         - Identifying its position between tick grid lines and using interpolation to estimate the (x, y) values of the circle center representing [{item_name}].
#         '''
#
#         if pred_feedback and isinstance(pred_feedback, list) and len(pred_feedback) >= 1:
#             pred = pred_feedback[-1]
#
#             # ===keep refine feedback for offset direction and distance===
#             base_prompt = f'''
#             You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
#             - X-axis ticks: [{x_tick_str}]
#             - Y-axis ticks: [{y_tick_str}]
#             These grid lines divide the chart into rectangular cells aligned with axis ticks.
#             The center of each visual mark (e.g., a circle) falls into a grid cell.
#             The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
#             The predicted coordinates in the previous round is drawn as a red crosshair on the given chart.
#             Please follow the steps below to refine your estimate for the coordinates of the circle representing [{item_name}]:
#             Step 1: Compare the center of the circle representing [{item_name}] to the red crosshair’s center point, which marks the previous prediction at coordinates (x = {pred[0]:.2f}, y = {pred[1]:.2f}).
#             Step 2: Accurately identify the horizontal and vertical offset direction and distance between the red crosshair’s intersection and the exact center of the circle representing [{item_name}].
#             Step 3: Apply both the direction and magnitude of the offset along the horizontal and vertical axes to fine-tune your x and y predictions, so that they can minimize the offset with the targeted circle center representing [{item_name}].
#             '''
#         base_prompt += f'''
#             You must respond in this JSON format: {{"datapoints": [{{"{item_name}": [x, y]}}]}}
#                 '''
#         return base_prompt
#
#
#     # ===very good for c2 and c4===
#     elif prompt_type == "feedback_cropped":
#         base_prompt = f'''
#         You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
#         - X-axis ticks: [{x_tick_str}]
#         - Y-axis ticks: [{y_tick_str}]
#         These grid lines divide the chart into rectangular cells aligned with axis ticks.
#         The center of each visual mark (e.g., a circle) falls into a grid cell.
#         The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
#         Your task is to accurately extract the coordinates of the circle center representing [{item_name}] by:
#         - Locating the center of the circle representing [{item_name}];
#         - Identifying its position between tick grid lines and using interpolation to estimate the (x, y) values of the circle center representing [{item_name}].
#         '''
#
#         base_prompt += f'''
#             Only respond in this JSON format:
#             {{"datapoints": [{{"{item_name}": [x, y]}}]}}
#             Important: Do not output null. If unsure, make a reasonable estimate of the position.
#         '''
#         return base_prompt
#
#     else:
#         raise ValueError(f"Unknown prompt_type: {prompt_type}")

def generate_prompt(
        item_name: str,
        prompt_type: str,
        x_ticks: list,
        y_ticks: list,
        pred_feedback: list = None,
        x_pixels: list = None,
        y_pixels: list = None,
        feedback_round: int = 0,
        current_round: int = 1
) -> str:
    import numpy as np

    x_tick_str = ", ".join(str(x) for x in x_ticks)
    y_tick_str = ", ".join(str(y) for y in y_ticks)

    print(f"🎯 generate_prompt 收到 x_ticks: {x_ticks}")
    print(f"🎯 generate_prompt 收到 y_ticks: {y_ticks}")
    json_contract = (
        f'CRITICAL OUTPUT RULE: Return only valid JSON and nothing else. '
        f'Do not write explanations, steps, Markdown fences, or prose. '
        f'The complete response must be exactly one JSON object like this: '
        f'{{"datapoints": [{{"{item_name}": [x, y]}}]}}. '
        f'Use numeric x and y values; if uncertain, make the best estimate.'
    )

    # === 1. Baseline Prompt ===
    if prompt_type == "baseline":
        return f'''
        {json_contract}

        You are given a chart image.
        Please extract the coordinates of the circle which represents [{item_name}].
        Identify the graphical element that represents the target data item and extract its coordinates based on its visual center, not the location of any associated label or annotation.        
         '''

    # === 2. Grid Prompt ===
    elif prompt_type == "grid":
        return f'''
        {json_contract}

        You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        These grid lines divide the chart into rectangular cells aligned with axis ticks.
        The center of each visual mark (e.g., a circle) falls into a grid cell.
        The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
        Your task is to accurately extract the coordinates of the circle center representing [{item_name}] by:
        - Locating the center of the circle representing the data item;
        - Identifying its position between tick grid lines and using interpolation to estimate the (x, y) values of the circle center representing [{item_name}].
        '''

    # === 3. Grid + Feedback Prompt ===
    elif prompt_type == "feedback":
        base_prompt = f'''
        {json_contract}

        You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        These grid lines divide the chart into rectangular cells aligned with axis ticks.        
        The center of each visual mark (e.g., a circle) falls into a grid cell.
        The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
        Your task is to accurately extract the coordinates of the circle center representing [{item_name}] by:
        - Locating the center of the circle representing the data item;
        - Identifying its position between tick grid lines and using interpolation to estimate the (x, y) values of the circle center representing [{item_name}].
        '''

        # === 🔥 新增逻辑：自动从模型预测值 -> 像素 -> 红十字真实数据坐标 ===
        if pred_feedback and isinstance(pred_feedback, (list, tuple)) and len(pred_feedback) == 2 \
                and x_pixels is not None and y_pixels is not None:
            # -------------------------------------
            # ✅ ① 数据坐标 → 像素坐标
            # x 正常插值
            pred_x_pixel = float(np.interp(pred_feedback[0], x_ticks, x_pixels))

            # y 必须反转映射：高数值 → 小像素
            pred_y_pixel = float(np.interp(pred_feedback[1],
                                           y_ticks,  # 数值升序
                                           y_pixels[::-1]))  # 像素倒序
            # 举例：y=74.0 → 低像素值 75（靠上）
            #       y=56.0 → 高像素值 512（靠下）

            # -------------------------------------
            # ✅ ② 像素坐标边界裁剪
            pred_x_pixel = int(np.clip(pred_x_pixel, 0, max(x_pixels)))
            pred_y_pixel = int(np.clip(pred_y_pixel, 0, max(y_pixels)))

            # -------------------------------------
            # ✅ ③ 像素坐标 → 数据坐标（逆映射）
            pred_x_val = float(np.interp(pred_x_pixel, x_pixels, x_ticks))
            pred_y_val = float(np.interp(pred_y_pixel,
                                         y_pixels[::-1],  # 像素倒序
                                         y_ticks))  # 数值升序

            print(f"🔁 模型预测值 ({pred_feedback[0]:.2f}, {pred_feedback[1]:.2f}) "
                  f"→ 像素 ({pred_x_pixel}, {pred_y_pixel}) "
                  f"→ 红十字真实数据坐标 ({pred_x_val:.2f}, {pred_y_val:.2f})")

            # # === 🔥 新增逻辑：自动从模型预测值 -> 像素 -> 红十字真实数据坐标 ===
        # if pred_feedback and isinstance(pred_feedback, (list, tuple)) and len(pred_feedback) == 2 \
        #         and x_pixels is not None and y_pixels is not None:
        #
        #     # ① 模型预测（数据坐标）→ 像素
        #     pred_x_pixel = float(np.interp(pred_feedback[0], x_ticks, x_pixels))
        #     pred_y_pixel = float(np.interp(pred_feedback[1], y_ticks, y_pixels))
        #
        #     # ② 模拟红十字绘制后的实际像素位置（取整）
        #     pred_x_pixel = int(np.clip(pred_x_pixel, 0, max(x_pixels)))
        #     pred_y_pixel = int(np.clip(pred_y_pixel, 0, max(y_pixels)))
        #
        #     # ③ 像素 → 真实红十字对应的数据坐标（逆映射）
        #     pred_x_val = float(np.interp(pred_x_pixel, x_pixels, x_ticks))
        #     pred_y_val = float(np.interp(pred_y_pixel, y_pixels, y_ticks))
        #
        #     # ④ 打印验证
        #     print(f"🔁 模型预测值 ({pred_feedback[0]:.2f}, {pred_feedback[1]:.2f}) "
        #           f"→ 像素 ({pred_x_pixel}, {pred_y_pixel}) → "
        #           f"红十字真实坐标 ({pred_x_val:.2f}, {pred_y_val:.2f})")

            # ⑤ 构建反馈提示
            base_prompt = f'''
            {json_contract}

            You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
            - X-axis ticks: [{x_tick_str}]
            - Y-axis ticks: [{y_tick_str}]
            These grid lines divide the chart into rectangular cells aligned with axis ticks.
            The center of each visual mark (e.g., a circle) falls into a grid cell.
            The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
            Your task is to accurately extract the coordinates of the circle representing [{item_name}].
            The predicted coordinates in the previous round is drawn as a red crosshair on the given chart.
            Please follow the steps below to refine your estimate for the coordinates of the circle representing [{item_name}]:
            Step 1: Find the circle that appears closest to the red crosshair and clarify its characteristics such as color, shape, and other visual features to help locate its position within the grid, as it is most likely the one corresponding to [{item_name}].            
            Step 1: Compare the circle center representing [{item_name}] to the red crosshair’s center, which marks the previous prediction at (x = {pred_x_val:.2f}, y = {pred_y_val:.2f}) in data coordinates.            
            Step 3: Accurately identify the horizontal and vertical offset direction and distance between the red crosshair’s intersection and the exact center of the circle.
            Step 4: Apply both the direction and magnitude of the offset along the horizontal and vertical axes to fine-tune your x and y predictions, so that they can minimize the offset with the targeted circle center representing [{item_name}]. Do follow the correct direction to minimize the offset to the circle center on both the x and y values. Note the spacing between adjacent grid lines is 2.5 units in both the horizontal (x) and vertical (y) directions, and refer to the relative distances among the red crosshair, grid lines, and the circle center to interpolate carefully—avoiding excessive correction caused by misjudging the coordinate scale. 
            '''

        base_prompt += f'''
            You must respond in this JSON format: {{"datapoints": [{{"{item_name}": [x, y]}}]}}
                '''
        return base_prompt

    # ===very good for c2 and c4===
    elif prompt_type == "feedback_cropped":
        base_prompt = f'''
        {json_contract}

        You are analyzing a chart that includes **reference grid lines**, in which the vertical and horizontal lines are aligned with the X-axis ticks and Y-axis ticks as follows:
        - X-axis ticks: [{x_tick_str}]
        - Y-axis ticks: [{y_tick_str}]
        These grid lines divide the chart into rectangular cells aligned with axis ticks.        
        The center of each visual mark (e.g., a circle) falls into a grid cell.
        The coordinate of the circle center can be determined by doing interpolation between the x/y values represented by the two adjacent grid lines that the circle center falls in.
        Your task is to accurately extract the coordinates of the circle center representing [{item_name}] by:
        - Locating the center of the circle representing [{item_name}];
        - Identifying its position between tick grid lines and using interpolation to estimate the (x, y) values of the circle center representing [{item_name}].
        '''

        base_prompt += f'''
            Return only this JSON format:
            {{"datapoints": [{{"{item_name}": [x, y]}}]}}
            Important: Do not output null. If unsure, make a reasonable estimate of the position.
        '''
        return base_prompt

    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")


def build_axis_mapping(tick_values, tick_pixels):
    """
    输入：tick数值（如 [0, 1, ..., 10]），对应像素（如 [60, 130, ..., 760]）
    输出：一个函数，可以输入任意坐标值，返回插值得到的像素位置
    """
    return lambda v: np.interp(v, tick_values, tick_pixels)


# === pixel误差计算 ===
def compute_pixel_relative_error_xy(pred_px, pred_py, gt_px, gt_py, img_width, img_height):
    x_rel_err = abs(pred_px - gt_px) / img_width
    y_rel_err = abs(pred_py - gt_py) / img_height
    return round(x_rel_err, 4), round(y_rel_err, 4)


# # === mae误差计算 ===
def compute_mae(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    if (
            pred is None or gt is None or
            len(pred) != 2 or len(gt) != 2 or
            pred[0] is None or pred[1] is None or
            gt[0] is None or gt[1] is None
    ):
        print(f"⚠️ 无法计算 MAE，包含 None：pred={pred}, gt={gt}")
        return None  # 或 return float('inf') 取决于评估策略
    return round(abs(pred[0] - gt[0]) + abs(pred[1] - gt[1]), 2)


# def compute_mae(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
#     return round(abs(pred[0] - gt[0]) + abs(pred[1] - gt[1]), 2)

def compute_re(pred: Tuple[float, float], gt: Tuple[float, float]) -> Tuple[float, float]:
    if (
            pred is None or gt is None or
            len(pred) != 2 or len(gt) != 2 or
            pred[0] is None or pred[1] is None or
            gt[0] is None or gt[1] is None
    ):
        print(f"⚠️ 无法计算 RE，包含 None：pred={pred}, gt={gt}")
        return None, None  # 或其他 sentinel 值

    if gt[0] == 0:
        x_re = -1
    else:
        x_re = abs(pred[0] - gt[0]) / (abs(gt[0]) + 1e-6)

    if gt[1] == 0:
        y_re = -1
    else:
        y_re = abs(pred[1] - gt[1]) / (abs(gt[1]) + 1e-6)

    return round(x_re, 4), round(y_re, 4)


# --- 新增：仅保留 grid+with_grid 的最后一轮预测 ---
def filter_final_round_for_feedback(df: pd.DataFrame) -> pd.DataFrame:
    # 标记每个配置下的轮次
    df["round_index"] = df.groupby(["chart_id", "point_name", "prompt_type", "image_type"]).cumcount()

    # 仅对 grid+with_grid 保留最后一轮，其它保留全部
    # mask_feedback = (df["prompt_type"] == "feedback") & (df["image_type"] == "with_grid")
    mask_feedback = (df["prompt_type"].isin(
        ["feedback", "feedback_cropped", "feedback_crop_final", "feedback_crop_from_feedback"])) & (
                                df["image_type"] == "with_grid")

    # ✅ 关键：transform 会返回与 df 等长的 Series，可以直接比较
    df["max_round_index"] = df.groupby(["chart_id", "point_name", "prompt_type", "image_type"])[
        "round_index"].transform("max")

    # ✅ 只保留非 feedback 的所有 or feedback 类型中最后一轮
    df_filtered = df[~mask_feedback | (df["round_index"] == df["max_round_index"])].drop(columns=["max_round_index"])

    return df_filtered


# def evaluate_results(df: pd.DataFrame, result_dir: str):
#     # -- 先清洗预测失败的数据
#     df_clean = df[(df["pred_x"] != -1) & (df["pred_y"] != -1)]
#
#     # -- 汇总每种prompt+image配置下的平均误差（包括新增的 x/y MAE）
#     summary = df_clean.groupby(["prompt_type", "image_type"]).agg(
#         avg_mae=("mae", "mean"),
#         std_mae=("mae", "std"),
#         avg_px_rel_x=("pixel_rel_x", "mean"),
#         std_px_rel_x=("pixel_rel_x", "std"),
#         avg_px_rel_y=("pixel_rel_y", "mean"),
#         std_px_rel_y=("pixel_rel_y", "std"),
#         avg_x_mae=("pred_x", lambda x: (x - df_clean.loc[x.index, "gt_x"]).abs().mean()),
#         avg_y_mae=("pred_y", lambda y: (y - df_clean.loc[y.index, "gt_y"]).abs().mean()),
#         avg_x_re=("x_re", "mean"),
#         std_x_re=("x_re", "std"),
#         avg_y_re=("y_re", "mean"),
#         std_y_re=("y_re", "std")
#     ).reset_index()
#
#     # -- 保存误差统计表
#     summary.to_csv(os.path.join(result_dir, "mae_summary.csv"), index=False)
#     print("📄 已保存评估汇总表 mae_summary.csv")
#
#     # -- 保存各配置差异对比表
#     pivot = summary.pivot_table(
#         index=["prompt_type", "image_type"],
#         values=["avg_mae"]
#     ).reset_index()
#     pivot.to_csv(os.path.join(result_dir, "prompt_comparison.csv"), index=False)
#     print("📄 已保存提示方式对比表 prompt_comparison.csv")
#
#     # -- 原图：总 MAE + 相对误差
#     labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
#     x = range(len(labels))
#     fig, ax1 = plt.subplots(figsize=(12, 6))
#     bar_width = 0.35
#
#     bars_mae = ax1.bar(x, summary["avg_mae"], width=bar_width, label="MAE", color="#D98880")
#     ax1.set_ylabel("MAE (absolute error)", fontsize=12, color="#D98880")
#     ax1.tick_params(axis='y', labelcolor="#D98880")
#     ax1.set_ylim(0, summary["avg_mae"].max() * 1.2)
#
#     for bar in bars_mae:
#         height = bar.get_height()
#         ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                      xytext=(0, 3), textcoords="offset points",
#                      ha='center', va='bottom', fontsize=10, color="#B03A2E")
#
#     ax2 = ax1.twinx()
#     offset = bar_width
#     bars_xre = ax2.bar([i + offset for i in x], summary["avg_x_re"], width=bar_width / 2,
#                        label="X Relative Error", color="#82E0AA")
#     bars_yre = ax2.bar([i + offset + bar_width / 2 for i in x], summary["avg_y_re"], width=bar_width / 2,
#                        label="Y Relative Error", color="#5DADE2")
#
#     ax2.set_ylabel("Relative Error", fontsize=12, color="#5DADE2")
#     ax2.tick_params(axis='y', labelcolor="#5DADE2")
#     ax2.set_ylim(0, max(summary["avg_x_re"].max(), summary["avg_y_re"].max()) * 1.5)
#
#     for bars in [bars_xre, bars_yre]:
#         for bar in bars:
#             height = bar.get_height()
#             ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
#                          xytext=(0, 3), textcoords="offset points",
#                          ha='center', va='bottom', fontsize=9, color="#2874A6")
#
#     # plt.xticks([i + bar_width/2 for i in x], labels, rotation=20, fontsize=11)
#     plt.xticks([i + bar_width / 2 for i in x], labels, rotation=60, fontsize=11)
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=11)
#
#     plt.title("Average MAE and Relative Errors by Prompt+Image Setting", fontsize=14, fontweight="bold")
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "error_comparison_plot_optimized_with_labels.png"))
#     print("📊 已保存高级版误差对比图 error_comparison_plot_optimized_with_labels.png")
#     plt.close()
#
#     # -- 新图：X/Y 分开 MAE 图
#     plt.figure(figsize=(10, 6))
#     plt.bar([i - bar_width / 2 for i in x], summary["avg_x_mae"], width=bar_width, label="X MAE", color="#85C1E9")
#     plt.bar([i + bar_width / 2 for i in x], summary["avg_y_mae"], width=bar_width, label="Y MAE", color="#F5B7B1")
#
#     for i, (xmae, ymae) in enumerate(zip(summary["avg_x_mae"], summary["avg_y_mae"])):
#         plt.text(i - bar_width / 2, xmae + 0.05, f"{xmae:.2f}", ha='center', va='bottom', fontsize=10, color="#21618C")
#         plt.text(i + bar_width / 2, ymae + 0.05, f"{ymae:.2f}", ha='center', va='bottom', fontsize=10, color="#922B21")
#
#     # plt.xticks(x, labels, rotation=20)
#     plt.xticks(x, labels, rotation=60)
#     plt.ylabel("MAE (per axis)")
#     plt.title("Separate MAE for X and Y by Prompt+Image Setting")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "y_mae_comparison_plot.png"))
#     print(f"📊 已保存MAE对比图 {os.path.join(result_dir, 'y_mae_comparison_plot.png')}")
#     plt.close()
#
#     # === 新图：Pixel Relative Error (X / Y) ===
#     plt.figure(figsize=(10, 6))
#     bar_width = 0.35
#     plt.bar([i - bar_width / 2 for i in x], summary["avg_px_rel_x"], width=bar_width, label="Pixel X Rel Error",
#             color="#AED6F1")
#     plt.bar([i + bar_width / 2 for i in x], summary["avg_px_rel_y"], width=bar_width, label="Pixel Y Rel Error",
#             color="#F9E79F")
#
#     for i, (px, py) in enumerate(zip(summary["avg_px_rel_x"], summary["avg_px_rel_y"])):
#         plt.text(i - bar_width / 2, px + 0.001, f"{px:.4f}", ha='center', fontsize=9, color="#21618C")
#         plt.text(i + bar_width / 2, py + 0.001, f"{py:.4f}", ha='center', fontsize=9, color="#B7950B")
#
#     # plt.xticks(x, labels, rotation=20)
#     plt.xticks(x, labels, rotation=60)
#     plt.ylabel("Relative Pixel Error (by width/height)")
#     plt.title("Pixel-level Relative Error (X and Y) by Prompt+Image Setting")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "pixel_relative_error_xy_plot.png"))
#     print(f"📊 已保存像素相对误差图 {os.path.join(result_dir, 'pixel_relative_error_xy_plot.png')}")
#     plt.close()
#
#     # === 综合图：MAE + 相对误差 + 像素误差
#     fig, ax1 = plt.subplots(figsize=(12, 6))
#     bar_width = 0.15
#
#     # MAE 柱状图（左轴）
#     bars_mae = ax1.bar([i - 1.5 * bar_width for i in x], summary["avg_mae"], width=bar_width, label="MAE",
#                        color="#D98880")
#     ax1.set_ylabel("MAE (absolute error)", fontsize=12, color="#D98880")
#     ax1.tick_params(axis='y', labelcolor="#D98880")
#     ax1.set_ylim(0, summary["avg_mae"].max() * 1.3)
#
#     # 标注 MAE 数值
#     for i, bar in enumerate(bars_mae):
#         height = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.1, f"{height:.1f}", ha='center', fontsize=9,
#                  color="#B03A2E")
#
#     # 相对误差 + 像素误差共用右轴
#     ax2 = ax1.twinx()
#     ax2.set_ylabel("Relative Errors (normalized)", fontsize=12)
#     ax2.tick_params(axis='y')
#
#     # 数值空间的相对误差
#     bars_xre = ax2.bar([i - 0.5 * bar_width for i in x], summary["avg_x_re"], width=bar_width, label="X Rel Error",
#                        color="#82E0AA")
#     bars_yre = ax2.bar([i + 0.5 * bar_width for i in x], summary["avg_y_re"], width=bar_width, label="Y Rel Error",
#                        color="#5DADE2")
#
#     # 图像空间的像素相对误差
#     bars_px_x = ax2.bar([i + 1.5 * bar_width for i in x], summary["avg_px_rel_x"], width=bar_width,
#                         label="Pixel X Rel Err", color="#AED6F1")
#     bars_px_y = ax2.bar([i + 2.5 * bar_width for i in x], summary["avg_px_rel_y"], width=bar_width,
#                         label="Pixel Y Rel Err", color="#F9E79F")
#
#     # 标注 pixel 相对误差
#     for bars, color in zip([bars_xre, bars_yre, bars_px_x, bars_px_y], ["#1E8449", "#21618C", "#2980B9", "#B7950B"]):
#         for bar in bars:
#             height = bar.get_height()
#             ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f"{height:.3f}", ha='center', fontsize=8,
#                      color=color)
#
#     # 设置坐标轴与标签
#     # plt.xticks(x, labels, rotation=20, fontsize=11)
#     plt.xticks(x, labels, rotation=60, fontsize=11)
#
#     ax1.set_title("Combined MAE + Relative Errors + Pixel-Level Errors", fontsize=14, fontweight="bold")
#
#     # 合并图例
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=10)
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(result_dir, "error_comparison_plot.png"))
#     print(f"📊 已保存综合误差图 {os.path.join(result_dir, 'error_combined_plot.png')}")
#     plt.close()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def evaluate_results(df: pd.DataFrame, result_dir: str):
    """
    评估函数（与当前 records 字段对齐）：
    - 支持 MAE / 相对误差 / 像素相对误差
    - 支持 |Δx|/range_x, |Δy|/range_y
    - 支持联合归一化误差 xy_err_over_range
    """

    os.makedirs(result_dir, exist_ok=True)

    # -- 1. 过滤预测失败样本
    df_clean = df[(df["pred_x"] != -1) & (df["pred_y"] != -1)].copy()

    # ===== 2. 确保必要列存在，如缺则兜底计算 =====
    # 2.1 绝对误差
    if "x_abs_err" not in df_clean.columns:
        df_clean["x_abs_err"] = (df_clean["pred_x"] - df_clean["gt_x"]).abs()
    if "y_abs_err" not in df_clean.columns:
        df_clean["y_abs_err"] = (df_clean["pred_y"] - df_clean["gt_y"]).abs()

    # 2.2 轴范围（如果记录里没有 range，但你现在是有的，一般不会走这里）
    if "x_range" not in df_clean.columns or "y_range" not in df_clean.columns:
        # 没有的话就先设为 NaN，避免 KeyError
        if "x_range" not in df_clean.columns:
            df_clean["x_range"] = np.nan
        if "y_range" not in df_clean.columns:
            df_clean["y_range"] = np.nan

    # 2.3 归一化误差
    if "x_err_over_range" not in df_clean.columns:
        df_clean["x_err_over_range"] = df_clean["x_abs_err"] / df_clean["x_range"]
    if "y_err_over_range" not in df_clean.columns:
        df_clean["y_err_over_range"] = df_clean["y_abs_err"] / df_clean["y_range"]

    # 2.4 联合 X+Y 归一化误差
    if "xy_err_over_range" not in df_clean.columns:
        df_clean["xy_err_over_range"] = (
            df_clean["x_err_over_range"] + df_clean["y_err_over_range"]
        ) / 2.0

    # ===== 3. 汇总每种 prompt_type + image_type 的表现 =====
    summary = df_clean.groupby(["prompt_type", "image_type"]).agg(
        # 原始 MAE
        avg_mae=("mae", "mean"),
        std_mae=("mae", "std"),

        # 像素相对误差
        avg_px_rel_x=("pixel_rel_x", "mean"),
        std_px_rel_x=("pixel_rel_x", "std"),
        avg_px_rel_y=("pixel_rel_y", "mean"),
        std_px_rel_y=("pixel_rel_y", "std"),

        # 数值空间绝对误差（按轴）
        avg_x_mae=("x_abs_err", "mean"),
        avg_y_mae=("y_abs_err", "mean"),

        # 数值空间相对误差
        avg_x_re=("x_re", "mean"),
        std_x_re=("x_re", "std"),
        avg_y_re=("y_re", "mean"),
        std_y_re=("y_re", "std"),

        # 归一化误差（误差 / tick-range）
        avg_x_err_over_range=("x_err_over_range", "mean"),
        avg_y_err_over_range=("y_err_over_range", "mean"),

        # ⭐ 联合 X+Y 归一化误差
        avg_xy_err_over_range=("xy_err_over_range", "mean"),
    ).reset_index()

    # -- 保存误差统计表
    summary.to_csv(os.path.join(result_dir, "mae_summary.csv"), index=False)
    print("📄 已保存评估汇总表 mae_summary.csv")

    # -- 保存各配置差异对比表（仅用 avg_mae）
    pivot = summary.pivot_table(
        index=["prompt_type", "image_type"],
        values=["avg_mae"]
    ).reset_index()
    pivot.to_csv(os.path.join(result_dir, "prompt_comparison.csv"), index=False)
    print("📄 已保存提示方式对比表 prompt_comparison.csv")

    # ===== 4. 绘图部分 =====
    labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
    x = range(len(labels))

    # ---------- 图 1：MAE + 数值相对误差 ----------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bar_width = 0.35

    bars_mae = ax1.bar(x, summary["avg_mae"], width=bar_width, label="MAE", color="#D98880")
    ax1.set_ylabel("MAE (absolute error)", fontsize=12, color="#D98880")
    ax1.tick_params(axis='y', labelcolor="#D98880")
    ax1.set_ylim(0, summary["avg_mae"].max() * 1.2)

    for bar in bars_mae:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, color="#B03A2E")

    ax2 = ax1.twinx()
    offset = bar_width
    bars_xre = ax2.bar([i + offset for i in x], summary["avg_x_re"], width=bar_width / 2,
                       label="X Relative Error", color="#82E0AA")
    bars_yre = ax2.bar([i + offset + bar_width / 2 for i in x], summary["avg_y_re"], width=bar_width / 2,
                       label="Y Relative Error", color="#5DADE2")

    ax2.set_ylabel("Relative Error", fontsize=12, color="#5DADE2")
    ax2.tick_params(axis='y', labelcolor="#5DADE2")
    ax2.set_ylim(0, max(summary["avg_x_re"].max(), summary["avg_y_re"].max()) * 1.5)

    for bars in [bars_xre, bars_yre]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9, color="#2874A6")

    plt.xticks([i + bar_width / 2 for i in x], labels, rotation=60, fontsize=11)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=11)

    plt.title("Average MAE and Relative Errors by Prompt+Image Setting", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "error_comparison_plot_optimized_with_labels.png"))
    print("📊 已保存高级版误差对比图 error_comparison_plot_optimized_with_labels.png")
    plt.close()

    # ---------- 图 2：X/Y 分开 MAE 图 ----------
    plt.figure(figsize=(10, 6))
    plt.bar([i - bar_width / 2 for i in x], summary["avg_x_mae"], width=bar_width,
            label="X MAE", color="#85C1E9")
    plt.bar([i + bar_width / 2 for i in x], summary["avg_y_mae"], width=bar_width,
            label="Y MAE", color="#F5B7B1")

    for i, (xmae, ymae) in enumerate(zip(summary["avg_x_mae"], summary["avg_y_mae"])):
        plt.text(i - bar_width / 2, xmae + 0.05, f"{xmae:.2f}", ha='center',
                 va='bottom', fontsize=10, color="#21618C")
        plt.text(i + bar_width / 2, ymae + 0.05, f"{ymae:.2f}", ha='center',
                 va='bottom', fontsize=10, color="#922B21")

    plt.xticks(x, labels, rotation=60)
    plt.ylabel("MAE (per axis)")
    plt.title("Separate MAE for X and Y by Prompt+Image Setting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "y_mae_comparison_plot.png"))
    print(f"📊 已保存MAE对比图 {os.path.join(result_dir, 'y_mae_comparison_plot.png')}")
    plt.close()

    # ---------- 图 3：Pixel Relative Error (X / Y) ----------
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    plt.bar([i - bar_width / 2 for i in x], summary["avg_px_rel_x"], width=bar_width,
            label="Pixel X Rel Error", color="#AED6F1")
    plt.bar([i + bar_width / 2 for i in x], summary["avg_px_rel_y"], width=bar_width,
            label="Pixel Y Rel Error", color="#F9E79F")

    for i, (px, py) in enumerate(zip(summary["avg_px_rel_x"], summary["avg_px_rel_y"])):
        plt.text(i - bar_width / 2, px + 0.001, f"{px:.4f}", ha='center',
                 fontsize=9, color="#21618C")
        plt.text(i + bar_width / 2, py + 0.001, f"{py:.4f}", ha='center',
                 fontsize=9, color="#B7950B")

    plt.xticks(x, labels, rotation=60)
    plt.ylabel("Relative Pixel Error (by width/height)")
    plt.title("Pixel-level Relative Error (X and Y) by Prompt+Image Setting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "pixel_relative_error_xy_plot.png"))
    print(f"📊 已保存像素相对误差图 {os.path.join(result_dir, 'pixel_relative_error_xy_plot.png')}")
    plt.close()

    # ---------- 图 4：综合图：MAE + 相对误差 + 像素误差 ----------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    bar_width = 0.15

    bars_mae = ax1.bar([i - 1.5 * bar_width for i in x], summary["avg_mae"], width=bar_width,
                       label="MAE", color="#D98880")
    ax1.set_ylabel("MAE (absolute error)", fontsize=12, color="#D98880")
    ax1.tick_params(axis='y', labelcolor="#D98880")
    ax1.set_ylim(0, summary["avg_mae"].max() * 1.3)

    for i, bar in enumerate(bars_mae):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar_width / 2, height + 0.1, f"{height:.1f}",
                 ha='center', fontsize=9, color="#B03A2E")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Relative Errors (normalized)", fontsize=12)
    ax2.tick_params(axis='y')

    bars_xre = ax2.bar([i - 0.5 * bar_width for i in x], summary["avg_x_re"], width=bar_width,
                       label="X Rel Error", color="#82E0AA")
    bars_yre = ax2.bar([i + 0.5 * bar_width for i in x], summary["avg_y_re"], width=bar_width,
                       label="Y Rel Error", color="#5DADE2")
    bars_px_x = ax2.bar([i + 1.5 * bar_width for i in x], summary["avg_px_rel_x"], width=bar_width,
                        label="Pixel X Rel Err", color="#AED6F1")
    bars_px_y = ax2.bar([i + 2.5 * bar_width for i in x], summary["avg_px_rel_y"], width=bar_width,
                        label="Pixel Y Rel Err", color="#F9E79F")

    for bars, color in zip(
        [bars_xre, bars_yre, bars_px_x, bars_px_y],
        ["#1E8449", "#21618C", "#2980B9", "#B7950B"]
    ):
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar_width / 2, height + 0.001, f"{height:.3f}",
                     ha='center', fontsize=8, color=color)

    plt.xticks(x, labels, rotation=60, fontsize=11)
    ax1.set_title("Combined MAE + Relative Errors + Pixel-Level Errors", fontsize=14, fontweight="bold")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()
    combined_path = os.path.join(result_dir, "error_comparison_plot.png")
    plt.savefig(combined_path)
    print(f"📊 已保存综合误差图 {combined_path}")
    plt.close()

    # ---------- 图 5：⭐ 联合 X+Y “误差 / 轴范围” ----------
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, summary["avg_xy_err_over_range"], width=0.55,
                   label="( |ΔX|/rangeX + |ΔY|/rangeY ) / 2", color="#BB8FCE")

    for i, v in enumerate(summary["avg_xy_err_over_range"]):
        plt.text(i, v + 0.001, f"{v:.3f}", ha='center', va='bottom',
                 fontsize=9, color="#6C3483")

    plt.xticks(x, labels, rotation=60, fontsize=11)
    plt.ylabel("Normalized Error over Axis Range (Combined X+Y)")
    plt.title("Normalized Error over Axis Range (Combined X and Y)", fontsize=13)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(result_dir, "error_over_range_xy_combined_plot.png")
    plt.savefig(out_path)
    print(f"📊 已保存联合归一化误差图 {out_path}")
    plt.close()


# import os
# import json
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def evaluate_results(df: pd.DataFrame, result_dir: str, config_dir: str = "chart_configs"):
#     """
#     完全匹配你的 JSON 格式的评估函数：
#     - JSON 必含 x_ticks / y_ticks
#     - 对每条记录根据 chart_id 读取 ticks，并构造 min/max
#     - 计算误差 / 轴范围（单独 + 联合）
#     - 保留你原来的所有图
#     - 新增联合归一化误差图
#     """
#
#     # -------------------------
#     # 1) 清洗预测失败的数据
#     # -------------------------
#     df_clean = df[(df["pred_x"] != -1) & (df["pred_y"] != -1)].copy()
#
#     # -------------------------
#     # 2) 从 JSON 读取 x_ticks / y_ticks
#     # -------------------------
#     axis_rows = []
#     for cid in df_clean["chart_id"].unique():
#         json_path = Path(config_dir) / f"{cid}.json"
#         if not json_path.exists():
#             print(f"⚠️ JSON 未找到：{json_path}")
#             continue
#
#         with open(json_path, "r", encoding="utf-8") as f:
#             cfg = json.load(f)
#
#         x_ticks = cfg["x_ticks"]
#         y_ticks = cfg["y_ticks"]
#
#         axis_rows.append({
#             "chart_id": cid,
#             "x_min_tick": min(x_ticks),
#             "x_max_tick": max(x_ticks),
#             "y_min_tick": min(y_ticks),
#             "y_max_tick": max(y_ticks),
#         })
#
#     # 将 JSON 的轴范围 merge 进 df
#     axis_df = pd.DataFrame(axis_rows)
#     df_clean = df_clean.merge(axis_df, on="chart_id", how="left")
#
#     # -------------------------
#     # 3) 计算归一化误差 = |Δx|/range_x
#     # -------------------------
#     df_clean["x_abs_err"] = (df_clean["pred_x"] - df_clean["gt_x"]).abs()
#     df_clean["y_abs_err"] = (df_clean["pred_y"] - df_clean["gt_y"]).abs()
#
#     df_clean["x_range"] = df_clean["x_max_tick"] - df_clean["x_min_tick"]
#     df_clean["y_range"] = df_clean["y_max_tick"] - df_clean["y_min_tick"]
#
#     df_clean["x_err_over_range"] = df_clean["x_abs_err"] / df_clean["x_range"]
#     df_clean["y_err_over_range"] = df_clean["y_abs_err"] / df_clean["y_range"]
#
#     # -------------------------
#     # ⭐ 4) 联合归一化误差
#     # -------------------------
#     df_clean["xy_err_over_range"] = (
#         df_clean["x_err_over_range"] + df_clean["y_err_over_range"]
#     ) / 2.0
#
#     # -------------------------
#     # 5) 汇总到 summary
#     # -------------------------
#     summary = df_clean.groupby(["prompt_type", "image_type"]).agg(
#         avg_mae=("mae", "mean"),
#         avg_x_re=("x_re", "mean"),
#         avg_y_re=("y_re", "mean"),
#         avg_px_rel_x=("pixel_rel_x", "mean"),
#         avg_px_rel_y=("pixel_rel_y", "mean"),
#
#         avg_x_mae=("x_abs_err", "mean"),
#         avg_y_mae=("y_abs_err", "mean"),
#
#         avg_x_err_over_range=("x_err_over_range", "mean"),
#         avg_y_err_over_range=("y_err_over_range", "mean"),
#
#         # ⭐ 联合指标
#         avg_xy_err_over_range=("xy_err_over_range", "mean"),
#     ).reset_index()
#
#     os.makedirs(result_dir, exist_ok=True)
#     summary.to_csv(os.path.join(result_dir, "mae_summary.csv"), index=False)
#
#     # -------------------------
#     # 6) 新图：联合归一化误差
#     # -------------------------
#     labels = summary.apply(lambda row: f"{row['prompt_type']}+{row['image_type']}", axis=1)
#     x = range(len(labels))
#
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(x, summary["avg_xy_err_over_range"], width=0.55,
#                    color="#BB8FCE", label="Normalized XY Error")
#
#     for i, v in enumerate(summary["avg_xy_err_over_range"]):
#         plt.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=10, color="#6C3483")
#
#     plt.xticks(x, labels, rotation=60)
#     plt.ylabel("Normalized XY Error ((|ΔX|/rangeX + |ΔY|/rangeY)/2)")
#     plt.title("Axis-Range Normalized Combined Error", fontsize=14)
#     plt.tight_layout()
#
#     out_path = os.path.join(result_dir, "xy_normalized_error_plot.png")
#     plt.savefig(out_path)
#     plt.close()
#
#     print(f"📊 已保存联合归一化误差：{out_path}")


async def estimate_diameter_via_llm(image_path: str, point_name: str) -> float:
    prompt = f"""
You are analyzing a chart that contains visual markers indicating the location of a point on a chart.
Your task is to estimate the **diameter (in pixels)** of the [{point_name}] in the image.
Return only one number between {MIN_MARK_DIAMETER:g} and {MAX_MARK_DIAMETER:g}. Do not repeat digits. Example:
65
"""
    print(f"📤 [Prompt for Diameter Estimation - {point_name}]:\n{prompt.strip()}\n")
    response_text = await call_llm_response(prompt, image_path, point_name, task="diameter_estimation")

    print(f"📄 提取的文本内容：\n{response_text}")

    candidates = []
    for match in re.findall(r"\d+(?:\.\d+)?", str(response_text)):
        try:
            value = float(match)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(value) and MIN_MARK_DIAMETER <= value <= MAX_MARK_DIAMETER:
            candidates.append(value)

    if candidates:
        diameter = candidates[0]
        print(f"📏 圆直径估计为 {diameter:.2f}px")
        return diameter

    print(f"⚠️ 无有效圆直径估计，使用默认值 {DEFAULT_MARK_DIAMETER:g}px")
    return DEFAULT_MARK_DIAMETER


import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


def format_tick_value(val):
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.2f}"


def draw_tick_label(draw, position, text, axis, image_size, font):
    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
    if axis == "x":
        x = position
        y = image_size[1] - text_h - 4
        draw.text((x - text_w / 2, y), text, font=font, fill="red")
    else:
        x = 5
        y = position - text_h / 2
        draw.text((x, y), text, font=font, fill="red")


def draw_grid_lines_only(draw, canvas_size, center, grid_span, axis):
    W, H = canvas_size
    if not np.isfinite(grid_span) or grid_span <= 0:
        print(f"⚠️ 跳过网格绘制：grid_span 非法 ({grid_span})")
        return

    i = 0
    while True:
        pos = center + i * grid_span
        if pos >= (W if axis == "x" else H): break
        if axis == "x":
            draw_dashed_line(draw, (pos, 0), (pos, H), dash_length=5, gap_length=5, fill="gray", width=1)
        else:
            draw_dashed_line(draw, (0, pos), (W, pos), dash_length=5, gap_length=5, fill="gray", width=1)
        i += 1

    i = 1
    while True:
        pos = center - i * grid_span
        if pos <= 0: break
        if axis == "x":
            draw_dashed_line(draw, (pos, 0), (pos, H), dash_length=5, gap_length=5, fill="gray", width=1)
        else:
            draw_dashed_line(draw, (0, pos), (W, pos), dash_length=5, gap_length=5, fill="gray", width=1)
        i += 1


def find_closest_tick(coord, ticks, pixels):
    idx = min(range(len(ticks)), key=lambda i: abs(ticks[i] - coord))
    return ticks[idx], pixels[idx]


from typing import Tuple, List, Callable


async def generate_expanded_crop_with_grid_by_diameter(
        image_path: str,
        pred_coord: Tuple[float, float],
        x_ticks: List[float],
        y_ticks: List[float],
        x_pixels: List[float],
        y_pixels: List[float],
        chart_id: str,
        point_name: str,
        feedback_round: int = 0,
        base_crop_size: int = 120,
        resize_to: Tuple[int, int] = (224, 224)
) -> Tuple[
    str, List[float], List[float], List[float], List[float],
    Callable[[float, float], Tuple[float, float]], int, int
]:
    import os
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from math import hypot

    def draw_dashed_line(draw, start, end, fill="gray", width=1, dash_length=1, gap_length=4):
        x1, y1 = start
        x2, y2 = end
        total_length = hypot(x2 - x1, y2 - y1)
        dash_gap = dash_length + gap_length
        num_dashes = int(total_length // dash_gap)
        for i in range(num_dashes + 1):
            start_frac = (i * dash_gap) / total_length
            end_frac = min((i * dash_gap + dash_length) / total_length, 1)
            sx = x1 + (x2 - x1) * start_frac
            sy = y1 + (y2 - y1) * start_frac
            ex = x1 + (x2 - x1) * end_frac
            ey = y1 + (y2 - y1) * end_frac
            draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)

    def draw_tick_label(draw, pos, text, axis, image_size, font):
        if axis == "x":
            # --- 横轴 → 45° 倾斜 ---
            # 先生成大透明图层
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            txt_img = Image.new("RGBA", (text_w * 4, text_h * 4), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((text_w * 2, text_h * 2), text, fill="red", font=font, anchor="mm")

            rotated = txt_img.rotate(45, expand=1, resample=Image.BICUBIC)

            # 🔑 计算旋转后实际文字边界
            rotated_bbox = rotated.getbbox()  # (x0,y0,x1,y1)
            cropped_rotated = rotated.crop(rotated_bbox)
            rw, rh = cropped_rotated.size

            # 横向中心对齐，纵向真正贴底（文字下边缘对齐 image_size[1]）
            paste_x = int(pos - rw / 2)
            paste_y = image_size[1] - rh  # 文字底边对齐画布底端

            resized.paste(cropped_rotated, (paste_x, paste_y), cropped_rotated)

        else:
            # --- 纵轴 → 保持水平 ---
            bbox = draw.textbbox((0, 0), text, font=font)
            text_h = bbox[3] - bbox[1]
            draw.text((4, pos - text_h / 2), text, fill="red", font=font)

    def format_tick_value(value):
        rounded = round(value, 2)
        return f"{int(rounded)}" if abs(rounded - int(rounded)) < 1e-6 else f"{rounded:.2f}"

    def find_closest_tick(coord, ticks, pixels):
        idx = min(range(len(ticks)), key=lambda i: abs(ticks[i] - coord))
        return ticks[idx], pixels[idx]

    x_mapper = lambda v: np.interp(v, x_ticks, x_pixels)
    y_mapper = lambda v: np.interp(v, y_ticks, y_pixels)
    pixel_x = x_mapper(pred_coord[0])
    pixel_y = y_mapper(pred_coord[1])

    diameter = await estimate_diameter_via_llm(image_path, point_name)
    if not np.isfinite(diameter):
        print(f"⚠️ Estimated diameter is not finite ({diameter}), using default {DEFAULT_MARK_DIAMETER:g}px.")
        diameter = DEFAULT_MARK_DIAMETER
    if diameter <= 5:
        print(f"⚠️ Estimated diameter {diameter:.1f}px is too small, forcing to 10px.")
        diameter = 10.0
    rounded_diameter = round(diameter / 10) * 10
    print(f"📏 Diameter: {diameter:.1f}px → Rounded: {rounded_diameter}px")

    # ✅ 自动调节 grid span 与 canvas size
    if rounded_diameter == 10:
        grid_span = diameter * 2
        canvas_size = int(grid_span * 6)
    elif rounded_diameter >= 20:
        grid_span = diameter * 1
        canvas_size = int(grid_span * 6)
    # else:
    #     grid_span = diameter * 1.5
    #     canvas_size = int(grid_span * 6)

    print(f"📐 使用 grid_span={grid_span:.1f}px，canvas_size={canvas_size}px")

    left = int(max(pixel_x - base_crop_size // 2, 0))
    upper = int(max(pixel_y - base_crop_size // 2, 0))
    right = left + base_crop_size
    lower = upper + base_crop_size

    base_img = Image.open(image_path).convert("RGB")
    cropped = base_img.crop((left, upper, right, lower))

    paste_x = (canvas_size - base_crop_size) // 2
    paste_y = (canvas_size - base_crop_size) // 2
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")
    canvas.paste(cropped, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)
    try:
        font_final = ImageFont.truetype("arial.ttf", 14)
    except:
        font_final = ImageFont.load_default()

    tick_x, px_x = find_closest_tick(pred_coord[0], x_ticks, x_pixels)
    tick_y, px_y = find_closest_tick(pred_coord[1], y_ticks, y_pixels)
    tick_value_span_x = x_ticks[1] - x_ticks[0] if len(x_ticks) > 1 else 1
    tick_value_span_y = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1
    tick_pixel_span_x = x_pixels[1] - x_pixels[0] if len(x_pixels) > 1 else 20
    tick_pixel_span_y = y_pixels[1] - y_pixels[0] if len(y_pixels) > 1 else 20

    grid_center_x = paste_x + (px_x - left)
    grid_center_y = paste_y + (px_y - upper)

    resized = canvas.resize(resize_to, Image.LANCZOS)
    draw_final = ImageDraw.Draw(resized)
    scale_factor = resize_to[0] / canvas_size

    new_xticks, new_xpixels = [], []
    new_yticks, new_ypixels = [], []

    for axis, center_px, tick_start, tick_value_span, tick_pixel_span, tick_list, pixel_list in [
        ("x", grid_center_x, tick_x, tick_value_span_x, tick_pixel_span_x, new_xticks, new_xpixels),
        ("y", grid_center_y, tick_y, tick_value_span_y, tick_pixel_span_y, new_yticks, new_ypixels)
    ]:
        i = 0
        while True:
            pos = center_px + i * grid_span
            if pos >= canvas_size:
                break
            tick_val = tick_start + i * (grid_span / tick_pixel_span) * tick_value_span
            tick_list.append(round(tick_val, 2))
            pixel_list.append(pos * scale_factor)
            draw_dashed_line(draw_final, (pos * scale_factor, 0) if axis == "x" else (0, pos * scale_factor),
                             (pos * scale_factor, resize_to[1]) if axis == "x" else (resize_to[0], pos * scale_factor),
                             fill="gray", width=1)
            draw_tick_label(draw_final, pos * scale_factor, format_tick_value(tick_val), axis, resized.size, font_final)
            i += 1

        i = 1
        while True:
            pos = center_px - i * grid_span
            if pos <= 0:
                break
            tick_val = tick_start - i * (grid_span / tick_pixel_span) * tick_value_span
            tick_list.insert(0, round(tick_val, 2))
            pixel_list.insert(0, pos * scale_factor)
            draw_dashed_line(draw_final, (pos * scale_factor, 0) if axis == "x" else (0, pos * scale_factor),
                             (pos * scale_factor, resize_to[1]) if axis == "x" else (resize_to[0], pos * scale_factor),
                             fill="gray", width=1)
            draw_tick_label(draw_final, pos * scale_factor, format_tick_value(tick_val), axis, resized.size, font_final)
            i += 1

    def pixel_to_cropped_coords(x: float, y: float) -> Tuple[float, float]:
        return (x - left) * scale_factor, (y - upper) * scale_factor

    # out_path = os.path.join("raw_crops", chart_id, f"{point_name}_round{feedback_round}_adaptive.png")
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # resized.save(out_path)
    # ✅ 将输出保存到 results/<chart_id>/raw_crops/
    out_dir = os.path.join("results_scatter_Gemini", chart_id, "raw_crops")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{point_name}_round{feedback_round}_adaptive.png")
    resized.save(out_path)

    new_yticks.reverse()
    print(f"✅ Resize后标注tick完成，图保存于: {out_path}")
    print(f"🎯 返回的 new_xticks 用于 prompt：{[format_tick_value(t) for t in new_xticks]}")

    return out_path, new_xticks, new_yticks, new_xpixels, new_ypixels, pixel_to_cropped_coords, left, upper


async def generate_adaptive_crop(
        image_path: str,
        pred_coord: tuple,
        x_ticks: list,
        y_ticks: list,
        x_pixels: list,
        y_pixels: list,
        chart_id: str,
        point_name: str,
        feedback_round: int = 0,
        tolerance: float = 5.0  # ✅ 可调节的容差
) -> tuple:
    """
    自适应裁图：根据点直径与tick跨度设定grid_span策略，并容错判断是否加密。
    若加密，则自动调整tick字体大小为8，以避免文字重叠。
    """

    # Step 1: 圆直径估计
    circle_diameter = await estimate_diameter_via_llm(image_path, point_name)
    if not np.isfinite(circle_diameter):
        print(f"⚠️ 圆直径估计不是有限值 ({circle_diameter})，使用默认值 {DEFAULT_MARK_DIAMETER:g}px")
        circle_diameter = DEFAULT_MARK_DIAMETER
    print(f"📏 圆直径估计为 {circle_diameter:.2f}px")

    # Step 2: 平均tick跨度
    def avg_span(pixels: list) -> float:
        if len(pixels) < 2:
            return 1.0
        spans = [abs(pixels[i + 1] - pixels[i]) for i in range(len(pixels) - 1)]
        return sum(spans) / len(spans)

    x_tick_span = avg_span(x_pixels)
    y_tick_span = avg_span(y_pixels)

    # Step 3: 三段策略选择
    if circle_diameter <= 12:
        grid_span = 22.5
        crop_size = 120
        strategy = "Finding1"
    elif circle_diameter <= 50:
        grid_span = circle_diameter
        crop_size = 150
        strategy = "Finding2"
    else:
        fallback_span = circle_diameter
        found_aligned = False
        for k in [2, 3]:
            candidate_span = circle_diameter / k
            if abs(candidate_span - round(candidate_span)) < 2:
                grid_span = candidate_span
                found_aligned = True
                break
        if not found_aligned:
            grid_span = fallback_span
            strategy = "Finding2 (fallback from 3)"
        else:
            strategy = "Finding3"
        crop_size = 150

    print(f"🔍 使用策略：{strategy} → grid_span ≈ {grid_span:.1f}px")

    # Step 4: 插值密度函数（带容差判断）
    def compute_density_with_tolerance(tick_span: float, target_span: float, tol: float = 5.0, max_n: int = 6) -> int:
        if abs(tick_span - target_span) <= tol:
            return 0
        best_n = 1
        best_diff = abs((tick_span / 2) - target_span)
        for n in range(1, max_n + 1):
            interp_span = tick_span / (n + 1)
            diff = abs(interp_span - target_span)
            if diff < best_diff:
                best_diff = diff
                best_n = n
        return best_n

    # Step 5: 计算加密密度
    x_grid_density = compute_density_with_tolerance(x_tick_span, grid_span, tol=tolerance)
    y_grid_density = compute_density_with_tolerance(y_tick_span, grid_span, tol=tolerance)

    # Step 6: 自动设置字体大小（有加密时使用较小字体）
    font_size = 8 if (x_grid_density > 0 or y_grid_density > 0) else 10

    # Step 7: 输出日志
    print(f"🧮 tick跨度：x={x_tick_span:.1f}px, y={y_tick_span:.1f}px")
    print(f"📐 grid_density: x={x_grid_density}, y={y_grid_density}（tolerance={tolerance}px）")
    print(f"🔠 字体大小：{font_size}px")
    print(f"🖼️ 裁剪窗口尺寸：{crop_size}px")

    # Step 8: 执行裁剪
    return crop_draw_ticks_resize(
        image_path=image_path,
        pred_coord=pred_coord,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_pixels=x_pixels,
        y_pixels=y_pixels,
        chart_id=chart_id,
        point_name=point_name,
        feedback_round=feedback_round,
        window_size=crop_size,
        font_size=font_size,  # ✅ 新增字体大小控制
        x_grid_density=x_grid_density,
        y_grid_density=y_grid_density,
        return_ticks=True  # ✅ 添加此项
    )


def build_point_prompt(point_name: str) -> str:
    return (
        f"You are given a cropped scatter chart image around the target point {point_name}. "
        f"Please check if the point corresponding to {point_name} is visible in this cropped region. "
        "Only respond with a JSON object like: {\"exists\": true} or {\"exists\": false}."
    )


async def run_experiment(batch_size=None, chart_ids=None, datasets=None):
    datasets = list(datasets) if datasets is not None else DATASET_CONFIGS.copy()
    if chart_ids:
        datasets = [ds for ds in datasets if ds['chart_id'] in chart_ids]
    feedback_final_results = {}

    async def run_for_dataset(dataset):
        records = []
        for point_name, gt in dataset["data_points"].items():

            for prompt_type, image_type in EXPERIMENT_TYPES:
                # FEEDBACK_START_ROUND = 1 if prompt_type == "feedback_crop_from_feedback" else 2
                FEEDBACK_START_ROUND = 1 if prompt_type in ["feedback_crop_from_feedback",
                                                            'feedback_crop_adaptive'] else 2

                valid_runs = 0
                total_attempts = 0
                image_path = resolve_local_path(dataset["image_paths"][image_type])

                history_preds = []
                last_pred = None

                local_xticks = dataset["x_ticks"]
                local_yticks = dataset["y_ticks"]
                local_xpix = dataset["x_pixels"]
                local_ypix = dataset["y_pixels"]
                pred_img_path = image_path

                fixed_crop_info = None  # 先声明

                if prompt_type in ["feedback_crop_from_feedback", "feedback_crop_adaptive"]:
                    # if prompt_type == "feedback_crop_from_feedback":

                    feedback_key = (dataset["chart_id"], point_name)
                    feedback_final_pred = feedback_final_results.get(feedback_key)

                    if feedback_final_pred is None:
                        print(f"⚠️ 缺少 feedback 最终结果，跳过该点：{point_name}")
                        continue

                    last_pred = feedback_final_pred  # 初始化第一轮用的参考点

                while valid_runs < REPEAT_TIMES:
                    if total_attempts >= MAX_ATTEMPTS:
                        print(f"⚠️ 达到最大尝试次数 {MAX_ATTEMPTS}，停止该配置：{point_name} - {prompt_type} - {image_type}")
                        break

                    pred_img_path = image_path
                    local_xticks = dataset["x_ticks"]
                    local_yticks = dataset["y_ticks"]
                    local_xpix = dataset["x_pixels"]
                    local_ypix = dataset["y_pixels"]

                    if prompt_type in ["feedback", "feedback_cropped", "feedback_crop_final",
                                       "feedback_crop_from_feedback",
                                       "feedback_crop_adaptive"] and last_pred is not None and valid_runs + 1 >= FEEDBACK_START_ROUND:

                        if prompt_type == "feedback_cropped":
                            pred_img_path, local_xticks, local_yticks, local_xpix, local_ypix, _, left, upper = crop_draw_ticks_resize(
                                # pred_img_path, local_xticks, local_yticks, local_xpix, local_ypix, _, _, _ = crop_draw_ticks_resize(
                                image_path=image_path,
                                pred_coord=last_pred,
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                chart_id=dataset["chart_id"],
                                point_name=point_name,
                                feedback_round=valid_runs + 1
                            )

                        elif prompt_type == "feedback_crop_final":
                            if valid_runs + 1 == FEEDBACK_START_ROUND:
                                cropped_img_path, local_xticks, local_yticks, local_xpix, local_ypix, _, left, upper = crop_draw_ticks_resize(
                                    image_path=image_path,
                                    pred_coord=last_pred,
                                    x_ticks=dataset["x_ticks"],
                                    y_ticks=dataset["y_ticks"],
                                    x_pixels=dataset["x_pixels"],
                                    y_pixels=dataset["y_pixels"],
                                    chart_id=dataset["chart_id"],
                                    point_name=point_name,
                                    feedback_round=valid_runs + 1
                                )
                                fixed_crop_info = (
                                cropped_img_path, local_xticks, local_yticks, local_xpix, local_ypix, left, upper)

                            cropped_img_path, local_xticks, local_yticks, local_xpix, local_ypix, left, upper = fixed_crop_info

                            cropped_x, cropped_y = convert_data_coord_to_resized_crop_pixel(
                                data_coord=last_pred,
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                crop_origin=(left, upper),
                                crop_size=(150, 150),
                                resize_size=(224, 224)
                            )

                            import re, os
                            safe_chart_id = re.sub(r'[\\/*?:"<>| ]', "_", dataset["chart_id"])
                            tempy_dir = os.path.join("results_scatter_Gemini", safe_chart_id, "tempy")
                            os.makedirs(tempy_dir, exist_ok=True)

                            overlay_img_path = os.path.join(
                                tempy_dir,
                                f"overlay_{safe_chart_id}_{point_name}_{prompt_type}_{image_type}_run{valid_runs + 1}.png"
                            )

                            # overlay_img_path = f"temp/{dataset['chart_id']}/overlay_{dataset['chart_id']}_{point_name}_{prompt_type}_{image_type}_run{valid_runs + 1}.png"
                            # os.makedirs(os.path.join("temp", dataset['chart_id']), exist_ok=True)

                            draw_crosshair_on_resized_image(
                                img_path=cropped_img_path,
                                coords=[(cropped_x, cropped_y)],
                                output_path=overlay_img_path
                            )

                            pred_img_path = overlay_img_path

                        elif prompt_type == "feedback_crop_adaptive":
                            # ✅ 切换使用 no_grid 图
                            chart_id = dataset["chart_id"]
                            no_grid_path = resolve_local_path(
                                dataset["image_paths"].get("no_grid") or os.path.join("charts", "scatter", f"{chart_id}.png")
                            )

                            if valid_runs + 1 == FEEDBACK_START_ROUND:
                                crop_coord = feedback_final_results.get((chart_id, point_name))
                                if crop_coord is None:
                                    print(f"⚠️ 缺少 feedback 最终预测点，跳过 {point_name}")
                                    break
                                last_pred = crop_coord
                            else:
                                crop_coord = last_pred

                            res = await try_generate_crop_until_point_detected(
                                image_path=no_grid_path,
                                pred_coord=crop_coord,
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                chart_id=chart_id,
                                point_name=point_name,
                                feedback_round=valid_runs + 1,
                                judge_prompt=build_point_prompt(point_name),  # ✅ 用来调用 LLM 判断
                                init_crop_size=120,
                                max_attempts=MAX_ATTEMPTS
                            )

                            if res is None:
                                print(f"⚠️ adaptive 裁剪未检测到点，跳过 {point_name}")
                                total_attempts += 1
                                if total_attempts >= MAX_ATTEMPTS:
                                    print(f"⚠️ adaptive 裁剪连续失败达到上限，停止该配置：{point_name} - {prompt_type} - {image_type}")
                                    break
                                await asyncio.sleep(2)
                                continue

                            pred_img_path, local_xticks, local_yticks, local_xpix, local_ypix, pixel_mapper, left, upper = res


                        # successful crop from feedback
                        elif prompt_type == "feedback_crop_from_feedback":
                            # ✅ 第1轮裁剪使用 feedback 最终预测点，之后用上一轮预测裁剪
                            if valid_runs + 1 == FEEDBACK_START_ROUND:
                                crop_coord = feedback_final_results.get((dataset["chart_id"], point_name))
                                if crop_coord is None:
                                    print(f"⚠️ 缺少 feedback 最终预测点，跳过 {point_name}")
                                    break
                                last_pred = crop_coord  # ✅ 初始化第一轮参考点
                            else:
                                crop_coord = last_pred

                            # ✅ 每轮重新裁剪
                            pred_img_path, local_xticks, local_yticks, local_xpix, local_ypix, _, left, upper = crop_draw_ticks_resize(
                                image_path=image_path,
                                pred_coord=crop_coord,
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                chart_id=dataset["chart_id"],
                                point_name=point_name,
                                feedback_round=valid_runs + 1
                            )

                        elif prompt_type == "feedback":
                            import re, os
                            safe_chart_id = re.sub(r'[\\/*?:"<>| ]', "_", dataset["chart_id"])
                            tempy_dir = os.path.join("results_scatter_Gemini", safe_chart_id, "tempy")
                            os.makedirs(tempy_dir, exist_ok=True)

                            overlay_img_path = os.path.join(
                                tempy_dir,
                                f"overlay_{safe_chart_id}_{point_name}_{prompt_type}_{image_type}_run{valid_runs + 1}.png"
                            )

                            # overlay_img_path = f"temp/{dataset['chart_id']}/overlay_{dataset['chart_id']}_{point_name}_{prompt_type}_{image_type}_run{valid_runs + 1}.png"
                            # os.makedirs(os.path.join("temp", dataset['chart_id']), exist_ok=True)
                            generate_overlayed_image_multi_with_mapping(
                                original_img_path=image_path,
                                pred_coords=[last_pred],
                                x_ticks=dataset["x_ticks"],
                                y_ticks=dataset["y_ticks"],
                                x_pixels=dataset["x_pixels"],
                                y_pixels=dataset["y_pixels"],
                                output_path=overlay_img_path,
                                feedback_round=valid_runs + 1,
                                draw_all=False
                            )
                            pred_img_path = overlay_img_path

                    print(f"✅ 当前用于 prompt 的 tick 值为：X={local_xticks}, Y={local_yticks}")

                    if prompt_type == "feedback_crop_final":
                        prompt_type_for_prompt = "feedback"
                    elif prompt_type in ["feedback_crop_from_feedback", "feedback_cropped", "feedback_crop_adaptive"]:
                        prompt_type_for_prompt = "feedback_cropped"
                    else:
                        prompt_type_for_prompt = prompt_type

                    # prompt_type_for_prompt = "feedback" if prompt_type in ["feedback_crop_final", "feedback_cropped", "feedback_crop_from_feedback"] else prompt_type
                    prompt = generate_prompt(
                        item_name=point_name,
                        prompt_type="feedback",
                        x_ticks=local_xticks,
                        y_ticks=local_yticks,
                        x_pixels=local_xpix,
                        y_pixels=local_ypix,
                        pred_feedback=last_pred,  # 模型原始预测值（还没映射）
                        feedback_round=1,
                        current_round=2
                    )

                    # prompt = generate_prompt(
                    #     item_name=point_name,
                    #     prompt_type=prompt_type_for_prompt,
                    #     x_ticks=local_xticks,
                    #     y_ticks=local_yticks,
                    #     pred_feedback=history_preds[-2:] if prompt_type.startswith("feedback") and len(
                    #         history_preds) > 0 else None,
                    #     feedback_round=FEEDBACK_START_ROUND,
                    #     current_round=valid_runs + 1
                    # )

                    print("\n==============================")
                    print(f"📌 Round {valid_runs + 1} | Point: {point_name} | Type: {prompt_type} - {image_type}")
                    print(f"🖼️  使用图像路径: {pred_img_path}")
                    print("📋 Prompt 内容如下：\n")
                    print(prompt)
                    print("==============================\n")

                    pred = await call_llm_response(prompt, pred_img_path, point_name)
                    pred = normalize_predicted_point(pred, point_name)

                    if pred is None:
                        print(f"⚠️ 第 {total_attempts + 1} 次预测失败 [{prompt_type} - {image_type}] @ {point_name}")
                        total_attempts += 1
                        if total_attempts >= MAX_ATTEMPTS:
                            print(f"⚠️ 连续预测失败达到上限 {MAX_ATTEMPTS}，停止该配置：{point_name} - {prompt_type} - {image_type}")
                            break
                        await asyncio.sleep(5)
                        continue

                    last_pred = pred
                    history_preds.append(pred)
                    valid_runs += 1
                    total_attempts += 1

                    print(f"✅ 成功 {valid_runs}/{REPEAT_TIMES} [{prompt_type} - {image_type}] @ {point_name}")

                    x_mapper = build_axis_mapping(dataset["x_ticks"], dataset["x_pixels"])
                    y_mapper = build_axis_mapping(dataset["y_ticks"], dataset["y_pixels"])

                    gt_x, gt_y = gt
                    pred_x, pred_y = pred

                    # ===== 1) 像素相对误差（你原来的逻辑） =====
                    gt_px = x_mapper(gt_x)
                    gt_py = y_mapper(gt_y)
                    pred_px = x_mapper(pred_x)
                    pred_py = y_mapper(pred_y)

                    img = Image.open(image_path)
                    img_width, img_height = img.size
                    px_rel_x, px_rel_y = compute_pixel_relative_error_xy(
                        pred_px, pred_py, gt_px, gt_py, img_width, img_height
                    )

                    # ===== 2) 数值绝对误差 =====
                    x_abs_err = abs(pred_x - gt_x)
                    y_abs_err = abs(pred_y - gt_y)

                    # ===== 3) 轴范围（直接用当前 chart 的 ticks） =====
                    x_ticks = dataset["x_ticks"]
                    y_ticks = dataset["y_ticks"]

                    x_min, x_max = min(x_ticks), max(x_ticks)
                    y_min, y_max = min(y_ticks), max(y_ticks)

                    x_range = x_max - x_min
                    y_range = y_max - y_min

                    # 避免除 0
                    x_err_over_range = x_abs_err / x_range if x_range != 0 else float("nan")
                    y_err_over_range = y_abs_err / y_range if y_range != 0 else float("nan")

                    # ===== 4) 联合 X+Y 归一化误差 =====
                    if np.isnan(x_err_over_range) or np.isnan(y_err_over_range):
                        xy_err_over_range = float("nan")
                    else:
                        xy_err_over_range = (x_err_over_range + y_err_over_range) / 2.0

                    # ===== 5) 其他已有误差 =====
                    x_re, y_re = compute_re(pred, gt)

                    records.append({
                        "chart_id": dataset["chart_id"],
                        "point_name": point_name,
                        "prompt_type": prompt_type,
                        "image_type": image_type,

                        "gt_x": gt_x,
                        "gt_y": gt_y,
                        "pred_x": pred_x,
                        "pred_y": pred_y,

                        "pixel_rel_x": px_rel_x,
                        "pixel_rel_y": px_rel_y,

                        "mae": compute_mae(pred, gt),
                        "x_re": x_re,
                        "y_re": y_re,

                        # ⭐ 新增：数值误差 + 轴范围 + 归一化误差
                        "x_abs_err": x_abs_err,
                        "y_abs_err": y_abs_err,
                        "x_range": x_range,
                        "y_range": y_range,
                        "x_err_over_range": x_err_over_range,
                        "y_err_over_range": y_err_over_range,
                        "xy_err_over_range": xy_err_over_range,
                    })

                    # x_mapper = build_axis_mapping(dataset["x_ticks"], dataset["x_pixels"])
                    # y_mapper = build_axis_mapping(dataset["y_ticks"], dataset["y_pixels"])
                    # gt_px = x_mapper(gt[0])
                    # gt_py = y_mapper(gt[1])
                    # pred_px = x_mapper(pred[0])
                    # pred_py = y_mapper(pred[1])
                    # img = Image.open(image_path)
                    # img_width, img_height = img.size
                    # px_rel_x, px_rel_y = compute_pixel_relative_error_xy(pred_px, pred_py, gt_px, gt_py, img_width,
                    #                                                      img_height)
                    #
                    # records.append({
                    #     "chart_id": dataset["chart_id"],
                    #     "point_name": point_name,
                    #     "prompt_type": prompt_type,
                    #     "image_type": image_type,
                    #     "gt_x": gt[0],
                    #     "gt_y": gt[1],
                    #     "pred_x": pred[0],
                    #     "pred_y": pred[1],
                    #     "pixel_rel_x": px_rel_x,
                    #     "pixel_rel_y": px_rel_y,
                    #     "mae": compute_mae(pred, gt),
                    #     "x_re": compute_re(pred, gt)[0],
                    #     "y_re": compute_re(pred, gt)[1]
                    # })

                    if total_attempts >= MAX_ATTEMPTS:
                        print(f"⚠️ 达到最大尝试次数 {MAX_ATTEMPTS}，停止该配置：{point_name} - {prompt_type} - {image_type}")
                        break

                # if prompt_type == "feedback_crop_final" and len(history_preds) > 0:
                if prompt_type in ["feedback_crop_final"] and len(history_preds) > 0:
                    final_img_path = f"temp/{dataset['chart_id']}/final_overlay_{dataset['chart_id']}_{point_name}_{prompt_type}_{image_type}.png"
                    os.makedirs(os.path.join("temp", dataset['chart_id']), exist_ok=True)

                    history_preds_resized = [
                        convert_data_coord_to_resized_crop_pixel(
                            data_coord=p,
                            x_ticks=dataset["x_ticks"],
                            y_ticks=dataset["y_ticks"],
                            x_pixels=dataset["x_pixels"],
                            y_pixels=dataset["y_pixels"],
                            crop_origin=(left, upper),
                            crop_size=(150, 150),
                            resize_size=(224, 224)
                        ) for p in history_preds
                    ]

                    # 复制图像作为起始画布
                    base_img = Image.open(pred_img_path).copy()

                    # 迭代绘制不同颜色的十字线
                    colors = ["red", "purple", "orange", "cyan", "magenta"]
                    for idx, (x, y) in enumerate(history_preds_resized):
                        color = colors[idx % len(colors)]
                        draw_crosshair_on_resized_image(
                            img_path=base_img,
                            coords=[(x, y)],
                            output_path=final_img_path,
                            color=color
                        )

                    print(f"🖼️ 已生成最终反馈图像：{final_img_path}")

                if prompt_type == "feedback" and len(history_preds) > 0:
                    feedback_final_results[(dataset["chart_id"], point_name)] = history_preds[-1]
                    import re, os
                    safe_chart_id = re.sub(r'[\\/*?:"<>| ]', "_", dataset["chart_id"])
                    tempy_dir = os.path.join("results_scatter_Gemini", safe_chart_id, "tempy")
                    os.makedirs(tempy_dir, exist_ok=True)

                    final_img_path = os.path.join(
                        tempy_dir,
                        f"final_overlay_{safe_chart_id}_{point_name}_{prompt_type}_{image_type}.png"
                    )

                    # final_img_path = f"temp/final_overlay_{dataset['chart_id']}_{point_name}_{prompt_type}_{image_type}.png"
                    # os.makedirs("temp", exist_ok=True)
                    generate_overlayed_image_multi_with_mapping(
                        original_img_path=pred_img_path,
                        pred_coords=history_preds,
                        x_ticks=local_xticks,
                        y_ticks=local_yticks,
                        x_pixels=local_xpix,
                        y_pixels=local_ypix,
                        output_path=final_img_path,
                        feedback_round=valid_runs,
                        draw_all=True
                    )
                    print(f"🖼️ 已生成最终反馈图像：{final_img_path}")

        return records

    records = []
    # 分批处理数据集
    if batch_size and batch_size > 0:
        total_batches = (len(datasets) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            batch_datasets = datasets[start:end]
            print(f"📦 处理批次 {batch_idx + 1}/{total_batches}（图表 {start + 1}-{min(end, len(datasets))}）")
            tasks = [run_for_dataset(ds) for ds in batch_datasets]
            batch_results = await asyncio.gather(*tasks)
            for res in batch_results:
                records.extend(res)
    else:
        tasks = [run_for_dataset(dataset) for dataset in datasets]
        results = await asyncio.gather(*tasks)
        for result in results:
            if result is not None:
                records.extend(result)

    df = pd.DataFrame(records)

    if df.empty:
        print("⚠️ No experiment records generated. No results to process.")
        return

    # ✅ 按图表ID拆分保存结果和评估图
    for chart_id, group_df in df.groupby("chart_id"):
        result_dir = os.path.join("results_scatter_Gemini", chart_id)
        os.makedirs(result_dir, exist_ok=True)

        result_path = os.path.join(result_dir, "experiment_results.csv")
        group_df.to_csv(result_path, index=False)

        print(f"✅ 已保存 {chart_id} 的结果至 {result_path}")

        # 筛选最后一轮，再进行评估（误差图、统计图等将保存到该图表子目录）
        final_df = filter_final_round_for_feedback(group_df)
        evaluate_results(final_df, result_dir=result_dir)

        print(f"📊 {chart_id} 的评估结果图已生成并保存在 {result_dir}")

    print("✅ 全部图表实验完成，结果按 chart_id 拆分保存。")

    # df = pd.DataFrame(records)
    # # 构建图表特定的结果目录
    # if chart_ids:
    #     result_dir = os.path.join('results', chart_ids[0])
    # else:
    #     result_dir = 'results'
    # os.makedirs(result_dir, exist_ok=True)
    # df.to_csv(os.path.join(result_dir, "experiment_results.csv"), index=False)
    # if not df.empty:
    #     df = filter_final_round_for_feedback(df)
    #     evaluate_results(df, result_dir=result_dir)
    #     print(f"✅ 全部实验完成，结果保存至 {result_dir}")
    # else:
    #     print("⚠️ No experiment records generated. No results to process.")


if __name__ == "__main__":
    import argparse
    import asyncio
    import socket

    async def check_api_endpoints():
        """检查API端点是否可用"""
        print("🔍 正在检查API端点可用性...")
        for url in API_URLS:
            try:
                # 从URL中提取主机和端口
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                
                # 尝试建立TCP连接
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)
                s.connect((host, port))
                s.close()
                print(f"✅ {url} - 可用")
            except Exception as e:
                print(f"❌ {url} - 不可用: {e}")
        print("🔍 API端点检查完成")

    async def main():
        global _session
        try:
            parser = argparse.ArgumentParser(description='图表数据批量处理工具')
            parser.add_argument('--batch-size', type=int, default=None,
                                help='每次处理的图表数量（批量大小）')
            parser.add_argument('--chart-ids', nargs='+', default=None,
                                help='指定处理的图表ID列表，如 --chart-ids chart08 chart11')
            parser.add_argument('--chart-path', default=None,
                                help='单独传入的图表图片路径；若未提供细分图片路径，则会同时作为 no_grid/with_grid/grid_with_grid 使用')
            parser.add_argument('--config-path', default=None,
                                help='单独传入的图表 config JSON 路径')
            parser.add_argument('--chart-id', default=None,
                                help='覆盖 config 中的 chart_id，单图模式可选')
            parser.add_argument('--no-grid-chart-path', default=None,
                                help='单图模式下 no_grid 图片路径；默认使用 --chart-path')
            parser.add_argument('--with-grid-chart-path', default=None,
                                help='单图模式下 with_grid 图片路径；默认使用 --chart-path')
            parser.add_argument('--grid-with-grid-chart-path', default=None,
                                help='单图模式下 grid_with_grid 图片路径；默认使用 --chart-path')
            parser.add_argument('--log-path', default=None,
                                help='保存完整控制台输出的日志文件路径；默认写入 scatter_bubble/logs/')
            args = parser.parse_args()
            setup_console_log(args.log_path)

            # 检查API端点
            await check_api_endpoints()

            custom_datasets = None
            if args.chart_path or args.config_path:
                if not args.chart_path or not args.config_path:
                    raise ValueError("单图模式需要同时提供 --chart-path 和 --config-path")
                custom_datasets = [
                    build_dataset_from_inputs(
                        config_path=args.config_path,
                        chart_path=args.chart_path,
                        chart_id=args.chart_id,
                        no_grid_chart_path=args.no_grid_chart_path,
                        with_grid_chart_path=args.with_grid_chart_path,
                        grid_with_grid_chart_path=args.grid_with_grid_chart_path,
                    )
                ]

            await run_experiment(
                batch_size=args.batch_size,
                chart_ids=args.chart_ids,
                datasets=custom_datasets,
            )
        except KeyboardInterrupt:
            print("\n⚠️ 程序被中断，正在清理资源...")
        except Exception as e:
            print(f"\n❌ 程序出错: {type(e).__name__} - {e}")
        finally:
            # 关闭aiohttp会话
            if _session and not _session.closed:
                await _session.close()
                print("✅ aiohttp会话已关闭")
            close_console_log()

    asyncio.run(main())

