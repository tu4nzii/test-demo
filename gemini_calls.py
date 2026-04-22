# -*- coding: utf-8 -*-
"""
Gemini-2.0-flash 简单对话模块
"""

import asyncio
import aiohttp
import json
import os

# ======== Gemini API 配置 ======== #

# API 端点
url = "https://api.vveai.com/v1/chat/completions"

# API 密钥列表（自动轮换）
API_KEYS = [
    "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d",
    "sk-2nzrUYD0JWLFzopWF477111f78E746AbAcA9Ed8534C3A481",
    "sk-CiD5WVUNIkBeXDgYB46b90C06aD24636BcEaBaFa993970C4",
    "sk-WvF4fU10VeOkfFMq579610Fc01E8496d827d0d3e04C44d0a",
    "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
]

# 当前使用的密钥索引
key_index = 0

# ======== 全局控制参数 ======== #

# 超时设置
BASE_TIMEOUT = aiohttp.ClientTimeout(total=180, connect=30, sock_connect=30, sock_read=120)

# 最大重试次数
MAX_RETRIES = 3

# ======== API 工具函数 ======== #

def get_headers() -> dict:
    """获取当前 key 对应的 headers"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_index]}"
    }


def rotate_key() -> None:
    """切换到下一个 key"""
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)
    print(f"🔑 已切换至新的 API Key [{key_index + 1}/{len(API_KEYS)}]")


# ======== 对话相关函数 ======== #

async def chat_with_gemini(messages: list) -> str:
    """
    与Gemini进行对话
    messages格式: [{"role": "user", "content": "消息内容"}, ...]
    """
    payload = {
        "model": "gemini-2.5-pro",
        "messages": messages,
        "temperature": 0.7
    }

    async with aiohttp.ClientSession(timeout=BASE_TIMEOUT) as session:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.post(url, headers=get_headers(), json=payload) as response:
                    if response.status == 429:
                        print(f"🚫 请求频率超限，切换 Key 重试...")
                        rotate_key()
                        await asyncio.sleep(3)
                        continue

                    # 处理响应
                    text = await response.text()
                    if response.status != 200:
                        print(f"⚠️ HTTP {response.status}: {text[:200]}")
                        await asyncio.sleep(2)
                        continue

                    # 解析响应
                    result = json.loads(text)
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return content
                    else:
                        print(f"⚠️ 响应格式错误: {text}")
                        await asyncio.sleep(2)
                        continue

            except Exception as e:
                print(f"❌ 第 {attempt} 次尝试失败: {e}")
                await asyncio.sleep(2)
                continue

    print("❌ 所有尝试均失败")
    return "抱歉，我暂时无法回应您的请求。"


async def run_chat():
    """
    运行交互式对话
    """
    print("====================================")
    print("Gemini-2.5-pro 简单对话工具")
    print("====================================")
    print("输入 'quit' 或 'exit' 退出对话")
    print("输入 'clear' 清空对话历史")
    print("====================================\n")
    
    # 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个友好的AI助手，请用自然、简洁的语言回答用户的问题。"}
    ]
    
    while True:
        try:
            # 获取用户输入
            user_input = input("用户: ").strip()
            
            # 处理特殊命令
            if user_input.lower() in ["quit", "exit"]:
                print("\n� 对话结束，再见！")
                break
            
            if user_input.lower() == "clear":
                messages = [
                    {"role": "system", "content": "你是一个友好的AI助手，请用自然、简洁的语言回答用户的问题。"}
                ]
                print("\n✅ 对话历史已清空")
                continue
            
            if not user_input:
                continue
            
            # 添加用户输入到对话历史
            messages.append({"role": "user", "content": user_input})
            
            # 获取Gemini响应
            print("\nAI: ", end="", flush=True)
            response = await chat_with_gemini(messages)
            
            # 显示响应
            print(response)
            
            # 添加AI响应到对话历史
            messages.append({"role": "assistant", "content": response})
            
            print("\n" + "-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 对话结束，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("\n" + "-" * 50 + "\n")


# ======== 测试函数 ======== #

async def test_all_keys():
    """
    测试所有 API 密钥是否可用
    """
    print("====================================")
    print("测试所有 API 密钥")
    print("====================================")
    
    global key_index
    original_index = key_index
    
    test_message = [{"role": "user", "content": "测试 API 密钥"}]
    
    for i, api_key in enumerate(API_KEYS):
        print(f"\n测试密钥 {i+1}/{len(API_KEYS)}: {api_key[:20]}...")
        
        # 切换到当前密钥
        key_index = i
        
        try:
            response = await chat_with_gemini(test_message)
            print(f"结果: {response}")
        except Exception as e:
            print(f"错误: {e}")
    
    # 恢复原始密钥索引
    key_index = original_index
    print("\n====================================")
    print("测试完成")
    print("====================================")

# ======== 主函数 ======== #

if __name__ == "__main__":
    # 测试所有密钥
    asyncio.run(test_all_keys())
    # 运行交互式对话
    # asyncio.run(run_chat())