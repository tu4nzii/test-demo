import cv2
import base64
import numpy as np
import json
import requests
import re
import os


class ChartTypeDetector:
    """图表类型检测器类
    使用大模型来判断图表类型（玫瑰图、雷达图等）
    """
    
    def __init__(self):
        """初始化图表类型检测器
        配置API参数和支持的图表类型列表
        """
        # 配置参数 - 复用项目中的大模型API配置
        self.api_key = "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
        self.url = "https://api.vveai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 支持的图表类型列表
        self.supported_types = ["rose", "radar"]
    
    def extract_json_response(self, content: str):
        """从LLM响应中提取JSON内容
        
        Args:
            content: LLM返回的原始内容
            
        Returns:
            dict: 解析后的JSON对象，失败则返回None
        """
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            json_str = match.group(1)
            return json.loads(json_str)
        except Exception as e:
            print(f"❌ JSON解析失败: {e}")
            return None
    
    def image_to_base64(self, image_path):
        """将图像转换为base64编码
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            str: base64编码的图像数据
        """
        # 读取图像并转换格式
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        
        # 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保数据类型正确
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        
        # 编码为JPEG并转换为base64
        success, encoded_image = cv2.imencode('.jpg', image_rgb)
        if not success:
            raise ValueError("图像编码失败")
        
        # 转换为base64字符串
        image_data = np.ascontiguousarray(encoded_image)
        return base64.b64encode(image_data).decode('utf-8')
    
    def detect_chart_type(self, image_path):
        """检测图表类型
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            dict: 包含检测结果的字典，格式为 {"type": "chart_type", "confidence": float}
        """
        try:
            # 将图像转换为base64
            base64_image = self.image_to_base64(image_path)
            
            # 构建提示和请求体
            prompt = f"""
            请你分析这个图表图像，并判断它属于哪种图表类型。
            
            可能的图表类型包括：
            - rose: 玫瑰图，通常有多个扇区从中心向外延伸，类似于饼图但角度均匀分布
            - radar: 雷达图，有多个坐标轴从中心向外辐射，形成多边形
            
            请严格按照以下JSON格式返回结果：
            ```json
            {{
                "type": "<chart_type>",
                "confidence": <confidence_score>
            }}
            ```
            
            其中：
            - <chart_type> 必须是 "rose" 或 "radar" 中的一个
            - <confidence_score> 是一个0到1之间的浮点数，表示你对判断结果的置信度
            """
            
            # 构建请求体
            payload = {
                "model": "gpt-4.1",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            # 发送请求到LLM API
            response = requests.post(self.url, headers=self.headers, data=json.dumps(payload))
            
            # 检查响应状态
            if response.status_code != 200:
                raise Exception(f"API请求失败: {response.status_code}, {response.text}")
            
            # 解析响应内容
            response_data = response.json()
            if "choices" not in response_data or len(response_data["choices"]) == 0:
                raise Exception("无效的API响应格式")
            
            # 提取LLM生成的内容
            content = response_data["choices"][0]["message"]["content"]
            
            # 提取JSON响应
            result = self.extract_json_response(content)
            
            # 验证结果格式
            if result is None or "type" not in result or "confidence" not in result:
                raise ValueError("LLM返回的结果格式不符合要求")
            
            # 验证图表类型是否在支持列表中
            chart_type = result["type"]
            if chart_type not in self.supported_types:
                raise ValueError(f"不支持的图表类型: {chart_type}")
            
            # 验证置信度是否在有效范围内
            confidence = result["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                raise ValueError(f"无效的置信度值: {confidence}")
            
            print(f"✅ 图表类型检测成功: {chart_type}, 置信度: {confidence}")
            return {
                "type": chart_type,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            print(f"❌ 图表类型检测失败: {e}")
            # 返回默认值，以便系统可以继续工作
            return {
                "type": "rose",  # 默认类型为玫瑰图
                "confidence": 0.5,
                "error": str(e)
            }


# 辅助函数：快速检测图表类型
def detect_chart(image_path):
    """检测图表类型的便捷函数
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        dict: 检测结果
    """
    detector = ChartTypeDetector()
    return detector.detect_chart_type(image_path)

if __name__ == "__main__":
    # 测试图表类型检测
    test_image = "./data/upload/radar_000.png"
    result = detect_chart(test_image)
    print(result)