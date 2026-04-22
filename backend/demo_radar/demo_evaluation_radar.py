import json
import os
from typing import List, Dict, Tuple, Optional
import base64
import requests
import re
import time
import cv2
import math
import numpy as np

class RadarChartEvaluator:
    def __init__(self):
        # 初始化API参数和配置
        self.api_key = "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d"
        self.url = "https://api.vveai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.llm_model = "gemini-2.5-pro"
        
        # 定义临时文件目录
        self.feedback_image_dir = './data/feedback'
        self.amplifier_image_dir = './data/amplifier/radar'
        
        # 创建必要的目录
        self._create_directories()
        
        # 存储结果
        self.results_by_image = {}
    
    def _create_directories(self):
        """创建必要的临时文件目录"""
        if not os.path.exists(self.feedback_image_dir):
            os.makedirs(self.feedback_image_dir)
        if not os.path.exists(self.amplifier_image_dir):
            os.makedirs(self.amplifier_image_dir)
    
    def load_dataset(self, json_path: str) -> dict:
        """加载单个JSON数据集文件"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            return {}
    
    def extract_json_response(self, content: str) -> Optional[dict]:
        """从LLM响应中提取JSON格式的内容"""
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            json_str = match.group(1)
            return json.loads(json_str)
        except Exception as e:
            print(f"❌ JSON解析失败: {e}")
            return None
    
    def validate_coordinates(self, coords: Tuple) -> bool:
        """验证坐标是否有效"""
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            return False
        valid = lambda x: isinstance(x, (int, float)) or x is None
        return valid(coords[0]) and valid(coords[1])
    
    def encode_image(self, image, center_x: int, center_y: int, arg_a: float, arg_b: float, r_ticks: List[float]) -> np.ndarray:
        """在图像上绘制加密网格线"""
        line_color = (128, 128, 128)
        thickness = 1
        if r_ticks and r_ticks[0] == 0:
            r_ticks.pop(0)
        
        count = 0
        for tick in r_ticks:
            count += 1
            if count % 4 == 0:
                continue
            radius = int(arg_a * tick + arg_b)
            circumference = int(2 * math.pi * radius)
            dash_length = 2
            gap_length = 3
            
            for i in range(0, circumference, dash_length + gap_length):
                angle_start = 2 * math.pi * i / circumference
                angle_end = 2 * math.pi * (i + dash_length) / circumference
                x1 = int(center_x + radius * math.cos(angle_start))
                y1 = int(center_y + radius * math.sin(angle_start))
                x2 = int(center_x + radius * math.cos(angle_end))
                y2 = int(center_y + radius * math.sin(angle_end))
                cv2.line(image, (x1, y1), (x2, y2), line_color, thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def crop_axis_label_region(self, image_path: str, center_x: int, center_y: int, angle_deg: float, 
                             outer_radius: int, angle_width: int = 30, inner_radius: int = 0, 
                             label_offset: int = 30, scale_factor: float = 1.0, 
                             r_ticks: List[float] = None, arg_a: float = 0, arg_b: float = 0) -> np.ndarray:
        """根据角度裁剪扇形区域"""
        if r_ticks is None:
            r_ticks = []
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        h, w = image.shape[:2]
        
        # 绘制加密网格线
        self.encode_image(image, center_x, center_y, arg_a, arg_b, r_ticks)
        
        # 创建掩码（黑色背景）
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 计算扇形角度范围
        start_angle = angle_deg - angle_width / 2
        end_angle = angle_deg + angle_width / 2
        
        # 在图像上添加角度文字标注
        for tick in r_ticks:
            radius = int(arg_a * tick + arg_b)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.3
            font_color = (0, 0, 0)  # 黑色
            thickness = 1
            
            # 计算起始角度文字的位置
            start_angle_rad = math.radians(start_angle + 4)
            text_radius = radius 
            
            # 获取文字尺寸以实现中心对齐
            text = str(int(tick) if tick % 1 == 0 else tick)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # 计算文字中心坐标
            start_x_center = int(center_x + text_radius * math.cos(start_angle_rad))
            start_y_center = int(center_y - text_radius * math.sin(start_angle_rad))
            
            # 调整为左下角坐标
            start_x = start_x_center - text_size[0] // 2
            start_y = start_y_center + text_size[1] // 2
            
            # 计算结束角度文字的位置
            end_angle_rad = math.radians(end_angle - 4)
            end_x_center = int(center_x + text_radius * math.cos(end_angle_rad))
            end_y_center = int(center_y - text_radius * math.sin(end_angle_rad))
            end_x = end_x_center - text_size[0] // 2
            end_y = end_y_center + text_size[1] // 2
            
            # 添加文字标注
            cv2.putText(image, text, (start_x, start_y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
            cv2.putText(image, text, (end_x, end_y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
        
        # 转换为OpenCV角度系统
        start_angle_cv = -end_angle
        end_angle_cv = -start_angle
        
        # 绘制扇形掩码
        out_axes = (outer_radius, outer_radius)
        cv2.ellipse(
            mask,
            (center_x, center_y),
            out_axes,
            angle=0,
            startAngle=start_angle_cv,
            endAngle=end_angle_cv,
            color=255,
            thickness=-1,  # 填充
            lineType=cv2.LINE_AA
        )
        
        # 如果有内半径，绘制内圆并减去
        if inner_radius > 0:
            inner_axes = (inner_radius, inner_radius)
            cv2.ellipse(
                mask,
                (center_x, center_y),
                inner_axes,
                angle=0,
                startAngle=start_angle_cv - 10,
                endAngle=end_angle_cv + 10,
                color=0,
                thickness=-1,
                lineType=cv2.LINE_AA
            )
        
        # 应用掩码
        sector_img = cv2.bitwise_and(image, image, mask=mask)
        # 将背景从黑色改为白色
        sector_img[mask == 0] = 255
        
        # 计算扇形边界框并裁剪
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image
        
        x, y, w_sector, h_sector = cv2.boundingRect(coords)
        
        # 确保裁剪区域在图像有效范围内
        img_height, img_width = sector_img.shape[:2]
        y_start = max(0, y - 20)
        y_end = min(img_height, y + h_sector + 20)
        x_start = max(0, x - 20)
        x_end = min(img_width, x + w_sector + 20)
        
        crop_img = sector_img[y_start:y_end, x_start:x_end]
        
        # 添加图像放大功能
        if scale_factor != 1.0 and crop_img.size > 0:
            new_width = int(crop_img.shape[1] * scale_factor)
            new_height = int(crop_img.shape[0] * scale_factor)
            crop_img = cv2.resize(
                crop_img,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC
            )
        
        return crop_img
    
    def draw_angle_indicator(self, image: np.ndarray, center_x: int, center_y: int, target_angle: float, radius: int, 
                           arc_color: Tuple[int, int, int] = (0, 0, 255), line_color: Tuple[int, int, int] = (0, 0, 255),
                           arc_thickness: int = 2, line_thickness: int = 2, arc_angle_width: int = 10, 
                           line_length_ratio: float = 0.3) -> np.ndarray:
        """在图像上绘制特定角度的扇形弧线和对应轴上的标记"""
        # 计算弧线的起始和结束角度
        start_angle = -target_angle - arc_angle_width // 2
        end_angle = -target_angle + arc_angle_width // 2  
        
        # 绘制扇形弧线
        cv2.ellipse(image, (center_x, center_y), (radius, radius), 0, start_angle, end_angle, 
                   arc_color, arc_thickness, lineType=cv2.LINE_AA)
        
        # 将角度转换为弧度并计算线段的端点坐标
        angle_rad = math.radians(target_angle)
        outer_x = int(center_x + (radius + line_length_ratio * radius) * math.cos(angle_rad))
        outer_y = int(center_y - (radius + line_length_ratio * radius) * math.sin(angle_rad))
        
        # 计算指向圆心的线段的内端点坐标
        inner_radius = radius * (1 - line_length_ratio)
        inner_x = int(center_x + inner_radius * math.cos(angle_rad))
        inner_y = int(center_y - inner_radius * math.sin(angle_rad))
        
        # 绘制指向圆心的线
        cv2.line(image, (outer_x, outer_y), (inner_x, inner_y), line_color, line_thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def generate_prompt(self, item_name: str, prompt_type: str, dataset: dict, tick: float = 0) -> str:
        """根据图表类型和提示类型生成对应的提示文本"""
        # 获取图表类型
        chart_type = dataset.get('chart_type', '')
        #start_angle = dataset.get('start_angle', 0)
        # print(f"当前处理: {item_name}, 网格类型: {prompt_type}, 图表类型: {chart_type}")
        # print(f"{dataset.get('axis_labels')[str(start_angle)]}对应范围为{start_angle}-{dataset.get('axes_angles')[1]}")
        if prompt_type == "with_grid":
            if chart_type == 'radar':
                return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} angle positions

Your task is to estimate the value of the data point labeled "{item_name}":

1.Locate the "{item_name}" data point on the radar chart.
2.Estimate its radial position by interpolating between concentric circles.** Remember to always interpolate and make good use of the encrypted grid **

⚠️ Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
           
        elif prompt_type == "no_grid":
            if chart_type == 'radar':
                return f'''
Your task is to estimate the value of the data point labeled "{item_name}":

⚠️ Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
        elif prompt_type == "feedback":
            if chart_type == 'radar':
                return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} positions

Your task is to estimate the value of the data point labeled "{item_name}":

**重要提示**：图表中已添加红色圆环，表示上一轮对"{item_name}"的预测值约为{tick}。
请比较红色圆环与真实数据点的位置差距，重新优化您的预测：
1. 确定红色圆环与真实数据点之间的位置关系（偏内、偏外）
2. 根据这种关系，调整您的预测值
3. 确保新的预测值与真实点的位置对齐 以实现尽可能准确的预测

⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
        elif prompt_type == "amplifier":
            if chart_type == 'radar':
                return f'''
您正在分析雷达图的一部分。该图在二维平面上使用从同一点出发的坐标轴展示多变量数据。

- 共有{len(dataset.get('series_color', {}))}个实体：{', '.join(dataset.get('series_color', {}).keys())}，分别对应颜色{', '.join(dataset.get('series_color', {}).values())}
现在的局部放大图为{item_name.split(',')[1].strip()}轴对应的局部放大
您的任务是估计标记为"{item_name}"对应的值，即{item_name}对应实体颜色的数值。
请先找到{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
然后找到该颜色对应的点，并插值出数值

⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
            elif chart_type == 'rose':
                return f'''
                该图片为玫瑰图中{item_name}数据的放大，请你根据该图片，估计{item_name}对应的数值。
-**找到扇形并且找到其最远端的边界**
-然后找到该边界处于哪两个基准线之间
-最后依据基准线的数值，插值出数值
⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
        else:
            raise ValueError("Unknown prompt_type")
    
    def call_llm_response(self, prompt: str, image_path: str, item_name: str, dataset: dict) -> Tuple[Optional[float], Optional[float]]:
        """调用LLM接口获取响应"""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            print(f"❌ 读取图像文件失败: {e}")
            return (None, None)
        
        max_retries = 10
        retry_delay = 0.5  # 秒
        retry_count = 0
        
        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.5
        }
        
        while retry_count < max_retries:
            try:
                response = requests.post(url=self.url, headers=self.headers, json=payload, timeout=10)
                response.raise_for_status()  # 检查HTTP错误状态码
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                coords_json = self.extract_json_response(content)
                
                if coords_json and "datapoints" in coords_json:
                    for item in coords_json["datapoints"]:
                        if item_name in item:
                            coords = item[item_name]
                            if self.validate_coordinates(coords):
                                return tuple(coords)
                
                # 如果未找到数据但请求成功，不重试直接返回
                return (None, None)
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                print(f"❌ 请求异常: {e}, 正在进行第 {retry_count}/{max_retries} 次重试...")
                if retry_count < max_retries:
                    time.sleep(retry_delay)
            except Exception as e:
                retry_count += 1
                print(f"❌ 解析异常: {e}, 正在进行第 {retry_count}/{max_retries} 次重试...")
                if retry_count < max_retries:
                    time.sleep(retry_delay)
        
        print(f"⚠️ 已达到最大重试次数 ({max_retries}次)")
        return (None, None)
    
    def process_single_image(self, json_path: str) -> None:
        """处理单个图像的评估逻辑"""
        # 加载数据集
        dataset = self.load_dataset(json_path)
        if not dataset:
            print("❌ 数据集为空，无法进行评估")
            return
        
        # 确保是雷达图且有数据
        if dataset.get('chart_type') != 'radar' or not dataset.get('data'):
            print("❌ 不是雷达图或没有数据，跳过处理")
            return
        
        chart_id = dataset.get('chart_id', 'unknown')
        print(f"开始处理图表: {chart_id}")
        
        # 初始化结果字典
        self.results_by_image[chart_id] = {
            'chart_type': dataset.get('chart_type', 'rose'),
            'data': {}
        }
        
        # 遍历每个实体及其数据点（双层遍历）
        for entity_name, entity_data in dataset.get('data', {}).items():
            # 初始化实体结果字典
            self.results_by_image[chart_id]['data'][entity_name] = {}
            
            # 遍历该实体的每个数据点
            for axis_label, value in entity_data.items():
                # 初始化该数据点的结果字典
                self.results_by_image[chart_id]['data'][entity_name][axis_label] = {}
                
                # 处理带网格和无网格两种情况
                for grid_type in ['with_grid', 'no_grid']:
                    # 获取对应的图像路径
                    image_path = dataset.get("image_paths", {}).get(grid_type)
                    if not image_path:
                        print(f"⚠️ 未找到{grid_type}对应的图像路径，跳过该类型处理")
                        continue
                    
                    # 替换路径分隔符以确保兼容性
                    image_path = image_path.replace('\\', '/')
                    
                    # 生成提示并调用LLM
                    try:
                        # 将实体名称和轴标签都传递给prompt生成函数
                        prompt = self.generate_prompt(f"{entity_name}_{axis_label}", grid_type, dataset)
                        # print(f"当前处理: {entity_name} - {axis_label}, 网格类型: {grid_type}, 提示: {prompt}")
                        coords = self.call_llm_response(prompt, image_path, f"{entity_name}_{axis_label}", dataset)
                        
                        # 获取轴标签和角度映射
                        axis_labels = dataset.get('axis_labels', {})
                        label_to_angle = {v: int(k) for k, v in axis_labels.items()}  # 建立标签到角度的反向映射
                        
                        # 检查轴标签是否存在
                        if axis_label not in label_to_angle:
                            print(f"警告: 未找到{axis_label}对应的角度，跳过当前数据点处理")
                            continue
                        
                        angle_width = int(360 / len(axis_labels)) if axis_labels else 30
                        target_angle = label_to_angle[axis_label]
                        
                        # 处理反馈模式
                        if grid_type == 'with_grid' and coords[0] is not None:
                            feedback_counts = 0
                            feedback_tick = [coords[0]] if coords[0] is not None else [value]
                            feedback_times = 1
                            
                            while feedback_counts > 0:
                                try:
                                    temp_image = cv2.imread(dataset["image_paths"][grid_type].replace('\\', '/'))
                                    if temp_image is None:
                                        print(f"❌ 无法读取图像: {dataset['image_paths'][grid_type]}")
                                        break
                                    
                                    feedback_image = temp_image.copy()
                                    feedback_image_path = os.path.join(self.feedback_image_dir, 
                                                                     f'{chart_id}_{grid_type}_{entity_name}_{axis_label}_{feedback_times}.png')
                                    
                                    center_x = dataset["pred_coords"][0]
                                    center_y = dataset["pred_coords"][1]
                                    a = dataset["argument"]["a"]
                                    b = dataset["argument"]["b"]
                                    pre_r = int(a * feedback_tick[-1] + b)
                                    
                                    # 绘制角度指示器
                                    feedback_image = self.draw_angle_indicator(feedback_image, center_x, center_y, 
                                                                              target_angle, pre_r, line_thickness=2, 
                                                                              arc_angle_width=angle_width, 
                                                                              line_length_ratio=0.05)
                                    
                                    # 保存反馈图像
                                    cv2.imwrite(feedback_image_path, feedback_image)
                                    
                                    # 生成反馈提示并调用LLM
                                    feedback_prompt = self.generate_prompt(f"{entity_name}_{axis_label}", 'feedback', dataset, feedback_tick[-1])
                                    feedback_coords = self.call_llm_response(feedback_prompt, feedback_image_path, 
                                                                             f"{entity_name}_{axis_label}", dataset)
                                    
                                    # 更新反馈tick列表
                                    if feedback_coords[0] is not None:
                                        feedback_tick.append(feedback_coords[0])
                                    else:
                                        feedback_tick.append(coords[0])
                                    
                                    print(f"反馈结果: {feedback_tick}")
                                    
                                    # 删除临时反馈图像
                                    if os.path.exists(feedback_image_path):
                                        os.remove(feedback_image_path)
                                    
                                    feedback_times += 1
                                    feedback_counts -= 1
                                    
                                except Exception as e:
                                    print(f"❌ 反馈处理异常: {e}")
                                    break
                            
                            # 保存反馈结果
                            if feedback_tick:
                                self.results_by_image[chart_id]['data'][entity_name][axis_label]['feedback'] = feedback_tick
                            
                            # 处理放大模式
                            try:
                                amplifier_path = dataset["image_paths"].get('no_grid', '').replace('\\', '/')
                                if not amplifier_path or not os.path.exists(amplifier_path):
                                    print(f"⚠️ 未找到no_grid图像路径或文件不存在: {amplifier_path}")
                                    continue
                                
                                center_x, center_y = dataset["pred_coords"]
                                arg_a = dataset["argument"]["a"]
                                arg_b = dataset["argument"]["b"]
                                radius = int(arg_a * coords[0] + arg_b) if coords[0] is not None else 0
                                r_ticks = dataset["r_ticks"]
                                
                                # 确定内外半径
                                if coords[0] is not None:
                                    inner_radius = 0
                                    outer_radius = radius + 150
                                    # 确保不超过最大半径
                                    if outer_radius > dataset['r_ticks'][-1] * arg_a + arg_b:
                                        outer_radius = radius
                                else:
                                    inner_radius = 0
                                    outer_radius = radius
                                print(f"当前处理: {entity_name} - {axis_label}, 半径范围: {inner_radius}-{outer_radius}")
                                # 裁剪并放大图像
                                scale_factor = 2
                                amplifier_image_path = os.path.join(self.amplifier_image_dir, 
                                                                    f'{chart_id}_{grid_type}_{entity_name}_{axis_label}.png')
                                
                                amplifier_image = self.crop_axis_label_region(amplifier_path, center_x, center_y, target_angle, outer_radius, angle_width, inner_radius, 30, scale_factor, r_ticks, arg_a, arg_b)
                                
                                # 保存放大图像
                                if amplifier_image.size > 0:
                                    cv2.imwrite(amplifier_image_path, amplifier_image)
                                else:
                                    print(f"警告: 无法保存图像 {amplifier_image_path}，因为裁剪区域为空")
                                    continue
                                
                                # 生成放大提示并调用LLM
                                amplifier_prompt = self.generate_prompt(f"{entity_name}_{axis_label}", 'amplifier', dataset)
                                amplifier_coords = self.call_llm_response(amplifier_prompt, amplifier_image_path, 
                                                                          f"{entity_name}_{axis_label}", dataset)
                                print(f"放大结果: {amplifier_prompt}")
                                # 保存放大结果
                                if amplifier_coords is not None:
                                    self.results_by_image[chart_id]['data'][entity_name][axis_label]['amplifier'] = amplifier_coords[0]
                            except Exception as e:
                                print(f"❌ 放大处理异常: {e}")
                    except Exception as e:
                        print(f"❌ 处理{entity_name}的{axis_label}时异常: {e}")
                        continue
                    
                    # 保存结果
                    if coords is not None:
                        self.results_by_image[chart_id]['data'][entity_name][axis_label][grid_type] = coords
                    self.results_by_image[chart_id]['data'][entity_name][axis_label]['origin'] = value
                    
                    # 打印结果
                    if grid_type == 'with_grid':
                        amplifier_value = self.results_by_image[chart_id]['data'][entity_name][axis_label].get('amplifier', 'N/A')
                        print(f"{entity_name} - {axis_label} origin:{value} {grid_type}:{coords} amplifier:{amplifier_value}")
                    else:
                        print(f"{entity_name} - {axis_label} origin:{value} {grid_type}:{coords}")
    
    def save_results(self, output_path: str = None) -> None:
        """保存结果到JSON文件"""
        if not self.results_by_image:
            print("❌ 没有结果可保存")
            return
        
        # 如果未指定输出路径，则使用默认路径
        if output_path is None:
            # 获取第一个图表的类型作为文件名的一部分
            first_chart_id = next(iter(self.results_by_image.keys()), '')
            chart_type = self.results_by_image.get(first_chart_id, {}).get('chart_type', 'unknown')
            output_path = f'coordinates_by_image_{chart_type}_{self.llm_model}.json'
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results_by_image, f, ensure_ascii=False, indent=4)
            print(f"✅ 结果已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")

# 主程序入口
if __name__ == '__main__':
    # 创建评估器实例
    evaluator = RadarChartEvaluator()
    
    # 指定要处理的单个JSON文件路径
    json_file_path = './data/output/radar/result/radar_001_evalution_datasets.json'
    
    # 处理单个图像
    evaluator.process_single_image(json_file_path)
    
    # 保存结果
    evaluator.save_results()