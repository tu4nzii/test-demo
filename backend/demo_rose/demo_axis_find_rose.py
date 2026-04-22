import numpy as np
import math
import json
import cv2
import re
import requests
import base64
import os

class RoseChartAxisFinder:
    def __init__(self):
        # 配置参数
        self.api_key = "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d"
        self.url = "https://api.vveai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 统一输出路径配置
        self.output_dir = "./data/output/rose"  # 主输出目录
        self.axes_output_dir = os.path.join(self.output_dir)  # 轴线检测结果目录
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.axes_output_dir, exist_ok=True)
        
        # 处理结果存储
        self.axes_angles = []
        self.axis_labels = {}  # 键:角度, 值:标签
        self.center = [0, 0]
        self.radius = 0

    def extract_json_response(self, content: str):
        """从LLM响应中提取JSON内容"""
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            json_str = match.group(1)
            return json.loads(json_str)
        except Exception as e:
            print(f"❌ JSON解析失败: {e}")
            return None

    def crop_axis_label_region(self, image_path, center_x, center_y, angle_deg, radius, 
                              label_offset=100, label_width=150, label_height=150):
        """根据角度裁剪坐标轴名称区域"""
        # 读取图像并转换角度为弧度
        image = cv2.imread(image_path)
        angle_rad = math.radians(angle_deg)
        
        # 计算名称区域中心坐标（在角度方向上，距离圆心radius+offset处）
        label_center_x = int(center_x + (radius + label_offset) * math.cos(angle_rad))
        label_center_y = int(center_y - (radius + label_offset) * math.sin(angle_rad))  # 图像y轴向下，故减号
        
        # 计算裁剪区域左上角和右下角坐标
        x1 = max(0, label_center_x - label_width // 2)
        y1 = max(0, label_center_y - label_height // 2)
        x2 = min(image.shape[1], label_center_x + label_width // 2)
        y2 = min(image.shape[0], label_center_y + label_height // 2)
        
        # 裁剪区域并返回
        return image[y1:y2, x1:x2]

    def call_llm_letter(self, crop_img):
        """调用LLM识别图像中的字母"""
        # 转换图像格式并确保数据类型正确
        image_area = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
        if image_area.dtype != np.uint8:
            image_area = image_area.astype(np.uint8)
            print("已转换图像数据类型为uint8")
            
        # 编码图像为JPEG并转换为base64
        success, encoded_image = cv2.imencode('.jpg', image_area)
        if not success:
            print("图像编码失败")
            return None
            
        image_data = np.ascontiguousarray(encoded_image)
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # 构建提示和请求体
        prompt = f"""
        请你分析该图片中的字母，并返回，该字母为黑色，且处在大约中心位置
        ```json
        {{
            "letter": <字母>
        }}
        如果无法识别某个值，请使用 `null`
        """
        
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}} ,
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.5
        }
        
        try:
            response = requests.post(url=self.url, headers=self.headers, json=payload)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            data = self.extract_json_response(content)
            return data
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
            return None

    def call_llm_nums(self, image_path: str):
        """调用LLM识别图像中的轴的数量和名称"""
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
        # 构建提示和请求体
        prompt = f"""
        请你分析该图片中的含有的轴的名称的个数，并返回，轴的延申最外出有字母组合的轴名称，轴的名称为英文
        轴的名称个数和轴的个数应该保持相同，轴的名称不重复
        记住，只有延长线上有名称的才算一个轴，如果没有名称则不算，比如有些自带的轴只是为了标注刻度，这样的就不算轴

        请判断轴是在色块的中心位置，还是在色块的边缘位置
        ```json
        {{
            "axis_name": <轴的名称>,
            "nums": <轴的名称个数>,
            "position": <轴的位置>,
            "reason": <原因>
        }}
        如果无法识别某个值，请使用 `null`
        """
        
        payload = {
            "model": "gpt-4.1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}} ,
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.5
        }
        
        try:
            response = requests.post(url=self.url, headers=self.headers, json=payload)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            data = self.extract_json_response(content)
            return data
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {e}")
            print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")
            return None

    def get_start_angle(self, image_path: str, center_x: int, center_y: int, radius: int):
        """获取起始角度"""
        pred_angle = [0, 90]
        start_angle = []
        
        for angle in pred_angle:
            image_area = self.crop_axis_label_region(image_path, center_x, center_y, angle, radius)
            data = self.call_llm_letter(image_area)
            if data and data["letter"] != 'None' and data["letter"] is not None:
                start_angle.append(angle)
                
        print(f"找到标签的角度: {start_angle}")
        if start_angle:
            return start_angle[0]
        else:
            return None

    def find_rose_axes(self, image_path: str, center, start_angle: int, max_radius):
        """识别玫瑰图的轴线角度"""
        img = cv2.imread(image_path)
        axes_angles = []
        
        # 调用LLM识别轴的数量
        axes = self.call_llm_nums(image_path)
        print(f"LLM识别结果: {axes}")
        
        if not axes or 'nums' not in axes:
            print("无法识别轴的数量")
            return []
            
        axes_nums = axes['nums']
        
        # 计算每个轴的角度
        for i in range(axes_nums):
            best_interval = int(360 / axes_nums)
            angle = round(start_angle + i * best_interval) % 360
            axes_angles.append(angle)
        
        # 可视化结果（可选）
        output_img = img.copy()
        for angle in axes_angles:
            current_angle_rad = math.radians(angle)
            
            # 沿着找到的角度画一条线
            end_x = int(center[0] + max_radius * math.cos(current_angle_rad))
            end_y = int(center[1] + max_radius * math.sin(current_angle_rad))
            
            # 在图像上绘制红线，用于可视化
            cv2.line(output_img, center, (end_x, end_y), (0, 0, 255), 1)
            
            # 在终点处绘制一个圆点
            cv2.circle(output_img, (end_x, end_y), 2, (0, 255, 0), -1)
        
        print(f"找到的轴线角度: {axes_angles}")
        
        # 保存可视化结果（使用统一的输出目录）
        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)
        # output_path = os.path.join(self.axes_output_dir, f"axes_detected_{file_name}{file_ext}")
        # cv2.imwrite(output_path, output_img)
        # print(f"轴线检测结果已保存至: {output_path}")
        
        self.axes_angles = axes_angles
        return axes_angles

    def recognize_axis_labels(self, image_path: str, center, radius, axes_angles):
        """识别每个轴的标签"""
        axis_labels = {}
        
        for axis in axes_angles:
            try:
                crop_img = self.crop_axis_label_region(image_path, center[0], center[1], axis, radius)
                axis_data = self.call_llm_letter(crop_img)
                
                # 提取识别结果
                if isinstance(axis_data, dict) and 'letter' in axis_data and axis_data['letter'] is not None:
                    letter = axis_data['letter']
                    axis_labels[axis] = letter
                    print(f"轴角度: {axis}, 识别结果: {letter}")
                else:
                    print(f"轴角度: {axis}, 识别结果无效: {axis_data}")
            except Exception as e:
                print(f"处理轴角度 {axis} 时发生错误: {str(e)}")
                continue
        
        self.axis_labels = axis_labels
        return axis_labels

    def process_single_image(self, image_path, center=None, radius=None, output_json_path=None):
        """处理单张玫瑰图，识别轴线和标签"""
        try:
            # 如果未指定圆心和半径，尝试从JSON文件中读取
            if center is None or radius is None:
                # 使用统一的输出目录查找JSON文件
                base_name = os.path.basename(image_path)
                file_name, _ = os.path.splitext(base_name)
                json_path = os.path.join(self.output_dir, f"{file_name}.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        
                    if center is None and 'pred_coords' in json_data:
                        center = json_data['pred_coords']
                        print(f"从JSON文件中读取圆心: {center}")
                        
                    if radius is None and 'argument' in json_data and 'r_ticks' in json_data:
                        a = json_data['argument']['a']
                        b = json_data['argument']['b']
                        max_tick = json_data['r_ticks'][-1]
                        radius = a * max_tick + b - 5
                        print(f"计算得到半径: {radius}")
                        
            # 确保有圆心和半径
            if center is None:
                raise ValueError("未提供圆心坐标，也无法从JSON文件中读取")
            if radius is None:
                raise ValueError("未提供半径，也无法从JSON文件中计算")
            
            self.center = center
            self.radius = radius
            
            # 获取起始角度
            start_angle = self.get_start_angle(image_path, center[0], center[1], radius)
            if start_angle is None:
                print("无法确定起始角度")
                return None
            
            print(f"起始角度: {start_angle}")
            
            # 识别轴线
            found_axes = self.find_rose_axes(image_path, center, start_angle, radius)
            if not found_axes:
                print("未找到任何轴线")
                return None
            
            # 识别轴标签
            axis_labels = self.recognize_axis_labels(image_path, center, radius, found_axes)
            
            # 准备结果
            result = {
                'image_path': image_path,
                'center': center,
                'radius': radius,
                'start_angle': start_angle,
                'axes_angles': found_axes,
                'axis_labels': axis_labels
            }
            
            # 保存结果到JSON文件（使用统一的输出目录）
            base_name = os.path.basename(image_path)
            file_name, _ = os.path.splitext(base_name)
            
            if output_json_path is None:
                output_json_path = os.path.join(self.output_dir, f"{file_name}_axes.json")
                
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"识别结果已保存至: {output_json_path}")
            
            # 如果原JSON文件存在，更新它
            original_json_path = os.path.join(self.output_dir, f"{file_name}.json")
            if os.path.exists(original_json_path):
                with open(original_json_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                    
                # 更新原数据
                original_data['axis_labels'] = axis_labels
                original_data['axes_angles'] = found_axes
                original_data['start_angle'] = start_angle
                
                # 保存更新后的数据
                with open(original_json_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, ensure_ascii=False, indent=2)
                    
                print(f"原JSON文件已更新: {original_json_path}")
            
            return result
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 示例用法
    finder = RoseChartAxisFinder()
    
    # 可以在此处修改输出路径（如果需要）
    # finder.output_dir = "custom_output"
    # finder.axes_output_dir = os.path.join(finder.output_dir, "custom_axes")
    
    # 指定要处理的图像路径
    image_path = "./data/rose/rose_001.png"  # 根据需要修改
    
    # 可以手动指定圆心和半径，也可以让程序自动从JSON文件中读取
    # 手动指定示例:
    # center = [300, 300]  # 根据实际情况修改
    # radius = 250  # 根据实际情况修改
    # result = finder.process_single_image(image_path, center, radius)
    
    # 自动读取示例:
    result = finder.process_single_image(image_path)
    
    if result:
        print(f"处理完成！轴线角度: {result['axes_angles']}")
        print(f"轴标签: {result['axis_labels']}")
    else:
        print("处理失败！")