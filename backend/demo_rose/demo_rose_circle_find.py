import cv2
import base64
import numpy as np
import json 
import requests
import os
import math
from PIL import Image, ImageDraw, ImageFont
import re

class RoseChartEncoder:
    def __init__(self):
        # 配置参数
        self.api_key = "sk-wI6yoFNGxIi8kFHuE68882A8Ed06427aAaA3548662439c8d"
        self.url = "https://api.vveai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.tick_density = 2
        
        # 处理结果存储
        self.result_image = None
        self.r_ticks = []
        self.argument = {}
        self.coords = [0, 0]  # [cx, cy]
        self.first_r = 0
        self.second_r = 0

    def show_image_with_scaling(self, window_name, image, max_width=800, max_height=600):
        """显示缩放后的图像"""
        height, width = image.shape[:2]
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)  # 不放大图像
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        cv2.imshow(window_name, resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    def visualize_ring_mask(self, image_path, ring_width=5):
        """创建环状掩码并返回处理后的图像"""
        # 读取并预处理图像
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 霍夫变换检测第一个圆
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                 param1=20, param2=30, minRadius=int(height/4), maxRadius=int(height))
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            first_circle = circles[0, 0]
            cx, cy, r = first_circle[0], first_circle[1], first_circle[2]
            
            # 创建环状掩膜
            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), int(r+ring_width), 255, -1)
            cv2.circle(mask, (cx, cy), int(r-ring_width), 0, -1)
            
            # 应用环状掩膜
            masked_blurred = image.copy()
            masked_blurred[mask == 255] = 255
            
            self.coords = [cx, cy]
            self.first_r = r
            return masked_blurred
        else:
            return image

    def second_circle_find(self, image):
        """检测第二个圆"""
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                 param1=20, param2=50, minRadius=self.first_r+30, maxRadius=int(height / 2))
        
        second_r = self.first_r + 50
        if circles is not None:
            circles = np.uint16(np.around(circles))
            second_circle = circles[0, 0]
            second_r = second_circle[2]
            
        self.second_r = second_r
        return second_r

    def crop_tick_region(self, image, target_radius, pixel_range=25):
        """裁剪指定半径附近的环形区域"""
        center_x, center_y = self.coords
        
        # 创建与原图相同大小的掩码
        mask = np.zeros_like(image)
        
        # 计算内外圆半径（确保内圆半径不为负）
        outer_radius = target_radius + pixel_range
        inner_radius = max(0, target_radius - 10)
        
        # 绘制环形掩码（白色为保留区域）
        cv2.circle(mask, (center_x, center_y), outer_radius, (255, 255, 255), -1)
        cv2.circle(mask, (center_x, center_y), inner_radius, (0, 0, 0), -1)
        
        # 应用掩码获取环形区域
        masked_image = cv2.bitwise_and(image, mask)
        
        # 裁剪最小外接矩形以去除多余黑色区域
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 获取最大轮廓的边界框
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return masked_image[y:y+h, x:x+w]
        
        # 如果没有找到有效区域，返回原始掩码图像
        return masked_image

    def find_tick(self, target_radius, image_path):
        """使用LLM识别指定半径处的刻度值"""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        cropped_image = self.crop_tick_region(image, target_radius, pixel_range=25)
        
        # 转换为RGB并编码为base64
        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            
        retval, buffer = cv2.imencode('.png', rgb_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""
        这是一张圆环图
        图片中已经用绿色圆圈标出了一个重要的圆
        请您分析图片内容，并提供该信息：
        1. 这个**绿色圆圈**对应的的刻度值是多少？（会出现在该圆周围，该刻度值仅为一个数值(如50% = 50)，且仅出现在图片中）
        **只读取存在的数**
        **注意,仅读取图上原本的数值，而不做任何推算**
        **仔细检查图片，确保读取的数值是正确的**
        **不存在300！！！若识别为300，则是识别错误，实际为200**

        请以严格的 JSON 格式返回这些信息，不要包含任何额外文字或解释，例如：
        ```json
        {{
            "tick": <刻度值>,
            "res":<分析过程>
        }}
        ```
        **如果有多个数字，使用null**
        如果无法识别某个值，请使用 `null`。
        **再次声明，如果没有数字，则使用null**
        **再次声明，如果有多个数字，使用null**
        **若包含字母，则为null**
        **若为0则为null**
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
            print(f"API请求失败: {e}")
            return None

    def call_llm_response(self, image_path):
        """调用LLM获取图表的刻度信息"""
        center_x, center_y = self.coords
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        prompt = f"""
        这是一张极坐标图（例如雷达图或极坐标散点图）。
        其中心大致在 ({center_x}, {center_y}) 像素位置。


        请您分析图片内容，并提供以下信息:
        1. 图表中所有**同心圆刻度线中，即最外圈的刻度，最大的刻度值是多少？**
        2. 图表中所有**同心圆刻度线中，即最外圈的刻度，最小的刻度值是多少？**
        3. 图表中所有**同心圆刻度的间隔，刻度为径向刻度，而非环状刻度**

        **注意,仅读取图上原本的数值，而不做任何推算**
        且仅返回数值，如（50% = 50）

        请以严格的 JSON 格式返回这些信息，不要包含任何额外文字或解释，例如：
        ```json
        {{
            "max_tick_value": <最大刻度值>
            "min_tick_value": <最小刻度值>
            "tick_interval": <刻度间隔>
            "res":<分析过程>
        }}
        该图的最大刻度为100，最小刻度为50 间隔为50
        ```
        如果无法识别某个值，请使用 `null`。例如，如果 `max_tick_value` 是0，请返回`0`。
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

    def encrypt_rose_chart_with_tick(self, image_path, tick_interval, tick1, tick2, max_tick_value, min_tick_value):
        """使用两个刻度值进行网格加密"""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # 确保r2 > r1
        r1, r2 = (self.first_r, self.second_r) if self.second_r > self.first_r else (self.second_r, self.first_r)
        interval = tick_interval / self.tick_density
        
        # 计算半径与刻度的线性关系 (r = a*tick + b)
        a = r1 / max_tick_value
        b = r1 - a * max_tick_value
        
        result = image.copy()
        
        # 设置字体和颜色
        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        image_area = height * width
        scale = math.sqrt(image_area)
        font_scale = scale * 0.006
        font_color = (0, 0, 0)  # 黑色
        line_color = (128, 128, 128)
        thickness = 1
        
        # 绘制加密圆环并标注刻度
        tick = max_tick_value
        r_ticks = []
        self.argument = {'a': a, 'b': b}
        
        current_px_distance = 10000
        radius = 0
        count = 0
        
        while tick > 0 and radius >= 0:
            tick -= interval
            if tick < 0:
                break
                
            # 计算当前刻度对应的半径
            radius = int(a * tick + b)
            current_px_distance = abs(radius - 0)
            
            if radius <= 0:
                print(f"⚠️ 跳过无效半径: tick={tick}, 计算半径={radius}")
                continue
                
            if current_px_distance <= 3:
                print(f"已达到圆心附近，停止绘制 (tick={tick}, radius={radius})")
                break
                
            text_x_up = center_x 
            text_y_up = center_y - radius 

            # 绘制刻度标注
            if tick % 1 == 0:
                tick = int(tick)
                
            if tick > 0:
                pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 设置字体
                try:
                    font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)
                # 获取文本边界框
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 右侧旋转90度文本
                temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((0, 0), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)  # 顺时针旋转90度
                
                # 调整右侧位置
                pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                
                # 左侧旋转-90度文本
                temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((0, 0), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)  # 逆时针旋转90度
                
                # 调整左侧位置
                pos_left = (center_x - radius - int(width*0.0125), 
                        center_y - rotated_left.size[1]//2)
                
                # 底部正常文本
                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius - int(height*0.0122) 
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                
                # 转换回OpenCV格式
                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            # 保存生成的刻度值
            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1, lineType=cv2.LINE_AA)
            
            # 绘制虚线圆环
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
                cv2.line(result, (x1, y1), (x2, y2), line_color, thickness, lineType=cv2.LINE_AA)
        
        self.result_image = result
        self.r_ticks = r_ticks
        return result, r_ticks, self.argument

    def encrypt_rose_chart_one_tick(self, image_path, tick_interval, tick1, r, max_tick_value, min_tick_value):
        """使用单个刻度值进行网格加密"""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        interval = tick_interval / self.tick_density
        
        # 计算半径与刻度的线性关系
        if min_tick_value > 0:
            pixels_per_value = float(r) / (tick1 - (min_tick_value - tick_interval))
        else:
            pixels_per_value = float(r) / tick1
            
        a = pixels_per_value 
        b = r - a * tick1
        result = image.copy()
        
        # 设置字体和颜色
        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        image_area = height * width
        scale = math.sqrt(image_area)
        font_scale = scale * 0.006
        font_color = (0, 0, 0)  # 黑色
        line_color = (128, 128, 128)
        thickness = 1
        
        # 绘制加密圆环并标注刻度
        tick = max_tick_value
        r_ticks = []
        self.argument = {'a': a, 'b': b}
        
        current_px_distance = 10000
        radius = 0
        count = 0
        
        while tick > 0 and radius >= 0:
            tick -= interval
            # 计算当前刻度对应的半径
            radius = int(a * tick + b)
            current_px_distance = abs(radius - 0)
            
            if radius <= 0:
                print(f"⚠️ 跳过无效半径: tick={tick}, 计算半径={radius}")
                continue
                
            if current_px_distance <= 3:
                print(f"已达到圆心附近，停止绘制 (tick={tick}, radius={radius})")
                break
                
            text_x_up = center_x 
            text_y_up = center_y - radius 

            # 绘制刻度标注
            if tick % 1 == 0:
                tick = int(tick)
                
            if tick > 0:
                pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 设置字体
                try:
                    font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)
                # 获取文本边界框
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 右侧旋转90度文本
                temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((0, 0), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)  # 顺时针旋转90度
                
                # 调整右侧位置
                pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                
                # 左侧旋转-90度文本
                temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((0, 0), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)  # 逆时针旋转90度
                
                # 调整左侧位置
                pos_left = (center_x - radius - int(width*0.0125), 
                        center_y - rotated_left.size[1]//2)
                
                # 底部正常文本
                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius - int(height*0.0122) 
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                
                # 转换回OpenCV格式
                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            # 保存生成的刻度值
            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1, lineType=cv2.LINE_AA)
            
            # 绘制虚线圆环
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
                cv2.line(result, (x1, y1), (x2, y2), line_color, thickness, lineType=cv2.LINE_AA)
        
        self.result_image = result
        self.r_ticks = r_ticks
        return result, r_ticks, self.argument

    def process_single_image(self, image_path, output_dir=None):
        """处理单张雷达图，执行完整的加密流程"""
        try:
            # 如果未指定输出目录，使用当前目录
            if output_dir is None:
                output_dir = os.path.dirname(image_path) or '.'
                
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取文件名和扩展名
            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            
            # 找第二个圆
            second_circle = self.visualize_ring_mask(image_path)
            self.second_circle_find(second_circle)
            
            print(f"检测到的圆心坐标: {self.coords}")
            print(f"第一个圆半径: {self.first_r}")
            print(f"第二个圆半径: {self.second_r}")
            
            # 创建临时输出路径
            temp_output_path = os.path.join(output_dir, f"temp_marked_{file_name}{file_ext}")
            
            # 画出两个圆并保存临时图像
            image = cv2.imread(image_path)
            cv2.circle(image, (self.coords[0], self.coords[1]), self.first_r, (0, 255, 0), 1)  # 绿色外圆
            cv2.circle(image, (self.coords[0], self.coords[1]), self.second_r, (0, 255, 0), 1)  # 绿色内圆
            cv2.circle(image, (self.coords[0], self.coords[1]), 2, (255, 0, 0), -1)  # 蓝色实心中心点
            
            cv2.imwrite(temp_output_path, image)
            print(f"临时标记图像已保存至: {temp_output_path}")
            
            # 识别刻度
            result_1 = self.find_tick(self.first_r, temp_output_path)
            tick1 = result_1.get("tick")
            reason1 = result_1.get("res")
            
            result_2 = self.find_tick(self.second_r, temp_output_path)
            tick2 = result_2.get("tick")
            reason2 = result_2.get("res")
            
            print(f"第一个圆刻度识别: {reason1}")
            print(f"第一个圆刻度值: {tick1}")
            print(f"第二个圆刻度识别: {reason2}")
            print(f"第二个圆刻度值: {tick2}")
            
            # 调用LLM获取刻度信息
            response_data = self.call_llm_response(temp_output_path)
            max_tick_value = response_data.get("max_tick_value")
            min_tick_value = response_data.get("min_tick_value")
            tick_interval = response_data.get("tick_interval")
            res = response_data.get("res")
            
            print(f"LLM分析结果: {tick1}, {tick2}, {max_tick_value}, {res}")
            
            # 加密处理
            if tick1 and tick2:
                result, r_ticks, argument = self.encrypt_rose_chart_with_tick(
                    image_path, tick_interval, tick1, tick2, max_tick_value, min_tick_value
                )
            elif tick1 and tick2 is None:
                result, r_ticks, argument = self.encrypt_rose_chart_one_tick(
                    image_path, tick_interval, tick1, self.first_r, max_tick_value, min_tick_value
                )
            elif tick1 is None and tick2:
                result, r_ticks, argument = self.encrypt_rose_chart_one_tick(
                    image_path, tick_interval, tick2, self.second_r, max_tick_value, min_tick_value
                )
            else:
                print("未识别出正确的刻度")
                return None
            
            # 保存最终结果
            output_path = os.path.join(output_dir, f"{file_name}_encode{file_ext}")
            cv2.imwrite(output_path, result)
            print(f"加密后的图像已保存至: {output_path}")
            
            # 处理JSON数据（如果存在）
            json_fname = f"{file_name}.json"
            json_path = os.path.join(os.path.dirname(image_path), json_fname)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    
                # 添加r_ticks字段
                r_ticks = r_ticks[::-1]
                r_ticks.append(r_ticks[-1] + (tick_interval / self.tick_density))
                json_data['r_ticks'] = r_ticks
                
                # 添加预测圆心
                pred_coords = [int(self.coords[0]), int(self.coords[1])]
                
                # 尝试获取真实圆心
                try:
                    if 'center' in json_data:
                        if isinstance(json_data['center'], dict):
                            real_coords = [json_data['center']['x'], json_data['center']['y']]
                        else:
                            real_coords = json_data['center']
                        
                        # 计算圆心误差
                        err_center = np.linalg.norm(np.array(pred_coords) - np.array(real_coords))
                        json_data['err_center'] = err_center
                except Exception as e:
                    print(f"计算圆心误差时出错: {e}")
                    
                json_data['pred_coords'] = pred_coords
                json_data['argument'] = argument
                
                # 保存更新后的JSON
                output_json_path = os.path.join(output_dir, json_fname)
                with open(output_json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
                print(f"更新后的JSON已保存至: {output_json_path}")
            else:
                print(f"JSON文件未找到: {json_path}")
                
            # 可选：删除临时文件
            # os.remove(temp_output_path)
            
            return output_path
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            return None


if __name__ == "__main__":
    # 示例用法
    encoder = RoseChartEncoder()
    
    # 指定要处理的图像路径和输出目录
    image_path = "./data/rose/rose_001.png"  # 可以根据需要修改
    output_dir = "./data/output/rose"      # 可以根据需要修改
    
    # 处理单张图像
    result_path = encoder.process_single_image(image_path, output_dir)
    
    if result_path:
        print(f"处理完成！加密后的图像保存在: {result_path}")
    else:
        print("处理失败！")