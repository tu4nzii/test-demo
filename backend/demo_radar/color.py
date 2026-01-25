import numpy as np
import json
import cv2
import re
import requests
import base64
import os

class RadarColorMatcher:
    """雷达图实体颜色匹配器"""
    def __init__(self):
        # API配置
        self.api_key = "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
        self.url = "https://api.vveai.com/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 输出配置
        self.output_dir = "./data/output/radar"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 结果存储
        self.entity_colors = {}
        
        # 颜色识别参数
        self.min_block_area = 30         # 最小颜色块面积
        self.max_block_area = 1000       # 最大颜色块面积
        self.color_diff_threshold = 30   # 颜色差异阈值
        self.min_saturation = 30         # 最小饱和度
        self.min_value = 50              # 最小亮度

    def parse_json(self, content: str):
        """从文本中解析JSON内容"""
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            return json.loads(match.group(1))
        except Exception as e:
            print(f"❌ JSON解析失败: {e}")
            return None

    def load_image(self, image_path):
        """加载图像文件"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        return image

    def crop_legend(self, image, ratio=0.3, scale=2):
        """裁剪图像左上角的图例区域并放大
        
        Args:
            image: OpenCV图像对象
            ratio: 裁剪比例（左上角区域）
            scale: 放大倍数
            
        Returns:
            裁剪并放大后的图像
        """
        height, width = image.shape[:2]
        # 裁剪左上角区域
        crop_region = image[:int(height*ratio), :int(width*ratio)]
        
        # 放大图像
        new_height = int(crop_region.shape[0] * scale)
        new_width = int(crop_region.shape[1] * scale)
        return cv2.resize(crop_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def image_to_base64(self, image):
        """将OpenCV图像转换为base64编码"""
        # 转换为RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 确保数据类型正确
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
            print("已转换图像数据类型为uint8")
            
        # 编码为JPEG并转换为base64
        success, encoded = cv2.imencode('.jpg', rgb_image)
        if not success:
            print("图像编码失败")
            return None
            
        return base64.b64encode(np.ascontiguousarray(encoded)).decode('utf-8')

    def detect_legend(self, base64_image):
        """使用大模型检测图例位置
        
        Args:
            base64_image: 完整雷达图的base64编码
            
        Returns:
            包含中心点坐标和范围的字典，失败则返回None
        """
        prompt = """
        请分析这张雷达图，识别图例区域的位置和范围。
        一定要包含整个图例区域，不能只包含部分。
        图例区域通常包含雷达图中各个实体的名称及其对应的颜色标记。
        
        请以JSON格式返回：
        - position: 图例区域中心点的坐标[x, y]
        - range: 图例区域的宽度和高度[w, h]
        
        ```json
        {
            "position": [x, y],
            "range": [w, h]
        }
        ```
        
        请确保返回合理的数值，无法识别时返回null。
        """
        
        payload = {
            "model": "gemini-2.0-flash",
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
        
        try:
            response = requests.post(url=self.url, headers=self.headers, json=payload)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            data = self.parse_json(content)
            if data and "position" in data and "range" in data:
                return data
            else:
                print(f"无法提取有效的图例位置信息: {content}")
                return None
                
        except Exception as e:
            print(f"API请求错误: {e}")
            if 'response' in locals():
                print(f"响应内容: {response.text}")
            return None
    
    def auto_crop_legend(self, image, scale=2):
        """智能裁剪图例区域
        
        使用大模型检测图例位置，精确裁剪并放大
        
        Args:
            image: OpenCV图像对象
            scale: 放大倍数
            
        Returns:
            裁剪并放大后的图例区域
        """
        # 转换为base64
        base64_image = self.image_to_base64(image)
        if base64_image is None:
            print("警告：图像转换失败，使用默认裁剪")
            return self.crop_legend(image, scale=scale)
        
        # 检测图例位置
        print("正在检测图例位置...")
        legend_info = self.detect_legend(base64_image)
        
        # 图像尺寸
        height, width = image.shape[:2]
        
        # 验证图例信息
        if legend_info is None or not self._validate_legend_info(legend_info, width, height):
            print("警告：图例检测失败，使用默认裁剪")
            return self.crop_legend(image, scale=scale)
        
        # 计算裁剪坐标
        center_x, center_y = legend_info["position"]
        region_width, region_height = legend_info["range"]
        
        # 添加边距并确保在图像范围内
        margin = int(min(region_width, region_height) * 0.1)
        x1 = max(0, int(center_x - region_width / 2) - margin)
        y1 = max(0, int(center_y - region_height / 2) - margin)
        x2 = min(width, int(center_x + region_width / 2) + margin)
        y2 = min(height, int(center_y + region_height / 2) + margin)
        
        print(f"图例位置: ({x1}, {y1}) 到 ({x2}, {y2})")
        
        # 裁剪图例区域
        crop_region = image[y1:y2, x1:x2]
        
        # 检查裁剪区域是否过小
        if crop_region.size == 0 or crop_region.shape[0] < 50 or crop_region.shape[1] < 50:
            print("警告：图例区域过小，使用默认裁剪")
            return self.crop_legend(image, scale=scale)
        
        # 放大图像
        new_height = int(crop_region.shape[0] * scale)
        new_width = int(crop_region.shape[1] * scale)
        return cv2.resize(crop_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def _validate_legend_info(self, legend_info, width, height):
        """验证图例信息是否有效"""
        try:
            # 检查position
            position = legend_info.get("position", [])
            if not isinstance(position, list) or len(position) != 2:
                return False
            
            x, y = position
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            
            # 检查range
            region_range = legend_info.get("range", [])
            if not isinstance(region_range, list) or len(region_range) != 2:
                return False
            
            w, h = region_range
            if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
                return False
            
            # 检查合理性
            if w <= 0 or h <= 0 or w > width or h > height:
                return False
            
            if x < 0 or x > width or y < 0 or y > height:
                return False
                
            return True
        except:
            return False
    
    def extract_colors(self, image):
        """从图例图像中提取唯一颜色及其位置信息
        
        Args:
            image: OpenCV图像对象
            
        Returns:
            颜色块信息列表，每个元素包含颜色和位置信息
        """
        # 转换到HSV并创建颜色掩码
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, self.min_saturation, self.min_value])
        upper_bound = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 形态学操作优化掩码
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
        color_mask = cv2.dilate(color_mask, kernel, iterations=2)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
        
        # 查找轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 提取颜色块和位置信息
        block_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 过滤面积和形状
            if (self.min_block_area < area < self.max_block_area and 
                0.3 < aspect_ratio < 3.0):
                # 计算平均颜色
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_color = cv2.mean(image, mask=mask)[:3]  # BGR
                
                # 检查饱和度
                bgr_color = np.uint8([[mean_color]])
                hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
                if hsv_color[1] >= self.min_saturation:
                    # 计算中心点坐标
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 存储颜色块信息（包含位置信息）
                    block_info.append({
                        'color': np.uint8(mean_color),
                        'position': {
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'center_x': center_x,
                            'center_y': center_y
                        },
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        # 颜色去重（保留位置信息）
        unique_color_info = []
        for block in block_info:
            is_unique = True
            current_color = block['color']
            
            for unique_block in unique_color_info:
                unique_color = unique_block['color']
                # 计算HSV颜色差异
                hsv1 = cv2.cvtColor(np.uint8([[current_color]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv2 = cv2.cvtColor(np.uint8([[unique_color]]), cv2.COLOR_BGR2HSV)[0][0]
                
                # 计算色调差异（考虑环形特性）
                h_diff = min(abs(int(hsv1[0]) - int(hsv2[0])), 180 - abs(int(hsv1[0]) - int(hsv2[0])))
                s_diff = abs(int(hsv1[1]) - int(hsv2[1]))
                v_diff = abs(int(hsv1[2]) - int(hsv2[2]))
                
                # 加权距离
                weighted_distance = int(h_diff) * 2 + int(s_diff) + int(v_diff)
                
                if weighted_distance <= self.color_diff_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_color_info.append(block)
        
        # 按色调排序
        if unique_color_info:
            hsv_color_info = []
            for block in unique_color_info:
                hsv = cv2.cvtColor(np.uint8([[block['color']]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv_color_info.append((hsv[0], block))
            
            hsv_color_info.sort(key=lambda x: x[0])
            unique_color_info = [block for _, block in hsv_color_info]
        
        return unique_color_info
    
    def bgr_to_hex(self, bgr_color):
        """将BGR颜色转换为十六进制格式
        
        Args:
            bgr_color: BGR格式的颜色值
            
        Returns:
            十六进制颜色字符串
        """
        rgb_color = bgr_color[::-1]  # BGR -> RGB
        return f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}".upper()
    
    def match_entities_colors(self, base64_image, entity_names, color_info_list):
        """使用大模型匹配实体和颜色，提供颜色位置信息辅助大模型匹配
        
        Args:
            base64_image: 图例图像的base64编码
            entity_names: 实体名称列表
            color_info_list: 提取到的颜色块信息列表，包含颜色和位置信息
            
        Returns:
            实体和颜色的简单映射字典 {实体: 颜色}
        """
        # 准备颜色信息（包含十六进制和位置）
        color_with_positions = []
        hex_colors = []
        
        for i, color_info in enumerate(color_info_list):
            hex_color = self.bgr_to_hex(color_info['color'])
            hex_colors.append(hex_color)
            pos = color_info['position']
            color_with_positions.append(
                f"颜色{i+1}: {hex_color} (左上角: {pos['x']}, {pos['y']}, 中心点: {pos['center_x']}, {pos['center_y']})"
            )
        
        prompt = f"""
        请分析这张雷达图图例，并将提供的实体名称与提取到的颜色进行一一对应。
        
        实体名称列表：
        {', '.join(entity_names)}
        
        提取到的颜色列表（包含位置信息，用于辅助匹配）：
        {'; '.join(color_with_positions)}
        
        请以JSON格式返回实体和颜色的对应关系（仅包含实体名称和十六进制颜色值）：
        
        ```json
        {{
            "实体1": "颜色1",
            "实体2": "颜色2",
            ...
        }}
        ```
        
        请确保每个实体都有对应的颜色。
        """
        
        payload = {
            "model": "gemini-2.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.3
        }
        
        try:
            response = requests.post(url=self.url, headers=self.headers, json=payload)
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            return self.parse_json(content)
                
        except Exception as e:
            print(f"API请求错误: {e}")
            if 'response' in locals():
                print(f"响应内容: {response.text}")
            return None
    
    def process_image(self, image_path, use_auto_crop=True, entity_names=None):
        """处理雷达图，识别实体和颜色
        
        Args:
            image_path: 图像路径
            use_auto_crop: 是否使用智能裁剪
            entity_names: 已知的实体名称列表
            
        Returns:
            包含识别结果的字典，失败则返回None
        """
        try:
            # 读取图像
            print(f"正在读取图像: {image_path}")
            image = self.load_image(image_path)
            
            # 裁剪图例
            print("正在裁剪图例区域...")
            legend_image = self.auto_crop_legend(image) if use_auto_crop else self.crop_legend(image)
            
            # 保存裁剪图像
            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            legend_path = os.path.join(self.output_dir, f"legend_{file_name}{file_ext}")
            print(f"正在保存图例图像: {legend_path}")
            cv2.imwrite(legend_path, legend_image)
            
            # 转换为base64
            print("正在转换图像格式...")
            base64_image = self.image_to_base64(legend_image)
            if base64_image is None:
                raise ValueError("图像转换失败")
            
            # 提取颜色和位置信息
            print("正在提取颜色块...")
            color_info_list = self.extract_colors(legend_image)
            hex_colors = [self.bgr_to_hex(info['color']) for info in color_info_list]
            print(f"成功提取到 {len(hex_colors)} 个颜色: {', '.join(hex_colors)}")
            
            # 验证实体名称
            if entity_names is None:
                print("错误：请提供实体名称列表")
                return None
            else:
                print(f"使用实体名称: {', '.join(entity_names)}")
            
            # 匹配实体和颜色
            print("正在匹配实体和颜色...")
            entity_colors = self.match_entities_colors(base64_image, entity_names, color_info_list)
            
            # 备选方案：简单一一对应
            if entity_colors is None:
                print("匹配失败，使用默认对应")
                entity_colors = {}
                for i, entity in enumerate(entity_names):
                    if i < len(color_info_list):
                        color_info = color_info_list[i]
                        entity_colors[entity] = self.bgr_to_hex(color_info['color'])
                    else:
                        entity_colors[entity] = "#000000"  # 默认为黑色
            
            self.entity_colors = entity_colors
            
            # 准备结果
            result = {
                'image_path': image_path,
                'entity_colors': entity_colors,
                'extracted_colors': hex_colors,
                'legend_path': legend_path,
                'crop_method': 'auto' if use_auto_crop else 'default'
            }
            
            # 保存结果
            output_path = os.path.join(self.output_dir, f"{file_name}_colors.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"结果已保存至: {output_path}")
            print("\n识别结果：")
            print(json.dumps(entity_colors, ensure_ascii=False, indent=2))
            
            return result
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    # 创建匹配器实例
    matcher = RadarColorMatcher()
    
    # 配置参数
    # matcher.output_dir = "custom_output"  # 自定义输出目录
    
    # 图像路径
    image_path = r"d:/home work/Agent.paper/test demo/backend/data/upload/radar_001.png"
    
    # 实体名称
    # entity_names = ["WDULR", "ZTJUP", "QCBOR", "RFLDM", "UCKIV"]
    entity_names =["LMIEXG","KBGCVO","AZC","OAAKCP"]
    # 处理图像
    result = matcher.process_image(image_path, entity_names=entity_names)
    
    if result:
        print(f"\n处理完成！成功识别到 {len(result['entity_colors'])} 个实体")
        print("\n实体-颜色对应关系：")
        for entity, color in result['entity_colors'].items():
            print(f"{entity}: {color}")
    else:
        print("处理失败！")


if __name__ == "__main__":
    main()