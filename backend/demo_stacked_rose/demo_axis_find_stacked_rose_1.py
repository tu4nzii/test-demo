import numpy as np
import math
import json
import cv2
import re
import requests
import base64
import os
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from model_api_config import get_chat_completion_url, get_headers, get_model_name

class RoseChartAxisFinder:
    def __init__(self):
        # 配置参数
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        self.ocr_reader = None
        self.ocr_scale = 2.0
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("OCR识别器初始化完成")
        except Exception as e:
            print(f"OCR识别器初始化失败: {e}")

        # 统一输出路径配置
        self.output_dir = "./data/output/stacked_rose"  # 主输出目录
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

    @staticmethod
    def _angle_distance(angle1, angle2):
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    @staticmethod
    def _normalize_text(text):
        return re.sub(r'\s+', '', text or '').strip()

    def _ocr_text_candidates(self, image_path: str, center):
        if self.ocr_reader is None:
            print("OCR识别器不可用，跳过OCR轴检测")
            return []

        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return []

        scaled_image = cv2.resize(
            image,
            None,
            fx=self.ocr_scale,
            fy=self.ocr_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        scaled_center_x = float(center[0]) * self.ocr_scale
        scaled_center_y = float(center[1]) * self.ocr_scale

        results = self.ocr_reader.readtext(scaled_image)
        if not results:
            return []

        candidates = []

        for bbox, text, conf in results:
            clean_text = self._normalize_text(text)
            if conf < 0.35 or len(clean_text) < 2:
                continue

            alpha_count = sum(ch.isalpha() for ch in clean_text)
            if alpha_count / max(len(clean_text), 1) < 0.6:
                continue

            pts = np.array(bbox, dtype=float)
            bbox_center_x = float(np.mean(pts[:, 0]))
            bbox_center_y = float(np.mean(pts[:, 1]))
            distance = float(math.hypot(bbox_center_x - scaled_center_x, bbox_center_y - scaled_center_y))
            # 使用向量 (dx, dy) = (bbox_x - center_x, bbox_y - center_y)
            # 计算得到的角度基于数学惯例（从 +x 逆时针为正），但图像 y 轴向下。
            # 我们需要：正上方 = 0°, 顺时针为正方向。
            dx = bbox_center_x - scaled_center_x
            dy = bbox_center_y - scaled_center_y
            raw_deg = math.degrees(math.atan2(dy, dx))
            # 将数学角（以 +x 逆时针为正）转换为“以正上方为 0 且顺时针为正”的角度：
            # angle = (raw_deg + 90) mod 360
            angle = float((raw_deg + 90.0) % 360.0)
            width = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
            height = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))

            candidates.append({
                'text': clean_text,
                'confidence': float(conf),
                'bbox': pts.tolist(),
                'center': [bbox_center_x, bbox_center_y],
                'distance': distance,
                'angle': angle,
                'width': width,
                'height': height,
            })

        return candidates

    def _filter_axis_candidates(self, candidates):
        if not candidates:
            return []

        distances = np.array([item['distance'] for item in candidates], dtype=float)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        tolerance = max(30.0, median_distance * 0.18, mad * 2.5)

        filtered = [item for item in candidates if abs(item['distance'] - median_distance) <= tolerance]
        if len(filtered) < 2:
            filtered = sorted(candidates, key=lambda item: abs(item['distance'] - median_distance))

        return filtered

    def _merge_close_angles(self, candidates, angle_tolerance=8.0):
        merged = []
        for item in sorted(candidates, key=lambda entry: entry['angle']):
            if not merged:
                merged.append(item)
                continue

            if self._angle_distance(item['angle'], merged[-1]['angle']) <= angle_tolerance:
                if item['confidence'] > merged[-1]['confidence']:
                    merged[-1] = item
            else:
                merged.append(item)

        return merged

    def _estimate_step(self, angles):
        if len(angles) < 2:
            return 360.0

        sorted_angles = sorted(angles)
        gaps = []
        for index in range(len(sorted_angles)):
            next_index = (index + 1) % len(sorted_angles)
            gaps.append((sorted_angles[next_index] - sorted_angles[index]) % 360)

        median_gap = float(np.median(gaps))
        if median_gap < 10 or median_gap > 180:
            median_gap = 360.0 / max(len(sorted_angles), 1)
        return median_gap

    def detect_axes_by_ocr(self, image_path: str, center):
        """使用OCR识别外圈轴名称位置，并由名称中心连线推导轴角度。"""
        candidates = self._ocr_text_candidates(image_path, center)
        print(f"OCR初筛候选数量: {len(candidates)}")
        if not candidates:
            return [], {}

        filtered = self._filter_axis_candidates(candidates)
        filtered = self._merge_close_angles(filtered)
        print(f"距离筛选后候选数量: {len(filtered)}")

        if not filtered:
            return [], {}

        # 按角度顺时针（从正上方 0° 开始）排序候选项
        ordered_candidates = sorted(filtered, key=lambda item: item['angle'] % 360)
        raw_angles = [item['angle'] for item in ordered_candidates]
        step = self._estimate_step(raw_angles)
        print(f"OCR识别角度间隔估计: {step}")

        # 以正上方 0° 为起点，顺时针间隔为 step
        axes_angles = [round((0 + index * step) % 360, 2) for index in range(len(ordered_candidates))]
        axis_labels = {}
        for axis_angle, candidate in zip(axes_angles, ordered_candidates):
            axis_labels[int(round(axis_angle)) % 360] = candidate['text']

        print(f"OCR推导的轴角度: {axes_angles}")
        print(f"OCR推导的轴标签: {axis_labels}")
        return axes_angles, axis_labels

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
            "model": self.model_name,
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
            "model": self.model_name,
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
        """堆叠玫瑰图默认起始角度为90度"""
        # 修改为：默认起始角度为 0°（正上方），顺时针为正方向
        return 0

    def find_rose_axes(self, image_path: str, center, start_angle: int, max_radius):
        """识别堆叠玫瑰图的轴线角度"""
        axes_angles, axis_labels = self.detect_axes_by_ocr(image_path, center)
        self.axes_angles = axes_angles
        self.axis_labels = axis_labels
        return axes_angles

    def recognize_axis_labels(self, image_path: str, center, radius, axes_angles):
        """返回OCR识别得到的标签映射"""
        if self.axis_labels:
            return self.axis_labels

        _, axis_labels = self.detect_axes_by_ocr(image_path, center)
        self.axis_labels = axis_labels
        return axis_labels

    def process_single_image(self, image_path, center=None, radius=None, output_json_path=None):
        """处理单张玫瑰图，识别轴线和标签"""
        print(f"正在处理图像: {image_path}")
        try:
            input_json_path = None
            # 如果未指定圆心和半径，尝试从JSON文件中读取
            if center is None or radius is None:
                # 优先读取显式传入的JSON文件，其次再回退到输出目录同名文件
                if isinstance(output_json_path, str):
                    if os.path.exists(output_json_path):
                        input_json_path = output_json_path
                        print(f"从指定的JSON文件路径读取: {input_json_path}")
                    else:
                        # 宽容解析：尝试在 self.output_dir 中查找同名文件
                        candidate = os.path.join(self.output_dir, os.path.basename(output_json_path))
                        if os.path.exists(candidate):
                            input_json_path = candidate
                            print(f"指定路径不存在，改为从输出目录读取: {input_json_path}")
                        else:
                            print(f"指定的JSON路径不存在: {output_json_path}")
                else:
                    base_name = os.path.basename(image_path)
                    file_name, _ = os.path.splitext(base_name)
                    print(f"未指定JSON文件路径，尝试从输出目录同名文件读取: {file_name}.json")
                    input_json_path = os.path.join(self.output_dir, f"{file_name}.json")

                if input_json_path and os.path.exists(input_json_path):
                    with open(input_json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        # print(json_data)
                    if center is None and 'pred_coords' in json_data:
                        center = json_data["pred_coords"]
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

            if output_json_path is None or (isinstance(output_json_path, str) and os.path.exists(output_json_path) and output_json_path == input_json_path):
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
    image_path = r"backend\charts\stacked_rose\stacked_rose_003.png"  # 根据需要修改
    json_path = r"data\output\stacked_rose\stacked_rose_003.json"  # 根据需要修改
    # 可以手动指定圆心和半径，也可以让程序自动从JSON文件中读取
    # 手动指定示例:
    # center = [300, 300]  # 根据实际情况修改
    # radius = 250  # 根据实际情况修改
    # result = finder.process_single_image(image_path, center, radius)

    # 自动读取示例:
    result = finder.process_single_image(image_path, output_json_path=json_path)

    if result:
        print(f"处理完成！轴线角度: {result['axes_angles']}")
        print(f"轴标签: {result['axis_labels']}")
    else:
        print("处理失败！")