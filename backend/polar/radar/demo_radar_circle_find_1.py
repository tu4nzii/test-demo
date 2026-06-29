import cv2
import base64
import numpy as np
import json 
import requests
import os
import math
from PIL import Image, ImageDraw, ImageFont
import re
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
from model_api_config import get_chat_completion_url, get_headers, get_model_name

class RadarChartEncoder:
    def __init__(self):
        # 配置参数
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()

        self.tick_density = 2
        
        # 处理结果存储
        self.result_image = None
        self.r_ticks = []
        self.argument = {}
        self.coords = [0, 0]  # [cx, cy]
        self.first_r = 0
        self.second_r = 0
        self.detection_size = 800
        self.detection_source = None
        self.detection_transform = None
        self.detection_image = None

        # ── 兜底机制: 质量评估与 fallback 标志 ──
        self.fallback_flag = False
        self.fallback_reason = ""
        self.last_edge_support = 0.0
        self.last_concentric_score = 0.0

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

    def get_annotation_metrics(self, image_shape, radial_spacing):
        """Scale generated tick labels to the detected chart area and ring spacing."""
        height, width = image_shape[:2]
        if self.detection_transform is not None:
            x0, y0, x1, y1 = self.detection_transform["roi"]
            chart_scale = math.sqrt((x1 - x0) * (y1 - y0))
        else:
            chart_scale = min(height, width)

        roi_font_size = max(7, int(round(chart_scale * 0.018)))
        spacing_font_limit = max(7, int(round(abs(radial_spacing) * 0.55)))
        font_size = min(roi_font_size, spacing_font_limit)
        label_offset = max(2, int(round(font_size * 0.25)))
        cv_font_scale = max(0.25, font_size / 30.0)
        return font_size, label_offset, cv_font_scale

    def prepare_detection_image(self, image):
        """Build a conservative square ROI and letterbox it for circle detection."""
        height, width = image.shape[:2]
        background_center = self.find_background_disk_center(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        foreground = np.uint8(gray < 250) * 255

        kernel_size = max(11, int(round(min(height, width) * 0.04)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        merged = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), np.uint8),
        )

        component_count, _, stats, _ = cv2.connectedComponentsWithStats(merged)
        if component_count <= 1:
            return None
        component_indices = [
            index for index in range(1, component_count)
            if not (
                stats[index, cv2.CC_STAT_WIDTH] >= width * 0.95
                and stats[index, cv2.CC_STAT_HEIGHT] >= height * 0.95
            )
        ]
        if not component_indices:
            component_indices = list(range(1, component_count))
        component_index = max(
            component_indices,
            key=lambda index: stats[index, cv2.CC_STAT_AREA],
        )
        x, y, component_width, component_height, _ = [
            int(value) for value in stats[component_index]
        ]

        side = int(round(max(component_width, component_height) * 1.5))
        side = min(max(side, component_width, component_height), width, height)
        center_x = x + component_width / 2.0
        center_y = y + component_height / 2.0
        x0 = max(0, min(int(round(center_x - side / 2.0)), width - side))
        y0 = max(0, min(int(round(center_y - side / 2.0)), height - side))
        x1 = x0 + side
        y1 = y0 + side

        # Never use a crop that clips the selected main content component.
        if x0 > x or y0 > y or x1 < x + component_width or y1 < y + component_height:
            return None

        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        scale = min(self.detection_size / roi.shape[1], self.detection_size / roi.shape[0])
        resized_width = max(1, int(round(roi.shape[1] * scale)))
        resized_height = max(1, int(round(roi.shape[0] * scale)))
        resized = cv2.resize(roi, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        pad_x = (self.detection_size - resized_width) // 2
        pad_y = (self.detection_size - resized_height) // 2
        normalized = np.full((self.detection_size, self.detection_size, 3), 255, dtype=np.uint8)
        normalized[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        normalized_content_center = (
            (center_x - x0) * scale + pad_x,
            (center_y - y0) * scale + pad_y,
        )
        if background_center is not None:
            normalized_content_center = (
                (background_center[0] - x0) * scale + pad_x,
                (background_center[1] - y0) * scale + pad_y,
            )
        return {
            "image": normalized,
            "roi": (x0, y0, x1, y1),
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "content_center": normalized_content_center,
            "reliable_center": background_center is not None,
        }

    def find_background_disk_center(self, image):
        """Find the center of a large, uniformly colored neutral background disk."""
        height, width = image.shape[:2]
        neutral = (
            (image[:, :, 0] == image[:, :, 1])
            & (image[:, :, 1] == image[:, :, 2])
            & (image[:, :, 0] >= 180)
            & (image[:, :, 0] < 250)
        )
        values, counts = np.unique(image[:, :, 0][neutral], return_counts=True)
        if len(counts) == 0:
            return None
        dominant_index = int(np.argmax(counts))
        dominant_value = int(values[dominant_index])
        if counts[dominant_index] < height * width * 0.05:
            return None

        mask = np.uint8(
            (image[:, :, 0] == dominant_value)
            & (image[:, :, 1] == dominant_value)
            & (image[:, :, 2] == dominant_value)
        ) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if component_count <= 1:
            return None
        component_index = max(
            range(1, component_count),
            key=lambda index: stats[index, cv2.CC_STAT_AREA],
        )
        x, y, component_width, component_height, area = [
            int(value) for value in stats[component_index]
        ]
        if area < height * width * 0.05:
            return None
        aspect_ratio = component_width / max(1, component_height)
        if not 0.9 <= aspect_ratio <= 1.1:
            return None
        return (
            x + component_width / 2.0,
            y + component_height / 2.0,
        )

    def normalize_with_transform(self, image, transform):
        """Apply the first detection's ROI transform to a modified source image."""
        x0, y0, x1, y1 = transform["roi"]
        roi = image[y0:y1, x0:x1]
        resized_width = max(1, int(round(roi.shape[1] * transform["scale"])))
        resized_height = max(1, int(round(roi.shape[0] * transform["scale"])))
        resized = cv2.resize(roi, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        normalized = np.full((self.detection_size, self.detection_size, 3), 255, dtype=np.uint8)
        pad_x = transform["pad_x"]
        pad_y = transform["pad_y"]
        normalized[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        return normalized

    def map_circle_to_original(self, circle, transform):
        """Map a normalized-image circle back to original image coordinates."""
        cx, cy, radius = [float(value) for value in circle]
        x0, y0, _, _ = transform["roi"]
        scale = transform["scale"]
        return (
            int(round((cx - transform["pad_x"]) / scale + x0)),
            int(round((cy - transform["pad_y"]) / scale + y0)),
            int(round(radius / scale)),
        )

    def circle_is_safe(self, circle, image_shape, margin=3):
        """Check that the full circle remains inside an image with a small margin."""
        height, width = image_shape[:2]
        cx, cy, radius = circle
        return (
            radius > 0
            and cx - radius >= margin
            and cy - radius >= margin
            and cx + radius < width - margin
            and cy + radius < height - margin
        )

    def enhance_detection_gray(self, image):
        """Increase faint grid contrast without modifying the source image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.GaussianBlur(enhanced, (5, 5), 1.2)

    def circle_edge_support(self, edges, circle):
        """Measure how much of a candidate circumference is supported by edges."""
        cx, cy, radius = circle
        height, width = edges.shape[:2]
        supported = 0
        sample_count = 180
        for angle in np.linspace(0, 2 * math.pi, sample_count, endpoint=False):
            found = False
            for radius_offset in (-2, -1, 0, 1, 2):
                x = int(round(cx + (radius + radius_offset) * math.cos(angle)))
                y = int(round(cy + (radius + radius_offset) * math.sin(angle)))
                if 0 <= x < width and 0 <= y < height and edges[y, x] > 0:
                    found = True
                    break
            supported += int(found)
        return supported / sample_count

    def find_safe_normalized_circle(self, normalized, transform, min_radius, max_radius,
                                    param2, original_shape, required_center=None):
        enhanced = self.enhance_detection_gray(normalized)
        edges = cv2.Canny(enhanced, 30, 100)
        candidates = []
        seen = set()
        thresholds = sorted({param2, 25, 20, 15}, reverse=True)
        for accumulator_threshold in thresholds:
            circles = cv2.HoughCircles(
                enhanced, cv2.HOUGH_GRADIENT, dp=1.0, minDist=12,
                param1=50, param2=accumulator_threshold,
                minRadius=max(1, min_radius), maxRadius=max_radius,
            )
            if circles is None:
                continue
            for circle in np.around(circles[0]).astype(int):
                key = tuple(int(value) for value in circle)
                if key not in seen:
                    seen.add(key)
                    candidates.append(circle)

        safe_candidates = []
        content_center = transform["content_center"]
        for circle in candidates:
            mapped = self.map_circle_to_original(circle, transform)
            x0, y0, x1, y1 = transform["roi"]
            roi_circle = (mapped[0] - x0, mapped[1] - y0, mapped[2])
            if not (
                self.circle_is_safe(roi_circle, (y1 - y0, x1 - x0))
                and self.circle_is_safe(mapped, original_shape)
            ):
                continue
            if required_center is not None:
                center_distance = math.hypot(
                    mapped[0] - required_center[0],
                    mapped[1] - required_center[1],
                )
                if center_distance > max(3, mapped[2] * 0.04):
                    continue
            else:
                center_distance = math.hypot(
                    circle[0] - content_center[0],
                    circle[1] - content_center[1],
                ) / transform["scale"]
                if transform.get("reliable_center") and center_distance > 6:
                    continue
            edge_support = self.circle_edge_support(edges, circle)
            safe_candidates.append((circle, mapped, edge_support, center_distance))

        if not safe_candidates:
            self.last_edge_support = 0.0
            self.last_concentric_score = 0.0
            return None

        scored_candidates = []
        for _, mapped, edge_support, center_distance in safe_candidates:
            concentric_radii = set()
            if required_center is None and not transform.get("reliable_center"):
                center_tolerance = max(4, mapped[2] * 0.04)
                for _, other_mapped, _, _ in safe_candidates:
                    if math.hypot(
                        mapped[0] - other_mapped[0],
                        mapped[1] - other_mapped[1],
                    ) <= center_tolerance:
                        concentric_radii.add(int(round(other_mapped[2] / 8.0)))
            concentric_score = min(len(concentric_radii), 6) * 3.0
            score = edge_support * 100.0 + concentric_score - center_distance * 1.5
            scored_candidates.append((score, mapped, edge_support, concentric_score))

        best_score, best_mapped, best_edge, best_concentric = max(
            scored_candidates, key=lambda item: item[0]
        )
        self.last_edge_support = best_edge
        self.last_concentric_score = best_concentric
        return best_mapped

    def find_full_image_circle(self, image, min_radius, max_radius, param2,
                               required_center=None):
        """Use the enhanced adaptive detector on the full image as fallback."""
        height, width = image.shape[:2]
        short_side = min(height, width)
        transform = {
            "roi": (0, 0, width, height),
            "scale": 1.0,
            "pad_x": 0,
            "pad_y": 0,
            "content_center": (width / 2.0, height / 2.0),
            "reliable_center": False,
        }
        adaptive_min_radius = min_radius
        adaptive_max_radius = max_radius
        if required_center is None:
            adaptive_min_radius = min(min_radius, int(short_side * 0.08))
            adaptive_max_radius = max(max_radius, int(short_side * 0.48))
        return self.find_safe_normalized_circle(
            image,
            transform,
            min_radius=adaptive_min_radius,
            max_radius=adaptive_max_radius,
            param2=param2,
            original_shape=image.shape,
            required_center=required_center,
        )

    def save_detection_debug_images(self, image_path, output_dir, file_name):
        """Save the selected ROI and normalized detector input for verification."""
        image = cv2.imread(image_path)
        if image is None:
            return
        roi_preview = image.copy()
        if self.detection_transform is not None:
            x0, y0, x1, y1 = self.detection_transform["roi"]
            cv2.rectangle(roi_preview, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 255), 2)
            cv2.imwrite(
                os.path.join(output_dir, f"detection_normalized_{file_name}.png"),
                self.detection_transform["image"],
            )
            cv2.imwrite(
                os.path.join(output_dir, f"detection_enhanced_{file_name}.png"),
                self.enhance_detection_gray(self.detection_transform["image"]),
            )
        cv2.putText(
            roi_preview,
            f"detection_source: {self.detection_source}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(os.path.join(output_dir, f"detection_roi_{file_name}.png"), roi_preview)

    def visualize_ring_mask(self, image_path, ring_width=5):
        """创建环状掩码并返回处理后的图像"""
        # 读取并预处理图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        height, width = image.shape[:2]

        self.detection_transform = self.prepare_detection_image(image)
        first_circle = None
        if self.detection_transform is not None:
            self.detection_image = self.detection_transform["image"]
            first_circle = self.find_safe_normalized_circle(
                self.detection_image,
                self.detection_transform,
                min_radius=int(self.detection_size * 0.12),
                max_radius=int(self.detection_size * 0.32),
                param2=30,
                original_shape=image.shape,
            )
        if first_circle is not None:
            self.detection_source = "cropped"
        else:
            first_circle = self.find_full_image_circle(
                image,
                min_radius=int(height / 5),
                max_radius=int(height / 4),
                param2=30,
            )
            self.detection_source = "full_image_fallback"

        if first_circle is None:
            self.detection_source = "failed"
            self.coords = [0, 0]
            self.first_r = 0
            return image

        cx, cy, r = first_circle
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(r + ring_width), 255, -1)
        cv2.circle(mask, (cx, cy), max(0, int(r - ring_width)), 0, -1)
        masked_image = image.copy()
        masked_image[mask == 255] = 255
        self.coords = [cx, cy]
        self.first_r = r
        return masked_image
        
        # 霍夫变换检测第一个圆


            # 创建环状掩膜
            
            # 应用环状掩膜
            

    def second_circle_find(self, image):
        height, width = image.shape[:2]
        if self.first_r <= 0:
            self.second_r = 0
            return 0
        second_r = 0
        second_circle = None
        if self.detection_transform is not None:
            normalized = self.normalize_with_transform(image, self.detection_transform)
            scale = self.detection_transform["scale"]
            second_circle = self.find_safe_normalized_circle(
                normalized,
                self.detection_transform,
                min_radius=int(round((self.first_r + 30) * scale)),
                max_radius=int(self.detection_size / 2),
                param2=50,
                original_shape=image.shape,
                required_center=self.coords,
            )
            if second_circle is None and self.first_r > 30:
                second_circle = self.find_safe_normalized_circle(
                    normalized,
                    self.detection_transform,
                    min_radius=max(5, int(round(self.first_r * 0.2 * scale))),
                    max_radius=max(6, int(round((self.first_r - 20) * scale))),
                    param2=30,
                    original_shape=image.shape,
                    required_center=self.coords,
                )
        if second_circle is None:
            second_circle = self.find_full_image_circle(
                image,
                min_radius=self.first_r + 30,
                max_radius=int(height / 2),
                param2=50,
                required_center=self.coords,
            )
        if second_circle is None and self.first_r > 30:
            second_circle = self.find_full_image_circle(
                image,
                min_radius=max(5, int(round(self.first_r * 0.2))),
                max_radius=self.first_r - 20,
                param2=30,
                required_center=self.coords,
            )
        if second_circle is not None:
            candidate_radius = second_circle[2]
            min_separation = max(10, int(round(self.first_r * 0.08)))
            if abs(candidate_radius - self.first_r) >= min_separation:
                second_r = candidate_radius
        self.second_r = second_r

        # ── 修复 r1 > r2 反转 ──
        if self.second_r > 0 and self.first_r > self.second_r:
            print(f"  [FIX] r1 > r2 inversion detected: r1={self.first_r}, r2={self.second_r} -> swapping")
            self.first_r, self.second_r = self.second_r, self.first_r

        return second_r

    def check_circle_quality(self, image_shape):
        """
        霍夫圆检测质量评估 -> 决定是否触发兜底机制.
        Returns: (pass_quality: bool, reason: str)
        """
        h, w = image_shape[:2]
        min_size = min(h, w)
        if self.first_r <= 0:
            return False, "no_circle_detected"
        if self.detection_source == "failed":
            return False, "detection_source_is_failed"
        if self.last_edge_support < 0.20:
            return False, f"low_edge_support({self.last_edge_support:.2f})"
        if self.first_r / min_size < 0.08:
            return False, f"radius_too_small(ratio={self.first_r/min_size:.3f})"
        return True, "ok"

    def _init_ocr(self):
        if not hasattr(self, '_ocr_reader') or self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                print(f"[OCR] init failed: {e}")
                self._ocr_reader = None
        return self._ocr_reader

    def ocr_find_tick(self, target_radius, image_path):
        reader = self._init_ocr()
        if reader is None:
            return self._llm_find_tick(target_radius, image_path)
        image = cv2.imread(image_path)
        if image is None:
            return None, 'error', 0.0
        cropped = self.crop_tick_region(image, target_radius, pixel_range=35)
        if cropped is None or cropped.size == 0:
            return None, 'crop_failed', 0.0
        h, w = cropped.shape[:2]
        if max(h, w) < 100:
            scale = max(1.5, 200.0 / max(h, w))
            cropped = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ocr_results = reader.readtext(cropped)
        if not ocr_results:
            return self._llm_find_tick(target_radius, image_path)
        numbers = []
        for bbox, text, conf in ocr_results:
            text = re.sub(r'[^0-9.]', '', text or '').strip()
            if not text or conf < 0.2:
                continue
            try:
                val = float(text)
            except ValueError:
                continue
            numbers.append((val, float(conf), text))
        if not numbers:
            return self._llm_find_tick(target_radius, image_path)
        best = max(numbers, key=lambda x: x[1])
        tick_val = int(best[0]) if best[0] == int(best[0]) else best[0]
        conf = best[1]
        if conf < 0.5:
            return self._llm_find_tick(target_radius, image_path)
        return tick_val, 'ocr', conf

    def _llm_find_tick(self, target_radius, image_path):
        result = self.find_tick(target_radius, image_path)
        if result is None:
            return None, 'llm_failed', 0.0
        tick = result.get("tick")
        if tick is None:
            return None, 'llm_null', 0.0
        try:
            tick = int(tick) if int(tick) == tick else float(tick)
        except (ValueError, TypeError):
            return None, 'llm_parse_error', 0.0
        return tick, 'llm_fallback', 0.7

    def ocr_tick_range(self, image_path):
        reader = self._init_ocr()
        if reader is None:
            return self._llm_tick_range(image_path)
        image = cv2.imread(image_path)
        if image is None:
            return self._llm_tick_range(image_path)
        center_x, center_y = self.coords
        ocr_results = reader.readtext(image)
        if not ocr_results:
            return self._llm_tick_range(image_path)
        numbers = []
        for bbox, text, conf in ocr_results:
            text = re.sub(r'[^0-9.]', '', text or '').strip()
            if not text or conf < 0.25:
                continue
            try:
                val = float(text)
            except ValueError:
                continue
            pts = np.array(bbox, dtype=float)
            bcx = float(np.mean(pts[:, 0]))
            bcy = float(np.mean(pts[:, 1]))
            dist = math.hypot(bcx - center_x, bcy - center_y)
            numbers.append({"value": val, "dist": dist, "conf": conf, "text": text})
        if len(numbers) < 3:
            return self._llm_tick_range(image_path)
        sorted_nums = sorted(numbers, key=lambda n: n["dist"])
        n = len(sorted_nums)
        best_k, best_labels, best_sep = 2, None, -1
        for k in range(2, min(6, n)):
            chunk = n // k
            labels_list = [min(i // max(chunk, 1), k - 1) for i in range(n)]
            labels = np.array(labels_list)
            dists = np.array([num["dist"] for num in sorted_nums], dtype=float).reshape(-1, 1)
            centers, stds = [], []
            for lbl in range(k):
                mask = labels == lbl
                if mask.sum() > 0:
                    centers.append(dists[mask].mean())
                    stds.append(dists[mask].std())
            if len(centers) >= 2 and sum(stds) > 0:
                sep = (max(centers) - min(centers)) / max(sum(stds), 1e-6)
                if sep > best_sep:
                    best_sep, best_k, best_labels = sep, k, labels
        if best_labels is None:
            return self._llm_tick_range(image_path)
        clusters = {}
        for num, lbl in zip(sorted_nums, best_labels):
            lbl = int(lbl)
            clusters.setdefault(lbl, []).append(num)
        sorted_clusters = sorted(clusters.items(), key=lambda kv: np.mean([x["dist"] for x in kv[1]]))
        inner_vals = [x["value"] for x in sorted_clusters[0][1]]
        outer_vals = [x["value"] for x in sorted_clusters[-1][1]]
        min_tick = min(inner_vals + outer_vals)
        max_tick = max(inner_vals + outer_vals)
        all_vals = sorted(set(x["value"] for x in numbers))
        if len(all_vals) >= 3:
            diffs = [all_vals[i+1] - all_vals[i] for i in range(len(all_vals)-1)]
            interval = float(np.median(diffs))
        else:
            interval = (max_tick - min_tick) / max(len(clusters) - 1, 1)
        if interval <= 0 or max_tick <= min_tick:
            return self._llm_tick_range(image_path)
        return {"max_tick_value": max_tick, "min_tick_value": min_tick, "tick_interval": interval, "source": "ocr", "res": f"OCR clustering k={best_k} n={len(numbers)}"}

    def _llm_tick_range(self, image_path):
        result = self.call_llm_response(image_path)
        if result is None:
            return {"max_tick_value": None, "min_tick_value": None, "tick_interval": None, "source": "llm_failed", "res": "LLM call failed"}
        result["source"] = "llm_fallback"
        return result

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
        # Keep each radius paired with the tick recognized from that radius.
        r1, r2 = self.first_r, self.second_r
        interval = tick_interval / self.tick_density
        if tick2 == tick1:
            raise ValueError("Two detected circles cannot have the same tick value.")
        a = float(r2 - r1) / float(tick2 - tick1)

        # 计算半径与刻度的线性关系 (r = a*tick + b)
        b = r1 - a * tick1
        if a <= 0:
            raise ValueError(
                f"Invalid radius mapping from detected pairs: "
                f"({r1}, {tick1}) and ({r2}, {tick2})"
            )
        print(f"Radius mapping: r = {a} * tick + {b}")
        
        result = image.copy()
        
        # 设置字体和颜色
        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        font_size, label_offset, font_scale = self.get_annotation_metrics(
            image.shape,
            radial_spacing=a * interval,
        )
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
                    font = ImageFont.truetype("arial.ttf", size=font_size)
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)
                # 获取文本边界框
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 右侧旋转90度文本
                temp_img_right = Image.new('RGBA', (text_width + 4, text_height + 4), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((2, 2), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)  # 顺时针旋转90度
                
                # 调整右侧位置
                pos_right = (center_x + radius + label_offset,
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                
                # 左侧旋转-90度文本
                temp_img_left = Image.new('RGBA', (text_width + 4, text_height + 4), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((2, 2), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)  # 逆时针旋转90度
                
                # 调整左侧位置
                pos_left = (center_x - radius - rotated_left.size[0] - label_offset,
                        center_y - rotated_left.size[1]//2)
                
                # 底部正常文本
                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius + label_offset
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                
                # 转换回OpenCV格式
                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            # 保存生成的刻度值
            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale, font_color, 1, lineType=cv2.LINE_AA)
            
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
        font_size, label_offset, font_scale = self.get_annotation_metrics(
            image.shape,
            radial_spacing=a * interval,
        )
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
                    font = ImageFont.truetype("arial.ttf", size=font_size)
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)
                # 获取文本边界框
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 右侧旋转90度文本
                temp_img_right = Image.new('RGBA', (text_width + 4, text_height + 4), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((2, 2), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)  # 顺时针旋转90度
                
                # 调整右侧位置
                pos_right = (center_x + radius + label_offset,
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                
                # 左侧旋转-90度文本
                temp_img_left = Image.new('RGBA', (text_width + 4, text_height + 4), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((2, 2), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)  # 逆时针旋转90度
                
                # 调整左侧位置
                pos_left = (center_x - radius - rotated_left.size[0] - label_offset,
                        center_y - rotated_left.size[1]//2)
                
                # 底部正常文本
                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius + label_offset
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                
                # 转换回OpenCV格式
                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            # 保存生成的刻度值
            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale, font_color, 1, lineType=cv2.LINE_AA)
            
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
            self.fallback_flag = False
            self.fallback_reason = ""
            if output_dir is None:
                output_dir = os.path.dirname(image_path) or '.'
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            second_circle = self.visualize_ring_mask(image_path)
            self.second_circle_find(second_circle)
            self.save_detection_debug_images(image_path, output_dir, file_name)
            print(f"Circle detection source: {self.detection_source}")
            print(f"Quality metrics: edge_support={self.last_edge_support:.3f}, concentric_score={self.last_concentric_score:.1f}")
            if self.detection_transform is not None:
                print(f"Detection ROI: {self.detection_transform['roi']}")
            pass_quality, quality_reason = self.check_circle_quality(image.shape)
            if not pass_quality:
                self.fallback_flag = True
                self.fallback_reason = f"circle_quality_failed: {quality_reason}"
                print(f"  Circle quality insufficient -> fallback: {self.fallback_reason}")
                return {"fallback": True, "reason": self.fallback_reason}
            if self.first_r <= 0:
                print("Circle detection failed; skipping later processing.")
                self.fallback_flag = True
                self.fallback_reason = "no_circle_detected"
                return {"fallback": True, "reason": self.fallback_reason}
            print(f"检测到的圆心坐标: {self.coords}")
            print(f"第一个圆半径: {self.first_r}")
            print(f"第二个圆半径: {self.second_r}")
            temp_output_path = os.path.join(output_dir, f"temp_marked_{file_name}{file_ext}")
            image = cv2.imread(image_path)
            cv2.circle(image, (self.coords[0], self.coords[1]), self.first_r, (0, 255, 0), 1)
            if self.second_r > 0:
                cv2.circle(image, (self.coords[0], self.coords[1]), self.second_r, (0, 255, 0), 1)
            cv2.circle(image, (self.coords[0], self.coords[1]), 2, (255, 0, 0), -1)
            cv2.imwrite(temp_output_path, image)
            print(f"临时标记图像已保存至: {temp_output_path}")
            # OCR-first tick reading
            tick1, source1, conf1 = self.ocr_find_tick(self.first_r, temp_output_path)
            reason1 = f"source={source1}, conf={conf1:.2f}" if tick1 is not None else f"{source1}"
            if self.second_r > 0:
                tick2, source2, conf2 = self.ocr_find_tick(self.second_r, temp_output_path)
                reason2 = f"source={source2}, conf={conf2:.2f}" if tick2 is not None else f"{source2}"
            else:
                tick2, source2, conf2 = None, 'no_second_circle', 0.0
                reason2 = "Second circle was not detected reliably."
            if tick1 is not None and tick2 == tick1:
                print("Two circles resolved to the same tick; using the first circle only.")
                tick2 = None
            print(f"第一个圆刻度: tick={tick1}, {reason1}")
            print(f"第二个圆刻度: tick={tick2}, {reason2}")
            range_data = self.ocr_tick_range(temp_output_path)
            max_tick_value = range_data.get("max_tick_value")
            min_tick_value = range_data.get("min_tick_value")
            tick_interval = range_data.get("tick_interval")
            range_source = range_data.get("source", "unknown")
            range_res = range_data.get("res", "")
            print(f"刻度范围: max={max_tick_value}, min={min_tick_value}, interval={tick_interval}, source={range_source}")
            
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
                if not isinstance(json_data, dict):
                    print("JSON root is not an object; skipping metadata update.")
                    return output_path
                    
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
    encoder = RadarChartEncoder()
    
    # 指定要处理的图像路径和输出目录
    image_path = "./backend/real/RadarChart-18 & RoseChart-6/RadarChart-18-final/RadarChart15.png"  # 可以根据需要修改
    output_dir = "./backend/data/polar/output/radar"      # 可以根据需要修改
    
    # 处理单张图像
    result_path = encoder.process_single_image(image_path, output_dir)
    
    if result_path:
        print(f"处理完成！加密后的图像保存在: {result_path}")
    else:
        print("处理失败！")
