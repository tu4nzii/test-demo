import cv2
import base64
import numpy as np
import json 
import requests
import os
import math
from PIL import Image, ImageDraw, ImageFont
import re

from model_api_config import get_chat_completion_url, get_headers, get_model_name

class RadarChartEncoder:
    def __init__(self):

        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        self.tick_density = 2
        

        self.result_image = None
        self.r_ticks = []
        self.argument = {}
        self.coords = [0, 0]  # [cx, cy]
        self.first_r = 0
        self.second_r = 0
        self._source_image = None

    def show_image_with_scaling(self, window_name, image, max_width=800, max_height=600):
        """Show a resized image."""
        height, width = image.shape[:2]
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height, 1.0)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        
        cv2.imshow(window_name, resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def extract_json_response(self, content: str):
        """Extract JSON content from an LLM response."""
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            json_str = match.group(1)
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON parse failed: {e}")
            return None

    def visualize_ring_mask(self, image_path, ring_width=5):
        """Create an annular mask and return the processed image."""

        image = cv2.imread(image_path)
        self._source_image = image.copy() if image is not None else None
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                 param1=20, param2=30, minRadius=int(height/5), maxRadius=int(height/4))
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            first_circle = circles[0, 0]
            cx, cy, r = first_circle[0], first_circle[1], first_circle[2]
            

            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), int(r+ring_width), 255, -1)
            cv2.circle(mask, (cx, cy), int(r-ring_width), 0, -1)
            

            masked_blurred = image.copy()
            masked_blurred[mask == 255] = 255
            
            self.coords = [cx, cy]
            self.first_r = r
            return masked_blurred
        else:
            return image

    def _circle_edge_support(self, image, center, radius, sample_count=240, band=2):
        """Measure how much edge evidence exists around a proposed circle."""
        try:
            radius = float(radius)
            cx, cy = float(center[0]), float(center[1])
        except (TypeError, ValueError, IndexError):
            return 0.0
        if image is None or radius <= 0:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        height, width = gray.shape[:2]
        hits = 0
        total = 0
        for angle in np.linspace(0, 2 * math.pi, sample_count, endpoint=False):
            x = int(round(cx + math.cos(angle) * radius))
            y = int(round(cy + math.sin(angle) * radius))
            if not (0 <= x < width and 0 <= y < height):
                continue
            total += 1
            patch = edges[max(0, y - band):min(height, y + band + 1), max(0, x - band):min(width, x + band + 1)]
            if patch.size and patch.max() > 0:
                hits += 1
        return hits / max(total, 1)

    def _bbox_from_mask(self, mask, *, min_area=50, max_components=10, morph_open=True):
        mask = mask.astype("uint8") * 255
        if morph_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area >= min_area:
                components.append((int(x), int(y), int(width), int(height), int(area)))
        if not components:
            return None, []
        components = sorted(components, key=lambda item: item[4], reverse=True)
        if max_components is not None:
            components = components[:max_components]
        xs = [item[0] for item in components]
        ys = [item[1] for item in components]
        rights = [item[0] + item[2] for item in components]
        bottoms = [item[1] + item[3] for item in components]
        return (min(xs), min(ys), max(rights), max(bottoms)), components

    def _line_intersection(self, line_a, line_b):
        x1, y1, x2, y2 = line_a
        x3, y3, x4, y4 = line_b
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-6:
            return None
        px = (
            (x1 * y2 - y1 * x2) * (x3 - x4)
            - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / denominator
        py = (
            (x1 * y2 - y1 * x2) * (y3 - y4)
            - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / denominator
        return px, py

    def _line_angle(self, line):
        x1, y1, x2, y2 = line
        return math.atan2(y2 - y1, x2 - x1)

    def _angle_diff(self, angle_a, angle_b):
        return abs((angle_a - angle_b + math.pi / 2) % math.pi - math.pi / 2)

    def _cluster_points(self, weighted_points, radius):
        clusters = []
        for x, y, weight in weighted_points:
            best_cluster = None
            best_dist = float("inf")
            for cluster in clusters:
                cx, cy = cluster["center"]
                dist = (x - cx) ** 2 + (y - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster
            if best_cluster is not None and best_dist <= radius * radius:
                best_cluster["points"].append((x, y, weight))
                total_weight = sum(item[2] for item in best_cluster["points"])
                if total_weight > 0:
                    best_cluster["center"] = (
                        sum(item[0] * item[2] for item in best_cluster["points"]) / total_weight,
                        sum(item[1] * item[2] for item in best_cluster["points"]) / total_weight,
                    )
            else:
                clusters.append({"center": (x, y), "points": [(x, y, weight)]})
        clusters.sort(
            key=lambda cluster: (
                len(cluster["points"]),
                sum(item[2] for item in cluster["points"]),
            ),
            reverse=True,
        )
        return clusters

    def _estimate_polygon_center_from_lines(self, image):
        """Find the common intersection of polygon radar spokes/grid lines."""
        if image is None:
            return None
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        border = np.concatenate([gray[:5, :].ravel(), gray[-5:, :].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
        dark_background = float(np.median(border)) < 80

        if dark_background:
            line_mask = ((val > 60) & (sat < 120)) | ((sat > 20) & (val > 70))
        else:
            line_mask = (sat <= 80) & (gray > 50) & (gray < 252)
        line_mask = line_mask.astype("uint8") * 255
        edges = cv2.Canny(line_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=24 if not dark_background else 28,
            minLineLength=max(24, int(min(width, height) * 0.07)),
            maxLineGap=15,
        )
        if lines is None:
            return None

        segments = []
        for raw_line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(value) for value in raw_line]
            length = math.hypot(x2 - x1, y2 - y1)
            if length < max(24, min(width, height) * 0.06):
                continue
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            if not (width * 0.03 < mx < width * 0.97 and height * 0.03 < my < height * 0.97):
                continue
            segments.append((x1, y1, x2, y2, length, self._line_angle((x1, y1, x2, y2))))
        if len(segments) < 4:
            return None

        intersections = []
        for index, first in enumerate(segments):
            for second in segments[index + 1:]:
                diff = self._angle_diff(first[5], second[5])
                if diff < math.radians(18):
                    continue
                point = self._line_intersection(first[:4], second[:4])
                if point is None:
                    continue
                x, y = point
                if not (width * 0.05 <= x <= width * 0.95 and height * 0.05 <= y <= height * 0.95):
                    continue
                intersections.append((x, y, math.sin(diff) * min(first[4], second[4])))
        if not intersections:
            return None

        clusters = self._cluster_points(intersections, max(10, min(width, height) * 0.025))
        if not clusters:
            return None

        best = clusters[0]
        if len(best["points"]) < 6:
            return None
        cx, cy = best["center"]
        return [int(round(cx)), int(round(cy))], len(best["points"])

    def _estimate_nearby_grid_contour_center(self, image, bbox_center, outer_radius):
        """Use a dominant grid contour only when it agrees with the coarse center."""
        if image is None or not bbox_center or outer_radius <= 0:
            return None
        height, width = image.shape[:2]
        if abs(width - height) / max(width, height) > 0.25:
            return None
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        border = np.concatenate([gray[:5, :].ravel(), gray[-5:, :].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
        dark_background = float(np.median(border)) < 80

        if dark_background:
            mask = ((val > 45) & (sat < 95)).astype("uint8") * 255
        else:
            mask = ((sat < 55) & (gray > 80) & (gray < 245)).astype("uint8") * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = max(150, image.shape[0] * image.shape[1] * 0.0005)
        best_contour = None
        best_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            if area > best_area:
                best_area = area
                best_contour = contour
        if best_contour is None:
            return None

        moments = cv2.moments(best_contour)
        if not moments.get("m00"):
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        distance = math.hypot(cx - bbox_center[0], cy - bbox_center[1])

        # This is intentionally tight. Some charts have asymmetric colored
        # fills, and their contour centroids are not valid chart centers.
        if distance <= max(6, outer_radius * 0.06):
            return [int(round(cx)), int(round(cy))]
        return None

    def _dominant_light_grid_bbox(self, image):
        if image is None:
            return None
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        border = np.concatenate([gray[:5, :].ravel(), gray[-5:, :].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
        if float(np.median(border)) < 80:
            return None

        mask = ((sat < 55) & (gray > 80) & (gray < 245)).astype("uint8") * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < max(40, width * 0.12) or box_height < max(40, height * 0.12):
                continue
            if best is None or area > best[0]:
                best = (area, (int(x), int(y), int(x + box_width), int(y + box_height)))
        return best[1] if best is not None else None

    def _estimate_plot_geometry_from_masks(self, image):
        """Detect polygon radar center/radius without disturbing circular charts.

        The original path relies on Hough circles, which is excellent for the
        synthetic circular-grid dataset. Real-world radar charts often use
        polygon grids or very faint gridlines. For those, we estimate the
        polygon center from spoke intersections and grid/foreground extent.
        """
        if image is None:
            return None
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        border = np.concatenate([gray[:5, :].ravel(), gray[-5:, :].ravel(), gray[:, :5].ravel(), gray[:, -5:].ravel()])
        dark_background = float(np.median(border)) < 80

        if dark_background:
            mask = (sat > 25) & (val > 60)
            bbox, components = self._bbox_from_mask(mask, min_area=50, max_components=None)
        else:
            color_mask = (sat > 25) & (val > 45)
            color_bbox, color_components = self._bbox_from_mask(color_mask, min_area=50)
            if color_components:
                mask = color_mask | ((sat <= 55) & (gray > 100) & (gray < 250))
                bbox, components = self._bbox_from_mask(mask, min_area=50, max_components=20)
            else:
                mask = (gray > 20) & (gray < 252)
                bbox, components = self._bbox_from_mask(mask, min_area=8, max_components=None, morph_open=False)
        if bbox is None:
            return None

        grid_bbox = self._dominant_light_grid_bbox(image)
        if grid_bbox is not None:
            bbox = grid_bbox

        left, top, right, bottom = bbox
        box_width = right - left
        box_height = bottom - top
        if box_width < max(40, width * 0.12) or box_height < max(40, height * 0.12):
            return None

        bbox_center = [int(round((left + right) / 2)), int(round((top + bottom) / 2))]
        line_center_result = self._estimate_polygon_center_from_lines(image)
        if line_center_result is not None:
            line_center, line_votes = line_center_result
            distance = math.hypot(line_center[0] - bbox_center[0], line_center[1] - bbox_center[1])
            no_color_polygon = not dark_background and not ((sat > 25) & (val > 45)).any()
            if distance <= max(box_width, box_height) * 0.18 or (no_color_polygon and distance <= max(box_width, box_height) * 0.25):
                cx, cy = line_center
            else:
                cx, cy = bbox_center
        else:
            cx, cy = bbox_center

        outer_radius = int(round(min(box_width, box_height) / 2))
        max_radius = int(min(cx, cy, width - 1 - cx, height - 1 - cy))
        if max_radius > 0:
            outer_radius = min(outer_radius, max_radius)
        if outer_radius < max(25, min(width, height) * 0.08):
            return None

        contour_center = self._estimate_nearby_grid_contour_center(image, [cx, cy], outer_radius)
        if contour_center is not None:
            cx, cy = contour_center

        inner_radius = int(round(outer_radius * 0.65))
        return [cx, cy], max(1, inner_radius), max(1, outer_radius)

    def _hint_requests_polygon_geometry(self, radar_grid_hint):
        if not isinstance(radar_grid_hint, dict):
            return False
        nested = radar_grid_hint.get("radar_grid")
        if isinstance(nested, dict):
            shape = nested.get("shape", radar_grid_hint.get("radar_grid_shape", ""))
            confidence = nested.get("confidence", radar_grid_hint.get("radar_grid_confidence", 0))
        else:
            shape = radar_grid_hint.get("radar_grid_shape", radar_grid_hint.get("shape", ""))
            confidence = radar_grid_hint.get("radar_grid_confidence", radar_grid_hint.get("confidence", 0))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return str(shape).strip().lower() == "polygon" and confidence >= 0.55

    def _hint_requests_circular_geometry(self, radar_grid_hint):
        if not isinstance(radar_grid_hint, dict):
            return False
        nested = radar_grid_hint.get("radar_grid")
        if isinstance(nested, dict):
            shape = nested.get("shape", radar_grid_hint.get("radar_grid_shape", ""))
            confidence = nested.get("confidence", radar_grid_hint.get("radar_grid_confidence", 0))
        else:
            shape = radar_grid_hint.get("radar_grid_shape", radar_grid_hint.get("shape", ""))
            confidence = radar_grid_hint.get("radar_grid_confidence", radar_grid_hint.get("confidence", 0))
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        return str(shape).strip().lower() == "circular" and confidence >= 0.55

    def _estimate_circular_grid_geometry(self, image):
        if image is None:
            return None
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]

        grid_mask = ((sat < 80) & (gray > 80) & (gray < 245)).astype("uint8") * 255
        grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        min_area = max(300, height * width * 0.0004)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < max(60, width * 0.12) or box_height < max(60, height * 0.12):
                continue
            ratio = box_width / max(box_height, 1)
            if ratio < 0.65 or ratio > 1.45:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < max(35, min(width, height) * 0.05):
                continue
            if best is None or area > best[0]:
                best = (area, int(round(cx)), int(round(cy)), int(round(radius)))
        if best is None:
            return None

        _, cx, cy, radius = best
        max_radius = int(min(cx, cy, width - 1 - cx, height - 1 - cy))
        if max_radius > 0:
            radius = min(radius, max_radius)
        inner_radius = int(round(radius * 0.72))
        return [cx, cy], max(1, inner_radius), max(1, radius)

    def _should_use_geometry_fallback(self, image, proposed_second_r, fallback=None, prefer_polygon_geometry=False):
        if image is None:
            return False
        height, width = image.shape[:2]
        try:
            cx, cy = int(self.coords[0]), int(self.coords[1])
            first_r = float(self.first_r)
            second_r = float(proposed_second_r)
        except (TypeError, ValueError, IndexError):
            return fallback is not None
        if first_r <= 0 or second_r <= 0:
            return fallback is not None
        if cx <= 2 or cy <= 2 or cx >= width - 3 or cy >= height - 3:
            return fallback is not None
        if fallback is None:
            return False
        if not prefer_polygon_geometry:
            return False

        _, _, fallback_outer_r = fallback
        try:
            fallback_outer_r = float(fallback_outer_r)
        except (TypeError, ValueError):
            return False
        if fallback_outer_r <= 0:
            return False

        # Polygon radar fallback should describe the same plotting area scale.
        # This guards circular charts where a colored data polygon creates a
        # small bbox even though the original Hough circle is still preferable.
        if fallback_outer_r < second_r * 0.55:
            return False

        return True

    def _should_use_circular_grid_refinement(self, image, proposed_second_r, circular_geometry):
        if image is None or circular_geometry is None:
            return False
        height, width = image.shape[:2]
        if min(width, height) <= 1000:
            return False
        if abs(width - height) / max(width, height) < 0.05:
            return False
        try:
            cx, cy = int(self.coords[0]), int(self.coords[1])
            first_r = float(self.first_r)
            second_r = float(proposed_second_r)
            center, _, grid_outer_r = circular_geometry
            grid_outer_r = float(grid_outer_r)
        except (TypeError, ValueError, IndexError):
            return False
        if first_r <= 0 or second_r <= 0 or grid_outer_r <= 0:
            return False
        if grid_outer_r / max(1, min(width, height)) > 0.23:
            return False

        distance = math.hypot(cx - center[0], cy - center[1])
        radius_mismatch = max(first_r, second_r) / grid_outer_r
        return distance > max(18, grid_outer_r * 0.12) and radius_mismatch > 1.30

    def second_circle_find(self, image, radar_grid_hint=None):
        """Detect the second circle."""
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

        source = self._source_image if self._source_image is not None else image
        fallback = self._estimate_plot_geometry_from_masks(source)
        prefer_polygon_geometry = self._hint_requests_polygon_geometry(radar_grid_hint)
        if self._should_use_geometry_fallback(source, second_r, fallback, prefer_polygon_geometry):
            if fallback is not None:
                center, first_r, outer_r = fallback
                self.coords = center
                self.first_r = first_r
                second_r = outer_r
        elif self._hint_requests_circular_geometry(radar_grid_hint):
            circular_geometry = self._estimate_circular_grid_geometry(source)
            if self._should_use_circular_grid_refinement(source, second_r, circular_geometry):
                center, first_r, outer_r = circular_geometry
                self.coords = center
                self.first_r = first_r
                second_r = outer_r
            
        self.second_r = second_r
        return second_r

    def crop_tick_region(self, image, target_radius, pixel_range=25):
        """Crop an annular region near the specified radius."""
        center_x, center_y = self.coords
        

        mask = np.zeros_like(image)
        

        outer_radius = target_radius + pixel_range
        inner_radius = max(0, target_radius - 10)
        

        cv2.circle(mask, (center_x, center_y), outer_radius, (255, 255, 255), -1)
        cv2.circle(mask, (center_x, center_y), inner_radius, (0, 0, 0), -1)
        

        masked_image = cv2.bitwise_and(image, mask)
        

        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:

            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return masked_image[y:y+h, x:x+w]
        

        return masked_image

    def find_tick(self, target_radius, image_path):
        """Use the LLM to recognize the tick value near a radius."""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
            
        cropped_image = self.crop_tick_region(image, target_radius, pixel_range=25)
        

        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            
        retval, buffer = cv2.imencode('.png', rgb_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""
You are reading a cropped region from a polar chart. A green circular marker highlights one radial tick ring.

Task:
1. Identify the visible numeric tick value corresponding to the green ring.
2. Read only numbers that are actually visible in the image. Do not infer missing labels.
3. If multiple numbers are visible, no number is visible, the text is not numeric, or the value appears to be 0/100 due to ambiguity, return null.
4. Percent labels should be returned as their numeric value, for example 50% -> 50.

Return strict JSON only, with no explanation outside JSON:
{{
  "tick": <number_or_null>,
  "res": "brief visual reasoning"
}}
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
            print(f"API request failed: {e}")
            return None

    def call_llm_response(self, image_path):
        """Use the LLM to extract chart tick information."""
        center_x, center_y = self.coords
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        prompt = f"""
You are analyzing a polar/radar/rose chart. The approximate chart center is ({center_x}, {center_y}) in pixel coordinates.

Task:
1. Read the largest radial tick value shown on the outer visible radial grid/tick labels.
2. Read the smallest radial tick value shown on the visible radial grid/tick labels.
3. Read the radial tick interval if it is explicitly visible or can be directly determined from visible consecutive tick labels.

Rules:
- Read only numbers printed in the chart. Do not infer labels that are not visible.
- Percent labels should be returned as their numeric value, for example 50% -> 50.
- If a field cannot be recognized, use null. If a visible value is 0, return 0.

Return strict JSON only, with no explanation outside JSON:
{{
  "max_tick_value": <number_or_null>,
  "min_tick_value": <number_or_null>,
  "tick_interval": <number_or_null>,
  "res": "brief visual reasoning"
}}
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
            print(f"API request error: {e}")
            print(f"Response content: {response.text if 'response' in locals() else 'no response'}")
            return None

    def _annotation_scale(self, width, height, plot_radius):
        base_scale = math.sqrt(width * height)
        try:
            plot_radius = float(plot_radius)
        except (TypeError, ValueError):
            return base_scale
        if plot_radius > 0 and plot_radius / max(1, min(width, height)) < 0.23:
            return min(base_scale, max(180.0, plot_radius * 2.8))
        return base_scale

    def encrypt_rose_chart_with_tick(self, image_path, tick_interval, tick1, tick2, max_tick_value, min_tick_value):
        """Encrypt the grid using two tick values."""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        

        r1, r2 = (self.first_r, self.second_r) if self.second_r > self.first_r else (self.second_r, self.first_r)
        interval = tick_interval / self.tick_density
        # Radar charts use the center as radial value 0. Avoid shifting the
        # origin when the MLLM misses an inner visible tick label.
        a = float(r2) / max(float(tick2), 1e-6)


        b = 0
        
        result = image.copy()
        

        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        scale = self._annotation_scale(width, height, r2)
        font_scale = scale * 0.006
        font_color = (0, 0, 0)
        line_color = (128, 128, 128)
        thickness = 1
        

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
                

            radius = int(a * tick + b)
            current_px_distance = abs(radius - 0)
            
            if radius <= 0:
                print(f"Skipping invalid radius: tick={tick}, radius={radius}")
                continue
                
            if current_px_distance <= 3:
                print(f"Skipping invalid radius: tick={tick}, radius={radius}")
                break
                
            text_x_up = center_x 
            text_y_up = center_y - radius 


            if tick % 1 == 0:
                tick = int(tick)
                
            if tick > 0:
                pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                

                try:
                    font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)

                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                

                temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((0, 0), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)
                

                pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                

                temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((0, 0), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)
                

                pos_left = (center_x - radius - int(width*0.0125), 
                        center_y - rotated_left.size[1]//2)
                

                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius - int(height*0.0122) 
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                

                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                

            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1, lineType=cv2.LINE_AA)
            

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
        """Encrypt the grid using one tick value."""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        interval = tick_interval / self.tick_density
        

        # Radar charts use the center as radial value 0. Avoid shifting the
        # origin when the MLLM misses an inner visible tick label.
        a = float(r) / max(float(tick1), 1e-6)
        b = 0
        result = image.copy()
        

        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        scale = self._annotation_scale(width, height, r)
        font_scale = scale * 0.006
        font_color = (0, 0, 0)
        line_color = (128, 128, 128)
        thickness = 1
        

        tick = max_tick_value
        r_ticks = []
        self.argument = {'a': a, 'b': b}
        
        current_px_distance = 10000
        radius = 0
        count = 0
        
        while tick > 0 and radius >= 0:
            tick -= interval

            radius = int(a * tick + b)
            current_px_distance = abs(radius - 0)
            
            if radius <= 0:
                print(f"Skipping invalid radius: tick={tick}, radius={radius}")
                continue
                
            if current_px_distance <= 3:
                print(f"Skipping invalid radius: tick={tick}, radius={radius}")
                break
                
            text_x_up = center_x 
            text_y_up = center_y - radius 


            if tick % 1 == 0:
                tick = int(tick)
                
            if tick > 0:
                pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                

                try:
                    font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)

                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                

                temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((0, 0), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)
                

                pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                

                temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((0, 0), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)
                

                pos_left = (center_x - radius - int(width*0.0125), 
                        center_y - rotated_left.size[1]//2)
                

                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius - int(height*0.0122) 
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                

                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                

            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1, lineType=cv2.LINE_AA)
            

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

    def _as_positive_number(self, value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number <= 0:
            return None
        return number

    def _normalize_scale_info(self, response_data):
        if not isinstance(response_data, dict):
            return None

        max_tick_value = self._as_positive_number(response_data.get("max_tick_value"))
        tick_interval = self._as_positive_number(response_data.get("tick_interval"))
        min_tick_raw = response_data.get("min_tick_value")
        try:
            min_tick_value = float(min_tick_raw)
        except (TypeError, ValueError):
            min_tick_value = 0
        if not math.isfinite(min_tick_value) or min_tick_value < 0:
            min_tick_value = 0

        if max_tick_value is None:
            return None
        if tick_interval is None or tick_interval >= max_tick_value:
            tick_interval = max_tick_value / 5
        if tick_interval <= 0:
            return None

        return max_tick_value, min_tick_value, tick_interval

    def _default_polygon_scale_info(self):
        """Fallback scale for polygon radar charts without readable r labels."""
        max_tick_value = 100.0
        min_tick_value = 0.0
        tick_interval = 20.0
        return max_tick_value, min_tick_value, tick_interval

    def _outer_radius(self):
        radii = []
        for radius in [self.first_r, self.second_r]:
            try:
                radius = float(radius)
            except (TypeError, ValueError):
                continue
            if math.isfinite(radius) and radius > 0:
                radii.append(radius)
        return max(radii) if radii else None

    def process_single_image(self, image_path, output_dir=None, radar_grid_hint=None):
        """Process one radar chart through the full encryption flow."""
        try:

            if output_dir is None:
                output_dir = os.path.dirname(image_path) or '.'
                
            os.makedirs(output_dir, exist_ok=True)
            

            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            

            second_circle = self.visualize_ring_mask(image_path)
            self.second_circle_find(second_circle, radar_grid_hint=radar_grid_hint)
            
            print(f"Detected center: {self.coords}")
            print(f"First circle radius: {self.first_r}")
            print(f"Second circle radius: {self.second_r}")
            

            temp_output_path = os.path.join(output_dir, f"temp_marked_{file_name}{file_ext}")
            

            image = cv2.imread(image_path)
            cv2.circle(image, (self.coords[0], self.coords[1]), self.first_r, (0, 255, 0), 1)
            cv2.circle(image, (self.coords[0], self.coords[1]), self.second_r, (0, 255, 0), 1)
            cv2.circle(image, (self.coords[0], self.coords[1]), 2, (255, 0, 0), -1)
            
            cv2.imwrite(temp_output_path, image)
            print(f"Temporary marked image saved to: {temp_output_path}")
            

            response_data = self.call_llm_response(temp_output_path)
            scale_info = self._normalize_scale_info(response_data)
            scale_source = "llm"
            if scale_info is None:
                if self._hint_requests_polygon_geometry(radar_grid_hint):
                    scale_info = self._default_polygon_scale_info()
                    scale_source = "simulated_polygon_default"
                    print(f"Unable to read radial numeric ticks; using polygon radar fallback scale: {scale_info}")
                else:
                    print(f"Unable to obtain valid radar tick information: {response_data}")
                    return None
            max_tick_value, min_tick_value, tick_interval = scale_info
            outer_radius = self._outer_radius()
            if outer_radius is None:
                print("No valid outer radar radius detected")
                return None
            res = f"outer_radius={outer_radius}"
            
            print(f"LLM analysis: max={max_tick_value}, min={min_tick_value}, interval={tick_interval}, {res}")
            

            result, r_ticks, argument = self.encrypt_rose_chart_one_tick(
                image_path, tick_interval, max_tick_value, outer_radius, max_tick_value, min_tick_value
            )
            

            output_path = os.path.join(output_dir, f"{file_name}_encode{file_ext}")
            cv2.imwrite(output_path, result)
            print(f"Output saved to: {output_path}")
            

            json_fname = f"{file_name}.json"
            output_json_path = os.path.join(output_dir, json_fname)
            generated_r_ticks = r_ticks[::-1]
            if generated_r_ticks:
                generated_r_ticks.append(generated_r_ticks[-1] + (tick_interval / self.tick_density))
            json_data = {
                "image_path": image_path,
                "r_ticks": generated_r_ticks,
                "pred_coords": [int(self.coords[0]), int(self.coords[1])],
                "argument": argument,
                "scale_source": scale_source,
            }
            if scale_source == "simulated_polygon_default":
                json_data["scale_note"] = "No readable radial numeric tick labels were detected; a default 0-100 mapping was used for encryption."
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"JSON saved to: {output_json_path}")


            # os.remove(temp_output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Operation failed: {e}")
            return None


if __name__ == "__main__":

    encoder = RadarChartEncoder()
    

    image_path = "./data/upload/radar_000.png"
    output_dir = "./data/output/radar"
    

    result_path = encoder.process_single_image(image_path, output_dir)
    
    if result_path:
        print(f"Processing complete. Encrypted image saved to: {result_path}")
    else:
        print("Processing failed")
