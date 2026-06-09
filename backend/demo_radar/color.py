import numpy as np
import json
import cv2
import re
import requests
import base64
import os

from model_api_config import get_chat_completion_url, get_headers, get_model_name

class RadarColorMatcher:
    """RadarColorMatcher helper."""
    def __init__(self):

        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        

        self.output_dir = "./data/output/radar"
        os.makedirs(self.output_dir, exist_ok=True)
        

        self.entity_colors = {}
        

        self.min_block_area = 30
        self.max_block_area = 1000
        self.color_diff_threshold = 30
        self.min_saturation = 30
        self.min_value = 50

    def parse_json(self, content: str):
        """Parse json."""
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            return json.loads(match.group(1))
        except Exception as e:
            print(f"JSON parse failed: {e}")
            return None

    def load_image(self, image_path):
        """Load image."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        return image

    def crop_legend(self, image, ratio=0.3, scale=2):
        """Crop legend."""
        height, width = image.shape[:2]

        crop_region = image[:int(height*ratio), :int(width*ratio)]
        

        new_height = int(crop_region.shape[0] * scale)
        new_width = int(crop_region.shape[1] * scale)
        return cv2.resize(crop_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def image_to_base64(self, image):
        """Image to base64."""

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
            print("Converted image data to uint8")
            

        success, encoded = cv2.imencode('.jpg', rgb_image)
        if not success:
            print("Image encoding failed")
            return None
            
        return base64.b64encode(np.ascontiguousarray(encoded)).decode('utf-8')

    def detect_legend(self, base64_image):
        """Detect legend."""
        prompt = """
Analyze this radar chart and locate the legend area.
The legend usually contains entity names and color markers. Include the full legend, not just part of it.

Return strict JSON only:
{
  "position": [x, y],
  "range": [w, h]
}

position is the legend center in pixels, and range is width/height in pixels. Use null if the legend cannot be located.
"""
        
        payload = {
            "model": self.model_name,
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
                print(f"Unable to extract valid legend location: {content}")
                return None
                
        except Exception as e:
            print(f"API request failed: {e}")
            if 'response' in locals():
                print(f"Response content: {response.text}")
            return None
    
    def auto_crop_legend(self, image, scale=2):
        """Auto crop legend."""

        base64_image = self.image_to_base64(image)
        if base64_image is None:
            print("Warning: image conversion failed; using default legend crop")
            return self.crop_legend(image, scale=scale)
        

        print("Detecting legend location...")
        legend_info = self.detect_legend(base64_image)
        

        height, width = image.shape[:2]
        

        if legend_info is None or not self._validate_legend_info(legend_info, width, height):
            print("Warning: legend detection failed; using default crop")
            return self.crop_legend(image, scale=scale)
        

        center_x, center_y = legend_info["position"]
        region_width, region_height = legend_info["range"]
        

        margin = int(min(region_width, region_height) * 0.1)
        x1 = max(0, int(center_x - region_width / 2) - margin)
        y1 = max(0, int(center_y - region_height / 2) - margin)
        x2 = min(width, int(center_x + region_width / 2) + margin)
        y2 = min(height, int(center_y + region_height / 2) + margin)
        
        print(f"Legend crop: ({x1}, {y1}) to ({x2}, {y2})")
        

        crop_region = image[y1:y2, x1:x2]
        

        if crop_region.size == 0 or crop_region.shape[0] < 50 or crop_region.shape[1] < 50:
            print("Warning: legend crop is too small; using default crop")
            return self.crop_legend(image, scale=scale)
        

        if not self.extract_colors(crop_region):
            print("Warning: detected legend crop has no color markers, falling back to default crop")
            return self.crop_legend(image, scale=scale)

        new_height = int(crop_region.shape[0] * scale)
        new_width = int(crop_region.shape[1] * scale)
        return cv2.resize(crop_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def _validate_legend_info(self, legend_info, width, height):
        """ validate legend info."""
        try:

            position = legend_info.get("position", [])
            if not isinstance(position, list) or len(position) != 2:
                return False
            
            x, y = position
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            

            region_range = legend_info.get("range", [])
            if not isinstance(region_range, list) or len(region_range) != 2:
                return False
            
            w, h = region_range
            if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
                return False
            

            if w <= 0 or h <= 0 or w > width or h > height:
                return False
            
            if x < 0 or x > width or y < 0 or y > height:
                return False
                
            return True
        except:
            return False
    
    def extract_colors(self, image):
        """Extract colors."""

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, self.min_saturation, self.min_value])
        upper_bound = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        

        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
        color_mask = cv2.dilate(color_mask, kernel, iterations=2)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
        

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        

        block_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            

            if (self.min_block_area < area < self.max_block_area and 
                0.3 < aspect_ratio < 3.0):

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_color = cv2.mean(image, mask=mask)[:3]  # BGR
                

                bgr_color = np.uint8([[mean_color]])
                hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
                if hsv_color[1] >= self.min_saturation:

                    center_x = x + w // 2
                    center_y = y + h // 2
                    

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
        

        unique_color_info = []
        for block in block_info:
            is_unique = True
            current_color = block['color']
            
            for unique_block in unique_color_info:
                unique_color = unique_block['color']

                hsv1 = cv2.cvtColor(np.uint8([[current_color]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv2 = cv2.cvtColor(np.uint8([[unique_color]]), cv2.COLOR_BGR2HSV)[0][0]
                

                h_diff = min(abs(int(hsv1[0]) - int(hsv2[0])), 180 - abs(int(hsv1[0]) - int(hsv2[0])))
                s_diff = abs(int(hsv1[1]) - int(hsv2[1]))
                v_diff = abs(int(hsv1[2]) - int(hsv2[2]))
                

                weighted_distance = int(h_diff) * 2 + int(s_diff) + int(v_diff)
                
                if weighted_distance <= self.color_diff_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_color_info.append(block)
        

        if unique_color_info:
            hsv_color_info = []
            for block in unique_color_info:
                hsv = cv2.cvtColor(np.uint8([[block['color']]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv_color_info.append((hsv[0], block))
            
            hsv_color_info.sort(key=lambda x: x[0])
            unique_color_info = [block for _, block in hsv_color_info]
        
        return unique_color_info
    
    def bgr_to_hex(self, bgr_color):
        """Bgr to hex."""
        rgb_color = bgr_color[::-1]  # BGR -> RGB
        return f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}".upper()
    
    def match_entities_colors(self, base64_image, entity_names, color_info_list):
        """Match entities colors."""

        color_with_positions = []
        hex_colors = []
        
        for i, color_info in enumerate(color_info_list):
            hex_color = self.bgr_to_hex(color_info['color'])
            hex_colors.append(hex_color)
            pos = color_info['position']
            color_with_positions.append(
                f"Color {i+1}: {hex_color} (top-left {pos['x']}, {pos['y']}, center {pos['center_x']}, {pos['center_y']})"
            )
        
        prompt = f"""
Analyze this radar chart legend and match each provided entity name to one extracted color.

Entity names:
{', '.join(entity_names)}

Extracted colors with positions:
{'; '.join(color_with_positions)}

Return strict JSON only as an object mapping entity names to #RRGGBB colors:
{{
  "Entity name": "#RRGGBB"
}}

Every entity should receive the most plausible matching color. Do not invent new entity names.
"""
        
        payload = {
            "model": self.model_name,
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
            print(f"API request failed: {e}")
            if 'response' in locals():
                print(f"Response content: {response.text}")
            return None

    def extract_legend_series_colors(self, image_path, use_auto_crop=True):
        """Extract radar legend series names and colors from the image only."""
        try:
            image = self.load_image(image_path)
            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            crop_attempts = ["default"]
            if use_auto_crop:
                crop_attempts.append("auto")

            best_output = None

            for crop_method in crop_attempts:
                legend_image = self.auto_crop_legend(image) if crop_method == "auto" else self.crop_legend(image)
                legend_path = os.path.join(self.output_dir, f"legend_{file_name}{file_ext}")
                if crop_method != "auto":
                    legend_path = os.path.join(self.output_dir, f"legend_{crop_method}_{file_name}{file_ext}")
                cv2.imwrite(legend_path, legend_image)

                base64_image = self.image_to_base64(legend_image)
                if base64_image is None:
                    continue

                color_info_list = self.extract_colors(legend_image)
                color_info_list.sort(
                    key=lambda item: (
                        item.get("position", {}).get("center_y", 0),
                        item.get("position", {}).get("center_x", 0),
                    )
                )
                color_candidates = []
                for index, color_info in enumerate(color_info_list):
                    pos = color_info.get("position", {})
                    color_candidates.append(
                        {
                            "index": index + 1,
                            "hex": self.bgr_to_hex(color_info["color"]),
                            "center": [pos.get("center_x"), pos.get("center_y")],
                        }
                    )

                prompt = f"""
Analyze this radar chart legend crop. Extract every visible legend item and pair
each series name with its displayed color.

Detected color candidates from image processing:
{json.dumps(color_candidates, ensure_ascii=False)}

Rules:
- Use only information visible in this legend crop.
- Do not use any dataset JSON or ground truth.
- Preserve the legend order from top to bottom.
- Return strict JSON only.
- Use hex colors in #RRGGBB format. If a candidate color matches the legend
  marker, prefer that candidate hex value.

Expected JSON shape:
{{
  "series_color": {{
    "Series name 1": "#RRGGBB",
    "Series name 2": "#RRGGBB"
  }}
}}
"""
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "temperature": 0.0,
                }

                response = requests.post(url=self.url, headers=self.headers, json=payload, timeout=180)
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                parsed = self.parse_json(content)
                series_color = parsed.get("series_color") if isinstance(parsed, dict) else None
                if not isinstance(series_color, dict):
                    series_color = parsed if isinstance(parsed, dict) else None
                if not isinstance(series_color, dict):
                    continue

                normalized = {}
                for name, color in series_color.items():
                    name_text = str(name or "").strip()
                    color_text = str(color or "").strip()
                    if not name_text:
                        continue
                    match = re.search(r"#?[0-9a-fA-F]{6}", color_text)
                    if not match:
                        continue
                    hex_color = match.group(0).upper()
                    if not hex_color.startswith("#"):
                        hex_color = f"#{hex_color}"
                    normalized[name_text] = hex_color

                if not normalized:
                    continue

                if color_candidates and len(normalized) == len(color_candidates):
                    normalized = {
                        name: color_candidates[index]["hex"]
                        for index, name in enumerate(normalized.keys())
                    }

                output = {
                    "image_path": image_path,
                    "series_color": normalized,
                    "entity_colors": normalized,
                    "extracted_colors": [item["hex"] for item in color_candidates],
                    "legend_path": legend_path,
                    "crop_method": crop_method,
                }
                best_output = best_output or output

                has_black_placeholder = (
                    not color_candidates
                    and all(str(color).upper() == "#000000" for color in normalized.values())
                )
                too_few_series = len(color_candidates) >= 2 and len(normalized) < len(color_candidates)
                if has_black_placeholder or too_few_series:
                    continue

                output_path = os.path.join(self.output_dir, f"{file_name}_colors.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                return output

            if best_output:
                output_path = os.path.join(self.output_dir, f"{file_name}_colors.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(best_output, f, ensure_ascii=False, indent=2)
                return best_output
            return None
        except Exception as e:
            print(f"Radar legend series-color extraction failed: {e}")
            return None
    
    def process_image(self, image_path, use_auto_crop=True, entity_names=None):
        """Process image."""
        try:

            print(f"Reading image: {image_path}")
            image = self.load_image(image_path)
            

            print("Cropping legend area...")
            legend_image = self.auto_crop_legend(image) if use_auto_crop else self.crop_legend(image)
            

            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            legend_path = os.path.join(self.output_dir, f"legend_{file_name}{file_ext}")
            print(f"Saving legend image: {legend_path}")
            cv2.imwrite(legend_path, legend_image)
            

            print("Converting image format...")
            base64_image = self.image_to_base64(legend_image)
            if base64_image is None:
                raise ValueError("Image conversion failed")
            

            print("Extracting color blocks...")
            color_info_list = self.extract_colors(legend_image)
            hex_colors = [self.bgr_to_hex(info['color']) for info in color_info_list]
            print(f"Extracted {len(hex_colors)} colors: {', '.join(hex_colors)}")
            

            if entity_names is None:
                print("Error: entity name list is required")
                return None
            else:
                print(f"Using entity names: {', '.join(entity_names)}")
            

            print("Matching entities to colors...")
            entity_colors = self.match_entities_colors(base64_image, entity_names, color_info_list)
            

            if entity_colors is None:
                print("Color matching failed; using default order mapping")
                entity_colors = {}
                for i, entity in enumerate(entity_names):
                    if i < len(color_info_list):
                        color_info = color_info_list[i]
                        entity_colors[entity] = self.bgr_to_hex(color_info['color'])
                    else:
                        entity_colors[entity] = "#000000"
            
            self.entity_colors = entity_colors
            

            result = {
                'image_path': image_path,
                'entity_colors': entity_colors,
                'extracted_colors': hex_colors,
                'legend_path': legend_path,
                'crop_method': 'auto' if use_auto_crop else 'default'
            }
            

            output_path = os.path.join(self.output_dir, f"{file_name}_colors.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"Output saved to: {output_path}")
            print("\nRecognition results:")
            print(json.dumps(entity_colors, ensure_ascii=False, indent=2))
            
            return result
            
        except Exception as e:
            print(f"Operation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Manual smoke test."""

    matcher = RadarColorMatcher()
    


    

    image_path = r"d:/home work/Agent.paper/test demo/backend/data/upload/radar_001.png"
    

    # entity_names = ["WDULR", "ZTJUP", "QCBOR", "RFLDM", "UCKIV"]
    entity_names =["LMIEXG","KBGCVO","AZC","OAAKCP"]

    result = matcher.process_image(image_path, entity_names=entity_names)
    
    if result:
        print(f"\nProcessing complete: {len(result['entity_colors'])} entities")
        print("\nEntity-color mapping:")
        for entity, color in result['entity_colors'].items():
            print(f"{entity}: {color}")
    else:
        print("Processing failed")


if __name__ == "__main__":
    main()
