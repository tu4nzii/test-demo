import numpy as np
import math
import json
import cv2
import re
import requests
import base64
import os

from model_api_config import get_chat_completion_url, get_headers, get_model_name

class RadarChartAxisFinder:
    def __init__(self):

        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        

        self.output_dir = "./data/output/radar"
        self.axes_output_dir = os.path.join(self.output_dir)
        

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.axes_output_dir, exist_ok=True)
        

        self.axes_angles = []
        self.axis_labels = {}
        self.center = [0, 0]
        self.radius = 0

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

    def crop_axis_label_region(self, image_path, center_x, center_y, angle_deg, radius, 
                              label_offset=100, label_width=150, label_height=150):
        """Crop an axis-label region by angle."""

        image = cv2.imread(image_path)
        angle_rad = math.radians(angle_deg)
        

        label_center_x = int(center_x + (radius + label_offset) * math.cos(angle_rad))
        label_center_y = int(center_y - (radius + label_offset) * math.sin(angle_rad))
        

        x1 = max(0, label_center_x - label_width // 2)
        y1 = max(0, label_center_y - label_height // 2)
        x2 = min(image.shape[1], label_center_x + label_width // 2)
        y2 = min(image.shape[0], label_center_y + label_height // 2)
        

        return image[y1:y2, x1:x2]

    def call_llm_letter(self, crop_img):
        """Use the LLM to recognize letters in an image."""

        image_area = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
        if image_area.dtype != np.uint8:
            image_area = image_area.astype(np.uint8)
            print("Converted image data to uint8")
            

        success, encoded_image = cv2.imencode('.jpg', image_area)
        if not success:
            print("Image encoding failed")
            return None
            
        image_data = np.ascontiguousarray(encoded_image)
        base64_image = base64.b64encode(image_data).decode('utf-8')
        

        prompt = f"""
You are given a cropped image region from a radar/rose chart axis label area.
Read the central visible axis label text only.

Return strict JSON only:
{{
  "letter": <string_or_null>
}}

Use null if the label cannot be read clearly.
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

    def call_llm_nums(self, image_path: str):
        """Use the LLM to recognize axis count and names."""
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            

        prompt = f"""
You are analyzing a radar/rose chart image.
Identify the visible axis labels around the chart and estimate how many named axes are present.
Count only axes with visible text labels near their outer end. Do not count unlabeled radial/grid helper lines.
If possible, determine whether labels are near the center of a sector or on a sector boundary.

Return strict JSON only:
{{
  "axis_name": [<label_1>, <label_2>, ...],
  "nums": <integer_or_null>,
  "position": <"center"|"edge"|null>,
  "reason": "brief visual reasoning"
}}

Use null for fields that cannot be recognized.
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
            print(f"API request error: {e}")
            print(f"Response content: {response.text if 'response' in locals() else 'no response'}")
            return None

    def get_start_angle(self, image_path: str, center_x: int, center_y: int, radius: int):
        """Get the start angle."""
        pred_angle = [0, 90]
        start_angle = []
        
        for angle in pred_angle:
            image_area = self.crop_axis_label_region(image_path, center_x, center_y, angle, radius)
            data = self.call_llm_letter(image_area)
            if data and data["letter"] != 'None' and data["letter"] is not None:
                start_angle.append(angle)
                
        print(f"Starting angle: {start_angle}")
        if start_angle:
            return start_angle[0]
        else:
            return None

    def find_radar_axes(self, image_path: str, center, start_angle: int, max_radius):
        """Detect radar axis angles."""
        img = cv2.imread(image_path)
        axes_angles = []
        

        axes = self.call_llm_nums(image_path)
        print(f"LLM recognition result: {axes}")
        
        if not axes or 'nums' not in axes:
            print("Unable to recognize axis count")
            return []
            
        axes_nums = axes['nums']
        

        for i in range(axes_nums):
            best_interval = int(360 / axes_nums)
            angle = round(start_angle + i * best_interval) % 360
            axes_angles.append(angle)
        

        output_img = img.copy()
        for angle in axes_angles:
            current_angle_rad = math.radians(angle)
            

            end_x = int(center[0] + max_radius * math.cos(current_angle_rad))
            end_y = int(center[1] + max_radius * math.sin(current_angle_rad))
            

            cv2.line(output_img, center, (end_x, end_y), (0, 0, 255), 1)
            

            cv2.circle(output_img, (end_x, end_y), 2, (0, 255, 0), -1)
        
        print(f"Detected axis angles: {axes_angles}")
        

        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)
        # output_path = os.path.join(self.axes_output_dir, f"axes_detected_{file_name}{file_ext}")
        # cv2.imwrite(output_path, output_img)

        
        self.radar_axes_angles = axes_angles
        return axes_angles

    def recognize_radar_axis_labels(self, image_path: str, center, radius, axes_angles):
        """Recognize labels for each axis."""
        axis_labels = {}
        
        for axis in axes_angles:
            try:
                crop_img = self.crop_axis_label_region(image_path, center[0], center[1], axis, radius)
                axis_data = self.call_llm_letter(crop_img)
                

                if isinstance(axis_data, dict) and 'letter' in axis_data and axis_data['letter'] is not None:
                    letter = axis_data['letter']
                    axis_labels[axis] = letter
                    print(f"Axis angle {axis}, recognition result: {letter}")
                else:
                    print(f"Axis angle {axis}, invalid recognition result: {axis_data}")
            except Exception as e:
                print(f"Failed to process axis angle {axis}: {str(e)}")
                continue
        
        self.axis_labels = axis_labels
        return axis_labels

    def process_single_image(self, image_path, center=None, radius=None, output_json_path=None):
        """Process one radar chart and recognize axes and labels."""
        try:

            if center is None or radius is None:

                base_name = os.path.basename(image_path)
                file_name, _ = os.path.splitext(base_name)
                json_path = os.path.join(self.output_dir, f"{file_name}.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        
                    if center is None and 'pred_coords' in json_data:
                        center = json_data['pred_coords']
                        print(f"Loaded center from JSON file: {center}")
                        
                    if radius is None and 'argument' in json_data and 'r_ticks' in json_data:
                        a = json_data['argument']['a']
                        b = json_data['argument']['b']
                        max_tick = json_data['r_ticks'][-1]
                        radius = a * max_tick + b - 5
                        print(f"Computed radius: {radius}")
                        

            if center is None:
                raise ValueError("Center was not provided and could not be read from JSON")
            if radius is None:
                raise ValueError("Radius was not provided and could not be computed from JSON")
            
            self.center = center
            self.radius = radius
            

            start_angle = self.get_start_angle(image_path, center[0], center[1], radius)
            if start_angle is None:
                print("Unable to determine starting angle")
                return None
            
            print(f"Starting angle: {start_angle}")
            

            found_axes = self.find_radar_axes(image_path, center, start_angle, radius)
            if not found_axes:
                print("No axes found")
                return None
            

            axis_labels = self.recognize_radar_axis_labels(image_path, center, radius, found_axes)
            

            result = {
                'image_path': image_path,
                'center': center,
                'radius': radius,
                'start_angle': start_angle,
                'axes_angles': found_axes,
                'axis_labels': axis_labels
            }
            

            base_name = os.path.basename(image_path)
            file_name, _ = os.path.splitext(base_name)
            
            if output_json_path is None:
                output_json_path = os.path.join(self.output_dir, f"{file_name}_axes.json")
                
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"JSON saved to: {output_json_path}")
            

            original_json_path = os.path.join(self.output_dir, f"{file_name}.json")
            if os.path.exists(original_json_path):
                with open(original_json_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                    

                original_data['axis_labels'] = axis_labels
                original_data['axes_angles'] = found_axes
                original_data['start_angle'] = start_angle
                

                with open(original_json_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, ensure_ascii=False, indent=2)
                    
                print(f"Original JSON file updated: {original_json_path}")
            
            return result
            
        except Exception as e:
            print(f"Operation failed: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":

    finder = RadarChartAxisFinder()
    

    # finder.output_dir = "custom_output"
    # finder.axes_output_dir = os.path.join(finder.output_dir, "custom_axes")
    

    image_path = "./data/upload/radar_001.png"
    




    # result = finder.process_single_image(image_path, center, radius)
    

    result = finder.process_single_image(image_path)
    
    if result:
        print(f"Processing complete. Axis angles: {result['axes_angles']}")
        print(f"Axis labels: {result['axis_labels']}")
    else:
        print("Processing failed")
