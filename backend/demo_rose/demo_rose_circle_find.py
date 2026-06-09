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

class RoseChartEncoder:
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
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                 param1=20, param2=30, minRadius=int(height/4), maxRadius=int(height))
        
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

    def second_circle_find(self, image):
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

    def encrypt_rose_chart_with_tick(self, image_path, tick_interval, tick1, tick2, max_tick_value, min_tick_value):
        """Encrypt the grid using two tick values."""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        

        r1, r2 = (self.first_r, self.second_r) if self.second_r > self.first_r else (self.second_r, self.first_r)
        interval = tick_interval / self.tick_density
        

        a = r1 / max_tick_value
        b = r1 - a * max_tick_value
        
        result = image.copy()
        

        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        image_area = height * width
        scale = math.sqrt(image_area)
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
        

        if min_tick_value > 0:
            pixels_per_value = float(r) / (tick1 - (min_tick_value - tick_interval))
        else:
            pixels_per_value = float(r) / tick1
            
        a = pixels_per_value 
        b = r - a * tick1
        result = image.copy()
        

        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        image_area = height * width
        scale = math.sqrt(image_area)
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

    def process_single_image(self, image_path, output_dir=None):
        """Process one rose chart through the full encryption flow."""
        try:

            if output_dir is None:
                output_dir = os.path.dirname(image_path) or '.'
                
            os.makedirs(output_dir, exist_ok=True)
            

            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            

            second_circle = self.visualize_ring_mask(image_path)
            self.second_circle_find(second_circle)
            
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
            

            result_1 = self.find_tick(self.first_r, temp_output_path)
            tick1 = result_1.get("tick")
            reason1 = result_1.get("res")
            
            result_2 = self.find_tick(self.second_r, temp_output_path)
            tick2 = result_2.get("tick")
            reason2 = result_2.get("res")
            
            print(f"First circle tick reasoning: {reason1}")
            print(f"First circle tick value: {tick1}")
            print(f"Second circle tick reasoning: {reason2}")
            print(f"Second circle tick value: {tick2}")
            

            response_data = self.call_llm_response(temp_output_path)
            max_tick_value = response_data.get("max_tick_value")
            min_tick_value = response_data.get("min_tick_value")
            tick_interval = response_data.get("tick_interval")
            res = response_data.get("res")
            
            print(f"LLM analysis: tick1={tick1}, tick2={tick2}, max={max_tick_value}, {res}")
            

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
                print("No valid tick values recognized")
                return None
            

            output_path = os.path.join(output_dir, f"{file_name}_encode{file_ext}")
            cv2.imwrite(output_path, result)
            print(f"Output saved to: {output_path}")
            

            json_fname = f"{file_name}.json"
            json_path = os.path.join(os.path.dirname(image_path), json_fname)
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    

                r_ticks = r_ticks[::-1]
                r_ticks.append(r_ticks[-1] + (tick_interval / self.tick_density))
                json_data['r_ticks'] = r_ticks
                

                pred_coords = [int(self.coords[0]), int(self.coords[1])]
                

                try:
                    if 'center' in json_data:
                        if isinstance(json_data['center'], dict):
                            real_coords = [json_data['center']['x'], json_data['center']['y']]
                        else:
                            real_coords = json_data['center']
                        

                        err_center = np.linalg.norm(np.array(pred_coords) - np.array(real_coords))
                        json_data['err_center'] = err_center
                except Exception as e:
                    print(f"Operation failed: {e}")
                    
                json_data['pred_coords'] = pred_coords
                json_data['argument'] = argument
                

                output_json_path = os.path.join(output_dir, json_fname)
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                    
                print(f"JSON saved to: {output_json_path}")
            else:
                print(f"JSON file not found: {json_path}")
                

            # os.remove(temp_output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Operation failed: {e}")
            return None


if __name__ == "__main__":

    encoder = RoseChartEncoder()
    

    image_path = "./data/rose/rose_001.png"
    output_dir = "./data/output/rose"
    

    result_path = encoder.process_single_image(image_path, output_dir)
    
    if result_path:
        print(f"Processing complete. Encrypted image saved to: {result_path}")
    else:
        print("Processing failed")
