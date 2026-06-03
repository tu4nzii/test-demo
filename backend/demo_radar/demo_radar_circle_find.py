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
        # 閰嶇疆鍙傛暟
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        self.tick_density = 2
        
        # 澶勭悊缁撴灉瀛樺偍
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
        scale = min(scale_width, scale_height, 1.0)  # 涓嶆斁澶у浘鍍?
        
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
            print(f"JSON瑙ｆ瀽澶辫触: {e}")
            return None

    def visualize_ring_mask(self, image_path, ring_width=5):
        """Create an annular mask and return the processed image."""
        # 璇诲彇骞堕澶勭悊鍥惧儚
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 闇嶅か鍙樻崲妫€娴嬬涓€涓渾
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                 param1=20, param2=30, minRadius=int(height/5), maxRadius=int(height/4))
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            first_circle = circles[0, 0]
            cx, cy, r = first_circle[0], first_circle[1], first_circle[2]
            
            # 鍒涘缓鐜姸鎺╄啘
            mask = np.zeros_like(gray)
            cv2.circle(mask, (cx, cy), int(r+ring_width), 255, -1)
            cv2.circle(mask, (cx, cy), int(r-ring_width), 0, -1)
            
            # 搴旂敤鐜姸鎺╄啘
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
        
        # 鍒涘缓涓庡師鍥剧浉鍚屽ぇ灏忕殑鎺╃爜
        mask = np.zeros_like(image)
        
        # 璁＄畻鍐呭鍦嗗崐寰勶紙纭繚鍐呭渾鍗婂緞涓嶄负璐燂級
        outer_radius = target_radius + pixel_range
        inner_radius = max(0, target_radius - 10)
        
        # 缁樺埗鐜舰鎺╃爜锛堢櫧鑹蹭负淇濈暀鍖哄煙锛?
        cv2.circle(mask, (center_x, center_y), outer_radius, (255, 255, 255), -1)
        cv2.circle(mask, (center_x, center_y), inner_radius, (0, 0, 0), -1)
        
        # 搴旂敤鎺╃爜鑾峰彇鐜舰鍖哄煙
        masked_image = cv2.bitwise_and(image, mask)
        
        # 瑁佸壀鏈€灏忓鎺ョ煩褰互鍘婚櫎澶氫綑榛戣壊鍖哄煙
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 鑾峰彇鏈€澶ц疆寤撶殑杈圭晫妗?
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return masked_image[y:y+h, x:x+w]
        
        # 濡傛灉娌℃湁鎵惧埌鏈夋晥鍖哄煙锛岃繑鍥炲師濮嬫帺鐮佸浘鍍?
        return masked_image

    def find_tick(self, target_radius, image_path):
        """Use the LLM to recognize the tick value near a radius."""
        center_x, center_y = self.coords
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"鏃犳硶璇诲彇鍥惧儚: {image_path}")
            
        cropped_image = self.crop_tick_region(image, target_radius, pixel_range=25)
        
        # 杞崲涓篟GB骞剁紪鐮佷负base64
        if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            
        retval, buffer = cv2.imencode('.png', rgb_image)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        prompt = f"""
        杩欐槸涓€寮犲渾鐜浘
        鍥剧墖涓凡缁忕敤缁胯壊鍦嗗湀鏍囧嚭浜嗕竴涓噸瑕佺殑鍦?
        璇锋偍鍒嗘瀽鍥剧墖鍐呭锛屽苟鎻愪緵璇ヤ俊鎭細
        1. 杩欎釜**缁胯壊鍦嗗湀**瀵瑰簲鐨勭殑鍒诲害鍊兼槸澶氬皯锛燂紙浼氬嚭鐜板湪璇ュ渾鍛ㄥ洿锛岃鍒诲害鍊间粎涓轰竴涓暟鍊?濡?0% = 50)锛屼笖浠呭嚭鐜板湪鍥剧墖涓級
        **鍙鍙栧瓨鍦ㄧ殑鏁?*
        **娉ㄦ剰,浠呰鍙栧浘涓婂師鏈殑鏁板€硷紝鑰屼笉鍋氫换浣曟帹绠?*
        **浠旂粏妫€鏌ュ浘鐗囷紝纭繚璇诲彇鐨勬暟鍊兼槸姝ｇ‘鐨?*
        **涓嶅瓨鍦?00锛侊紒锛佽嫢璇嗗埆涓?00锛屽垯鏄瘑鍒敊璇紝瀹為檯涓?00**

        璇蜂互涓ユ牸鐨?JSON 鏍煎紡杩斿洖杩欎簺淇℃伅锛屼笉瑕佸寘鍚换浣曢澶栨枃瀛楁垨瑙ｉ噴锛屼緥濡傦細
        ```json
        {{
            "tick": <鍒诲害鍊?,
            "res":<鍒嗘瀽杩囩▼>
        }}
        ```
        **濡傛灉鏈夊涓暟瀛楋紝浣跨敤null**
        濡傛灉鏃犳硶璇嗗埆鏌愪釜鍊硷紝璇蜂娇鐢?`null`銆?
        **鍐嶆澹版槑锛屽鏋滄病鏈夋暟瀛楋紝鍒欎娇鐢╪ull**
        **鍐嶆澹版槑锛屽鏋滄湁澶氫釜鏁板瓧锛屼娇鐢╪ull**
        **鑻ュ寘鍚瓧姣嶏紝鍒欎负null**
        **鑻ヤ负0鍒欎负null**
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
            print(f"API璇锋眰澶辫触: {e}")
            return None

    def call_llm_response(self, image_path):
        """Use the LLM to extract chart tick information."""
        center_x, center_y = self.coords
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        prompt = f"""
        杩欐槸涓€寮犳瀬鍧愭爣鍥撅紙渚嬪闆疯揪鍥炬垨鏋佸潗鏍囨暎鐐瑰浘锛夈€?
        鍏朵腑蹇冨ぇ鑷村湪 ({center_x}, {center_y}) 鍍忕礌浣嶇疆銆?


        璇锋偍鍒嗘瀽鍥剧墖鍐呭锛屽苟鎻愪緵浠ヤ笅淇℃伅:
        1. 鍥捐〃涓墍鏈?*鍚屽績鍦嗗埢搴︾嚎涓紝鍗虫渶澶栧湀鐨勫埢搴︼紝鏈€澶х殑鍒诲害鍊兼槸澶氬皯锛?*
        2. 鍥捐〃涓墍鏈?*鍚屽績鍦嗗埢搴︾嚎涓紝鍗虫渶澶栧湀鐨勫埢搴︼紝鏈€灏忕殑鍒诲害鍊兼槸澶氬皯锛?*
        3. 鍥捐〃涓墍鏈?*鍚屽績鍦嗗埢搴︾殑闂撮殧锛屽埢搴︿负寰勫悜鍒诲害锛岃€岄潪鐜姸鍒诲害**

        **娉ㄦ剰,浠呰鍙栧浘涓婂師鏈殑鏁板€硷紝鑰屼笉鍋氫换浣曟帹绠?*
        涓斾粎杩斿洖鏁板€硷紝濡傦紙50% = 50锛?

        璇蜂互涓ユ牸鐨?JSON 鏍煎紡杩斿洖杩欎簺淇℃伅锛屼笉瑕佸寘鍚换浣曢澶栨枃瀛楁垨瑙ｉ噴锛屼緥濡傦細
        ```json
        {{
            "max_tick_value": <鏈€澶у埢搴﹀€?
            "min_tick_value": <鏈€灏忓埢搴﹀€?
            "tick_interval": <鍒诲害闂撮殧>
            "res":<鍒嗘瀽杩囩▼>
        }}
        璇ュ浘鐨勬渶澶у埢搴︿负100锛屾渶灏忓埢搴︿负50 闂撮殧涓?0
        ```
        濡傛灉鏃犳硶璇嗗埆鏌愪釜鍊硷紝璇蜂娇鐢?`null`銆備緥濡傦紝濡傛灉 `max_tick_value` 鏄?锛岃杩斿洖`0`銆?
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
        
        # 纭繚r2 > r1
        r1, r2 = (self.first_r, self.second_r) if self.second_r > self.first_r else (self.second_r, self.first_r)
        interval = tick_interval / self.tick_density
        # Radar charts use the center as radial value 0. Avoid shifting the
        # origin when the MLLM misses an inner visible tick label.
        a = float(r2) / max(float(tick2), 1e-6)

        # 璁＄畻鍗婂緞涓庡埢搴︾殑绾挎€у叧绯?(r = a*tick + b)
        b = 0
        
        result = image.copy()
        
        # 璁剧疆瀛椾綋鍜岄鑹?
        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        image_area = height * width
        scale = math.sqrt(image_area)
        font_scale = scale * 0.006
        font_color = (0, 0, 0)  # 榛戣壊
        line_color = (128, 128, 128)
        thickness = 1
        
        # 缁樺埗鍔犲瘑鍦嗙幆骞舵爣娉ㄥ埢搴?
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
                
            # 璁＄畻褰撳墠鍒诲害瀵瑰簲鐨勫崐寰?
            radius = int(a * tick + b)
            current_px_distance = abs(radius - 0)
            
            if radius <= 0:
                print(f"璺宠繃鏃犳晥鍗婂緞: tick={tick}, 璁＄畻鍗婂緞={radius}")
                continue
                
            if current_px_distance <= 3:
                print(f"宸茶揪鍒板渾蹇冮檮杩戯紝鍋滄缁樺埗 (tick={tick}, radius={radius})")
                break
                
            text_x_up = center_x 
            text_y_up = center_y - radius 

            # 缁樺埗鍒诲害鏍囨敞
            if tick % 1 == 0:
                tick = int(tick)
                
            if tick > 0:
                pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 璁剧疆瀛椾綋
                try:
                    font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)
                # 鑾峰彇鏂囨湰杈圭晫妗?
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 鍙充晶鏃嬭浆90搴︽枃鏈?
                temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((0, 0), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)  # 椤烘椂閽堟棆杞?0搴?
                
                # 璋冩暣鍙充晶浣嶇疆
                pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                
                # 宸︿晶鏃嬭浆-90搴︽枃鏈?
                temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((0, 0), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)  # 閫嗘椂閽堟棆杞?0搴?
                
                # 璋冩暣宸︿晶浣嶇疆
                pos_left = (center_x - radius - int(width*0.0125), 
                        center_y - rotated_left.size[1]//2)
                
                # 搴曢儴姝ｅ父鏂囨湰
                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius - int(height*0.0122) 
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                
                # 杞崲鍥濷penCV鏍煎紡
                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            # 淇濆瓨鐢熸垚鐨勫埢搴﹀€?
            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1, lineType=cv2.LINE_AA)
            
            # 缁樺埗铏氱嚎鍦嗙幆
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
        
        # 璁＄畻鍗婂緞涓庡埢搴︾殑绾挎€у叧绯?
        # Radar charts use the center as radial value 0. Avoid shifting the
        # origin when the MLLM misses an inner visible tick label.
        a = float(r) / max(float(tick1), 1e-6)
        b = 0
        result = image.copy()
        
        # 璁剧疆瀛椾綋鍜岄鑹?
        font_CV = cv2.FONT_HERSHEY_DUPLEX 
        image_area = height * width
        scale = math.sqrt(image_area)
        font_scale = scale * 0.006
        font_color = (0, 0, 0)  # 榛戣壊
        line_color = (128, 128, 128)
        thickness = 1
        
        # 缁樺埗鍔犲瘑鍦嗙幆骞舵爣娉ㄥ埢搴?
        tick = max_tick_value
        r_ticks = []
        self.argument = {'a': a, 'b': b}
        
        current_px_distance = 10000
        radius = 0
        count = 0
        
        while tick > 0 and radius >= 0:
            tick -= interval
            # 璁＄畻褰撳墠鍒诲害瀵瑰簲鐨勫崐寰?
            radius = int(a * tick + b)
            current_px_distance = abs(radius - 0)
            
            if radius <= 0:
                print(f"璺宠繃鏃犳晥鍗婂緞: tick={tick}, 璁＄畻鍗婂緞={radius}")
                continue
                
            if current_px_distance <= 3:
                print(f"宸茶揪鍒板渾蹇冮檮杩戯紝鍋滄缁樺埗 (tick={tick}, radius={radius})")
                break
                
            text_x_up = center_x 
            text_y_up = center_y - radius 

            # 缁樺埗鍒诲害鏍囨敞
            if tick % 1 == 0:
                tick = int(tick)
                
            if tick > 0:
                pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_img)
                
                # 璁剧疆瀛椾綋
                try:
                    font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))
                except IOError:
                    font = ImageFont.load_default()
                    
                text = str(tick)
                # 鑾峰彇鏂囨湰杈圭晫妗?
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 鍙充晶鏃嬭浆90搴︽枃鏈?
                temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_right = ImageDraw.Draw(temp_img_right)
                temp_draw_right.text((0, 0), text, font=font, fill=font_color)
                rotated_right = temp_img_right.rotate(-90, expand=True)  # 椤烘椂閽堟棆杞?0搴?
                
                # 璋冩暣鍙充晶浣嶇疆
                pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                            center_y - rotated_right.size[1]//2)
                pil_img.paste(rotated_right, pos_right, rotated_right)
                
                # 宸︿晶鏃嬭浆-90搴︽枃鏈?
                temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
                temp_draw_left = ImageDraw.Draw(temp_img_left)
                temp_draw_left.text((0, 0), text, font=font, fill=font_color)
                rotated_left = temp_img_left.rotate(90, expand=True)  # 閫嗘椂閽堟棆杞?0搴?
                
                # 璋冩暣宸︿晶浣嶇疆
                pos_left = (center_x - radius - int(width*0.0125), 
                        center_y - rotated_left.size[1]//2)
                
                # 搴曢儴姝ｅ父鏂囨湰
                text_x_bottom = center_x - text_width//2
                text_y_bottom = center_y + radius - int(height*0.0122) 
                draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)
                
                pil_img.paste(rotated_left, pos_left, rotated_left)
                
                # 杞崲鍥濷penCV鏍煎紡
                result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            # 淇濆瓨鐢熸垚鐨勫埢搴﹀€?
            r_ticks.append(tick)
            count += 1
            
            if count % self.tick_density == 0:
                continue
                
            cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1, lineType=cv2.LINE_AA)
            
            # 缁樺埗铏氱嚎鍦嗙幆
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

    def process_single_image(self, image_path, output_dir=None):
        """Process one radar chart through the full encryption flow."""
        try:
            # 濡傛灉鏈寚瀹氳緭鍑虹洰褰曪紝浣跨敤褰撳墠鐩綍
            if output_dir is None:
                output_dir = os.path.dirname(image_path) or '.'
                
            os.makedirs(output_dir, exist_ok=True)
            
            # 鑾峰彇鏂囦欢鍚嶅拰鎵╁睍鍚?
            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            
            # 鎵剧浜屼釜鍦?
            second_circle = self.visualize_ring_mask(image_path)
            self.second_circle_find(second_circle)
            
            print(f"妫€娴嬪埌鐨勫渾蹇冨潗鏍? {self.coords}")
            print(f"绗竴涓渾鍗婂緞: {self.first_r}")
            print(f"绗簩涓渾鍗婂緞: {self.second_r}")
            
            # 鍒涘缓涓存椂杈撳嚭璺緞
            temp_output_path = os.path.join(output_dir, f"temp_marked_{file_name}{file_ext}")
            
            # 鐢诲嚭涓や釜鍦嗗苟淇濆瓨涓存椂鍥惧儚
            image = cv2.imread(image_path)
            cv2.circle(image, (self.coords[0], self.coords[1]), self.first_r, (0, 255, 0), 1)  # 缁胯壊澶栧渾
            cv2.circle(image, (self.coords[0], self.coords[1]), self.second_r, (0, 255, 0), 1)  # 缁胯壊鍐呭渾
            cv2.circle(image, (self.coords[0], self.coords[1]), 2, (255, 0, 0), -1)  # 钃濊壊瀹炲績涓績鐐?
            
            cv2.imwrite(temp_output_path, image)
            print(f"涓存椂鏍囪鍥惧儚宸蹭繚瀛樿嚦: {temp_output_path}")
            
            # 璋冪敤LLM鑾峰彇鍒诲害淇℃伅
            response_data = self.call_llm_response(temp_output_path)
            scale_info = self._normalize_scale_info(response_data)
            if scale_info is None:
                print(f"鏃犳硶鑾峰彇鏈夋晥鐨勯浄杈惧浘鍒诲害淇℃伅: {response_data}")
                return None
            max_tick_value, min_tick_value, tick_interval = scale_info
            outer_radius = self._outer_radius()
            if outer_radius is None:
                print("鏈娴嬪埌鏈夋晥鐨勯浄杈惧浘澶栧湀鍗婂緞")
                return None
            res = f"outer_radius={outer_radius}"
            
            print(f"LLM鍒嗘瀽缁撴灉: max={max_tick_value}, min={min_tick_value}, interval={tick_interval}, {res}")
            
            # 鍔犲瘑澶勭悊
            result, r_ticks, argument = self.encrypt_rose_chart_one_tick(
                image_path, tick_interval, max_tick_value, outer_radius, max_tick_value, min_tick_value
            )
            
            # 淇濆瓨鏈€缁堢粨鏋?
            output_path = os.path.join(output_dir, f"{file_name}_encode{file_ext}")
            cv2.imwrite(output_path, result)
            print(f"鍔犲瘑鍚庣殑鍥惧儚宸蹭繚瀛樿嚦: {output_path}")
            
            # 澶勭悊JSON鏁版嵁锛堝鏋滃瓨鍦級
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
            }
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"闆疯揪鍥炬娴婮SON宸蹭繚瀛樿嚦: {output_json_path}")

            # 鍙€夛細鍒犻櫎涓存椂鏂囦欢
            # os.remove(temp_output_path)
            
            return output_path
            
        except Exception as e:
            print(f"澶勭悊鍥惧儚鏃跺嚭閿? {e}")
            return None


if __name__ == "__main__":
    # 绀轰緥鐢ㄦ硶
    encoder = RadarChartEncoder()
    
    # 鎸囧畾瑕佸鐞嗙殑鍥惧儚璺緞鍜岃緭鍑虹洰褰?
    image_path = "./data/upload/radar_000.png"  # 鍙互鏍规嵁闇€瑕佷慨鏀?
    output_dir = "./data/output/radar"      # 鍙互鏍规嵁闇€瑕佷慨鏀?
    
    # 澶勭悊鍗曞紶鍥惧儚
    result_path = encoder.process_single_image(image_path, output_dir)
    
    if result_path:
        print(f"澶勭悊瀹屾垚锛佸姞瀵嗗悗鐨勫浘鍍忎繚瀛樺湪: {result_path}")
    else:
        print("Processing failed")
