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
        # 閰嶇疆鍙傛暟
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        
        # 缁熶竴杈撳嚭璺緞閰嶇疆
        self.output_dir = "./data/output/radar"  # 涓昏緭鍑虹洰褰?
        self.axes_output_dir = os.path.join(self.output_dir)  # 杞寸嚎妫€娴嬬粨鏋滅洰褰?
        
        # 纭繚杈撳嚭鐩綍瀛樺湪
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.axes_output_dir, exist_ok=True)
        
        # 澶勭悊缁撴灉瀛樺偍
        self.axes_angles = []
        self.axis_labels = {}  # 閿?瑙掑害, 鍊?鏍囩
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
            print(f"鉂?JSON瑙ｆ瀽澶辫触: {e}")
            return None

    def crop_axis_label_region(self, image_path, center_x, center_y, angle_deg, radius, 
                              label_offset=100, label_width=150, label_height=150):
        """Crop an axis-label region by angle."""
        # 璇诲彇鍥惧儚骞惰浆鎹㈣搴︿负寮у害
        image = cv2.imread(image_path)
        angle_rad = math.radians(angle_deg)
        
        # 璁＄畻鍚嶇О鍖哄煙涓績鍧愭爣锛堝湪瑙掑害鏂瑰悜涓婏紝璺濈鍦嗗績radius+offset澶勶級
        label_center_x = int(center_x + (radius + label_offset) * math.cos(angle_rad))
        label_center_y = int(center_y - (radius + label_offset) * math.sin(angle_rad))  # 鍥惧儚y杞村悜涓嬶紝鏁呭噺鍙?
        
        # 璁＄畻瑁佸壀鍖哄煙宸︿笂瑙掑拰鍙充笅瑙掑潗鏍?
        x1 = max(0, label_center_x - label_width // 2)
        y1 = max(0, label_center_y - label_height // 2)
        x2 = min(image.shape[1], label_center_x + label_width // 2)
        y2 = min(image.shape[0], label_center_y + label_height // 2)
        
        # 瑁佸壀鍖哄煙骞惰繑鍥?
        return image[y1:y2, x1:x2]

    def call_llm_letter(self, crop_img):
        """Use the LLM to recognize letters in an image."""
        # 杞崲鍥惧儚鏍煎紡骞剁‘淇濇暟鎹被鍨嬫纭?
        image_area = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
        if image_area.dtype != np.uint8:
            image_area = image_area.astype(np.uint8)
            print("宸茶浆鎹㈠浘鍍忔暟鎹被鍨嬩负uint8")
            
        # 缂栫爜鍥惧儚涓篔PEG骞惰浆鎹负base64
        success, encoded_image = cv2.imencode('.jpg', image_area)
        if not success:
            print("鍥惧儚缂栫爜澶辫触")
            return None
            
        image_data = np.ascontiguousarray(encoded_image)
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # 鏋勫缓鎻愮ず鍜岃姹備綋
        prompt = f"""
        璇蜂綘鍒嗘瀽璇ュ浘鐗囦腑鐨勫瓧姣嶏紝骞惰繑鍥烇紝璇ュ瓧姣嶄负榛戣壊锛屼笖澶勫湪澶х害涓績浣嶇疆
        ```json
        {{
            "letter": <瀛楁瘝>
        }}
        濡傛灉鏃犳硶璇嗗埆鏌愪釜鍊硷紝璇蜂娇鐢?`null`
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
            
        # 鏋勫缓鎻愮ず鍜岃姹備綋
        prompt = f"""
        璇蜂綘鍒嗘瀽璇ュ浘鐗囦腑鐨勫惈鏈夌殑杞寸殑鍚嶇О鐨勪釜鏁帮紝骞惰繑鍥烇紝杞寸殑寤剁敵鏈€澶栧嚭鏈夊瓧姣嶇粍鍚堢殑杞村悕绉帮紝杞寸殑鍚嶇О涓鸿嫳鏂?
        杞寸殑鍚嶇О涓暟鍜岃酱鐨勪釜鏁板簲璇ヤ繚鎸佺浉鍚岋紝杞寸殑鍚嶇О涓嶉噸澶?
        璁颁綇锛屽彧鏈夊欢闀跨嚎涓婃湁鍚嶇О鐨勬墠绠椾竴涓酱锛屽鏋滄病鏈夊悕绉板垯涓嶇畻锛屾瘮濡傛湁浜涜嚜甯︾殑杞村彧鏄负浜嗘爣娉ㄥ埢搴︼紝杩欐牱鐨勫氨涓嶇畻杞?

        璇峰垽鏂酱鏄湪鑹插潡鐨勪腑蹇冧綅缃紝杩樻槸鍦ㄨ壊鍧楃殑杈圭紭浣嶇疆
        ```json
        {{
            "axis_name": <杞寸殑鍚嶇О>,
            "nums": <杞寸殑鍚嶇О涓暟>,
            "position": <杞寸殑浣嶇疆>,
            "reason": <鍘熷洜>
        }}
        濡傛灉鏃犳硶璇嗗埆鏌愪釜鍊硷紝璇蜂娇鐢?`null`
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
                
        print(f"鎵惧埌鏍囩鐨勮搴? {start_angle}")
        if start_angle:
            return start_angle[0]
        else:
            return None

    def find_radar_axes(self, image_path: str, center, start_angle: int, max_radius):
        """Detect radar axis angles."""
        img = cv2.imread(image_path)
        axes_angles = []
        
        # 璋冪敤LLM璇嗗埆杞寸殑鏁伴噺
        axes = self.call_llm_nums(image_path)
        print(f"LLM璇嗗埆缁撴灉: {axes}")
        
        if not axes or 'nums' not in axes:
            print("鏃犳硶璇嗗埆杞寸殑鏁伴噺")
            return []
            
        axes_nums = axes['nums']
        
        # 璁＄畻姣忎釜杞寸殑瑙掑害
        for i in range(axes_nums):
            best_interval = int(360 / axes_nums)
            angle = round(start_angle + i * best_interval) % 360
            axes_angles.append(angle)
        
        # 鍙鍖栫粨鏋滐紙鍙€夛級
        output_img = img.copy()
        for angle in axes_angles:
            current_angle_rad = math.radians(angle)
            
            # 娌跨潃鎵惧埌鐨勮搴︾敾涓€鏉＄嚎
            end_x = int(center[0] + max_radius * math.cos(current_angle_rad))
            end_y = int(center[1] + max_radius * math.sin(current_angle_rad))
            
            # 鍦ㄥ浘鍍忎笂缁樺埗绾㈢嚎锛岀敤浜庡彲瑙嗗寲
            cv2.line(output_img, center, (end_x, end_y), (0, 0, 255), 1)
            
            # 鍦ㄧ粓鐐瑰缁樺埗涓€涓渾鐐?
            cv2.circle(output_img, (end_x, end_y), 2, (0, 255, 0), -1)
        
        print(f"鎵惧埌鐨勮酱绾胯搴? {axes_angles}")
        
        # 淇濆瓨鍙鍖栫粨鏋滐紙浣跨敤缁熶竴鐨勮緭鍑虹洰褰曪級
        base_name = os.path.basename(image_path)
        file_name, file_ext = os.path.splitext(base_name)
        # output_path = os.path.join(self.axes_output_dir, f"axes_detected_{file_name}{file_ext}")
        # cv2.imwrite(output_path, output_img)
        # print(f"杞寸嚎妫€娴嬬粨鏋滃凡淇濆瓨鑷? {output_path}")
        
        self.radar_axes_angles = axes_angles
        return axes_angles

    def recognize_radar_axis_labels(self, image_path: str, center, radius, axes_angles):
        """Recognize labels for each axis."""
        axis_labels = {}
        
        for axis in axes_angles:
            try:
                crop_img = self.crop_axis_label_region(image_path, center[0], center[1], axis, radius)
                axis_data = self.call_llm_letter(crop_img)
                
                # 鎻愬彇璇嗗埆缁撴灉
                if isinstance(axis_data, dict) and 'letter' in axis_data and axis_data['letter'] is not None:
                    letter = axis_data['letter']
                    axis_labels[axis] = letter
                    print(f"杞磋搴? {axis}, 璇嗗埆缁撴灉: {letter}")
                else:
                    print(f"杞磋搴? {axis}, 璇嗗埆缁撴灉鏃犳晥: {axis_data}")
            except Exception as e:
                print(f"澶勭悊杞磋搴?{axis} 鏃跺彂鐢熼敊璇? {str(e)}")
                continue
        
        self.axis_labels = axis_labels
        return axis_labels

    def process_single_image(self, image_path, center=None, radius=None, output_json_path=None):
        """Process one radar chart and recognize axes and labels."""
        try:
            # 濡傛灉鏈寚瀹氬渾蹇冨拰鍗婂緞锛屽皾璇曚粠JSON鏂囦欢涓鍙?
            if center is None or radius is None:
                # 浣跨敤缁熶竴鐨勮緭鍑虹洰褰曟煡鎵綣SON鏂囦欢
                base_name = os.path.basename(image_path)
                file_name, _ = os.path.splitext(base_name)
                json_path = os.path.join(self.output_dir, f"{file_name}.json")
                
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        
                    if center is None and 'pred_coords' in json_data:
                        center = json_data['pred_coords']
                        print(f"浠嶫SON鏂囦欢涓鍙栧渾蹇? {center}")
                        
                    if radius is None and 'argument' in json_data and 'r_ticks' in json_data:
                        a = json_data['argument']['a']
                        b = json_data['argument']['b']
                        max_tick = json_data['r_ticks'][-1]
                        radius = a * max_tick + b - 5
                        print(f"璁＄畻寰楀埌鍗婂緞: {radius}")
                        
            # 纭繚鏈夊渾蹇冨拰鍗婂緞
            if center is None:
                raise ValueError("Center was not provided and could not be read from JSON")
            if radius is None:
                raise ValueError("Radius was not provided and could not be computed from JSON")
            
            self.center = center
            self.radius = radius
            
            # 鑾峰彇璧峰瑙掑害
            start_angle = self.get_start_angle(image_path, center[0], center[1], radius)
            if start_angle is None:
                print("鏃犳硶纭畾璧峰瑙掑害")
                return None
            
            print(f"璧峰瑙掑害: {start_angle}")
            
            # 璇嗗埆杞寸嚎
            found_axes = self.find_radar_axes(image_path, center, start_angle, radius)
            if not found_axes:
                print("No axes found")
                return None
            
            # 璇嗗埆杞存爣绛?
            axis_labels = self.recognize_radar_axis_labels(image_path, center, radius, found_axes)
            
            # 鍑嗗缁撴灉
            result = {
                'image_path': image_path,
                'center': center,
                'radius': radius,
                'start_angle': start_angle,
                'axes_angles': found_axes,
                'axis_labels': axis_labels
            }
            
            # 淇濆瓨缁撴灉鍒癑SON鏂囦欢锛堜娇鐢ㄧ粺涓€鐨勮緭鍑虹洰褰曪級
            base_name = os.path.basename(image_path)
            file_name, _ = os.path.splitext(base_name)
            
            if output_json_path is None:
                output_json_path = os.path.join(self.output_dir, f"{file_name}_axes.json")
                
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"璇嗗埆缁撴灉宸蹭繚瀛樿嚦: {output_json_path}")
            
            # 濡傛灉鍘烰SON鏂囦欢瀛樺湪锛屾洿鏂板畠
            original_json_path = os.path.join(self.output_dir, f"{file_name}.json")
            if os.path.exists(original_json_path):
                with open(original_json_path, 'r', encoding='utf-8') as f:
                    original_data = json.load(f)
                    
                # 鏇存柊鍘熸暟鎹?
                original_data['axis_labels'] = axis_labels
                original_data['axes_angles'] = found_axes
                original_data['start_angle'] = start_angle
                
                # 淇濆瓨鏇存柊鍚庣殑鏁版嵁
                with open(original_json_path, 'w', encoding='utf-8') as f:
                    json.dump(original_data, f, ensure_ascii=False, indent=2)
                    
                print(f"鍘烰SON鏂囦欢宸叉洿鏂? {original_json_path}")
            
            return result
            
        except Exception as e:
            print(f"澶勭悊鍥惧儚鏃跺嚭閿? {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 绀轰緥鐢ㄦ硶
    finder = RadarChartAxisFinder()
    
    # 鍙互鍦ㄦ澶勪慨鏀硅緭鍑鸿矾寰勶紙濡傛灉闇€瑕侊級
    # finder.output_dir = "custom_output"
    # finder.axes_output_dir = os.path.join(finder.output_dir, "custom_axes")
    
    # 鎸囧畾瑕佸鐞嗙殑鍥惧儚璺緞
    image_path = "./data/upload/radar_001.png"  # 鏍规嵁闇€瑕佷慨鏀?
    
    # 鍙互鎵嬪姩鎸囧畾鍦嗗績鍜屽崐寰勶紝涔熷彲浠ヨ绋嬪簭鑷姩浠嶫SON鏂囦欢涓鍙?
    # 鎵嬪姩鎸囧畾绀轰緥:
    # center = [300, 300]  # 鏍规嵁瀹為檯鎯呭喌淇敼
    # radius = 250  # 鏍规嵁瀹為檯鎯呭喌淇敼
    # result = finder.process_single_image(image_path, center, radius)
    
    # 鑷姩璇诲彇绀轰緥:
    result = finder.process_single_image(image_path)
    
    if result:
        print(f"澶勭悊瀹屾垚锛佽酱绾胯搴? {result['axes_angles']}")
        print(f"杞存爣绛? {result['axis_labels']}")
    else:
        print("Processing failed")
