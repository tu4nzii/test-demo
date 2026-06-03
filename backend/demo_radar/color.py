import numpy as np
import json
import cv2
import re
import requests
import base64
import os

from model_api_config import get_chat_completion_url, get_headers, get_model_name

class RadarColorMatcher:
    """闆疯揪鍥惧疄浣撻鑹插尮閰嶅櫒"""
    def __init__(self):
        # API閰嶇疆
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.model_name = get_model_name()
        
        # 杈撳嚭閰嶇疆
        self.output_dir = "./data/output/radar"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 缁撴灉瀛樺偍
        self.entity_colors = {}
        
        # 棰滆壊璇嗗埆鍙傛暟
        self.min_block_area = 30         # 鏈€灏忛鑹插潡闈㈢Н
        self.max_block_area = 1000       # 鏈€澶ч鑹插潡闈㈢Н
        self.color_diff_threshold = 30   # 棰滆壊宸紓闃堝€?
        self.min_saturation = 30         # 鏈€灏忛ケ鍜屽害
        self.min_value = 50              # 鏈€灏忎寒搴?

    def parse_json(self, content: str):
        """浠庢枃鏈腑瑙ｆ瀽JSON鍐呭"""
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            return json.loads(match.group(1))
        except Exception as e:
            print(f"鉂?JSON瑙ｆ瀽澶辫触: {e}")
            return None

    def load_image(self, image_path):
        """鍔犺浇鍥惧儚鏂囦欢"""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"鏃犳硶璇诲彇鍥剧墖: {image_path}")
        return image

    def crop_legend(self, image, ratio=0.3, scale=2):
        """瑁佸壀鍥惧儚宸︿笂瑙掔殑鍥句緥鍖哄煙骞舵斁澶?
        
        Args:
            image: OpenCV鍥惧儚瀵硅薄
            ratio: 瑁佸壀姣斾緥锛堝乏涓婅鍖哄煙锛?
            scale: 鏀惧ぇ鍊嶆暟
            
        Returns:
            瑁佸壀骞舵斁澶у悗鐨勫浘鍍?
        """
        height, width = image.shape[:2]
        # 瑁佸壀宸︿笂瑙掑尯鍩?
        crop_region = image[:int(height*ratio), :int(width*ratio)]
        
        # 鏀惧ぇ鍥惧儚
        new_height = int(crop_region.shape[0] * scale)
        new_width = int(crop_region.shape[1] * scale)
        return cv2.resize(crop_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def image_to_base64(self, image):
        """灏哋penCV鍥惧儚杞崲涓篵ase64缂栫爜"""
        # 杞崲涓篟GB鏍煎紡
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 纭繚鏁版嵁绫诲瀷姝ｇ‘
        if rgb_image.dtype != np.uint8:
            rgb_image = rgb_image.astype(np.uint8)
            print("宸茶浆鎹㈠浘鍍忔暟鎹被鍨嬩负uint8")
            
        # 缂栫爜涓篔PEG骞惰浆鎹负base64
        success, encoded = cv2.imencode('.jpg', rgb_image)
        if not success:
            print("鍥惧儚缂栫爜澶辫触")
            return None
            
        return base64.b64encode(np.ascontiguousarray(encoded)).decode('utf-8')

    def detect_legend(self, base64_image):
        """浣跨敤澶фā鍨嬫娴嬪浘渚嬩綅缃?
        
        Args:
            base64_image: 瀹屾暣闆疯揪鍥剧殑base64缂栫爜
            
        Returns:
            鍖呭惈涓績鐐瑰潗鏍囧拰鑼冨洿鐨勫瓧鍏革紝澶辫触鍒欒繑鍥濶one
        """
        prompt = """
        璇峰垎鏋愯繖寮犻浄杈惧浘锛岃瘑鍒浘渚嬪尯鍩熺殑浣嶇疆鍜岃寖鍥淬€?
        涓€瀹氳鍖呭惈鏁翠釜鍥句緥鍖哄煙锛屼笉鑳藉彧鍖呭惈閮ㄥ垎銆?
        鍥句緥鍖哄煙閫氬父鍖呭惈闆疯揪鍥句腑鍚勪釜瀹炰綋鐨勫悕绉板強鍏跺搴旂殑棰滆壊鏍囪銆?
        
        璇蜂互JSON鏍煎紡杩斿洖锛?
        - position: 鍥句緥鍖哄煙涓績鐐圭殑鍧愭爣[x, y]
        - range: 鍥句緥鍖哄煙鐨勫搴﹀拰楂樺害[w, h]
        
        ```json
        {
            "position": [x, y],
            "range": [w, h]
        }
        ```
        
        璇风‘淇濊繑鍥炲悎鐞嗙殑鏁板€硷紝鏃犳硶璇嗗埆鏃惰繑鍥瀗ull銆?
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
                print(f"鏃犳硶鎻愬彇鏈夋晥鐨勫浘渚嬩綅缃俊鎭? {content}")
                return None
                
        except Exception as e:
            print(f"API璇锋眰閿欒: {e}")
            if 'response' in locals():
                print(f"鍝嶅簲鍐呭: {response.text}")
            return None
    
    def auto_crop_legend(self, image, scale=2):
        """鏅鸿兘瑁佸壀鍥句緥鍖哄煙
        
        浣跨敤澶фā鍨嬫娴嬪浘渚嬩綅缃紝绮剧‘瑁佸壀骞舵斁澶?
        
        Args:
            image: OpenCV鍥惧儚瀵硅薄
            scale: 鏀惧ぇ鍊嶆暟
            
        Returns:
            瑁佸壀骞舵斁澶у悗鐨勫浘渚嬪尯鍩?
        """
        # 杞崲涓篵ase64
        base64_image = self.image_to_base64(image)
        if base64_image is None:
            print("璀﹀憡锛氬浘鍍忚浆鎹㈠け璐ワ紝浣跨敤榛樿瑁佸壀")
            return self.crop_legend(image, scale=scale)
        
        # 妫€娴嬪浘渚嬩綅缃?
        print("姝ｅ湪妫€娴嬪浘渚嬩綅缃?..")
        legend_info = self.detect_legend(base64_image)
        
        # 鍥惧儚灏哄
        height, width = image.shape[:2]
        
        # 楠岃瘉鍥句緥淇℃伅
        if legend_info is None or not self._validate_legend_info(legend_info, width, height):
            print("璀﹀憡锛氬浘渚嬫娴嬪け璐ワ紝浣跨敤榛樿瑁佸壀")
            return self.crop_legend(image, scale=scale)
        
        # 璁＄畻瑁佸壀鍧愭爣
        center_x, center_y = legend_info["position"]
        region_width, region_height = legend_info["range"]
        
        # 娣诲姞杈硅窛骞剁‘淇濆湪鍥惧儚鑼冨洿鍐?
        margin = int(min(region_width, region_height) * 0.1)
        x1 = max(0, int(center_x - region_width / 2) - margin)
        y1 = max(0, int(center_y - region_height / 2) - margin)
        x2 = min(width, int(center_x + region_width / 2) + margin)
        y2 = min(height, int(center_y + region_height / 2) + margin)
        
        print(f"鍥句緥浣嶇疆: ({x1}, {y1}) 鍒?({x2}, {y2})")
        
        # 瑁佸壀鍥句緥鍖哄煙
        crop_region = image[y1:y2, x1:x2]
        
        # 妫€鏌ヨ鍓尯鍩熸槸鍚﹁繃灏?
        if crop_region.size == 0 or crop_region.shape[0] < 50 or crop_region.shape[1] < 50:
            print("璀﹀憡锛氬浘渚嬪尯鍩熻繃灏忥紝浣跨敤榛樿瑁佸壀")
            return self.crop_legend(image, scale=scale)
        
        # 鏀惧ぇ鍥惧儚
        if not self.extract_colors(crop_region):
            print("Warning: detected legend crop has no color markers, falling back to default crop")
            return self.crop_legend(image, scale=scale)

        new_height = int(crop_region.shape[0] * scale)
        new_width = int(crop_region.shape[1] * scale)
        return cv2.resize(crop_region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    def _validate_legend_info(self, legend_info, width, height):
        """楠岃瘉鍥句緥淇℃伅鏄惁鏈夋晥"""
        try:
            # 妫€鏌osition
            position = legend_info.get("position", [])
            if not isinstance(position, list) or len(position) != 2:
                return False
            
            x, y = position
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            
            # 妫€鏌ange
            region_range = legend_info.get("range", [])
            if not isinstance(region_range, list) or len(region_range) != 2:
                return False
            
            w, h = region_range
            if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
                return False
            
            # 妫€鏌ュ悎鐞嗘€?
            if w <= 0 or h <= 0 or w > width or h > height:
                return False
            
            if x < 0 or x > width or y < 0 or y > height:
                return False
                
            return True
        except:
            return False
    
    def extract_colors(self, image):
        """浠庡浘渚嬪浘鍍忎腑鎻愬彇鍞竴棰滆壊鍙婂叾浣嶇疆淇℃伅
        
        Args:
            image: OpenCV鍥惧儚瀵硅薄
            
        Returns:
            棰滆壊鍧椾俊鎭垪琛紝姣忎釜鍏冪礌鍖呭惈棰滆壊鍜屼綅缃俊鎭?
        """
        # 杞崲鍒癏SV骞跺垱寤洪鑹叉帺鐮?
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, self.min_saturation, self.min_value])
        upper_bound = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # 褰㈡€佸鎿嶄綔浼樺寲鎺╃爜
        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
        color_mask = cv2.dilate(color_mask, kernel, iterations=2)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
        
        # 鏌ユ壘杞粨
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 鎻愬彇棰滆壊鍧楀拰浣嶇疆淇℃伅
        block_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 杩囨护闈㈢Н鍜屽舰鐘?
            if (self.min_block_area < area < self.max_block_area and 
                0.3 < aspect_ratio < 3.0):
                # 璁＄畻骞冲潎棰滆壊
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_color = cv2.mean(image, mask=mask)[:3]  # BGR
                
                # 妫€鏌ラケ鍜屽害
                bgr_color = np.uint8([[mean_color]])
                hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]
                if hsv_color[1] >= self.min_saturation:
                    # 璁＄畻涓績鐐瑰潗鏍?
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # 瀛樺偍棰滆壊鍧椾俊鎭紙鍖呭惈浣嶇疆淇℃伅锛?
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
        
        # 棰滆壊鍘婚噸锛堜繚鐣欎綅缃俊鎭級
        unique_color_info = []
        for block in block_info:
            is_unique = True
            current_color = block['color']
            
            for unique_block in unique_color_info:
                unique_color = unique_block['color']
                # 璁＄畻HSV棰滆壊宸紓
                hsv1 = cv2.cvtColor(np.uint8([[current_color]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv2 = cv2.cvtColor(np.uint8([[unique_color]]), cv2.COLOR_BGR2HSV)[0][0]
                
                # 璁＄畻鑹茶皟宸紓锛堣€冭檻鐜舰鐗规€э級
                h_diff = min(abs(int(hsv1[0]) - int(hsv2[0])), 180 - abs(int(hsv1[0]) - int(hsv2[0])))
                s_diff = abs(int(hsv1[1]) - int(hsv2[1]))
                v_diff = abs(int(hsv1[2]) - int(hsv2[2]))
                
                # 鍔犳潈璺濈
                weighted_distance = int(h_diff) * 2 + int(s_diff) + int(v_diff)
                
                if weighted_distance <= self.color_diff_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_color_info.append(block)
        
        # 鎸夎壊璋冩帓搴?
        if unique_color_info:
            hsv_color_info = []
            for block in unique_color_info:
                hsv = cv2.cvtColor(np.uint8([[block['color']]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv_color_info.append((hsv[0], block))
            
            hsv_color_info.sort(key=lambda x: x[0])
            unique_color_info = [block for _, block in hsv_color_info]
        
        return unique_color_info
    
    def bgr_to_hex(self, bgr_color):
        """灏咮GR棰滆壊杞崲涓哄崄鍏繘鍒舵牸寮?
        
        Args:
            bgr_color: BGR鏍煎紡鐨勯鑹插€?
            
        Returns:
            鍗佸叚杩涘埗棰滆壊瀛楃涓?
        """
        rgb_color = bgr_color[::-1]  # BGR -> RGB
        return f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}".upper()
    
    def match_entities_colors(self, base64_image, entity_names, color_info_list):
        """浣跨敤澶фā鍨嬪尮閰嶅疄浣撳拰棰滆壊锛屾彁渚涢鑹蹭綅缃俊鎭緟鍔╁ぇ妯″瀷鍖归厤
        
        Args:
            base64_image: 鍥句緥鍥惧儚鐨刡ase64缂栫爜
            entity_names: 瀹炰綋鍚嶇О鍒楄〃
            color_info_list: 鎻愬彇鍒扮殑棰滆壊鍧椾俊鎭垪琛紝鍖呭惈棰滆壊鍜屼綅缃俊鎭?
            
        Returns:
            瀹炰綋鍜岄鑹茬殑绠€鍗曟槧灏勫瓧鍏?{瀹炰綋: 棰滆壊}
        """
        # 鍑嗗棰滆壊淇℃伅锛堝寘鍚崄鍏繘鍒跺拰浣嶇疆锛?
        color_with_positions = []
        hex_colors = []
        
        for i, color_info in enumerate(color_info_list):
            hex_color = self.bgr_to_hex(color_info['color'])
            hex_colors.append(hex_color)
            pos = color_info['position']
            color_with_positions.append(
                f"棰滆壊{i+1}: {hex_color} (宸︿笂瑙? {pos['x']}, {pos['y']}, 涓績鐐? {pos['center_x']}, {pos['center_y']})"
            )
        
        prompt = f"""
        璇峰垎鏋愯繖寮犻浄杈惧浘鍥句緥锛屽苟灏嗘彁渚涚殑瀹炰綋鍚嶇О涓庢彁鍙栧埌鐨勯鑹茶繘琛屼竴涓€瀵瑰簲銆?
        
        瀹炰綋鍚嶇О鍒楄〃锛?
        {', '.join(entity_names)}
        
        鎻愬彇鍒扮殑棰滆壊鍒楄〃锛堝寘鍚綅缃俊鎭紝鐢ㄤ簬杈呭姪鍖归厤锛夛細
        {'; '.join(color_with_positions)}
        
        璇蜂互JSON鏍煎紡杩斿洖瀹炰綋鍜岄鑹茬殑瀵瑰簲鍏崇郴锛堜粎鍖呭惈瀹炰綋鍚嶇О鍜屽崄鍏繘鍒堕鑹插€硷級锛?
        
        ```json
        {{
            "瀹炰綋1": "棰滆壊1",
            "瀹炰綋2": "棰滆壊2",
            ...
        }}
        ```
        
        璇风‘淇濇瘡涓疄浣撻兘鏈夊搴旂殑棰滆壊銆?
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
            print(f"API璇锋眰閿欒: {e}")
            if 'response' in locals():
                print(f"鍝嶅簲鍐呭: {response.text}")
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
        """澶勭悊闆疯揪鍥撅紝璇嗗埆瀹炰綋鍜岄鑹?
        
        Args:
            image_path: 鍥惧儚璺緞
            use_auto_crop: 鏄惁浣跨敤鏅鸿兘瑁佸壀
            entity_names: 宸茬煡鐨勫疄浣撳悕绉板垪琛?
            
        Returns:
            鍖呭惈璇嗗埆缁撴灉鐨勫瓧鍏革紝澶辫触鍒欒繑鍥濶one
        """
        try:
            # 璇诲彇鍥惧儚
            print(f"姝ｅ湪璇诲彇鍥惧儚: {image_path}")
            image = self.load_image(image_path)
            
            # 瑁佸壀鍥句緥
            print("姝ｅ湪瑁佸壀鍥句緥鍖哄煙...")
            legend_image = self.auto_crop_legend(image) if use_auto_crop else self.crop_legend(image)
            
            # 淇濆瓨瑁佸壀鍥惧儚
            base_name = os.path.basename(image_path)
            file_name, file_ext = os.path.splitext(base_name)
            legend_path = os.path.join(self.output_dir, f"legend_{file_name}{file_ext}")
            print(f"姝ｅ湪淇濆瓨鍥句緥鍥惧儚: {legend_path}")
            cv2.imwrite(legend_path, legend_image)
            
            # 杞崲涓篵ase64
            print("姝ｅ湪杞崲鍥惧儚鏍煎紡...")
            base64_image = self.image_to_base64(legend_image)
            if base64_image is None:
                raise ValueError("鍥惧儚杞崲澶辫触")
            
            # 鎻愬彇棰滆壊鍜屼綅缃俊鎭?
            print("姝ｅ湪鎻愬彇棰滆壊鍧?..")
            color_info_list = self.extract_colors(legend_image)
            hex_colors = [self.bgr_to_hex(info['color']) for info in color_info_list]
            print(f"鎴愬姛鎻愬彇鍒?{len(hex_colors)} 涓鑹? {', '.join(hex_colors)}")
            
            # 楠岃瘉瀹炰綋鍚嶇О
            if entity_names is None:
                print("閿欒锛氳鎻愪緵瀹炰綋鍚嶇О鍒楄〃")
                return None
            else:
                print(f"浣跨敤瀹炰綋鍚嶇О: {', '.join(entity_names)}")
            
            # 鍖归厤瀹炰綋鍜岄鑹?
            print("姝ｅ湪鍖归厤瀹炰綋鍜岄鑹?..")
            entity_colors = self.match_entities_colors(base64_image, entity_names, color_info_list)
            
            # 澶囬€夋柟妗堬細绠€鍗曚竴涓€瀵瑰簲
            if entity_colors is None:
                print("Color matching failed; using default order mapping")
                entity_colors = {}
                for i, entity in enumerate(entity_names):
                    if i < len(color_info_list):
                        color_info = color_info_list[i]
                        entity_colors[entity] = self.bgr_to_hex(color_info['color'])
                    else:
                        entity_colors[entity] = "#000000"  # 榛樿涓洪粦鑹?
            
            self.entity_colors = entity_colors
            
            # 鍑嗗缁撴灉
            result = {
                'image_path': image_path,
                'entity_colors': entity_colors,
                'extracted_colors': hex_colors,
                'legend_path': legend_path,
                'crop_method': 'auto' if use_auto_crop else 'default'
            }
            
            # 淇濆瓨缁撴灉
            output_path = os.path.join(self.output_dir, f"{file_name}_colors.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"缁撴灉宸蹭繚瀛樿嚦: {output_path}")
            print("\nRecognition results:")
            print(json.dumps(entity_colors, ensure_ascii=False, indent=2))
            
            return result
            
        except Exception as e:
            print(f"澶勭悊鍥惧儚鏃跺嚭閿? {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Manual smoke test."""
    # 鍒涘缓鍖归厤鍣ㄥ疄渚?
    matcher = RadarColorMatcher()
    
    # 閰嶇疆鍙傛暟
    # matcher.output_dir = "custom_output"  # 鑷畾涔夎緭鍑虹洰褰?
    
    # 鍥惧儚璺緞
    image_path = r"d:/home work/Agent.paper/test demo/backend/data/upload/radar_001.png"
    
    # 瀹炰綋鍚嶇О
    # entity_names = ["WDULR", "ZTJUP", "QCBOR", "RFLDM", "UCKIV"]
    entity_names =["LMIEXG","KBGCVO","AZC","OAAKCP"]
    # 澶勭悊鍥惧儚
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
