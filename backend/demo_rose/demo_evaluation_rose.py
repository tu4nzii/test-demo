import json
import os
from typing import List, Dict, Tuple, Optional
import base64
import requests
import re
import time
import cv2
import math
import numpy as np

from model_api_config import get_chat_completion_url, get_headers, get_model_name

class RoseChartEvaluator:
    def __init__(self):
        # 鍒濆鍖朅PI鍙傛暟鍜岄厤缃?
        self.url = get_chat_completion_url()
        self.headers = get_headers()
        self.llm_model = get_model_name()
        
        # 瀹氫箟涓存椂鏂囦欢鐩綍
        self.feedback_image_dir = './data/feedback'
        self.amplifier_image_dir = './data/amplifier/rose'
        
        # 鍒涘缓蹇呰鐨勭洰褰?
        self._create_directories()
        
        # 瀛樺偍缁撴灉
        self.results_by_image = {}
    
    def _create_directories(self):
        """鍒涘缓蹇呰鐨勪复鏃舵枃浠剁洰褰?""
        if not os.path.exists(self.feedback_image_dir):
            os.makedirs(self.feedback_image_dir)
        if not os.path.exists(self.amplifier_image_dir):
            os.makedirs(self.amplifier_image_dir)
    
    def load_dataset(self, json_path: str) -> dict:
        """鍔犺浇鍗曚釜JSON鏁版嵁闆嗘枃浠?""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"鉂?鍔犺浇鏁版嵁闆嗗け璐? {e}")
            return {}
    
    def extract_json_response(self, content: str) -> Optional[dict]:
        """浠嶭LM鍝嶅簲涓彁鍙朖SON鏍煎紡鐨勫唴瀹?""
        try:
            match = re.search(r'(\{[\s\S]*\})', content)
            if not match:
                return None
            json_str = match.group(1)
            return json.loads(json_str)
        except Exception as e:
            print(f"鉂?JSON瑙ｆ瀽澶辫触: {e}")
            return None
    
    def validate_coordinates(self, coords: Tuple) -> bool:
        """楠岃瘉鍧愭爣鏄惁鏈夋晥"""
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            return False
        valid = lambda x: isinstance(x, (int, float)) or x is None
        return valid(coords[0]) and valid(coords[1])
    
    def encode_image(self, image, center_x: int, center_y: int, arg_a: float, arg_b: float, r_ticks: List[float]) -> np.ndarray:
        """鍦ㄥ浘鍍忎笂缁樺埗鍔犲瘑缃戞牸绾?""
        line_color = (128, 128, 128)
        thickness = 1
        if r_ticks and r_ticks[0] == 0:
            r_ticks.pop(0)
        
        count = 0
        for tick in r_ticks:
            count += 1
            if count % 4 == 0:
                continue
            radius = int(arg_a * tick + arg_b)
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
                cv2.line(image, (x1, y1), (x2, y2), line_color, thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def crop_axis_label_region(self, image_path: str, center_x: int, center_y: int, angle_deg: float, 
                             outer_radius: int, angle_width: int = 30, inner_radius: int = 0, 
                             label_offset: int = 30, scale_factor: float = 1.0, 
                             r_ticks: List[float] = None, arg_a: float = 0, arg_b: float = 0) -> np.ndarray:
        """鏍规嵁瑙掑害瑁佸壀鎵囧舰鍖哄煙"""
        if r_ticks is None:
            r_ticks = []
        
        # 璇诲彇鍥惧儚
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"鏃犳硶璇诲彇鍥惧儚: {image_path}")
        h, w = image.shape[:2]
        
        # 缁樺埗鍔犲瘑缃戞牸绾?
        self.encode_image(image, center_x, center_y, arg_a, arg_b, r_ticks)
        
        # 鍒涘缓鎺╃爜锛堥粦鑹茶儗鏅級
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 璁＄畻鎵囧舰瑙掑害鑼冨洿
        start_angle = angle_deg - angle_width / 2
        end_angle = angle_deg + angle_width / 2
        
        # 鍦ㄥ浘鍍忎笂娣诲姞瑙掑害鏂囧瓧鏍囨敞
        for tick in r_ticks:
            radius = int(arg_a * tick + arg_b)
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.3
            font_color = (0, 0, 0)  # 榛戣壊
            thickness = 1
            
            # 璁＄畻璧峰瑙掑害鏂囧瓧鐨勪綅缃?
            start_angle_rad = math.radians(start_angle + 4)
            text_radius = radius 
            
            # 鑾峰彇鏂囧瓧灏哄浠ュ疄鐜颁腑蹇冨榻?
            text = str(int(tick) if tick % 1 == 0 else tick)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # 璁＄畻鏂囧瓧涓績鍧愭爣
            start_x_center = int(center_x + text_radius * math.cos(start_angle_rad))
            start_y_center = int(center_y - text_radius * math.sin(start_angle_rad))
            
            # 璋冩暣涓哄乏涓嬭鍧愭爣
            start_x = start_x_center - text_size[0] // 2
            start_y = start_y_center + text_size[1] // 2
            
            # 璁＄畻缁撴潫瑙掑害鏂囧瓧鐨勪綅缃?
            end_angle_rad = math.radians(end_angle - 4)
            end_x_center = int(center_x + text_radius * math.cos(end_angle_rad))
            end_y_center = int(center_y - text_radius * math.sin(end_angle_rad))
            end_x = end_x_center - text_size[0] // 2
            end_y = end_y_center + text_size[1] // 2
            
            # 娣诲姞鏂囧瓧鏍囨敞
            cv2.putText(image, text, (start_x, start_y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
            cv2.putText(image, text, (end_x, end_y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
        
        # 杞崲涓篛penCV瑙掑害绯荤粺
        start_angle_cv = -end_angle
        end_angle_cv = -start_angle
        
        # 缁樺埗鎵囧舰鎺╃爜
        out_axes = (outer_radius, outer_radius)
        cv2.ellipse(
            mask,
            (center_x, center_y),
            out_axes,
            angle=0,
            startAngle=start_angle_cv,
            endAngle=end_angle_cv,
            color=255,
            thickness=-1,  # 濉厖
            lineType=cv2.LINE_AA
        )
        
        # 濡傛灉鏈夊唴鍗婂緞锛岀粯鍒跺唴鍦嗗苟鍑忓幓
        if inner_radius > 0:
            inner_axes = (inner_radius, inner_radius)
            cv2.ellipse(
                mask,
                (center_x, center_y),
                inner_axes,
                angle=0,
                startAngle=start_angle_cv - 10,
                endAngle=end_angle_cv + 10,
                color=0,
                thickness=-1,
                lineType=cv2.LINE_AA
            )
        
        # 搴旂敤鎺╃爜
        sector_img = cv2.bitwise_and(image, image, mask=mask)
        # 灏嗚儗鏅粠榛戣壊鏀逛负鐧借壊
        sector_img[mask == 0] = 255
        
        # 璁＄畻鎵囧舰杈圭晫妗嗗苟瑁佸壀
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image
        
        x, y, w_sector, h_sector = cv2.boundingRect(coords)
        crop_img = sector_img[y-20:y+h_sector+20, x-20:x+w_sector+20]
        
        # 娣诲姞鍥惧儚鏀惧ぇ鍔熻兘
        if scale_factor != 1.0 and crop_img.size > 0:
            new_width = int(crop_img.shape[1] * scale_factor)
            new_height = int(crop_img.shape[0] * scale_factor)
            crop_img = cv2.resize(
                crop_img,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC
            )
        
        return crop_img
    
    def draw_angle_indicator(self, image: np.ndarray, center_x: int, center_y: int, target_angle: float, radius: int, 
                           arc_color: Tuple[int, int, int] = (0, 0, 255), line_color: Tuple[int, int, int] = (0, 0, 255),
                           arc_thickness: int = 2, line_thickness: int = 2, arc_angle_width: int = 10, 
                           line_length_ratio: float = 0.3) -> np.ndarray:
        """鍦ㄥ浘鍍忎笂缁樺埗鐗瑰畾瑙掑害鐨勬墖褰㈠姬绾垮拰瀵瑰簲杞翠笂鐨勬爣璁?""
        # 璁＄畻寮х嚎鐨勮捣濮嬪拰缁撴潫瑙掑害
        start_angle = -target_angle - arc_angle_width // 2
        end_angle = -target_angle + arc_angle_width // 2  
        
        # 缁樺埗鎵囧舰寮х嚎
        cv2.ellipse(image, (center_x, center_y), (radius, radius), 0, start_angle, end_angle, 
                   arc_color, arc_thickness, lineType=cv2.LINE_AA)
        
        # 灏嗚搴﹁浆鎹负寮у害骞惰绠楃嚎娈电殑绔偣鍧愭爣
        angle_rad = math.radians(target_angle)
        outer_x = int(center_x + (radius + line_length_ratio * radius) * math.cos(angle_rad))
        outer_y = int(center_y - (radius + line_length_ratio * radius) * math.sin(angle_rad))
        
        # 璁＄畻鎸囧悜鍦嗗績鐨勭嚎娈电殑鍐呯鐐瑰潗鏍?
        inner_radius = radius * (1 - line_length_ratio)
        inner_x = int(center_x + inner_radius * math.cos(angle_rad))
        inner_y = int(center_y - inner_radius * math.sin(angle_rad))
        
        # 缁樺埗鎸囧悜鍦嗗績鐨勭嚎
        cv2.line(image, (outer_x, outer_y), (inner_x, inner_y), line_color, line_thickness, lineType=cv2.LINE_AA)
        
        return image
    
    def generate_prompt(self, item_name: str, prompt_type: str, dataset: dict, tick: float = 0) -> str:
        """鏍规嵁鍥捐〃绫诲瀷鍜屾彁绀虹被鍨嬬敓鎴愬搴旂殑鎻愮ず鏂囨湰"""
        # 鑾峰彇鍥捐〃绫诲瀷
        chart_type = dataset.get('chart_type', '')
        start_angle = dataset.get('start_angle', 0)
        # print(f"褰撳墠澶勭悊: {item_name}, 缃戞牸绫诲瀷: {prompt_type}, 鍥捐〃绫诲瀷: {chart_type}")
        # print(f"{dataset.get('axis_labels')[str(start_angle)]}瀵瑰簲鑼冨洿涓簕start_angle}-{dataset.get('axes_angles')[1]}")
        if prompt_type == "with_grid":
            if chart_type == 'radar':
                return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} angle positions

Your task is to estimate the value of the data point labeled "{item_name}":

1.Locate the "{item_name}" data point on the radar chart.
2.Estimate its radial position by interpolating between concentric circles.** Remember to always interpolate and make good use of the encrypted grid **

鈿狅笍 Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
            elif chart_type == 'rose':
                return f'''
鍥捐〃鍖呭惈**铏氭嫙鍙傝€冪嚎**锛?
鎮ㄦ鍦ㄥ垎鏋愪竴寮犵帿鐟板浘銆傚畠閫氳繃鎵囧舰鐨?*鏈€杩滅鍗婂緞**鏉ュ睍绀烘暟鎹紝姣忎釜鎵囧舰浠ｈ〃涓€涓被鍒紝鍏堕暱搴﹁〃绀烘暟鎹€肩殑澶у皬銆?
浠ヤ笅涓哄浘琛ㄧ殑璇︾粏淇℃伅锛?
    - 瀛樺湪浠ヤ笅寰勫悜缃戞牸绾匡紙鍚屽績鍦嗭級锛屽搴旂殑鍒诲害鍊间负{dataset.get('r_ticks', [])},鏍囨敞鍦ㄥ搴旂綉鏍艰櫄绾夸笂
    - 瀛樺湪浠ヤ笅瑙掑害缃戞牸绾匡紝灏嗗渾鍒嗘垚澶氫釜鎵囧舰鍖哄煙锛寋dataset.get('axis_labels')}鍒嗗埆瀵瑰簲姣忎釜鎵囧舰鍖哄煙锛屾墖褰㈠尯鍩熺殑鍒嗙晫涓簕dataset.get('axes_angles', [])}锛堝崟浣嶄负搴︼級

鎮ㄧ殑浠诲姟鏄及璁℃爣璁颁负"{item_name}"鐨勫搴旀墖褰㈢殑鐨勫€硷細
浠ヤ笅涓烘彁绀猴細
    1. **鍦ㄧ帿鐟板浘涓婃壘鍒?{item_name}"瀵瑰簲鐨勬墖褰㈠尯鍩燂紝鍗崇‘瀹氬叾瑙掑害鑼冨洿銆?*,{dataset.get('axis_labels')[str(start_angle)]}瀵瑰簲鑼冨洿涓簕start_angle}-{dataset.get('axes_angles')[1]}"
    渚嬪瓙锛氳鍥剧殑{dataset.get('axis_labels')[str(start_angle)]}瀵瑰簲鑼冨洿涓簕start_angle}-{dataset.get('axes_angles')[1]}搴︼紝澶勫湪鍥炬渶鍙崇
    2. 纭畾鍏跺緞鍚戜綅缃紝鎵惧埌鍏跺浜庡摢涓や釜缃戞牸绾夸箣闂达紝缃戞牸绾垮寘鍚互涓嬪埢搴dataset.get('r_ticks', [])}锛屽繀椤诲噯纭殑璇嗗埆鍏朵綅浜庡摢涓や釜缃戞牸绾夸箣闂?

    3. 鏍规嵁鍏舵墖褰㈠拰鐩稿浜庝袱涓綉鏍肩嚎鐨勪綅缃紝鎻掑€艰绠楀叾鏁版嵁鍊笺€?

**璁颁綇锛屼竴瀹氳鎻掑€硷紝鍒╃敤濂界綉鏍肩嚎鐨勫埢搴﹀€?*
鍦ㄩ娴嬩箣鍓嶏紝鍐嶆鍥為【浠ヤ笅鎴戠粰浣犵殑鎻愮ず
涓€瀹氳缁欐垜涓€涓€硷紝涓嶈兘缁欐垜澶氫釜鍊硷紝涔熶笉鑳界粰鎴戞病鏈夊€肩殑鎯呭喌

鈿狅笍 浠呬互浠ヤ笅纭垏鐨凧SON鏍煎紡鍝嶅簲锛?
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

涓嶈鍖呭惈浠讳綍瑙ｉ噴鎴栭澶栨枃鏈€?
'''.strip()
        elif prompt_type == "no_grid":
            if chart_type == 'radar':
                return f'''
Your task is to estimate the value of the data point labeled "{item_name}":

鈿狅笍 Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
            elif chart_type == 'rose':
                return f'''
鎮ㄧ殑浠诲姟鏄及璁℃爣璁颁负"{item_name}"瀵瑰簲鐨勫€硷細

鈿狅笍 浠呬互浠ヤ笅纭垏鐨凧SON鏍煎紡鍝嶅簲锛?
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

涓嶈鍖呭惈浠讳綍瑙ｉ噴鎴栭澶栨枃鏈€?
'''.strip()
        elif prompt_type == "feedback":
            if chart_type == 'radar':
                return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} positions

Your task is to estimate the value of the data point labeled "{item_name}":

**閲嶈鎻愮ず**锛氬浘琛ㄤ腑宸叉坊鍔犵孩鑹插渾鐜紝琛ㄧず涓婁竴杞"{item_name}"鐨勯娴嬪€肩害涓簕tick}銆?
璇锋瘮杈冪孩鑹插渾鐜笌鐪熷疄鏁版嵁鐐圭殑浣嶇疆宸窛锛岄噸鏂颁紭鍖栨偍鐨勯娴嬶細
1. 纭畾绾㈣壊鍦嗙幆涓庣湡瀹炴暟鎹偣涔嬮棿鐨勪綅缃叧绯伙紙鍋忓唴銆佸亸澶栵級
2. 鏍规嵁杩欑鍏崇郴锛岃皟鏁存偍鐨勯娴嬪€?
3. 纭繚鏂扮殑棰勬祴鍊间笌鐪熷疄鐐圭殑浣嶇疆瀵归綈 浠ュ疄鐜板敖鍙兘鍑嗙‘鐨勯娴?

鈿狅笍 浠呬互浠ヤ笅纭垏鐨凧SON鏍煎紡鍝嶅簲锛?
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

涓嶈鍖呭惈浠讳綍瑙ｉ噴鎴栭澶栨枃鏈€?
'''.strip()
            elif chart_type == 'rose':
                return f'''
鍥捐〃鍖呭惈**铏氭嫙鍙傝€冪嚎**锛?
鎮ㄦ鍦ㄥ垎鏋愪竴寮犵帿鐟板浘銆傚畠閫氳繃鎵囧舰鐨?*鏈€杩滅鍗婂緞**鏉ュ睍绀烘暟鎹紝姣忎釜鎵囧舰浠ｈ〃涓€涓被鍒紝鍏堕暱搴﹁〃绀烘暟鎹€肩殑澶у皬銆?
浠ヤ笅涓哄浘琛ㄧ殑璇︾粏淇℃伅锛?
    - 瀛樺湪浠ヤ笅寰勫悜缃戞牸绾匡紙鍚屽績鍦嗭級锛屽搴旂殑鍒诲害鍊间负{dataset.get('r_ticks', [])},鏍囨敞鍦ㄥ搴旂綉鏍艰櫄绾夸笂
    - 瀛樺湪浠ヤ笅瑙掑害缃戞牸绾匡紝灏嗗渾鍒嗘垚澶氫釜鎵囧舰鍖哄煙锛寋dataset.get('axis_lables')}鍒嗗埆瀵瑰簲姣忎釜鎵囧舰鍖哄煙锛屾墖褰㈠尯鍩熺殑鍒嗙晫涓簕dataset.get('axes_angles', [])}锛堝崟浣嶄负搴︼級

鎮ㄧ殑浠诲姟鏄及璁℃爣璁颁负"{item_name}"鐨勫搴旀墖褰㈢殑鍊硷細

**閲嶈鎻愮ず**锛氬浘琛ㄤ腑宸叉坊鍔犵孩鑹插渾鐜紝琛ㄧず涓婁竴杞"{item_name}"鐨勯娴嬪€肩害涓簕tick}銆?
璇锋瘮杈冪孩鑹插渾鐜笌鐪熷疄鏁版嵁鐐圭殑浣嶇疆宸窛锛岄噸鏂颁紭鍖栨偍鐨勯娴嬶細
1. 纭畾绾㈣壊鍦嗙幆涓庣湡瀹炴暟鎹偣涔嬮棿鐨勪綅缃叧绯伙紙鍋忓唴銆佸亸澶栵級
2. 鏍规嵁杩欑鍏崇郴锛岃皟鏁存偍鐨勯娴嬪€?
3. 纭繚鏂扮殑棰勬祴鍊间笌鐪熷疄鐐圭殑浣嶇疆瀵归綈 浠ュ疄鐜板敖鍙兘鍑嗙‘鐨勯娴?
涓€瀹氳缁欐垜涓€涓€硷紝涓嶈兘缁欐垜澶氫釜鍊硷紝涔熶笉鑳界粰鎴戞病鏈夊€肩殑鎯呭喌

**璁颁綇锛屼竴瀹氳鍒╃敤缃戞牸绾胯繘琛岀簿纭彃鍊?*

鈿狅笍 浠呬互浠ヤ笅纭垏鐨凧SON鏍煎紡鍝嶅簲锛?
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

涓嶈鍖呭惈浠讳綍瑙ｉ噴鎴栭澶栨枃鏈€?
'''.strip()
        elif prompt_type == "amplifier":
            if chart_type == 'radar':
                return f'''
鎮ㄦ鍦ㄥ垎鏋愰浄杈惧浘鐨勪竴閮ㄥ垎銆傝鍥惧湪浜岀淮骞抽潰涓婁娇鐢ㄤ粠鍚屼竴鐐瑰嚭鍙戠殑鍧愭爣杞村睍绀哄鍙橀噺鏁版嵁銆?

- 鍏辨湁{len(dataset.get('series_color', {}))}涓疄浣擄細{', '.join(dataset.get('series_color', {}).keys())}锛屽垎鍒搴旈鑹瞷', '.join(dataset.get('series_color', {}).values())}
鐜板湪鐨勫眬閮ㄦ斁澶у浘涓簕item_name.split(',')[1].strip()}杞村搴旂殑灞€閮ㄦ斁澶?
鎮ㄧ殑浠诲姟鏄及璁℃爣璁颁负"{item_name}"瀵瑰簲鐨勫€硷紝鍗硔item_name}瀵瑰簲瀹炰綋棰滆壊鐨勬暟鍊笺€?
璇峰厛鎵惧埌{item_name.split(',')[0].strip()}瀵瑰簲瀹炰綋棰滆壊涓簕dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '鏈煡棰滆壊')}
鐒跺悗鎵惧埌璇ラ鑹插搴旂殑鐐癸紝骞舵彃鍊煎嚭鏁板€?

鈿狅笍 浠呬互浠ヤ笅纭垏鐨凧SON鏍煎紡鍝嶅簲锛?
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

涓嶈鍖呭惈浠讳綍瑙ｉ噴鎴栭澶栨枃鏈€?
'''.strip()
            elif chart_type == 'rose':
                return f'''
                璇ュ浘鐗囦负鐜懓鍥句腑{item_name}鏁版嵁鐨勬斁澶э紝璇蜂綘鏍规嵁璇ュ浘鐗囷紝浼拌{item_name}瀵瑰簲鐨勬暟鍊笺€?
-**鎵惧埌鎵囧舰骞朵笖鎵惧埌鍏舵渶杩滅鐨勮竟鐣?*
-鐒跺悗鎵惧埌璇ヨ竟鐣屽浜庡摢涓や釜鍩哄噯绾夸箣闂?
-鏈€鍚庝緷鎹熀鍑嗙嚎鐨勬暟鍊硷紝鎻掑€煎嚭鏁板€?
鈿狅笍 浠呬互浠ヤ笅纭垏鐨凧SON鏍煎紡鍝嶅簲锛?
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

涓嶈鍖呭惈浠讳綍瑙ｉ噴鎴栭澶栨枃鏈€?
'''.strip()
        else:
            raise ValueError("Unknown prompt_type")
    
    def call_llm_response(self, prompt: str, image_path: str, item_name: str, dataset: dict) -> Tuple[Optional[float], Optional[float]]:
        """璋冪敤LLM鎺ュ彛鑾峰彇鍝嶅簲"""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            print(f"鉂?璇诲彇鍥惧儚鏂囦欢澶辫触: {e}")
            return (None, None)
        
        max_retries = 10
        retry_delay = 0.5  # 绉?
        retry_count = 0
        
        payload = {
            "model": self.llm_model,
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
        
        while retry_count < max_retries:
            try:
                response = requests.post(url=self.url, headers=self.headers, json=payload, timeout=10)
                response.raise_for_status()  # 妫€鏌TTP閿欒鐘舵€佺爜
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                coords_json = self.extract_json_response(content)
                
                if coords_json and "datapoints" in coords_json:
                    for item in coords_json["datapoints"]:
                        if item_name in item:
                            coords = item[item_name]
                            if self.validate_coordinates(coords):
                                return tuple(coords)
                
                # 濡傛灉鏈壘鍒版暟鎹絾璇锋眰鎴愬姛锛屼笉閲嶈瘯鐩存帴杩斿洖
                return (None, None)
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                print(f"鉂?璇锋眰寮傚父: {e}, 姝ｅ湪杩涜绗?{retry_count}/{max_retries} 娆￠噸璇?..")
                if retry_count < max_retries:
                    time.sleep(retry_delay)
            except Exception as e:
                retry_count += 1
                print(f"鉂?瑙ｆ瀽寮傚父: {e}, 姝ｅ湪杩涜绗?{retry_count}/{max_retries} 娆￠噸璇?..")
                if retry_count < max_retries:
                    time.sleep(retry_delay)
        
        print(f"鈿狅笍 宸茶揪鍒版渶澶ч噸璇曟鏁?({max_retries}娆?")
        return (None, None)
    
    def process_single_image(self, json_path: str) -> None:
        """澶勭悊鍗曚釜鍥惧儚鐨勮瘎浼伴€昏緫"""
        # 鍔犺浇鏁版嵁闆?
        dataset = self.load_dataset(json_path)
        if not dataset:
            print("鉂?鏁版嵁闆嗕负绌猴紝鏃犳硶杩涜璇勪及")
            return
        
        # 纭繚鏄帿鐟板浘涓旀湁鏁版嵁
        if dataset.get('chart_type') != 'rose' or not dataset.get('data'):
            print("鉂?涓嶆槸鐜懓鍥炬垨娌℃湁鏁版嵁锛岃烦杩囧鐞?)
            return
        
        chart_id = dataset.get('chart_id', 'unknown')
        print(f"寮€濮嬪鐞嗗浘琛? {chart_id}")
        
        # 鍒濆鍖栫粨鏋滃瓧鍏?
        self.results_by_image[chart_id] = {
            'chart_type': dataset.get('chart_type', 'rose'),
            'data': {}
        }
        
        # 閬嶅巻姣忎釜鏁版嵁椤?
        for item_name, value in dataset.get('data', {}).items():
            self.results_by_image[chart_id]['data'][item_name] = {}
            
            # 澶勭悊甯︾綉鏍煎拰鏃犵綉鏍间袱绉嶆儏鍐?
            for grid_type in ['with_grid', 'no_grid']:
                # 鑾峰彇瀵瑰簲鐨勫浘鍍忚矾寰?
                image_path = dataset.get("image_paths", {}).get(grid_type)
                if not image_path:
                    print(f"鈿狅笍 鏈壘鍒皗grid_type}瀵瑰簲鐨勫浘鍍忚矾寰勶紝璺宠繃璇ョ被鍨嬪鐞?)
                    continue
                
                # 鏇挎崲璺緞鍒嗛殧绗︿互纭繚鍏煎鎬?
                image_path = image_path.replace('\\', '/')
                
                # 鐢熸垚鎻愮ず骞惰皟鐢↙LM
                try:
                    prompt = self.generate_prompt(item_name, grid_type, dataset)
                    # print(f"褰撳墠澶勭悊: {item_name}, 缃戞牸绫诲瀷: {grid_type}, 鎻愮ず: {prompt}")
                    coords = self.call_llm_response(prompt, image_path, item_name, dataset)
                    
                    # 鑾峰彇杞存爣绛惧拰瑙掑害鏄犲皠
                    axis_labels = dataset.get('axis_labels', {})
                    label_to_angle = {v: int(k) for k, v in axis_labels.items()}  # 寤虹珛鏍囩鍒拌搴︾殑鍙嶅悜鏄犲皠
                    
                    item_label = item_name
                    if item_label not in label_to_angle:
                        print(f"璀﹀憡: 鏈壘鍒皗item_label}瀵瑰簲鐨勮搴︼紝璺宠繃褰撳墠鍥捐〃澶勭悊")
                        continue
                    
                    angle_width = int(360 / len(axis_labels)) if axis_labels else 30
                    target_angle = label_to_angle[item_label]
                    
                    # 澶勭悊鍙嶉妯″紡
                    if grid_type == 'with_grid' and coords[0] is not None:
                        feedback_counts = 0
                        feedback_tick = [coords[0]] if coords[0] is not None else [value]
                        feedback_times = 1
                        
                        while feedback_counts > 0:
                            try:
                                temp_image = cv2.imread(dataset["image_paths"][grid_type].replace('\\', '/'))
                                if temp_image is None:
                                    print(f"鉂?鏃犳硶璇诲彇鍥惧儚: {dataset['image_paths'][grid_type]}")
                                    break
                                
                                feedback_image = temp_image.copy()
                                feedback_image_path = os.path.join(self.feedback_image_dir, 
                                                                 f'{chart_id}_{grid_type}_{item_name}_{feedback_times}.png')
                                
                                center_x = dataset["pred_coords"][0]
                                center_y = dataset["pred_coords"][1]
                                a = dataset["argument"]["a"]
                                b = dataset["argument"]["b"]
                                pre_r = int(a * feedback_tick[-1] + b)
                                
                                # 缁樺埗瑙掑害鎸囩ず鍣?
                                feedback_image = self.draw_angle_indicator(feedback_image, center_x, center_y, 
                                                                          target_angle, pre_r, line_thickness=2, 
                                                                          arc_angle_width=angle_width, 
                                                                          line_length_ratio=0.05)
                                
                                # 淇濆瓨鍙嶉鍥惧儚
                                cv2.imwrite(feedback_image_path, feedback_image)
                                
                                # 鐢熸垚鍙嶉鎻愮ず骞惰皟鐢↙LM
                                feedback_prompt = self.generate_prompt(item_name, 'feedback', dataset, feedback_tick[-1])
                                feedback_coords = self.call_llm_response(feedback_prompt, feedback_image_path, 
                                                                         item_name, dataset)
                                
                                # 鏇存柊鍙嶉tick鍒楄〃
                                if feedback_coords[0] is not None:
                                    feedback_tick.append(feedback_coords[0])
                                else:
                                    feedback_tick.append(coords[0])
                                
                                print(f"鍙嶉缁撴灉: {feedback_tick}")
                                
                                # 鍒犻櫎涓存椂鍙嶉鍥惧儚
                                if os.path.exists(feedback_image_path):
                                    os.remove(feedback_image_path)
                                
                                feedback_times += 1
                                feedback_counts -= 1
                                
                            except Exception as e:
                                print(f"鉂?鍙嶉澶勭悊寮傚父: {e}")
                                break
                        
                        # 淇濆瓨鍙嶉缁撴灉
                        if feedback_tick:
                            self.results_by_image[chart_id]['data'][item_name]['feedback'] = feedback_tick
                        
                        # 澶勭悊鏀惧ぇ妯″紡
                        try:
                            amplifier_path = dataset["image_paths"].get('no_grid', '').replace('\\', '/')
                            if not amplifier_path or not os.path.exists(amplifier_path):
                                print(f"鈿狅笍 鏈壘鍒皀o_grid鍥惧儚璺緞鎴栨枃浠朵笉瀛樺湪: {amplifier_path}")
                                continue
                            
                            center_x, center_y = dataset["pred_coords"]
                            arg_a = dataset["argument"]["a"]
                            arg_b = dataset["argument"]["b"]
                            radius = int(arg_a * coords[0] + arg_b) if coords[0] is not None else 0
                            r_ticks = dataset["r_ticks"]
                            
                            # 纭畾鍐呭鍗婂緞
                            if coords[0] is not None:
                                inner_radius = 0
                                outer_radius = radius + 150
                                # 纭繚涓嶈秴杩囨渶澶у崐寰?
                                if outer_radius > dataset['r_ticks'][-1] * arg_a + arg_b:
                                    outer_radius = radius
                            else:
                                inner_radius = 0
                                outer_radius = radius
                            print(f"褰撳墠澶勭悊: {item_name}, 鍗婂緞鑼冨洿: {inner_radius}-{outer_radius}")
                            # 瑁佸壀骞舵斁澶у浘鍍?
                            scale_factor = 2
                            amplifier_image_path = os.path.join(self.amplifier_image_dir, 
                                                                f'{chart_id}_{grid_type}_{item_name}.png')
                            
                            amplifier_image = self.crop_axis_label_region(amplifier_path, center_x, center_y, target_angle, outer_radius, angle_width, inner_radius,30, scale_factor,r_ticks,arg_a,arg_b)
                            
                            # 淇濆瓨鏀惧ぇ鍥惧儚
                            if amplifier_image.size > 0:
                                cv2.imwrite(amplifier_image_path, amplifier_image)
                            else:
                                print(f"璀﹀憡: 鏃犳硶淇濆瓨鍥惧儚 {amplifier_image_path}锛屽洜涓鸿鍓尯鍩熶负绌?)
                                continue
                            
                            # 鐢熸垚鏀惧ぇ鎻愮ず骞惰皟鐢↙LM
                            amplifier_prompt = self.generate_prompt(item_name, 'amplifier', dataset)
                            amplifier_coords = self.call_llm_response(amplifier_prompt, amplifier_image_path, 
                                                                      item_name, dataset)
                            print(f"鏀惧ぇ缁撴灉: {amplifier_prompt}")
                            # 淇濆瓨鏀惧ぇ缁撴灉
                            if amplifier_coords is not None:
                                self.results_by_image[chart_id]['data'][item_name]['amplifier'] = amplifier_coords[0]
                        except Exception as e:
                            print(f"鉂?鏀惧ぇ澶勭悊寮傚父: {e}")
                except Exception as e:
                    print(f"鉂?澶勭悊{item_name}鏃跺紓甯? {e}")
                    continue
                
                # 淇濆瓨缁撴灉
                if coords is not None:
                    self.results_by_image[chart_id]['data'][item_name][grid_type] = coords
                self.results_by_image[chart_id]['data'][item_name]['origin'] = value
                
                # 鎵撳嵃缁撴灉
                if grid_type == 'with_grid':
                    amplifier_value = self.results_by_image[chart_id]['data'][item_name].get('amplifier', 'N/A')
                    print(f"{item_name} origin:{value} {grid_type}:{coords} amplifier:{amplifier_value}")
                else:
                    print(f"{item_name} origin:{value} {grid_type}:{coords}")
    
    def save_results(self, output_path: str = None) -> None:
        """淇濆瓨缁撴灉鍒癑SON鏂囦欢"""
        if not self.results_by_image:
            print("鉂?娌℃湁缁撴灉鍙繚瀛?)
            return
        
        # 濡傛灉鏈寚瀹氳緭鍑鸿矾寰勶紝鍒欎娇鐢ㄩ粯璁よ矾寰?
        if output_path is None:
            # 鑾峰彇绗竴涓浘琛ㄧ殑绫诲瀷浣滀负鏂囦欢鍚嶇殑涓€閮ㄥ垎
            first_chart_id = next(iter(self.results_by_image.keys()), '')
            chart_type = self.results_by_image.get(first_chart_id, {}).get('chart_type', 'unknown')
            output_path = f'coordinates_by_image_{chart_type}_{self.llm_model}.json'
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results_by_image, f, ensure_ascii=False, indent=4)
            print(f"鉁?缁撴灉宸蹭繚瀛樺埌: {output_path}")
        except Exception as e:
            print(f"鉂?淇濆瓨缁撴灉澶辫触: {e}")

# 涓荤▼搴忓叆鍙?
if __name__ == '__main__':
    # 鍒涘缓璇勪及鍣ㄥ疄渚?
    evaluator = RoseChartEvaluator()
    
    # 鎸囧畾瑕佸鐞嗙殑鍗曚釜JSON鏂囦欢璺緞
    json_file_path = './data/output/rose/result/chart_1761120786_evalution_datasets.json'
    
    # 澶勭悊鍗曚釜鍥惧儚
    evaluator.process_single_image(json_file_path)
    
    # 淇濆瓨缁撴灉
    evaluator.save_results()
