# -*- coding: utf-8 -*-
"""
鐙珛鐨凧SON瑙ｆ瀽妯″潡锛岀敤浜庡鐞嗘ā鍨嬭緭鍑虹殑JSON鏍煎紡瑙ｆ瀽锛堢洿瑙掑潗鏍囩郴涓撶敤锛?
"""

import json
import os
import re
import asyncio
import aiohttp
from aiohttp import ClientTimeout

from prediction_core.model_config import get_chat_completion_url, get_headers, get_model_name


GEMINI_URL = get_chat_completion_url()
GEMINI_HEADERS = get_headers()
GEMINI_MODEL = get_model_name()


def normalize_value(data):
    """
    閫掑綊灏嗘暟鎹腑鐨勬暟鍊艰浆鎹负float绫诲瀷
    璇嗗埆骞惰浆鎹㈢櫨鍒嗘瘮瀛楃涓诧紝閬垮厤灏嗗勾浠界瓑瀛楃涓茶浆鎹负鏁板€?
    """
    if isinstance(data, dict):
        return {k: normalize_value(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 鍙浆鎹㈠垪琛ㄤ腑鐨勭涓€涓厓绱狅紙鍋囪鏄暟鍊硷級锛岀浜屼釜鍏冪礌淇濇寔鍘熸牱
        if len(data) >= 2:
            return [normalize_value(data[0]), data[1]]
        else:
            return [normalize_value(item) for item in data]
    elif isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, str):
        # 璇嗗埆骞惰浆鎹㈢櫨鍒嗘瘮瀛楃涓?
        if "%" in data:
            try:
                # 鎻愬彇鐧惧垎姣斿€煎苟杞崲涓篺loat
                return float(data.replace("%", "")) / 100
            except ValueError:
                # 濡傛灉杞崲澶辫触锛屼繚鎸佸師鏍?
                return data
        # 璇嗗埆骞惰浆鎹㈡暟鍊煎瓧绗︿覆
        try:
            return float(data)
        except ValueError:
            # 濡傛灉杞崲澶辫触锛屼繚鎸佸師鏍?
            return data
    else:
        return data


def extract_json_with_regex(txt):
    """
    浣跨敤姝ｅ垯琛ㄨ揪寮忔彁鍙朖SON瀛楃涓?
    浼樺厛浣跨敤浠ｇ爜鍧楄В鏋愶紝鍙繘琛屾渶鍩烘湰鐨勪慨澶?
    瀵逛簬澶嶆潅鏍煎紡鐩存帴杩斿洖None锛岃Gemini鏉ュ鐞?
    """
    try:
        # 1. 棰勫鐞嗭細绉婚櫎鏃犳晥鐨勮浆涔夊瓧绗?
        # 淇濈暀鏈夋晥鐨凧SON杞箟瀛楃
        json_str = re.sub(r'\\(?!(["\\\/bfnrt]|u[0-9a-fA-F]{4}))', '', txt)
        
        # 2. 濡傛灉鏈変唬鐮佸潡 ```json ... ```锛屽厛鎻愬彇閲岄潰鐨?
        codeblock_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
        if codeblock_match:
            json_str = codeblock_match.group(1).strip()
        
        # 3. 鏌ユ壘鏈€澶栧眰鐨勫畬鏁碕SON缁撴瀯
        # 浼樺厛鏌ユ壘浠寮€澶村埌鏈€鍚庝竴涓獇缁撳熬鐨勭粨鏋?
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_candidate = json_str[first_brace:last_brace+1]
        else:
            # 鍚﹀垯鏌ユ壘浠寮€澶村埌鏈€鍚庝竴涓猐缁撳熬鐨勭粨鏋?
            first_bracket = json_str.find('[')
            last_bracket = json_str.rfind(']')
            
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_candidate = json_str[first_bracket:last_bracket+1]
            else:
                # 濡傛灉閮芥病鎵惧埌锛屽皾璇曟煡鎵句换浣曞彲鑳界殑JSON缁撴瀯
                json_objects = re.findall(r"(\{.*\}|\[.*\])", json_str, re.DOTALL)
                if json_objects:
                    json_candidate = max(json_objects, key=len)
                else:
                    raise ValueError("No JSON object or array found in output")
        
        json_str = json_candidate.strip()
        
        # ----------- 鏈€鍩烘湰鐨凧SON淇 ----------- #
        # 1. 绉婚櫎浠ｇ爜鍧楁爣璁?
        json_str = re.sub(r'```json|```', '', json_str)
        
        # 2. 淇鎷彿涓嶅尮閰?
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")
        
        if open_brackets > close_brackets:
            json_str += "]" * (open_brackets - close_brackets)
        if open_braces > close_braces:
            json_str += "}" * (open_braces - close_braces)
        
        # 3. 绉婚櫎澶氫綑鐨勯€楀彿
        json_str = re.sub(r',\s*\]', r']', json_str)
        json_str = re.sub(r',\s*\}', r'}', json_str)
        # ----------------------------------- #

        # 灏濊瘯鐩存帴瑙ｆ瀽锛屽鏋滃け璐ュ垯璁〨emini鏉ュ鐞?
        try:
            parsed = json.loads(json_str)
            parsed = normalize_value(parsed)   # 鉁?缁熶竴 float 鍖?
            return parsed
        except:
            # 瀵逛簬澶嶆潅鏍煎紡锛岀洿鎺ヨ繑鍥濶one璁〨emini澶勭悊
            print("鈿狅笍 姝ｅ垯寮忚В鏋愰亣鍒板鏉傛牸寮忥紝浜ょ粰Gemini澶勭悊")
            return None
    except Exception as e:
        print(f"鈿狅笍 姝ｅ垯寮忚В鏋愬け璐? {e}")
        # 瑙ｆ瀽澶辫触鍚庣洿鎺ヨ繑鍥濶one锛岃Gemini鏉ュ鐞?
        return None


async def call_gemini_for_json_fix(original_output, prompt):
    """
    璋冪敤Gemini妯″瀷鏉ヤ慨澶岼SON鏍煎紡
    """
    # 鏋勯€犱慨澶嶈姹傜殑鎻愮ず璇?
    fix_prompt = f"""
    浣犳槸涓€涓笓涓氱殑JSON鏍煎紡淇宸ュ叿銆傝淇浠ヤ笅鏂囨湰涓殑JSON鏍煎紡閿欒锛岀‘淇濆畠鍙互琚獼SON.parse姝ｇ‘瑙ｆ瀽銆?
    璇峰彧杩斿洖淇鍚庣殑JSON瀛楃涓诧紝涓嶈娣诲姞浠讳綍瑙ｉ噴鎴栧叾浠栧唴瀹广€?

    鍘熷鏂囨湰锛?
    ```
    {original_output}
    ```

    淇鍚庣殑JSON锛?
    """
    
    # 鏋勯€燗PI璇锋眰鐨刾ayload
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a professional JSON formatter. Fix the JSON format errors in the given text."},
            {"role": "user", "content": fix_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    try:
        # 璁剧疆瓒呮椂鏃堕棿涓?鍒嗛挓
        timeout = ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(GEMINI_URL, json=payload, headers=GEMINI_HEADERS) as response:
                if response.status != 200:
                    print(f"鉂?Gemini API璇锋眰澶辫触锛岀姸鎬佺爜: {response.status}")
                    return None
                
                response_data = await response.json()
                
                # 妫€鏌ュ搷搴旀牸寮?
                if not response_data or "choices" not in response_data or not response_data["choices"]:
                    print(f"鉂?Gemini API杩斿洖鏍煎紡閿欒: {response_data}")
                    return None
                
                # 鎻愬彇淇鍚庣殑鍐呭
                fixed_content = response_data["choices"][0]["message"]["content"].strip()
                
                # 灏濊瘯鎻愬彇JSON閮ㄥ垎
                fixed_json = extract_json_with_regex(fixed_content)
                if fixed_json:
                    return fixed_json
                else:
                    # 鍐嶆灏濊瘯鐩存帴瑙ｆ瀽
                    try:
                        parsed = json.loads(fixed_content)
                        return normalize_value(parsed)
                    except:
                        print(f"鉂?Gemini淇鍚庣殑鍐呭浠嶆棤娉曡В鏋? {fixed_content}")
                        return None
    except asyncio.TimeoutError:
        print(f"鉂?Gemini API璇锋眰瓒呮椂")
        return None
    except json.JSONDecodeError as e:
        print(f"鉂?Gemini API杩斿洖鍐呭涓嶆槸鏈夋晥鐨凧SON: {e}")
        return None
    except Exception as e:
        print(f"鉂?Gemini淇杩囩▼涓彂鐢熼敊璇? {e}")
        return None


async def parse_model_output(original_output, prompt):
    """
    瑙ｆ瀽妯″瀷杈撳嚭鐨凧SON鏍煎紡
    浼樺厛浣跨敤姝ｅ垯琛ㄨ揪寮忔彁鍙栵紝澶辫触鏃惰皟鐢℅emini API淇
    """
    # 1. 棣栧厛灏濊瘯浣跨敤姝ｅ垯琛ㄨ揪寮忔彁鍙栧拰淇
    parsed = extract_json_with_regex(original_output)
    if parsed:
        print("Regex JSON parsing succeeded")
        return parsed
    
    # 2. 濡傛灉姝ｅ垯琛ㄨ揪寮忚В鏋愬け璐ワ紝璋冪敤Gemini API淇
    print("馃攧 姝ｅ垯寮忚В鏋愬け璐ワ紝灏濊瘯浣跨敤Gemini淇")
    fixed_parsed = await call_gemini_for_json_fix(original_output, prompt)
    if fixed_parsed:
        print("鉁?Gemini淇鎴愬姛")
        return fixed_parsed
    
    # 3. 鎵€鏈夋柟娉曢兘澶辫触
    print("鉂?鎵€鏈夎В鏋愭柟娉曞潎澶辫触")
    return None


async def process_parsed_json(parsed_json, chart_type):
    """
    澶勭悊瑙ｆ瀽鍚庣殑JSON鏁版嵁锛屾牴鎹浘琛ㄧ被鍨嬭繘琛岄€傞厤
    """
    if not parsed_json:
        return None
    
    # 妫€鏌ユ槸鍚﹀寘鍚玡xists瀛楁
    if "exists" in parsed_json and parsed_json["exists"] is False:
        return {
            "exists": False,
            "datapoints": []
        }
    
    # 鏍规嵁鍥捐〃绫诲瀷澶勭悊鏁版嵁
    if chart_type == "scatter":
        # 鐩磋鍧愭爣绯绘暟鎹鐞?
        if "datapoints" in parsed_json:
            datapoints = parsed_json["datapoints"]
            # 纭繚datapoints鏄垪琛ㄦ牸寮?
            if isinstance(datapoints, list):
                return {
                    "exists": True,
                    "datapoints": datapoints
                }
            else:
                print("鉂?datapoints涓嶆槸鍒楄〃鏍煎紡")
                return None
        else:
            print("鉂?缂哄皯datapoints瀛楁")
            return None
    elif chart_type == "pie":
        # 楗煎浘鏁版嵁澶勭悊
        if "datapoints" in parsed_json:
            datapoints = parsed_json["datapoints"]
            # 纭繚datapoints鏄垪琛ㄦ牸寮?
            if isinstance(datapoints, list):
                return {
                    "exists": True,
                    "datapoints": datapoints
                }
            else:
                print("鉂?datapoints涓嶆槸鍒楄〃鏍煎紡")
                return None
        else:
            print("鉂?缂哄皯datapoints瀛楁")
            return None
    else:
        print(f"鉂?涓嶆敮鎸佺殑鍥捐〃绫诲瀷: {chart_type}")
        return None
