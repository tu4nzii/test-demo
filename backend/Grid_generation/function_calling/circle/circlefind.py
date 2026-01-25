from tkinter import N
import cv2
import numpy as np
import os
from PIL import Image
from google import genai

import json # 用于解析VLM的JSON响应
import traceback

# --- 配置 Google Gemini API Key ---
# !!! 警告：在生产环境中，不要将 API Key 直接硬编码在代码中。
# !!! 而是使用环境变量、配置文件或秘密管理服务。
API_KEY = "AIzaSyDg8boGTENhxXrKO2qNBTbQNCVCDIeiACs"
client = genai.Client(api_key=API_KEY)
def find_center_by_hough_transform(image_path):
    """
    使用霍夫圆变换来寻找图表中心。
    适用于极坐标图、雷达图等有明显圆形的图表。
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return image, None

    # 2. 预处理：转换为灰度图并进行高斯模糊以减少噪声
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)


    # 根据图像文件名确定参数以更好地调整
    # 这是一种启发式方法，在实际系统中，您可以使用更健壮的图像分析
    # 或用户输入/配置。

    min_dist_val = 10 # 默认值
    param1_val = 20  # 默认值
    param2_val = 30   # 默认值
    min_radius_val = 10 # 默认值
    max_radius_val = 300 # 默认值

    fname = os.path.basename(image_path)

    # if "circle8" in fname:
    #     # 对于旭日图，中心的白色孔是最突出的圆。
    #     # 它相对较小且边缘清晰。
    #     min_dist_val = 20  # 允许更近的圆，对于同心圆很重要
    #     param1_val = 70    # Canny 的更强边缘
    #     param2_val = 25    # 更宽容的累加器阈值以找到中心圆
    #     min_radius_val = 10 # 最小内圈半径
    #     max_radius_val = 80 # 直到中心扇区的大小
    #     print(f"为 {fname} 应用特定参数: minDist={min_dist_val}, param1={param1_val}, param2={param2_val}, minRadius={min_radius_val}, maxRadius={max_radius_val}")
    # else:
    #     print(f"为 {fname} 应用默认参数: minDist={min_dist_val}, param1={param1_val}, param2={param2_val}, minRadius={min_radius_val}, maxRadius={max_radius_val}")

    # 3. 应用霍夫圆变换
    detected_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist_val,
                                         param1=param1_val, param2=param2_val,
                                         minRadius=min_radius_val, maxRadius=max_radius_val)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        # 只选取第一组检测到的圆
        first_circle = detected_circles[0, 0]
        cx, cy, r = first_circle[0], first_circle[1], first_circle[2]
        r = round(r+0.5)
        center_coordinates = (cx, cy)
        # 绘制该圆
        cv2.circle(image, center_coordinates, r, (0, 255, 0), 2)
        cv2.circle(image, center_coordinates, 2, (0, 0, 255), -1)
        cv2.putText(image, f"Center: ({cx}, {cy})", (cx + 15, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"方法二 (霍夫变换) 第一组圆的中心点,半径像素: {center_coordinates,r}")
        return image, center_coordinates, r
    else:
        print("方法二 (霍夫变换) 失败: 未检测到任何圆形。")
        return image, None, None



def get_radial_chart_scale_info_from_vlm(cv_image, center_x, center_y, hough_radius_px):
    """
    通过 Google Gemini VLM 获取极坐标图的刻度信息。
    接收 cv2 图像对象作为输入。
    """
    print(f"--- 调用 Google Gemini VLM ---")
    
    # 将 OpenCV 图像（BGR 格式）转换为 PIL Image（RGB 格式）
    # OpenCV 默认是 BGR，PIL 默认是 RGB，需要转换通道顺序
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    # 构造给 VLM 的提示词。
    prompt = f"""
    这是一张极坐标图（例如雷达图或极坐标散点图）。
    图片中已经用绿色圆圈标出了一个重要的同心圆，其中心大致在 ({center_x}, {center_y}) 像素位置。
    这个绿色圆圈的像素半径是 {hough_radius_px}。

    请您分析图片内容，并提供以下信息：
    1. 这个**绿色圆圈**（霍夫圆）在图表上对应的tick是多少？以circle2为例该tick为90
    2. 图表的**中心点**（即像素半径为0的位置）对应的刻度值是多少？如果中心没有明确标注刻度（即不为0），请根据现有刻度推断出最内圈（最小）的刻度值作为中心值（如这张图为65）。
    3. 图表中所有**同心圆刻度线中，即最外圈的刻度，最大的刻度值是多少？**
    请以严格的 JSON 格式返回这些信息，不要包含任何额外文字或解释，例如：
    ```json
    {{
        "hough_circle_value": <霍夫圆半径>,
        "polar_center_value": <极坐标中心刻度值>,
        "max_tick_value": <最大刻度值>
    }}
    ```
    如果无法识别某个值，请使用 `null`。例如，如果 `polar_center_value` 是0，请返回`0`。
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[pil_image, prompt] # 直接传递PIL图像对象
        )
        vlm_raw_text = response.text.strip()
        print(f"Gemini VLM 原始响应:\n{vlm_raw_text}")

        # 尝试解析 JSON 响应
        if vlm_raw_text.startswith("```json"):
            vlm_raw_text = vlm_raw_text.strip("```json").strip("```").strip()

        vlm_data = json.loads(vlm_raw_text)

        # 提取信息，使用 .get() 方法安全地访问字典，避免 KeyError
        hough_val = vlm_data.get("hough_circle_value")
        polar_center_val = vlm_data.get("polar_center_value")
        max_tick_val = vlm_data.get("max_tick_value")

        return_info = {
            "hough_circle_value": hough_val,
            "polar_center_value": polar_center_val,
            "max_tick_value": max_tick_val
        }
        
        print(f"解析后的 VLM 信息: {return_info}")
        return return_info

    except Exception as e:
        print(f"调用 Google Gemini VLM 失败或解析响应失败: {e}")
        # 返回一个包含 None 值的字典，以便主函数能优雅地处理错误
        return {
            "hough_circle_value": None,
            "polar_center_value": None,
            "max_tick_value": None
        }
def encrypt_radial_chart_with_tick(image_path,interval):
    '''
    对极坐标图进行刻度加密。
    它首先通过霍夫变换识别图表中心和主要圆，然后调用 VLM 获取刻度信息，
    最后根据线性映射在图表的每个刻度间隔中绘制一个加密圆。

    Args:
        image_path (str): 输入图像文件的路径。

    Returns:
        numpy.ndarray: 绘制了加密圆的图像，如果处理失败则返回 None。
    '''
    # 1. 霍夫变换寻找图表中心和霍夫圆
    initial_image, center_coords, hough_radius_px = find_center_by_hough_transform(image_path)
    
    if center_coords is None or hough_radius_px is None:
        print(f"未能从 {image_path} 中找到霍夫圆信息。")
        return None

    center_x, center_y = center_coords

    # 2. 调用 VLM 获取刻度信息
    scale_info = get_radial_chart_scale_info_from_vlm(initial_image, center_x, center_y, hough_radius_px)

    hough_circle_value = scale_info.get("hough_circle_value")
    chart_origin_value = scale_info.get("polar_center_value")
    max_tick_value = scale_info.get("max_tick_value")

    if None in [hough_circle_value, chart_origin_value, max_tick_value]:
        print(f"VLM 未能提供完整的刻度信息，无法加密 {image_path}。")
        return None

    # 3. 计算值到像素半径的映射
    # 假设刻度值与像素半径之间是线性关系
    # pixels_per_value = 霍夫圆像素半径 / (霍夫圆刻度值 - 中心刻度值)
    if (hough_circle_value - chart_origin_value) == 0:
        print("错误：霍夫圆刻度值与中心刻度值相同，无法计算像素映射。")
        return None
        
    pixels_per_value = float(hough_radius_px) / (hough_circle_value - chart_origin_value)
    print(f"每单位刻度值对应的像素数: {pixels_per_value}")

    # 4. 在每个现有刻度间隔中添加一个加密圆
    current_image = initial_image.copy() # 在绘制了霍夫圆的图像基础上进行

    # 遍历所有可能的刻度间隔

    current_tick = chart_origin_value
    while current_tick <= max_tick_value:
        # 计算加密圆的像素半径
        encrypted_radius_px = round((current_tick - chart_origin_value) * pixels_per_value + 0.5)

        # 在图像上绘制加密圆（使用不同颜色或线型以便区分，这里用洋红色）
        # 注意：圆的颜色是 BGR 格式 (255, 0, 255) 是洋红色
        cv2.circle(current_image, center_coords, encrypted_radius_px, (255, 0, 255), 1)
        
        # 在加密圆旁边标注 tick 值
        text_x = center_coords[0] + encrypted_radius_px + 10
        text_y = center_coords[1]
        cv2.putText(current_image, str(current_tick), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        print(f"添加了刻度为 {current_tick} 的加密圆，像素半径为 {encrypted_radius_px}")
        
        current_tick += interval

    return current_image

# --- 主运行部分 ---
if __name__ == "__main__":
    img_dir = "./data/basic_images/circle" 
    out_dir = "./data/out_encrypted_gemini_multiple" # 更改输出目录以区分多点加密
    os.makedirs(out_dir, exist_ok=True) # 确保输出目录存在

    print(f"请确保您的图片文件（例如 circle1.jpg, circle6.jpg, circle7.jpg）位于 '{img_dir}' 目录下。")

    # 遍历指定目录下的所有图片文件
    # for fname in os.listdir(img_dir):
    #     if fname.lower().startswith("circle") and fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
    #         img_path = os.path.join(img_dir, fname)
    #         print(f"\n--- 处理图片: {img_path} ---")
            
    #         # 调用加密函数，它会在内部处理霍夫变换、VLM调用和加密圆绘制
    #         encrypted_image = encrypt_radial_chart_with_tick(img_path)
            
    #         # 构造输出文件路径
    #         base_name = os.path.splitext(fname)[0]
    #         out_path = os.path.join(out_dir, 
    #                                 f"{base_name}_encrypted_intervals.jpg") # 文件名表示加密了多个间隔
            
    #         if encrypted_image is not None:
    #             cv2.imwrite(out_path, encrypted_image)
    #             print(f"加密结果已保存到 {out_path}")
    #         else:
    #             print(f"处理 {fname} 失败，未生成加密图像。")
    fname = "circle2.png"
    img_path = os.path.join(img_dir, "circle2.png")
    print(f"\n--- 处理图片: {img_path} ---")
    interval = 2.5
    # 调用加密函数，它会在内部处理霍夫变换、VLM调用和加密圆绘制
    encrypted_image = encrypt_radial_chart_with_tick(img_path,interval)
            
    # 构造输出文件路径
    base_name = os.path.splitext(fname)[0]
    out_path = os.path.join(out_dir, f"{base_name}_encrypted_intervals.jpg") # 文件名表示加密了多个间隔
            
    if encrypted_image is not None:
        cv2.imwrite(out_path, encrypted_image)
        print(f"加密结果已保存到 {out_path}")
    else:
        print(f"处理 {fname} 失败，未生成加密图像。")