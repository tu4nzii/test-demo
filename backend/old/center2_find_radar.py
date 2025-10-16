from ctypes import DEFAULT_MODE
from email import message
from gettext import find
import re
from unittest import result
import cv2
import base64
from matplotlib import image
import numpy as np
from numpy.random import f
from openai import OpenAI
from openai.types.chat.chat_completion_assistant_message_param import ContentArrayOfContentPart
from urllib3 import response
import json 
import requests
import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
api_key = "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
url = "https://api.vveai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}
tick_denstiy = 2
# 修改后的代码 - 添加图像缩放功能
def show_image_with_scaling(window_name, image, max_width=800, max_height=600):
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算缩放比例
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height, 1.0)  # 不放大图像

    # 计算新尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))

    # 显示缩放后的图像
    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_vertical_text(image, text, x, y, font_size=12, color=(0, 0, 0)):
    """
    在图像上绘制垂直于x轴的文本
    :param image: 输入图像(PIL格式)
    :param text: 要绘制的文本
    :param x: 文本起始x坐标
    :param y: 文本起始y坐标
    :param font_size: 字体大小
    :param color: 文本颜色
    :return: 绘制后的图像
    """
    draw = ImageDraw.Draw(image)
    try:
        # 尝试加载系统字体
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        # 如果找不到指定字体，使用默认字体
        font = ImageFont.load_default()

    # 绘制垂直文本（旋转90度）
    for i, char in enumerate(text):
        # 计算每个字符的位置
        char_x = x
        char_y = y + i * font_size
        # 绘制单个字符
        draw.text((char_x, char_y), char, font=font, fill=color)

    return image


def extract_json_response(content: str):
    try:
        match = re.search(r'(\{[\s\S]*\})', content)
        if not match:
            return None
        json_str = match.group(1)
        return json.loads(json_str)
    except Exception as e:
        print(f"❌ JSON解析失败: {e}")
        return None

def visualize_ring_mask(image_path, ring_width=5):
    # 1. 读取并预处理图像
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 2. 第一次霍夫变换检测第一个圆
    #radar max R = height / 4
    #rose min R = 40
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                             param1=20, param2=30, minRadius = int(height/4), maxRadius=int(height))
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        first_circle = circles[0, 0]
        cx, cy, r = first_circle[0], first_circle[1], first_circle[2]
        
        # 3. 创建环状掩膜
        mask = np.zeros_like(gray)
        # 外圆
        cv2.circle(mask, (cx, cy), int(r+ring_width), 255, -1)
        # 内圆
        cv2.circle(mask, (cx, cy), int(r-ring_width), 0, -1)
        
        # 4. 应用环状掩膜
        masked_blurred = image.copy()
        masked_blurred[mask == 255] = 255
        
        # 5. 显示结果
        # cv2.imshow('Original', image)
        # cv2.imshow('Ring Mask', mask)
        # cv2.imshow('After Ring Masking', masked_blurred)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return masked_blurred,r,cx,cy
    else:
        return image

def second_circle_find(image,r):
    # 2. 第一次霍夫变换检测第一个圆    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                             param1=20, param2=50, minRadius=r+30, maxRadius=int(height / 2))
    second_r = r+50
    if circles is not None:
        circles = np.uint16(np.around(circles))
        second_circle = circles[0, 0]
        cx, cy, r = second_circle[0], second_circle[1], second_circle[2]
        cv2.circle(image, (second_circle[0], second_circle[1]), second_circle[2], (255, 0, 0), 1)
        cv2.putText(image, f"Center: ({second_circle[0]}, {second_circle[1]})", (second_circle[0] + 15, second_circle[1] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.circle(image, (second_circle[0], second_circle[1]), 2, (0, 0, 255), -1)
        # cv2.imshow("Second Circle", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        second_r = second_circle[2]
    return second_r

def crop_tick_region(image, center_x, center_y, target_radius, pixel_range=25):
    """
    裁剪指定半径附近的环形区域
    :param image: 原始图像
    :param center_x: 圆心X坐标
    :param center_y: 圆心Y坐标
    :param target_radius: 目标tick半径
    :param pixel_range: 裁剪范围（±像素）
    :return: 裁剪后的图像
    """
    # 创建与原图相同大小的掩码
    mask = np.zeros_like(image)
    
    # 计算内外圆半径（确保内圆半径不为负）
    outer_radius = target_radius + pixel_range
    inner_radius = max(0, target_radius - 10)
    
    # 绘制环形掩码（白色为保留区域）
    cv2.circle(mask, (center_x, center_y), outer_radius, (255, 255, 255), -1)
    cv2.circle(mask, (center_x, center_y), inner_radius, (0, 0, 0), -1)
    
    # 应用掩码获取环形区域
    masked_image = cv2.bitwise_and(image, mask)
    
    # 裁剪最小外接矩形以去除多余黑色区域
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 获取最大轮廓的边界框
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped_image = masked_image[y:y+h, x:x+w]
        #show_image_with_scaling('Cropped Tick Region', cropped_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return cropped_image
    
    # 如果没有找到有效区域，返回原始掩码图像
    return masked_image

def find_tick(target_radius,center_x,center_y,image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image}")
    cropped_image = crop_tick_region(image, center_x, center_y, target_radius,pixel_range=25)
    cv2.imshow('Cropped Tick Region', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if len(cropped_image.shape) == 3 and cropped_image.shape[2] == 3:
        rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
    retval, buffer = cv2.imencode('.png', rgb_image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    prompt = f"""
    这是一张圆环图
    图片中已经用绿色圆圈标出了一个重要的圆
    请您分析图片内容，并提供该信息：
    1. 这个**绿色圆圈**对应的的刻度值是多少？（会出现在该圆周围，该刻度值仅为一个数值(如50% = 50)，且仅出现在图片中）
    **只读取存在的数**
    **注意,仅读取图上原本的数值，而不做任何推算**
    **仔细检查图片，确保读取的数值是正确的**
    **不存在300！！！若识别为300，则是识别错误，实际为200**

    请以严格的 JSON 格式返回这些信息，不要包含任何额外文字或解释，例如：
    ```json
    {{
        "tick": <刻度值>,
        "res":<分析过程>
    }}
    ```
    **如果有多个数字，使用null**
    如果无法识别某个值，请使用 `null`。
    **再次声明，如果没有数字，则使用null**
    **再次声明，如果有多个数字，使用null**
    **若包含字母，则为null**
    **若为0则为null**
    """
    payload = {
        "model": "gpt-4.1",
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
        response = requests.post(url=url, headers=headers, json=payload)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        data = extract_json_response(content)
        return data
    except requests.exceptions.RequestException as e:

        print(f"API请求失败: {e}")
        return None

def call_llm_response(image_path: str,center_x,center_y):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    prompt = f"""
    这是一张极坐标图（例如雷达图或极坐标散点图）。
    其中心大致在 ({center_x}, {center_y}) 像素位置。


    请您分析图片内容，并提供以下信息:
    1. 图表中所有**同心圆刻度线中，即最外圈的刻度，最大的刻度值是多少？**
    2. 图表中所有**同心圆刻度线中，即最外圈的刻度，最小的刻度值是多少？**
    3. 图表中所有**同心圆刻度的间隔，刻度为径向刻度，而非环状刻度**

    **注意,仅读取图上原本的数值，而不做任何推算**
    且仅返回数值，如（50% = 50）

    请以严格的 JSON 格式返回这些信息，不要包含任何额外文字或解释，例如：
    ```json
    {{
        "max_tick_value": <最大刻度值>
        "min_tick_value": <最小刻度值>
        "tick_interval": <刻度间隔>
        "res":<分析过程>
    }}
    该图的最大刻度为100，最小刻度为50 间隔为50
    ```
    如果无法识别某个值，请使用 `null`。例如，如果 `max_tick_value` 是0，请返回`0`。
    """
    payload = {
        "model": "gpt-4.1",
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
        response = requests.post(url=url, headers=headers, json=payload)
        #print(response.json())
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        data = extract_json_response(content)
        #print(data)

        return data
    except requests.exceptions.RequestException as e:

        print(f"API请求错误: {e}")
        print(f"响应内容: {response.text if 'response' in locals() else '无响应'}")

def encrypt_radial_chart_with_tick(image_path,tick_interval,center_x,center_y,tick1,tick2,r1,r2,max_tick_value,min_tick_value):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    # 计算半径与tick的线性关系
    # r = a*tick + b
    if(r2 < r1):
        r2,r1 = r1,r2
    interval = tick_interval / tick_denstiy
    #a = (r2 - r1) / (tick2 - tick1)
    #rose
    a = r1 / max_tick_value 
    # 创建副本用于绘制
    # if(min_tick_value > 0):
    #     pixels_per_value = float(r1) / (tick1 - (min_tick_value - tick_interval))
    # else:
    #     pixels_per_value = float(r1) / (tick1)
    # a = (pixels_per_value + a) / 2
    b = r1 - a * max_tick_value
    result = image.copy()
     # 设置字体和颜色
    font_CV = cv2.FONT_HERSHEY_DUPLEX 
    # 计算图像面积
    image_area = height * width
    scale = math.sqrt(image_area)
    # 以面积的平方根为基准计算缩放比例，0.0006 是根据实际情况调整的系数
    font_scale = scale * 0.006
    font_color = (0, 0, 0)  # 红色
    line_color = (128, 128, 128)
    thickness = 1
    #绘制加密圆环并标注tick
    tick = max_tick_value
    # 用于存储生成的r_tick值
    r_ticks = []
    argument = {
        'a': a,
        'b': b
    }
    current_px_distance = 10000
    radius = 0
    count = 0
    while ( tick > 0 and radius >= 0):

        tick -= interval
        if(tick < 0):
            break
        # 计算当前tick对应的半径
        radius = int(a * tick + b)
        current_px_distance = abs(radius - 0)
        if radius <= 0:
            print(f"⚠️ 跳过无效半径: tick={tick}, 计算半径={radius}")
            continue
        if current_px_distance <= 3:
            print(f"已达到圆心附近，停止绘制 (tick={tick}, radius={radius})")
            break
        # 绘制圆环
        text_x_up = center_x 
        text_y_up = center_y - radius 

        # 绘制右侧标注
        if(tick % 1 == 0):
            tick = int(tick)
        if(tick > 0):  
            pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
        
        # 设置更大的字体
            try:
                font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))  # 大号字体
            except IOError:
                font = ImageFont.load_default()
                font.size = 30
            
            text = str(tick)
        # 获取文本边界框
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # 右侧旋转90度文本
            temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
            temp_draw_right = ImageDraw.Draw(temp_img_right)
            temp_draw_right.text((0, 0), text, font=font, fill=font_color)
            rotated_right = temp_img_right.rotate(-90, expand=True)  # 顺时针旋转90度
        # 调整右侧位置
            pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                    center_y - rotated_right.size[1]//2)
            pil_img.paste(rotated_right, pos_right, rotated_right)
        
        # 左侧旋转-90度文本
            temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
            temp_draw_left = ImageDraw.Draw(temp_img_left)
            temp_draw_left.text((0, 0), text, font=font, fill=font_color)
            rotated_left = temp_img_left.rotate(90, expand=True)  # 逆时针旋转90度
        # 调整左侧位置
            pos_left = (center_x - radius - int(width*0.0125), 
                   center_y - rotated_left.size[1]//2)
            
            # 底部正常文本
            text_x_bottom = center_x - text_width//2
            text_y_bottom = center_y + radius - int(height*0.0122) 
            draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)  # 这里使用已定义的text
            
            pil_img.paste(rotated_left, pos_left, rotated_left)
        
        
        # 转换回OpenCV格式
            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 新增：保存生成的tick值
        r_ticks.append(tick)
        count+=1
        if(count % tick_denstiy == 0):
            continue
        cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1,lineType=cv2.LINE_AA)
        # 绘制虚线圆环
        circumference = int(2 * math.pi * radius)
        # dash_length 表示虚线中每段实线的长度（像素）
        dash_length = 2
        # gap_length 表示虚线中每段实线之间的间隔长度（像素）
        gap_length = 3
        for i in range(0, circumference, dash_length + gap_length):
            angle_start = 2 * math.pi * i / circumference
            angle_end = 2 * math.pi * (i + dash_length) / circumference
            x1 = int(center_x + radius * math.cos(angle_start))
            y1 = int(center_y + radius * math.sin(angle_start))
            x2 = int(center_x + radius * math.cos(angle_end))
            y2 = int(center_y + radius * math.sin(angle_end))
            cv2.line(result, (x1, y1), (x2, y2), line_color, thickness, lineType=cv2.LINE_AA)
        # 在圆环右侧标注tick值
    
    # 保存结果
    # cv2.imwrite('encrypted_circles.png', result)
    # cv2.imshow("image",result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result , r_ticks , argument

def encrypt_radial_chart_one_tick(image_path,tick_interval,center_x,center_y,tick1,r,max_tick_value,min_tick_value):
    image = cv2.imread(image_path)
    # 计算半径与tick的线性关系
    # r = a*tick + b
    height, width = image.shape[:2]
    interval = tick_interval / tick_denstiy
    
    # 创建副本用于绘制
    if(min_tick_value > 0):
        pixels_per_value = float(r) / (tick1 - (min_tick_value - tick_interval))
    else:
        pixels_per_value = float(r) / (tick1)
    a = pixels_per_value 
    b = r - a * tick1
    result = image.copy()
     # 设置字体和颜色
    font_CV = cv2.FONT_HERSHEY_DUPLEX 
    image_area = height * width
    scale = math.sqrt(image_area)
    # 以面积的平方根为基准计算缩放比例，0.0006 是根据实际情况调整的系数
    font_scale = scale * 0.006
    font_color = (0, 0, 0)  # 红色
    line_color = (128, 128, 128)
    thickness = 1
    #绘制加密圆环并标注tick
    tick = max_tick_value
    # 用于存储生成的r_tick值
    r_ticks = []
    argument = {
        'a': a,
        'b': b
    }
    current_px_distance = 10000
    radius = 0
    count = 0
    while ( tick > 0 and radius >= 0):

        tick -= interval
        # 计算当前tick对应的半径
        radius = int(a * tick + b)
        current_px_distance = abs(radius - 0)
        if radius <= 0:
            print(f"⚠️ 跳过无效半径: tick={tick}, 计算半径={radius}")
            continue
        if current_px_distance <= 3:
            print(f"已达到圆心附近，停止绘制 (tick={tick}, radius={radius})")
            break
        # 绘制圆环
        text_x_up = center_x 
        text_y_up = center_y - radius 

        # 绘制右侧标注
        if(tick % 1 == 0):
            tick = int(tick)
        if(tick > 0):  
            pil_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
        
        # 设置更大的字体
            try:
                font = ImageFont.truetype("arial.ttf", size=int(0.025*scale))  # 大号字体
            except IOError:
                font = ImageFont.load_default()
                font.size = 30
            
            text = str(tick)
        # 获取文本边界框
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        
        # 右侧旋转90度文本
            temp_img_right = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
            temp_draw_right = ImageDraw.Draw(temp_img_right)
            temp_draw_right.text((0, 0), text, font=font, fill=font_color)
            rotated_right = temp_img_right.rotate(-90, expand=True)  # 顺时针旋转90度
        # 调整右侧位置
            pos_right = (center_x + radius - rotated_right.size[0] + int(width*0.0125), 
                    center_y - rotated_right.size[1]//2)
            pil_img.paste(rotated_right, pos_right, rotated_right)
        
        # 左侧旋转-90度文本
            temp_img_left = Image.new('RGBA', (text_width, text_height+100), (255, 255, 255, 0))
            temp_draw_left = ImageDraw.Draw(temp_img_left)
            temp_draw_left.text((0, 0), text, font=font, fill=font_color)
            rotated_left = temp_img_left.rotate(90, expand=True)  # 逆时针旋转90度
        # 调整左侧位置
            pos_left = (center_x - radius - int(width*0.0125), 
                   center_y - rotated_left.size[1]//2)
            
            # 底部正常文本
            text_x_bottom = center_x - text_width//2
            text_y_bottom = center_y + radius - int(height*0.0122) 
            draw.text((text_x_bottom, text_y_bottom), text, font=font, fill=font_color)  # 这里使用已定义的text
            
            pil_img.paste(rotated_left, pos_left, rotated_left)
        
        
        # 转换回OpenCV格式
            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 新增：保存生成的tick值
        r_ticks.append(tick)
        count+=1
        if(count % tick_denstiy == 0):
            continue
        cv2.putText(result, str(tick), (text_x_up, text_y_up), font_CV, font_scale*0.1, font_color, 1,lineType=cv2.LINE_AA)
        # 绘制虚线圆环
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
        # 在圆环右侧标注tick值
    
    # 保存结果
    # cv2.imwrite('encrypted_circles.png', result)
    # cv2.imshow("image",result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result, r_ticks, argument

    
    
if __name__ == "__main__":
    
    image_types = [
        #{"type": "radar", "input_dir": "./data/circle/radar/", "output_dir": "./data/out/circle/radar/"},
        {"type": "rose", "input_dir": "./data/circle/rose/", "output_dir": "./data/out/circle/rose/"}
    ]
    for img_type in image_types:
        input_dir = img_type["input_dir"]
        output_dir = img_type["output_dir"]
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在e
        for fname in os.listdir(input_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                pass
                print(f"Processing {img_type['type']} image: {fname}")
                coords = [0, 0]
                image_path_1 = os.path.join(input_dir, fname)

                try:
                    #找第二个圆
                    second_circle, first_r, coords[0], coords[1] = visualize_ring_mask(image_path_1)
                    second_r = second_circle_find(second_circle, first_r)
                    print(coords)
                    print(first_r)
                    print(second_r)
                    fname2 = fname.replace(".png", "") + "+encode" + ".jpg"
                    output_path2 = os.path.join(output_dir, fname2)
                    json_fname = fname.replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json')
                    json_path = os.path.join(input_dir, json_fname)

                    #画出两个圆
                    image = cv2.imread(image_path_1)
                    cv2.circle(image, (coords[0], coords[1]), first_r, (0, 255, 0), 1)  # 绿色外圆
                    cv2.circle(image, (coords[0], coords[1]), second_r, (0, 255, 0), 1)  # 绿色内圆
                    # 添加中心点标记
                    cv2.circle(image, (coords[0], coords[1]), 2, (255, 0, 0), -1)  # 蓝色实心中心点
                    #保存中间结果（即识别出的两个圆）

                    temp_output_path = os.path.join(output_dir, f"temp_{fname}")
                    cv2.imwrite(temp_output_path, image)
                    print(f"Temporary image saved to: {temp_output_path}")
                    image_out = cv2.imread(temp_output_path)

                    # 识别刻度
                    result_1 = find_tick(first_r, coords[0], coords[1], temp_output_path)
                    tick1 = result_1.get("tick")
                    reason1 = result_1.get("res")
                    result_2 = find_tick(second_r, coords[0], coords[1], temp_output_path)
                    tick2 = result_2.get("tick")
                    reason2 = result_2.get("res")
                    print(reason1)
                    print(tick1)
                    print(reason2)
                    print(tick2)

                     # 调用LLM获取刻度信息
                    response_data = call_llm_response(temp_output_path, coords[0], coords[1])
                    max_tick_value = response_data.get("max_tick_value")
                    min_tick_value = response_data.get("min_tick_value")
                    tick_interval = response_data.get("tick_interval")
                    res = response_data.get("res")
                    print(tick1, tick2, max_tick_value, res)

                    # 加密处理
                    argument = {}
                    if(tick1 and tick2):
                        result, r_ticks, argument = encrypt_radial_chart_with_tick(image_path_1, tick_interval, coords[0], coords[1], tick1, tick2, first_r, second_r, max_tick_value, min_tick_value)
                    elif(tick1 and tick2 == None):
                        result, r_ticks, argument = encrypt_radial_chart_one_tick(image_path_1, tick_interval, coords[0], coords[1], tick1, first_r, max_tick_value, min_tick_value)
                    elif(tick1 == None and tick2):
                        result, r_ticks, argument = encrypt_radial_chart_one_tick(image_path_1, tick_interval, coords[0], coords[1], tick2, second_r, max_tick_value, min_tick_value)
                    elif(tick1 == None and tick2 == None):
                        print("未识别出正确的刻度")
                        continue
                    # 保存最终结果
                    print(output_path2)
                    print(json_path)
                    #添加r_ticks
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                            print(True,"1")
                        # 添加r_ticks字段
                        r_ticks = r_ticks[::-1]
                        r_ticks.append(r_ticks[-1]+(tick_interval / tick_denstiy))
                        json_data['r_ticks'] = r_ticks
                        #添加预测圆心
                        pred_coords = []
                        pred_coords.append(int(coords[0]))
                        pred_coords.append(int(coords[1]))
                        real_coords = []
                        # real_coords.append(json_data['center'][0])
                        # real_coords.append(json_data['center'][1])
                        real_coords.append(json_data['center']['x'])
                        real_coords.append(json_data['center']['y'])
                        err_center = np.linalg.norm(np.array(pred_coords) - np.array(real_coords))
                        json_data['err_center'] = err_center
                        print(pred_coords)
                        json_data['pred_coords'] = pred_coords
                        #添加线性参数
                        json_data['argument'] = argument
                        # 保存到输出目录
                        output_json_path = os.path.join(output_dir, json_fname)
                        with open(output_json_path, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        print(f"Updated JSON saved to: {output_json_path}")
                    else:
                        print(f"JSON file not found: {json_path}")
                    cv2.imwrite(output_path2, result)

                    print(f"Encrypted image saved to: {output_path2}")

                    # 清理临时文件
                    #os.remove(temp_output_path)

                except Exception as e:
                    print(f"Error processing {fname}: {e}")
                    continue
    """        
    coords = [0,0]
    fname = "circle21.jpg"
    image_path_1 = "./data/basic_images/circle/{fname}".format(fname=fname)
    second_circle,first_r,coords[0],coords[1] = visualize_ring_mask(image_path_1)
    second_r = second_circle_find(second_circle,first_r)
    print(coords)
    print (first_r)
    print(second_r)
    image = cv2.imread(image_path_1)

    cv2.circle(image, (coords[0], coords[1]), first_r, (0, 255, 0), 1)  # 红色外圆
    cv2.circle(image, (coords[0], coords[1]), second_r, (0, 255, 0),1)  # 绿色内圆
    
    # 添加中心点标记
    cv2.circle(image, (coords[0], coords[1]), 5, (255, 0, 0), -1)  # 蓝色实心中心点
    output_dir = "./data/out/circle2/"
    os.makedirs(output_dir, exist_ok=True)  # 确保目录存在
    output_path = os.path.join(output_dir, fname)
    cv2.imwrite(output_path, image)
    print(f"Image saved to: {output_path}")
    image_out = cv2.imread(output_path)

    # 添加文字标注
    # cv2.putText(image, f"R1: {first_r}", (coords[0]+10, coords[1]+30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    # cv2.putText(image, f"R2: {second_r}", (coords[0]+10, coords[1]+60), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    image_path = "./data/out/circle2/{fname}".format(fname=fname)
    #crop_tick_region(image_out,coords[0],coords[1],first_r)
    result_1 = find_tick(first_r,coords[0],coords[1],output_path)
    tick1 = result_1.get("tick")
    reason1 = result_1.get("res")
    result_2 = find_tick(second_r,coords[0],coords[1],output_path)
    tick2 = result_2.get("tick")
    reason2 = result_2.get("res")
    
    print(reason1)
    print(tick1)
    print(reason2)
    print(tick2)

    # print(response_data.json())    

    # 初始call llm 


    #tick1 = response_data.get("tick1")
    #tick2 = response_data.get("tick2")
    response_data = call_llm_response(image_path,coords[0],coords[1])
    max_tick_value = response_data.get("max_tick_value")
    min_tick_value = response_data.get("min_tick_value")
    tick_interval = response_data.get("tick_interval")
    res = response_data.get("res")
    print(tick1,tick2,max_tick_value,res)
    #interval = 1
    if(tick1 and tick2):
        result = encrypt_radial_chart_with_tick(image_path_1,tick_interval,coords[0],coords[1],tick1,tick2,first_r,second_r,max_tick_value,min_tick_value) 
    elif(tick1 and tick2 == None):
        result = encrypt_radial_chart_one_tick(image_path_1,tick_interval,coords[0],coords[1],tick1,first_r,max_tick_value,min_tick_value) 
    elif(tick1 == None and tick2):
        result = encrypt_radial_chart_one_tick(image_path_1,tick_interval,coords[0],coords[1],tick2,second_r,max_tick_value,min_tick_value) 
    elif(tick1 == None and tick2 == None):
        print("未识别出正确的刻度")
    #写入结果
    fname2 = fname.replace(".jpg","") + "+encode" + ".jpg"
    output_path2 = os.path.join(output_dir, fname2)
    cv2.imwrite(output_path2, result)
    """
"""
仍存在问题：
1. 图片规范性问题
2. 最内侧的判断问题
"""
