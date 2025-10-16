import json
import os
from typing import List, Dict
from typing import Tuple
import base64
import requests
import re
import time
import cv2
import math
import numpy as np


#json 数据库导入
llm_model = "gemini-2.0-flash"

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

def validate_coordinates(coords: Tuple) -> bool:
    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        return False
    valid = lambda x: isinstance(x, (int, float)) or x is None
    return valid(coords[0]) and valid(coords[1])

#裁剪图像
def encode_image(image,center_x, center_y,arg_a,arg_b,r_ticks):
    line_color = (128, 128, 128)
    thickness = 1
    if(r_ticks[0] == 0):
        r_ticks.pop(0)
    count= 0
    for tick in r_ticks:
        count+=1
        if(count%2==0):
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

    
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    return image


def crop_axis_label_region(image_path, center_x, center_y, angle_deg, outer_radius, angle_width=30, inner_radius=0, label_offset=30, scale_factor=1.0,r_ticks=[],arg_a=0,arg_b=0):
    """
    根据角度裁剪扇形区域（圆心到指定角度方向，带角度宽度和半径范围）
    :param image_path: 图像路径
    :param center_x, center_y: 圆心坐标
    :param angle_deg: 扇形中心角度(度)
    :param outer_radius: 扇形外半径
    :param angle_width: 扇形角度宽度(度)，默认30度
    :param inner_radius: 扇形内半径，默认0（从圆心开始）
    :param label_offset: 兼容旧参数，实际已不再使用
    :return: 裁剪后的扇形区域图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    h, w = image.shape[:2]
    encode_image(image,center_x, center_y,arg_a,arg_b,r_ticks)

    # 创建掩码（黑色背景）
    mask = np.zeros((h, w), dtype=np.uint8)

    # 计算扇形角度范围
    start_angle = angle_deg - angle_width / 2
    end_angle = angle_deg + angle_width / 2
    for tick in r_ticks:
        radius = int(arg_a * tick + arg_b)
         # 在图像上添加角度文字标注
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.3
        font_color = (0, 0, 0)  # 红色
        thickness = 1
    
                # 计算起始角度文字的位置（位于外圆外一点）
        start_angle_rad = math.radians(start_angle+4)
        text_radius = radius 
        
        # 获取文字尺寸以实现中心对齐
        text = str(tick)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 计算文字中心坐标
        start_x_center = int(center_x + text_radius * math.cos(start_angle_rad))
        start_y_center = int(center_y - text_radius * math.sin(start_angle_rad))
        
        # 调整为左下角坐标（考虑文字尺寸）
        start_x = start_x_center - text_size[0] // 2
        start_y = start_y_center + text_size[1] // 2

        # 计算结束角度文字的位置
        end_angle_rad = math.radians(end_angle-4)
        end_x_center = int(center_x + text_radius * math.cos(end_angle_rad))
        end_y_center = int(center_y - text_radius * math.sin(end_angle_rad))
        end_x = end_x_center - text_size[0] // 2
        end_y = end_y_center + text_size[1] // 2
        
        if(tick%1==0):
            tick = int(tick)
        # 添加文字标注（使用计算好的左下角坐标）
        cv2.putText(image, str(tick), (start_x, start_y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(image, str(tick), (end_x, end_y), font, font_scale, font_color, thickness, lineType=cv2.LINE_AA)

        
    # cv2.imshow('image', image)  
    # cv2.waitKey(0)

    # 转换为OpenCV角度系统（0度为右向，顺时针方向）
    # OpenCV的ellipse角度是顺时针，需要转换
    start_angle_cv = -end_angle
    end_angle_cv = -start_angle



    # 绘制扇形掩码
    # 定义椭圆的长轴和短轴，由于是绘制圆形，长轴和短轴长度均为外半径
    axes = (outer_radius, outer_radius)

    # 绘制外圆弧
    cv2.ellipse(
        mask,
        (center_x, center_y),
        axes,
        angle=0,
        startAngle=start_angle_cv,
        endAngle=end_angle_cv,
        color=255,
        thickness=-1,  # -1表示填充
        lineType=cv2.LINE_AA  # 外圆弧同样应用
    )

    # 如果有内半径，绘制内圆并减去（形成环形扇形）
    if inner_radius > 0:
        inner_axes = (inner_radius, inner_radius)
        cv2.ellipse(
            mask,
            (center_x, center_y),
            inner_axes,
            angle=0,
            startAngle=start_angle_cv-10,
            endAngle=end_angle_cv+10,
            color=0,
            thickness=-1,
            lineType=cv2.LINE_AA  # 外圆弧同样应用抗锯齿
        )

    # 应用掩码
    sector_img = cv2.bitwise_and(image, image, mask=mask)
       # --- 修改开始 ---
    # 将背景从黑色改为白色
    # 在掩码值为0（黑色）的区域，将sector_img中对应的像素设置为255（白色）
    sector_img[mask == 0] = 255
    
    # 计算扇形边界框并裁剪（减少空白区域）
    # 找到掩码中非零区域的坐标
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image

    x, y, w_sector, h_sector = cv2.boundingRect(coords)
    # 直接使用边界框坐标裁剪，确保包含从内半径到外半径的整个区域
    crop_img = sector_img[y-20:y+h_sector+20, x-20:x+w_sector+20]


     # 添加图像放大功能
    if scale_factor != 1.0 and crop_img.size > 0:
        # 计算新尺寸
        new_width = int(crop_img.shape[1] * scale_factor)
        new_height = int(crop_img.shape[0] * scale_factor)
        # 使用双三次插值放大
        crop_img = cv2.resize(
            crop_img,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC

        )
    # cv2.imshow('crop_image', crop_img)
    # cv2.waitKey(0)

    return crop_img


# # 加载数据集（使用实际JSON根目录）
# datasets = load_evaluation_datasets(json_root=os.path.join('./','data', 'out', 'circle'))


def generate_prompt(item_name: str, prompt_type: str, dataset: dict,tick = 0) -> str:
    # 获取图表类型（假设dataset中已包含chart_type字段）
    chart_type = dataset.get('chart_type', '')
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

⚠️ Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
        if chart_type == 'rose':
            return f'''
图表包含**虚拟参考线**：
您正在分析一张玫瑰图。它通过扇形的**最远端半径**来展示数据，每个扇形代表一个类别，其长度表示数据值的大小。
以下为图表的详细信息：
    - 存在以下径向网格线（同心圆），对应的刻度值为{dataset.get('r_ticks', [])},标注在对应网格虚线上
    - 存在以下角度网格线，将圆分成多个扇形区域，{dataset.get('theta_ticks', [])}分别对应每个扇形区域，扇形区域的分界为{dataset.get('theta_angles', [])}（单位为度）

您的任务是估计标记为"{item_name}"的对应扇形的的值：
以下为提示：
    1. **在玫瑰图上找到"{item_name}"对应的扇形区域，即确定其角度范围。**,非常重要，以x轴正方向为0度，可以理解为{dataset.get('theta_ticks')[0]}对应范围为0-{dataset.get('theta_angles')[1]}度，{dataset.get('theta_ticks')[1]}对应范围为{dataset.get('theta_angles')[1]}-{dataset.get('theta_angles')[2]}度，以此类推
    例子：该图的{dataset.get('theta_ticks')[0]}对应范围为0-{dataset.get('theta_angles')[1]}度，处在图最右端
    2. 确定其径向位置，找到其处于哪两个网格线之间，网格线包含以下刻度{dataset.get('r_ticks', [])}，必须准确的识别其位于哪两个网格线之间
    例子：该图的"{dataset.get('theta_ticks')[0]}"的值就为{dataset['data_points'][dataset.get('theta_ticks')[0]]}
    3. 根据其扇形和相对于两个网格线的位置，插值计算其数据值。

**记住，一定要插值，利用好网格线的刻度值**
在预测之前，再次回顾以下我给你的提示
一定要给我一个值，不能给我多个值，也不能给我没有值的情况

⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "no_grid":
        if chart_type == 'radar':
            return f'''
            
Your task is to estimate the value of the data point labeled "{item_name}":

⚠️ Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
        elif chart_type == 'rose':
            return f'''
您的任务是估计标记为"{item_name}"对应的值：

⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
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

**重要提示**：图表中已添加红色圆环，表示上一轮对"{item_name}"的预测值约为{tick}。
请比较红色圆环与真实数据点的位置差距，重新优化您的预测：
1. 确定红色圆环与真实数据点之间的位置关系（偏内、偏外）
2. 根据这种关系，调整您的预测值
3. 确保新的预测值与真实点的位置对齐 以实现尽可能准确的预测

⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
        elif chart_type == 'rose':
            return f'''
图表包含**虚拟参考线**：
您正在分析一张玫瑰图。它通过扇形的**最远端半径**来展示数据，每个扇形代表一个类别，其长度表示数据值的大小。
以下为图表的详细信息：
    - 存在以下径向网格线（同心圆），对应的刻度值为{dataset.get('r_ticks', [])},标注在对应网格虚线上
    - 存在以下角度网格线，将圆分成多个扇形区域，{dataset.get('theta_ticks', [])}分别对应每个扇形区域，扇形区域的分界为{dataset.get('theta_angles', [])}（单位为度）

您的任务是估计标记为"{item_name}"的对应扇形的值：

**重要提示**：图表中已添加红色圆环，表示上一轮对"{item_name}"的预测值约为{tick}。
请比较红色圆环与真实数据点的位置差距，重新优化您的预测：
1. 确定红色圆环与真实数据点之间的位置关系（偏内、偏外）
2. 根据这种关系，调整您的预测值
3. 确保新的预测值与真实点的位置对齐 以实现尽可能准确的预测
一定要给我一个值，不能给我多个值，也不能给我没有值的情况

**记住，一定要利用网格线进行精确插值**

⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "amplifier":
        if chart_type == 'radar':
            return f'''
您正在分析雷达图的一部分。该图在二维平面上使用从同一点出发的坐标轴展示多变量数据。

- 共有{len(dataset.get('series_color', {}))}个实体：{', '.join(dataset.get('series_color', {}).keys())}，分别对应颜色{', '.join(dataset.get('series_color', {}).values())}
现在的局部放大图为{item_name.split(',')[1].strip()}轴对应的局部放大
您的任务是估计标记为"{item_name}"对应的值，即{item_name.split(',')[0].strip()}对应实体颜色的数值。
请先找到{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
然后找到该颜色对应的点，并插值出数值

⚠️ 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()

    else:
        raise ValueError("Unknown prompt_type")
def call_llm_response(prompt: str, image_path: str, item_name: str, dataset: dict) -> Tuple[float | None, float | None]:
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        # vveai API参数
    max_retries = 10
    retry_delay = 0.5  # 秒
    retry_count = 0

    api_key = "sk-BH7u759dHuV910Xo9d3bC5A45d524b7c9c2b95528d09D92d"
    url = "https://api.vveai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": llm_model,
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

    while retry_count < max_retries:
        try:
            response = requests.post(url=url, headers=headers, json=payload, timeout=10)

            response.raise_for_status()  # 检查HTTP错误状态码

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            coords_json = extract_json_response(content)
            #print(coords_json)
            if coords_json and "datapoints" in coords_json:
                for item in coords_json["datapoints"]:
                    if item_name in item:
                        coords = item[item_name]
                        if validate_coordinates(coords):
                            return tuple(coords)
            
            # 如果未找到数据但请求成功，不重试直接返回
            return (None, None)
            
        except requests.exceptions.RequestException as e:
            retry_count += 1
            print(f"❌ 请求异常: {e}, 正在进行第 {retry_count}/{max_retries} 次重试...")
            if retry_count < max_retries:
                time.sleep(retry_delay)
        except Exception as e:
            retry_count += 1
            print(f"❌ 解析异常: {e}, 正在进行第 {retry_count}/{max_retries} 次重试...")
            if retry_count < max_retries:
                time.sleep(retry_delay)

    print(f"⚠️ 已达到最大重试次数 ({max_retries}次)")
    return (None, None)

if __name__ == '__main__':
    #将数据集保存到JSON文件
    #生成数据集的prompt
    with open(f'evaluation_datasets_with_axis_labels.json', 'r', encoding='utf-8') as f:

        datasets = json.load(f)
   
    feedback_image_dir = './data/feedback'
    amplifier_image_dir = './data/amplifier'
    if not os.path.exists(feedback_image_dir):
        os.makedirs(feedback_image_dir)
    if not os.path.exists(amplifier_image_dir):
        os.makedirs(amplifier_image_dir)


    # 创建一个字典来存储结果，以图片路径为键
    results_by_image = {}
    for grid_type in ['with_grid','no_grid']:
        for dataset in datasets[:28]:
            if dataset['chart_type'] != 'radar' or not dataset['data']:
                continue
            
            image_path = dataset["image_paths"][grid_type].replace('\\', '/')
            chart_id = dataset['chart_id']
            print(chart_id)
            if chart_id not in results_by_image:
                results_by_image[chart_id] = {
                    'chart_type': dataset['chart_type'],
                    'data': {}
                }
             # 遍历顶层键（如WDULR, ZTJUP等）
            for top_key, nested_dict in dataset['data'].items():
                # 遍历嵌套字典中的键值对（如A, B, C, D）
                for sub_key, value in nested_dict.items():
                    # 构建格式为"顶层键,子键"的item_name
                    item_name = f"{top_key},{sub_key}"
                    prompt = generate_prompt(item_name, grid_type, dataset)
                    coords = call_llm_response(prompt, image_path, item_name, dataset)
                    if item_name not in results_by_image[chart_id]['data']:
                        results_by_image[chart_id]['data'][item_name] = {
                            'origin': value
                        }
                    #feedback
                    
                    # if(grid_type == 'with_grid'):
                    #     feedback_counts = 2
                    #     feedback_tick = []
                    #     if(coords[0]!=None):
                    #         feedback_tick.append(coords[0])
                    #     else:
                    #         feedback_tick.append(origin_value)
                    #     feedback_times = 1
                    #     while(feedback_counts):
                    #         temp_image = cv2.imread(dataset["image_paths"]['with_grid'].replace('\\', '/'))
                    #         feedback_image = temp_image.copy()
                    #         feedback_image_path = os.path.join(feedback_image_dir, f'{chart_id}_{grid_type}_{item_name}_{feedback_times}.png')
                    #         center_x = dataset["pred_coords"][0]
                    #         center_y = dataset["pred_coords"][1]
                    #         a = dataset["argument"]["a"]
                    #         b = dataset["argument"]["b"]
                    #         pre_r = int(a*feedback_tick[-1]+b)
                    #         cv2.circle(feedback_image, (center_x, center_y), pre_r, (0, 0, 255), 1 ,lineType=cv2.LINE_AA)
                    #         cv2.imwrite(feedback_image_path, feedback_image)
                    #         # 在循环结束后删除生成的反馈图片
                    #         feedback_prompt = generate_prompt(item_name, 'feedback', dataset,feedback_tick[-1])
                    #         feedback_coords = call_llm_response(feedback_prompt, feedback_image_path, item_name,dataset)
                    #         if(feedback_coords[0]!=None):
                    #             feedback_tick.append(feedback_coords[0])
                    #         else:
                    #             feedback_tick.append(coords[0])
                    #         print(feedback_tick)

                    #         os.remove(feedback_image_path)
                    #         feedback_times += 1
                    #         feedback_counts -= 1
                    #     if(feedback_tick):    
                    #         results_by_image[chart_id]['data'][item_name]['feedback'] = feedback_tick
                            
                    #amplifier 
                    #流程 1.获取item对应轴以及位置  2.根据对应轴位置裁剪图片 3.放大裁剪后的图片 4.call_llm获取返回值
                    if(grid_type == 'with_grid'):

                        axis_labels = dataset.get('axis_labels')
                        #print(axis_labels)
                        label_to_angle = {v: int(k) for k, v in axis_labels.items()}  # 建立标签到角度的反向映射
                        # print(label_to_angle)
                        # 从item名称提取轴标签（格式："UOHP,E" -> 提取"E"）
                        #print(item_name)
                        item_label = item_name.split(',')[1].strip()
                        # print(item_label)
                        target_angle = label_to_angle.get(item_label,0)
                        #print(target_angle)
                        amplifier_path = dataset["image_paths"]['no_grid'].replace('\\', '/')
                        center_x, center_y = dataset["pred_coords"]
                        arg_a = dataset["argument"]["a"]
                        arg_b = dataset["argument"]["b"]
                        radius = int(arg_a * coords[0] + arg_b)
                        r_ticks = dataset["r_ticks"]
                        # print(r_ticks)
                        if(coords[0]!=None):
                            inner_radius = int(arg_a*coords[0]+arg_b - 0.5*r_ticks[-1])
                            outer_radius = int(arg_a*coords[0]+arg_b + 0.5*r_ticks[-1])
                            if(outer_radius>dataset['r_pixels'][-1]):
                                outer_radius = dataset['r_pixels'][-1]+30
                            if(inner_radius<0):
                                inner_radius = 0
                        else:
                            inner_radius = 0
                            outer_radius = radius
                        scale_factor = 2
                        amplifier_image_path = os.path.join(amplifier_image_dir, f'{chart_id}_{grid_type}_{item_name}.png')
                        amplifier_image = crop_axis_label_region(amplifier_path, center_x, center_y, target_angle, outer_radius, 30, inner_radius, 30, scale_factor, r_ticks, arg_a, arg_b)
                        if amplifier_image.size > 0:
                            cv2.imwrite(amplifier_image_path, amplifier_image)
                        else:
                            print(f"警告: 无法保存图像 {amplifier_image_path}，因为裁剪区域为空")
                        amplifier_prompt = generate_prompt(item_name, 'amplifier', dataset)
                        amplifier_coords = call_llm_response(amplifier_prompt, amplifier_image_path, item_name, dataset)
                        #print(amplifier_prompt)
                        if amplifier_coords is not None:
                            results_by_image[chart_id]['data'][item_name]['amplifier'] = amplifier_coords[0]
                    
                    # 只存储有效坐标值
                    if coords is not None:
                        results_by_image[chart_id]['data'][item_name][grid_type] = coords
                    results_by_image[chart_id]['data'][item_name]['origin'] = value
                    if(grid_type == 'with_grid'):
                        print(item_name,"origin:",value,grid_type,":",coords,"amplifier:",amplifier_coords)
                    else:
                        print(item_name,"origin:",value,grid_type,":",coords)

                    #print(results_by_image)
    
    # 保存结果到JSON文件
    with open(f'coordinates_by_image_{datasets[0]['chart_type']}_{llm_model}.json', 'w', encoding='utf-8') as f:
        json.dump(results_by_image, f, ensure_ascii=False, indent=4)
