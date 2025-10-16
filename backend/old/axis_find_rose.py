import numpy as np
from sklearn.cluster import DBSCAN
import math
import json
import cv2
import re
import requests
import base64

api_key = "sk-1fZigErRE5Mv2Y2d910c8b8f86354dF3AeD8B8F2Bb385dEb"
url = "https://api.vveai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


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


#裁剪图像
def crop_axis_label_region(image_path, center_x, center_y, angle_deg, radius, label_offset=100, label_width=150, label_height=150):
    """
    根据角度裁剪坐标轴名称区域
    :param image: 原始图像
    :param center_x, center_y: 圆心坐标
    :param angle_deg: 角度(度)
    :param radius: 极坐标图半径
    :param label_offset: 名称区域相对圆心的偏移量
    :param label_width: 裁剪区域宽度
    :param label_height: 裁剪区域高度
    :return: 裁剪后的区域图像
    """
    # 角度转弧度
    image = cv2.imread(image_path)

    angle_rad = math.radians(angle_deg)
    
    # 计算名称区域中心坐标（在角度方向上，距离圆心radius+offset处）
    label_center_x = int(center_x + (radius + label_offset) * math.cos(angle_rad))
    label_center_y = int(center_y - (radius + label_offset) * math.sin(angle_rad))  # 图像y轴向下，故减号
    
    # 计算裁剪区域左上角和右下角坐标
    x1 = max(0, label_center_x - label_width // 2)
    y1 = max(0, label_center_y - label_height // 2)
    x2 = min(image.shape[1], label_center_x + label_width // 2)
    y2 = min(image.shape[0], label_center_y + label_height // 2)
    
    crop_img = image[y1:y2, x1:x2]
    # cv2.imshow('crop_img',crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 裁剪区域并返回
    return crop_img


def call_llm_letter(crop_img):
    """
    调用LLM
    :return: 模型回复
    """
    image_area = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        
        # 确保图像数据类型正确
    if image_area.dtype != np.uint8:
        image_area = image_area.astype(np.uint8)
        print("已转换图像数据类型为uint8")
        
        # 关键修复：使用JPEG编码而非原始像素数据
    success, encoded_image = cv2.imencode('.jpg', image_area)
    if not success:
        print("图像编码失败")
        return None
        
        # 转换为C连续数组并编码为base64
    image_data = np.ascontiguousarray(encoded_image)
    base64_image = base64.b64encode(image_data).decode('utf-8')
    prompt = f"""
    请你分析该图片中的字母，并返回，该字母为黑色，且处在大约中心位置
    ```json
    {{
        "letter": <字母>
    }}
    如果无法识别某个值，请使用 `null`
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


def call_llm_nums(image_path:str):
    """
    调用LLM
    :return: 模型回复
    """
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        prompt = f"""
    请你分析该图片中的含有的轴的名称的个数，并返回，轴的延申最外出有字母组合的轴名称，轴的名称为英文
    轴的名称个数和轴的个数应该保持相同，轴的名称不重复
    记住，只有延长线上有名称的才算一个轴，如果没有名称则不算，比如有些自带的轴只是为了标注刻度，这样的就不算轴

    请判断轴是在色块的中心位置，还是在色块的边缘位置
    ```json
    {{
        "axis_name": <轴的名称>,
        "nums": <轴的名称个数>,
        "position": <轴的位置>,
        "reason": <原因>
    }}
    如果无法识别某个值，请使用 `null`
    """
    payload = {
        "model": "gpt-4.1",
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


def get_start_angle(image_path:str,center_x:int,center_y:int,radius:int):

    """
    获取起始角度
    :param axes_angles: 轴角度列表
    :return: 起始角度
    """
    pred_angle = [0,90]
    start_angle= []
    for angle in pred_angle:
        image_area = crop_axis_label_region(image_path, center_x, center_y, angle, radius)
        data = call_llm_letter(image_area)
        if data["letter"]!='None':
            start_angle.append(angle)
    print("have_lable:",start_angle)
    if start_angle:
        return start_angle[0]
    else:
        return None


def find_rose_axes(image_path:str,center,start_angel:int,max_radius):
    img = cv2.imread(image_path)
    axes_angles = []
    axes = call_llm_nums(image_path)
    print(axes)
    axes_nums = len(axes['axis_name'])
    for i in range(axes_nums):
        best_interval = int(360 / axes_nums)
        angle = round(start_angel + i * best_interval) % 360
        axes_angles.append(angle)
    # 4. 可视化结果（可选）
    output_img = img.copy()
    for angle in axes_angles:
        current_angle_rad = math.radians(angle)
        
        # 沿着找到的角度画一条线
        end_x = int(center[0] + max_radius * math.cos(current_angle_rad))
        end_y = int(center[1] + max_radius * math.sin(current_angle_rad))
        
        # 在图像上绘制红线，用于可视化
        cv2.line(output_img, center, (end_x, end_y), (0, 0, 255), 1)
        
        # 在终点处绘制一个圆点
        cv2.circle(output_img, (end_x, end_y), 2, (0, 255, 0), -1)

    print(f"找到的轴线角度: {axes_angles}")
    # cv2.imshow("Radar Axes Detection", output_img)
    # cv2.waitKey(0)

    return axes_angles

if __name__ == "__main__":
    dataset_path = 'd:/home work/Agent.paper/Agent/evaluation_datasets.json'
    dataset_path_with_axes = 'd:/home work/Agent.paper/Agent/evaluation_datasets_with_axes.json'
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {} 
        print("错误: 未找到 evluation_datasets.json 文件")
    
    # 确定是否处理特定数量的图表，None表示处理所有
    max_charts = 20                                              # 可以设置为具体数字，如10来处理前10个图表
    
    if data and isinstance(data, list):
        # 确定要处理的图表数量
        total_charts = len(data)
        charts_to_process = total_charts
        if max_charts is not None and max_charts > 0:
            charts_to_process = min(max_charts, total_charts)
            print(f"将处理前 {charts_to_process}/{total_charts} 张图表")
        else:
            print(f"将处理所有 {total_charts} 张图表")
        
        # 遍历图表
        for i in range(charts_to_process):
            try:
                now_chart = data[i]
                print(f"\n处理图表 {i+1}/{charts_to_process}: {now_chart.get('chart_id', '未命名图表')}")
                
                # 获取图表数据
                image_path = now_chart['image_paths']['no_grid']
                coords = now_chart['pred_coords']
                r = now_chart['argument']['a'] * now_chart['r_ticks'][-1] + now_chart['argument']['b'] - 5
                
                # 获取起始角度并识别轴线
                start_angle = get_start_angle(image_path, coords[0], coords[1], r)
                if start_angle is None:
                    print(f"跳过图表 {now_chart.get('chart_id', i)}: 无法确定起始角度")
                    continue
                
                print(f"起始角度: {start_angle}")
                found_axes = find_rose_axes(image_path, coords, start_angle, r)
                
                # 识别轴标签
                axis_labels = {}
                for axis in found_axes:
                    try:
                        crop_img = crop_axis_label_region(image_path, coords[0], coords[1], axis, r)
                        axis_data = call_llm_letter(crop_img)
                        
                        # 提取识别结果
                        if isinstance(axis_data, dict) and 'letter' in axis_data and axis_data['letter'] is not None:
                            letter = axis_data['letter']
                            axis_labels[axis] = letter
                            print(f"轴角度: {axis}, 识别结果: {letter}")
                        else:
                            print(f"轴角度: {axis}, 识别结果无效: {axis_data}")
                    except Exception as e:
                        print(f"处理轴角度 {axis} 时发生错误: {str(e)}")
                        continue
                
                # 验证轴标签数量与数据点数量是否匹配
                num_vars = len(now_chart['data_points'])
                if len(axis_labels) == num_vars:
                    # 更新图表数据
                    print("轴角度与识别结果对应关系:", axis_labels)
                    now_chart['axis_labels'] = axis_labels
                    data[i] = now_chart
                    print(f"成功保存图表 {now_chart.get('chart_id', i)} 的数据")
                else:
                    print(f"跳过图表 {now_chart.get('chart_id', i)}: 轴标签数量不匹配 ({len(axis_labels)}/{num_vars})")
            except Exception as e:
                print(f"处理图表索引 {i} 时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # 保存更新后的数据到JSON文件
        try:
            with open(dataset_path_with_axes, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n数据已成功更新并保存到{dataset_path_with_axes}")
        except Exception as e:
            print(f"保存数据失败: {str(e)}")
    else:
        print("数据格式不正确或为空")

    # max_charts = min(10, len(data))  # 确保不超出data的实际长度
    # for i in range(max_charts):
    #     try:
    #         now_chart = data[i]
    #         print(f"\n处理图表索引 {i}: {now_chart.get('chart_id', '未命名图表')}")
            
    #         # 检查image_paths是否存在且包含'no_grid'
    #         if 'image_paths' not in now_chart or 'no_grid' not in now_chart['image_paths']:
    #             print(f"警告: 图表索引 {i} 缺少图像路径信息")
    #             continue
            
    #         image_path = now_chart["image_paths"]['no_grid']
    #         axis_num = call_llm_nums(image_path)
            
    #         # 检查axis_num是否为有效字典
    #         if not isinstance(axis_num, dict) or 'nums' not in axis_num:
    #             print(f"警告: 图表索引 {i} 的LLM响应格式不正确")
    #             continue
            
    #         # 获取实际轴个数
    #         actual_axis_count = len(now_chart.get('data_points', {}))
    #         llm_axis_count = axis_num['nums']
    #         reason = axis_num['reason']
            
    #         # 比较轴的个数
    #         if llm_axis_count != actual_axis_count:
    #             print(f"轴的个数不一致")
    #             print(f'LLM识别的轴的个数为：{llm_axis_count}')
    #             print(f'实际轴的个数为：{actual_axis_count}')
    #             print(f'LLM的识别原因：{reason}')
    #             print(f'LLM的轴的位置：{axis_num["position"]}')
    #         else:
    #             print(f"轴的个数一致")
    #             print(f'LLM识别的轴的个数为：{llm_axis_count}')
    #             print(f'实际轴的个数为：{actual_axis_count}')
    #             print(f'LLM的识别原因：{reason}')
    #             print(f'LLM的轴的位置：{axis_num["position"]}')

    #         # 输出轴名称（如果有）
    #         if 'axis_name' in axis_num and axis_num['axis_name']:
    #             print(f'识别的轴名称：{axis_num['axis_name']}')
    #     except Exception as e:
    #         print(f"处理图表索引 {i} 时发生错误: {str(e)}")
