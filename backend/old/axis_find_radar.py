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
def find_radar_axes(image_path: str,center_coords:dict):

    """
    通过计算机视觉流程，自动检测雷达图的圆心和四个主轴的角度。

    Args:
        image_path (str): 输入雷达图图像的路径。

    Returns:
        tuple: 包含圆心坐标和轴角度字典的元组 (center, final_axes)。
               如果无法找到足够的轴，则返回 (None, None)。
    """
    # 0. 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像 {image_path}")
        return None, None
    
    output_image = image.copy() # 创建一个副本用于绘制结果

    # 1. 图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊减少噪声，提高圆和直线检测的准确性
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # center_x = center_coords['x']
    # center_y = center_coords['y']
    center_x = center_coords[0]
    center_y = center_coords[1]


    
    # 在输出图上标记圆心
    cv2.circle(output_image, (center_x, center_y), 3, (0, 0, 255), -1) # 红色圆心

    # 3. 直线检测 (使用Canny边缘 + 概率霍夫变换)
    # 使用Canny边缘检测算法检测图像边缘
    # blurred: 输入的经过高斯模糊处理后的灰度图像
    # 50: 滞后阈值处理中的低阈值
    # 150: 滞后阈值处理中的高阈值
    # apertureSize=3: Sobel算子的孔径大小，用于计算图像梯度
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    
    # 调整HoughLinesP参数以适应图像
    # minLineLength: 轴线很长，所以这个值可以设得比较大
    lines = cv2.HoughLinesP(
        edges,  # 输入的边缘图像，由Canny边缘检测算法得到
        rho=1,  # 累加器的距离分辨率（像素），表示直线搜索时的距离步长
        theta=np.pi / 180,  # 累加器的角度分辨率（弧度），表示直线搜索时的角度步长
        threshold=50,  # 累加器阈值，只有投票数超过该阈值的直线才会被检测出来
        minLineLength=int(image.shape[0] * 0.2),  # 最小直线长度，至少是图像高度的20%，短于该长度的直线会被忽略
        maxLineGap=50  # 同一条直线上连接点之间的最大允许间隔，超过该间隔的点不会被视为同一条直线
    )

    if lines is None:
        print("错误: 未能检测到足够的直线。")
        return None, None

    # 4. 筛选和提纯轴线
    axis_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 筛选条件1: 直线必须靠近圆心
        # 计算点到直线的距离公式: |Ax+By+C| / sqrt(A^2+B^2)
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([center_x, center_y])
        vec1 = np.append(p2 - p1, 0)  # 线段方向向量
        vec2 = np.append(p1 - p3, 0)  # 点到线段起点的向量
        cross_product = np.cross(vec1, vec2)
        dist_to_center = np.linalg.norm(cross_product) / np.linalg.norm(vec1)
        # 容忍距离设为5个像素
        if dist_to_center < 2:
            # 使用 np.arctan2 计算直线的角度（弧度），p1[1] - p2[1] 为 y 坐标差，p1[0] - p2[0] 为 x 坐标差
            # np.arctan2 返回值范围为 [-π, π]，表示从 x 轴正方向逆时针旋转到直线的角度
            angle_rad = np.arctan2(p3[1] - p1[1], p3[0] - p1[0])
            # 将角度归一化到 [0, 360) 范围
            angle_deg = (np.rad2deg(angle_rad) + 360) % 360
            axis_candidates.append([angle_deg]) # 角度作为特征

            
            # 在输出图上绘制通过筛选的候选线 (蓝色)
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    print(axis_candidates)

    if not axis_candidates:
        print("错误: 没有找到任何穿过圆心的候选轴线。")
        return None, None
        
     # 5. 使用DBSCAN对角度进行聚类，找出所有可能的轴方向
    # eps: 角度差10度以内算作同一类
    # min_samples: 至少有1条线才能构成一个簇
    clustering = DBSCAN(eps=5, min_samples=1).fit(axis_candidates)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    # 移除噪声点标签 (-1)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    # 计算每个有效聚类的平均角度
    avg_angles = []
    for label in unique_labels:
        cluster_angles = np.array(axis_candidates)[labels == label].flatten()
        # 使用矢量法计算平均角度，以处理360°环绕问题
        radians = np.deg2rad(cluster_angles)
        avg_sin = np.mean(np.sin(radians))
        avg_cos = np.mean(np.cos(radians))
        avg_rad = np.arctan2(avg_sin, avg_cos)
        avg_deg = (np.rad2deg(avg_rad) + 360) % 360
        avg_angles.append(avg_deg)
    
    # 按角度大小排序，便于后续处理
    avg_angles.sort()
    
    num_detected_axes = len(avg_angles)
    if num_detected_axes < 2:
        print(f"错误: 检测到的轴数量 ({num_detected_axes}) 过少，无法判断其几何分布。")
        return center_coords, {}


    print(f"信息: 初步检测到 {num_detected_axes} 个可能的轴。正在验证其几何分布...")

    # 6. 分析轴的几何分布（是否等角度）
    angle_diffs = []
    for i in range(num_detected_axes):
        # 计算相邻两个角度的差值
        diff = avg_angles[i] - avg_angles[i-1]
        angle_diffs.append(diff)
    
    # 最后一个角度和第一个角度的差值需要特殊处理（环绕）
    angle_diffs[0] = (angle_diffs[0] + 360) % 360

    # 检查角度差是否一致
    # 计算所有角度差的标准差，如果标准差很小，说明它们几乎相等
    std_dev = np.std(angle_diffs)
    
    # 如果标准差大于一个阈值（例如5度），则认为轴不是等角度分布
    if std_dev > 5.0:
        print(f"警告: 检测到的轴似乎不是等角度分布。角度差的标准差为 {std_dev:.2f}°。")
        print("将按原样输出所有检测到的轴。")
        # 在这种情况下，我们仍将所有找到的轴视为有效，但不保证其规律性
        final_axes = {f"Axis-{i+1}": angle for i, angle in enumerate(avg_angles)}
    else:
        # 如果分布均匀，则确认这些都是有效的轴
        print("成功: 检测到的轴符合等角度分布。")
        angular_separation = np.mean(angle_diffs)
        print(f"推断的轴数量: {num_detected_axes}, 角度间隔约为: {angular_separation:.2f}°")
        # 使用 "Axis-N" 作为标签
        final_axes = {f"Axis-{i+1}": angle for i, angle in enumerate(avg_angles)}


    # 7. 在图像上绘制最终识别的轴和标签
    # (此部分取代了原先的绘制代码)
    for name, angle in final_axes.items():
        angle_rad = np.deg2rad(angle)
        # 从圆心画一条长线来表示轴
        end_x = int(center_x + image.shape[1] * 0.5 * math.cos(angle_rad))
        # y轴在图像坐标系中是反的
        end_y = int(center_y - image.shape[1] * 0.5 * math.sin(angle_rad))
        #cv2.line(output_image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 1) # 绿色粗线
        # 在轴末端添加标签
        cv2.putText(output_image, name, (end_x - 40, end_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imshow('Radar Axes Detection', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('radar_axes_result.png', output_image)

    return center_coords, final_axes

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
def crop_axis_label_region(image_path, center_x, center_y, angle_deg, radius, label_offset=30, label_width=50, label_height=50):
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


#识别标签
def call_llm(crop_img):
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


#获取起始角度
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
        data = call_llm(image_area)
        if data['letter']:
            start_angle.append(angle)
    print("have_lable:",start_angle)
    if start_angle:
        return start_angle[0]
    else:
        return None

#识别轴线
def find_radar_axes_by_pixel(image_path, center,max_radius,start_angle):


    """
    通过遍历圆心周围的像素，寻找雷达图的轴线。

    Args:
        image_path (str): 雷达图文件的路径。
        center (tuple): 圆心的坐标 (x, y)。

    Returns:
        list: 找到的轴线的角度列表。
    """
    # 1. 读取图像并转换为灰度图像，方便处理
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return []

    # 转换为灰度图，如果需要也可以在RGB空间进行处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, _ = img.shape
    
    # 2. 确定背景色和颜色差异阈值
    # 假设图像背景是白色。我们可以通过检查圆心周围的像素来估计背景色。
    # 也可以手动指定背景色，这里我们假设白色 (255, 255, 255)。
    background_color = np.array([255, 255, 255], dtype=np.uint8)


    
    # 我们使用RGB颜色差异，所以需要用彩色图像
    bgr_background = background_color
    
    # 设定一个颜色差异阈值。欧几里得距离超过这个值就认为不是背景色。
    # 这个值可能需要根据具体图像进行调整。
    color_diff_threshold = 5

    # 3. 极坐标遍历寻找轴线
    axes_angles = []
    
    # 遍历360个角度
    num_angles = 360
    for angle in range(num_angles):
        is_axis = False
        current_angle_rad = math.radians(angle)
        
        # 从圆心向外扫描，最大半径取图像宽度或高度的一半
        
        for r in range(int(max_radius)-15, int(max_radius)-3):

            x = int(center[0] + r * math.cos(current_angle_rad))
            y = int(center[1] + r * math.sin(current_angle_rad))
            
            # 检查坐标是否在图像范围内
            if 0 <= x < width and 0 <= y < height:
                current_pixel = img[y, x]
                
                # 计算与背景色的欧几里得距离
                diff = np.sqrt(np.sum((current_pixel - bgr_background) ** 2))
                
                # 如果颜色差异超过阈值，认为找到轴线
                if diff > color_diff_threshold:
                    # 为了避免将同一条轴线识别多次，我们只记录第一次找到的角度
                    # 并且确保新角度与已找到的角度有足够大的间隔（例如，5度）
                    if not axes_angles or abs(angle - axes_angles[-1]) > 10:
                        axes_angles.append(angle)
                        is_axis = True
                        
            # 如果找到轴，停止当前角度的扫描，进入下一个角度
            if is_axis:
                break
    print('first axes_angles: ',axes_angles)
    # 5. 优化轴角度间隔 (新增代码)
    if len(axes_angles) >= 2:
        # 对角度排序并计算相邻间隔
        sorted_angles = sorted(axes_angles)
        intervals = []
        for i in range(len(sorted_angles)):
            if i < len(sorted_angles) - 1:
                diff = sorted_angles[i+1] - sorted_angles[i]
            else:
                # 处理首尾角度差（考虑360度循环）
                diff = (sorted_angles[0] + 360) - sorted_angles[i]
            intervals.append(diff)

        # 计算间隔中位数并寻找最优360因数
        median_interval = np.median(intervals)
        median_interval = int(median_interval)
        print('median_interval: ',median_interval)
        best_interval = None
        delta_remainders = {}  # 存储delta与对应的最小余数

        # 在中位数±5度范围内搜索所有可能的delta
        for delta in range(-5, 6):
            candidate = median_interval + delta
            if candidate < 0:  # 排除非正值
                continue
            # 计算360对candidate的最小余数（考虑商和余数的两种情况）
            remainder = 360 % candidate
            min_remainder = min(remainder, candidate - remainder)
            delta_remainders[delta] = min_remainder
            # print('delta: ',delta,'candidate: ',candidate,'remainder: ',remainder,'min_remainder: ',min_remainder)

        # 找到最小余数的delta，优先选择余数为0的情况
        if delta_remainders:
            # 按余数从小到大排序，余数相同则按delta绝对值（离中位数距离）排序
            sorted_items = sorted(delta_remainders.items(), key=lambda x: (x[1], abs(x[0])))
            
            # 1. 优先寻找余数为0的精确解
            for delta, rem in sorted_items:
                if rem == 0:
                    best_interval = median_interval + delta
                    break
            
            # 2. 如果没有精确解，选择余数最小且离中位数最近的近似解
            if best_interval is None:
                best_delta, min_rem = sorted_items[0]
                best_interval = median_interval + best_delta

        # 如果找到有效间隔，补充可能缺失的轴角度
        if best_interval:
            print('best_interval: ',best_interval)            # 清除原有角度列表，按最优间隔重新生成
            axes_angles = []
            num_axes = int(round(360 / best_interval))
            for i in range(num_axes):
                # 计算角度并确保在0-360度范围内
                angle = round(start_angle + i * best_interval) % 360
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
    # cv2.destroyAllWindows()
    
    return axes_angles

if __name__ == '__main__':
    # 替换为你的图片路径
    dataset_path = 'd:/home work/Agent.paper/Agent/evaluation_datasets.json'
    # 设置要处理的最大图表数量，None 表示处理所有图表
    max_charts = 30  # 可以修改为具体数字，如 5 表示只处理前5张
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {} 
        print("错误: 未找到 evluation_datasets.json 文件")
    
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
            now_chart = data[i]
            print(f"处理图表 {i+1}/{charts_to_process}: {now_chart.get('chart_id')}")
            
            image_path = now_chart['image_paths']['no_grid']
            coords = now_chart['pred_coords']
            r = now_chart['argument']['a'] * now_chart['r_ticks'][-1] + now_chart['argument']['b'] -5
            # 流程为：1.获取起始角度 2.识别轴线 3.识别标签 4.返回结果

            start_angle = get_start_angle(image_path, coords[0], coords[1], r)
            found_axes = find_radar_axes_by_pixel(image_path, coords, r, start_angle)

            print(f"最终找到的轴线角度: {found_axes}")
            axis_labels = {}
            
            for axis in found_axes:
                crop_img = crop_axis_label_region(image_path, coords[0], coords[1], axis, r)
                axis_data = call_llm(crop_img)
                # 提取识别结果（假设API返回格式为{'letter': 'A'}）
                letter = axis_data.get('letter') if isinstance(axis_data, dict) else axis_data
                axis_labels[axis] = letter
                
                print(f"轴角度: {axis}, 识别结果: {letter}")
                # 检查now_chart中是否存在num_vars字段
            if 'num_vars' not in now_chart:
                print(f"跳过图表 {now_chart.get('chart_id')}: 缺少num_vars字段")
                continue
            
            # 验证axis_labels的长度是否等于num_vars
            num_vars = now_chart['num_vars']
            if len(axis_labels) == num_vars:
                # 更新图表数据
                print("轴角度与识别结果对应关系:", axis_labels)
                now_chart['axis_labels'] = axis_labels
                data[i] = now_chart
                print(f"成功保存图表 {now_chart.get('chart_id')} 的数据")
            else:
                print(f"跳过图表 {now_chart.get('chart_id')}: 轴标签数量不匹配 ({len(axis_labels)}/{num_vars})")

        
        # 保存更新后的数据到JSON文件
        try:
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"数据已成功更新并保存到{dataset_path}")
        except Exception as e:
            print(f"保存数据失败: {str(e)}")
    else:
        print("数据格式不正确或为空")

# # --- 执行脚本 ---
# if __name__ == '__main__':
#     # 替换为你的图片路径
#     dataset_path = 'd:/home work/Agent.paper/Agent/evaluation_datasets.json'
#     try:
#         with open(dataset_path, 'r', encoding='utf-8') as f:

#             data = json.load(f)
#     except FileNotFoundError:
#         data = {}
#         print("错误: 未找到 evluation_datasets.json 文件")
        
#     now_chart = data[0]
#     image_path = now_chart['image_paths']['no_grid']
#     coords = now_chart['pred_coords']
#     r = now_chart['argument']['a'] * now_chart['r_ticks'][-1] + now_chart['argument']['b'] -5
#     #流程为：1.获取起始角度 2.识别轴线 3.识别标签 4.返回结果

#     start_angle = get_start_angle(image_path,coords[0],coords[1],r)

#     found_axes = find_radar_axes_by_pixel(image_path, coords,r,start_angle)

#     print(f"最终找到的轴线角度: {found_axes}")
#     axis_labels = {}
    
#     for axis in found_axes:
#         crop_img = crop_axis_label_region(image_path, coords[0], coords[1], axis, r)
#         axis_data = call_llm(crop_img)
#         # 提取识别结果（假设API返回格式为{'letter': 'A'}）
#         # 根据实际API响应结构调整键名
#         letter = axis_data.get('letter') if isinstance(axis_data, dict) else axis_data
#         axis_labels[axis] = letter
        
#         print(f"轴角度: {axis}, 识别结果: {letter}")

#     # 返回完整的对应关系字典
#     print("轴角度与识别结果对应关系:", axis_labels)
#     now_chart['axis_labels'] = axis_labels
#     print(now_chart)
#     # 确保data对象被正确更新并保存
#     if data and isinstance(data, list):
#         # 查找当前图表在data中的索引并更新
#         chart_index = next((i for i, chart in enumerate(data) if chart.get('id') == now_chart.get('id')), 0)
#         print('chart_index: ',chart_index)
#         data[chart_index] = now_chart
#         # 保存更新后的数据到JSON文件
#     try:
#         with open(dataset_path, 'w', encoding='utf-8') as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
#         print(f"数据已成功更新并保存到{dataset_path}")
#     except Exception as e:
#         print(f"保存数据失败: {str(e)}")





    
    # center_coords, axes_angles = find_radar_axes_by_pixel(image_path, coords)
    # if center_coords and axes_angles:
    #     print("\n--- 检测结果 ---")
    #     print(f"检测到的圆心坐标: {center_coords}")
    #     print("检测到的轴角度:")
    #     for axis, angle in sorted(axes_angles.items()):
    #         print(f"  - 轴 {axis}: {angle:.2f}°")
    #     print("\n结果图像已保存为 'radar_axes_result.png'")


    # 1.份数识别
    # 2.起始角度的识别 （0 / 90） 边缘/中间
    # 3.