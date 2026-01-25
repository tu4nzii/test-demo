import cv2
import numpy as np
import os
from config import DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
import shutil

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

    min_dist_val = 100 # 默认值
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
        center_coordinates = (cx, cy)
        # 绘制该圆
        cv2.circle(image, center_coordinates, r, (0, 255, 0), 2)
        cv2.circle(image, center_coordinates, 2, (0, 0, 255), -1)
        cv2.putText(image, f"Center: ({cx}, {cy})", (cx + 15, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"方法二 (霍夫变换) 第一组圆的中心点: {center_coordinates}")
        return image, center_coordinates
    else:
        print("方法二 (霍夫变换) 失败: 未检测到任何圆形。")
        return image, None

if __name__ == "__main__":
    img_dir = "./data/basic_images/circle" # 假设您的图片在这里
    out_dir = "./data/out/circle"
    os.makedirs(out_dir, exist_ok=True)

    # 创建虚拟图片进行测试（如果它们不存在）
    # 在实际场景中，您将拥有实际的图片。
    # 为了演示，我们假设 circle8.jpg 和 circle6.jpg 位于 data/basic_images 中
    # 如果需要，您会将提供的图片保存到该目录中。
    
# 遍历所有以circle开头的图片
    for fname in os.listdir(img_dir):
        if fname.startswith("circle") and fname.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            img_path = os.path.join(img_dir, fname)
            print(f"处理图片: {img_path}")
            result_image_hough, center_hough = find_center_by_hough_transform(img_path)
            out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_center_hough.jpg")
            if result_image_hough is not None:
                cv2.imwrite(out_path, result_image_hough)
                print(f"霍夫变换结果已保存到 {out_path}")



#华东院 新华三
