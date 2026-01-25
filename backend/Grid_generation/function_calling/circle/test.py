import cv2
import numpy as np
import os
from config import DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
import shutil

out_dir = DEBUG_OUTPUT_DIRS['test']
if CLEAR_OUTPUT_BEFORE_RUN.get('test', False):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

def find_chart_center_advanced(image_path):
    """
    通用高级方法来寻找图表中心，适用于极坐标图、雷达图。
    该方法结合了更鲁棒的预处理、区域识别和基于径向/轴线汇聚的中心寻找。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return image, None

    original_image = image.copy() # 保留原始图像用于绘制

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 步骤1：鲁棒的二值化和噪声去除
    # 尝试多种阈值方法，这里使用自适应阈值，因为它对光照变化更鲁棒
    # BLOCK_SIZE 必须是奇数，C是一个从均值中减去的常数
    # 尝试对浅色线条进行二值化，所以需要反转图像
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 5) # 调整 block_size 和 C

    # 步骤2：形态学操作增强线条和连接虚线
    # 先进行小的腐蚀，去除噪声
    eroded = cv2.erode(binary, np.ones((2,2), np.uint8), iterations=1)
    # 再进行闭运算，连接虚线
    kernel_close = np.ones((5,5), np.uint8) # 增大核以连接更远的虚线
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close, iterations=2) # 增加迭代次数

    # 步骤3：填充小的空白区域（对于某些情况可能有用）
    # 寻找外部轮廓，然后填充内部
    contours_fill, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(closed)
    for cnt in contours_fill:
        # 填充面积较大的轮廓，避免填充小的噪声
        if cv2.contourArea(cnt) > 200: # 调整面积阈值
            cv2.drawContours(filled_image, [cnt], -1, 255, -1)

    # 步骤4：再次进行形态学操作，清理和确保连接
    kernel_final = np.ones((3,3), np.uint8)
    processed_image = cv2.morphologyEx(filled_image, cv2.MORPH_OPEN, kernel_final, iterations=1) # 开运算去除小点
    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel_final, iterations=1) # 再次闭运算

    # 步骤5：寻找最大轮廓作为图表主体
    contours_main, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_main:
        print("高级分析：未检测到任何主要轮廓。")
        return original_image, None

    main_chart_contour = max(contours_main, key=cv2.contourArea)

    # 步骤6：计算图表主体的边界框和质心
    x, y, w, h = cv2.boundingRect(main_chart_contour)
    
    # 限制分析区域在主要轮廓的边界框内，去除外部文字和图例干扰
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_chart_contour], -1, 255, -1)
    
    # 将处理后的图像与蒙版结合，只保留图表区域
    chart_only_processed = cv2.bitwise_and(processed_image, processed_image, mask=mask)

    # 计算 chart_only_processed 的质心
    M = cv2.moments(chart_only_processed)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center_coordinates = (cx, cy)
        
        # (可视化) 在原图上绘制结果
        cv2.circle(original_image, center_coordinates, 6, (0, 0, 255), -1) # 圆心点
        cv2.putText(original_image, f"Center: {center_coordinates}", (cx + 20, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 绘制主轮廓的边界框（可选）
        # cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # 绘制识别出的主轮廓（可选）
        # cv2.drawContours(original_image, [main_chart_contour], -1, (0, 255, 0), 2)


        print(f"高级方法找到的中心点: {center_coordinates}")
        return original_image, center_coordinates
    else:
        print("高级分析失败: 未能确定中心点。")
        return original_image, None

if __name__ == "__main__":
    img_dir = "./data/basic_images"
    # out_dir = "./data/out"
    # os.makedirs(out_dir, exist_ok=True)

    # 您需要将 circle6.jpg, circle7.jpg, circle8.jpg 放置在 data/basic_images 目录下

    files_to_process = ["circle6.jpg", "circle7.jpg", ] # 尝试处理所有三张图

    for fname_to_process in files_to_process:
        img_path = os.path.join(img_dir, fname_to_process)
        
        if not os.path.exists(img_path):
            print(f"警告: 文件 {img_path} 不存在，请确保已将图片放置在此目录。")
            continue

        print(f"\n处理图片: {img_path}")
        
        # 对于霍夫变换可能仍然有效的图（如 circle8.jpg 的中心圆），可以考虑单独处理
        # 但我们这里尝试通用方法
        result_image, center = find_chart_center_advanced(img_path)
        
        if result_image is not None:
            out_path = os.path.join(out_dir, f"{os.path.splitext(fname_to_process)[0]}_center_advanced.jpg")
            cv2.imwrite(out_path, result_image)
            print(f"高级方法结果已保存到 {out_path}")
        else:
            print(f"图片 {fname_to_process} 的中心点未找到。")