import cv2  # 图像处理
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 可视化
from pathlib import Path  # 文件路径处理

# 输入图片路径和结果保存路径（请根据实际情况修改）
img_dir = "./data/basic_images"
save_dir = "./data/out"


def detect_circle_center(image, dp=1.2, minDist=100, param1=50, param2=30, minRadius=0, maxRadius=0, debug=False):
    """
    利用 HoughCircles 检测图像中的圆形，用以定位极坐标系图的外圈圆，从而确定圆心。

    参数：
        image: 待处理的彩色图像（BGR格式）
        dp: 累加器分辨率与图像分辨率的反比，值越小检测越精细
        minDist: 检测到的圆之间的最小距离
        param1: Canny 边缘检测的高阈值
        param2: 检测圆心时的累加器阈值，值越小检测到的圆越多
        minRadius: 允许检测的最小圆半径
        maxRadius: 允许检测的最大圆半径
        debug: 是否显示中间调试结果

    返回：
        center: (x, y) 圆心坐标；未检测到则返回 None
        radius: 圆的半径；未检测到则返回 None
        image: 绘制结果后的图像
    """
    # 转换为灰度图并进行高斯模糊降噪
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 使用 HoughCircles 检测圆形
    circles = cv2.HoughCircles(gray_blurred,
    # circles=cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               dp=dp,
                               minDist=minDist,
                               param1=param1,
                               param2=param2,
                               minRadius=minRadius,
                               maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 选取第一组检测到的圆，一般该圆为极坐标系的外圈圆
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            if debug:
                # 绘制圆心（绿色小点）和圆轮廓（红色圆圈）
                cv2.circle(image, center, 3, (0, 255, 0), -1)  # 圆心
                cv2.circle(image, center, radius, (0, 0, 255), 2)  # 圆轮廓
            return center, radius, image
    else:
        if debug:
            print("未检测到圆形！")
    return None, None, image
    # return center, radius, image

def draw_radial_lines(image, center, radius, num_lines=12, color=(255, 0, 0), thickness=2):
    """
    以圆心为中心，绘制极坐标图中的径向刻度线（射线形式）

    参数：
        image: 图像
        center: 圆心坐标 (x, y)
        radius: 射线长度，通常为检测到的圆半径
        num_lines: 射线数量（如12根表示每隔30度一根）
        color: 射线颜色（默认蓝色）
        thickness: 射线线宽
    返回：
        image: 绘制后的图像
    """
    cx, cy = center
    for i in range(num_lines):
        angle = 2 * np.pi * i / num_lines
        x = int(cx + radius * np.cos(angle))
        y = int(cy + radius * np.sin(angle))
        cv2.line(image, center, (x, y), color, thickness)
    return image

def draw_concentric_circles(image, center, max_radius, num_circles=4, color=(0, 255, 255), thickness=1):
    """
    在图像上以圆心为中心绘制若干同心圆，用以表示极坐标图上的环形刻度。

    参数：
        image: 原始图像
        center: 圆心 (x, y)
        max_radius: 最大半径（一般是检测到的外圈圆半径）
        num_circles: 要绘制的圆环数量
        color: 圆环颜色（默认黄色）
        thickness: 线条粗细
    返回：
        image: 添加圆环后的图像
    """
    step = max_radius / num_circles
    for i in range(1, num_circles + 1):
        r = int(i * step)
        cv2.circle(image, center, r, color, thickness)
    return image


if __name__ == "__main__":
    # 保证输出目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 遍历图片目录下所有图像
    for path in Path(img_dir).iterdir():
        if "circle" not in path.name.lower():
            continue
        filepath = str(path)
        image = cv2.imread(filepath)
        if image is None:
            continue

        # 定位圆心，启用调试模式以便显示检测结果
        center, radius, annotated_image = detect_circle_center(
            image,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=0,
            maxRadius=0,
            debug=True
        )

        print(f"图像: {filepath}")
        # if center is not None:
        #     print(f"检测到的圆心坐标: {center}， 半径: {radius}")

        #     # 绘制极坐标径向刻度线
        #     annotated_image = draw_radial_lines(
        #         annotated_image,
        #         center=center,
        #         radius=radius,
        #         num_lines=12,
        #         color=(255, 0, 0),
        #         thickness=2
        #     )

        #     # 绘制环形刻度线（同心圆）
        #     annotated_image = draw_concentric_circles(
        #         annotated_image,
        #         center=center,
        #         max_radius=radius,
        #         num_circles=4,  # 可以调整为更多
        #         color=(0, 255, 0),  # 绿色线
        #         thickness=2
        #     )
        # else:
        #     print("未检测到圆形！")
        # 显示调试结果
        # plt.figure()
        # plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        # plt.title(f"检测结果: {path.name}")
        # plt.axis("off")
        # plt.show()

        # 保存结果图像
        save_path = Path(save_dir) / (path.stem + "_polar_center.png")
        cv2.imwrite(str(save_path), annotated_image)
