# 霍夫变换或其他直线检测方法
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def detect_candidate_lines(image_gray, 
                           canny_threshold1=50, 
                           canny_threshold2=150, 
                           hough_threshold=20, 
                           min_length=20, 
                           max_gap=20):
    """
    对灰度图像进行边缘检测与直线检测，返回线段列表
    每条线段格式为 [x1, y1, x2, y2]
    """
    edges = cv2.Canny(image_gray, canny_threshold1, canny_threshold2)
    lines = cv2.HoughLinesP(edges, 
                            rho=1, 
                            theta=np.pi / 180, 
                            threshold=hough_threshold,
                            minLineLength=min_length, 
                            maxLineGap=max_gap)
    
    if lines is None:
        return []
    
    return [line[0] for line in lines]  # 每个line是 [[x1, y1, x2, y2]] 的结构

# 🧪 内置测试
def detect_lines_main():
    from config import IMG_PATHS, DEBUG_OUTPUT_DIRS, CLEAR_OUTPUT_BEFORE_RUN
    from utils.image_io import load_image, save_image
    import os
    import shutil

    output_dir = DEBUG_OUTPUT_DIRS['detect_lines']
    # 配置：是否清空输出目录
    if CLEAR_OUTPUT_BEFORE_RUN.get('detect_lines', False):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for name, path in IMG_PATHS.items():
        img = load_image(path)
        print(f"[Info] 处理: {name} => {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lines = detect_candidate_lines(gray)

        # 可视化线段
        vis = img.copy()
        for x1, y1, x2, y2 in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        out_path = os.path.join(output_dir, f"result_{name}.png")
        save_image(vis, out_path)
        print(f"[Debug] 共检测到 {len(lines)} 条线段，结果保存至: {out_path}")

if __name__ == '__main__':
    detect_lines_main()