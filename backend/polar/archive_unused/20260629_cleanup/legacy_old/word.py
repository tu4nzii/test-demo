import cv2
import easyocr
import numpy as np
import json

# 加载图像
image_path = 'data/output/radar/legend_radar_001.png'
image = cv2.imread(image_path)

# 初始化 OCR 识别器
reader = easyocr.Reader(['en'])
results = reader.readtext(image_path)

# 用于存储识别结果
text_data = []

# 遍历识别结果并绘制
for (bbox, text, conf) in results:
    clean_text = text.strip()
    if conf > 0.6 and clean_text.isalpha() and len(clean_text) > 2:
        # 获取四个顶点坐标
        pts = [tuple(map(int, point)) for point in bbox]
        # 计算中心点
        center_x = int(sum(p[0] for p in pts) / 4)
        center_y = int(sum(p[1] for p in pts) / 4)
        center = (center_x, center_y)

        # 绘制矩形框和中心点
        cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(image, center, radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(image, clean_text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 存储文本、坐标和中心点
        text_data.append({
            'text': clean_text,
            'confidence': round(conf, 2),
            'bbox': pts,
            'center': center
        })

# 保存图像
cv2.imwrite('output_visualized.png', image)

# 保存识别结果为 JSON 文件
with open('recognized_text.json', 'w', encoding='utf-8') as f:
    json.dump(text_data, f, ensure_ascii=False, indent=2)

# 打印结果
for item in text_data:
    print(f"文本: {item['text']}，中心点: {item['center']}")
