import cv2
import numpy as np
import json
import os

class SimpleLegendColorExtractor:
    def __init__(self):
        self.min_color_block_area = 30
        self.max_color_block_area = 1000
        self.color_distance_threshold = 30
        self.min_saturation = 30
        self.min_value = 50

    def extract_legend_colors(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([0, self.min_saturation, self.min_value])
        upper_bound = np.array([180, 255, 255])
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        kernel = np.ones((3, 3), np.uint8)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)
        color_mask = cv2.dilate(color_mask, kernel, iterations=2)
        color_mask = cv2.erode(color_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        block_info = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            if self.min_color_block_area < area < self.max_color_block_area and 0.3 < aspect_ratio < 3.0:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mean_color = cv2.mean(image, mask=mask)[:3]
                bgr_color = np.uint8([[mean_color]])
                hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0][0]

                if hsv_color[1] >= self.min_saturation:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    block_info.append({
                        'color': np.uint8(mean_color),
                        'position': {
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'center_x': center_x,
                            'center_y': center_y
                        },
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })

        unique_color_info = []
        for block in block_info:
            is_unique = True
            current_color = block['color']
            for unique_block in unique_color_info:
                unique_color = unique_block['color']
                hsv1 = cv2.cvtColor(np.uint8([[current_color]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv2 = cv2.cvtColor(np.uint8([[unique_color]]), cv2.COLOR_BGR2HSV)[0][0]
                h_diff = min(abs(int(hsv1[0]) - int(hsv2[0])), 180 - abs(int(hsv1[0]) - int(hsv2[0])))
                s_diff = abs(int(hsv1[1]) - int(hsv2[1]))
                v_diff = abs(int(hsv1[2]) - int(hsv2[2]))
                weighted_distance = h_diff * 2 + s_diff + v_diff
                if weighted_distance <= self.color_distance_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_color_info.append(block)

        # 按色调排序
        hsv_color_info = []
        for block in unique_color_info:
            hsv = cv2.cvtColor(np.uint8([[block['color']]]), cv2.COLOR_BGR2HSV)[0][0]
            hsv_color_info.append((hsv[0], block))
        hsv_color_info.sort(key=lambda x: x[0])
        return [block for _, block in hsv_color_info]

    def convert_to_hex(self, bgr_color):
        rgb_color = bgr_color[::-1]
        return f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}".upper()

def main():
    image_path = 'data/output/radar/legend_radar_001.png'
    output_json = 'legend_colors_with_position.json'
    output_image = 'legend_color_marked.png'

    extractor = SimpleLegendColorExtractor()
    legend_color_info = extractor.extract_legend_colors(image_path)

    image = cv2.imread(image_path)
    export_data = []

    for i, color_info in enumerate(legend_color_info):
        hex_color = extractor.convert_to_hex(color_info['color'])
        pos = color_info['position']
        center = (pos['center_x'], pos['center_y'])

        # 可视化标注
        cv2.circle(image, center, 4, (0, 0, 255), -1)
        cv2.putText(image, f"{i+1}", (center[0]+5, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        export_data.append({
            'index': i + 1,
            'hex_color': hex_color,
            'top_left': {'x': pos['x'], 'y': pos['y']},
            'center': {'x': center[0], 'y': center[1]},
            'area': round(color_info['area'], 2)
        })

    # 保存图像和数据
    cv2.imwrite(output_image, image)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"已提取 {len(export_data)} 个图例颜色，结果已保存为：\n- {output_image}\n- {output_json}")

if __name__ == "__main__":
    main()
