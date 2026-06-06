"""Visual crop and overlay helpers for radar-chart prediction flows."""

from __future__ import annotations

import math

import cv2
import numpy as np

def draw_sector_grid(image, center_x, center_y, arg_a, arg_b, r_ticks, start_angle_deg, end_angle_deg, extension_deg=5, outer_radius=300, inner_radius=0):
    # 保持原逻辑不变
    line_color = (100, 100, 100)
    thickness = 1
    draw_start = start_angle_deg - extension_deg
    draw_end = end_angle_deg + extension_deg
    local_ticks = [t for t in r_ticks if t != 0]

    for tick in local_ticks:
        radius = int(arg_a * tick + arg_b)
        if radius <= 0 or radius >= outer_radius or radius <= inner_radius: 
            continue
        circumference = 2 * math.pi * radius
        if circumference == 0: continue
        
        dash_len_px = 3
        gap_len_px = 4
        total_step_rad = (dash_len_px + gap_len_px) / radius
        dash_step_rad = dash_len_px / radius
        
        current_rad = math.radians(draw_start)
        end_rad = math.radians(draw_end)
        
        if end_rad < current_rad:
            end_rad += 2 * math.pi

        while current_rad < end_rad:
            seg_end_rad = current_rad + dash_step_rad
            if seg_end_rad > end_rad: 
                seg_end_rad = end_rad
            
            p1_x = int(center_x + radius * math.cos(current_rad))
            p1_y = int(center_y - radius * math.sin(current_rad))
            p2_x = int(center_x + radius * math.cos(seg_end_rad))
            p2_y = int(center_y - radius * math.sin(seg_end_rad))
            
            cv2.line(image, (p1_x, p1_y), (p2_x, p2_y), line_color, thickness, cv2.LINE_AA)
            current_rad += total_step_rad
    return image

def process_crop_axis_label_region_memory(source_image, center_x, center_y, angle_deg, outer_radius, angle_width=30, inner_radius=0, label_offset=30, scale_factor=2.0, r_ticks=[], arg_a=0, arg_b=0):
    if source_image is None:
        return None
    h, w = source_image.shape[:2]

    # --- 修复开始：防御性检查 ---
    # 强制将半径转换为非负整数
    out_r = max(0, int(outer_radius))
    in_r = max(0, int(inner_radius))
    
    # 如果外径为0，说明无法裁剪，直接返回原图或者空图
    if out_r == 0:
        return source_image
    # --- 修复结束 ---

    start_angle = angle_deg - angle_width / 2
    end_angle = angle_deg + angle_width / 2
    
    start_angle_cv = -end_angle
    end_angle_cv = -start_angle

    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 使用安全的 out_r 和 in_r
    cv2.ellipse(mask, (int(center_x), int(center_y)), (out_r, out_r), 
                angle=0, startAngle=start_angle_cv, endAngle=end_angle_cv, 
                color=255, thickness=-1, lineType=cv2.LINE_AA)

    if in_r > 0:
        cv2.ellipse(mask, (int(center_x), int(center_y)), (in_r, in_r), 
                    angle=0, startAngle=start_angle_cv - 5, endAngle=end_angle_cv + 5, 
                    color=0, thickness=-1, lineType=cv2.LINE_AA)

    # ... (后续代码保持不变，注意用到 radius 的地方都要确保安全，但前面已经处理了掩码，后面问题不大) ...
    # 为了安全，后续用到 scaled_outer_radius 的地方也要注意
    
    coords = cv2.findNonZero(mask)
    if coords is None: return source_image

    x, y, w_sector, h_sector = cv2.boundingRect(coords)
    pad = 30
    
    crop_x1 = max(0, x - pad)
    crop_y1 = max(0, y - pad)
    crop_x2 = min(w, x + w_sector + pad)
    crop_y2 = min(h, y + h_sector + pad)
    
    crop_img = source_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    
    crop_img[crop_mask == 0] = 255

    if scale_factor != 1.0:
        new_w = int(crop_img.shape[1] * scale_factor)
        new_h = int(crop_img.shape[0] * scale_factor)
        crop_img_scaled = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        crop_img_scaled = crop_img

    scaled_center_x = (center_x - crop_x1) * scale_factor
    scaled_center_y = (center_y - crop_y1) * scale_factor
    
    scaled_arg_a = arg_a * scale_factor
    scaled_arg_b = arg_b * scale_factor
    
    scaled_outer_radius = out_r * scale_factor # 使用安全的 out_r
    scaled_inner_radius = in_r * scale_factor  # 使用安全的 in_r

    interpolated_ticks = []
    for i in range(len(r_ticks)):
        interpolated_ticks.append(r_ticks[i])
        if i < len(r_ticks) - 1:
            mid_val = (r_ticks[i] + r_ticks[i+1]) / 2
            interpolated_ticks.append(mid_val)

    draw_sector_grid(
        crop_img_scaled,
        scaled_center_x,
        scaled_center_y,
        scaled_arg_a,
        scaled_arg_b,
        interpolated_ticks,
        start_angle,
        end_angle,
        extension_deg=8,
        outer_radius=scaled_outer_radius,
        inner_radius=scaled_inner_radius
    )
    
    # ... (字体绘制代码保持不变) ...
    base_font_scale = 0.3  
    final_font_scale = base_font_scale * (scale_factor ** 0.5) 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_color = (0, 0, 0)
    font_thickness = 1 
    axis_offset_deg = 3.0  
    text_rad_offset = 0  

    for tick in interpolated_ticks:
        radius_scaled = int(scaled_arg_a * tick + scaled_arg_b)
        if radius_scaled <= 0 or radius_scaled >= scaled_outer_radius or radius_scaled <= scaled_inner_radius:
            continue
        final_text_radius = radius_scaled + text_rad_offset
        text = str(int(tick)) if tick % 1 == 0 else f"{tick:.2f}"
        (t_w, t_h), baseline = cv2.getTextSize(text, font, final_font_scale, font_thickness)
        target_angle_rad = math.radians(angle_deg + axis_offset_deg)
        tx = scaled_center_x + final_text_radius * math.cos(target_angle_rad)
        ty = scaled_center_y - final_text_radius * math.sin(target_angle_rad)
        cv2.putText(crop_img_scaled, text, (int(tx - t_w/2), int(ty + t_h/2)), 
                    font, final_font_scale, font_color, font_thickness, cv2.LINE_AA)

    return crop_img_scaled

def draw_angle_indicator(image, center_x, center_y, target_angle, radius, arc_color=(0, 0, 255), line_color=(0, 0, 255), 
                         arc_thickness=2, line_thickness=2, arc_angle_width=10, line_length_ratio=0.3):
    start_angle =  - target_angle - arc_angle_width // 2
    end_angle =  - target_angle + arc_angle_width // 2
    if radius <= 0:
        radius = 0
    radius = int(radius)
    cv2.ellipse(image, (int(center_x), int(center_y)), (radius, radius), 0, start_angle, end_angle, arc_color, arc_thickness, lineType=cv2.LINE_AA)
    angle_rad = math.radians(target_angle)
    outer_x = int(center_x + (radius+line_length_ratio*radius) * math.cos(angle_rad))
    outer_y = int(center_y - (radius+line_length_ratio*radius) * math.sin(angle_rad))
    inner_radius = int(radius * (1 - line_length_ratio))
    inner_x = int(center_x + inner_radius * math.cos(angle_rad))
    inner_y = int(center_y - inner_radius * math.sin(angle_rad))
    cv2.line(image, (outer_x, outer_y), (inner_x, inner_y), line_color, line_thickness, lineType=cv2.LINE_AA)
    return image
