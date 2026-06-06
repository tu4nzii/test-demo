"""Amplifier crop/search loop for radar-chart prediction flows."""

from __future__ import annotations

import asyncio
from typing import Tuple

import aiohttp

from .model import check_find_point_async, encode_cv2_to_base64
from .visual import process_crop_axis_label_region_memory

async def async_crop_and_find(
    session: aiohttp.ClientSession, 
    executor, 
    base_image_cv, 
    dataset: dict, 
    start_outer_radius: float, 
    start_inner_radius: float,
    angle: float,
    intervals: Tuple[float, float], 
    max_radius: float,
    item_color_rgb
):
    center_x, center_y = dataset["pred_coords"]
    arg_a, arg_b = dataset["argument"]["a"], dataset["argument"]["b"]
    r_ticks = dataset["r_ticks"]
    
    # --- 【关键修复】传入前也做一次保护 ---
    curr_out = max(0, float(start_outer_radius))
    curr_in = max(0, float(start_inner_radius))
    
    outer_interval, inner_interval = intervals
    angle_width = 30
    loop_limit = 5
    loop = asyncio.get_running_loop()
    final_img_b64 = None
    
    for _ in range(loop_limit):
        def crop_task():
            # 这里的 process_crop_axis_label_region_memory 已经是修复版了
            img = process_crop_axis_label_region_memory(
                base_image_cv, center_x, center_y, angle, curr_out, angle_width, curr_in, 
                30, 3.0, r_ticks, arg_a, arg_b
            )
            return img

        try:
            cropped_img = await loop.run_in_executor(executor, crop_task)
            img_b64 = await loop.run_in_executor(executor, encode_cv2_to_base64, cropped_img)
            
            if not img_b64:
                break # 图片编码失败，可能原图是空的
                
            final_img_b64 = img_b64
            
            check_result = await check_find_point_async(session, img_b64, item_color_rgb)
            if check_result == "True":
                return img_b64
                
        except Exception as e:
            # 如果这里报错（比如CV崩溃），打印日志但不中断程序，尝试下一次循环或返回 None
            print(f"Warning: Crop/Check failed: {e}")
            break

        if curr_out >= max_radius:
            break
            
        curr_out += outer_interval
        curr_in -= inner_interval
        if curr_in < 0: curr_in = 0
    
    return final_img_b64
