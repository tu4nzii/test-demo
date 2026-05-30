"""Amplifier crop/search loop for rose-chart prediction flows."""

from __future__ import annotations

import asyncio
from typing import Tuple

import aiohttp

from .model import check_find_point_async, encode_cv2_to_base64
from .visual import crop_axis_label_region_memory

async def async_crop_and_find_rose(
    session: aiohttp.ClientSession,
    executor,
    base_image_cv, # 原图 numpy
    dataset: dict,
    start_outer_radius: float,
    start_inner_radius: float,
    angle: float,
    intervals: Tuple[float, float], # (outer_int, inner_int)
    max_radius: float,
    item_color_rgb
):
    """
    异步版的 crop + find_point 循环逻辑 (Rose Chart)
    """
    center_x, center_y = dataset["pred_coords"]
    arg_a, arg_b = dataset["argument"]["a"], dataset["argument"]["b"]
    r_ticks = dataset["r_ticks"]
    
    # Rose Chart 特有的参数
    angle_width = int(360 / len(dataset['axis_labels']))
    scale_factor = 3
    
    # --- 【关键修复】传入前也做一次保护 ---
    curr_out = max(0, float(start_outer_radius))
    curr_in = max(0, float(start_inner_radius))
    
    outer_interval, inner_interval = intervals
    
    loop_limit = 5
    loop = asyncio.get_running_loop()
    final_img_b64 = None
    
    for _ in range(loop_limit):
        # 1. 裁剪 (CPU Bound -> Executor)
        def crop_task():
            return crop_axis_label_region_memory(
                base_image_cv, center_x, center_y, angle, curr_out, angle_width, curr_in, 
                30, scale_factor, r_ticks, arg_a, arg_b
            )
            
        cropped_img = await loop.run_in_executor(executor, crop_task)
        img_b64 = await loop.run_in_executor(executor, encode_cv2_to_base64, cropped_img)
        
        if not img_b64:
            break
            
        final_img_b64 = img_b64 # Fallback
        
        # 2. 检查颜色 (IO Bound -> Async)
        check_result = await check_find_point_async(session, img_b64, item_color_rgb)
        
        if check_result == "True":
            return img_b64
        
        # 3. 调整半径
        if curr_out >= max_radius:
            break
            
        curr_out += outer_interval
        curr_in -= inner_interval
        if curr_in < 0: curr_in = 0
        
    return final_img_b64
