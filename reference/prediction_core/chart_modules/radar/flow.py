import json
import os
from typing import Dict
import time
import cv2
import asyncio
import aiohttp
import random
import sys
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "prediction_core").is_dir():
        sys.path.insert(0, str(_parent))
        break
from .model import (
    call_llm_response_async,
    encode_cv2_to_base64,
    llm_model,
    read_file_to_base64,
)
from .amplifier import async_crop_and_find
from .prompts import generate_prompt
from .visual import draw_angle_indicator

try:
    from tqdm.asyncio import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# --- 全局配置 ---

# --- 并发控制配置 ---
# 建议设置为 20-50，过高会导致 API 报 429 错误
CONCURRENT_LIMIT = 30 

# --- 辅助函数 (数学逻辑保持不变) ---



# --- 内存辅助工具 ---



# --- Async HTTP 请求封装 ---



# --- Prompt 生成器 (保持不变) ---


# --- 核心逻辑 ---


async def process_single_task(sem: asyncio.Semaphore, session: aiohttp.ClientSession, executor, task_info: Dict):
    dataset = task_info['dataset']
    grid_type = task_info['grid_type']
    item_name = task_info['item_name']
    origin_value = task_info['value']
    
    chart_id = dataset['chart_id']
    image_path = dataset["image_paths"][grid_type].replace('\\', '/')

    # 结果容器
    result_data = {
        'origin': origin_value,
        'initial': None
    }
    
    # 异步读取主图
    if not os.path.exists(image_path):
        return (chart_id, item_name, grid_type, result_data)

    loop = asyncio.get_running_loop()
    # 读图为 B64 用于发送，同时读图为 CV2 对象用于后续处理
    base_img_b64_fut = loop.run_in_executor(executor, read_file_to_base64, image_path)
    
    # 只有 with_grid 需要 CV2 对象进行后续的画图操作
    base_img_cv_fut = None
    if grid_type == 'with_grid':
        base_img_cv_fut = loop.run_in_executor(executor, cv2.imread, image_path)
        
    base_img_b64 = await base_img_b64_fut
    
    # --- Step 1: Initial Prediction ---
    initial_prompt = generate_prompt(item_name, grid_type, dataset)
    
    async with sem:
        coords = await call_llm_response_async(session, initial_prompt, base_img_b64, item_name)
    
    result_data['initial'] = coords[0] if coords else None
    
    # 如果是 no_grid，到此结束
    if grid_type != 'with_grid':
        return (chart_id, item_name, grid_type, result_data)

    # --- Step 2: Feedback & Amplifier (Only for with_grid) ---
    
    # 获取 CV2 图片对象
    base_img_cv = await base_img_cv_fut
    if base_img_cv is None:
        return (chart_id, item_name, grid_type, result_data)

    # Feedback 配置
    feedback_tick = []
    if coords[0] is not None:
        feedback_tick.append(coords[0])
    else:
        feedback_tick.append(origin_value)

    axis_labels = dataset.get('axis_labels')
    label_to_angle = {v: int(k) for k, v in axis_labels.items()}
    item_label = item_name.split(',')[1].strip()
    target_angle = label_to_angle.get(item_label, 0)
    
    center_x, center_y = dataset["pred_coords"]
    a, b = dataset["argument"]["a"], dataset["argument"]["b"]
    
    # Feedback Loop (串行依赖)
    feedback_counts = 3
    for _ in range(feedback_counts):
        curr_val = feedback_tick[-1] if feedback_tick[-1] is not None else feedback_tick[0]
        pre_r = int(a * curr_val + b)
        
        # 在 Executor 中画图
        def draw_task():
            temp = base_img_cv.copy()
            draw_angle_indicator(temp, center_x, center_y, target_angle, pre_r, 
                               arc_thickness=1, line_thickness=1, arc_angle_width=20, line_length_ratio=0.1)
            return encode_cv2_to_base64(temp)
            
        fb_img_b64 = await loop.run_in_executor(executor, draw_task)
        fb_prompt = generate_prompt(item_name, 'feedback', dataset, curr_val)
        
        async with sem:
            fb_coords = await call_llm_response_async(session, fb_prompt, fb_img_b64, item_name)
            
        if fb_coords[0] is not None:
            feedback_tick.append(fb_coords[0])
        else:
            feedback_tick.append(feedback_tick[-1])
            
    result_data['feedback'] = feedback_tick

    # Amplifier Logic
    # 准备 Amplifier 需要的资源
    no_grid_path = dataset["image_paths"]['no_grid'].replace('\\', '/')
    # 读取 no_grid 图片用于裁剪 (只读一次)
    no_grid_img_cv = await loop.run_in_executor(executor, cv2.imread, no_grid_path)
    
    if no_grid_img_cv is not None:
        color = dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')
        # rgb_colors 用于 find_point
        
        r_ticks = dataset["r_ticks"]
        max_radius = r_ticks[-1]*a + b
        max_radius_pixel = (r_ticks[-1] - r_ticks[0])*a + b
        
        amplifier_outter_interval = max_radius_pixel / len(r_ticks)
        amplifier_inner_interval = max_radius_pixel / len(r_ticks)
        intervals = (amplifier_outter_interval, amplifier_inner_interval)
        
        amplifier_feedback_ticks = []
        amplifier_grid_ticks = []
        
        current_feedback_radius = int(a * feedback_tick[-1] + b) if feedback_tick[-1] is not None else 0
        current_grid_radius = int(a * coords[0] + b) if coords and coords[0] is not None else 0
        
        amplifier_counts = 3
        
        for _ in range(amplifier_counts):
            # 计算内外半径
            def get_radii(curr_rad):
                if curr_rad != 0:
                    inner = curr_rad - amplifier_outter_interval
                    outer = curr_rad + amplifier_inner_interval
                    if outer > max_radius: outer = max_radius + 50
                    if inner < 0: inner = 0
                    return inner, outer
                return 0, curr_rad
                
            fb_in, fb_out = get_radii(current_feedback_radius)
            gd_in, gd_out = get_radii(current_grid_radius)
            
            # 并发执行带 find_point 逻辑的裁剪
            # 这里调用 async_crop_and_find，它内部会进行多次 LLM 检查，直到找到点或循环结束
            task_fb = async_crop_and_find(
                session, executor, no_grid_img_cv, dataset, 
                fb_out, fb_in, target_angle, intervals, max_radius, color
            )
            task_gd = async_crop_and_find(
                session, executor, no_grid_img_cv, dataset, 
                gd_out, gd_in, target_angle, intervals, max_radius, color
            )
            
            # 等待两个裁剪任务完成 (大部分时间花在网络等待上)
            async with sem:
                 amp_fb_b64, amp_gd_b64 = await asyncio.gather(task_fb, task_gd)
            
            # 拿到符合条件的图片后，进行 Amplifier 预测
            amp_prompt = generate_prompt(item_name, 'amplifier', dataset)
            
            pred_tasks = []
            if amp_fb_b64:
                pred_tasks.append(call_llm_response_async(session, amp_prompt, amp_fb_b64, item_name))
            else:
                pred_tasks.append(asyncio.sleep(0, result=(None, None)))
                
            if amp_gd_b64:
                pred_tasks.append(call_llm_response_async(session, amp_prompt, amp_gd_b64, item_name))
            else:
                pred_tasks.append(asyncio.sleep(0, result=(None, None)))
            
            async with sem:
                results = await asyncio.gather(*pred_tasks)
            
            res_fb, res_gd = results
            
            # 更新逻辑
            if res_fb and res_fb[0] is not None:
                amplifier_feedback_ticks.append(res_fb[0])
                current_feedback_radius = int(a * res_fb[0] + b)
            elif amplifier_feedback_ticks:
                amplifier_feedback_ticks.append(amplifier_feedback_ticks[-1])
                
            if res_gd and res_gd[0] is not None:
                amplifier_grid_ticks.append(res_gd[0])
                current_grid_radius = int(a * res_gd[0] + b)
            elif amplifier_grid_ticks:
                amplifier_grid_ticks.append(amplifier_grid_ticks[-1])
        
        result_data['amplifier_feedback_ticks'] = amplifier_feedback_ticks
        result_data['amplifier_grid_ticks'] = amplifier_grid_ticks
        
    return (chart_id, item_name, grid_type, result_data)


# --- 主程序入口 ---

async def main(chart_ids: list[str] | None = None):
    # 1. 加载数据
    try:
        with open(f'evaluation_datasets_with_axes_radar.json', 'r', encoding='utf-8') as f:
            datasets = json.load(f)
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return

    # 2. 生成任务列表
    tasks_info = []
    results_by_image = {}
    
    max_charts = len(datasets) 
    print(f"Loading {max_charts} charts...")
    chart_ids = chart_ids or [f'radar_{str(i).zfill(3)}' for i in range(0, 51)]
    print(f"Processing charts: {chart_ids}")
    for dataset in datasets[:max_charts]:
        chart_id = dataset['chart_id']
        if chart_id not in chart_ids: continue
        if dataset['chart_type'] != 'radar' or not dataset['data']: continue
        if dataset['num_entities'] > 5: continue
        
        # 初始化结果字典结构
        if chart_id not in results_by_image:
            results_by_image[chart_id] = {
                'chart_type': dataset['chart_type'],
                'data': {}
            }
            
        for grid_type in ['with_grid', 'no_grid']:
            for top_key, nested_dict in dataset['data'].items():
                for sub_key, value in nested_dict.items():
                    item_name = f"{top_key},{sub_key}"
                    
                    if item_name not in results_by_image[chart_id]['data']:
                        results_by_image[chart_id]['data'][item_name] = {'origin': value}
                        
                    task = {
                        'dataset': dataset,
                        'grid_type': grid_type,
                        'item_name': item_name,
                        'value': value
                    }
                    tasks_info.append(task)

    print(f"Total tasks: {len(tasks_info)}")
    if not tasks_info:
        print("No matching radar chart tasks. Nothing to run.")
        return
    
    # 3. 异步执行
    start_time = time.time()
    # 随机打乱任务顺序, 避免扎堆请求
    random.shuffle(tasks_info) 
    # 限制并发请求数
    sem = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    # 创建线程池，用于 OpenCV 操作，核心数可根据机器配置调整
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    
    # 开启 HTTP Session
    timeout = aiohttp.ClientTimeout(total=None) # 单个请求通过 headers 和参数控制超时，Session 不设总超时
    connector = aiohttp.TCPConnector(limit=None, ttl_dns_cache=300) 
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建 Task 列表
        futures = [process_single_task(sem, session, executor, t) for t in tasks_info]
        
        processed_results = []
        # 使用 as_completed 结合 tqdm 显示进度
        for f in tqdm(asyncio.as_completed(futures), total=len(futures), desc="Processing"):
            res = await f
            if res:
                processed_results.append(res)
    
    # 4. 汇总结果
    print("Aggregating results...")
    for chart_id, item_name, grid_type, data in processed_results:
        target = results_by_image[chart_id]['data'][item_name]
        
        # 写入数据
        if data['initial'] is not None:
            target[grid_type] = data['initial']
            
        if grid_type == 'with_grid':
            if 'feedback' in data: 
                target['feedback'] = data['feedback']
            if 'amplifier_feedback_ticks' in data: 
                target['amplifier_feedback_ticks'] = data['amplifier_feedback_ticks']
            if 'amplifier_grid_ticks' in data: 
                target['amplifier_grid_ticks'] = data['amplifier_grid_ticks']

    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds.")

    # 5. 保存文件
    output_filename = f'coordinates_by_image_radar_{llm_model}_async.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_by_image, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Radar chart value prediction')
    parser.add_argument('--chart-ids', nargs='+', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()
    # Windows 需要设置策略
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main(chart_ids=args.chart_ids))
