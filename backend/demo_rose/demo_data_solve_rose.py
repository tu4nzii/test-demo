import json
import os
from typing import List, Dict
from typing import Tuple
import base64
import requests
import re
import time
import cv2
target_type = 'rose'

def process_rose_evaluation_data(chart_id: str) -> List[Dict]:
    """处理并整合玫瑰图评估数据
    功能：加载特定的评估用JSON文件、进行数据验证和筛选、整合轴线标签信息、保存处理结果
    """
    datasets = []
    chart_type = 'rose'
    
    # 定义文件路径
    json_root = os.path.join('./', 'data', 'output')
    rose_json_path = os.path.join(json_root, chart_type, f'{chart_id}.json')
    axes_json_path = os.path.join(json_root, chart_type, f'{chart_id}_axes.json')
    output_path = os.path.join('./', 'data', 'output', 'result', chart_type, f'{chart_id}_evalution_datasets.json')
    
    # 检查文件是否存在
    if not os.path.exists(rose_json_path):
        print(f"❌ 文件不存在: {rose_json_path}")
        return datasets
    
    try:
        # 读取rose_001.json文件
        with open(rose_json_path, 'r', encoding='utf-8') as f:
            rose_data = json.load(f)
        
        # 读取rose_001_axes.json文件并整合axis_labels
        if os.path.exists(axes_json_path):
            with open(axes_json_path, 'r', encoding='utf-8') as f:
                axes_data = json.load(f)
            if 'axis_labels' in axes_data:
                rose_data['axis_labels'] = axes_data['axis_labels']
                print(f"✅ 成功整合axis_labels数据")
        else:
            print(f"⚠️ 轴线数据文件不存在: {axes_json_path}")
        
        # 检查必要的字段
        if 'err_center' not in rose_data:
            print(f"❌ 文件缺少必要字段: err_center")
            return datasets
        
        err_center = rose_data['err_center']
        
        # 检查err_center，大于10则不添加到数据集
        if err_center > 10:
            print(f"err_center > 10, 跳过 {chart_id}")
            return datasets
            
        # 计算r_pixel_err
        if 'r_pixels' not in rose_data or 'r_ticks' not in rose_data or 'argument' not in rose_data:
            print(f"❌ 文件缺少必要的字段用于计算r_pixel_err")
            return datasets
            
        total_pred_r_pixel_err = 0
        for i in range(len(rose_data['r_pixels'])):
            if(rose_data['r_ticks'][0] == 0):
                if(len(rose_data['r_ticks'])==5):
                    r_tick = rose_data['r_ticks'][i*2+2]
                else:
                    r_tick = rose_data['r_ticks'][i+3]
            else:
                if(len(rose_data['r_ticks'])==4):
                    r_tick = rose_data['r_ticks'][i*2+1]
                else:
                    r_tick = rose_data['r_ticks'][i]
            pred_r_pixel = r_tick*rose_data['argument']['a']+rose_data['argument']['b']
            pred_r_err = abs(pred_r_pixel - rose_data['r_pixels'][i])
            total_pred_r_pixel_err += pred_r_err
        
        avg_err = total_pred_r_pixel_err/len(rose_data['r_pixels'])
        if(avg_err > 10):
            print(f"r_pixel_err > 10, 跳过 {chart_id}", avg_err)
            return datasets
            
        # 添加必要的信息
        rose_data['chart_id'] = chart_id
        rose_data['chart_type'] = chart_type
        rose_data['image_paths'] = {
            'no_grid': os.path.join('./','data',chart_type, f'{chart_id}.png'),
            'with_grid': os.path.join(json_root, chart_type, f'{chart_id}_encode.png')
        }
        
        # 转换labels和values为data键值对
        rose_data['data'] = {}
        if 'data_points' in rose_data:
            #rose
            for label, value in rose_data['data_points'].items():
                rose_data['data'][label] = value
        
        datasets.append(rose_data)
        
        # 输出统计信息
        print('图表类型:', chart_type, '总数:', 1)
        print(f"圆心检测精准率 (err_center < 5): {'1.000000' if err_center < 5 else '0.000000'}")
        print(f"圆心检测RMSE: {err_center:.6f}")
        print(f"r_pixel_err MAE : {avg_err:6f}")
        
        # 保存整合后的文件
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rose_data, f, ensure_ascii=False, indent=4)
        
        print(f"✅ 成功整合JSON文件并保存到: {output_path}")
        
    except Exception as e:
        print(f"❌ 处理文件失败: {e}")
        
    return datasets


if __name__ == '__main__':
    # 处理并整合玫瑰图评估数据
    datasets = process_rose_evaluation_data()
    
    # 可选：如果需要仍然保存原始的evaluation_datasets.json文件
    # if os.path.exists('evaluation_datasets.json'):
    #     os.remove('evaluation_datasets.json')
    # with open(f'evaluation_datasets.json', 'w', encoding='utf-8') as f:
    #     json.dump(datasets, f, ensure_ascii=False, indent=4)
    
#12 45 75 112 118 126 164 167 179 197 205 221 228