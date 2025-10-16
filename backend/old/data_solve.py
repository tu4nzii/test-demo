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
def load_evaluation_datasets(json_root: str) -> List[Dict]:
    """从指定目录加载所有评估用JSON文件
    实际JSON文件结构：data/circle/{类型}/{图表ID}.json
    """
    datasets = []
    correct_count = 0
    total_count = 0
    sum_squared_err = 0.0
    pred_r_pixel_err = []
    chart_target_type= target_type
    # 遍历circle目录下的所有子目录（radar/rose等）
    for chart_type in os.listdir(json_root):
        chart_type_path = os.path.join(json_root, chart_type)
        if not os.path.isdir(chart_type_path):
            print(chart_type_path)
            continue
        # 处理目录下的所有JSON文件
        for filename in os.listdir(chart_type_path):
            if total_count >= 100:
                break
            #if filename.endswith('.json') and filename.startswith(f'{chart_type}'): #radar
            if (filename.endswith('.json') and chart_type == chart_target_type): #rose

                chart_id = os.path.splitext(filename)[0]
                json_path = os.path.join(chart_type_path, filename)
                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        dataset = json.load(f)
                        # 添加图表类型和ID信息
                        
                        # 检查err_center，大于5则不添加到数据集
                        err_center = dataset.get('err_center')
                        total_count += 1
                        if err_center > 10:
                            print(f"err_center > 10, 跳过 {chart_id}")
                            continue
                         # 更新圆心检测统计数据
                        if err_center < 5:
                            correct_count += 1
                        sum_squared_err += err_center **2
                        ##r_pixel_err
                        total_pred_r_pixel_err = 0
                        for i in range(len(dataset['r_pixels'])):
                            if(dataset['r_ticks'][0] == 0):
                                if(len(dataset['r_ticks'])==9):
                                    r_tick = dataset['r_ticks'][i*4+4]
                                else:
                                    r_tick = dataset['r_ticks'][i+3]
                            else:
                                if(len(dataset['r_ticks'])==8):
                                    r_tick = dataset['r_ticks'][i*4+3]
                                else:
                                    r_tick = dataset['r_ticks'][i]
                            pred_r_pixel = r_tick*dataset['argument']['a']+dataset['argument']['b']
                            pred_r_err = abs(pred_r_pixel - dataset['r_pixels'][i])
                            total_pred_r_pixel_err += pred_r_err
                        avg_err = total_pred_r_pixel_err/len(dataset['r_pixels'])
                        if(avg_err > 10):
                            print(f"r_pixel_err > 10, 跳过 {chart_id}", avg_err)
                            continue
                        pred_r_pixel_err.append(avg_err)
                        dataset['chart_id'] = chart_id
                        dataset['chart_type'] = chart_type
                        dataset['image_paths'] = {
                            'no_grid': os.path.join('./','data','circle',chart_type, f'{chart_id}.png'),
                            'with_grid': os.path.join(json_root, chart_type, f'{chart_id}+encode.jpg')
                        }
                        # 转换labels和values为data键值对
                        dataset['data'] = {}
                        # 遍历每个实体的数据点
                        if 'data_points' in dataset:
                            #遍历每个实体的数据点
                            #radar
                            # for entity, values in dataset['data_points'].items():
                            #     #遍历该实体的每个标签值对
                            #     for label, value in values.items():
                            #         # 生成单一名称格式的键
                            #         key_name = f"{entity},{label}"
                            #         dataset['data'][key_name] = value

                            #rose
                            for label, value in dataset['data_points'].items():
                                dataset['data'][label] = value
                        
                        else:
                            # 当data_points不存在时，直接使用labels和values
                            for label, value in zip(dataset['labels'], dataset['values']):
                                dataset['data'][label] = value
                    
                        datasets.append(dataset)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON解析失败 {filename}: {e}")
    if total_count > 0:
        precision = correct_count / total_count
        rmse = (sum_squared_err / total_count)** 0.5
        print('图表类型:', chart_target_type, '总数:', total_count)
        print(f"圆心检测精准率 (err_center < 5): {precision:.6f}")
        print(f"圆心检测RMSE: {rmse:.6f}")
        print(f"r_pixel_err MAE : {sum(pred_r_pixel_err)/len(pred_r_pixel_err):6f}")
    else:
        print("没有符合条件的数据集用于计算圆心检测指标")
    return datasets

datasets = load_evaluation_datasets(json_root=os.path.join('./','data', 'out', 'circle'))


if __name__ == '__main__':
    #将数据集保存到JSON文件
    #生成数据集的prompt
    # 若文件存在，删除evaluation_datasets.json文件
    if os.path.exists('evaluation_datasets.json'):
        os.remove('evaluation_datasets.json')
    with open(f'evaluation_datasets.json', 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=4)
    
#12 45 75 112 118 126 164 167 179 197 205 221 228