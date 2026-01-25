import json
import os
from typing import List, Dict
from typing import Tuple
import base64
import requests
import re
import time
import cv2


class RadarChartDataProcessor:
    """雷达图数据处理类
    功能：加载特定的评估用JSON文件、进行数据验证和筛选、整合轴线标签信息、保存处理结果
    """
    
    def __init__(self):
        """初始化雷达图数据处理器"""
        self.chart_type = 'radar'
        self.target_type = 'radar'
        self.json_root = os.path.join('./', 'data', 'output')
    
    def process_evaluation_data(self, chart_id: str) -> List[Dict]:
        """处理并整合雷达图评估数据
        
        Args:
            chart_id: 图表ID
            
        Returns:
            List[Dict]: 处理后的数据集
        """
        datasets = []
        
        # 定义文件路径
        radar_json_path = os.path.join(self.json_root, self.chart_type, f'{chart_id}.json')
        axes_json_path = os.path.join(self.json_root, self.chart_type, f'{chart_id}_axes.json')
        output_path = os.path.join('./', 'data', 'output', self.chart_type, 'result', f'{chart_id}_evalution_datasets.json')
        
        # 检查文件是否存在
        if not os.path.exists(radar_json_path):
            print(f"❌ 文件不存在: {radar_json_path}")
            return datasets
        
        try:
            # 读取JSON文件
            radar_data = self._load_json_file(radar_json_path)
            
            # 整合轴线标签信息
            self._integrate_axis_labels(radar_data, axes_json_path)
            
            # 验证数据有效性
            if not self._validate_data(radar_data, chart_id):
                return datasets
            
            # 计算误差并验证
            err_center = radar_data['err_center']
            avg_err = self._calculate_error(radar_data)
            if avg_err > 10:
                print(f"r_pixel_err > 10, 跳过 {chart_id}", avg_err)
                return datasets
            
            # 添加元数据信息
            self._add_metadata(radar_data, chart_id)
            
            # 转换数据格式
            self._convert_data_format(radar_data)
            
            datasets.append(radar_data)
            
            # 输出统计信息
            self._print_statistics(radar_data, err_center, avg_err)
            
            # 保存处理结果
            self._save_result(radar_data, output_path)
            
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            
        return datasets
    
    def _load_json_file(self, file_path: str) -> Dict:
        """加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            Dict: 加载的数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _integrate_axis_labels(self, rose_data: Dict, axes_json_path: str) -> None:
        """整合轴线标签信息
        
        Args:
            rose_data: 玫瑰图数据
            axes_json_path: 轴线数据文件路径
        """
        if os.path.exists(axes_json_path):
            axes_data = self._load_json_file(axes_json_path)
            if 'axis_labels' in axes_data:
                rose_data['axis_labels'] = axes_data['axis_labels']
                print(f"✅ 成功整合axis_labels数据")
        else:
            print(f"⚠️ 轴线数据文件不存在: {axes_json_path}")
    
    def _validate_data(self, radar_data: Dict, chart_id: str) -> bool:
        """验证数据有效性
        
        Args:
            radar_data: 雷达图数据
            chart_id: 图表ID
            
        Returns:
            bool: 数据是否有效
        """
        # 检查必要的字段
        if 'err_center' not in radar_data:
            print(f"❌ 文件缺少必要字段: err_center")
            return False
        
        err_center = radar_data['err_center']
        
        # 检查err_center，大于10则不添加到数据集
        if err_center > 10:
            print(f"err_center > 10, 跳过 {chart_id}")
            return False
            
        # 检查计算误差所需的字段
        if 'r_pixels' not in radar_data or 'r_ticks' not in radar_data or 'argument' not in radar_data:
            print(f"❌ 文件缺少必要的字段用于计算r_pixel_err")
            return False
            
        return True
    
    def _calculate_error(self, radar_data: Dict) -> float:
        """计算像素误差
        
        Args:
            radar_data: 雷达图数据
            
        Returns:
            float: 平均误差
        """
        total_pred_r_pixel_err = 0
        
        for i in range(len(radar_data['r_pixels'])):
            if(radar_data['r_ticks'][0] == 0):
                if(len(radar_data['r_ticks'])==11):
                    r_tick = radar_data['r_ticks'][i*2+1]
                else:
                    r_tick = radar_data['r_ticks'][i+1]
            else:
                if(len(radar_data['r_ticks'])==10):
                    r_tick = radar_data['r_ticks'][i*2+1]
                    print(r_tick)
                else:
                    r_tick = radar_data['r_ticks'][i]
            
            pred_r_pixel = r_tick*radar_data['argument']['a']+radar_data['argument']['b']
            pred_r_err = abs(pred_r_pixel - radar_data['r_pixels'][i])
            total_pred_r_pixel_err += pred_r_err
        
        return total_pred_r_pixel_err / len(radar_data['r_pixels'])
    
    def _add_metadata(self, radar_data: Dict, chart_id: str) -> None:
        """添加元数据信息
        
        Args:
            rose_data: 玫瑰图数据
            chart_id: 图表ID
        """
        radar_data['chart_id'] = chart_id
        radar_data['chart_type'] = self.chart_type
        radar_data['image_paths'] = {
            'no_grid': os.path.join('./','data','upload', f'{chart_id}.png'),
            'with_grid': os.path.join(self.json_root, self.chart_type, f'{chart_id}_encode.png')
        }
    
    def _convert_data_format(self, radar_data: Dict) -> None:
        """转换数据格式
        
        Args:
            radar_data: 雷达图数据
        """
        radar_data['data'] = {}
        if 'data_points' in radar_data:
            for label, value in radar_data['data_points'].items():
                radar_data['data'][label] = value
    
    def _print_statistics(self, radar_data: Dict, err_center: float, avg_err: float) -> None:
        """输出统计信息
        
        Args:
            radar_data: 雷达图数据
            err_center: 圆心误差
            avg_err: 平均像素误差
        """
        print('图表类型:', self.chart_type, '总数:', 1)
        print(f"圆心检测精准率 (err_center < 5): {'1.000000' if err_center < 5 else '0.000000'}")
        print(f"圆心检测RMSE: {err_center:.6f}")
        print(f"r_pixel_err MAE : {avg_err:6f}")
    
    def _save_result(self, radar_data: Dict, output_path: str) -> None:
        """保存处理结果
        
        Args:
            radar_data: 雷达图数据
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(radar_data, f, ensure_ascii=False, indent=4)
        
        print(f"✅ 成功整合JSON文件并保存到: {output_path}")


# 保持向后兼容性，可以直接调用原函数

def process_radar_evaluation_data(chart_id: str) -> List[Dict]:
    """处理并整合雷达图评估数据
    功能：加载特定的评估用JSON文件、进行数据验证和筛选、整合轴线标签信息、保存处理结果
    """
    processor = RadarChartDataProcessor()
    return processor.process_evaluation_data(chart_id)


if __name__ == '__main__':
    # 处理并整合雷达图评估数据
    datasets = process_radar_evaluation_data(chart_id='radar_001')
    
    # 可选：如果需要仍然保存原始的evaluation_datasets.json文件
    # if os.path.exists('evaluation_datasets.json'):
    #     os.remove('evaluation_datasets.json')
    # with open(f'evaluation_datasets.json', 'w', encoding='utf-8') as f:
    #     json.dump(datasets, f, ensure_ascii=False, indent=4)
    
#12 45 75 112 118 126 164 167 179 197 205 221 228