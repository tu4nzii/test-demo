import json
import os
from typing import List, Dict
from typing import Tuple
import base64
import requests
import re
import time
import cv2


class RoseChartDataProcessor:
    """玫瑰图数据处理类
    功能：加载特定的评估用JSON文件、进行数据验证和筛选、整合轴线标签信息、保存处理结果
    """
    
    def __init__(self):
        """初始化玫瑰图数据处理器"""
        self.chart_type = 'rose'
        self.target_type = 'rose'
        self.json_root = os.path.join('./', 'data', 'output')
    
    def process_evaluation_data(self, chart_id: str) -> List[Dict]:
        """处理并整合玫瑰图评估数据
        
        Args:
            chart_id: 图表ID
            
        Returns:
            List[Dict]: 处理后的数据集
        """
        datasets = []
        
        # 定义文件路径
        rose_json_path = os.path.join(self.json_root, self.chart_type, f'{chart_id}.json')
        axes_json_path = os.path.join(self.json_root, self.chart_type, f'{chart_id}_axes.json')
        output_path = os.path.join('./', 'data', 'output', 'result', self.chart_type, f'{chart_id}_evalution_datasets.json')
        
        # 检查文件是否存在
        if not os.path.exists(rose_json_path):
            print(f"❌ 文件不存在: {rose_json_path}")
            return datasets
        
        try:
            # 读取JSON文件
            rose_data = self._load_json_file(rose_json_path)
            
            # 整合轴线标签信息
            self._integrate_axis_labels(rose_data, axes_json_path)
            
            # 验证数据有效性
            if not self._validate_data(rose_data, chart_id):
                return datasets
            
            # 计算误差并验证
            err_center = rose_data['err_center']
            avg_err = self._calculate_error(rose_data)
            if avg_err > 10:
                print(f"r_pixel_err > 10, 跳过 {chart_id}", avg_err)
                return datasets
            
            # 添加元数据信息
            self._add_metadata(rose_data, chart_id)
            
            # 转换数据格式
            self._convert_data_format(rose_data)
            
            datasets.append(rose_data)
            
            # 输出统计信息
            self._print_statistics(rose_data, err_center, avg_err)
            
            # 保存处理结果
            self._save_result(rose_data, output_path)
            
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
    
    def _validate_data(self, rose_data: Dict, chart_id: str) -> bool:
        """验证数据有效性
        
        Args:
            rose_data: 玫瑰图数据
            chart_id: 图表ID
            
        Returns:
            bool: 数据是否有效
        """
        # 检查必要的字段
        if 'err_center' not in rose_data:
            print(f"❌ 文件缺少必要字段: err_center")
            return False
        
        err_center = rose_data['err_center']
        
        # 检查err_center，大于10则不添加到数据集
        if err_center > 10:
            print(f"err_center > 10, 跳过 {chart_id}")
            return False
            
        # 检查计算误差所需的字段
        if 'r_pixels' not in rose_data or 'r_ticks' not in rose_data or 'argument' not in rose_data:
            print(f"❌ 文件缺少必要的字段用于计算r_pixel_err")
            return False
            
        return True
    
    def _calculate_error(self, rose_data: Dict) -> float:
        """计算像素误差
        
        Args:
            rose_data: 玫瑰图数据
            
        Returns:
            float: 平均误差
        """
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
        
        return total_pred_r_pixel_err / len(rose_data['r_pixels'])
    
    def _add_metadata(self, rose_data: Dict, chart_id: str) -> None:
        """添加元数据信息
        
        Args:
            rose_data: 玫瑰图数据
            chart_id: 图表ID
        """
        rose_data['chart_id'] = chart_id
        rose_data['chart_type'] = self.chart_type
        rose_data['image_paths'] = {
            'no_grid': os.path.join('./','data', self.chart_type, f'{chart_id}.png'),
            'with_grid': os.path.join(self.json_root, self.chart_type, f'{chart_id}_encode.png')
        }
    
    def _convert_data_format(self, rose_data: Dict) -> None:
        """转换数据格式
        
        Args:
            rose_data: 玫瑰图数据
        """
        rose_data['data'] = {}
        if 'data_points' in rose_data:
            for label, value in rose_data['data_points'].items():
                rose_data['data'][label] = value
    
    def _print_statistics(self, rose_data: Dict, err_center: float, avg_err: float) -> None:
        """输出统计信息
        
        Args:
            rose_data: 玫瑰图数据
            err_center: 圆心误差
            avg_err: 平均像素误差
        """
        print('图表类型:', self.chart_type, '总数:', 1)
        print(f"圆心检测精准率 (err_center < 5): {'1.000000' if err_center < 5 else '0.000000'}")
        print(f"圆心检测RMSE: {err_center:.6f}")
        print(f"r_pixel_err MAE : {avg_err:6f}")
    
    def _save_result(self, rose_data: Dict, output_path: str) -> None:
        """保存处理结果
        
        Args:
            rose_data: 玫瑰图数据
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rose_data, f, ensure_ascii=False, indent=4)
        
        print(f"✅ 成功整合JSON文件并保存到: {output_path}")


# 保持向后兼容性，可以直接调用原函数

def process_rose_evaluation_data(chart_id: str) -> List[Dict]:
    """处理并整合玫瑰图评估数据
    功能：加载特定的评估用JSON文件、进行数据验证和筛选、整合轴线标签信息、保存处理结果
    """
    processor = RoseChartDataProcessor()
    return processor.process_evaluation_data(chart_id)


if __name__ == '__main__':
    # 处理并整合玫瑰图评估数据
    datasets = process_rose_evaluation_data()
    
    # 可选：如果需要仍然保存原始的evaluation_datasets.json文件
    # if os.path.exists('evaluation_datasets.json'):
    #     os.remove('evaluation_datasets.json')
    # with open(f'evaluation_datasets.json', 'w', encoding='utf-8') as f:
    #     json.dump(datasets, f, ensure_ascii=False, indent=4)
    
#12 45 75 112 118 126 164 167 179 197 205 221 228