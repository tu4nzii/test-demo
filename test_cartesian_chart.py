#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试直角坐标系图表处理功能
"""

import os
import sys
import json
from backend.function_call.chart_type import detect_chart

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接测试直角坐标系图表处理器，避免依赖 demo_rose
from backend.Grid_generation.grid_generation import process_chart

def test_cartesian_chart_processing():
    """测试直角坐标系图表处理"""
    print("开始测试直角坐标系图表处理功能...")
    
    # 测试数据目录
    test_data_dir = os.path.join('backend', 'VisHintPrompt_datasets')
    
    # 测试的图表类型
    test_types = ['v_bar', 'h_bar', 'line', 'scatter', 'bubble', 'donut', 'pie']
    
    for chart_type in test_types:
        chart_dir = os.path.join(test_data_dir, chart_type)
        if not os.path.exists(chart_dir):
            print(f"跳过 {chart_type} - 目录不存在")
            continue
        
        # 获取测试文件
        test_files = [f for f in os.listdir(chart_dir) if f.endswith('.png') and '_grid' not in f]
        if not test_files:
            print(f"跳过 {chart_type} - 无测试文件")
            continue
        
        # 选择第一个文件进行测试
        test_file = os.path.join(chart_dir, test_files[0])
        print(f"\n测试 {chart_type} 图表: {test_file}")
        
        try:
            # 1. 检测图表类型
            chart_type_result = detect_chart(test_file)
            print(f"图表类型检测结果: {chart_type_result}")
            
            # 2. 测试 process_chart 函数
            output_dir = os.path.join('test_output', chart_type)
            os.makedirs(output_dir, exist_ok=True)
            
            result = process_chart(test_file, output_dir)
            print(f"process_chart 结果: {'成功' if result else '失败'}")
            
            if result:
                print(f"处理后的数据包含 {len(result.get('x_ticks', []))} 个X轴刻度")
                print(f"处理后的数据包含 {len(result.get('y_ticks', []))} 个Y轴刻度")
                print(f"基础网格图像: {result.get('basic_grid_path', 'N/A')}")
                print(f"加密网格图像: {result.get('encrypted_grid_path', 'N/A')}")
            
            print(f"{chart_type} 测试完成")
            
        except Exception as e:
            print(f"测试 {chart_type} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_cartesian_chart_processing()
