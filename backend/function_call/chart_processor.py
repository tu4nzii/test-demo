import os
import json
import asyncio
from typing import Dict, Protocol

# 从现有的玫瑰图处理模块导入必要的类
from demo_rose.demo_rose_circle_find import RoseChartEncoder
from demo_rose.demo_axis_find_rose import RoseChartAxisFinder
from demo_rose.demo_evaluation_rose import RoseChartEvaluator

# 从直角坐标系处理模块导入必要的函数
from Grid_generation.grid_generation import process_chart

# 图表处理器接口定义
class ChartProcessor(Protocol):
    """图表处理器的抽象接口"""
    def encode_image(self, image_path: str, output_dir: str) -> str:
        """加密图表图像"""
        ...
    
    def find_axis(self, image_path: str) -> dict:
        """查找图表坐标轴"""
        ...
    
    def process_data(self, chart_id: str, image_path: str, json_path: str, output_dir: str) -> dict:
        """处理图表数据"""
        ...
    
    def evaluate(self, eval_data_path: str) -> dict:
        """评估图表处理结果"""
        ...
    
    def save_evaluation_results(self, results: dict, output_path: str) -> None:
        """保存评估结果"""
        ...

# 玫瑰图处理器实现
class RoseChartProcessor:
    """玫瑰图处理器的具体实现"""
    def encode_image(self, image_path: str, output_dir: str) -> str:
        """使用demo_rose_circle_find处理图片，获取加密后的图片"""
        encoder = RoseChartEncoder()
        return encoder.process_single_image(image_path, output_dir)
    
    def find_axis(self, image_path: str) -> dict:
        """使用demo_axis_find_rose找到坐标轴"""
        axis_finder = RoseChartAxisFinder()
        return axis_finder.process_single_image(image_path)
    
    def process_data(self, chart_id: str, image_path: str, json_path: str, output_dir: str) -> dict:
        """使用demo_data_solve_rose处理数据"""
        # 准备文件路径
        target_json_path = os.path.join(output_dir, f"{chart_id}.json")
        
        if not os.path.exists(target_json_path):
            return None
        
        # 读取现有的结果文件
        with open(target_json_path, 'r', encoding='utf-8') as f:
            rose_data = json.load(f)
        
        # 添加必要的字段
        rose_data['chart_id'] = chart_id
        rose_data['chart_type'] = 'rose'
        
        # 读取原始JSON数据
        with open(json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # 转换labels和values为data键值对
        rose_data['data'] = {}
        if 'data_points' in original_data:
            for label, value in original_data['data_points'].items():
                rose_data['data'][label] = value
        
        return rose_data
    
    def evaluate(self, eval_data_path: str) -> dict:
        """使用demo_evaluation_rose进行评估"""
        evaluator = RoseChartEvaluator()
        evaluator.process_single_image(eval_data_path)
        # 尝试获取结果，如果evaluator没有直接暴露results属性
        # 我们可以先保存到临时文件，然后读取
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w', encoding='utf-8') as temp_file:
            temp_path = temp_file.name
        
        try:
            evaluator.save_results(temp_path)
            # 检查文件是否存在且不为空
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                # 如果没有结果，返回空结果结构
                return {"error": "评估未产生结果，可能是数据格式不符合要求"}
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return {"error": "评估结果为空"}
                results = json.loads(content)
            return results
        except json.JSONDecodeError as e:
            # 如果 JSON 解析失败，返回错误信息
            return {"error": f"JSON解析失败: {str(e)}"}
        except Exception as e:
            return {"error": f"评估过程出错: {str(e)}"}
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def save_evaluation_results(self, results: dict, output_path: str) -> None:
        """保存评估结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

# 直角坐标系图表处理器实现
class CartesianChartProcessor:
    """直角坐标系图表处理器的具体实现"""
    def encode_image(self, image_path: str, output_dir: str) -> str:
        """处理图片，获取加密后的图片"""
        # 调用grid_generation模块处理图像
        result = process_chart(image_path, output_dir)
        if result and 'encrypted_grid_path' in result:
            return result['encrypted_grid_path']
        return None
    
    def find_axis(self, image_path: str) -> dict:
        """查找图表坐标轴"""
        # 创建临时输出目录
        temp_output = os.path.join(os.path.dirname(image_path), 'temp_output')
        os.makedirs(temp_output, exist_ok=True)
        
        try:
            # 调用grid_generation模块处理图像
            result = process_chart(image_path, temp_output)
            if result:
                # 提取坐标轴信息
                axis_info = {
                    'x_ticks': result.get('x_ticks', []),
                    'y_ticks': result.get('y_ticks', []),
                    'x_axis_type': '数值轴',
                    'y_axis_type': '数值轴'
                }
                return axis_info
        finally:
            # 清理临时目录
            import shutil
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
        
        return {}
    
    def process_data(self, chart_id: str, image_path: str, json_path: str, output_dir: str) -> dict:
        """处理图表数据"""
        # 调用grid_generation模块处理图像
        result = process_chart(image_path, output_dir)
        
        if result:
            # 构建返回数据结构
            chart_data = {
                'chart_id': chart_id,
                'chart_type': 'cartesian',
                'x_ticks': result.get('x_ticks', []),
                'y_ticks': result.get('y_ticks', []),
                'x_axis_type': '数值轴',
                'y_axis_type': '数值轴',
                'colors': [],
                'image_path': result.get('image_path', ''),
                'basic_grid_path': result.get('basic_grid_path', ''),
                'encrypted_grid_path': result.get('encrypted_grid_path', '')
            }
            
            # 尝试读取原始JSON数据并添加data字段
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)
                    if 'data_points' in original_data:
                        chart_data['data'] = original_data['data_points']
                except:
                    pass
            
            return chart_data
        
        return None
    
    def evaluate(self, eval_data_path: str) -> dict:
        """评估图表处理结果"""
        # 创建临时输出目录
        temp_output = os.path.join(os.path.dirname(eval_data_path), 'temp_eval')
        os.makedirs(temp_output, exist_ok=True)
        
        try:
            # 调用grid_generation模块处理图像
            result = process_chart(eval_data_path, temp_output)
            if result:
                # 构建评估结果
                evaluation_result = {
                    'chart_id': result.get('chart_id', ''),
                    'x_ticks_count': len(result.get('x_ticks', [])),
                    'y_ticks_count': len(result.get('y_ticks', [])),
                    'colors_count': 0,
                    'success': True
                }
                return evaluation_result
        finally:
            # 清理临时目录
            import shutil
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
        
        return {"error": "评估失败"}
    
    def save_evaluation_results(self, results: dict, output_path: str) -> None:
        """保存评估结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

class ChartProcessorFactory:
    """根据图表类型创建相应的处理器"""
    @staticmethod
    def create_processor(chart_type: str) -> ChartProcessor:
        """创建并返回指定类型的图表处理器"""
        if chart_type == 'rose':
            return RoseChartProcessor()
        elif chart_type in ['v_bar', 'h_bar', 'line', 'scatter', 'bubble', 'donut', 'pie']:
            return CartesianChartProcessor()
        # 可以在这里添加更多图表类型的处理器
        else:
            # 默认使用直角坐标系处理器作为回退
            return CartesianChartProcessor()

# 支持的图表类型
SUPPORTED_CHART_TYPES = ['rose', 'v_bar', 'h_bar', 'line', 'scatter', 'bubble', 'donut', 'pie']