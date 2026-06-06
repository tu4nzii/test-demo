"""Prompt construction for radar-chart prediction flows."""

from __future__ import annotations

from .geometry import hex_to_rgb

def generate_prompt(item_name: str, prompt_type: str, dataset: dict, tick=0) -> str:
    color = dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')
    rgb_colors = hex_to_rgb(color)
    chart_type = dataset.get('chart_type', '')
    
    if prompt_type == "with_grid":
        return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} angle positions


1.请先找到{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
2.然后找到该颜色对应的点，并插值出数值,rgb颜色为{rgb_colors}
3.Locate the "{item_name}" data point on the radar chart.
4.Estimate its radial position by interpolating between concentric circles.** Remember to always interpolate and make good use of the encrypted grid **

警告 Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
    elif prompt_type == "no_grid":
        return f'''
Your task is to estimate the value of the data point labeled "{item_name}":

警告 Respond ONLY in the exact JSON format:
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

Do not include any explanations or additional text.
'''.strip()
    elif prompt_type == "feedback":
        return f'''
You are analyzing a radar chart. It displays multivariate data on a 2D plane using axes that originate from a common point.

The chart contains virtual reference lines :

- Radial grid lines (concentric circles) represent data values, with corresponding tick values {dataset.get('r_ticks', [])}
- There are {len(dataset.get('series_color', {}))} entities: {', '.join(dataset.get('series_color', {}).keys())}, corresponding to colors {', '.join(dataset.get('series_color', {}).values())} respectively
- There are {len(dataset.get('theta_ticks', []))} positions, corresponding to {dataset.get('theta_ticks', [])}, distributed sequentially around the circle at {dataset.get('theta_angles', [])} positions

Your task is to estimate the value of the data point labeled "{item_name}":

**重要提示**：图表中已添加红色十字，在对应轴上，表示上一轮对"{item_name}"的预测值约为{tick}。
请比较红色十字与真实数据点的位置差距，重新优化您的预测：
{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
1. 确定红色十字与真实数据点之间的位置关系（偏内、偏外）
2. 根据这种关系，调整您的预测值
3. 确保新的预测值与真实点的位置对齐 以实现尽可能准确的预测
然后找到该颜色对应的点，并插值出数值,rgb颜色为{rgb_colors}
警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "amplifier":
        return f'''
您正在分析雷达图的一部分。该图在二维平面上使用从同一点出发的坐标轴展示多变量数据。

- 共有{len(dataset.get('series_color', {}))}个实体：{', '.join(dataset.get('series_color', {}).keys())}，分别对应颜色{', '.join(dataset.get('series_color', {}).values())}
现在的局部放大图为{item_name.split(',')[1].strip()}轴对应的局部放大
您的任务是估计标记为"{item_name}"对应的值，即{item_name.split(',')[0].strip()}对应实体颜色点的数值。
请先找到{item_name.split(',')[0].strip()}对应实体颜色为{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')}
若不存在该点，但存在该颜色的线，说明上一次预测的数值过小，请往大预测
若不存在该颜色，则说说明上一次预测值过大，超出范围，请往小预测
然后找到该{dataset.get('series_color', {}).get(item_name.split(',')[0].strip(), '未知颜色')},rgb颜色为{rgb_colors}对应的点，并插值出数值，记住，是一个圆点！

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    return ""
