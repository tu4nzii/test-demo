"""Prompt construction for rose-chart prediction flows."""

from __future__ import annotations

def generate_prompt(item_name: str, prompt_type: str, dataset: dict, tick=0) -> str:
    chart_type = dataset.get('chart_type', '')
    if prompt_type == "with_grid":
        if chart_type == 'rose':
            return f'''
图表包含**虚拟参考线**：
您正在分析一张玫瑰图。它通过扇形的**最远端半径**来展示数据，每个扇形代表一个类别，其长度表示数据值的大小。
以下为图表的详细信息：
    - 存在以下径向网格线（同心圆），对应的刻度值为{dataset.get('r_ticks', [])},标注在对应网格虚线上
    - 存在以下角度网格线，将圆分成多个扇形区域，{dataset.get('theta_ticks', [])}分别对应每个扇形区域，扇形区域的分界为{dataset.get('theta_angles', [])}（单位为度）

你的任务是估计对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形的径向最远边界值为多少
以下为提示：
    1. **在玫瑰图上找到"{item_name}"对应的扇形区域，即确定其角度范围。**,非常重要，以x轴正方向为0度，可以理解为{dataset.get('theta_ticks')[0]}对应范围为0-{dataset.get('theta_angles')[1]}度，{dataset.get('theta_ticks')[1]}对应范围为{dataset.get('theta_angles')[1]}-{dataset.get('theta_angles')[2]}度，以此类推
    例子：该图的{dataset.get('theta_ticks')[0]}对应范围为0-{dataset.get('theta_angles')[1]}度，处在图最右端
    2. 确定其径向位置，找到其处于哪两个网格线之间，网格线包含以下刻度{dataset.get('r_ticks', [])}，必须准确的识别其位于哪两个网格线之间
    例子：该图的"{dataset.get('theta_ticks')[0]}"的值就为{dataset['data_points'][dataset.get('theta_ticks')[0]]}
    3. 根据其扇形和相对于两个网格线的位置，插值计算其数据值。

**记住，一定要插值，利用好网格线的刻度值**
在预测之前，再次回顾以下我给你的提示
一定要给我一个值，不能给我多个值，也不能给我没有值的情况

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "no_grid":
        if chart_type == 'rose':
            return f'''
您的任务是估计标记为"{item_name}"对应的值：

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "feedback":
        if chart_type == 'rose':
            return f'''
图表包含**虚拟参考线**：
您正在分析一张玫瑰图。它通过扇形的**最远端半径**来展示数据，每个扇形代表一个类别，其长度表示数据值的大小。
以下为图表的详细信息：
    - 存在以下径向网格线（同心圆），对应的刻度值为{dataset.get('r_ticks', [])},标注在对应网格虚线上
    - 存在以下角度网格线，将圆分成多个扇形区域，{dataset.get('theta_ticks', [])}分别对应每个扇形区域，扇形区域的分界为{dataset.get('theta_angles', [])}（单位为度）
你的任务是估计对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形的径向最远边界值为多少

**重要提示**：图表中已添加红色十字，表示上一轮对"{item_name}"的预测值约为{tick}。
请比较红色十字与真实数据点的位置差距，重新优化您的预测：
1. 确定红色十字与真实数据点之间的位置关系（偏内、偏外）
2. 根据这种关系，调整您的预测值
3. 确保新的预测值与真实点的位置对齐 以实现尽可能准确的预测
一定要给我一个值，不能给我多个值，也不能给我没有值的情况

**记住，一定要利用网格线进行精确插值**

警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    elif prompt_type == "amplifier":
        if chart_type == 'rose':
            return f'''
你的任务是估计对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形的径向最远边界值为多少
-请先找到{item_name}对应颜色为{dataset.get('series_color', {}).get(item_name, '未知颜色')}的扇形，**并且找到其最远端**
-然后找到该边界处于哪两个基准线之间
-最后依据基准线的数值，插值出数值
一定要是颜色的最远端！！
警告 仅以以下确切的JSON格式响应：
{{"datapoints": [{{"{item_name}": [r_value, null]}}]}}

不要包含任何解释或额外文本。
'''.strip()
    return ""
