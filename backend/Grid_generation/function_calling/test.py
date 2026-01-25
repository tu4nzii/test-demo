import matplotlib.pyplot as plt
import numpy as np

# 数据准备（保留小数点后两位）
metrics = ['Precision', 'Recall', 'MSE', 'Max Error', 'Abs Error']
x_axis = [0.98, 0.98, 0.10, 0.55, 0.10]  # X轴刻度评估结果
y_axis = [0.95, 0.93, 0.03, 0.10, 0.03]  # Y轴刻度评估结果

# 设置图形
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(metrics))

# 绘制柱形图
bar1 = ax.bar(index, x_axis, bar_width, label='X-axis ticks', color='#1f77b4')
bar2 = ax.bar(index + bar_width, y_axis, bar_width, label='Y-axis ticks', color='#ff7f0e')

# 添加标签和标题
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores/Errors')
ax.set_title('Evaluation Metrics Comparison (499 images evaluated)')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend()

# 在柱子上方显示数值（保留两位小数）
for rect in bar1 + bar2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., height,
            f'{height:.2f}',
            ha='center', va='bottom', fontsize=9)

# 调整布局
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图形
plt.show()

# 打印数值结果
print("""
[Summary]

Total images evaluated: 499

X-axis ticks:
Avg Precision: 0.98
Avg Recall: 0.98
Avg MSE: 0.10
Avg Max Error: 0.55
Avg Abs Error: 0.10

Y-axis ticks:
Avg Precision: 0.95
Avg Recall: 0.93
Avg MSE: 0.03
Avg Max Error: 0.10
Avg Abs Error: 0.03
""")