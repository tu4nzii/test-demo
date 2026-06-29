# Polar Pipeline

极坐标相关代码统一放在这里，包含 radar、rose、pie、donut 的检测、fallback、加密和评估。

## 目录

```text
backend/polar/
  scripts/      # 推荐命令入口
  evaluation/   # 轴检测、fallback、几何误差评估
  encryption/   # 真实图表 GT 加密数据准备
  value_eval/   # 后续数值评估入口
  radar/        # radar 底层实现
  rose/         # rose 底层实现
  pie/          # pie 底层实现
  donut/        # donut 底层实现
  legacy/       # 旧脚本归档
```

## 常用命令

单张图从头到尾:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type radar --json "backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final\RadarChart24.json"
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type rose --dataset real_corrected --json "backend\real\RadarChart-18 & RoseChart-6\RoseChart-6\Rose1_gt_encrypt.json"
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type pie --json "backend\real\pie\pie_001.json"
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type donut --json "backend\real\donut\donut_001.json"
```

批量统计:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type radar --dataset real --tick-mode gt-nearest
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type rose --dataset real_corrected --tick-mode gt-nearest
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type pie-donut --dataset all
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_grid_encrypt.py --chart-type all --mode gt
D:\anaconda3\envs\ADtry\python.exe backend\polar\value_eval\run_real_radar_value_evaluation.py --dry-run
```

## 数据和文档

```text
backend/data/polar   # 输出、manifest、归档
backend/docs/polar   # 流程说明和 fallback 说明
backend/real         # 真实/合成原始图表输入
```

先看:

```text
backend/docs/polar/polar_file_inventory.md
backend/docs/polar/polar_axis_grid_pipeline.md
backend/docs/polar/polar_fallback_policy.md
```
