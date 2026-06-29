# Polar Axis Prior and Grid Encryption Pipeline

本文档记录当前极坐标图表的复现流程。文件位置先看 `backend/docs/polar/polar_file_inventory.md`。

## 1. 输入数据

真实 radar / rose:

```text
backend/real/RadarChart-18 & RoseChart-6/RadarChart-18-final
backend/real/RadarChart-18 & RoseChart-6/RoseChart-6
```

合成 radar / rose / pie / donut:

```text
backend/real/radar
backend/real/rose
backend/real/pie
backend/real/donut
```

真实 pie / donut:

```text
backend/real/PieChart-11 & DonutChart-14
```

## 2. 稳定入口

单张图完整流程:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type radar --json "backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final\RadarChart24.json"
```

该入口按以下顺序执行:

```text
原图/JSON
-> 轴检测或圆检测
-> runtime fallback 判断
-> radar/rose 成功时生成加密图和评估 JSON
-> pie/donut 成功时生成圆检测评估 JSON 和检测预览
-> 输出 pipeline_summary.json
```

批量入口只用于论文表格统计:

轴检测与 fallback 评估:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type radar --dataset real --tick-mode gt-nearest
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type rose --dataset real_corrected --tick-mode gt-nearest
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type pie-donut --dataset all
```

真实 radar / rose 的 GT 控制加密数据:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_grid_encrypt.py --chart-type all --mode gt
```

真实 radar value evaluation:

```powershell
$env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\ADtry\python.exe backend\polar\value_eval\run_real_radar_value_evaluation.py
```

单图 smoke test:

```powershell
$env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\ADtry\python.exe backend\polar\value_eval\run_real_radar_value_evaluation.py --charts RadarChart24 --max-points 1
```

## 3. 输出位置

```text
backend/data/polar/output/radar_grid_eval
backend/data/polar/output/rose_grid_eval
backend/data/polar/output/pie_donut_circle_eval
backend/data/polar/output/axis_sample_selection
backend/data/polar/real_evaluation_data
backend/data/polar/manifests
```

## 4. Fallback 复现

生成 manifest:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\build_axis_manifest.py --csv backend\data\polar\output\radar_grid_eval\radar_grid_eval_real_gt-nearest.csv --chart-type radar --dataset real --policy-version polar_axis_v1 --output backend\data\polar\manifests\real_radar_axis_manifest.json

D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\build_axis_manifest.py --csv backend\data\polar\output\rose_grid_eval\rose_grid_eval_real_corrected_gt-nearest.csv --chart-type rose --dataset real_corrected --policy-version polar_axis_v1 --output backend\data\polar\manifests\real_rose_corrected_axis_manifest.json
```

检查 manifest:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\check_fallback_repro.py --manifest backend\data\polar\manifests\real_radar_axis_manifest.json --csv backend\data\polar\output\radar_grid_eval\radar_grid_eval_real_gt-nearest.csv

D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\check_fallback_repro.py --manifest backend\data\polar\manifests\real_rose_corrected_axis_manifest.json --csv backend\data\polar\output\rose_grid_eval\rose_grid_eval_real_corrected_gt-nearest.csv
```

期望:

```text
[fallback-repro] OK: 18 charts
[fallback-repro] OK: 3 charts
```

## 5. 指标口径

非 fallback 图表才计算:

```text
center_error_px
center_error_ratio
radius_error_max_px
radius_error_max_ratio
radius_tick_mapping_error_max_px
radius_tick_mapping_error_max_ratio
tolerance_pass
```

Fallback 图表只统计 fallback rate 和 fallback reason，不进入误差均值。

## 6. Radar Value Evaluation 逻辑

当前真实 radar value evaluation 保持旧版多轮逻辑:

```text
with_grid 整图输入模型
-> feedback_counts = 3
-> feedback = [初始 with_grid 值, 第1轮反馈, 第2轮反馈, 第3轮反馈]
-> amplifier_counts = 3
-> 依据 grid 初始值 amp 三次，输出 amplifier_grid_ticks
-> 依据 feedback 最后一轮值 amp 三次，输出 amplifier_feedback_ticks
```

局部放大输出:

```text
backend/data/polar/real_evaluation_data/radar/data/amplifier/radar/grid
backend/data/polar/real_evaluation_data/radar/data/amplifier/radar/feedback
```
