# 极坐标代码目录索引

当前极坐标相关内容分三层：

```text
backend/polar        # 代码
backend/docs/polar   # 文档
backend/data/polar   # 输出、manifest、归档数据
```

## 1. 现在主要看这两个入口

| 任务 | 入口 |
| --- | --- |
| 单张图完整流程：原图/JSON -> fallback -> 轴/圆检测 -> 加密或圆检测结果 | `backend/polar/scripts/run_one_chart.py` |
| 后续值评估，也就是你之后说“评估”时默认指的脚本 | `backend/polar/legacy/demo_radar/demo_evaluation_radar_1.py` |

`backend/polar/value_eval/run_real_radar_value_evaluation.py` 只是一个轻量包装器，用来批量调用上面的 legacy evaluator。

## 2. 单张图流程

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type radar --json "backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final\RadarChart24.json"
```

输出默认在：

```text
backend/data/polar/single_runs/<chart_type>/<json_name>
```

主要输出：

```text
01_axis_detection/axis_eval.json       # radar/rose
01_circle_detection/circle_eval.json   # pie/donut
02_encrypted/*.png                     # radar/rose 成功且未 fallback 时
02_encrypted/*.json                    # 后续值评估可用的 JSON
pipeline_summary.json                  # 一张图的总结果
```

## 3. 当前代码层级

```text
backend/polar/
  scripts/       # 只保留 run_one_chart.py
  evaluation/    # run_one_chart 依赖的轴/圆检测评估模块
  encryption/    # run_one_chart 依赖的 GT 网格加密模块
  value_eval/    # 调用 legacy evaluator 的包装器
  radar/         # radar 底层检测/加密实现
  rose/          # rose 底层检测/加密实现
  pie/           # pie 圆检测实现
  donut/         # donut 圆检测实现
  legacy/        # 当前保留 demo_radar/demo_evaluation_radar_1.py
  archive_unused/# 旧脚本归档
```

## 4. 已归档的旧脚本

旧批处理、论文统计、fallback manifest 复现、合成图抽样、重复 demo 等已经移到：

```text
backend/polar/archive_unused/20260629_cleanup
```

这些文件没有删除，只是不再作为当前主入口。

## 5. 数据位置

输入数据仍在：

```text
backend/real
```

极坐标输出在：

```text
backend/data/polar
```

真实图表后续值评估输入一般在：

```text
backend/data/polar/real_evaluation_data
```

## 6. 特别说明

`RadarChart19` 当前应进入 fallback：

```text
fallback_reason = center_not_at_origin_zero_tick_line_not_through_center
```

它不参与成功图表的误差均值统计。
