# 极坐标单张图流程

这份文档只记录当前保留的主流程。旧批处理和论文统计脚本已经归档到：

```text
backend/polar/archive_unused/20260629_cleanup
```

## 1. 主入口

```text
backend/polar/scripts/run_one_chart.py
```

示例：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type radar --json "backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final\RadarChart24.json"
```

## 2. 流程顺序

```text
原图/JSON
-> 轴检测或圆检测
-> runtime fallback 判断
-> radar/rose 未 fallback: 生成加密图和对应评估 JSON
-> pie/donut 未 fallback: 生成圆心/半径检测结果
-> pipeline_summary.json
```

pie/donut 当前只做坐标先验评估，没有网格加密阶段。

## 3. 输出结构

默认输出：

```text
backend/data/polar/single_runs/<chart_type>/<json_name>
```

radar/rose：

```text
01_axis_detection/axis_eval.json
02_encrypted/<chart>_gt_encrypt.png
02_encrypted/<chart>.json
pipeline_summary.json
```

pie/donut：

```text
01_circle_detection/circle_eval.json
01_circle_detection/<chart_type>/<dataset>/detections
pipeline_summary.json
```

## 4. 后续值评估

之后你说“评估”时，默认指这个脚本：

```text
backend/polar/radar/demo_evaluation_radar_1 copy.py
```

包装器入口：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\value_eval\run_real_radar_value_evaluation.py --dry-run
```

包装器只负责批量调用和切换工作目录，实际评估逻辑仍在 legacy evaluator 中。
