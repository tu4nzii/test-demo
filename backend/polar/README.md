# Polar Chart Pipeline

这里现在只保留两个主入口：

```text
backend/polar/scripts/run_one_chart.py
backend/polar/radar/demo_evaluation_radar_1 copy.py
```

## 1. 单张图完整流程

后面如果要检查一张图，从这里跑：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type radar --json "backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final\RadarChart24.json"
```

它的流程是：

```text
原图/JSON
-> 轴检测或圆检测
-> fallback 判断
-> radar/rose: 生成加密图和对应评估 JSON
-> pie/donut: 生成圆心/半径检测结果
-> pipeline_summary.json
```

pie/donut 当前没有网格加密阶段，只评估圆心和半径先验。

## 2. 后续值评估

你说的“评估”默认指这个文件：

```text
backend/polar/radar/demo_evaluation_radar_1 copy.py
```

如果想用包装器批量跑已经准备好的真实 radar JSON，可以用：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\value_eval\run_real_radar_value_evaluation.py --dry-run
```

包装器只是帮你切工作目录和批量传 JSON，实际 evaluator 仍然是上面的 legacy 文件。

## 3. 主目录说明

```text
backend/polar/
  scripts/       # 只放单张图主流程
  evaluation/    # run_one_chart 依赖的轴/圆检测评估模块
  encryption/    # run_one_chart 依赖的 GT 网格加密模块
  value_eval/    # legacy evaluator 的轻量包装器
  radar/         # radar 底层检测/加密实现
  rose/          # rose 底层检测/加密实现
  pie/           # pie 圆检测实现
  donut/         # donut 圆检测实现
  legacy/        # 其他旧版 demo 追溯
  archive_unused/# 旧批处理、历史 demo、论文统计脚本归档
```

## 4. 旧脚本归档

旧的批量轴评估、fallback 复现、50 张抽样、旧 demo 等已经放到：

```text
backend/polar/archive_unused/20260629_cleanup
```

这些文件没有删除，只是不再作为当前主流程入口。
