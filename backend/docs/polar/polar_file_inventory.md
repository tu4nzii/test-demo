# 极坐标目录索引

极坐标相关内容现在集中在三个位置:

```text
backend/polar        # 代码
backend/docs/polar   # 文档
backend/data/polar   # 结果、manifest、归档数据
```

## 1. 代码层级

```text
backend/polar/
  scripts/      # 推荐命令入口
  evaluation/   # 轴检测、fallback、几何误差评估
  encryption/   # GT 控制加密数据准备
  value_eval/   # radar/rose 后续数值评估入口
  radar/        # radar 底层检测、加密、value evaluator
  rose/         # rose 底层检测、加密
  pie/          # pie 圆心/外半径检测
  donut/        # donut 圆心/内外半径检测
  legacy/       # 旧脚本归档，只保留不用作主入口
```

## 2. 推荐入口

以后优先跑 `backend/polar/scripts`，不要直接从旧 demo 文件跑。

| 任务                                 | 入口                                                            |
| ------------------------------------ | --------------------------------------------------------------- |
| 轴检测、fallback、几何误差评估       | `backend/polar/scripts/run_axis_eval.py`                      |
| 真实 radar / rose 的 GT 加密数据准备 | `backend/polar/scripts/run_grid_encrypt.py`                   |
| 从 CSV 生成 fallback manifest        | `backend/polar/scripts/build_axis_manifest.py`                |
| 检查 fallback manifest 是否复现      | `backend/polar/scripts/check_fallback_repro.py`               |
| 真实 radar 数值评估                  | `backend/polar/value_eval/run_real_radar_value_evaluation.py` |
| 合成图表 50 张抽样                   | `backend/polar/scripts/select_axis_eval_samples.py`           |

常用命令:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type radar --dataset real --tick-mode gt-nearest
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type rose --dataset real_corrected --tick-mode gt-nearest
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_axis_eval.py --chart-type pie-donut --dataset all
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_grid_encrypt.py --chart-type all --mode gt
D:\anaconda3\envs\ADtry\python.exe backend\polar\value_eval\run_real_radar_value_evaluation.py --dry-run
```

## 3. 主流程文件

| 类型                        | 文件                                                                  |
| --------------------------- | --------------------------------------------------------------------- |
| radar 轴/网格评估           | `backend/polar/evaluation/evaluate_radar_grid_extraction.py`        |
| rose 轴/网格评估            | `backend/polar/evaluation/evaluate_rose_grid_extraction.py`         |
| pie / donut 圆检测评估      | `backend/polar/evaluation/evaluate_pie_donut_circle_extraction.py`  |
| 真实 GT 加密数据生成        | `backend/polar/encryption/prepare_real_evaluation_gt_encryption.py` |
| 真实 radar value evaluation | `backend/polar/value_eval/run_real_radar_value_evaluation.py`       |
| radar 底层实现              | `backend/polar/radar`                                               |
| rose 底层实现               | `backend/polar/rose`                                                |
| pie / donut 底层实现        | `backend/polar/pie`, `backend/polar/donut`                        |

## 4. 数据层级

```text
backend/data/polar/
  output/                # 主评估输出
  manifests/             # fallback 复现清单
  real_evaluation_data/  # 真实图表 GT 加密后的 value-eval 输入
  archive/               # 阶段性旧输出
```

主结果:

```text
backend/data/polar/output/radar_grid_eval
backend/data/polar/output/rose_grid_eval
backend/data/polar/output/pie_donut_circle_eval
backend/data/polar/output/axis_sample_selection
```

fallback manifest:

```text
backend/data/polar/manifests/real_radar_axis_manifest.json
backend/data/polar/manifests/real_rose_corrected_axis_manifest.json
```

## 5. 文档层级

```text
backend/docs/polar/polar_file_inventory.md
backend/docs/polar/polar_axis_grid_pipeline.md
backend/docs/polar/polar_fallback_policy.md
backend/docs/polar/archive
```

`archive` 只做追溯，不作为当前 README 主入口。

## 6. 真实数据入口

真实图表原始数据仍保留在:

```text
backend/real
```

这些是输入数据，不归入 `backend/data/polar`，避免和实验输出混在一起。

## 7. RadarChart19

`radarchart_19` 当前应当进入 fallback:

```text
fallback_reason = center_not_at_origin_zero_tick_line_not_through_center
```

它不参与成功图表误差均值。