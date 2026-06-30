# 当前直角系全链路证据

本报告用于确认审稿回复中的直角系证据来自当前完整流程，而不是旧版坐标轴/tick 扫描路径。

## 流程证据

- 最新 recheck 目录：`F:\program\test-demo\backend\evaluation\recheck_outputs\vishintprompt_full_grid_encryption_latest`
- priority decision 文件数：650
- x/y 两个方向都包含三类来源（`combined_mask`、`tick_supplement`、`semantic_guide`）的 decision 数：650
- 候选网格 binding 文件数：1809
- final binding 文件数：1176
- final selection 文件数：650
- 网格状态报告文件数：650
- 实际 failure/exit 报告数：14
- score prefill 后使用 MLLM 仲裁的次数：12

观察到的 selection policy：

- `score_first_mllm_when_needed`：650

轴方向来源选择统计：

| 来源 | X 轴方向选择次数 | Y 轴方向选择次数 |
| --- | ---: | ---: |
| `combined_mask` | 44 | 90 |
| `semantic_guide` | 232 | 204 |
| `tick_supplement` | 374 | 356 |

## 直角系完整流程指标

| 数据集 | 类型 | 样本数 | 已处理 | Tick MAE(px) | Tick Acc@2px | Label Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Final-RealDataset | bubble | 9 | 9 | 0.482 | 99.12% | 96.49% |
| Final-RealDataset | h_bar | 12 | 11 | 1.868 | 80.88% | 75.25% |
| Final-RealDataset | line | 23 | 21 | 0.960 | 92.03% | 97.23% |
| Final-RealDataset | scatter | 9 | 7 | 0.814 | 98.04% | 100.00% |
| Final-RealDataset | v_bar | 21 | 19 | 0.806 | 94.19% | 95.86% |
| Sy.Dataset | bubble | 50 | 50 | 0.874 | 95.40% | 99.40% |
| Sy.Dataset | h_bar | 50 | 49 | 0.771 | 98.09% | 93.46% |
| Sy.Dataset | line | 50 | 50 | 0.564 | 96.76% | 98.01% |
| Sy.Dataset | scatter | 50 | 50 | 0.442 | 98.60% | 100.00% |
| Sy.Dataset | v_bar | 51 | 51 | 0.462 | 98.78% | 97.20% |

直角系完整流程总体结果：

- 样本数：325
- 已处理：317
- Tick MAE：0.691 px
- Tick Acc@2px：96.37%
- Tick position MAE：0.849 px
- Label accuracy：96.13%

## 解释

`review/` 中的参数敏感性扫描只是一项旧版低层 Canny/Hough 候选生成器诊断。上面的最终直角系结果来自当前 active enhanced-grid-first 运行时 artifacts：三套候选网格经过 score 筛选，不可靠样本产生 failure/exit 报告，评估读取生成端输出的 final bindings。
