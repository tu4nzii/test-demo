# 极坐标参数敏感性诊断

本诊断只覆盖 Radar/Rose 的径向圆环检测器。它不评估当前直角系 enhanced-grid-first 主流程，也不代表 Pie/Donut 的端到端效果，因为 Pie/Donut 绘图区检测当前是颜色 mask 优先，Hough 只作为 fallback。

执行命令：

```bash
python backend/evaluation/scripts/evaluate_axis_prior_reviewer_questions.py --output review/axis_prior_reviewer_eval.json
```

输出文件：`review/axis_prior_reviewer_eval.json`。

## Radar/Rose HoughCircles `param2` 扫描

该诊断中的固定设置：Gaussian `(9,9), sigma=2`，`dp=1.2`，`minDist=100`，`param1=20`，radar 半径范围为 `height/5` 到 `height/4`，rose 半径范围为 `height/4` 到 `height`。

| HoughCircles `param2` | 样本数 | 圆检测返回率 | 中位圆心误差 | 中位第一候选半径误差 | 中位最佳半径误差 | 平均候选圆数量 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 99 | 100.00% | 2.236 px | 1.000 px | 1.000 px | 50.83 |
| **30** | **99** | **100.00%** | **2.236 px** | **2.000 px** | **1.000 px** | **26.87** |
| 40 | 99 | 97.98% | 2.236 px | 2.000 px | 2.000 px | 5.60 |
| 50 | 99 | 76.77% | 2.236 px | 2.000 px | 2.000 px | 2.21 |

## 解释边界

- `param2=30` 是 Radar/Rose 第一圈检测代码中的固定运行值。
- 在该离线诊断中，`param2=30` 与 `param2=20` 一样保持 100% 圆检测返回率，同时将平均候选圆数量从 `50.83` 降到 `26.87`。
- 更严格的阈值会继续减少候选数量，但也会降低圆检测返回率，尤其是 `param2=50`。
- GT 只用于本离线评分报告，不进入生成端流程。
