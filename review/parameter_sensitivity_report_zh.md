# 参数敏感性实验

本目录包含用于回复审稿意见的离线可复现实验。GT JSON 文件只用于该实验的离线打分，不用于系统生成端流程。

范围说明：该实验隔离考察旧版低层 Canny/Hough 候选生成器，不是当前直角系系统实际运行路径，也不是当前直角系网格重建主流程的端到端重跑。当前运行时流程使用 enhanced-grid-first mask reconstruction，生成三套网格候选（`combined_mask`、`tick_supplement` 和 `semantic_guide`），进行 score 筛选和退出检查，并写出 `final_bindings`；下方完整流程指标来自该最新主流程报告。

## 数据集

- 数据集根目录：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets`
- 评估的直角坐标系样本数：324
- 类型：bubble, h_bar, line, scatter, v_bar

## 当前直角坐标系实际运行参数

当前系统路径调用 `_process_chart_with_enhanced_grid_only` 和 `grid_line_filter.process_image`。其固定参数如下：

| 参数 | 取值 |
| --- | --- |
| 中性网格 mask | 饱和度 <= 70；灰度范围 [95, 255]；局部对比度 >= 7 |
| 可选 dark candidates | 默认关闭；开启时 dark cutoff 为 80 |
| 形态学线段长度 | min_line_frac 0.055，按图像宽/高计算；下限 15 px |
| 间隙闭合 | gap_frac 0.006，按图像宽/高计算；下限 3 px |
| 连通域厚度过滤 | max_thickness_frac 0.008，按图像短边计算；下限 3 px |
| 网格几何重建 | min_grid_span_frac 0.18；min_grid_lines 2；cluster_tolerance 3 px；grid_thickness 1 px |
| 基于 dark 轴线/tick 的补充候选 | tick_dark_cutoff 150 |
| OCR 过滤 | ocr_min_score 0.45；det_thresh 0.35；det_box_thresh 0.60；det_unclip_ratio 1.15；det_limit_side_len 960 |

## 旧版 Canny/Hough 诊断参数

| 参数 | 取值 |
| --- | --- |
| Canny 前的 Gaussian smoothing | 旧版线段检测器不使用全图 Gaussian smoothing；`(3,3), sigma=0` 只用于局部 OCR 文本裁剪阈值化 |
| 参数扫描中的 Gaussian 设置 | none；`(3,3), sigma=0`；`(5,5), sigma=1` |
| Canny 阈值 | 30 / 100 |
| Probabilistic Hough threshold | 15 |
| Hough 最小线段长度 / 最大间隙 | 20 px / 20 px |
| 旧版 tick 扫描范围 | 20 px |
| 旧版 tick 合并角度容差 | 10 degrees |

## 主要结果

这里展示的是适合写入审稿回复的候选数量稳定性。候选是否接近 GT 坐标轴位置等内部诊断列仅保留在 `parameter_sensitivity_samples.csv` 和 `parameter_sensitivity_summary.csv` 中，不作为审稿正文中的效果指标。

| 参数设置 | 平均 Hough 线段候选数 | 相对基线变化 | 解释 |
| --- | ---: | ---: | --- |
| none__canny_20_80__hough15_l20_g20 | 181.003 | -0.04% | 候选数量与旧版诊断基线接近。 |
| none__canny_30_100__hough15_l20_g20 | 181.083 | 0.00% | 旧版 Canny/Hough 诊断基线。 |
| none__canny_50_150__hough15_l20_g20 | 181.590 | +0.28% | 候选数量与旧版诊断基线接近。 |
| none__canny_70_210__hough15_l20_g20 | 178.870 | -1.22% | 候选数量少于旧版诊断基线。 |
| g3_s0__canny_30_100__hough15_l20_g20 | 189.349 | +4.56% | 候选数量多于旧版 no-blur 基线。 |
| g5_s1__canny_30_100__hough15_l20_g20 | 188.071 | +3.86% | 候选数量多于旧版 no-blur 基线。 |
| none__canny_30_100__hough10_l20_g20 | 203.639 | +12.45% | 旧版检测器中的候选数量显著增加。 |
| none__canny_30_100__hough20_l20_g20 | 159.534 | -11.90% | 旧版检测器中的候选数量显著减少。 |

## 基线参数下的类别结果

| 类型 | 样本数 | 平均 Hough 线段候选数 |
| --- | ---: | ---: |
| bubble | 59 | 215.542 |
| h_bar | 62 | 169.581 |
| line | 73 | 181.918 |
| scatter | 59 | 234.407 |
| v_bar | 71 | 117.324 |

## 当前完整流程参考指标

以下指标作为直角系端到端证据，评估的是当前 active enhanced-grid-first 三候选 score 筛选/退出流程，并使用生成后的 `final_bindings`。

- 直角坐标系样本数：325
- 直角坐标系已处理样本数：317
- 当前报告中的直角坐标系加权图表类型分类准确率：100.00%（`bubble` 与 `scatter` 按点图族互认）
- 直角坐标系 Tick MAE：0.691 px
- 直角坐标系 Tick Acc@2px：96.37%
- 直角坐标系标签准确率：96.13%

## 文件说明

- `parameter_sensitivity_samples.csv`：逐样本、逐参数组合的测量结果。
- `parameter_sensitivity_summary.csv`：每组参数设置的汇总结果。
- `parameter_sensitivity_baseline_by_type.csv`：基线参数下按图表类型汇总的结果。
- `parameter_sensitivity_summary.json`：机器可读的元信息与汇总结果。
