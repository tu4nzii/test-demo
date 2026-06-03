# 直角系清理后全量指标报告

数据来源：`backend/evaluation/reports/cartesian_cleaned_full_metrics_20260601.json`

评估范围：`backend/evaluation/recheck_outputs/cartesian_latest_full_results_20260528`

GT 来源：`backend/charts`

说明：本报告使用清理后的数据集统计结果，共 534 个直角系样本。`v_bar/v_bar_050` 等已确认源数据问题的样本不计入本次结果。

## 指标计算方式

### 坐标轴准确率

图表级指标。若一张图的必要坐标轴均被系统有效识别，则该图记为坐标轴正确。

计算公式：

```text
坐标轴准确率 = axis_valid / gt_total
```

其中，数值轴要求至少识别到 2 个有效加密 tick 及其像素位置；文字/类别轴要求至少识别到 1 个有效 tick。

### tick-value MAE

数值 tick 的平均绝对误差。仅统计可比较的数值 tick；缺失或非数值预测不进入 MAE 分母，但会在 tick-value 准确率中计为错误。

计算公式：

```text
tick-value MAE =
  (x_tick_numeric_mae_sum + y_tick_numeric_mae_sum)
  / (x_tick_numeric_mae_count + y_tick_numeric_mae_count)
```

### tick-value 准确率

tick 级别准确率。按 GT tick 顺序与预测 tick 对齐比较；数值 tick 使用 `1e-6` 容差，文字 tick 使用归一化后的精确匹配。缺失预测计为错误。

计算公式：

```text
tick-value 准确率 =
  (x_tick_correct + y_tick_correct)
  / (x_tick_total + y_tick_total)
```

### 图例-颜色准确率

颜色匹配指标。系列图按图例/系列标签匹配，必要时使用位置或单系列 fallback；scatter 按 `data_points` 中的对象标签匹配。bar/line/scatter 的预测颜色会映射到最近的 GT 调色板颜色，再判断是否等于对应 GT 对象/系列颜色；bubble 使用专用的同标签 HSV 色相容差口径。

计算公式：

```text
图例-颜色准确率 =
  (non_bubble_legend_color_nearest_correct + bubble_hue_tolerance_correct_15deg)
  / legend_color_total
```

注意：RGB 最近调色板口径适合 bar/line/scatter，但会明显低估 bubble。bubble 气泡通常以半透明方式绘制，且可能互相重叠，系统从图像中提取到的是实际可见的浅化/混合颜色，而 GT 中保存的是原始不透明调色板颜色。因此本报告对 bubble 采用下列专用口径，详见 `backend/evaluation/reports/bubble_color_metric_audit_20260602.md`。

bubble 专用计算方式：

```text
bubble 图例-颜色准确率 =
  count(同标签预测颜色与 GT 颜色的 HSV hue distance <= 15°)
  / bubble legend_color_total
```

### 标签名准确率

仅对 scatter 和 bubble 计算。GT 标签来自 `data_points` 的对象名；若归一化后的 GT 标签出现在预测的 `colors[].name` 中，则记为正确。

计算公式：

```text
标签名准确率 =
  point_label_name_exact / point_label_name_total
```

bar/line 图表没有散点/气泡对象标签，因此该指标为 `N/A`。

## 指标结果

| 类型 | 样本数 | 坐标轴准确率 | tick-value MAE | tick-value 准确率 | 图例-颜色准确率 | 标签名准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 534 | 100.00% | 0.000052 | 98.86% | 92.75% | 99.67% |
| v_bar | 150 | 100.00% | 0.000000 | 100.00% | 99.50% | N/A |
| h_bar | 90 | 100.00% | 0.000000 | 99.00% | 100.00% | N/A |
| line | 100 | 100.00% | 0.000000 | 99.95% | 100.00% | N/A |
| scatter | 94 | 100.00% | 0.000284 | 94.21% | 86.56% | 99.42% |
| bubble | 100 | 100.00% | 0.000000 | 99.89% | 96.12% | 100.00% |

说明：本表已采用 bubble 专用的同标签色相容差 `<= 15°` 重新审计结果。bubble 原 RGB 最近 GT 调色板口径为 `545 / 800 = 68.12%`，不作为最终颜色表现；若放宽到 `<= 25°`，结果为 `789 / 800 = 98.62%`。

## 关键计数

| 类型 | 数值 tick MAE 分母 | tick 总数 | tick 正确数 | 颜色总数 | 颜色正确数 | 标签总数 | 标签正确数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 9162 | 10629 | 10508 | 2372 | 2200 | 1842 | 1836 |
| v_bar | 2853 | 3264 | 3264 | 202 | 201 | N/A | N/A |
| h_bar | 1138 | 1993 | 1973 | 106 | 106 | N/A | N/A |
| line | 1683 | 1880 | 1879 | 222 | 222 | N/A | N/A |
| scatter | 1690 | 1692 | 1594 | 1042 | 902 | 1042 | 1036 |
| bubble | 1798 | 1800 | 1798 | 800 | 769 | 800 | 800 |

## 备注

- `tick-value MAE` 只反映数值 tick 的误差，不包含文字/类别 tick。
- `tick-value 准确率` 同时包含数值 tick 和文字/类别 tick，因此 h_bar 的文字类别轴会影响该指标。
- `图例-颜色准确率` 对 bar/line/scatter 使用最近 GT 调色板颜色匹配；对 bubble 使用同标签 HSV 色相容差 `<= 15°`。
- `标签名准确率` 只对 scatter/bubble 有意义；总体值也只由 scatter 和 bubble 的对象标签组成。
