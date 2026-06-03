# 网格加密报告

日期：2026-06-02

本报告汇总当前系统对直角坐标系与极坐标系图表的加密网格生成能力。直角坐标系指标来自清理后的全量评估报告；极坐标系指标来自 radar/rose 圆心、半径与轴提取评估结果。

## 评估范围

| 坐标系 | 图表类型 | 样本数 | 主要评估对象 |
| --- | --- | ---: | --- |
| 直角坐标系 | v_bar, h_bar, line, scatter, bubble | 534 | 坐标轴、tick-value、文字轴、图例颜色、对象标签 |
| 极坐标系 | radar, rose | 200 | 圆心、内外半径、径向像素误差、极轴标签 |
| 合计 | 7 类图表 | 734 | 加密网格所需几何与语义先验 |

直角坐标系来源：

- 指标报告：`backend/evaluation/reports/cartesian_cleaned_metrics_summary_20260602.md`
- 明细 JSON：`backend/evaluation/reports/cartesian_cleaned_full_metrics_20260601.json`
- 评估输出：`backend/evaluation/recheck_outputs/cartesian_latest_full_results_20260528`
- GT 来源：`backend/charts`

极坐标系来源：

- radar 样本数：100
- rose 样本数：100
- radar 轴筛选输出：`d:/home work/Agent.paper/Agent/evaluation_datasets_with_axes_radar.json`
- rose 轴筛选输出：`d:/home work/Agent.paper/Agent/evaluation_datasets_with_axes_rose.json`

## 加密网格依赖项

加密网格生成依赖两类信息：

| 坐标系 | 几何先验 | 语义先验 | 加密网格生成含义 |
| --- | --- | --- | --- |
| 直角坐标系 | X/Y 轴、tick 像素、网格线位置 | tick 值、轴类型、颜色/标签 | 在数值轴相邻 tick 中插入加密 tick，并绘制加密网格 |
| 极坐标系 | 圆心、内半径、外半径、径向像素尺度 | 极轴标签 axis_labels、径向 tick | 根据圆心和半径范围生成极坐标角度/半径网格 |

因此，直角坐标系重点看轴与 tick-value；极坐标系重点看圆心、半径与极轴标签。

## 总览

| 坐标系 | 样本数 | 轴/极轴准确率 | 关键几何误差 | 语义补充 |
| --- | ---: | ---: | --- | --- |
| 直角坐标系 | 534 | 100.00% | tick-value MAE 0.000052 | tick-value 准确率 98.86%，文字轴准确率 98.83% |
| radar | 100 | 93.75%（90/96，约 94%） | 圆心 RMSE 2.469818，r_pixel_err 1.948472 | 90 条包含 axis_labels |
| rose | 100 | 96.59%（85/88，约 96%） | 圆心 RMSE 2.000000，r_pixel_err 1.190341 | 85 条包含 axis_labels |
| 极坐标合计 | 200 | 95.11%（175/184） | 见 radar/rose 分项 | 184 条参与轴提取评估，175 条含 axis_labels |

补充说明：若按极坐标总样本数计算端到端 `axis_labels` 覆盖率，则 radar 为 90.00%（90/100），rose 为 85.00%（85/100），极坐标合计为 87.50%（175/200）。上表的轴/极轴准确率采用参与轴提取评估的数据作为分母。

## 直角坐标系结果

| 类型 | 样本数 | 坐标轴准确率 | tick-value MAE | tick-value 准确率 | 图例-颜色准确率 | 标签名准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 534 | 100.00% | 0.000052 | 98.86% | 92.75% | 99.67% |
| v_bar | 150 | 100.00% | 0.000000 | 100.00% | 99.50% | N/A |
| h_bar | 90 | 100.00% | 0.000000 | 99.00% | 100.00% | N/A |
| line | 100 | 100.00% | 0.000000 | 99.95% | 100.00% | N/A |
| scatter | 94 | 100.00% | 0.000284 | 94.21% | 86.56% | 99.42% |
| bubble | 100 | 100.00% | 0.000000 | 99.89% | 96.12% | 100.00% |

直角坐标系关键计数：

| 类型 | 数值 tick MAE 分母 | tick 总数 | tick 正确数 | 颜色总数 | 颜色正确数 | 标签总数 | 标签正确数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 9162 | 10629 | 10508 | 2372 | 2200 | 1842 | 1836 |
| v_bar | 2853 | 3264 | 3264 | 202 | 201 | N/A | N/A |
| h_bar | 1138 | 1993 | 1973 | 106 | 106 | N/A | N/A |
| line | 1683 | 1880 | 1879 | 222 | 222 | N/A | N/A |
| scatter | 1690 | 1692 | 1594 | 1042 | 902 | 1042 | 1036 |
| bubble | 1798 | 1800 | 1798 | 800 | 769 | 800 | 800 |

说明：bubble 的图例-颜色准确率采用同标签 HSV 色相容差 `<= 15°` 的专用口径，即 `769 / 800 = 96.12%`。

## 极坐标系圆心与半径结果

### radar

| 指标 | 数值 |
| --- | ---: |
| 图表类型 | radar |
| 样本数 | 100 |
| 圆心检测精准率（err_center < 5） | 1.000000 |
| 圆心检测 RMSE | 2.469818 |
| r_pixel_err | 1.948472 |
| inner_r_err_MAE | 1.365278 |
| inner_r_err_RE | 0.007140 |
| outer_r_err_MAE | 2.218750 |
| outer_r_err_RE | 0.006915 |

### rose

| 指标 | 数值 |
| --- | ---: |
| 图表类型 | rose |
| 样本数 | 100 |
| 圆心检测精准率（err_center < 5） | 0.990000 |
| 圆心检测 RMSE | 2.000000 |
| r_pixel_err | 1.190341 |
| inner_r_err_MAE | 1.625000 |
| inner_r_err_RE | 0.005839 |
| outer_r_err_MAE | 1.625000 |
| outer_r_err_RE | 0.005839 |

## 极坐标系轴提取结果

| 类型 | 总样本数 | 参与轴评估样本 | 含 axis_labels 样本 | 参与评估口径准确率 | 端到端覆盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| radar | 100 | 96 | 90 | 93.75%（约 94%） | 90.00% |
| rose | 100 | 88 | 85 | 96.59%（约 96%） | 85.00% |
| 极坐标合计 | 200 | 184 | 175 | 95.11% | 87.50% |

轴提取口径说明：

- radar：从 96 条数据中筛选出 90 条包含 `axis_labels` 的数据，轴准确率约为 94%。
- rose：从 88 条数据中筛选出 85 条包含 `axis_labels` 的数据，轴准确率约为 96%。
- `参与评估口径准确率 = 含 axis_labels 样本 / 参与轴评估样本`。
- `端到端覆盖率 = 含 axis_labels 样本 / 总样本数`。

## 指标定义

### 圆心检测精准率

```text
圆心检测精准率 = count(err_center < 5) / total
```

表示预测圆心与 GT 圆心之间的像素距离小于 5 的样本比例。

### 圆心检测 RMSE

```text
圆心检测 RMSE = sqrt(mean(err_center^2))
```

反映圆心定位误差的整体稳定性。

### r_pixel_err

径向像素尺度误差，用于衡量半径方向上像素到数值映射的偏差。该指标越低，极坐标径向网格越稳定。

### inner/outer 半径误差

```text
inner_r_err_MAE = mean(abs(pred_inner_r - gt_inner_r))
outer_r_err_MAE = mean(abs(pred_outer_r - gt_outer_r))
inner_r_err_RE = mean(abs(pred_inner_r - gt_inner_r) / gt_inner_r)
outer_r_err_RE = mean(abs(pred_outer_r - gt_outer_r) / gt_outer_r)
```

其中 MAE 使用像素误差，RE 使用相对误差。内外半径决定极坐标网格的起止范围。

### 极轴准确率

```text
极轴准确率 = count(samples with valid axis_labels) / count(samples participating in axis extraction evaluation)
```

该指标反映系统是否能为极坐标图提取角度方向的标签先验。

## 结论

1. 直角坐标系加密网格已经具备稳定的轴和 tick 先验，坐标轴准确率为 100.00%，tick-value 准确率为 98.86%。
2. 直角坐标系颜色指标已按图表类型修正口径，bubble 使用同标签 HSV 色相容差 `<= 15°` 后，颜色准确率为 96.12%，整体颜色准确率为 92.75%。
3. radar 的圆心检测精准率达到 100.00%，圆心 RMSE 为 2.469818，内外半径相对误差均低于 1%，说明径向网格几何基础稳定。
4. rose 的圆心检测精准率达到 99.00%，圆心 RMSE 为 2.000000，内外半径相对误差约 0.58%，几何定位稳定性优于 radar。
5. 极坐标轴标签提取仍是主要改进点。按参与轴评估口径，radar 为 93.75%，rose 为 96.59%；若按全部 200 个极坐标样本端到端统计，`axis_labels` 覆盖率为 87.50%。
6. 当前全量结果表明，系统已经覆盖直角坐标系和极坐标系的加密网格生成关键先验；后续优化重点应放在极坐标 `axis_labels` 端到端覆盖率，以及极坐标处理流程进一步脱离外部 JSON 依赖。

