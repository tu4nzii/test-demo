# 四类图表加密/圆检测结果总结

更新时间：2026-06-28

本文汇总当前 `data/output` 下 radar、rose、pie、donut 四类图表的加密或圆检测评估结果。主指标优先使用相对误差，分母为图像短边；容差统一为图像短边的 5%。像素误差主要保留圆心误差，半径与映射误差同时给出相对误差和必要的像素参考。

## 输出目录

- Radar：`data/output/radar_grid_eval`
- Rose：`data/output/rose_grid_eval`
- Rose 修正 GT 加密图：`data/output/rose_gt_encrypt_corrected_real`
- Pie/Donut：`data/output/pie_donut_circle_eval`

## Radar

真实 radar：

- 数据文件：`data/output/radar_grid_eval/radar_grid_eval_real_gt-nearest.csv`
- 汇总文件：`data/output/radar_grid_eval/radar_grid_eval_real_gt-nearest.json`
- total：18
- fallback：11，fallback 率 61.11%
- success：7
- tolerance_fail：0
- 圆心误差：mean 2.2143 px / 0.37%，median 2.2361 px / 0.37%，max 4.4721 px / 0.67%
- 半径最大误差：mean 0.92%，median 0.83%，max 2.67%
- 半径-tick 映射最大误差：mean 1.19%，median 0.83%，max 3.33%

真实 radar fallback：

- `radarchart_1`：polygon_radar_excluded
- `radarchart_5`：polygon_radar_excluded
- `radarchart_6`：polygon_radar_excluded
- `radarchart_8`：polygon_radar_excluded
- `radarchart_9`：axis_line_insufficient:0<2
- `radarchart_16`：polygon_radar_excluded
- `radarchart_17`：polygon_radar_excluded
- `radarchart_18`：polygon_radar_excluded
- `radarchart_19`：center_not_at_origin_zero_tick_line_not_through_center
- `radarchart_21`：circle_quality_failed:low_edge_support(0.00)
- `radarchart_23`：polygon_radar_excluded

合成 radar：

- 数据文件：`data/output/radar_grid_eval/radar_grid_eval_synthetic_gt-nearest.csv`
- total：50
- fallback：1，fallback 率 2.00%
- success：49
- tolerance_fail：0
- 圆心误差：mean 1.3456 px / 0.20%，median 1.0 px / 0.17%，max 3.6056 px / 0.50%
- 半径最大误差：mean 0.20%，median 0.17%，max 0.42%
- 半径-tick 映射最大误差：mean 0.44%，median 0.28%，max 1.53%

说明：

- 真实 radar 的最大半径/映射误差来自 `radarchart_19`，像素 max 为 21 px，但相对误差为 3.07%，仍低于 5% 容差。
- 合成 radar 的映射 max 11 px 来自 `radar_032`，相对误差为 1.53%。这是两条检测半径拟合全部 tick 时产生的映射外推误差，并不是圆半径本身错了 11 px。

## Rose

真实 rose 原始标注评估：

- 数据文件：`data/output/rose_grid_eval/rose_grid_eval_real_gt-nearest.csv`
- total：7
- fallback：4，fallback 率 57.14%
- success：3
- tolerance_fail：2
- 圆心误差：mean 0.0 px / 0.00%，median 0.0 px / 0.00%，max 0.0 px / 0.00%
- 半径最大误差：原始标注中存在尺度问题，像素误差偏大，不建议作为最终论文数值。

真实 rose 修正 GT 评估：

- 数据文件：`data/output/rose_grid_eval/rose_grid_eval_real_corrected_gt-nearest.csv`
- 汇总文件：`data/output/rose_grid_eval/rose_grid_eval_real_corrected_gt-nearest.json`
- Markdown：`data/output/rose_grid_eval/rose_grid_eval_real_corrected_gt-nearest.md`
- total：3
- fallback：0，fallback 率 0.00%
- success：3
- tolerance_fail：0
- 圆心误差：mean 1.2168 px / 0.31%，median 1.4142 px / 0.14%，max 2.2361 px / 0.78%
- 半径最大误差：mean 0.74%，median 0.10%，max 2.13%
- 半径-tick 映射最大误差：mean 0.75%，median 0.08%，max 2.17%

真实 rose 修正 GT 单图结果：

| chart_id                       | pass | center px | center % | radius max % | mapping max % | 结论 |
| ------------------------------ | ---: | --------: | -------: | -----------: | ------------: | ---- |
| Rose1                          | True |       0.0 |    0.00% |        2.13% |         2.17% | 通过 |
| RoseDiagramExample2            | True |    2.2361 |    0.78% |        0.00% |         0.00% | 通过 |
| plotivy-nightingale-rose-chart | True |    1.4142 |    0.14% |        0.10% |         0.08% | 通过 |

plotivy 处理说明：

- `plotivy-nightingale-rose-chart` 已补画最外圈 `300k` 标记。
- 因此 `300k -> 398px` 作为正常可见 tick 参与加密和半径映射评估。
- 补画 `300k` 后，plotivy 半径最大相对误差为 0.10%，映射最大相对误差为 0.08%。

合成 rose：

- 数据文件：`data/output/rose_grid_eval/rose_grid_eval_synth_gt-nearest.csv`
- total：50
- fallback：0，fallback 率 0.00%
- success：50
- tolerance_fail：0
- 圆心误差：mean 1.5785 px，median 1.4142 px，max 3.1623 px
- 半径最大误差：mean 0.84 px，median 1.0 px，max 2.0 px
- 半径-tick 映射最大误差：mean 0.84 px，median 1.0 px，max 2.0 px

## Pie

Pie 当前只评估圆心和最外圈半径，输出为检测叠加图，不做同心网格加密。

真实 pie：

- 数据文件：`data/output/pie_donut_circle_eval/pie_real_circle_eval.csv`
- 检测图目录：`data/output/pie_donut_circle_eval/pie/real/detections`
- total：11
- fallback：0，fallback 率 0.00%
- success：11
- tolerance_fail：0
- 圆心误差：mean 0.6819 px / 0.20%，median 0.621 px / 0.22%，max 1.5738 px / 0.35%
- 外半径误差：mean 0.18%，median 0.15%，max 0.45%

合成 pie：

- 数据文件：`data/output/pie_donut_circle_eval/pie_synth_circle_eval.csv`
- 检测图目录：`data/output/pie_donut_circle_eval/pie/synth/detections`
- total：50
- fallback：2，fallback 率 4.00%
- success：48
- tolerance_fail：0
- 圆心误差：mean 1.2142 px / 0.15%，median 1.5522 px / 0.16%，max 1.9801 px / 0.24%
- 外半径误差：mean 0.07%，median 0.04%，max 0.27%

合成 pie fallback：

- `110`：circle_quality_failed:low_pie_edge_support(0.13)
- `228`：circle_quality_failed:low_pie_edge_support(0.25)

## Donut

Donut 当前只评估圆心、内圈半径、外圈半径，输出为内外圆检测叠加图，不做同心网格加密。

真实 donut：

- 数据文件：`data/output/pie_donut_circle_eval/donut_real_circle_eval.csv`
- 检测图目录：`data/output/pie_donut_circle_eval/donut/real/detections`
- total：13
- fallback：4，fallback 率 30.77%
- success：9
- tolerance_fail：0
- 圆心误差：mean 0.8212 px / 0.20%，median 0.7089 px / 0.17%，max 1.6941 px / 0.32%
- 外半径误差：mean 0.14%，median 0.00%，max 0.50%
- 内半径误差：mean 0.04%，median 0.00%，max 0.20%

真实 donut fallback：

- `DonutChart11`：circle_quality_failed:exploded_or_nonconcentric_ring
- `DonutChart13`：circle_quality_failed:no_reliable_donut_boundaries
- `DonutChart14`：circle_quality_failed:exploded_or_nonconcentric_ring
- `DonutChart16`：circle_quality_failed:tiny_image(short_side=43)

合成 donut：

- 数据文件：`data/output/pie_donut_circle_eval/donut_synth_circle_eval.csv`
- 检测图目录：`data/output/pie_donut_circle_eval/donut/synth/detections`
- total：50
- fallback：0，fallback 率 0.00%
- success：50
- tolerance_fail：0
- 圆心误差：mean 0.6451 px / 0.11%，median 0.524 px / 0.10%，max 1.0579 px / 0.21%
- 外半径误差：mean 0.03%，median 0.00%，max 0.25%
- 内半径误差：mean 0.13%，median 0.06%，max 0.40%

## 当前结论

- Radar：合成集稳定；真实集 fallback 来自多边形 radar（8 张）、圆心偏移 0 刻度线不经过圆心（1 张，`radarchart_19`）、圆检测质量（1 张）及轴线不足（1 张）。非 fallback 样本全部低于 5% 容差。
- Rose：合成集稳定；真实修正 GT 三张均通过。plotivy 已补画最外圈 `300k` 标记，因此按正常可见 tick 处理。
- Pie：真实集全部通过；合成集中 2 张低圆周边缘支撑图进入 fallback 后，成功样本全部通过。
- Donut：合成集全部通过；真实集 fallback 主要对应爆炸/非共圆、小图、无法可靠检测边界，成功样本全部通过。

## 复跑命令

Radar：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\evaluate_radar_grid_extraction.py --dataset real --tick-mode gt-nearest
D:\anaconda3\envs\ADtry\python.exe backend\evaluate_radar_grid_extraction.py --dataset synthetic --tick-mode gt-nearest
```

Rose 修正 GT 加密图：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\demo_rose\encrypt_rose_from_json_gt.py --json-dir "backend\real\RadarChart-18 & RoseChart-6\RoseChart-6" --charts Rose1_gt_encrypt RoseDiagramExample2_gt_encrypt plotivy-nightingale-rose-chart_gt_encrypt --output-dir "data\output\rose_gt_encrypt_corrected_real"
```

Pie/Donut：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\evaluate_pie_donut_circle_extraction.py --chart-type all --dataset all
```
