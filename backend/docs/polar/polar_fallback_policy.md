# Polar Fallback Policy

本文档描述当前 `polar_axis_v1` 的 fallback 机制。该机制应被理解为运行时可靠性检查，只使用算法自身输出，不依赖 ground truth。

## 1. 总原则

流程:

```text
输入图表
-> 轴线 / 圆心 / 半径 / tick 映射检测
-> 运行时可靠性检查
-> 若 fallback=True: 不进入轴误差计算，交给 full fallback 或直接整图 LLM
-> 若 fallback=False: 与 GT 比较，计算 center / radius / radial tick assoc. 误差
```

因此，fallback rate 和 axis-prior accuracy 是两个不同指标:

```text
fallback rate = fallback 图表数量 / 全部图表数量
axis-prior accuracy = 非 fallback 图表中满足容差的比例
```

## 2. Radar Fallback

当前真实 radar 的主要 fallback reason:

```text
polygon_radar_excluded
axis_line_insufficient:<detected><<threshold>
center_not_at_origin_zero_tick_line_not_through_center
circle_quality_failed:<reason>
```

含义:

- `polygon_radar_excluded`: 当前圆形 radar 流程不处理多边形 radar。
- `axis_line_insufficient`: 轴线聚类数量不足，不能可靠建立角度先验。
- `center_not_at_origin_zero_tick_line_not_through_center`: 检测圆心与 0 tick / 轴线几何关系不一致。
- `circle_quality_failed`: 圆检测质量不足，例如边缘支持过低、未检测到可靠圆或半径异常。

当前真实集里 `radarchart_19` 属于 `center_not_at_origin_zero_tick_line_not_through_center`，因为检测到的圆心不满足 0 tick / 半径映射应有的原点一致性，因此进入 fallback，不参与成功样本误差均值。

单张复现命令:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type radar --json "backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final\RadarChart19.json"
```

当前真实 radar fallback 清单由以下 manifest 固定:

```text
backend/data/polar/manifests/real_radar_axis_manifest.json
```

检查命令:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\check_fallback_repro.py --manifest backend\data\polar\manifests\real_radar_axis_manifest.json --csv backend\data\polar\output\radar_grid_eval\radar_grid_eval_real_gt-nearest.csv
```

## 3. Rose Fallback

Rose 使用灰度网格边缘、圆心/半径 profile、轴线数量与质量门控。难图如果无法通过 OCR/视觉结构可靠建立圆心、半径、tick 映射，则进入 fallback。

当前修正真实 rose 评估使用三张可用图:

```text
Rose1
RoseDiagramExample2
plotivy-nightingale-rose-chart
```

当前 manifest:

```text
backend/data/polar/manifests/real_rose_corrected_axis_manifest.json
```

检查命令:

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\check_fallback_repro.py --manifest backend\data\polar\manifests\real_rose_corrected_axis_manifest.json --csv backend\data\polar\output\rose_grid_eval\rose_grid_eval_real_corrected_gt-nearest.csv
```

## 4. Pie / Donut Fallback

Pie 评估圆心和外半径。Donut 评估圆心、内半径和外半径。

常见 fallback:

```text
circle_quality_failed:exploded_or_nonconcentric_ring
circle_quality_failed:no_reliable_donut_boundaries
circle_quality_failed:tiny_image(...)
circle_quality_failed:low_pie_edge_support(...)
```

含义:

- 爆炸型 pie/donut 或非共圆结构不进入圆环先验检测。
- 小图或边界不清晰图不进入半径误差计算。
- 无法可靠检测内外环边界时进入 fallback。

## 5. 论文表述建议

可以写成:

```text
Before using coordinate priors, we apply a runtime reliability check based on
chart-structure cues such as axis-line support, circle-edge support,
concentricity, radius scale, and tick-radius monotonicity. Charts that fail
this check are routed to a full-image fallback and are excluded from coordinate
prior error computation.
```

中文解释:

```text
在使用坐标先验前，我们通过轴线支持度、圆边缘支持度、同心性、半径尺度和 tick-radius 单调性进行运行时可靠性检查。未通过检查的图表进入整图 fallback，不参与坐标先验误差计算。
```
