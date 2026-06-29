# 极坐标 Fallback 机制

本机制是运行时可靠性检查，只使用算法自身从图像中得到的结构信息，不使用 ground truth 做 fallback 判断。Ground truth 只用于 fallback 之后的误差量化。

## 1. 总原则

```text
输入图表
-> 轴线 / 圆心 / 半径 / tick 映射检测
-> runtime fallback 判断
-> fallback=True: 不进入坐标先验误差统计，后续走整图兜底
-> fallback=False: 再和 GT 比较，计算 center / radius / radial tick assoc. 误差
```

论文里可以这样解释：

```text
Before using coordinate priors, we apply a runtime reliability check based on chart-structure cues such as axis-line support, circle-edge support, concentricity, radius scale, and tick-radius monotonicity. Charts that fail this check are routed to a full-image fallback and are excluded from coordinate-prior error computation.
```

## 2. Radar Fallback

当前 radar 主要 fallback reason：

```text
polygon_radar_excluded
axis_line_insufficient:<detected><<threshold>
center_not_at_origin_zero_tick_line_not_through_center
circle_quality_failed:<reason>
```

含义：

- `polygon_radar_excluded`：当前圆形 radar 流程不处理多边形 radar。
- `axis_line_insufficient`：轴线聚类数量不足，不能可靠建立角度先验。
- `center_not_at_origin_zero_tick_line_not_through_center`：检测圆心和 0 tick / 半径映射的几何关系不一致。
- `circle_quality_failed`：圆检测质量不足，例如边缘支持过低、未检测到可靠圆或半径异常。

`RadarChart19` 当前应进入 fallback：

```text
fallback_reason = center_not_at_origin_zero_tick_line_not_through_center
```

单张复现：

```powershell
D:\anaconda3\envs\ADtry\python.exe backend\polar\scripts\run_one_chart.py --chart-type radar --json "backend\real\RadarChart-18 & RoseChart-6\RadarChart-18-final\RadarChart19.json"
```

## 3. Rose Fallback

Rose 使用灰度网格边缘、圆心/半径 profile、轴线数量和质量门控。困难真实图如果 OCR/视觉结构无法可靠建立圆心、半径、tick 映射，则进入 fallback。

当前修正后的真实 rose 可用图包括：

```text
Rose1
RoseDiagramExample2
plotivy-nightingale-rose-chart
```

其他真实 rose 若结构不可靠，进入 fallback，不参与成功图表误差均值。

## 4. Pie / Donut Fallback

Pie 评估圆心和外半径。Donut 评估圆心、内半径和外半径。

常见 fallback：

```text
circle_quality_failed:exploded_or_nonconcentric_ring
circle_quality_failed:no_reliable_donut_boundaries
circle_quality_failed:tiny_image(...)
circle_quality_failed:low_pie_edge_support(...)
```

含义：

- 爆炸型 pie/donut 或非共圆结构不进入圆环先验检测。
- 小图或边界不清晰图不进入半径误差统计。
- 无法可靠检测内外环边界时进入 fallback。

## 5. 当前代码入口

单张图流程：

```text
backend/polar/scripts/run_one_chart.py
```

后续值评估：

```text
backend/polar/legacy/demo_radar/demo_evaluation_radar_1.py
```

旧的批量统计和 manifest 复现脚本已经归档到：

```text
backend/polar/archive_unused/20260629_cleanup
```
