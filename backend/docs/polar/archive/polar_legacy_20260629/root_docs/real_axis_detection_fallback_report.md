# 真实 Radar/Rose 坐标轴检测 fallback 机制与评估记录

运行日期：2026-06-28

## 运行配置

- Python 环境：`D:\anaconda3\envs\ADtry\python.exe`
- LLM 模型：`gemini-2.5-flash-lite`
- Radar 输入：`backend/real/RadarChart-18 & RoseChart-6/RadarChart-18-final`
- Rose 输入：`backend/real/RadarChart-18 & RoseChart-6/RoseChart-6`

运行命令：

```powershell
$env:CHART_MODEL_NAME='gemini-2.5-flash-lite'
D:\anaconda3\envs\ADtry\python.exe backend\demo_radar\evaluate_radar.py --dataset real
D:\anaconda3\envs\ADtry\python.exe backend\demo_rose\evaluate_rose.py --dataset real
```

## fallback 机制说明

本机制不使用 groundtruth 做判断。groundtruth 只用于最终统计准确率，不参与图表是否进入 fallback 的判定。

流程是：

1. 先运行 CV/OCR 轴检测。
2. 若检测结果触发内部不可信条件，则进入 LLM salvage 尝试。
3. LLM salvage 的结果必须通过内部一致性检查。
4. 仍不可信时，图表标记为 fallback，不进入正常轴误差统计。

## Radar fallback 规则

Radar 当前使用以下内部规则：

- 已知多边形 radar 不进入本轮圆形 radar 算法，直接 `polygon_radar_excluded`。
- OCR/几何绑定失败或没有轴标签时，标记 fallback。
- numeric-axis 模式下，如果绑定分数显示混乱，则标记 `unreliable_numeric_mode`。判定依据为：
  - assignment score 中位数小于 0；或
  - 负分 assignment 比例大于 25%。
- 该 numeric 可信度检查对 LLM 结果同样生效。也就是说，LLM 可以帮助估计轴数或修正标签，但不能绕过绑定分数检查。

这条规则使 `radarchart_21` 被 LLM 成功救回，同时 `radarchart_9` 因 `negative_rate=0.40` 继续进入 fallback。

## Rose fallback 规则

Rose 当前使用以下内部规则：

- 图片读取失败：`image_read_failed`。
- 径向轴线证据不足：`no_axis_line_evidence`。
- OCR 候选不足：`insufficient_ocr_text` 或 `insufficient_ocr_candidates`。
- OCR 置信度不足：`low_ocr_confidence`。
- numeric-only 网格数量不规范：`non_canonical_numeric_grid`。
- 楔形内部数值被误识别为轴标签，且没有 Product/Label-X 顺序证据：`rose_wedge_no_sequential_evidence`。
- 单字母或问号占比过高，且没有顺序证据：`rose_sequential_no_evidence`。
- Label-X 轴数不规范时，先尝试 LLM salvage；若无法得到完整规范序列，则 fallback。

LLM salvage 不是无条件放行。LLM 输出必须满足以下内部一致性条件：

- 不能是无轴线证据图表的硬猜结果。
- numeric-only 非规范网格不靠 LLM 猜成正常结果。
- 如果不是完整 `Label A...` 或 `A...` 顺序序列，LLM 标签需要和高置信 OCR 候选在角度上基本一致。
- 若 LLM 标签与高置信 OCR 位置冲突，或没有任何 OCR 一致证据，则继续 fallback。

这套规则使 `plotivy-nightingale-rose-chart` 和 `RoseDiagramExample2` 被 LLM 成功救回，同时 `rose-diagram`、`rose`、`shot-nightingale-rose-chart-4` 不会因 LLM 硬猜而被错误放行。

## Radar 真实图表结果

总数：18 张。

| 类别 | 数量 |
| --- | ---: |
| 可用图表 | 9 |
| fallback 图表 | 9 |
| fallback 率 | 50.0% |
| 非 fallback 轴准确率 | 87/87 = 100.0% |

可用图表：

- `radarchart_3`
- `radarchart_4`
- `radarchart_10`
- `radarchart_15`
- `radarchart_19`
- `radarchart_20`
- `radarchart_21`
- `radarchart_22`
- `radarchart_24`

fallback 图表：

- 多边形排除：`radarchart_1`, `radarchart_5`, `radarchart_6`, `radarchart_8`, `radarchart_16`, `radarchart_17`, `radarchart_18`, `radarchart_23`
- numeric 绑定不可靠：`radarchart_9`

## Rose 真实图表结果

总数：7 张。

| 类别 | 数量 |
| --- | ---: |
| 可用图表 | 3 |
| fallback 图表 | 4 |
| fallback 率 | 57.1% |
| 非 fallback 轴准确率 | 29/29 = 100.0% |

可用图表：

- `plotivy-nightingale-rose-chart`
- `Rose1`
- `RoseDiagramExample2`

fallback 图表：

- `rose-diagram`
- `rose`
- `Rose2`
- `shot-nightingale-rose-chart-4`

## 总结

包含已知多边形 radar 在内，真实图表总数为 25 张。

| 图表类型 | 总数 | 可用 | fallback | fallback 率 | 非 fallback 轴准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Radar | 18 | 9 | 9 | 50.0% | 87/87 = 100.0% |
| Rose | 7 | 3 | 4 | 57.1% | 29/29 = 100.0% |
| 合计 | 25 | 12 | 13 | 52.0% | 116/116 = 100.0% |

如果不把已知多边形 radar 算入本轮圆形 radar/rose 算法范围，则本轮真实圆形 radar 与 rose 图表共 17 张，可用 12 张，fallback 5 张，fallback 率为 29.4%。
