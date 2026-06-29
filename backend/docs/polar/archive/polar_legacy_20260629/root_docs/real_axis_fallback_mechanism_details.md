# 真实 Radar/Rose 轴检测 Fallback 机制说明

本文档基于当前项目实际代码整理，主要对应：

- `backend/demo_radar/detect_radar_axes.py::detect_axes()`
- `backend/demo_radar/evaluate_radar.py::evaluate_one()`
- `backend/demo_rose/detect_rose_axes.py::detect_rose()`
- `backend/demo_rose/detect_rose_axes.py::_fallback_or_llm_salvage()`
- `backend/demo_rose/detect_rose_axes.py::_llm_salvage_reject_reason()`

每张真实图表会先经过 CV/OCR 轴检测；当检测结果触发内部不可信条件时，系统允许 LLM salvage 尝试补救。LLM salvage 不是无条件放行，必须通过内部一致性检查。若仍不可信，则该图表计入 fallback，不进入正常轴误差统计。

重要原则：fallback 判断只使用检测端证据，包括图像结构、轴线证据、OCR 候选、OCR 置信度、几何绑定分数、LLM salvage 输出与 OCR 的一致性等，不使用 groundtruth。Groundtruth 只用于最终统计非 fallback 样本的准确率。

## 总体机制

最终 fallback 主要由四类检查构成：

1. 图表类型与结构检查：排除本轮算法不处理或几何结构不可靠的图表。
2. OCR/几何绑定质量检查：防止 OCR 证据不足或绑定混乱的结果进入成功集合。
3. LLM salvage 一致性检查：允许 LLM 救回难图，但拒绝无图像证据支撑的硬猜结果。
4. 轴标签质量检查：防止问号、标题片段、楔形内部数值等错误文本被当成轴标签。

## 1. 图表类型与结构检查

### 目的

该检查用于排除当前圆形 radar/rose 轴检测机制不适合处理的图表，或缺少基础径向结构证据的图表。

本轮真实评估暂时不处理多边形 radar。对于 rose 图，如果图像中无法检测到足够的径向轴线证据，则认为后续 OCR 和 LLM 都缺少可靠几何锚点。

### 当前实际触发条件

| reason                     | 触发条件                                                     | 含义                                   |
| -------------------------- | ------------------------------------------------------------ | -------------------------------------- |
| `polygon_radar_excluded` | 真实 radar 图编号属于已知多边形集合`{1,5,6,8,16,17,18,23}` | 多边形 radar 不进入本轮圆形 radar 算法 |
| `image_read_failed`      | 图片无法读取                                                 | 无法进行检测                           |
| `no_axis_line_evidence`  | Rose 图 Hough 直线检测后，穿过圆心附近的径向轴线簇少于 3 个  | 径向结构证据不足                       |

### 例子

`shot-nightingale-rose-chart-4` 的径向轴线证据不足，触发：

```text
no_axis_line_evidence
```

即使 LLM 能从图中猜出一些水果名，该图仍不会放行，因为缺少足够的轴线几何证据。

### 论文表述

可以写成：

> We first reject charts that are outside the supported chart family or lack sufficient radial-axis evidence. Polygonal radar charts are excluded from the circular radar pipeline, and rose charts without reliable radial line evidence are marked as fallback before entering normal axis evaluation.

## 2. OCR/几何绑定质量检查

### 目的

该检查用于判断 OCR 和几何绑定是否足够可信。系统不会仅凭检测到若干文字就认为轴标签可靠，而是进一步检查 OCR 数量、置信度、候选是否位于合理的外圈标签区域，以及绑定分数是否显示混乱。

### Rose 触发条件

| reason                          | 当前条件                                                          | 含义                       |
| ------------------------------- | ----------------------------------------------------------------- | -------------------------- |
| `insufficient_ocr_text`       | `n_alpha + n_numeric < 3`，且没有 Product/Label-X 顺序证据      | 整体 OCR 文本证据不足      |
| `insufficient_ocr_candidates` | 过滤标题/说明文字后，可用 OCR 候选少于 3 个                       | 外圈可用 OCR 候选不足      |
| `low_ocr_confidence`          | 可用 OCR 候选最高置信度`< 0.45`，或 top-3 平均置信度 `< 0.40` | OCR 置信度整体过低         |
| `non_canonical_numeric_grid`  | numeric-only 模式下，检测出的轴数不属于常用规范集合               | 数值角度网格不完整或不可靠 |

### Radar 触发条件

| reason                      | 当前条件                                                                               | 含义                                 |
| --------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------ |
| `unreliable_numeric_mode` | numeric-axis 模式下，assignment score 中位数`< 0`，或负分 assignment 比例 `> 0.25` | OCR/几何绑定混乱，标签与轴位置不可信 |

这条规则对 LLM 结果同样生效。也就是说，LLM 可以帮助估计轴数或修正标签，但不能绕过 numeric 绑定分数检查。

### 例子

`radarchart_9` 经过 LLM 后曾被估计为 5 轴，但绑定结果中混入了 `Squirtle`、`80`、`3` 等非轴标签内容，且负分 assignment 比例为 `0.40`：

```text
negative_rate = 0.40 > 0.25
```

因此触发：

```text
unreliable_numeric_mode
```

该图最终仍计入 fallback。

### 论文表述

可以写成：

> We mark an axis as fallback when OCR evidence or geometric binding quality is insufficient. For rose charts, the gate checks the number and confidence of usable OCR candidates after removing titles and annotations. For radar charts, numeric-axis outputs are rejected when assignment scores indicate unstable binding, even if the axis count is suggested by the LLM.

## 3. LLM Salvage 一致性检查

### 目的

该检查用于允许 LLM 救回 OCR 难以处理但图像结构仍可信的图表，同时防止 LLM 在缺少图像证据时硬猜出一个看似合理的轴序列。

LLM salvage 只在检测端已经准备 fallback 时触发。LLM 会读取整图外圈轴标签，并输出：

```json
{"axes":[{"angle":90,"label":"example"}]}
```

但 LLM 输出必须再通过内部一致性检查，才能从 fallback 中救回。

### 当前实际通过条件

LLM salvage 满足以下任一条件时可放行：

| 条件                           | 含义                                                                  |
| ------------------------------ | --------------------------------------------------------------------- |
| 完整`Label A...Label N` 序列 | LLM 读出完整 Label-X 顺序标签，系统使用统一 rose 角度网格生成最终标签 |
| 完整`A...N` 单字母序列       | LLM 读出完整单字母顺序标签，系统使用统一 rose 角度网格生成最终标签    |
| 与高置信 OCR 角度一致          | LLM 标签与高置信 OCR 候选在角度上基本一致，且没有明显冲突             |

### 当前实际拒绝条件

| reject reason                               | 触发条件                                            | 含义                                         |
| ------------------------------------------- | --------------------------------------------------- | -------------------------------------------- |
| `llm_rejected_no_axis_line_evidence`      | 原始 reason 为`no_axis_line_evidence`             | 没有径向轴线证据时，不允许 LLM 硬猜放行      |
| `llm_rejected_non_canonical_numeric_grid` | 原始 reason 为`non_canonical_numeric_grid`        | numeric-only 非规范网格不靠 LLM 猜成正常结果 |
| `llm_rejected_numeric_only_salvage`       | LLM 输出全是数字标签                                | 避免将径向 tick 或角度数字误当成轴标签       |
| `llm_rejected_no_ocr_validation`          | LLM 输出不是完整顺序序列，且没有 OCR 候选可交叉验证 | 缺少图像证据                                 |
| `llm_rejected_ocr_angle_conflict`         | LLM 标签与高置信 OCR 位置冲突，且没有一致证据       | LLM 与图像 OCR 冲突                          |
| `llm_rejected_no_ocr_agreement`           | LLM 标签没有任何高置信 OCR 一致证据                 | LLM 结果缺少图像支撑                         |

### 例子

例子 1：LLM 成功救回。

`RoseDiagramExample2` 的 OCR 只识别到 11 个 Label-X 候选，触发：

```text
non_canonical_label_x_grid(n=11)
```

LLM salvage 后读出完整 `Label A` 到 `Label L`，属于完整 Label-X 序列。系统不直接使用 LLM 的角度，而是使用 rose 统一角度网格生成 12 个标签，因此该图被救回。

例子 2：LLM 不放行。

`rose` 是 numeric-only 角度图，LLM 可以输出 `0, 30, 60, ...` 这类数字标签，但该图原始触发 `non_canonical_numeric_grid`，且 LLM 输出全是数字。因此系统判为不可信，继续 fallback。

### 论文表述

可以写成：

> LLM salvage is used only as a secondary recovery step for charts that already fail the CV/OCR gate. The LLM output is not accepted unconditionally. It must either form a complete ordered label sequence or agree with high-confidence OCR anchors at compatible angular positions. Outputs that contradict OCR geometry, lack OCR support, or attempt to recover numeric-only non-canonical grids are rejected and kept in fallback.

## 4. 轴标签质量检查

### 目的

该检查用于防止明显错误的文本被当成轴标签，例如：

- `?`
- 空字符串；
- 标题或副标题片段；
- 带有 `$`、括号等明显标题/注释特征的长文本；
- rose 楔形内部数值；
- 无顺序证据的单字母或问号序列。

### 当前实际触发条件

| reason                                         | 触发条件                                                | 含义                                 |
| ---------------------------------------------- | ------------------------------------------------------- | ------------------------------------ |
| `unreliable_axis_labels(unknown=...)`        | 轴标签中出现`?` 或空标签                              | 标签未读出                           |
| `unreliable_axis_labels(long_fragments=...)` | 多个标签过长，或包含标题/注释特征                       | 可能把标题、说明文字或注释当成轴标签 |
| `rose_wedge_no_sequential_evidence`          | 大量标签是楔形内部数值，且没有 Product/Label-X 顺序证据 | 楔形数值被误识别为轴标签             |
| `rose_sequential_no_evidence`                | 单字母或问号占比高，但没有 Product/Label-X 顺序证据     | 不能确认顺序轴标签                   |

### 例子

`plotivy-nightingale-rose-chart` 的初始 OCR 结果混入标题片段和 `?`，触发标签质量问题。随后 LLM salvage 读出完整月份标签，且与图中高置信 OCR 月份候选一致，因此最终被救回。

### 论文表述

可以写成：

> We additionally check the quality of the recovered axis labels. Labels containing unknown tokens, title fragments, annotation-like long strings, or wedge-internal numeric values are treated as unreliable unless they can be validated by ordered-label evidence or high-confidence OCR anchors.

## 当前真实图表结果

使用 `gemini-2.5-flash-lite` 启用 LLM salvage 后，真实图表结果如下。

| 图表类型 | 总数 | 可用 | fallback | fallback 率 | 非 fallback 轴准确率 |
| -------- | ---: | ---: | -------: | ----------: | -------------------: |
| Radar    |   18 |    9 |        9 |       50.0% |       87/87 = 100.0% |
| Rose     |    7 |    3 |        4 |       57.1% |       29/29 = 100.0% |
| 合计     |   25 |   12 |       13 |       52.0% |     116/116 = 100.0% |

如果不把已知多边形 radar 算入本轮圆形 radar/rose 算法范围，则本轮真实圆形 radar 与 rose 图表共 17 张，可用 12 张，fallback 5 张，fallback 率为 29.4%。

## 简洁版本

如果论文正文只需要简洁说明，可以写：

> We use a detection-side fallback gate to prevent unreliable polar-axis detections from entering the success set. The gate is based only on image-side evidence and model outputs, without using ground-truth annotations. It first rejects charts outside the supported circular radar/rose family or charts lacking sufficient radial structure. It then checks OCR support, OCR confidence, axis-label quality, and geometric binding stability. For difficult cases, an LLM salvage step is allowed, but the LLM output is accepted only when it forms a complete ordered label sequence or agrees with high-confidence OCR anchors at compatible angular positions. Numeric-only non-canonical grids, no-axis-line cases, OCR/LLM conflicts, and unsupported hard guesses remain in fallback. Final accuracy is computed only on non-fallback charts.
