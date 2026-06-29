# 生成端 Fallback 机制说明

本文档基于当前项目实际代码整理，主要对应：

- `grid_line_filter.py::priority_axis_quality()`
- `grid_line_filter.py::repeated_semantic_label_failure()`
- `grid_line_filter.py::grid_failure_report()`
- `grid_adjudication.py::binding_quality()`

每张图生成阶段都会写出 `*_grid_failure.json`。只要 `reasons` 非空，系统就设置 `failed=true`。这些样本仍保留 final overlay、bindings、priority decision 等文件供人工检查，但在评估时计入 fallback/skipped，不进入成功样本的 MAE 和 accuracy 分母。

重要原则：fallback 判断只使用生成端证据，包括 P1/P2/P3 候选、OCR 框、MLLM 输出、绑定质量、线数和几何距离，不使用 GT。

## 总体机制

最终 fallback 主要由三类检查构成：

1. OCR/MLLM 支撑度检查：防止 MLLM-only label 被当成正式几何证据。
2. 重复语义 label 检查：防止高基数类别/时间轴里重复 label 造成不可唯一绑定。
3. 异常几何检查：防止线数、绑定数、OCR 距离、候选有效性明显异常的结果进入成功集合。

## 1. OCR/MLLM 支撑度检查

### 目的

该检查用于判断最终候选是否过度依赖 MLLM 语义，而缺少图像中的 OCR 框或物理几何证据。

项目不会简单地把 MLLM 输出的 tick label 全部当成正式 label。系统会统计每个轴的 OCR 支撑、MLLM-only 数量、唯一 OCR 绑定数量和 OCR 到网格线的几何距离。如果最终结果主要由 MLLM-only label 构成，且 OCR-bound 支撑比例不足，就会主动 fallback。

### 相关字段

这些字段来自 `grid_adjudication.py::binding_quality()`，并在 `grid_line_filter.py::priority_axis_quality()` 中用于 fallback 判断。

| 字段 | 含义 |
| --- | --- |
| `target_count` | MLLM 或融合语义认为该轴应有的 tick 数 |
| `line_count` | 当前候选实际生成的网格线数 |
| `labeled_count` | 成功绑定 label 的线数 |
| `ocr_bound_count` | label 来源包含真实 OCR 框的数量，包含 `ocr` 和 `ocr+mllm` |
| `strong_count` | OCR 和 MLLM 同时支持的绑定数量，即 `ocr+mllm` |
| `mllm_only_count` | 只有 MLLM 支持、没有 OCR 框支撑的 label 数 |
| `unique_ocr_count` | 唯一 OCR 框绑定数量，用于防止同一个 OCR 框重复绑定多条线 |
| `duplicate_ocr_count` | OCR 框重复绑定数量 |
| `mean_ocr_distance` | OCR 框到网格线的平均距离 |
| `max_ocr_distance` | OCR 框到网格线的最大距离 |

### 当前实际触发条件

| reason | 触发条件 |
| --- | --- |
| `x_axis_semantic_only_low_ocr_support` / `y_axis_semantic_only_low_ocr_support` | 最终选择 `semantic_guide`；轴类型为 `category/time/mixed`；`target_count >= 12`；`ocr_bound_count / target_count < 0.7`；且 `mllm_only_count >= 5` |
| `x_axis_dense_axis_low_unique_ocr_support` / `y_axis_dense_axis_low_unique_ocr_support` | 轴类型为 `numeric/time`；`target_count >= 12`；`labeled_count / target_count < 0.65`；且 `unique_ocr_count / target_count < 0.55` |
| `x_axis_numeric_semantic_only_low_ocr_support` / `y_axis_numeric_semantic_only_low_ocr_support` | 最终选择 `semantic_guide`；轴类型为 `numeric`；`target_count >= 4`；`ocr_bound_count / target_count < 0.25`；且 `mllm_only_count >= max(3, ceil(target_count * 0.7))` |
| `x_axis_numeric_selected_low_ocr_support` / `y_axis_numeric_selected_low_ocr_support` | 轴类型为 `numeric`；`target_count >= 6`；`ocr_bound_count / target_count < 0.25`；且 `mllm_only_count >= max(3, ceil(target_count * 0.65))` |

### 关于“几个 OCR anchor 才够”

当前项目不是固定要求“至少 2 个 OCR anchor”。它主要看 OCR-bound 比例。

对 numeric 轴，核心比例是：

```text
ocr_bound_count / target_count >= 0.25
```

因此：

| `target_count` | 通常不会被 `numeric_selected_low_ocr_support` 卡住的 OCR-bound 数 |
| ---: | --- |
| 6 | 2 个及以上 |
| 8 | 2 个及以上 |
| 10 | 3 个及以上 |
| 11 | 3 个及以上 |
| 12 | 3 个及以上 |

也就是说，2 个 OCR anchor 对 6 或 8 个 tick 的 numeric 轴通常够，但对 10 个以上 tick 的 numeric 轴不一定够。

### 例子

假设一张数值折线图中，MLLM 输出了 6 个 tick label：

```text
0, 20, 40, 60, 80, 100
```

如果图像 OCR 只检测到 1 个有效 tick label，其余 5 个 label 都是 MLLM-only，且没有可靠 P1 原生网格或 P2 短 tick 几何支撑，则：

```text
target_count = 6
ocr_bound_count = 1
mllm_only_count = 5
ocr_bound_count / target_count = 0.167 < 0.25
```

这会触发 `numeric_selected_low_ocr_support`，系统主动 fallback。

如果同样是 6 个 tick，但 OCR-bound 有 2 个，并且数值序列规则、OCR 距离小、候选线数合理，则通常不会因为这条 low-OCR 规则被卡住。

### 论文表述

可以写成：

> We mark an axis as fallback when the selected grid relies primarily on semantic-only MLLM labels without sufficient image-grounded OCR support. The gate checks the ratio of OCR-bound labels, unique OCR bindings, labeled tick coverage, and the number of MLLM-only labels. Dense numeric/time axes and high-cardinality category/time axes are rejected when OCR support is below the required threshold.

## 2. 重复语义 Label 检查

### 目的

该检查用于处理高基数类别轴或时间轴中的重复 label 歧义。

有些图表的 x 轴或 y 轴可能出现重复语义 label，例如季度序列：

```text
Q1, Q2, Q3, Q4, Q1, Q2, Q3, Q4
```

如果最终候选主要来自 `semantic_guide`，且没有 P1 原生网格或 P2 短 tick 形成交叉验证，那么系统无法唯一判断某个 `Q1` 应该绑定到哪一条网格线。因此，即使语义序列看起来合理，也会主动 fallback。

### 当前实际触发条件

这部分由 `grid_line_filter.py::repeated_semantic_label_failure()` 实现。

| 条件 | 当前代码要求 |
| --- | --- |
| 候选来源 | `axis_quality.choice == "semantic_guide"` |
| 有效候选数量 | `valid_candidate_count == 1` |
| 轴类型 | `category/time/mixed` |
| 目标 tick 数 | `target_count >= 8` |
| 实际有文本的 label 数 | `len(labels) >= 8` |
| 重复 label 数 | `duplicate_count >= max(3, ceil(len(labels) * 0.25))` |
| 唯一 label 比例 | `unique_ratio <= 0.75` |

触发后的 reason 为：

- `x_axis_semantic_only_repeated_labels`
- `y_axis_semantic_only_repeated_labels`

### 例子

假设一张 grouped bar 图的 x 轴是：

```text
Q1, Q2, Q3, Q4, Q1, Q2, Q3, Q4, Q1, Q2, Q3, Q4
```

总 label 数为 12，唯一 label 只有 4 个：

```text
unique_count = 4
duplicate_count = 8
unique_ratio = 4 / 12 = 0.333
```

因为重复数量超过 `max(3, ceil(12 * 0.25)) = 3`，且 `unique_ratio <= 0.75`，如果该轴最终只能靠 `semantic_guide`，系统会触发：

```text
x_axis_semantic_only_repeated_labels
```

### 论文表述

可以写成：

> For high-cardinality categorical, temporal, or mixed axes, repeated semantic labels make the correspondence between labels and grid lines ambiguous. If the final candidate is selected solely from the semantic-guide path and repeated labels occupy a substantial portion of the bound ticks, the sample is marked as fallback because the generation process lacks unique geometric evidence for the binding.

## 3. 异常几何检查

### 目的

该检查用于防止几何结构明显不可靠的候选进入成功集合。

它不是只看有没有线，而是综合检查：

- 最终是否有网格线；
- 至少一个轴是否有两个以上 labeled ticks；
- 候选线数是否严重少于目标 tick 数；
- 候选线数是否过多；
- label 绑定是否太少；
- OCR 是否被重复绑定；
- OCR 到网格线的距离是否存在离群；
- P2 物理 tick 候选和 P3 语义 guide 候选是否冲突。

### 最终硬检查

这部分由 `grid_line_filter.py::grid_failure_report()` 实现。

| reason | 触发条件 | 含义 |
| --- | --- | --- |
| `no_final_grid_lines` | `horizontal_count + vertical_count == 0` | 最终没有任何水平或垂直网格线 |
| `no_axis_with_two_labeled_ticks` | `max(x_labeled, y_labeled) < 2` | x/y 两个轴都没有至少 2 个可确认 tick label |
| `mllm_unavailable_for_priority_arbitration` | priority 未启用，且 MLLM 有 error | 需要 MLLM priority arbitration 时 MLLM 不可用 |

### 候选有效性检查

这部分来自 `grid_adjudication.py::binding_quality()`。如果候选被判为 invalid，最终会在 `priority_axis_quality()` 中产生：

- `x_axis_selected_candidate_invalid`
- `y_axis_selected_candidate_invalid`
- `x_axis_selected_candidate_has_invalid_reasons`
- `y_axis_selected_candidate_has_invalid_reasons`

候选 invalid reason 包括：

| invalid reason | 触发条件 |
| --- | --- |
| `no_grid_lines` | `line_count <= 0` |
| `too_few_lines_for_mllm_ticks` | `target_count >= 3` 且 `line_count < max(2, floor(target_count * 0.45))` |
| `too_few_bound_labels` | `target_count >= 3` 且 `labeled_count < max(1, floor(target_count * 0.35))` |
| `too_few_unique_ocr_bindings` | `target_count >= 4`，OCR-bound 数量看似足够，但 `unique_ocr_count < max(3, ceil(target_count * 0.6))` |
| `too_many_lines_for_mllm_ticks` | `target_count >= 3` 且 `line_count > max(target_count + 6, ceil(target_count * 2.2))` |

### 更具体的异常几何 fallback

| reason | 触发条件 | 含义 |
| --- | --- | --- |
| `x_axis_selected_axis_severely_undercovered` / `y_axis_selected_axis_severely_undercovered` | `target_count >= 5`，`score < 5.0`，且 `line_count <= max(2, floor(target_count * 0.35))` 或 `labeled_count <= max(2, floor(target_count * 0.35))` 或 `ocr_bound_count <= max(1, floor(target_count * 0.25))` | 最终轴相对于目标 tick 严重覆盖不足 |
| `x_axis_numeric_duplicate_ocr_tick_binding` / `y_axis_numeric_duplicate_ocr_tick_binding` | numeric 轴；`target_count >= 5`；`duplicate_ocr_count >= 1`；且 `unique_ocr_count < target_count` | 同一个 OCR tick 被重复绑定，数值轴不可信 |
| `x_axis_numeric_selected_large_ocr_distance` / `y_axis_numeric_selected_large_ocr_distance` | numeric 轴；`target_count >= 8`；`max_ocr_distance > 8.0`；且 `max_ocr_distance > mean_ocr_distance * 4.0` | 存在明显 OCR 到网格线距离离群点 |
| `x_axis_category_semantic_tick_position_ambiguous` / `y_axis_category_semantic_tick_position_ambiguous` | 最终选择 `semantic_guide`；轴类型为 `category/mixed`；`target_count >= 8`；P2 `tick_supplement` 有效；`target_count < tick_line_count <= target_count + 3`；`tick_labeled_count >= target_count`；`tick_ocr_bound_count >= max(3, ceil(target_count * 0.65))`；`semantic_score - tick_score <= 10.0`；且 semantic `max_ocr_distance > 2.5` | P2 物理 tick 和 P3 语义 guide 都有一定依据，但位置不完全一致，系统不强行选择 |

### 例子

例子 1：线数严重不足。

MLLM 认为 y 轴应有 6 个 tick，但最终候选只生成了 1 条水平线：

```text
target_count = 6
line_count = 1
```

因为 `line_count < max(2, floor(6 * 0.45)) = 2`，候选会被标记为 `too_few_lines_for_mllm_ticks`。如果该候选最终被选中，就会触发 selected candidate invalid 相关 fallback。

例子 2：线数过多。

MLLM 认为 x 轴应有 5 个 tick，但候选生成了 18 条竖线：

```text
target_count = 5
line_count = 18
```

因为 `line_count > max(5 + 6, ceil(5 * 2.2)) = 11`，候选会被标记为 `too_many_lines_for_mllm_ticks`。

例子 3：OCR 距离离群。

某 numeric 轴有 11 个 tick，大多数 OCR label 离网格线很近，但其中一个 OCR label 到绑定线的距离为 9.5 px，平均距离只有 1.267 px：

```text
target_count = 11
max_ocr_distance = 9.5
mean_ocr_distance = 1.267
```

因为 `max_ocr_distance > 8.0` 且 `max_ocr_distance > mean_ocr_distance * 4.0`，触发 `numeric_selected_large_ocr_distance`。

### 论文表述

可以写成：

> Geometric fallback is triggered when the generated grid fails basic structural checks or when the selected candidate is inconsistent with the expected axis evidence. The system rejects candidates with no grid lines, too few labeled ticks, severe under-coverage of the expected tick sequence, excessive extra lines, duplicated OCR bindings, large OCR-to-grid distance outliers, or ambiguous competition between physical tick evidence and semantic-guide evidence.

## 简洁版本

如果论文正文只需要简洁说明，可以写：

> We use a generation-side fallback gate to prevent unreliable grid reconstructions from entering the success set. The gate contains three groups of checks. First, OCR/MLLM support checks prevent semantic-only MLLM labels from being treated as image-grounded geometric evidence. Second, repeated-label checks reject high-cardinality categorical or temporal axes whose repeated semantic labels cannot be uniquely bound to grid lines. Third, geometric sanity checks reject outputs with no usable axes, severe under-coverage, excessive extra lines, duplicated OCR bindings, large OCR-to-grid distance outliers, or ambiguous conflicts between physical tick evidence and semantic-guide evidence. All fallback decisions are made without using ground-truth annotations.
