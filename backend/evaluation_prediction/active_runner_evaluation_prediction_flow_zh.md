## 实验版机制边界更新（2026-07-01）

本文件描述 active runner 时，以模型三阶段为准：grid 固定 1 次，feedback 最多 2 次，amplifier 最多 3 次。最终预测和指标只来自这些阶段的模型输出；不再使用颜色面积、几何扫描等本地确定性读数作为 fallback 预测。允许调整提示词、裁剪范围和缩放策略，但不得新增论文机制外的模型调用或非模型预测结果。

# 当前 Active Runner 的 3.评估预测流程说明

本文说明系统当前真正执行的第 3 步“评估预测”流程。范围仅覆盖 active runtime path：

```text
backend/main.py
  -> backend/evaluation_prediction/service.py
  -> backend/evaluation_prediction/chart_modules/*/runner.py
```

旧版 `pie/flow.py`、`donut/flow.py` 仍保留在代码中，但当前 service 和 CLI 不调用它们；阅读和描述当前系统行为时，应以 `runner.py` 为准。

## 1. 入口

前端第 3 步触发后，后端进入 `backend/main.py` 的 `evaluate_processed_chart(chart_info)`。

执行前置条件：

- `chart_info["processed"]` 必须为真，也就是第 2 步加密处理已经完成。
- 如果是数据集预览样例，系统会先检查 `backend/data/dataset_preview_cache/` 下是否已有可用评估缓存。
- 如果命中缓存，直接返回缓存结果，不再重复调用 prediction runner。

未命中缓存时，系统会调用：

```python
prediction_results = await run_prediction_async(chart_info["chart_type"], eval_json_path)
```

其中 `eval_json_path` 是第 2 步生成的系统 JSON，通常是：

```text
backend/data/output/<chart_type>/*_image.json
backend/data/output/<chart_type>/*_image_ticks.json
```

重要边界：

- 第 3 步使用系统生成的 JSON、ticks、pixels、grid image、original image。
- 正常运行时不读取数据集 ground truth 作为输入。
- 如果 active runner 没有产生可用预测，后端会尝试 `system_cv_predictions` 作为系统侧 CV fallback。

## 2. 类型调度

调度逻辑在 `backend/evaluation_prediction/service.py`。

当前支持的预测类型：

```text
v_bar, h_bar, line, scatter, bubble, pie, donut, radar, rose
```

实验版不支持的处理对象：

```text
v_stacked_bar
h_stacked_bar
```

它们不会再被映射到 `v_bar` / `h_bar` 参与本实验。

各类型调用的 active runner：

| chart_type | active runner |
|---|---|
| `v_bar` | `chart_modules/v_bar/runner.py` |
| `h_bar` | `chart_modules/h_bar/runner.py` |
| `line` | `chart_modules/line/runner.py` |
| `scatter` | `chart_modules/scatter/runner.py` |
| `bubble` | `chart_modules/bubble/runner.py` |
| `pie` | `chart_modules/pie/runner.py` |
| `donut` | `chart_modules/donut/runner.py` |
| `radar` | `chart_modules/radar/runner.py` -> `polar_value.py` |
| `rose` | `chart_modules/rose/runner.py` -> `polar_value.py` |

## 3. 运行参数

通用重复次数来自 `backend/evaluation_prediction/common/runtime.py`：

```text
CHART_REPEAT_TIMES
```

默认值为 `3`。

Bar 类 amplifier 轮数来自：

```text
CHART_BAR_AMPLIFIER_ROUNDS
```

如果未设置，则回退到：

```text
CHART_AMPLIFIER_ROUNDS
```

默认值为 `3`。

注意：Radar/Rose 的 polar runner 使用 `get_repeat_times(default=1)`，因此它们默认每种图像提示只跑 1 次，而不是 3 次。

## 4. 通用输出结构

每个 runner 返回若干 chart-level summary，典型字段包括：

```text
chart_id
result_dir
record_count
object_count
predictions
```

后端会把所有 `chart_result["predictions"]` 展平成统一的 `predictions` 列表，并过滤掉没有具体数值的预测。

最终保存为 evaluation JSON：

```text
backend/data/results/<chart_id>_evaluation.json
```

或数据集预览缓存：

```text
backend/data/dataset_preview_cache/<sample_id>/<sample_id>_evaluation.json
```

最终 JSON 的核心结构：

```json
{
  "success": true,
  "mode": "prediction_extraction",
  "chart_id": "...",
  "chart_type": "...",
  "source_json": "...",
  "summary": {
    "object_count": 0,
    "chart_runs": 0,
    "system_cv_fallback": false
  },
  "predictions": [],
  "artifacts": [],
  "processed_json": {}
}
```

## 5. Bar 类：v_bar / h_bar

文件：

```text
chart_modules/v_bar/runner.py
chart_modules/h_bar/runner.py
```

启用的实验类型：

```python
EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("amplifier", "grid_with_grid"),
]
```

每个目标条形段的步骤：

1. 跑 `baseline + no_grid`，GT 实验模式固定 1 次。
2. 跑 `grid + grid_with_grid`，GT 实验模式固定 1 次。
3. 跑 `feedback + grid_with_grid`，最多 2 次。
4. feedback 第 2 轮及以后会把历史预测画成 overlay，再交给模型修正。
5. 跑 `amplifier + grid_with_grid`，最多 3 轮。
6. amplifier 每轮围绕上一轮预测值裁剪局部窗口。
7. 裁剪后先用颜色/目标可见性 prompt 检查目标条形段是否在 crop 内。
8. 如果 crop 不包含目标，最多尝试多个 offset；仍失败则提前停止 amplifier。
10. 最终选择预测时优先级为：

```text
geometry > amplifier > feedback > grid > baseline
```

因此，Bar 类不是“1 次 grid + 3 次 feedback + 3 次 amplifier”。默认是 baseline/grid/feedback 各 3 次，amplifier 默认 3 轮，但 amplifier 可能提前停止。

## 6. Line

文件：

```text
chart_modules/line/runner.py
```

启用的实验类型：

```python
EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("amplifier", "grid_with_grid"),
]
```

每个目标点的步骤：

1. 跑 `baseline + no_grid`，默认 3 次。
2. 跑 `grid + grid_with_grid`，默认 3 次。
3. 跑 `feedback + grid_with_grid`，默认 3 次。
4. feedback 会把历史预测点画到 grid 图上，让模型比较并修正。
5. 跑 `amplifier + grid_with_grid`，默认 3 次。
6. amplifier 围绕目标 x 类别和当前 y 估计裁剪局部窗口。
7. 裁剪后会调用 point-exists prompt 检查目标点是否可见，但当前实现只是记录检查结果，不会像 Bar 那样循环寻找 crop。
8. 最终选择预测优先级为：

```text
amplifier > feedback > grid > baseline
```

## 7. Scatter / Bubble

文件：

```text
chart_modules/scatter/runner.py
chart_modules/bubble/runner.py
```

启用的实验类型：

```python
EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "grid_with_grid"),
    ("feedback", "grid_with_grid"),
    ("feedback_crop_adaptive", "grid_with_grid"),
]
```

每个目标点的步骤：

1. 跑 `baseline + no_grid`，默认 3 次。
2. 跑 `grid + grid_with_grid`，默认 3 次。
3. 跑 `feedback + grid_with_grid`，默认 3 次。
4. feedback 会把上一轮预测位置画到 grid 图上。
5. 跑 `feedback_crop_adaptive + grid_with_grid`，默认 3 次。
6. `feedback_crop_adaptive` 不是名为 `amplifier` 的 prompt type，但它承担 zoom-in 局部裁剪功能。
7. 该阶段需要先有有效 feedback 预测；如果没有，则该轮记录 `(-1, -1)` 并跳过。
8. 生成 crop 前会估计点/气泡直径。
9. 每轮 adaptive crop 内部最多尝试 5 个 crop 尺寸或位置。
10. 每次 crop 后使用目标可见性 prompt 判断目标点是否在局部图内。
11. 最终选择预测优先级为：

```text
feedback_crop_adaptive > feedback > grid > baseline
```

所以 point 类有“zoom-in-like adaptive crop”，但不叫 `amplifier`，且每轮内部还有最多 5 次 crop attempt。

## 8. Pie

文件：

```text
chart_modules/pie/runner.py
```

当前 active runner 的启用类型：

```python
EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "with_grid"),
]
```

当前 active 流程：

1. 优先调用 whole-chart prompt，在 `with_grid` 图上一次性识别全部扇区。
2. 如果 whole-chart 结果缺少系统目标标签，则对缺失目标逐个运行 `_run_target`。
3. `_run_target` 只跑 `baseline` 和 `grid`，默认各 3 次。
5. 最后调用 `complete_circular_predictions` 补齐系统期望的标签顺序和预测结构。

当前 active `pie.runner.py` 没有 feedback 链，也没有 amplifier 链。

注意：`chart_modules/pie/flow.py` 中保留了旧的 `grid -> feedback -> amplifier` 实验代码，但 service 和 CLI 当前都不调用它。

## 9. Donut

文件：

```text
chart_modules/donut/runner.py
```

Donut active runner 比 Pie 更接近旧式 refined chain。

整体步骤：

1. 先用 whole-chart prompt 在 `with_grid` 图上提取全部 ring sector。
2. 对每个系统目标标签运行 `_run_refined_target`。
3. 如果 whole-chart 已经给出该标签的初始预测，则先记录一条 `whole_chart` 记录。
4. refined feedback chain 的轮数为 `max(1, CHART_REPEAT_TIMES)`，默认 3。
5. 第 1 轮 prompt type 为 `grid`。
6. 第 2-3 轮 prompt type 为 `feedback`，使用上一轮角度预测绘制 feedback 图。
7. 如果没有有效角度预测，则 refined chain 提前停止。
8. 有有效角度后，进入 amplifier，固定尝试 3 轮：

```python
for amp_round in range(1, 4):
```

9. amplifier 每轮在 `no_grid` 图上围绕当前角度预测裁剪扇区局部图。
10. 模型在 crop 图上重新估计起止角度。
11. 如果 crop 或模型输出失败，amplifier 提前停止。
12. 最后生成 `amplifier_pct` 作为最终百分比记录。

最终选择预测优先级为：

```text
amplifier_pct > amplifier > feedback > grid > whole_chart > baseline
```

## 10. Radar / Rose

文件：

```text
chart_modules/radar/runner.py
chart_modules/rose/runner.py
chart_modules/polar_value.py
```

`radar.runner` 和 `rose.runner` 只是轻量 wrapper，实际逻辑在 `polar_value.py`。

默认重复次数：

```python
repeat_times = get_repeat_times(default=1)
```

目标级流程：

1. 对每个目标点运行 `baseline + no_grid`，默认 1 次。
2. 对每个目标点运行 `grid + grid_with_grid`，默认 1 次。
3. prompt 会包含系统生成的径向 tick、角度 tick、角度位置、颜色信息等。
4. 不执行 feedback overlay。
5. 不执行 zoom-in/amplifier crop。

Whole-chart fallback：

1. 如果目标级预测没有可用结果，则调用 `_run_whole_chart`。
2. whole-chart fallback 先尝试 `grid + grid_with_grid`。
3. 如果没有得到预测，再尝试 `baseline + no_grid`。
4. 有结果后用于生成最终 predictions。

最终选择预测时优先使用 `grid`，其次是 `baseline`。

## 11. 结果保存和后端 fallback

Runner 自己会在：

```text
backend/evaluation_prediction/results/<chart_type>/<chart_id>/
```

保存中间结果，例如：

```text
experiment_results.csv
selected_predictions.csv
predictions.json
run_summary.json
feedback / crop / amplifier 图片
```

然后 `backend/main.py` 会把 runner 的 `predictions` 汇总到 step 3 的 evaluation JSON。

如果 runner 报错或没有可用数值预测：

1. 后端捕获异常并记录 `prediction_runner_error`。
2. 后端尝试 `system_cv_predictions`。
3. 如果 CV fallback 产生预测，则设置：

```json
"system_cv_fallback": true
```

4. 如果仍无预测，会保存空预测列表，并说明 prediction runner 没有系统生成目标或结果。

## 12. 快速对照表

| 类型 | grid | feedback | zoom-in / amplifier | 当前最终优先级 |
|---|---:|---:|---:|---|
| `v_bar` / `h_bar` | 默认 3 次 | 默认 3 次 | 默认 3 轮，可提前停 | `geometry > amplifier > feedback > grid > baseline` |
| `line` | 默认 3 次 | 默认 3 次 | 默认 3 次 | `amplifier > feedback > grid > baseline` |
| `scatter` / `bubble` | 默认 3 次 | 默认 3 次 | `feedback_crop_adaptive` 默认 3 次，每次最多 5 个 crop attempt | `feedback_crop_adaptive > feedback > grid > baseline` |
| `pie` | whole-chart 优先；缺失目标时 baseline/grid 默认各 3 次 | 无 active feedback | 无 active amplifier | `grid > baseline`，并有 whole-chart/color-area 补齐 |
| `donut` | refined chain 第 1 轮 grid | refined chain 默认第 2-3 轮 feedback | 固定最多 3 轮 amplifier | `amplifier_pct > amplifier > feedback > grid > whole_chart > baseline` |
| `radar` / `rose` | 默认 1 次 | 无 | 无 | `grid > baseline`，失败时 whole-chart fallback |

## 13. 阅读建议

如果只想理解当前系统行为，建议按下面顺序读：

1. `backend/main.py` 的 `evaluate_processed_chart`
2. `backend/evaluation_prediction/service.py`
3. `backend/evaluation_prediction/common/runtime.py`
4. 对应图表类型的 `chart_modules/<type>/runner.py`
5. 对应类型的 `data.py`、`prompts.py`、`visual.py`

不要优先读 `pie/flow.py` 或 `donut/flow.py` 来判断当前 active 行为；它们主要是历史/参考实验流。
