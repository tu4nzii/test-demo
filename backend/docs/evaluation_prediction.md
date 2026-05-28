# 评估预测处理逻辑

本文档说明后端“评估预测”功能的业务目标、代码入口、运行时评估、离线全量评估和结果口径。

## 1. 功能定位

评估预测用于回答两个问题：

1. 当前图表处理结果是否具备可评估的数据结构。
2. 如果存在预测值和 ground truth，预测误差是多少。

它不负责生成加密网格。加密网格由 `/api/process/` 完成，评估预测在加密处理之后执行。

重要边界：

- 业务加密不依赖 JSON。
- 评估可以使用 JSON 作为 ground truth。
- JSON tick 数量不等于前端可见 tick 数量时，不能直接判定前端加密有问题。

## 2. 代码入口

运行时 API 入口：

| 文件 | 作用 |
| --- | --- |
| `backend/main.py` | `/api/evaluate/` 接口和评估流程编排 |
| `backend/type_detection/chart_processor.py` | 调用处理器的 `evaluate()` |
| `backend/evaluation/service.py` | 评估主逻辑 |
| `backend/evaluation/normalizer.py` | ground truth 和 prediction 数据归一化 |
| `backend/evaluation/metrics.py` | 误差指标计算 |

离线全量评估入口：

| 文件 | 作用 |
| --- | --- |
| `backend/evaluation/scripts/evaluate_cv_mllm_ticks.py` | 批量评估 CV + MLLM tick 检测效果 |
| `backend/evaluation/results/` | 评估输出目录 |

当前保留的全量评估结果：

```text
backend/evaluation/results/cv_mllm_current_flow_gpt4omini_full_reusedcache.json
backend/evaluation/results/problem_charts_current_flow_actionable.csv
backend/evaluation/results/problem_charts_current_flow_actionable.md
```

## 3. 运行时 API 调用链

前端点击“进行预测”时调用：

```text
POST /api/evaluate/?chart_id=...
```

后端调用链：

```text
evaluate_chart()
  -> get_chart(chart_id)
  -> evaluate_processed_chart(chart_info)
    -> 检查是否已经 process
    -> resolve_eval_json(chart_info)
    -> ChartProcessorFactory.create_processor(chart_type)
    -> processor.evaluate(eval_json_path)
      -> evaluation.evaluate_chart_data()
    -> processor.save_evaluation_results()
    -> 返回 results_url
```

前端再通过：

```text
GET /api/results/{filename}
```

读取评估结果。

## 4. 评估数据来源

`resolve_eval_json()` 会按顺序寻找或生成评估数据：

1. 优先找处理输出目录下以图片 stem 命名的 JSON。
2. 再找以 `chart_id` 命名的 JSON。
3. 如果都不存在，调用处理器的 `process_data()` 生成。

直角坐标系的 `process_data()` 会重新调用图片处理流程，拿到：

- `x_ticks`
- `y_ticks`
- `x_axis_type`
- `y_axis_type`
- `colors`
- `basic_grid_path`
- `encrypted_grid_path`

如果上传时有 JSON，则把其中的 `data_points` 或 `data` 写入评估数据，作为 ground truth。

这一步的重点是：JSON 只进入评估数据，不回流到加密生成。

## 5. 数据归一化

`backend/evaluation/normalizer.py` 负责把不同格式的数据整理成统一结构。

Ground truth 来源优先级：

```text
ground_truth
data_points
data
```

Prediction 来源优先级：

```text
predictions
prediction
predicted_data
prediction_data
extracted_data
estimated_data
```

归一化逻辑：

- 嵌套 dict 会被展开成路径 ID。
- 标量 list 会被当作一个向量值。
- 嵌套 list 会按索引继续展开。

示例：

```json
{
  "data": {
    "series_a": {
      "point_1": 10
    }
  }
}
```

会归一化为：

```text
series_a / point_1 -> 10
```

## 6. 运行时评估模式

`evaluate_chart_data()` 有两种模式。

### 6.1 prediction_evaluation

当评估数据中存在 prediction 字段时进入此模式。

输出内容包括：

- `total_items`：ground truth 条目数。
- `matched_items`：预测结果中匹配到的条目数。
- `missing_items`：缺失预测数。
- `extra_predictions`：多余预测数。
- `coverage`：覆盖率。
- `avg_mae`：平均绝对误差。
- `avg_relative_error`：平均相对误差。
- `records`：逐条预测对比。

### 6.2 data_readiness

当没有 prediction 字段时进入此模式。

此时系统不计算真实预测误差，而是报告当前处理结果是否具备评估条件，例如：

- 是否有 ground truth。
- 是否生成基础网格图。
- 是否生成加密网格图。
- X/Y tick 数量。
- 极坐标 r/theta tick 数量。
- 颜色数量。

这个模式适合前端展示“处理结果结构是否完整”，但不能当作模型预测准确率。

## 7. 指标计算

指标实现位于 `backend/evaluation/metrics.py`。

当前支持：

- `absolute_error()`：绝对误差。
- `relative_error()`：相对误差。
- `vector_mae()`：向量平均绝对误差。
- `vector_relative_error()`：向量平均相对误差。
- `safe_mean()`：忽略空值的均值。
- `round_metric()`：指标保留固定小数位。

标量示例：

```text
ground_truth = 100
predicted = 92
mae = 8
relative_error = 0.08
```

向量示例：

```text
ground_truth = [10, 20]
predicted = [13, 18]
vector_mae = (3 + 2) / 2 = 2.5
```

## 8. 离线全量评估

离线评估脚本：

```text
backend/evaluation/scripts/evaluate_cv_mllm_ticks.py
```

它用于批量检查当前图片识别流程的效果，尤其是：

- 坐标轴是否检测成功。
- tick 像素位置是否接近 ground truth。
- MLLM 识别的 tick 值是否接近 ground truth。
- 轴类型是否识别正确。
- 像素和值是否能配对成功。

常用命令：

```powershell
python backend\evaluation\scripts\evaluate_cv_mllm_ticks.py --types line scatter bubble v_bar h_bar --limit 0 --cache-only --dataset-id backend_charts_gpt4omini --output backend\evaluation\results\cv_mllm_current_flow.json
```

说明：

- `--limit 0` 表示每类图全量。
- `--cache-only` 表示只使用已有 MLLM 缓存，不重新调用模型。
- 去掉 `--cache-only` 才会在缓存缺失时调用模型接口。

## 9. 离线评估流程

离线脚本对每张图片执行：

1. 从 `backend/charts/{chart_type}` 读取原图。
2. 读取同名 JSON 作为评估 ground truth。
3. 使用与当前加密流程对齐的 CV 参数检测坐标轴。
4. 扫描并过滤 tick 像素位置。
5. 调用或读取缓存中的 MLLM tick 标签识别结果。
6. 使用 `refine_tick_pixels()` 修正 tick 像素。
7. 计算像素召回、值召回、配对召回和轴类型准确率。
8. 输出 summary 和逐图 rows。

与当前加密流程对齐的关键点：

- 使用更宽松的线检测参数。
- `scan_range = 20`。
- 使用 `filter_ticks()`。
- `refine_tick_pixels()` 不把可见 tick 扩展成 JSON 中的稠密 tick。

## 10. 结果口径

全量评估中有两类“问题”需要区分。

### 10.1 指标型问题

例如：

```text
x_tick_count_mismatch
y_tick_count_mismatch
value_recall_lt_0.50
pair_recall_lt_0.50
```

这类问题不一定代表前端视觉有 bug。原因是 JSON 里可能包含更密的内部刻度、网格值或数据结构，而前端业务只要求加密图片中可见的 tick。

所以，仅因为 JSON tick 数量和可见 tick 数量不同，不应直接判定当前加密错误。

### 10.2 业务可疑问题

当前更应该关注：

```text
x_axis_type_wrong
y_axis_type_wrong
x_numeric_ticks_lt_2
y_numeric_ticks_lt_2
cv_axis_failed
```

这些问题可能影响前端实际加密显示，例如：

- 数值轴被识别成文字轴，导致不加密。
- 文字轴被识别成数值轴，导致错误插值。
- 数值轴 tick 少于 2 个，无法插入中点。
- 坐标轴检测失败，无法生成网格。

当前整理后的可疑图表清单保存在：

```text
backend/evaluation/results/problem_charts_current_flow_actionable.csv
backend/evaluation/results/problem_charts_current_flow_actionable.md
```

## 11. 与 JSON 的关系

评估预测允许使用 JSON，但使用方式必须清楚：

允许：

- 用上传 JSON 中的 `data_points` 或 `data` 作为 ground truth。
- 用样本图同名 JSON 作为离线评估 ground truth。
- 用 JSON 计算覆盖率、误差、召回率。

禁止：

- 用 JSON 参与业务加密。
- 用 JSON 覆盖图片检测出来的 tick 像素。
- 用 JSON 覆盖图片识别出来的 tick 标签。
- 把 JSON 中的稠密 tick mismatch 直接当作前端加密 bug。

评估的职责是“检查”，不是“生成”。

