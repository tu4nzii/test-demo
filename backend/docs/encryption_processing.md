# 加密处理逻辑

本文档说明后端“加密处理”功能的业务目标、代码入口、直角坐标系处理流程、极坐标处理流程、输出结构和 JSON 边界。

## 1. 功能定位

加密处理发生在用户完成上传和图表类型识别之后。它的核心目标是基于原始图表图片生成带网格和加密刻度的可视化图片。

当前最重要的业务边界：

- 直角坐标系加密必须基于原始图片。
- 上传 JSON 不参与坐标轴检测、刻度位置检测、刻度值识别或加密刻度生成。
- 数值轴只加密一轮，也就是在相邻两个可见 tick 之间插入一个中点。
- 文字轴不做数值加密。

## 2. 代码入口

主要文件：

| 文件 | 作用 |
| --- | --- |
| `backend/main.py` | `/api/process/` 接口和处理流程编排 |
| `backend/type_detection/chart_processor.py` | 根据图表类型选择处理器 |
| `backend/Grid_generation/grid_generation.py` | 直角坐标系加密主流程 |
| `backend/Grid_generation/function_calling/axis/` | 坐标轴检测与推断 |
| `backend/Grid_generation/function_calling/ticks/` | tick 检测、过滤、修正 |
| `backend/Grid_generation/function_calling/label/` | 轴标签和轴类型识别 |
| `backend/Grid_generation/function_calling/color/` | 图例或系列颜色识别 |
| `backend/demo_rose/` | 玫瑰图加密处理 |
| `backend/demo_radar/` | 雷达图加密处理 |

调用链：

```text
POST /api/process/?chart_id=...
  -> process_chart_image()
    -> ChartProcessorFactory.create_processor(chart_type)
    -> processor.encode_image(image_path, output_dir)
    -> save_axis_data()
    -> enrich_generated_json()
    -> 返回 encrypted_image_url
```

## 3. 处理器选择

`ChartProcessorFactory` 根据图表类型分发：

| 图表类型 | 处理器 | 主流程 |
| --- | --- | --- |
| `rose` | `RoseChartProcessor` | 极坐标玫瑰图处理 |
| `radar` | `RadarChartProcessor` | 极坐标雷达图处理 |
| 其它已注册直角类型 | `CartesianChartProcessor` | `Grid_generation.process_chart()` |

`CartesianChartProcessor.encode_image()` 调用：

```text
Grid_generation.grid_generation.process_chart(image_path, output_dir)
```

并从返回结果中取：

```text
encrypted_grid_path
```

作为前端展示的加密图片。

## 4. 直角坐标系加密流程

直角坐标系图表包括 `v_bar`、`h_bar`、`line`、`scatter`、`bubble` 等。主函数是：

```text
backend/Grid_generation/grid_generation.py
  -> process_chart(image_path, output_dir)
```

处理步骤：

1. 读取原始图片。
2. 转为灰度图。
3. 使用 Canny + Hough 检测候选直线。
4. 合并相似线段。
5. 推断 X 轴和 Y 轴。
6. 沿坐标轴附近扫描 tick 短线。
7. 合并并过滤 tick。
8. 从图片中识别可见轴标签和轴类型。
9. 提取图表系列颜色。
10. 使用 `refine_tick_pixels()` 对 tick 像素位置做修正。
11. 将识别出的 tick 值与像素位置对齐。
12. 对数值轴生成一轮加密 tick。
13. 为新增加密 tick 生成中间像素位置。
14. 绘制基础网格图片。
15. 绘制加密网格图片。
16. 保存 tick、像素、轴类型、颜色和输出路径。

## 5. 坐标轴检测

坐标轴检测由以下模块完成：

```text
function_calling.axis.detect_lines.detect_candidate_lines()
function_calling.axis.merge_lines.merge_similar_lines()
function_calling.axis.infer_axes.infer_axes_from_lines()
```

基本逻辑：

- 先在灰度图上做边缘检测。
- 用 HoughLinesP 检出候选直线。
- 合并角度和位置接近的线段。
- 从候选线段中选择最可能的水平 X 轴和垂直 Y 轴。
- 如果推断失败，主流程会尝试用“底部水平线”和“左侧垂直线”做兜底。

当前加密流程和评估流程对齐后的关键参数包括：

```text
canny_threshold1 = 30
canny_threshold2 = 100
hough_threshold = 15
min_length = 15
max_gap = 15
scan_range = 20
```

## 6. Tick 检测与修正

Tick 检测由：

```text
function_calling.ticks.detect_ticks.scan_pixels_for_ticks()
function_calling.ticks.filter_ticks.filter_ticks()
function_calling.ticks.refine_ticks.refine_tick_pixels()
```

共同完成。

处理逻辑：

- X 轴：沿 X 方向扫描轴线上下的暗色短线。
- Y 轴：沿 Y 方向扫描轴线左右的暗色短线。
- 对连续暗色像素进行分组，得到 tick 短线。
- 合并相近 tick。
- 过滤明显噪声。
- 使用图片识别出的轴类型和 tick 数量做修正。

`refine_tick_pixels()` 的原则：

- 不读取 ground truth JSON。
- 不把稠密网格 tick 当作可见 tick。
- 只融合图像检测到的 tick 像素、轴几何和模型识别出的可见 tick 标签数量。
- 对水平条形图等特殊情况做轴类型和类别轴修正。

## 7. 轴标签和轴类型识别

轴标签识别由：

```text
function_calling.label.extract_tick_labels_with_llm.extract_tick_labels_with_llm()
```

完成。

它会基于原始图片或轴区域截图识别：

- `x_ticks`
- `y_ticks`
- `x_axis_type`
- `y_axis_type`

轴类型一般归为：

- 数值轴：tick 是数字，可以插入中间值。
- 文字轴：tick 是类别、年份、月份、公司名等文本，不做数值插值。

如果模型未能返回可用 tick，开发环境下会使用位置序号做 fallback，保证流程可以继续运行。但这只是兜底，不代表业务上的真实刻度。

## 8. 一轮加密规则

加密函数：

```text
generate_encrypted_ticks(original_ticks, is_numeric_axis=True)
```

规则：

- 数值轴：保留原 tick，并在相邻两个 tick 之间插入一个中点。
- 文字轴：直接返回原 tick，不插入新 tick。
- 不递归加密，不继续细分新生成的中点。

示例：

```text
原始可见 tick:
0, 10, 20, 30

一轮加密后:
0, 5, 10, 15, 20, 25, 30
```

如果继续生成 `2.5`、`7.5`、`12.5` 等，就属于二次加密，不符合当前需求。

对应像素位置规则：

- 原始 tick 使用检测到的原始像素位置。
- 新增加密 tick 使用相邻两个原始像素位置的中点。

## 9. 图片绘制规则

基础网格图：

```text
*_grid.png
```

内容：

- 原始图。
- 检测出的 X/Y 坐标轴。
- 基于原始可见 tick 的基础网格线。

加密网格图：

```text
*_with_grid.png
```

内容：

- 原始图。
- 基础网格。
- 新增加密 tick 对应的中间网格线。
- 新增加密 tick 的红色文本标签。

绘制原则：

- 原始 tick 不重复绘制红色标签。
- 只给新增的中间 tick 绘制加密标签。
- 文字轴不生成新增加密标签。

## 10. 输出结构

直角坐标系处理完成后返回并保存类似结构：

```json
{
  "chart_id": "line_000",
  "x_ticks": [0, 10, 20],
  "y_ticks": [0, 5, 10],
  "x_pixels": [80, 180, 280],
  "y_pixels": [320, 220, 120],
  "x_ticks_encrypted": [0, 5, 10, 15, 20],
  "y_ticks_encrypted": [0, 2.5, 5, 7.5, 10],
  "x_pixels_encrypted": [80, 130, 180, 230, 280],
  "y_pixels_encrypted": [320, 270, 220, 170, 120],
  "x_axis_type": "数值轴",
  "y_axis_type": "数值轴",
  "image_path": "...",
  "basic_grid_path": "..._grid.png",
  "encrypted_grid_path": "..._with_grid.png",
  "colors": []
}
```

`main.py` 之后会调用 `enrich_generated_json()` 给结果补充：

- `chart_id`
- `chart_type`
- `coordinate_system`
- `image_paths`
- 可选的原始数据 `data`

这里的原始数据只用于评估和结果展示，不参与已经完成的加密计算。

## 11. 极坐标处理

极坐标图表当前包括：

- `rose`
- `radar`

处理入口：

```text
RoseChartProcessor.encode_image()
RadarChartProcessor.encode_image()
```

它们分别调用 `demo_rose` 和 `demo_radar` 下的编码器。当前这部分保留了合作方实现的处理链路，可能仍对配套 JSON 或历史输出文件有兼容依赖。

业务方向上，用户入口不应强制要求 JSON；如果极坐标算法内部仍需要 JSON，需要在后续改造中把它收敛为可选评估输入或内部推断结果，而不是前端必填项。

## 12. 与 JSON 的关系

直角坐标系加密明确不基于上传 JSON。

允许使用 JSON 的位置：

- 上传阶段保存 JSON 文件路径。
- 加密完成后，把 JSON 中的原始数据写入结果文件，作为评估 ground truth。
- `/api/evaluate/` 读取结果文件做预测评估。

禁止使用 JSON 的位置：

- 坐标轴检测。
- tick 像素位置检测。
- tick 标签识别。
- 轴类型判断。
- 加密 tick 生成。
- 加密网格线生成。

一句话：JSON 可以解释结果，但不能生成加密结果。

