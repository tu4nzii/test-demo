# 直角系加密处理链路与代码模块说明

本文档介绍 `backend/Grid_generation` 中当前直角坐标系图表的处理链路。直角系包括：

- `v_bar`
- `h_bar`
- `v_stacked_bar`
- `h_stacked_bar`
- `line`
- `scatter`
- `bubble`

`pie`、`donut`、`radar`、`rose` 属于极坐标或角度网格链路，不是本文档重点。

## 1. 外部入口

直角系处理不是由前端直接调用 `Grid_generation`，而是通过后端统一分发进入。

调用链路：

```text
backend/main.py
  -> process_chart_image(chart_info)
  -> ChartProcessorFactory.create_processor(chart_type)
  -> CartesianChartProcessor.encode_image(...)
  -> Grid_generation.grid_generation.process_chart(...)
```

对应代码：

- `backend/main.py`
  - `register_chart(...)`：保存上传图像，并保存上传阶段的 `axis_repair` 先验。
  - `process_chart_image(...)`：根据图表类型创建 processor，调用加密处理。
  - `enrich_generated_json(...)`：把处理产物合并进系统生成 JSON。
- `backend/type_detection/chart_processor.py`
  - `CartesianChartProcessor.encode_image(...)`：直角系统一入口。
  - `CartesianChartProcessor.find_axis(...)`：需要轴信息时临时运行一次处理链路。
- `backend/Grid_generation/grid_generation.py`
  - `process_chart(...)`：直角系真正的核心处理函数。

## 2. Grid_generation 目录结构

当前目录中与直角系主流程关系最密切的是：

```text
backend/Grid_generation/
  grid_generation.py
  config.py
  test_grid.py
  function_calling/
    axis/
      detect_lines.py
      merge_lines.py
      infer_axes.py
      thickness_detection.py
    ticks/
      detect_ticks.py
      filter_ticks.py
      correct_ticks.py
      fit_ticks.py
      tick_thickness.py
      extract_ticks_with_llm.py
    label/
      extract_tick_labels_with_llm.py
      recognize_tick_labels.py
      filter_ticks_with_labels.py
      generate_ticks_from_labels.py
      recognize_tick_text_box.py
    color/
      extract_chart_colors.py
    image/
      draw_grid_from_ticks.py
  utils/
    image_io.py
    geometry.py
    drawing.py
```

说明：

- `grid_generation.py` 是现在直角系的编排层和主要算法承载文件。
- `function_calling/axis` 提供线段检测、线段合并、坐标轴推断等底层 CV 能力。
- `function_calling/ticks` 提供短 tick 扫描、过滤、修正等能力。
- `function_calling/label` 提供 MLLM 读取 tick 文本和柱端数值标签的能力。
- `function_calling/color` 提供图例颜色、点图对象和系列颜色读取。
- `function_calling/image` 主要是早期网格绘制工具，目前主流程更多使用 `grid_generation.py` 内部的 `draw_basic_grid` 和 `draw_encrypted_grid`。
- `utils` 是图像读写和辅助工具。
- `data/llm_cache`、`__pycache__` 是运行缓存，不属于代码模块。

## 3. process_chart 的主流程

核心函数：

```python
process_chart(
    image_path,
    output_dir,
    chart_type_override=None,
    chart_id_override=None,
    axis_repair_hint=None,
)
```

它完成一张直角系图表从原图到 `_grid.png`、`_with_grid.png`、`_ticks.json` 的完整过程。

主流程可以分为 10 个阶段。

## 4. 阶段一：加载图像与基础初始化

位置：

- `grid_generation.py::process_chart`

主要职责：

- 处理 Windows 中文路径，使用 `np.fromfile + cv2.imdecode` 读取图像。
- 初始化日志。
- 根据 `chart_type_override` 固定当前图表类型。
- 规范化上传阶段传入的 `axis_repair_hint`。

相关函数：

- `normalize_axis_repair_hint(...)`
- `axis_repair_enabled(...)`
- `_point_chart_grid_hint(...)`

`axis_repair_hint` 是当前真实图适配的重要入口。它来自上传阶段 MLLM，里面包含：

- 是否缺 X/Y 轴。
- 是否缺 X/Y tick。
- 轴角色是 numeric、category、date 还是 unknown。
- Y 轴是否在右侧。
- 绘图区是显式轴、弱轴、仅网格还是无轴。
- 是否存在背景网格。
- 上传阶段读到的 tick 文本先验。

## 5. 阶段二：线段检测与坐标轴初判

调用模块：

- `function_calling/axis/detect_lines.py`
- `function_calling/axis/merge_lines.py`
- `function_calling/axis/infer_axes.py`

核心函数：

- `detect_candidate_lines(...)`
- `merge_similar_lines(...)`
- `infer_axes_from_lines(...)`

处理逻辑：

1. 灰度图上做 Canny 边缘检测。
2. 用 HoughLinesP 检测候选线段。
3. 合并相邻、平行、投影重叠的线段。
4. 从长水平线和长垂直线中推断 X/Y 轴。
5. 优先选择能形成真实左下角或右侧轴关系的轴线组合。

这一阶段提供初始 `x_axis`、`y_axis`，但不保证完全正确。后面会继续基于图表类型、背景网格和 MLLM 先验修正。

## 6. 阶段三：缺轴、弱轴、右侧轴修复

位置：

- `grid_generation.py`

关键函数：

- `_mark_missing_axis_from_cv(...)`
- `repair_missing_axes(...)`
- `infer_line_axes_from_visual_structure(...)`
- `infer_point_axes_from_visual_structure(...)`
- `refine_point_chart_axes_from_gridlines(...)`
- `refine_point_chart_axes_for_bootstrap(...)`
- `refine_explicit_point_axes_from_strokes(...)`
- `refine_vbar_axes_from_grid_bounds(...)`
- `refine_vbar_positive_axes_from_plot_bounds(...)`

处理逻辑：

- 如果 CV 找不到轴，会把这个事实合并到 `axis_repair_hint` 中。
- 如果 MLLM 判断为 `weak_axes`、`grid_only` 或 `no_axes`，允许进入补轴逻辑。
- 对 bar 系列，优先利用柱体包围盒推断轴线范围。
- 对 `v_bar` 且 `y_axis_position=right` 的图，把 Y 轴放在右侧。
- 对 scatter/bubble，优先从背景网格、点云范围和视觉边界推断绘图区。
- 对 line 图，如果显式轴不完整，会从折线所在绘图区结构补出轴。

保护原则：

- 正常显式轴图不会主动补轴。
- 补轴逻辑必须由 MLLM 先验或 CV 失败触发。
- 修复结果会记录到 `axis_repair` 的 `repair_applied` 中，便于后续排查。

## 7. 阶段四：tick 短线检测与过滤

调用模块：

- `function_calling/ticks/detect_ticks.py`
- `function_calling/ticks/filter_ticks.py`
- `function_calling/axis/merge_lines.py`

核心函数：

- `scan_pixels_for_ticks(...)`
- `merge_similar_lines(...)`
- `filter_ticks(...)`

处理逻辑：

1. 沿 X/Y 轴扫描短黑线，得到原始 tick 候选。
2. 合并相邻 tick 线段。
3. 过滤明显不合理的 tick。
4. 如果 tick 不足，根据图表类型进入不同兜底：
   - line：从网格线或轴线等距点 bootstrap。
   - scatter/bubble：从背景网格投影或绘图区边界 bootstrap。
   - bar：从柱体中心合成类别 tick。

相关函数：

- `bootstrap_point_chart_tick_pixels(...)`
- `synthesize_tick_pixels_for_missing_axis(...)`
- `_synthetic_bar_tick_pixels(...)`
- `_cluster_bar_category_pixels(...)`
- `ticks_from_pixels(...)`

## 8. 阶段五：MLLM 读取 tick 标签

调用模块：

- `function_calling/label/extract_tick_labels_with_llm.py`

核心函数：

- `extract_tick_labels_with_llm(...)`
- `extract_axis_ticks_with_llm(...)`
- `build_tick_extraction_prompt(...)`
- `parse_llm_response(...)`

处理逻辑：

- 分别读取 X/Y 轴 tick 文本。
- 返回 `x_ticks`、`y_ticks`、`x_axis_type`、`y_axis_type`。
- 使用缓存避免重复调用模型。
- prompt 会根据图表类型调整规则：
  - h_bar 的 Y 轴只读类别标签。
  - scatter/bubble 忽略色条、尺寸图例和说明文字。
  - line 忽略数据点旁标注和右侧系列标签。
  - grouped bar 忽略年份、分组标题、图例等次级标签。

输出会进入后续 tick-value 与 tick-pixel 绑定阶段。

## 9. 阶段六：图例颜色和对象信息读取

调用模块：

- `function_calling/color/extract_chart_colors.py`

核心函数：

- `extract_chart_series_color(...)`
- `extract_point_chart_items(...)`

处理逻辑：

- 对 scatter/bubble，读取点图对象、标签、颜色等信息。
- 对 bar/line，读取图例系列颜色。
- 结果写入 `colors` 字段，并进入系统生成 JSON。

这部分不直接决定加密 tick，但会影响评估预测阶段的对象展示和系列识别。

## 10. 阶段七：tick 文本与像素绑定、校验和修正

位置：

- `grid_generation.py`

关键函数：

- `_apply_upload_numeric_axis_prior(...)`
- `coerce_chart_axis_numeric_ticks(...)`
- `recover_bar_value_axis_pixels_from_grid(...)`
- `select_projected_tick_pixels_for_values(...)`
- `bind_noisy_numeric_ticks_to_labels(...)`
- `bind_noisy_numeric_bar_ticks_to_labels(...)`
- `snap_numeric_ticks_to_visual_grid(...)`
- `repair_point_tick_span_from_plot_bounds(...)`
- `repair_suspicious_vbar_value_axis(...)`

处理逻辑：

1. 把 MLLM 读到的 tick 文本与 CV 得到的 tick 像素匹配。
2. 判断轴类型是 `数值轴` 还是 `文字轴`。
3. 对数值轴，保留数字、百分比、带单位数字等可解析内容。
4. 对文字轴，保留类别文本，不参与数值插值。
5. 如果当前 MLLM tick 明显不可靠，才使用上传阶段 `axis_tick_labels` 作为先验。
6. 对候选 tick 像素过多的情况，按 MLLM tick 数量重新绑定。
7. 对背景网格明显可用的 scatter/bubble，枚举候选网格线组合，尝试线性和对数映射，选择误差最小的一组。
8. 对轻微偏移的 tick，snap 到附近可见网格线。

这一步是鲁棒性的核心：不是简单按顺序绑定，而是结合轴线、背景网格、tick 值、像素候选、图表类型共同校验。

## 11. 阶段八：bar 特殊兜底

位置：

- `grid_generation.py`
- `function_calling/label/extract_tick_labels_with_llm.py`

关键函数：

- `extract_bar_value_labels_with_llm(...)`
- `_bar_value_axis_ticks_from_data_labels(...)`
- `_nice_numeric_ticks_from_data_labels(...)`
- `_nice_diverging_ticks_from_data_labels(...)`

使用场景：

- 柱形图没有可见数值轴 tick。
- 但柱体末端或顶部有数值标签。

处理逻辑：

1. MLLM 只读取柱端/柱顶数字，不读取轴 tick、类别、图例或标题。
2. 根据柱体长度/高度和数值标签估计数值轴映射。
3. 生成一组合理的数值 tick。
4. 再进入正常加密插值。

这避免了真实柱形图缺 tick 时直接失败。

## 12. 阶段九：生成加密 tick 和加密像素

位置：

- `grid_generation.py`

关键函数：

- `axis_scale_from_ticks_and_pixels(...)`
- `generate_encrypted_ticks(...)`
- `generate_encrypted_pixels` 相关内联逻辑

处理逻辑：

- 只对数值轴加密。
- 文字轴直接保留原 tick，不插入中间类别。
- 线性轴：相邻 tick 取算术中点。
- 对数轴：相邻 tick 取几何中点。
- X/Y 像素位置同步插入中间点。

结果字段：

- `x_ticks_encrypted`
- `y_ticks_encrypted`
- `x_pixels_encrypted`
- `y_pixels_encrypted`
- `x_axis_scale`
- `y_axis_scale`

## 13. 阶段十：绘制基础网格和加密网格

位置：

- `grid_generation.py`

关键函数：

- `draw_basic_grid(...)`
- `draw_encrypted_grid(...)`
- `_draw_tick_grid_line(...)`
- `_active_grid_line_color(...)`

输出图：

- `{chart_id}_grid.png`
- `{chart_id}_with_grid.png`

绘制策略：

- `_grid.png`：绘制原始 tick 对应的基础网格。
- `_with_grid.png`：在基础网格基础上，只额外绘制插入的加密 tick 网格线和加密文本。

当前第二步加密网格参数：

- 网格颜色：`#cccccc`
- 测试复查颜色：环境变量 `CHART_GRID_REVIEW_COLOR=green` 时使用绿色
- 线宽：`1px`
- 虚线：`2px` 实线 + `2px` 空白循环
- 绘制方式：OpenCV `cv2.LINE_AA`

加密文本：

- 只显示插入的中间 tick。
- 使用红色文本。
- 使用半透明白底降低遮挡。
- Y 轴在右侧时，文本优先放在右侧。

## 14. 输出 JSON

位置：

- `grid_generation.py::process_chart`
- `backend/main.py::merge_tick_sidecar`
- `backend/main.py::enrich_generated_json`

`process_chart` 会保存 `{chart_id}_ticks.json`，核心字段包括：

```json
{
  "chart_id": "...",
  "x_ticks": [],
  "y_ticks": [],
  "x_pixels": [],
  "y_pixels": [],
  "x_ticks_encrypted": [],
  "y_ticks_encrypted": [],
  "x_pixels_encrypted": [],
  "y_pixels_encrypted": [],
  "x_axis_type": "数值轴",
  "y_axis_type": "文字轴",
  "x_axis_scale": "linear",
  "y_axis_scale": "log",
  "basic_grid_path": "..._grid.png",
  "encrypted_grid_path": "..._with_grid.png",
  "colors": [],
  "axis_repair": {}
}
```

随后 `backend/main.py` 会把这些字段合并到系统生成 JSON，并补充：

- `chart_id`
- `chart_type`
- `coordinate_system`
- `image_paths.no_grid`
- `image_paths.with_grid`
- `image_paths.grid_with_grid`
- `image_paths.basic_grid`

评估预测阶段使用这个系统生成 JSON，而不是数据集自带 GT JSON。

## 15. 辅助和历史模块

### config.py

主要保存早期调试路径、图片路径和模块开关。当前后端主流程不依赖它驱动生产处理。

### test_grid.py

本地单图调试入口，会直接调用：

```python
from grid_generation import process_chart
```

适合快速验证单张图的加密产物。

### function_calling/image/draw_grid_from_ticks.py

早期根据 tick 线绘制网格的工具模块。当前主流程主要使用 `grid_generation.py` 内部的 `draw_basic_grid` 和 `draw_encrypted_grid`。

### function_calling/ticks/correct_ticks.py、fit_ticks.py、evaluate_ticks.py

早期或独立调试 tick 修正、拟合、评估的模块。当前主链路中部分逻辑已经内聚到 `grid_generation.py`。

### circular_angle_grid.py

用于 pie/donut 的角度网格，不属于直角系链路，但同在 `Grid_generation` 目录下。

## 16. 一句话总览

当前直角系链路的设计思路是：

```text
上传阶段 MLLM 给出图表结构先验
  -> CV 检测轴、tick、网格、柱体/点等视觉结构
  -> MLLM 读取 tick 文本和颜色/对象信息
  -> 用数值一致性和视觉网格绑定 tick 文本与像素
  -> 只对数值轴插入中间 tick
  -> 输出基础网格图、加密网格图和系统生成 JSON
```

这样可以让正常数据集图表继续走稳定的原始 CV 路径，同时在真实图中遇到缺轴、弱轴、右侧轴、背景网格、不均匀网格、分组柱形图等情况时，有更明确的分支逻辑进行修复。

