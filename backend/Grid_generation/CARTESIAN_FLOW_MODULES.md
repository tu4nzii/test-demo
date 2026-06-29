# 直角坐标系处理链路与模块说明

本文档描述当前 `backend/Grid_generation` 中直角坐标系图表的主流程。直角系以本项目 `main` 为准，不从 `origin/stacked_bar` 合入实现。

## 1. 范围

直角系类型：

- `v_bar`
- `h_bar`
- `line`
- `scatter`
- `bubble`

`pie`、`donut`、`radar`、`rose` 不属于本文件重点。

## 2. 外部入口

```text
backend/main.py
  -> process_chart_image(chart_info)
  -> ChartProcessorFactory.create_processor(chart_type)
  -> CartesianChartProcessor.encode_image(...)
  -> Grid_generation.grid_generation.process_chart(...)
```

`process_chart()` 是直角系从原图到网格图、加密图和系统 JSON 的核心入口。

## 3. 主要模块

| 路径 | 说明 |
| --- | --- |
| `grid_generation.py` | 主编排和大量图表专用修复逻辑 |
| `grid_ocr.py` | OCR 文本、label 框、轴角色判断 |
| `grid_line_filter.py` | 网格候选过滤、优先级和 fallback 质量判断 |
| `grid_adjudication.py` | 多来源网格候选裁决 |
| `grid_geometry.py` | 几何结构、背景网格、plot bounds 推断 |
| `grid_visual.py` | 可视化调试图绘制 |
| `function_calling/label/` | MLLM tick、bar value label、颜色等读取 |
| `function_calling/ticks/` | tick 扫描、过滤、早期工具 |
| `function_calling/axis/` | 轴线检测和合并 |

## 4. 主流程

1. 读取图像，标准化上传阶段 `axis_repair_hint`。
2. 生成轴线、网格线和 tick 候选。
3. 使用 OCR/MLLM 读取可见 tick 文本和轴类型。
4. 针对 bar、line、scatter、bubble 做图表类型相关修复。
5. 用 OCR label 框、数值拟合、背景网格和候选线质量绑定 tick-value 与 tick-pixel。
6. 判断数值轴是否可加密。
7. 生成中间 tick 和中间 pixel。
8. 绘制原生网格、标准灰色加密网格和彩色预览图。
9. 写出 sidecar JSON 和主 JSON。

## 5. 上传阶段先验

`axis_repair_hint` 来自类型识别阶段，包含：

- 是否缺 X/Y 轴或 tick。
- X/Y 轴角色：numeric、category、date、unknown。
- 轴位置：bottom、top、left、right、middle。
- plot area 风格：explicit_axes、weak_axes、grid_only、no_axes。
- 是否存在背景网格。
- 上传阶段可见 tick label。
- 图例/系列颜色先验。

这些先验是 opt-in hint，不直接覆盖 CV 结果。

## 6. Tick 绑定和 fallback

绑定原则：

- 可见 tick 数量优先来自 OCR/MLLM label。
- 候选像素必须和 label 数量、轴范围、数值拟合一致。
- 背景网格可用于 weak axis 或 point chart 的 tick 恢复。
- 如果模型没有返回可用 tick，不应把 GT JSON 拿来补。

fallback 原则：

- 正常图不主动修复。
- CV 失败、弱轴、缺轴或背景网格更可靠时才触发修复。
- 结果不可靠时可以退出或标记 fallback，不强行给错误结果。

## 7. 加密和绘制

数值轴中间值：

```text
linear: (tick_i + tick_{i+1}) / 2
log:    sqrt(tick_i * tick_{i+1})
```

类别轴不加密。

标准灰色样式：

- `#cccccc`
- `1px`
- 2px 实线 + 2px 空白

彩色预览用于检查，不作为正式主流程效果。

## 8. 输出字段

`*_ticks.json` 通常包含：

```json
{
  "x_ticks": [],
  "y_ticks": [],
  "x_pixels": [],
  "y_pixels": [],
  "x_ticks_encrypted": [],
  "y_ticks_encrypted": [],
  "x_pixels_encrypted": [],
  "y_pixels_encrypted": [],
  "x_axis_type": "numeric",
  "y_axis_type": "category",
  "basic_grid_path": "...",
  "encrypted_grid_path": "...",
  "colored_grid_path": "...",
  "encrypted_label_style_version": "..."
}
```

这些字段会被 `backend/main.py` 合并进评估预测使用的系统 JSON。

## 9. 当前维护边界

- 不把 `stacked_bar` 分支的直角系实现合入主线。
- 不用 GT 修 tick、label、颜色或轴。
- 改 label 样式后必须更新缓存版本。
- 改 fallback 阈值后要用全量报告检查退出样本和误接收样本变化。
