# 加密处理逻辑

本文档说明 `/api/process/` 的处理边界、直角坐标系加密流程、极坐标处理和输出结构。

## 1. 入口

```text
POST /api/process/?chart_id=...
  -> backend/main.py::process_chart()
  -> backend/main.py::process_chart_image()
  -> ChartProcessorFactory.create_processor(chart_type)
  -> processor.encode_image(...)
```

处理输入是上传图片和上传阶段识别出的系统先验。数据集 GT 或上传 JSON 不参与生成网格、tick、label、颜色或加密值。

## 2. 处理器路由

| 类型 | 处理器 |
| --- | --- |
| `v_bar`, `h_bar`, `line`, `scatter`, `bubble` | `CartesianChartProcessor` |
| `pie`, `donut` | `CircularAngleChartProcessor` |
| `radar` | `RadarChartProcessor` |
| `rose` | `RoseChartProcessor` |

直角系主链路以当前 `main` 为准。`origin/stacked_bar` 只参考极坐标系处理方式，不覆盖直角系。

## 3. 直角坐标系流程

核心入口：

```text
backend/Grid_generation/grid_generation.py::process_chart()
```

主要阶段：

1. 读取图片，初始化图表类型和上传阶段先验。
2. 检测轴线、背景网格、tick 短线和数据对象。
3. 结合 OCR/MLLM 读取可见 tick 文本、轴类型、图例/系列颜色。
4. 将 tick 文本与像素位置绑定。
5. 用 OCR label 框、背景网格、线性/对数拟合误差校验绑定结果。
6. 只对数值轴生成一轮中间 tick。
7. 根据 label 密度决定整条数值轴是否加密。
8. 生成基础网格、标准灰色加密网格和彩色预览网格。
9. 写出 `*_ticks.json` 和系统聚合 JSON。

## 4. 加密规则

数值轴：

```text
linear: (tick_i + tick_{i+1}) / 2
log:    sqrt(tick_i * tick_{i+1})
```

非数值轴：

```text
保持原 tick，不插入中间值。
```

密集保护：

- 比较加密后相邻 tick 的像素间距。
- 比较 OCR label 框宽度或高度。
- 空间不足时整轴跳过加密。
- 一个轴不做局部加密，避免语义混乱。

## 5. 绘制样式

标准模式：

- 原生网格和加密网格都使用 `#cccccc`。
- 线宽 `1px`。
- 虚线为 2px 实线 + 2px 空白。
- 新生成 label 使用黑色。

彩色模式：

- 用于人工预览和调试。
- 通过前端按钮切换。
- 不作为正式主流程的标准图。

## 6. 输出

典型输出：

| 文件 | 说明 |
| --- | --- |
| `*_grid.png` | 原生识别网格 |
| `*_with_grid.png` | 标准灰色加密网格 |
| `*_with_grid_color.png` | 彩色预览 |
| `*_ticks.json` | tick、像素、轴类型、颜色、路径等 |
| `*_image.json` 或同 stem JSON | 评估预测使用的系统 JSON |

核心 JSON 字段：

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
  "image_paths": {
    "no_grid": "...",
    "with_grid": "...",
    "grid_with_grid_color": "..."
  }
}
```

## 7. 极坐标处理

`radar`、`rose`、`pie`、`donut` 不使用直角系加密规则。

- `radar` 和 `rose` 使用现有 demo 处理器。
- `pie` 和 `donut` 使用角度网格。
- 后续优化可参考 `origin/stacked_bar` 中极坐标模块化和 fallback 组织方式。
- 生成端仍不得使用 GT。

## 8. 缓存

自定义上传：

- 默认不复用数据集预览缓存。
- 需要从图片开始完整处理。

数据集预览：

- 可以读取本项目当前链路生成的缓存。
- 网格缓存和评估预测缓存可以独立更新。
- 缓存版本由 label style、模型输入、prompt、图像内容等共同决定。
