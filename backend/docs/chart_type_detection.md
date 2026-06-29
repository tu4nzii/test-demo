# 图表类型识别处理逻辑

本文档说明当前上传阶段的图表类型识别、结构先验提取、异常策略和数据边界。

## 1. 入口

```text
POST /api/upload/
  -> backend/main.py::register_chart()
  -> backend/main.py::detect_chart_type()
  -> backend/type_detection/chart_type.py::ChartTypeDetector.detect_chart_type()
```

识别输入只有图片。上传 JSON、数据集 JSON 和 GT 不参与类型判断。

## 2. 支持类型

注册表位于 `backend/type_detection/chart_registry.py`。

| 类型 | 坐标系 | 后续处理 |
| --- | --- | --- |
| `rose` | polar | `backend/demo_rose/` |
| `radar` | polar | `backend/demo_radar/` |
| `v_bar` | cartesian | `backend/Grid_generation/` |
| `h_bar` | cartesian | `backend/Grid_generation/` |
| `line` | cartesian | `backend/Grid_generation/` |
| `scatter` | cartesian | `backend/Grid_generation/` |
| `bubble` | cartesian | `backend/Grid_generation/` |
| `donut` | polar | circular angle grid |
| `pie` | polar | circular angle grid |

当前主流程不注册 `v_stacked_bar` / `h_stacked_bar`。stacked bar 相关文本会归一到普通 bar 类型，避免改变直角系主线类别集合。

## 3. 模型输出

模型需要返回严格 JSON，核心字段包括：

```json
{
  "type": "scatter",
  "confidence": 0.95,
  "axis_repair": {
    "x_axis_missing": false,
    "y_axis_missing": false,
    "x_axis_role": "numeric",
    "y_axis_role": "numeric",
    "plot_area_style": "explicit_axes",
    "has_background_grid": true,
    "bar_layout": "single",
    "bar_orientation": "unknown"
  },
  "axis_tick_labels": {
    "x_axis_type": "numeric",
    "y_axis_type": "numeric",
    "x_ticks": [],
    "y_ticks": []
  },
  "radar_grid": {
    "shape": "polygon",
    "confidence": 0.9
  },
  "series_items": {
    "kind": "legend",
    "items": [
      {"name": "Series 1", "color": "#3366cc"}
    ]
  }
}
```

`axis_repair` 是保守先验，不直接覆盖 CV 结果。只有在 CV 失败、弱轴、缺轴、背景网格更可靠等情况下，后续处理才使用它触发修复。

## 4. 异常策略

当前策略是失败即报错：

- 模型调用失败，返回 HTTP 400。
- 模型返回空结果，返回 HTTP 400。
- 模型返回不支持类型，返回 HTTP 400。
- `confidence` 非法，返回 HTTP 400。

系统不再用默认 `v_bar` 兜底继续流程，因为错误类型会污染后续网格、缓存和评估预测。

## 5. 图例颜色

模型会尝试返回 `series_items`。对于可见图例，后端还会尝试在图像中局部采样图例色块，减少模型猜测颜色造成的误差。

颜色识别原则：

- 读取可见色块、线段、点标记或扇区颜色。
- 不使用默认调色板猜测。
- 不把轴文字、网格线或背景颜色当作系列颜色。
- 不确定时允许返回 `null`，比猜错更好。

## 6. Radar 网格形态

`radar_grid.shape` 会被标准化为：

```json
{
  "axis_repair": {
    "radar_grid_shape": "polygon",
    "radar_grid_confidence": 0.9
  }
}
```

该信息只作为 radar 极坐标处理的 hint：

- `polygon`：多边形 radar 几何检测。
- `circular`：圆形 radar 检测和大白边保护。
- `unknown`：保持原有处理，不触发专用修正。

## 7. 模型配置

模型配置集中在：

```text
backend/evaluation_prediction/common/model_config.py
backend/model_api_config.py
model_api_config.py
```

当前默认 Gemini profile 为 `gemini-2.5-flash-lite`。密钥通过本地 ignored secret 或环境变量注入。
