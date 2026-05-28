# 图表类型识别处理逻辑

本文档说明后端“图表类型识别”功能的业务目标、代码入口、处理流程、输入输出和异常策略。

## 1. 功能定位

图表类型识别发生在用户上传图片之后、加密处理之前。它的职责是根据原始图表图片判断图表属于哪一种类型，并为后续处理器路由提供依据。

当前支持的类型集中注册在 `backend/type_detection/chart_registry.py`：

| 类型 | 坐标系 | 后续处理路径 |
| --- | --- | --- |
| `rose` | polar | `backend/demo_rose/` |
| `radar` | polar | `backend/demo_radar/` |
| `v_bar` | cartesian | `backend/Grid_generation/` |
| `h_bar` | cartesian | `backend/Grid_generation/` |
| `line` | cartesian | `backend/Grid_generation/` |
| `scatter` | cartesian | `backend/Grid_generation/` |
| `bubble` | cartesian | `backend/Grid_generation/` |
| `donut` | cartesian | 当前走通用处理器 |
| `pie` | cartesian | 当前走通用处理器 |

默认兜底类型是 `v_bar`。如果模型识别失败、返回非法类型或系统异常，后端会使用默认类型继续流程，避免上传阶段直接中断。

## 2. 代码入口

主要文件：

| 文件 | 作用 |
| --- | --- |
| `backend/main.py` | 上传接口和流程编排 |
| `backend/type_detection/chart_type.py` | 调用多模态模型识别图表类型 |
| `backend/type_detection/chart_registry.py` | 图表类型、坐标系和能力注册表 |
| `backend/type_detection/chart_processor.py` | 根据图表类型创建对应处理器 |

调用链：

```text
POST /api/upload/
  -> register_chart()
    -> save_upload_file()
    -> detect_chart_type()
      -> ChartTypeDetector().detect_chart_type()
    -> normalize_chart_type()
    -> get_coordinate_system()
    -> 写入 charts_db
```

## 3. 输入与输出

输入：

- 必填：图表图片文件。
- 可选：JSON 文件。

重要边界：图表类型识别只依赖图片。上传 JSON 不参与类型判断。

输出给前端：

```json
{
  "chart_id": "...",
  "chart_type": "v_bar",
  "coordinate_system": "cartesian",
  "confidence": 0.87
}
```

后端同时把以下运行时信息写入内存 `charts_db`：

```json
{
  "chart_id": "...",
  "chart_type": "...",
  "coordinate_system": "...",
  "confidence": 0.87,
  "image_path": "...",
  "json_path": "... or null",
  "processed": false,
  "evaluated": false
}
```

注意：`charts_db` 是内存态，后端重启后上传记录会丢失。

## 4. 模型识别流程

`ChartTypeDetector.detect_chart_type(image_path)` 的核心步骤：

1. 使用 OpenCV 读取图片。
2. 将图片转为 RGB。
3. 将图片编码为 JPEG，再转为 base64。
4. 构造提示词，要求模型在注册表支持的类型中选择一个。
5. 调用外部多模态模型接口。
6. 从模型响应文本中提取 JSON。
7. 校验返回字段：
   - 必须包含 `type`。
   - 必须包含 `confidence`。
   - `type` 必须能被 `normalize_chart_type()` 归一化到支持列表。
   - `confidence` 必须是 0 到 1 之间的数字。
8. 返回标准化后的图表类型和置信度。

模型响应期望格式：

```json
{
  "type": "line",
  "confidence": 0.92
}
```

## 5. 类型注册与归一化

`chart_registry.py` 是类型系统的唯一来源。它定义：

- `ChartDefinition`：图表类型定义。
- `CoordinateSystem`：`polar` 或 `cartesian`。
- `ChartCapability`：类型识别、网格加密、评估能力。
- `SUPPORTED_CHART_TYPES`：模型可返回的类型白名单。
- `CARTESIAN_CHART_TYPES`：直角坐标图表集合。
- `POLAR_CHART_TYPES`：极坐标图表集合。
- `normalize_chart_type()`：非法类型回退到默认类型。
- `get_coordinate_system()`：根据图表类型返回坐标系。

这一层的作用是避免模型直接决定系统行为。模型只给出候选类型，真正进入系统前必须经过注册表校验。

## 6. 处理器路由关系

上传接口只负责识别类型，真正选择处理算法发生在后续 `/api/process/`：

```text
ChartProcessorFactory.create_processor(chart_type)
```

路由规则：

- `rose` -> `RoseChartProcessor`
- `radar` -> `RadarChartProcessor`
- `v_bar`、`h_bar`、`line`、`scatter`、`bubble`、`pie`、`donut` -> `CartesianChartProcessor`

其中 `pie` 和 `donut` 当前虽然注册为 cartesian 通用处理路径，但它们语义上并不是标准直角坐标轴图表。后续如果要正式支持，应补充专门处理器或调整注册表分类。

## 7. 异常与兜底

类型识别阶段不应该因为模型失败阻断整体上传流程。当前异常策略：

- 图片读取失败：返回默认类型，并带上错误信息。
- 模型接口失败：返回默认类型，并带上错误信息。
- 模型响应不是合法 JSON：返回默认类型，并带上错误信息。
- 模型返回未知类型：返回默认类型，并带上错误信息。

兜底返回示例：

```json
{
  "type": "v_bar",
  "confidence": 0.5,
  "error": "..."
}
```

## 8. 与 JSON 的关系

图表类型识别不依赖业务 JSON。这里涉及的 JSON 只有两种：

- 模型响应 JSON：用于表达模型识别结果。
- 上传 JSON 路径：仅被保存到 `charts_db`，供后续评估或结果富集使用。

禁止把上传 JSON 的字段用于判断图表类型。类型识别的业务依据必须是原始图片。

