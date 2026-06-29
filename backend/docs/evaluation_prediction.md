# 评估预测处理逻辑

本文档说明 `/api/evaluate/` 的职责、输入来源、缓存规则和指标边界。

## 1. 职责

评估预测是前端第 3 步。它读取第 2 步生成的系统 JSON 和网格图，执行值提取或返回结构化预测结果。

它不负责生成网格，也不能用 GT 修正生成结果。

## 2. 入口

```text
POST /api/evaluate/?chart_id=...
  -> backend/main.py::evaluate_chart()
  -> backend/main.py::evaluate_processed_chart()
  -> backend/evaluation_prediction/service.py
  -> backend/evaluation/service.py
```

结果通过：

```text
GET /api/results/{filename}
```

返回给前端。

## 3. 输入来源

优先使用处理阶段写出的系统 JSON：

```text
backend/data/output/<chart_type>/<image_stem>.json
backend/data/output/<chart_type>/<image_stem>_ticks.json
backend/data/output/<chart_type>/<chart_id>.json
```

如果缺失，会尝试由处理器 `process_data()` 生成评估输入。

自定义上传不会读取数据集 GT。数据集预览可以读取由当前系统链路预生成的评估预测缓存。

## 4. 支持类型

当前运行时类型：

```text
v_bar, h_bar, line, scatter, bubble, pie, donut, radar, rose
```

评估预测包内部可能保留 stacked bar helper，但主线类型注册表不暴露 `v_stacked_bar` / `h_stacked_bar`。

## 5. 结果结构

前端主要消费：

```json
{
  "success": true,
  "mode": "prediction_extraction",
  "chart_id": "...",
  "chart_type": "scatter",
  "system_json": "...",
  "summary": {
    "object_count": 10,
    "chart_runs": 1
  },
  "predictions": [],
  "processed_json": {},
  "quality": {}
}
```

如果模型预测 runner 不适用于某类型，系统可能返回基于当前系统 JSON 和图像像素的 CV fallback 预测，并在 summary 中标记 `system_cv_fallback`。

## 6. GT 使用边界

允许：

- 离线评估中使用 GT 计算指标。
- 管理/回归脚本中读取数据集 JSON 作为评估基准。

禁止：

- 用 GT 生成网格。
- 用 GT 生成 tick。
- 用 GT 生成前端预测展示。
- 用 GT 覆盖系统识别出的 label、颜色或类别。

## 7. 离线指标

全量报告中常用：

- 数值轴 tick-value MAE。
- 数值轴 tick-value Acc@2px。
- 图例颜色准确率。
- 标签名准确率。
- 图表分类准确率。

注意：tick MAE 和 tick Acc 只计算数值轴，分类轴不纳入。

## 8. 缓存

模型输入不变时应复用模型缓存。数据集预览预测缓存可快速返回结果，但必须由当前系统链路生成。

如果网格缓存已经正确，只需要更新第 3 步评估预测缓存时，可以从网格阶段之后重新跑全链路预测，不必重建网格。
