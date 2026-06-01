# 重构指南

本文档记录 `prediction_core` 当前的模块边界规则。它替代早期规划笔记，因为那些笔记已经不再匹配当前项目结构。

## 当前架构

活跃图表代码直接按图表类型组织：

```text
chart_modules/
  v_bar/
  h_bar/
  line/
  scatter/
  bubble/
  pie/
  donut/
  rose/
  radar/
```

不要再为活跃代码引入额外的 `cartesian/`、`polar/`、`xy_points/` 或 `legacy/` 目录。坐标系统分组属于 `chart_types/cartesian.py` 和 `chart_types/polar.py`，也就是注册表所在位置。

## 文件职责

当某个图表类型完成模块化后，使用以下文件名：

- `cli.py`：解析该图表类型的 CLI 参数并调用 runner。
- `angle_grid.py`：当图表类型需要在预测前生成角度网格叠加时，生成对应的 `with_grid` 图像。
- `data.py`：加载配置并构造预测目标。
- `prompts.py`：构造 prompt 并描述期望输出 schema。
- `parser.py`：把模型响应解析为该图表类型的数值。
- `geometry.py`：映射数据值、坐标轴、角度和像素。
- `visual.py`：创建网格、叠加图、裁剪图和视觉反馈图像。
- `model.py`：发起模型调用，并解包 OpenAI-compatible 响应。
- `evaluation.py`：暴露该图表类型的指标计算/保存入口。
- `runner.py`：编排该图表工作流。

对于极坐标图表，`flow.py` 保持为编排层。角度网格生成放在 `angle_grid.py`，prompt 文本放在 `prompts.py`，视觉反馈/裁剪放在 `visual.py`，API 调用放在 `model.py`，小型坐标/颜色辅助放在 `geometry.py`，如果存在迭代裁剪/搜索循环，则放在 `amplifier.py`。任何被抽取出来的职责都必须被 flow 导入并使用。避免创建没人调用的 wrapper 模块。

## 共享模块

只有当逻辑被多个图表类型使用，且不感知具体 chart id 或资产路径时，才放入顶层共享模块。

- `chart_io.py`：配置加载、安全文件名、目录创建、图像编码。
- `json_utils.py`：稳健的模型输出和 JSON 解析辅助。
- `axis_utils.py`：可复用的坐标轴映射辅助。
- `evaluation_utils.py`：共享指标、最终轮次、summary 和绘图辅助。
- `model_config.py`：endpoint、模型和 API key 解析。
- `runtime.py`：由环境变量驱动的运行参数。

图表专属代码应把路径、标签和列名传入共享辅助，而不是让共享辅助知道某个具体图表类型。

## 评估层

当前评估拆分如下：

- `h_bar`、`v_bar`、`line`：图表专属 `evaluation.py` 委托给共享单轴辅助。
- `scatter`、`bubble`：图表专属 `evaluation.py` 委托给共享 XY 点辅助。
- `pie`、`donut`：图表专属 `evaluation.py` 委托给共享极坐标扇区辅助。
- `rose`、`radar`：flow 在委托本地 model、prompt、visual、geometry 和 amplifier 辅助后，直接写坐标 JSON。

保留图表专属 `evaluation.py` 文件，因为 runner 和 flow 会导入它们。不要添加没有调用方的兼容函数。

## 注册表规则

每个受支持图表类型都必须在 `chart_types/cartesian.py` 或 `chart_types/polar.py` 中有一个 `ChartSpec` 条目。

每个 spec 必须定义：

- `chart_type`
- `coordinate_system`
- `script`
- `sample_chart_id`
- `data_path`
- `trim_strategy`
- `model_line`
- `note`
- 当后端必须从项目根目录运行时，定义 `workdir_override`

统一 CLI 必须能够列出并运行该 spec：

```powershell
python -m prediction_core.run_chart --list
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1 --dry-run
```

## 集成规则

外部系统应调用统一入口或执行适配器，不应直接导入图表模块内部实现。

CLI 约定：

```powershell
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1
```

Python 约定：

```python
from prediction_core.chart_types import get_spec
from prediction_core.execution.adapter import RunRequest, run_backend

spec = get_spec("v_bar")
exit_code = run_backend(RunRequest(spec=spec, chart_ids=["v_bar_002"], batch_size=1))
```

## 完成修改前的验证

对于结构或模块边界变更，运行：

```powershell
cd F:\program\test-demo
python -m py_compile prediction_core\<changed files>
python -m prediction_core.testing.run_single_object_e2e --chart-types v_bar h_bar line scatter bubble pie donut rose radar --dry-run --continue-on-failure
```

对于影响行为的变更，还要为每个被修改的图表类型运行一个完整样例：

```powershell
$env:CHART_REPEAT_TIMES="1"
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1
```

验证后删除生成的 `__pycache__` 目录，保持仓库树清爽可读。
