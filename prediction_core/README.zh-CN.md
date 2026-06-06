# Prediction Core 中文说明

`prediction_core` 为图表数值预测提供统一入口。它用一个 CLI 和一个图表注册表封装九类图表，同时把各图表专属的 prompt、几何映射、解析器、可视化、runner 和评估逻辑保留在各自模块中。

## 支持的图表类型

```text
cartesian: v_bar, h_bar, line, scatter, bubble
polar:     pie, donut, rose, radar
```

查看已注册图表类型：

```powershell
cd F:\program\test-demo
python -m prediction_core.run_chart --list
```

运行一种图表：

```powershell
cd F:\program\test-demo
$env:CHART_REPEAT_TIMES="1"
python -m prediction_core.run_chart scatter --chart-ids scatter_001 --batch-size 1
```

统一 CLI 参数：

- `chart_type`：`v_bar`、`h_bar`、`line`、`scatter`、`bubble`、`pie`、`donut`、`rose`、`radar` 之一。
- `--chart-ids`：该图表类型支持的一个或多个 chart id。
- `--batch-size`：可选的数据集批大小。
- `--dry-run`：只打印解析后的后端命令，不实际运行。
- `--list`：打印注册表。

## 模型配置

模型/API 配置集中在 `prediction_core/model_config.py`，后端通过 `model_api_config.py` 共享同一套配置。

用一个 profile 即可切换全系统模型：

```powershell
$env:CHART_MODEL_PROFILE="dsiclab_gpt54"  # 当前默认
$env:CHART_MODEL_PROFILE="vveai_gpt41"    # 原 vveai gpt-4.1
$env:CHART_MODEL_PROFILE="vveai_gemini"   # vveai gemini-3.1-flash-lite
```

环境变量始终优先于 profile：

```powershell
$env:CHART_BASE_URL="http://host/v1"
$env:CHART_MODEL_NAME="gpt-5.4"
$env:CHART_API_KEY="<your key>"
```

如需使用各脚本旧的本地 Pixtral endpoint 池：

```powershell
$env:CHART_USE_LEGACY_PIXTRAL="1"
```

如需显式覆盖 OpenAI-compatible chat completion URL：

```powershell
$env:CHART_API_URLS="http://localhost:8100/v1/chat/completions,http://localhost:8101/v1/chat/completions"
```

## 当前结构

```text
prediction_core/
  assets/              # 各图表类型的数据根目录和生成结果
  chart_modules/       # 图表实现，按图表类型直接组织
    v_bar/
    h_bar/
    line/
    scatter/
    bubble/
    pie/
    donut/
    rose/
    radar/
  chart_types/         # 按坐标系统分组的注册表条目
    cartesian.py
    polar.py
  execution/           # 统一 CLI 后端适配器
  testing/             # 单对象 E2E 和数据集裁剪辅助
  axis_utils.py        # 坐标轴映射辅助
  chart_io.py          # 配置加载、图像编码、路径处理
  evaluation_utils.py  # 共享评估辅助
  json_utils.py        # 模型输出提取和 JSON 解析
  model_config.py      # endpoint、模型和 API key 配置
  runtime.py           # 由环境变量驱动的运行参数
  run_chart.py         # 统一 CLI
  specs.py             # ChartSpec 数据类
```

当前活跃图表代码中没有 `legacy/`、`cartesian/` 或共享 `xy_points/` 实现目录。图表模块直接按图表类型组织。

## 模块边界

对于模块化图表类型，预期本地文件如下：

- `cli.py`：该图表类型的命令行适配器。
- `angle_grid.py`：扇区类图表从 `no_grid` 生成 `with_grid` 图像的角度网格逻辑。
- `data.py`：图表配置加载和目标对象枚举。
- `prompts.py`：prompt 构造和输出 schema 描述。
- `parser.py`：该图表类型的模型输出解析。
- `geometry.py`：数值、像素、坐标轴变换和裁剪几何。
- `visual.py`：叠加图、裁剪图、网格图和视觉反馈产物。
- `model.py`：模型请求辅助。
- `evaluation.py`：图表专属评估入口。
- `runner.py`：单图或批量流程编排。

对极坐标图表，`flow.py` 现在只负责编排：数据集遍历、反馈/放大器顺序、批处理和结果写入。Prompt 文本在 `prompts.py`，叠加图、裁剪图和视觉反馈在 `visual.py`，模型调用在 `model.py`。`pie` 和 `donut` 还使用 `angle_grid.py`、`data.py` 和 `evaluation.py`；`rose` 和 `radar` 使用 `geometry.py`，并通过 `amplifier.py` 执行裁剪/搜索循环。

## 集成约定

较大系统中的稳定集成入口是：

```powershell
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1
```

Python 层集成使用注册表和执行适配器：

```python
from prediction_core.chart_types import get_spec
from prediction_core.execution.adapter import RunRequest, run_backend

spec = get_spec("scatter")
request = RunRequest(spec=spec, chart_ids=["scatter_001"], batch_size=1)
exit_code = run_backend(request)
```

注册表的事实来源是 `chart_types/cartesian.py` 和 `chart_types/polar.py`。每个 `ChartSpec` 记录：

- `chart_type`
- `coordinate_system`
- 后端 `script`
- 标准 `sample_chart_id`
- `data_path`
- 单对象测试 `trim_strategy`
- 当前实现说明

## 输出

由于来源实验使用了不同输出约定，不同图表族的结果路径不同。当前已验证样例输出包括：

- `assets/v_bar/results_vbar_gemini/<chart_id>/experiment_results.csv`
- `assets/h_bar/results_Pixtral/<chart_id>/experiment_results.csv`
- `assets/line/results_line_gemini/<chart_id>/experiment_results.csv`
- `assets/scatter/results_scatter_Pixtral/<chart_id>/experiment_results.csv`
- `assets/bubble/results_bubble_Pixtral/<chart_id>/experiment_results.csv`
- `assets/pie/results_Pixtral/<chart_id>/experiment_results.csv`
- `assets/donut/results_Pixtral/<chart_id>/experiment_results.csv`
- `assets/rose/coordinates_by_image_rose_<model>_async.json`
- `assets/radar/coordinates_by_image_radar_<model>_async.json`

笛卡尔图表和扇区图表在有预测结果时，也会由评估层写入 summary CSV/PNG。

## 验证

对所有图表类型 dry-run 标准单对象路径：

```powershell
cd F:\program\test-demo
python -m prediction_core.testing.run_single_object_e2e --chart-types v_bar h_bar line scatter bubble pie donut rose radar --dry-run --continue-on-failure
```

使用配置好的 API 为每种图表运行一个真实样例：

```powershell
cd F:\program\test-demo
$env:CHART_REPEAT_TIMES="1"
python -m prediction_core.run_chart v_bar --chart-ids v_bar_002 --batch-size 1
python -m prediction_core.run_chart h_bar --chart-ids h_bar_001 --batch-size 1
python -m prediction_core.run_chart line --chart-ids line_001 --batch-size 1
python -m prediction_core.run_chart scatter --chart-ids scatter_001 --batch-size 1
python -m prediction_core.run_chart bubble --chart-ids bubble_023 --batch-size 1
python -m prediction_core.run_chart pie --chart-ids 001 --batch-size 1
python -m prediction_core.run_chart donut --chart-ids donut_135 --batch-size 1
python -m prediction_core.run_chart rose --chart-ids rose_004 --batch-size 1
python -m prediction_core.run_chart radar --chart-ids radar_009 --batch-size 1
```

最近一次结构审计确认：

- 修改过的模块通过 `python -m py_compile`。
- 九种图表类型的单对象 dry-run 均通过。
- 九种图表类型各一个完整 API 样例运行成功。
- `prediction_core` 中不再保留活跃的 `legacy`、`cartesian`、`xy_points` 或 `__pycache__` 目录。

模块边界规则和后续清理说明见 `REFACTORING_GUIDE.zh-CN.md`。
