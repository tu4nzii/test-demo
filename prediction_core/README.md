# Prediction Core

`prediction_core` provides a unified entry point for chart value prediction. It
wraps nine chart types behind one CLI and one chart registry, while keeping
chart-specific prompt, geometry, parser, visual, runner and evaluation logic
inside each chart module.

## Supported Chart Types

```text
cartesian: v_bar, h_bar, line, scatter, bubble
polar:     pie, donut, rose, radar
```

List the registered chart types:

```powershell
cd F:\program\test-demo
python -m prediction_core.run_chart --list
```

Run one chart type:

```powershell
cd F:\program\test-demo
$env:CHART_REPEAT_TIMES="1"
python -m prediction_core.run_chart scatter --chart-ids scatter_001 --batch-size 1
```

The unified CLI accepts:

- `chart_type`: one of `v_bar`, `h_bar`, `line`, `scatter`, `bubble`, `pie`,
  `donut`, `rose`, `radar`.
- `--chart-ids`: one or more chart ids supported by that chart type.
- `--batch-size`: optional dataset batch size.
- `--dry-run`: print the resolved backend command without running it.
- `--list`: print the registry.

## Model Configuration

Model/API configuration is centralized in `prediction_core/model_config.py` and
shared by the backend through `model_api_config.py`.

Use one profile to switch the whole system:

```powershell
$env:CHART_MODEL_PROFILE="gemini"         # current default, vveai gemini-3.1-flash-lite
$env:CHART_MODEL_PROFILE="vveai_gemini"   # same as gemini
$env:CHART_MODEL_PROFILE="gpt54"          # BUAA gpt-5.4 profile
$env:CHART_MODEL_PROFILE="dsiclab_gpt54"  # same as gpt54
$env:CHART_MODEL_PROFILE="vveai_gpt41"    # original vveai gpt-4.1
```

Environment variables always take precedence over a profile:

```powershell
$env:CHART_BASE_URL="http://host/v1"
$env:CHART_MODEL_NAME="gemini-3.1-flash-lite"
$env:CHART_API_KEY="<your key>"
```

To use each script's old local Pixtral endpoint pool:

```powershell
$env:CHART_USE_LEGACY_PIXTRAL="1"
```

To override with explicit OpenAI-compatible chat-completion URLs:

```powershell
$env:CHART_API_URLS="http://localhost:8100/v1/chat/completions,http://localhost:8101/v1/chat/completions"
```

## Current Structure

```text
prediction_core/
  assets/              # chart-type data roots and generated results
  chart_modules/       # chart implementations, organized directly by chart type
    v_bar/
    h_bar/
    line/
    scatter/
    bubble/
    pie/
    donut/
    rose/
    radar/
  chart_types/         # registry entries grouped by coordinate system
    cartesian.py
    polar.py
  execution/           # unified CLI backend adapter
  testing/             # single-object E2E and dataset trimming helpers
  axis_utils.py        # axis mapping helpers
  chart_io.py          # config loading, image encoding, paths
  evaluation_utils.py  # shared evaluation helpers
  json_utils.py        # model-output extraction and JSON parsing
  model_config.py      # endpoint, model and API-key config
  runtime.py           # environment-driven runtime knobs
  run_chart.py         # unified CLI
  specs.py             # ChartSpec dataclass
```

There is no `legacy/`, `cartesian/`, or shared `xy_points/` implementation
folder in the active chart code. Chart modules are organized directly by chart
type.

## Module Boundaries

For modular chart types, the expected local files are:

- `cli.py`: command-line adapter for that chart type.
- `angle_grid.py`: angle-grid generation for sector charts that create a
  `with_grid` image from `no_grid`.
- `data.py`: chart config loading and target enumeration.
- `prompts.py`: prompt construction and output-schema wording.
- `parser.py`: model output parsing for that chart type.
- `geometry.py`: value/pixel/axis transformations and crop math.
- `visual.py`: overlays, crops, grid images and visual feedback artifacts.
- `model.py`: model request helpers.
- `evaluation.py`: chart-specific evaluation entry points.
- `runner.py`: orchestration for one chart or batch.

For the polar chart types, `flow.py` now stays focused on orchestration:
dataset iteration, feedback/amplifier sequencing, batching and result writes.
Prompt text lives in `prompts.py`; overlays, crops and visual feedback live in
`visual.py`; model calls live in `model.py`. `pie` and `donut` also use
`angle_grid.py`, `data.py` and `evaluation.py`, while `rose` and `radar` use
`geometry.py` plus `amplifier.py` for their crop/search loop.

## Integration Contract

The stable integration point for a larger system is:

```powershell
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1
```

For Python-level integration, use the registry and execution adapter:

```python
from prediction_core.chart_types import get_spec
from prediction_core.execution.adapter import RunRequest, run_backend

spec = get_spec("scatter")
request = RunRequest(spec=spec, chart_ids=["scatter_001"], batch_size=1)
exit_code = run_backend(request)
```

The registry source of truth is `chart_types/cartesian.py` and
`chart_types/polar.py`. Each `ChartSpec` records:

- `chart_type`
- `coordinate_system`
- backend `script`
- canonical `sample_chart_id`
- `data_path`
- single-object test `trim_strategy`
- notes for the current implementation

## Outputs

Result locations differ by chart family because the source experiments used
different output conventions. Current verified sample outputs include:

- `assets/v_bar/results_vbar_gemini/<chart_id>/experiment_results.csv`
- `assets/h_bar/results_Pixtral/<chart_id>/experiment_results.csv`
- `assets/line/results_line_gemini/<chart_id>/experiment_results.csv`
- `assets/scatter/results_scatter_Pixtral/<chart_id>/experiment_results.csv`
- `assets/bubble/results_bubble_Pixtral/<chart_id>/experiment_results.csv`
- `assets/pie/results_Pixtral/<chart_id>/experiment_results.csv`
- `assets/donut/results_Pixtral/<chart_id>/experiment_results.csv`
- `assets/rose/coordinates_by_image_rose_<model>_async.json`
- `assets/radar/coordinates_by_image_radar_<model>_async.json`

Cartesian and sector charts also write summary CSV/PNG files from their
evaluation layer when predictions are available.

## Verification

Dry-run the canonical single-object path for every chart type:

```powershell
cd F:\program\test-demo
python -m prediction_core.testing.run_single_object_e2e --chart-types v_bar h_bar line scatter bubble pie donut rose radar --dry-run --continue-on-failure
```

Run one real sample per chart type with the configured API:

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

Last structural audit verified:

- `python -m py_compile` passes for changed modules.
- single-object dry-run passes for all nine chart types.
- one complete API-backed sample run succeeds for all nine chart types.
- no active `legacy`, `cartesian`, `xy_points`, or `__pycache__` directories
  remain in `prediction_core`.

See `REFACTORING_GUIDE.md` for module-boundary rules and future cleanup notes.
