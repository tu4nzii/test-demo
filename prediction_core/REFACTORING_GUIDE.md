# Refactoring Guide

This document records the current module boundary rules for `prediction_core`.
It replaces the earlier planning notes, which no longer matched the active
project structure.

## Current Architecture

Active chart code is organized directly by chart type:

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

Do not reintroduce an extra `cartesian/`, `polar/`, `xy_points/`, or `legacy/`
folder for active code. The coordinate-system grouping belongs in
`chart_types/cartesian.py` and `chart_types/polar.py`, where the registry lives.

## File Responsibilities

Use these file names when a chart type is fully modular:

- `cli.py`: parse chart-type CLI arguments and call the runner.
- `angle_grid.py`: generate chart-specific `with_grid` images when a chart
  type creates angular grid overlays before prediction.
- `data.py`: load configs and build prediction targets.
- `prompts.py`: build prompts and describe expected output schema.
- `parser.py`: parse model responses into chart-specific values.
- `geometry.py`: map data values, axes, angles and pixels.
- `visual.py`: create grids, overlays, crops and visual feedback images.
- `model.py`: make model calls and unwrap OpenAI-compatible responses.
- `evaluation.py`: expose chart-specific metric/save entry points.
- `runner.py`: orchestrate the chart workflow.

For polar chart types, keep `flow.py` as the orchestration layer. Angle-grid
generation belongs in `angle_grid.py`, prompt text in `prompts.py`, visual
feedback/crops in `visual.py`, API calls in `model.py`, small coordinate/color
helpers in `geometry.py`, and iterative crop/search loops in `amplifier.py`
when that logic exists. Any extracted responsibility must be imported and used
by the flow. Avoid creating unused wrapper modules.

## Shared Modules

Only put logic in top-level shared modules when it is used by more than one
chart type and does not know any specific chart id or asset path.

- `chart_io.py`: config loading, safe names, directory creation, image encoding.
- `json_utils.py`: robust model-output and JSON parsing helpers.
- `axis_utils.py`: reusable axis mapping helpers.
- `evaluation_utils.py`: shared metric, final-round, summary and plot helpers.
- `model_config.py`: endpoint, model and API-key resolution.
- `runtime.py`: environment-driven runtime knobs.

Chart-specific code should pass paths, labels and column names into shared
helpers rather than making shared helpers aware of one chart type.

## Evaluation Layer

The current evaluation split is:

- `h_bar`, `v_bar`, `line`: chart-specific `evaluation.py` delegates to shared
  single-axis helpers.
- `scatter`, `bubble`: chart-specific `evaluation.py` delegates to shared XY
  point helpers.
- `pie`, `donut`: chart-specific `evaluation.py` delegates to shared polar
  sector helpers.
- `rose`, `radar`: flows write coordinate JSON directly after delegating model,
  prompt, visual, geometry and amplifier helpers to their local modules.

Keep the chart-specific `evaluation.py` files because runners and flows import
them. Do not add compatibility functions that have no caller.

## Registry Rules

Every supported chart type must have one `ChartSpec` entry in either
`chart_types/cartesian.py` or `chart_types/polar.py`.

Each spec must define:

- `chart_type`
- `coordinate_system`
- `script`
- `sample_chart_id`
- `data_path`
- `trim_strategy`
- `model_line`
- `note`
- `workdir_override` when the backend must run from project root

The unified CLI must be able to list and run the spec:

```powershell
python -m prediction_core.run_chart --list
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1 --dry-run
```

## Integration Rule

External systems should call the unified entry point or the execution adapter.
They should not import chart-module internals directly.

CLI contract:

```powershell
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1
```

Python contract:

```python
from prediction_core.chart_types import get_spec
from prediction_core.execution.adapter import RunRequest, run_backend

spec = get_spec("v_bar")
exit_code = run_backend(RunRequest(spec=spec, chart_ids=["v_bar_002"], batch_size=1))
```

## Validation Before Finishing Changes

For structural or module-boundary changes, run:

```powershell
cd F:\program\test-demo
python -m py_compile prediction_core\<changed files>
python -m prediction_core.testing.run_single_object_e2e --chart-types v_bar h_bar line scatter bubble pie donut rose radar --dry-run --continue-on-failure
```

For behavior-affecting changes, also run one complete sample for every touched
chart type:

```powershell
$env:CHART_REPEAT_TIMES="1"
python -m prediction_core.run_chart <chart_type> --chart-ids <id> --batch-size 1
```

After validation, remove generated `__pycache__` directories so the repo tree
stays readable.
