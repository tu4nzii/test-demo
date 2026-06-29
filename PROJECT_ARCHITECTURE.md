# Project Architecture Notes

This repository is a FastAPI + Vue/Vite chart analysis demo. The current `main`
branch is the source of truth for cartesian chart handling. The `origin/stacked_bar`
branch is useful only as a polar-coordinate reference and must not overwrite the
cartesian grid/encryption flow.

## Runtime Flow

1. `frontend/chart-demo-ui/src/App.vue` accepts either a custom uploaded chart image or a dataset-preview sample.
2. `POST /api/upload/` saves a custom image under `backend/data/upload/` and calls MLLM chart-type detection.
3. Detection failures are surfaced as errors. The backend does not silently fall back to a wrong chart type.
4. `POST /api/process/` routes through `ChartProcessorFactory`.
5. Cartesian charts route to `backend/Grid_generation/`; polar and circular charts route to their registered processors.
6. Generated outputs are written under `backend/data/output/` or `backend/data/dataset_preview_cache/`.
7. `POST /api/evaluate/` runs evaluation prediction from the system-generated JSON and grid images.
8. Dataset-preview endpoints load cached current-system outputs by dataset source and category.

Normal generation must not consume dataset GT. GT is only allowed in offline metric calculation.

## Supported Types

The registry lives in `backend/type_detection/chart_registry.py`.

```text
rose, radar, v_bar, h_bar, line, scatter, bubble, donut, pie
```

There are no runtime `v_stacked_bar` or `h_stacked_bar` chart types in the current main flow. Stacked-bar wording may be normalized into ordinary bar types, but cartesian behavior remains the current main implementation.

## Key Files

| Path | Purpose |
| --- | --- |
| `backend/main.py` | FastAPI app, upload/process/evaluate orchestration, dataset preview cache routes. |
| `backend/type_detection/chart_type.py` | MLLM chart-type detection and upload-time structure priors. |
| `backend/type_detection/chart_registry.py` | Supported types and coordinate-system registry. |
| `backend/type_detection/chart_processor.py` | Processor protocol and chart-type routing. |
| `backend/Grid_generation/` | Main cartesian grid reconstruction and encryption flow. |
| `backend/evaluation_prediction/` | Runtime value extraction for step 3, independent from grid generation. |
| `backend/evaluation/` | Offline metrics, reports, and batch evaluation helpers. |
| `backend/datasets/VisHintPrompt_datasets/` | Dataset preview sources. |
| `frontend/chart-demo-ui/src/App.vue` | Current UI and API integration. |
| `reference/` | Archived/reference code only; runtime code must not import from it. |

## Dataset Preview

Dataset sources:

| source | directory |
| --- | --- |
| `realworld` | `backend/datasets/VisHintPrompt_datasets/Final-RealDataset` |
| `synthetic` | `backend/datasets/VisHintPrompt_datasets/Sy.Dataset` |

The UI loads categories first, then samples for the selected category. This keeps preview startup fast. Preview cache is stored under:

```text
backend/data/dataset_preview_cache/
```

Batch-generated current-system cache can also be seeded from:

```text
backend/evaluation/recheck_outputs/vishintprompt_full_grid_encryption_latest/
```

## Model Configuration

Model settings are centralized in:

```text
backend/evaluation_prediction/common/model_config.py
backend/model_api_config.py
model_api_config.py
```

The current Gemini default is `gemini-2.5-flash-lite`. API keys should come from ignored local secret files or environment variables, not from committed code.

## Verification

```powershell
python -m py_compile backend/main.py backend/type_detection/chart_type.py backend/type_detection/chart_registry.py backend/Grid_generation/grid_generation.py
cd frontend/chart-demo-ui
npm.cmd run build
```

Smoke checks after starting services:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173/
```

## Gotchas

- Do not use GT to generate grids, ticks, labels, colors, or prediction previews.
- Custom upload must run the current system chain from scratch and should not use dataset preview cache.
- Dataset preview cache must be regenerated when grid style, label style, or prediction output format changes.
- Runtime code must not import from `reference/`.
- Do not merge cartesian code from `origin/stacked_bar` into `main`; only inspect its polar handling when needed.
- There may be local modified dataset files. Treat them as user work unless explicitly asked to restore them.
