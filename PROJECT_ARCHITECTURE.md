# Project Architecture Notes

This repository is a chart analysis demo with a FastAPI backend and a Vue/Vite frontend. It uploads a chart image, detects the chart type, generates a processed/grid-enhanced image, and returns extracted prediction results. A hidden admin/backdoor mode can optionally upload GT JSON for metric calculation, but the normal user flow does not depend on dataset GT.

## Quick Start

Backend:

```powershell
cd backend
pip install -r requirements.txt
python main.py
```

The API runs at `http://127.0.0.1:8000`.

Frontend:

```powershell
cd frontend/chart-demo-ui
npm install
npm.cmd run dev -- --host 127.0.0.1
```

The UI runs at `http://127.0.0.1:5173`.

On Windows PowerShell, prefer `npm.cmd` if execution policy blocks `npm.ps1`.

## Runtime Flow

1. `frontend/chart-demo-ui/src/App.vue` accepts a chart image. In admin/backdoor mode it can also accept GT JSON.
2. `POST /api/upload/` in `backend/main.py` saves uploaded files under `backend/data/upload/`.
3. `backend/type_detection/chart_type.py` tries to classify the chart with an external LLM API. If that fails, it falls back to a default chart type so the app can continue.
4. `backend/type_detection/chart_registry.py` is the shared registry for supported chart types, coordinate-system categories, and capabilities.
5. `POST /api/process/` uses `ChartProcessorFactory` from `backend/type_detection/chart_processor.py`.
6. Polar charts route to `backend/demo_rose/` or `backend/demo_radar/`; cartesian charts route to `backend/Grid_generation/`.
7. Generated images and intermediate JSON are written under `backend/data/output/`.
8. `POST /api/evaluate/` runs `backend/evaluation_prediction/` to extract values from the system-generated image/JSON. In admin/backdoor mode, saved GT can also be used to calculate metrics.
9. `GET /api/images/{filename}` and `GET /api/results/{filename}` serve outputs to the frontend.

Runtime code must not import from `reference/`. That directory is kept only for archived/reference scripts such as the moved `reference/prediction_core`.

## Files Worth Reading First

| Path | Why it matters |
| --- | --- |
| `backend/main.py` | FastAPI app, route definitions, upload/process/evaluate orchestration helpers, runtime directories, in-memory `charts_db`. |
| `backend/type_detection/chart_registry.py` | Shared chart type registry: type names, coordinate-system categories, supported capabilities, and fallback type. |
| `backend/type_detection/chart_processor.py` | Processor protocol, shared polar processor base, cartesian processor, and chart type factory. |
| `backend/type_detection/chart_type.py` | Chart type detection and fallback behavior; contains external API configuration. |
| `backend/evaluation/` | Unified evaluation package extracted from the old `Final_scatterplot_0717` experiment scripts: data normalization, metric calculation, and API-ready summaries. |
| `backend/evaluation_prediction/` | Runtime value-extraction flows for bar, stacked bar, line, scatter, bubble, pie, donut, radar, and rose charts. Uses backend-generated JSON rather than dataset GT in the normal flow. |
| `backend/Grid_generation/grid_generation.py` | Main cartesian chart processing pipeline. |
| `backend/Grid_generation/function_calling/` | Axis, tick, label, color, and grid helper modules. |
| `backend/demo_rose/` | Rose chart encode, axis detection, and evaluation helpers. |
| `backend/demo_radar/` | Radar chart encode, axis detection, and evaluation helpers. |
| `frontend/chart-demo-ui/src/App.vue` | Entire current UI and API interaction logic. |

## Paths To Avoid Re-reading Unless Needed

| Path | Note |
| --- | --- |
| `backend/charts/` | Large local sample image/JSON corpus. Read specific files only when testing a sample. |
| `backend/data/` | Runtime upload/output/result/cache directories; safe to regenerate. |
| `frontend/chart-demo-ui/node_modules/` | Dependency install output. |
| `frontend/chart-demo-ui/dist/` | Vite build output. |
| `Final_scatterplot_0717/` | Legacy experiment workspace. Core metric ideas were migrated into `backend/evaluation/`; old batch scripts, venvs, temporary crops, and result images are not needed by the app. |
| `reference/` | Archived/reference scripts only. Runtime backend/frontend code should not import from here. |
| `test_output/` | Generated/test output. |

## Verification Commands

```powershell
python -m py_compile backend/main.py
python -m compileall -q backend/type_detection backend/Grid_generation backend/demo_rose backend/demo_radar
cd frontend/chart-demo-ui
npm.cmd run build
```

Smoke checks after starting both services:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173/
```

## Known Gotchas

- Many Chinese comments and some terminal output display as mojibake in PowerShell, but the app and browser UI can still render Chinese correctly.
- Full upload/process/evaluate flows may call external LLM APIs. They can fail without network access or valid credentials.
- Model API settings are centralized in `backend/evaluation_prediction/common/model_config.py` and exposed through `model_api_config.py`. Switch profiles with `CHART_MODEL_PROFILE` or override individual fields with `CHART_BASE_URL`, `CHART_MODEL_NAME`, and `CHART_API_KEY`.
- `charts_db` is in-memory, so uploaded `chart_id` values disappear when the backend restarts.
- Upload and process endpoints intentionally keep the same response fields used by the frontend: `chart_id`, `chart_type`, `confidence`, and `encrypted_image_url`.
- The repository currently has many pre-existing modified/deleted dataset files. Treat them as user work unless explicitly told to restore or remove them.

## Cleanup Policy

Safe generated paths are ignored in `.gitignore`: Python caches, frontend `dist`, frontend `node_modules`, backend runtime `data` output/cache/log directories, and `*.log`. Prefer cleaning those before reviewing diffs. Do not delete sample datasets or old source directories just because they are large.

