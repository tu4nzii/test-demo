# Project Architecture Notes

This repository is a chart analysis demo with a FastAPI backend and a Vue/Vite frontend. It uploads a chart image plus matching JSON, generates a processed/grid-enhanced image, and returns evaluation JSON.

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

1. `frontend/chart-demo-ui/src/App.vue` accepts an image file and a JSON file.
2. `POST /api/upload/` in `backend/main.py` saves both files under `backend/data/upload/`.
3. `backend/function_call/chart_type.py` tries to classify the chart with an external LLM API. If that fails, it falls back to a default chart type so the app can continue.
4. `POST /api/process/` uses `ChartProcessorFactory` from `backend/function_call/chart_processor.py`.
5. Rose charts route to `backend/demo_rose/`; radar charts route to `backend/demo_radar/`; cartesian-like charts route to `backend/Grid_generation/`.
6. Generated images and intermediate JSON are written under `backend/data/output/`.
7. `POST /api/evaluate/` writes result JSON under `backend/data/results/`.
8. `GET /api/images/{filename}` and `GET /api/results/{filename}` serve outputs to the frontend.

## Files Worth Reading First

| Path | Why it matters |
| --- | --- |
| `backend/main.py` | FastAPI app, route definitions, upload/process/evaluate orchestration helpers, runtime directories, in-memory `charts_db`. |
| `backend/function_call/chart_processor.py` | Processor protocol, shared polar processor base, cartesian processor, and chart type factory. |
| `backend/function_call/chart_type.py` | Chart type detection and fallback behavior; contains external API configuration. |
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
| `VishintPrompt_evaluatation/` | Pruned source/reference copy from the old grid/evaluation workspace. Generated charts, output, logs, venv, and nested `.git` were removed. |
| `test_output/` | Generated/test output. |

## Verification Commands

```powershell
python -m py_compile backend/main.py
python -m compileall -q backend/function_call backend/Grid_generation backend/demo_rose backend/demo_radar
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
- `charts_db` is in-memory, so uploaded `chart_id` values disappear when the backend restarts.
- Upload and process endpoints intentionally keep the same response fields used by the frontend: `chart_id`, `chart_type`, `confidence`, and `encrypted_image_url`.
- The repository currently has many pre-existing modified/deleted dataset files. Treat them as user work unless explicitly told to restore or remove them.
- `backend/old/` looks archival, but it is source code and should not be deleted without a deliberate cleanup decision.
- `VishintPrompt_evaluatation/function_calling/` and `VishintPrompt_evaluatation/utils/` are mostly mirrored by `backend/Grid_generation/`. The old folder still has reference-only files that are not fully wired into the app yet: `model_processor.py`, `grid_detection_generator.py`, `label_validator.py`, `axis_accuracy_annotator.py`, `conclusion.py`, and docs.

## Cleanup Policy

Safe generated paths are ignored in `.gitignore`: Python caches, frontend `dist`, frontend `node_modules`, backend runtime `data` output/cache/log directories, and `*.log`. Prefer cleaning those before reviewing diffs. Do not delete sample datasets or old source directories just because they are large.
