# Implementation Parameter Inventory

This note records the implementation details referenced in the reviewer response. It is an evidence checklist, not manuscript prose.

## Cartesian Axis And Tick Candidate Generation

- The active runtime path is `backend.main.process_chart_image` -> `ChartProcessorFactory` -> `Grid_generation.grid_generation.process_chart` -> `_process_chart_with_enhanced_grid_only` -> `grid_line_filter.process_image`.
- `process_chart` calls `_process_chart_with_enhanced_grid_only` first and returns its result when successful. If it fails, the function currently logs that the legacy CV Cartesian flow is disabled and returns `None`; it does not fall through to the old Canny/Hough block.
- The active Cartesian line candidates are produced by `backend/Grid_generation/grid_masks.py::make_line_masks`, not by the legacy `function_calling/axis/detect_lines.py` path.
- Runtime mask parameters are passed in `backend/Grid_generation/grid_generation.py::_run_enhanced_cartesian_grid_reconstruction`: `sat_max=70`, `white_cutoff=255`, `min_gray=95`, `contrast_min=7`, `include_dark=False`, `dark_cutoff=80`, `min_line_frac=0.055`, `gap_frac=0.006`, `max_thickness_frac=0.008`, `min_grid_span_frac=0.18`, `min_grid_lines=2`, `cluster_tolerance=3`, `grid_thickness=1`, and `tick_dark_cutoff=150`.
- OCR parameters in the active path are fixed as `ocr_min_score=0.45`, `ocr_det_thresh=0.35`, `ocr_det_box_thresh=0.60`, `ocr_det_unclip_ratio=1.15`, `ocr_det_limit_side_len=960`, and `ocr_det_limit_type=max`.
- The legacy Canny/Hough code remains after an early return in `grid_generation.py` and is kept only as historical/reference code, not as the current system path.

## Current Cartesian Grid Selection Path

- The current Cartesian generation pipeline is not the old axis-scan path.
- It builds three grid candidates: `combined_mask`, `tick_supplement`, and `semantic_guide`.
- `backend/Grid_generation/grid_adjudication.py::arbitrate_priority_grids` scores each candidate by label binding quality, target tick count, OCR support, MLLM guidance, and numeric-axis consistency.
- Selection is score-first; MLLM arbitration is used only when score/position evidence remains ambiguous.
- The latest full report under `backend/evaluation/results/vishintprompt_full_latest_report` evaluates the generated `final_bindings`, not raw legacy tick scanning output.

## OCR And Local Text Processing

- OCR/local text thresholding uses a small smoothing step, `cv2.GaussianBlur(gray, (3, 3), 0)`, in `backend/Grid_generation/grid_ocr.py`.
- This blur is local to text crop processing and is separate from the active Cartesian grid mask reconstruction.

## Pie/Donut Circular Angle Grid

- The active entry is `backend/type_detection/chart_processor.py::CircularAngleChartProcessor`, which calls `backend/Grid_generation/circular_angle_grid.py::process_circular_angle_chart`.
- Angle ticks are fixed by `ANGLE_STEP_DEGREES=15`.
- Plot-area detection first uses a color mask: HSV saturation `>12`, value `>40`, morphology ellipse kernel `(5,5)`, opening plus closing with `iterations=2`, connected-component min area `max(64, width*height*0.003)`, and `cv2.minEnclosingCircle`.
- Hough is a fallback only: Gaussian `(9,9), sigma=2`; `cv2.HoughCircles(dp=1.2, minDist=min(width,height)//3, param1=50, param2=30, minRadius=max(10,min_side//10), maxRadius=max(20,min_side//2))`.
- Angle grid rendering uses `grid_line_ratio=0.1` and line width `1`.

## Radar/Rose Radial Grid

- Radar and rose processors are registered in `backend/type_detection/chart_processor.py` and call `RadarChartEncoder` from `backend/demo_radar/demo_radar_circle_find.py` and `RoseChartEncoder` from `backend/demo_rose/demo_rose_circle_find.py`.
- Both use `tick_density=2`.
- Radar first circle: Gaussian `(9,9), sigma=2`, Hough `dp=1.2`, `minDist=100`, `param1=20`, `param2=30`, `minRadius=height/5`, `maxRadius=height/4`.
- Radar second circle: Gaussian `(9,9), sigma=2`, Hough `dp=1.2`, `minDist=100`, `param1=20`, `param2=50`, `minRadius=first_r+30`, `maxRadius=height/2`.
- Rose first circle: Gaussian `(9,9), sigma=2`, Hough `dp=1.2`, `minDist=100`, `param1=20`, `param2=30`, `minRadius=height/4`, `maxRadius=height`.
- Rose second circle: Gaussian `(9,9), sigma=2`, Hough `dp=1.2`, `minDist=100`, `param1=20`, `param2=50`, `minRadius=first_r+30`, `maxRadius=height/2`.
- Radar refinement includes polygon/circular evidence: Canny `50/150`, `HoughLinesP` threshold `24` or `28`, `minLineLength=max(24,min_side*0.07)`, `maxLineGap=15`; circular-grid masks use thresholds such as `sat<80`, `gray>80`, `gray<245`, and morphology close `(3,3)`.
- Radar/Rose radial tick values are read by the MLLM with `temperature=0.5`. Generation-side JSON stores generated `r_ticks` and geometry; GT is not used by the generation path.

## Chart Type Classification

- Type detection is implemented in `backend/type_detection/chart_type.py`.
- The MLLM call uses deterministic decoding (`temperature=0`) and a strict JSON response format.
- Supported runtime types are registered in `backend/type_detection/chart_registry.py`: `rose`, `radar`, `v_bar`, `h_bar`, `line`, `scatter`, `bubble`, `donut`, and `pie`.
- Unsupported or missing chart types raise an error instead of falling back to a default type.
- The latest full report records chart-type accuracy as `98.95%` overall, `100.00%` for Cartesian, and `96.64%` for polar; `bubble` and `scatter` are mutually accepted as point-chart types for this metric.

## Zoom-In Verification In Evaluation Prediction

- The zoom-in verification mechanism belongs to the third stage, evaluation prediction, not grid generation.
- Bar charts use `backend/evaluation_prediction/chart_modules/v_bar/runner.py`, `h_bar/runner.py`, and their `visual.py` crop helpers. The amplifier crop is centered around the predicted numeric value/category position; if the MLLM does not verify the target in the crop, the window is shifted and retried.
- Line charts use `backend/evaluation_prediction/chart_modules/line/runner.py` and `line/visual.py` to crop around the target x category and current y estimate; the contains-target check is recorded for diagnostics before localized prediction.
- Scatter and bubble charts use `feedback_crop_adaptive` in `scatter/runner.py` and `bubble/runner.py`; the crop is generated around the predicted mark, the target mark diameter is estimated, and crop size is expanded across attempts until the target is verified.
- Radar and rose value prediction is implemented in `backend/evaluation_prediction/chart_modules/polar_value.py`; model calls use `temperature=0.0`, prompts include generated `r_ticks`, `theta_ticks`, `theta_angles`, and color hints, and the module uses both grid and baseline images. If target-level predictions are missing, it falls back to a whole-chart prompt.
- Pie and donut charts use `crop_sector_for_amplifier` in `pie/visual.py` and `donut/visual.py` to crop and zoom the predicted sector, update the local center after crop/zoom, and refine over amplifier rounds. The fixed round settings are round 1 `pad=15 deg`, `grid=5 deg`, `zoom=2.0`; round 2 `pad=9 deg`, `grid=3 deg`, `zoom=2.0`; round 3 `pad=6 deg`, `grid=2 deg`, `zoom=3.0`.
- Pie/donut sector crops include an LLM contains-sector validation and can recrop with swapped angles when the previous start/end order is inconsistent.

## Diagnostic Outputs

- Legacy Cartesian Canny/Hough candidate diagnostic: `review/parameter_sensitivity_summary.csv` and `review/parameter_sensitivity_report.md`.
- Radar/Rose Hough `param2` diagnostic: `review/axis_prior_reviewer_eval.json`.
- Current Cartesian full-pipeline evidence: `review/current_cartesian_full_pipeline_evidence.json`.
- Claim/evidence boundary audit: `review/response_claim_evidence_audit.md`.
