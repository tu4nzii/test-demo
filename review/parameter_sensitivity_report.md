# Parameter Sensitivity Experiment

This folder contains an offline reproducibility experiment for the reviewer response.
Ground truth JSON files are used only for scoring in this experiment; they are not used by the runtime generation pipeline.

Scope note: this experiment isolates the legacy low-level Canny/Hough candidate generator. It is not the active Cartesian runtime path and is not an end-to-end rerun of the current Cartesian grid reconstruction pipeline. The current runtime pipeline uses enhanced-grid-first mask reconstruction, constructs three grid candidates (`combined_mask`, `tick_supplement`, and `semantic_guide`), applies score-based selection and exit checks, and writes `final_bindings`; the full-pipeline metrics below come from that latest pipeline report.

## Dataset

- Dataset root: `F:\program\test-demo\backend\datasets\VisHintPrompt_datasets`
- Cartesian samples evaluated: 324
- Types: bubble, h_bar, line, scatter, v_bar

## Active Cartesian Runtime Parameters

The current system path calls `_process_chart_with_enhanced_grid_only` and `grid_line_filter.process_image`. Its fixed parameters are:

| Parameter | Value |
| --- | --- |
| Neutral grid mask | saturation <= 70; gray range [95, 255]; local contrast >= 7 |
| Optional dark candidates | disabled by default; dark cutoff 80 when enabled |
| Morphological line length | min_line_frac 0.055 of image width/height; lower bound 15 px |
| Gap closing | gap_frac 0.006 of image width/height; lower bound 3 px |
| Component thickness filter | max_thickness_frac 0.008 of shorter side; lower bound 3 px |
| Grid geometry reconstruction | min_grid_span_frac 0.18; min_grid_lines 2; cluster_tolerance 3 px; grid_thickness 1 px |
| Tick supplement from dark axis/tick evidence | tick_dark_cutoff 150 |
| OCR filtering | ocr_min_score 0.45; det_thresh 0.35; det_box_thresh 0.60; det_unclip_ratio 1.15; det_limit_side_len 960 |

## Legacy Canny/Hough Diagnostic Settings

| Parameter | Value |
| --- | --- |
| Gaussian smoothing before Canny | none for the legacy line detector; `(3,3), sigma=0` only for local OCR crop thresholding |
| Gaussian settings tested in the sweep | none; `(3,3), sigma=0`; `(5,5), sigma=1` |
| Canny thresholds | 30 / 100 |
| Probabilistic Hough threshold | 15 |
| Hough min line length / max gap | 20 px / 20 px |
| Legacy tick scan range | 20 px |
| Legacy tick merge angle tolerance | 10 degrees |

## Main Result

This reviewer-facing summary reports candidate-volume stability. Internal diagnostic columns such as candidate hits near GT axis positions are retained only in `parameter_sensitivity_samples.csv` and `parameter_sensitivity_summary.csv`.

| Setting | Avg Hough line candidates | Change from baseline | Interpretation |
| --- | ---: | ---: | --- |
| none__canny_20_80__hough15_l20_g20 | 181.003 | -0.04% | Similar candidate volume to the legacy diagnostic baseline. |
| none__canny_30_100__hough15_l20_g20 | 181.083 | +0.00% | Legacy Canny/Hough diagnostic baseline. |
| none__canny_50_150__hough15_l20_g20 | 181.590 | +0.28% | Similar candidate volume to the legacy diagnostic baseline. |
| none__canny_70_210__hough15_l20_g20 | 178.870 | -1.22% | Fewer candidates than the legacy diagnostic baseline. |
| g3_s0__canny_20_80__hough15_l20_g20 | 189.772 | +4.80% | More candidates than the legacy no-blur baseline. |
| g3_s0__canny_30_100__hough15_l20_g20 | 189.349 | +4.56% | More candidates than the legacy no-blur baseline. |
| g3_s0__canny_50_150__hough15_l20_g20 | 185.201 | +2.27% | More candidates than the legacy no-blur baseline. |
| g3_s0__canny_70_210__hough15_l20_g20 | 181.614 | +0.29% | Similar candidate volume to the legacy diagnostic baseline. |
| g5_s1__canny_20_80__hough15_l20_g20 | 189.781 | +4.80% | More candidates than the legacy no-blur baseline. |
| g5_s1__canny_30_100__hough15_l20_g20 | 188.071 | +3.86% | More candidates than the legacy no-blur baseline. |
| g5_s1__canny_50_150__hough15_l20_g20 | 183.151 | +1.14% | More candidates than the legacy no-blur baseline. |
| g5_s1__canny_70_210__hough15_l20_g20 | 179.213 | -1.03% | Fewer candidates than the legacy diagnostic baseline. |
| none__canny_30_100__hough10_l20_g20 | 203.639 | +12.46% | Substantially more candidates in the legacy detector. |
| none__canny_30_100__hough20_l20_g20 | 159.534 | -11.90% | Substantially fewer candidates in the legacy detector. |

## Baseline By Chart Type

| Type | Samples | Avg Hough line candidates |
| --- | ---: | ---: |
| bubble | 59 | 215.542 |
| h_bar | 62 | 169.581 |
| line | 73 | 181.918 |
| scatter | 59 | 234.407 |
| v_bar | 71 | 117.324 |

## Current Full-Pipeline Reference

The latest full-pipeline report is included as the end-to-end Cartesian evidence. It evaluates the active enhanced-grid-first three-candidate scoring/exit pipeline and uses generated `final_bindings`.
- Cartesian samples: 325
- Cartesian processed samples: 317
- Cartesian Tick MAE: 0.691 px
- Cartesian Tick Acc@2px: 96.37%
- Cartesian Tick position MAE: 0.849 px
- Cartesian label accuracy: 96.13%

## Files

- `parameter_sensitivity_samples.csv`: per-sample, per-setting measurements.
- `parameter_sensitivity_summary.csv`: aggregate measurements for each parameter setting.
- `parameter_sensitivity_baseline_by_type.csv`: baseline results by chart type.
- `parameter_sensitivity_summary.json`: machine-readable metadata and summary.
