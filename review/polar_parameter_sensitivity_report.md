# Polar Parameter Sensitivity Diagnostic

This diagnostic covers the Radar/Rose radial-circle detector only. It does not evaluate the Cartesian enhanced-grid-first pipeline, and it does not represent Pie/Donut end-to-end performance because Pie/Donut plot-area detection uses a color-mask-first path with Hough only as fallback.

Source command:

```bash
python backend/evaluation/scripts/evaluate_axis_prior_reviewer_questions.py --output review/axis_prior_reviewer_eval.json
```

Source output: `review/axis_prior_reviewer_eval.json`.

## Radar/Rose HoughCircles `param2` Sweep

Fixed settings in this diagnostic: Gaussian `(9,9), sigma=2`, `dp=1.2`, `minDist=100`, `param1=20`, radar radius range `height/5` to `height/4`, rose radius range `height/4` to `height`.

| HoughCircles `param2` | Samples | Circle found rate | Median center error | Median first radius error | Median best radius error | Mean candidate count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 99 | 100.00% | 2.236 px | 1.000 px | 1.000 px | 50.83 |
| **30** | **99** | **100.00%** | **2.236 px** | **2.000 px** | **1.000 px** | **26.87** |
| 40 | 99 | 97.98% | 2.236 px | 2.000 px | 2.000 px | 5.60 |
| 50 | 99 | 76.77% | 2.236 px | 2.000 px | 2.000 px | 2.21 |

## Interpretation Boundary

- `param2=30` is the fixed runtime value for first-ring detection in the Radar/Rose code path.
- In this offline diagnostic, `param2=30` keeps the same 100% circle-found rate as `param2=20` while reducing the mean candidate count from `50.83` to `26.87`.
- More restrictive values reduce the candidate count further but also reduce circle-found rate, especially `param2=50`.
- Ground truth is used only for this offline scoring report. It is not used by the generation path.
