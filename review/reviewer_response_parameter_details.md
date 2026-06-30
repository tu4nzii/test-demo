# Reviewer Response Draft: Parameters, Reproducibility, And Verification Details

## Comment 1.7

**Comment:** How are the parameters chosen for Gaussian filters and Canny edge detection? Are they fixed across all images? How sensitive is performance to these parameters?

**Response:** The parameters are fixed for all images within each chart family; no per-image tuning is used. The role of Gaussian filtering and Canny edge detection differs by chart family.

For Cartesian charts, the active grid reconstruction pipeline does not use global Gaussian smoothing or Canny edge detection. It detects grid evidence from low-saturation gray pixels with sufficient local contrast, applies morphological horizontal/vertical line filtering, and then selects among three grid candidates using OCR/MLLM axis evidence. The only Gaussian operation in this branch is a local `(3,3), sigma=0` smoothing step for OCR text-crop thresholding.

Latest-code verification was run on the current generated artifacts. The Cartesian audit found `650` adjudication records, and all `650` contained the three candidate sources used by the active flow. The same audit found `1176` final-binding files, `650` grid status reports, and `14` failure/exit reports. On the current Cartesian evaluation set, `325` samples were audited and `317` were processed; the generated final bindings achieved Tick MAE `0.691 px`, Tick Acc@2px `96.37%`, tick-position MAE `0.849 px`, and label accuracy `96.13%`.

For pie/donut charts, plot-area detection is color-mask first: HSV saturation `>12`, value `>40`, morphology with a `(5,5)` elliptical kernel, connected-component filtering, and `minEnclosingCircle`. Hough circle detection is used only as fallback, with Gaussian `(9,9), sigma=2` and `HoughCircles(dp=1.2, param1=50, param2=30)`.

For radar/rose charts, radial-ring detection uses fixed Gaussian `(9,9), sigma=2` followed by Hough circle detection. The first-ring setting uses `dp=1.2`, `minDist=100`, `param1=20`, and `param2=30`; the second-ring setting uses `param2=50`. Radar and rose use different fixed radius ranges based on image height.

To quantify sensitivity, two bounded diagnostics were run. The Cartesian diagnostic isolates the legacy Canny/Hough candidate generator. It is not the active Cartesian runtime path, so it is reported only as a low-level sensitivity check.

| Setting | Avg. Hough line candidates | Change from baseline |
| --- | ---: | ---: |
| no blur, Canny 20/80, Hough 15 | 181.00 | -0.04% |
| **no blur, Canny 30/100, Hough 15** | **181.08** | **0.00%** |
| no blur, Canny 50/150, Hough 15 | 181.59 | +0.28% |
| no blur, Canny 70/210, Hough 15 | 178.87 | -1.22% |
| 3x3 blur, Canny 30/100, Hough 15 | 189.35 | +4.56% |
| 5x5 blur, Canny 30/100, Hough 15 | 188.07 | +3.86% |
| no blur, Canny 30/100, Hough 10 | 203.64 | +12.45% |
| no blur, Canny 30/100, Hough 20 | 159.53 | -11.90% |

This diagnostic indicates that moderate Canny thresholds have little effect on candidate count, while the Hough threshold changes the number of candidates more substantially.

For radar/rose, Hough `param2` was swept on 99 polar samples. Ground truth was used only for offline scoring.

| HoughCircles `param2` | Samples | Circle found rate | Median center error | Median best radius error | Mean candidate count |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 99 | 100.00% | 2.236 px | 1.000 px | 50.83 |
| **30** | **99** | **100.00%** | **2.236 px** | **1.000 px** | **26.87** |
| 40 | 99 | 97.98% | 2.236 px | 2.000 px | 5.60 |
| 50 | 99 | 76.77% | 2.236 px | 2.000 px | 2.21 |

The selected `param2=30` keeps the same detection return rate as `param2=20` while reducing the candidate count. In the full evaluation, Cartesian charts achieve Tick MAE `0.691 px`, Tick Acc@2px `96.37%`, label accuracy `96.13%`, and chart-type accuracy `100.00%`; polar chart-type accuracy is `96.64%`. For chart-type accuracy, `bubble` and `scatter` are mutually accepted as point-chart types.

The current-code polar diagnostic at the runtime setting `param2=30` returned circles for all `99/99` polar samples, with median center error `2.236 px`, median first-candidate radius error `2.000 px`, median best-radius error `1.000 px`, and mean candidate count `26.87`. These diagnostics use ground truth only for offline measurement; generation uses only the image-derived grid, OCR/MLLM evidence, and generated geometry.

## Comment 2.6[2]

**Comment:** Some technical details remain under-specified. Key implementation details that affect reproducibility are not described in enough depth, including the robustness of chart type classification, parameter settings for axis/tick extraction, and how the zoom-in verification mechanism is actually implemented.

**Response:** The implementation has three relevant components: chart type classification, grid/axis/tick extraction, and zoom-in verification during value prediction.

**Chart type classification.** Chart type classification is performed by an MLLM JSON extractor with `temperature=0`. The output type must be one of the registered types: `rose`, `radar`, `v_bar`, `h_bar`, `line`, `scatter`, `bubble`, `donut`, or `pie`. Missing or unsupported types produce an explicit error rather than a default fallback. In the reported chart-type accuracy, `bubble` and `scatter` are mutually accepted as point-chart types. The latest full evaluation gives chart-type accuracy of `98.95%` overall, `100.00%` for Cartesian charts, and `96.64%` for polar charts.

**Grid/axis/tick extraction.** Cartesian charts use fixed mask and morphology parameters: saturation `<=70`, gray range `[95,255]`, local contrast `>=7`, `min_line_frac=0.055`, `gap_frac=0.006`, `max_thickness_frac=0.008`, `min_grid_span_frac=0.18`, `min_grid_lines=2`, `cluster_tolerance=3 px`, and `grid_thickness=1 px`. OCR filtering uses `ocr_min_score=0.45`, detection threshold `0.35`, box threshold `0.60`, unclip ratio `1.15`, and side limit `960`.

Pie/donut charts use a 15-degree angular grid. Their circular plot area is detected by color mask first and by Hough circle fallback only when needed. Radar/rose charts use fixed Gaussian-Hough radial-ring detection with `tick_density=2`; radial tick values are read from the image by the MLLM.

The current-code evidence files also confirm the intended Cartesian selection mechanism: every audited adjudication record included the three candidate sources, and the final evaluation read the generated `final_bindings` rather than any reference annotations. Reference JSON files are used only by offline evaluation scripts to compute the reported errors.

**Zoom-in verification.** Zoom-in verification is part of the value-prediction stage, not grid reconstruction. Bar, scatter, and bubble charts crop around the predicted target and verify target visibility before accepting or retrying the crop. Line charts crop around the target x category and current y estimate and record a contains-target diagnostic. Radar/rose prediction uses both the generated grid image and the original image, with detected `r_ticks`, `theta_ticks`, `theta_angles`, and color hints in the prompt; when target-level prediction is incomplete, a whole-chart prompt is used. Pie/donut prediction refines the estimated sector over three crop-and-zoom rounds: `(pad=15 deg, grid=5 deg, zoom=2.0)`, `(pad=9 deg, grid=3 deg, zoom=2.0)`, and `(pad=6 deg, grid=2 deg, zoom=3.0)`, with an LLM contains-sector check for sector order and visibility.
