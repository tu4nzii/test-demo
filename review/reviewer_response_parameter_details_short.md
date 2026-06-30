# Short Reviewer Response: Parameters And Reproducibility

## Comment 1.7

**Comment:** How are the parameters chosen for Gaussian filters and Canny edge detection? Are they fixed across all images? How sensitive is performance to these parameters?

**Response:** The parameters are fixed within each chart family and are not tuned per image. Gaussian smoothing is used only in localized OCR text processing or in circle/ring detection, with `(3,3), sigma=0` for OCR crops and `(9,9), sigma=2` for circular/radial detection.

Sensitivity was evaluated with bounded diagnostics for edge and circle parameters. Moderate Canny threshold changes produced only small changes in line-candidate count, while changing the Hough threshold had a larger effect on candidate count. For radar/rose radial-ring detection, sweeping Hough `param2` on 99 polar samples showed that `param2=30` preserved a 100% circle-found rate while reducing the average candidate count compared with a looser threshold. In the full evaluation, Cartesian charts achieved Tick MAE `0.691 px`, Tick Acc@2px `96.37%`, label accuracy `96.13%`, and chart-type accuracy `100.00%`; polar chart-type accuracy was `96.64%`. For this metric, `bubble` and `scatter` are mutually accepted as point-chart types.

## Comment 2.6[2]

**Comment:** Some technical details remain under-specified. Key implementation details that affect reproducibility are not described in enough depth, including the robustness of chart type classification, parameter settings for axis/tick extraction, and how the zoom-in verification mechanism is actually implemented.

**Response:** Chart type classification is performed by an MLLM JSON extractor with `temperature=0`. The output must belong to the supported type registry; missing or unsupported types produce an explicit error. In the reported chart-type accuracy, `bubble` and `scatter` are mutually accepted as point-chart types. The latest full evaluation gives chart-type accuracy of `98.95%` overall, `100.00%` for Cartesian charts, and `96.64%` for polar charts.

Grid and tick extraction use fixed parameters across images and are not adjusted for individual images. We clarified the key settings raised by the reviewer, including the fixed use of image thresholds, morphology, line clustering, OCR thresholds, circular/radial structure detection, and `tick_density`. Radial tick values are read from the image by the MLLM.

Zoom-in verification is used during value prediction. It verifies target visibility on cropped regions before accepting or retrying a prediction; pie/donut charts use three crop-and-zoom rounds to refine sector estimates, while radar/rose predictions use both the generated grid image, the original image, and detected tick information.
