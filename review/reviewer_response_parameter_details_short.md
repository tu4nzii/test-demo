# Short Reviewer Response: Parameters And Reproducibility

## Comment 1.7

**Comment:** How are the parameters chosen for Gaussian filters and Canny edge detection? Are they fixed across all images? How sensitive is performance to these parameters?

**Response:** The parameters are fixed within each chart family and are not tuned per image. The system uses different visual evidence according to chart geometry: Cartesian grids are recovered from low-saturation gray pixels with sufficient local contrast and morphological horizontal/vertical line filtering; OCR/MLLM evidence is then used to select the final grid candidate. Pie/donut charts locate the circular plot area mainly through color-mask and connected-component evidence, while radar/rose charts use radial-ring evidence. Gaussian smoothing is therefore used only in localized OCR text processing or in circle/ring detection, with `(3,3), sigma=0` for OCR crops and `(9,9), sigma=2` for circular/radial detection. Canny edge detection is not part of the active Cartesian grid reconstruction.

Sensitivity was evaluated with bounded diagnostics for edge and circle parameters. Moderate Canny threshold changes produced only small changes in line-candidate count, while changing the Hough threshold had a larger effect on candidate count. For radar/rose radial-ring detection, sweeping Hough `param2` on 99 polar samples showed that `param2=30` preserved a 100% circle-found rate while reducing the average candidate count compared with a looser threshold. In the full evaluation, Cartesian charts achieved Tick MAE `0.691 px`, Tick Acc@2px `96.37%`, label accuracy `96.13%`, and chart-type accuracy `100.00%`; polar chart-type accuracy was `96.64%`. For this metric, `bubble` and `scatter` are mutually accepted as point-chart types.

## Comment 2.6[2]

**Comment:** Some technical details remain under-specified. Key implementation details that affect reproducibility are not described in enough depth, including the robustness of chart type classification, parameter settings for axis/tick extraction, and how the zoom-in verification mechanism is actually implemented.

**Response:** Chart type classification is performed by an MLLM JSON extractor with `temperature=0`. The output must belong to the supported type registry: `rose`, `radar`, `v_bar`, `h_bar`, `line`, `scatter`, `bubble`, `donut`, or `pie`. Missing or unsupported types produce an explicit error rather than a default fallback. In the reported chart-type accuracy, `bubble` and `scatter` are mutually accepted as point-chart types. The latest full evaluation gives chart-type accuracy of `98.95%` overall, `100.00%` for Cartesian charts, and `96.64%` for polar charts.

Grid and tick extraction use fixed parameters across images. The fixed settings cover saturation, gray-level, local contrast, morphology, line span, clustering, OCR thresholds, circular plot detection, angular grid spacing, radial-ring detection, and `tick_density`. Together, these parameters recover Cartesian horizontal/vertical grid candidates, pie/donut angular grids, and radar/rose radial tick structures; radial tick values are read from the image by the MLLM.

Zoom-in verification is used during value prediction rather than grid reconstruction. Bar, scatter, and bubble charts crop around the predicted target and verify target visibility before accepting or retrying the crop. Line charts crop around the target x category and current y estimate. Radar/rose prediction uses both the generated grid image and the original image, together with detected radial and angular tick information. Pie/donut prediction refines the estimated sector through three crop-and-zoom rounds and checks sector visibility/order with the MLLM.
