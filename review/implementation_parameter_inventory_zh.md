# 实现参数清单

本文档记录审稿回复中引用的关键实现细节。它是证据清单，不是论文正文。

## 直角坐标系轴线与 Tick 候选生成

- 当前系统实际入口是 `backend.main.process_chart_image` -> `ChartProcessorFactory` -> `Grid_generation.grid_generation.process_chart` -> `_process_chart_with_enhanced_grid_only` -> `grid_line_filter.process_image`。
- `process_chart` 会先调用 `_process_chart_with_enhanced_grid_only`；成功时直接返回其结果。如果该流程失败，当前函数会记录 legacy CV Cartesian flow disabled 并返回 `None`，不会继续落到旧版 Canny/Hough 代码。
- 当前直角系候选线段由 `backend/Grid_generation/grid_masks.py::make_line_masks` 生成，不是旧的 `function_calling/axis/detect_lines.py` 路径。
- 运行时 mask 参数在 `backend/Grid_generation/grid_generation.py::_run_enhanced_cartesian_grid_reconstruction` 中传入：`sat_max=70`，`white_cutoff=255`，`min_gray=95`，`contrast_min=7`，`include_dark=False`，`dark_cutoff=80`，`min_line_frac=0.055`，`gap_frac=0.006`，`max_thickness_frac=0.008`，`min_grid_span_frac=0.18`，`min_grid_lines=2`，`cluster_tolerance=3`，`grid_thickness=1`，`tick_dark_cutoff=150`。
- 当前路径中的 OCR 参数固定为 `ocr_min_score=0.45`，`ocr_det_thresh=0.35`，`ocr_det_box_thresh=0.60`，`ocr_det_unclip_ratio=1.15`，`ocr_det_limit_side_len=960`，`ocr_det_limit_type=max`。
- 旧版 Canny/Hough 代码仍保留在 `grid_generation.py` 的早返回之后，只作为历史/参考代码，不是当前系统会跑的路径。

## 当前直角系网格选择路径

- 当前直角系生成流程不是旧版坐标轴扫描路径。
- 系统会生成三套网格候选：`combined_mask`、`tick_supplement` 和 `semantic_guide`。
- `backend/Grid_generation/grid_adjudication.py::arbitrate_priority_grids` 会根据标签绑定质量、目标 tick 数量、OCR 支持、MLLM 引导和数值轴一致性为候选打分。
- 选择策略是 score-first；只有当 score/位置证据仍然存在歧义时才使用 MLLM 仲裁。
- 最新全量报告 `backend/evaluation/results/vishintprompt_full_latest_report` 评估的是生成后的 `final_bindings`，不是旧版原始 tick 扫描输出。

## OCR 与局部文本处理

- OCR/局部文本阈值化在 `backend/Grid_generation/grid_ocr.py` 中使用小尺度平滑：`cv2.GaussianBlur(gray, (3, 3), 0)`。
- 该 blur 只作用于文本裁剪区域，与当前直角系网格 mask 重建相互独立。

## Pie/Donut 圆形角度网格

- 当前入口是 `backend/type_detection/chart_processor.py::CircularAngleChartProcessor`，调用 `backend/Grid_generation/circular_angle_grid.py::process_circular_angle_chart`。
- 角度 tick 固定为 `ANGLE_STEP_DEGREES=15`。
- 绘图区检测优先使用颜色 mask：HSV 饱和度 `>12`、亮度 `>40`，形态学椭圆核 `(5,5)`，open + close 两次，连通域面积阈值 `max(64, width*height*0.003)`，并通过 `cv2.minEnclosingCircle` 得到圆形绘图区。
- Hough 只是 fallback：Gaussian `(9,9), sigma=2`；`cv2.HoughCircles(dp=1.2, minDist=min(width,height)//3, param1=50, param2=30, minRadius=max(10,min_side//10), maxRadius=max(20,min_side//2))`。
- 角度网格绘制使用 `grid_line_ratio=0.1`，线宽为 `1`。

## Radar/Rose 径向网格

- Radar 和 rose processor 注册在 `backend/type_detection/chart_processor.py` 中，分别调用 `backend/demo_radar/demo_radar_circle_find.py` 中的 `RadarChartEncoder` 和 `backend/demo_rose/demo_rose_circle_find.py` 中的 `RoseChartEncoder`。
- 两者均使用 `tick_density=2`。
- Radar 第一圈：Gaussian `(9,9), sigma=2`，Hough `dp=1.2`，`minDist=100`，`param1=20`，`param2=30`，`minRadius=height/5`，`maxRadius=height/4`。
- Radar 第二圈：Gaussian `(9,9), sigma=2`，Hough `dp=1.2`，`minDist=100`，`param1=20`，`param2=50`，`minRadius=first_r+30`，`maxRadius=height/2`。
- Rose 第一圈：Gaussian `(9,9), sigma=2`，Hough `dp=1.2`，`minDist=100`，`param1=20`，`param2=30`，`minRadius=height/4`，`maxRadius=height`。
- Rose 第二圈：Gaussian `(9,9), sigma=2`，Hough `dp=1.2`，`minDist=100`，`param1=20`，`param2=50`，`minRadius=first_r+30`，`maxRadius=height/2`。
- Radar refinement 包含多边形/圆形证据：Canny `50/150`，`HoughLinesP` threshold `24` 或 `28`，`minLineLength=max(24,min_side*0.07)`，`maxLineGap=15`；圆形网格 mask 使用 `sat<80`、`gray>80`、`gray<245` 等阈值和 `(3,3)` close。
- Radar/Rose 的径向 tick 数值由 MLLM 读取，调用温度为 `temperature=0.5`。生成端 JSON 保存的是系统生成的 `r_ticks` 和几何信息；GT 不进入生成流程。

## 图表类型分类

- 图表类型检测由 `backend/type_detection/chart_type.py` 实现。
- MLLM 调用使用确定性解码，即 `temperature=0`，并要求严格 JSON 响应格式。
- 运行时支持类型注册在 `backend/type_detection/chart_registry.py` 中，包括 `rose`、`radar`、`v_bar`、`h_bar`、`line`、`scatter`、`bubble`、`donut` 和 `pie`。
- 不支持或缺失的图表类型会触发显式错误，而不是回退到默认类型。
- 最新全量报告中的图表类型分类准确率为：整体 `98.95%`，直角坐标系 `100.00%`，极坐标系 `96.64%`；该指标中 `bubble` 与 `scatter` 按点图族互认。

## 3.评估预测阶段的 Zoom-In Verification

- zoom-in verification 属于第三阶段评估预测，不属于网格生成阶段。
- 柱状图使用 `backend/evaluation_prediction/chart_modules/v_bar/runner.py`、`h_bar/runner.py` 以及对应 `visual.py` 裁剪函数。amplifier crop 会围绕预测数值和类别位置生成；如果 MLLM 判断裁剪中没有目标对象，则移动窗口并重试。
- 折线图使用 `backend/evaluation_prediction/chart_modules/line/runner.py` 和 `line/visual.py`，围绕目标 x 类别与当前 y 估计进行裁剪；contains-target 检查用于诊断记录，然后基于局部 crop 预测。
- 散点图和气泡图使用 `scatter/runner.py`、`bubble/runner.py` 中的 `feedback_crop_adaptive`；系统围绕预测 mark 生成裁剪，估计目标 mark 直径，并在多次尝试中扩大 crop size，直到目标被验证存在。
- Radar 和 rose 的数值预测由 `backend/evaluation_prediction/chart_modules/polar_value.py` 实现；模型调用使用 `temperature=0.0`，提示词包含生成得到的 `r_ticks`、`theta_ticks`、`theta_angles` 和颜色提示，并同时使用网格图和原图。当目标级预测缺失时，会 fallback 到 whole-chart prompt。
- Pie 和 donut 使用 `pie/visual.py`、`donut/visual.py` 中的 `crop_sector_for_amplifier` 对预测扇区进行裁剪放大，随着裁剪/zoom 更新局部圆心，并通过固定三轮 amplifier 细化预测。第 1 轮 `pad=15°`、`grid=5°`、`zoom=2.0`；第 2 轮 `pad=9°`、`grid=3°`、`zoom=2.0`；第 3 轮 `pad=6°`、`grid=2°`、`zoom=3.0`。
- Pie/donut 扇区裁剪包含 LLM contains-sector 校验；当上一轮 start/end 顺序不一致时，可以触发 swapped-angle recrop。

## 诊断输出

- 旧版直角系 Canny/Hough 候选诊断：`review/parameter_sensitivity_summary.csv` 和 `review/parameter_sensitivity_report.md`。
- Radar/Rose Hough `param2` 诊断：`review/axis_prior_reviewer_eval.json`。
- 当前直角系完整流程证据：`review/current_cartesian_full_pipeline_evidence.json`。
- 表述与证据边界审计：`review/response_claim_evidence_audit.md`。
