# 审稿回复证据对账

本文档用于核查 `review/reviewer_response_parameter_details.md` 与 `review/reviewer_response_parameter_details_zh.md` 中的关键表述是否有实验或代码事实支撑。它不是论文正文，只是过程审计文件。

## 结论边界

- `parameter_sensitivity_*` 只支持“旧版低层 Canny/Hough 候选生成器的候选数量变化”这一类结论。
- 它不支持“当前直角系全链路性能由某个 Canny/Hough 参数导致”这类因果结论。
- 当前系统实际可跑的直角系路径是 enhanced-grid-first：`backend.main.process_chart_image` -> `ChartProcessorFactory` -> `Grid_generation.grid_generation.process_chart` -> `_process_chart_with_enhanced_grid_only` -> `grid_line_filter.process_image`。
- 旧版 Canny/Hough scan block 仍保留在源码后半段，但位于 `process_chart` 的 early return 之后；当前 enhanced-grid-first 成功时直接返回，失败时也直接退出，不会继续执行旧版扫描。
- 极坐标部分需要分开看：Pie/Donut 当前是颜色 mask 优先、Hough fallback；Radar/Rose 当前使用固定 Gaussian+Hough 圆环检测，并结合 MLLM 读取径向 tick。
- GT 只用于离线实验和评估打分，不进入生成端流程。

## 关键表述与证据

| 审稿回复中的表述 | 证据来源 | 是否直接支持 | 当前写法处理 |
| --- | --- | --- | --- |
| 当前直角系运行路径是 enhanced-grid-first，不是旧版 Canny/Hough scan | `backend/Grid_generation/grid_generation.py` 中 `process_chart` 先调用 `_process_chart_with_enhanced_grid_only`，成功即返回；失败后返回 `None`；`review/implementation_parameter_inventory.md` | 是 | 作为当前流程事实保留。 |
| 当前直角系候选由灰度/饱和度/局部对比度 mask 与形态学线段过滤产生 | `backend/Grid_generation/grid_masks.py::make_line_masks`；`backend/Grid_generation/grid_generation.py::_run_enhanced_cartesian_grid_reconstruction` | 是 | 作为当前参数事实保留。 |
| 当前直角系 mask 参数为 `sat_max=70`、`min_gray=95`、`contrast_min=7`、`min_line_frac=0.055` 等 | `backend/Grid_generation/grid_generation.py::_run_enhanced_cartesian_grid_reconstruction`；`backend/Grid_generation/grid_masks.py` | 是 | 写入直角系参数表。 |
| 旧版 Canny/Hough 诊断参数为 Canny `30/100`、Hough threshold `15` 等 | `review/parameter_sensitivity_summary.csv`；旧版残留代码 | 是，但仅支持 legacy diagnostic | 明确标注为旧版诊断，不写成当前流程参数。 |
| OCR 局部文本裁剪使用 Gaussian `(3,3), sigma=0` | `backend/Grid_generation/grid_ocr.py`；`review/implementation_parameter_inventory.md` | 是 | 限定为局部 OCR/text crop，不写成全局线段检测。 |
| Pie/Donut 绘图区检测颜色 mask 优先，Hough 只是 fallback | `backend/Grid_generation/circular_angle_grid.py::detect_circular_plot_area` 与 `_detect_circular_plot_area_hough` | 是 | 在 Comment 1.7 和 Comment 2.6 中作为极坐标/圆形分支单独说明。 |
| Pie/Donut 角度网格使用 `ANGLE_STEP_DEGREES=15` | `backend/Grid_generation/circular_angle_grid.py` | 是 | 写入极坐标/圆形参数表。 |
| Radar/Rose 使用 Gaussian `(9,9), sigma=2` 和固定 HoughCircles 参数 | `backend/demo_radar/demo_radar_circle_find.py`；`backend/demo_rose/demo_rose_circle_find.py` | 是 | 写入极坐标/圆形参数表。 |
| Radar/Rose 使用 `tick_density=2`，径向 tick 由 MLLM 读取，生成端不使用 GT | `backend/demo_radar/demo_radar_circle_find.py`；`backend/demo_rose/demo_rose_circle_find.py` | 是 | 写入极坐标处理说明。 |
| Radar/Rose Hough `param2=30` 的敏感性诊断结果 | `review/axis_prior_reviewer_eval.json`；`review/polar_parameter_sensitivity_report.md` | 是，但仅支持 Radar/Rose 圆环候选诊断 | 明确说明不是 Pie/Donut 端到端实验，也不是直角系实验。 |
| 当前完整直角系流程生成 `combined_mask`、`tick_supplement`、`semantic_guide` 三套候选并做 score 筛选/退出 | `review/current_cartesian_full_pipeline_evidence.json`；`backend/Grid_generation/grid_line_filter.py`；`backend/Grid_generation/grid_adjudication.py` | 是 | 用于说明当前实验和主流程边界。 |
| 当前完整直角系 Tick MAE `0.691 px`、Tick Acc@2px `96.37%`、Label Acc `96.13%` | `review/current_cartesian_full_pipeline_evidence.json` | 是 | 作为当前全链路结果保留。 |
| 当前图表分类准确率 overall `98.95%`、Cartesian `100.00%`、Polar `96.64%` | `backend/evaluation/results/vishintprompt_full_latest_report/details.json` 与报告汇总 | 是 | 作为评估事实保留；该指标中 `bubble` 与 `scatter` 按点图族互认。 |
| 图表类型分类使用 `temperature=0`、JSON response、注册表校验，unsupported/missing type 显式报错 | `backend/type_detection/chart_type.py`；`backend/type_detection/chart_registry.py` | 是 | 作为实现事实保留。 |
| zoom-in verification 属于 3.评估预测，而不是网格生成阶段 | `backend/evaluation_prediction/chart_modules/*`；`review/implementation_parameter_inventory.md` | 是 | 在回复中明确区分。 |
| 柱状图、散点图、气泡图等 crop gating/重试机制 | `backend/evaluation_prediction/chart_modules/v_bar/runner.py`、`h_bar/runner.py`、`scatter/runner.py`、`bubble/runner.py` | 是 | 按类型描述，不泛化到所有类型。 |
| 折线图 contains-target check | `backend/evaluation_prediction/chart_modules/line/runner.py` | 部分支持 | 写成日志/诊断检查后再基于局部 crop 预测，不写成强 gating。 |
| Radar/Rose 评估预测使用网格图和原图，提示词包含 `r_ticks`、`theta_ticks`、`theta_angles`、颜色提示，并在缺失目标级预测时 fallback whole-chart | `backend/evaluation_prediction/chart_modules/polar_value.py` | 是 | 写入 zoom-in/评估预测段落。 |
| Pie/Donut amplifier 三轮 crop 参数和 contains-sector 校验 | `backend/evaluation_prediction/chart_modules/pie/visual.py`；`backend/evaluation_prediction/chart_modules/donut/visual.py` | 是 | 写入 zoom-in/评估预测段落。 |

## 已删除或收紧的表述

| 原表述类型 | 原因 | 当前处理 |
| --- | --- | --- |
| “Gaussian blur 没有带来更可靠的后续结果” | 参数实验没有重跑完整后续流程，不能支持该因果结论。 | 改为“候选数量多于旧版 no-blur 诊断基线”。 |
| “高 Canny 阈值抑制弱/细边缘” | 实验表格只统计候选数量，未逐线验证边缘强弱。 | 改为“候选数量少于旧版诊断基线”。 |
| “Hough `20` 漏掉细线/tick 风险更高” | 当前实验只直接支持候选数减少。 | 改为“候选数量显著少于旧版诊断基线”。 |
| “zoom-in verification 对所有类型都先验证再接受 crop” | 不同 chart runner 的 gating 严格程度不同。 | 改为“对实现了显式 gating 的类型验证目标可见；折线图记录诊断检查；Radar/Rose 使用 grid/baseline 双提示与 whole-chart fallback”。 |
| “Canny/Hough 参数是当前直角系轴线/tick 提取参数” | 当前系统实际路径已经切到 enhanced-grid-first，旧版 Canny/Hough block 不可达。 | 改为“当前 runtime 参数表 + 旧版 Canny/Hough diagnostic 表”。 |
| “极坐标只有 Gaussian `(9,9), sigma=2`” | 这种写法太粗，会漏掉 Pie/Donut 的颜色 mask 优先和 Radar/Rose 的 Hough/MLLM tick 机制。 | 拆分为 Pie/Donut 与 Radar/Rose 两个分支。 |

## 可复现文件

- 参数实验脚本：`review/parameter_sensitivity_experiment.py`
- 旧版 Canny/Hough 参数实验汇总：`review/parameter_sensitivity_summary.csv`
- 旧版 Canny/Hough 参数实验逐样本结果：`review/parameter_sensitivity_samples.csv`
- Radar/Rose polar 诊断输出：`review/axis_prior_reviewer_eval.json`
- Radar/Rose polar 诊断报告：`review/polar_parameter_sensitivity_report.md`
- 当前直角系全链路证据脚本：`review/current_cartesian_full_pipeline_evidence.py`
- 当前直角系全链路证据 JSON：`review/current_cartesian_full_pipeline_evidence.json`
- 参数实现清单：`review/implementation_parameter_inventory.md`
- 审稿回复英文版：`review/reviewer_response_parameter_details.md`
- 审稿回复中文版：`review/reviewer_response_parameter_details_zh.md`
