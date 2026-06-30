# 审稿回复事实核查日志

核查日期：2026-06-30

## 1. 当前直角系系统路径核查

代码核查结论：

- 系统入口：`backend.main.process_chart_image` -> `ChartProcessorFactory` -> `Grid_generation.grid_generation.process_chart`。
- 当前直角系网格生成：`process_chart` 先调用 `_process_chart_with_enhanced_grid_only`，后者调用 `_run_enhanced_cartesian_grid_reconstruction` 和 `grid_line_filter.process_image`。
- `process_chart` 在 enhanced-grid-first 成功时直接返回；如果失败，当前代码记录 legacy CV Cartesian flow disabled 并返回 `None`。
- 因此，`process_chart` 后半段残留的 Canny/Hough + `scan_pixels_for_ticks` 代码不是当前系统实际会跑的流程。

当前系统路径中的固定参数：

| 参数 | 数值 |
| --- | --- |
| sat_max | 70 |
| white_cutoff | 255 |
| min_gray | 95 |
| contrast_min | 7 |
| include_dark | False |
| dark_cutoff | 80 |
| min_line_frac | 0.055 |
| gap_frac | 0.006 |
| max_thickness_frac | 0.008 |
| min_grid_span_frac | 0.18 |
| min_grid_lines | 2 |
| cluster_tolerance | 3 |
| grid_thickness | 1 |
| tick_dark_cutoff | 150 |
| ocr_min_score | 0.45 |
| ocr_det_thresh | 0.35 |
| ocr_det_box_thresh | 0.60 |
| ocr_det_unclip_ratio | 1.15 |
| ocr_det_limit_side_len | 960 |

## 2. 旧版 Canny/Hough 诊断实验数值

来源文件：`review/parameter_sensitivity_summary.csv`

复核字段：`setting`、`samples`、`avg_line_candidates`，并以 no-blur + Canny `30/100` + Hough `15` 作为 baseline 重新计算百分比变化。

| setting | samples | avg_line_candidates | delta_pct |
| --- | ---: | ---: | ---: |
| none__canny_20_80__hough15_l20_g20 | 324 | 181.003 | -0.04 |
| none__canny_30_100__hough15_l20_g20 | 324 | 181.083 | 0.00 |
| none__canny_50_150__hough15_l20_g20 | 324 | 181.590 | 0.28 |
| none__canny_70_210__hough15_l20_g20 | 324 | 178.870 | -1.22 |
| g3_s0__canny_30_100__hough15_l20_g20 | 324 | 189.349 | 4.56 |
| g5_s1__canny_30_100__hough15_l20_g20 | 324 | 188.071 | 3.86 |
| none__canny_30_100__hough10_l20_g20 | 324 | 203.639 | 12.46 |
| none__canny_30_100__hough20_l20_g20 | 324 | 159.534 | -11.90 |

核查结论：审稿回复中 Comment 1.7 表格只引用这些候选数量和百分比变化，并明确标注为 legacy Canny/Hough diagnostic，不再声称这些参数属于当前系统实际直角系流程，也不声称它们直接导致最终端到端性能变化。

## 3. 当前直角系完整流程指标

来源文件：`review/current_cartesian_full_pipeline_evidence.json`

| 指标 | 数值 |
| --- | ---: |
| sample_count | 325 |
| processed_count | 317 |
| tick_value_mae_px | 0.691468253968254 |
| tick_value_accuracy_2px | 0.9637083470801716 |
| tick_position_mae_px | 0.8492985002418965 |
| label_name_accuracy | 0.9613352545629202 |

核查结论：审稿回复中的 `0.691 px`、`96.37%`、`96.13%` 来自该 JSON 的完整流程结果，而不是参数敏感性实验。

## 4. 图表分类准确率

来源文件：`backend/evaluation/results/vishintprompt_full_latest_report/details.json`

复核方法：逐样本累加 `chart_type_total` 和 `chart_type_correct`，并按 `gt_type` 分为 Cartesian 与 Polar。当前报告口径中，`bubble` 与 `scatter` 按点图族互认。

| scope | total | correct | acc |
| --- | ---: | ---: | ---: |
| overall | 474 | 469 | 98.95% |
| cartesian | 325 | 325 | 100.00% |
| polar | 149 | 144 | 96.64% |

核查结论：审稿回复中的分类准确率来自当前评估结果。

## 5. 实现参数位置

| 事实 | 代码位置 |
| --- | --- |
| 系统上传后处理入口 | `backend/main.py::process_chart_image`、`/api/process/` |
| 直角系 processor 调用 | `backend/type_detection/chart_processor.py::CartesianChartProcessor.encode_image` |
| 当前 enhanced-grid-first 分支 | `backend/Grid_generation/grid_generation.py::_process_chart_with_enhanced_grid_only` |
| 当前 mask 参数传入位置 | `backend/Grid_generation/grid_generation.py::_run_enhanced_cartesian_grid_reconstruction` |
| 当前 mask/形态学候选生成 | `backend/Grid_generation/grid_masks.py::make_line_masks` |
| 当前三候选生成与仲裁调用 | `backend/Grid_generation/grid_line_filter.py` |
| 当前 score-first 仲裁 | `backend/Grid_generation/grid_adjudication.py` |
| 旧版 Canny/Hough 残留但不可达 | `backend/Grid_generation/grid_generation.py` 中 enhanced-grid-first 失败后 return `None` 之后的代码 |
| OCR local Gaussian `(3,3), sigma=0` | `backend/Grid_generation/grid_ocr.py` |
| polar/circular Gaussian `(9,9), sigma=2` | `backend/Grid_generation/circular_angle_grid.py` 与 circle 相关模块 |
| type detection `temperature=0` 与 JSON response | `backend/type_detection/chart_type.py` |
| unsupported/missing chart type 显式报错 | `backend/type_detection/chart_type.py`、`backend/main.py` |
| 评估读取生成端 `final_bindings` | `backend/evaluation/scripts/evaluate_vishintprompt_latest_metrics.py` |

## 6. 已收紧的表述

- 不再把参数敏感性实验说成当前直角系全链路端到端重跑。
- 不再把旧版 Canny/Hough 参数说成当前系统实际直角系流程参数。
- 不再声称 Gaussian blur 会改善或损害最终可靠性。
- 不再把 Hough/Canny 候选数量变化解释成已经验证过的弱边缘、细 tick 漏检。
- zoom-in verification 明确限定为 `3.评估预测` 阶段，并按 chart-specific runner 描述。

## 7. 极坐标/圆形分支补充核查

- Pie/Donut 当前入口为 `backend/type_detection/chart_processor.py::CircularAngleChartProcessor` -> `backend/Grid_generation/circular_angle_grid.py::process_circular_angle_chart`。
- Pie/Donut 绘图区检测优先使用颜色 mask：HSV saturation `>12`、value `>40`、椭圆核 `(5,5)`、open + close、连通域面积阈值 `max(64, width*height*0.003)`，并由 `cv2.minEnclosingCircle` 得到圆形绘图区；HoughCircles 仅为 fallback。
- Radar/Rose 当前入口分别为 `RadarChartEncoder` 与 `RoseChartEncoder`，使用固定 Gaussian `(9,9), sigma=2` 和 HoughCircles 检测径向圆环，并通过 MLLM 读取径向 tick。
- 新增 `review/axis_prior_reviewer_eval.json` 与 `review/polar_parameter_sensitivity_report*.md`。该诊断只支撑 Radar/Rose 的 Hough `param2` 选择说明：`param2=30` 在 99 张 polar 样本上保持 100% circle-found rate，同时把平均候选圆数量从 `50.83` 降到 `26.87`；更高阈值会降低返回率。
- `zoom-in verification` 已进一步明确：Radar/Rose 的 3.评估预测使用 grid/baseline 双图提示、`r_ticks`/`theta_ticks`/`theta_angles`/颜色提示和 whole-chart fallback；Pie/Donut 的 amplifier 使用三轮固定 pad/grid/zoom 参数，并包含 contains-sector 校验。

## 8. 最新代码测试复跑记录

复跑命令：

```bash
python review/current_cartesian_full_pipeline_evidence.py
python backend/evaluation/scripts/evaluate_axis_prior_reviewer_questions.py --output review/axis_prior_reviewer_eval.json
```

当前直角系完整流程证据：

| 指标 | 数值 |
| --- | ---: |
| adjudication records | 650 |
| records with all three candidate sources | 650 |
| final binding files | 1176 |
| grid status report files | 650 |
| failure/exit reports | 14 |
| audited Cartesian samples | 325 |
| processed Cartesian samples | 317 |
| Tick MAE | 0.691 px |
| Tick Acc@2px | 96.37% |
| Tick position MAE | 0.849 px |
| Label accuracy | 96.13% |

当前 polar/radial 诊断在运行参数 `param2=30` 下的结果：

| 指标 | 数值 |
| --- | ---: |
| samples | 99 |
| circle found rate | 100.00% |
| median center error | 2.236 px |
| median first radius error | 2.000 px |
| median best radius error | 1.000 px |
| mean candidate count | 26.87 |

这些结果已经补入完整版审稿回复 `review/reviewer_response_parameter_details.md` 和 `review/reviewer_response_parameter_details_zh.md`。GT/参考 JSON 只用于离线度量，不进入生成端流程。
