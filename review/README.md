# Review 补充材料

本目录用于回复审稿意见中关于参数选择、复现细节、分类鲁棒性和 zoom-in verification 机制不够清楚的问题。

## 文件说明

- `parameter_sensitivity_experiment.py`：旧版 Canny/Hough 候选生成器的离线参数敏感性诊断脚本。只读取数据集图片和 GT JSON 做离线打分，不参与系统生成端，也不会改变主流程。
- `parameter_sensitivity_samples.csv`：逐样本、逐参数组合的检测结果，便于检查每张图的轴线候选和数值轴 tick/grid-line 候选。
- `parameter_sensitivity_summary.csv`：14 组 Gaussian/Canny/Hough 参数的总体结果。
- `parameter_sensitivity_baseline_by_type.csv`：旧版 Canny/Hough 诊断基线参数在各直角系类别上的结果。
- `parameter_sensitivity_summary.json`：机器可读的实验元信息、参数配置和汇总结果。
- `parameter_sensitivity_report.md`：自动生成的实验报告。
- `parameter_sensitivity_report_zh.md`：参数敏感性实验报告中文版。
- `implementation_parameter_inventory.md`：当前方法中关键参数、固定策略和代码位置说明。
- `implementation_parameter_inventory_zh.md`：实现参数清单中文版。
- `reviewer_response_parameter_details.md`：面向审稿意见 Comment 1.7 和 Comment 2.6[2] 的英文回复草稿。
- `reviewer_response_parameter_details_zh.md`：面向审稿意见 Comment 1.7 和 Comment 2.6[2] 的中文回复草稿。
- `response_claim_evidence_audit.md`：审稿回复关键表述与实验/代码证据的逐项对账。
- `verification_log.md`：本轮事实核查中复算的关键数值、来源文件和实现位置。
- `current_cartesian_full_pipeline_evidence.py`：核验当前直角系完整流程 artifacts 的脚本，不调用旧版轴扫描，不调用模型。
- `current_cartesian_full_pipeline_evidence.md` / `current_cartesian_full_pipeline_evidence_zh.md`：当前三套网格候选 score 筛选/退出机制的证据报告。
- `current_cartesian_full_pipeline_evidence.json`：上述核验证据的机器可读结果。
- `axis_prior_reviewer_eval.json`：Radar/Rose 径向圆环检测的 Hough `param2` 离线诊断输出；同时包含脚本输出的其他旧版字段，审稿回复只引用 polar 相关部分。
- `polar_parameter_sensitivity_report.md` / `polar_parameter_sensitivity_report_zh.md`：Radar/Rose Hough `param2` 诊断报告。该报告不代表 Pie/Donut 端到端实验，Pie/Donut 当前是颜色 mask 优先、Hough fallback。

## 实验口径

旧版参数敏感性诊断使用 `backend/datasets/VisHintPrompt_datasets` 中的 324 张直角系图表，包括 `v_bar`、`h_bar`、`line`、`scatter` 和 `bubble`。最新完整流程评估的直角系口径为 325 张样本、317 张已处理样本。GT 只用于离线评分，生成端仍然只依赖系统自身的 OCR/MLLM/网格重建输出。

注意：该参数敏感性实验只隔离考察旧版低层 Canny/Hough 候选生成器，不是当前直角系完整主流程的端到端重跑，也不是系统实际运行路径。当前主流程是 enhanced-grid-first，会生成 `combined_mask`、`tick_supplement`、`semantic_guide` 三套网格候选，经过 score 筛选和退出检查后写出 `final_bindings`；审稿回复中的最终直角系效果使用 `backend/evaluation/results/vishintprompt_full_latest_report` 的最新完整流程结果。

当前系统实际运行路径的直角系候选生成基线是：中性网格 mask 使用 `sat_max=70`、`white_cutoff=255`、`min_gray=95`、`contrast_min=7`；形态学线段过滤使用 `min_line_frac=0.055`、`gap_frac=0.006`、`max_thickness_frac=0.008`；网格几何重建使用 `min_grid_span_frac=0.18`、`min_grid_lines=2`、`cluster_tolerance=3`、`grid_thickness=1`；OCR 过滤使用 `ocr_min_score=0.45`、`ocr_det_thresh=0.35`、`ocr_det_box_thresh=0.60`、`ocr_det_unclip_ratio=1.15`。

极坐标/圆形图表在回复中拆分为两类：Pie/Donut 使用颜色 mask 优先的圆形绘图区检测与 15 度角度网格，Hough 只作为 fallback；Radar/Rose 使用固定 Gaussian `(9,9), sigma=2` 和 HoughCircles 圆环检测，并结合 MLLM 读取径向 tick。`polar_parameter_sensitivity_report*` 只支撑 Radar/Rose 的 Hough `param2` 选择说明。

重新生成实验结果：

```powershell
python review\parameter_sensitivity_experiment.py
```
