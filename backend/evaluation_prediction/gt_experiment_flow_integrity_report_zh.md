# GT 实验版流程完整性测试报告

- 生成时间：2026-07-01 17:00:56
- 测试方式：每个图表类型取 1 个样本，不调用模型；实际加载 GT、grid-with-grid、提示词、feedback 图、amplifier crop。
- 判据：GT 只能作为输入映射、过程日志和指标真值；生成端提前停止和 crop 中心不能使用 GT 作为答案。

## 总览

| 类型 | 结论 | 样本 | 目标数 | grid 来源 | 过程图数量 |
| --- | --- | --- | ---: | --- | ---: |
| bubble | PASS | synthetic/bubble_023 | 8 | fallback-rendered | 2 |
| scatter | PASS | synthetic/scatter_001 | 15 | fallback-rendered | 2 |
| line | PASS | synthetic/line_001 | 10 | fallback-rendered | 2 |
| v_bar | PASS | synthetic/v_bar_002 | 5 | fallback-rendered | 2 |
| h_bar | PASS | synthetic/h_bar_001 | 3 | fallback-rendered | 2 |
| pie | WARN | synthetic/001 | 2 | fallback-rendered | 2 |
| donut | PASS | synthetic/donut_000 | 9 | fallback-rendered | 2 |
| radar | PASS | synthetic/radar_000 | 20 | fallback-rendered | 2 |
| rose | PASS | synthetic/rose_000 | 9 | existing-grid-with-grid | 2 |

## bubble

- 结论：PASS
- 样本：synthetic/bubble_023，sample_id=2792514cd74e9159
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Bubble_50\chart_configs\bubble_023.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Bubble_50\charts\bubble_023.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Bubble_50\bubble_023\bubble_023_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：S1，目标总数=8

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/bubble_023 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Bubble_50\chart_configs\bubble_023.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=8 |
| 原图 | PASS | 800x800, variance=238.04, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Bubble_50\charts\bubble_023.png |
| grid-with-grid 图 | PASS | 800x800, variance=268.77, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Bubble_50\bubble_023\bubble_023_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0009, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| GT tick-pixel 映射进入 grid/feedback prompt | PASS |  |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0521 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心不回退 GT | PASS |  |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\bubble\bubble_023\tempy\overlay_bubble_023_S1_feedback_grid_with_grid_run1.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\bubble\bubble_023\raw_crops\S1_round1_crop.png`

## scatter

- 结论：PASS
- 样本：synthetic/scatter_001，sample_id=cb54104704986e23
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Scatter_50\chart_config\scatter_001.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Scatter_50\charts\scatter_001.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Scatter_50\scatter_001\scatter_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：ITA，目标总数=15

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/scatter_001 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Scatter_50\chart_config\scatter_001.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=15 |
| 原图 | PASS | 500x400, variance=907.91, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Scatter_50\charts\scatter_001.png |
| grid-with-grid 图 | PASS | 500x400, variance=953.36, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Scatter_50\scatter_001\scatter_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0016, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| GT tick-pixel 映射进入 grid/feedback prompt | PASS |  |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0910 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心不回退 GT | PASS |  |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\scatter\scatter_001\tempy\overlay_scatter_001_ITA_feedback_grid_with_grid_run1.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\scatter\scatter_001\raw_crops\ITA_round1_crop.png`

## line

- 结论：PASS
- 样本：synthetic/line_001，sample_id=201ae44a17e02ef9
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Line_50\chart_configs\line_001.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Line_50\charts\line_001.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Line_50\line_001\line_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：Favorable, 2011，目标总数=10

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/line_001 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Line_50\chart_configs\line_001.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=10 |
| 原图 | PASS | 800x500, variance=491.68, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Line_50\charts\line_001.png |
| grid-with-grid 图 | PASS | 800x500, variance=539.74, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Line_50\line_001\line_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0027, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| GT tick-pixel 映射进入 grid/feedback prompt | PASS |  |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0060 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心来源 | PASS | 使用上一轮预测或类别轴定位；GT 仅用于记录/指标 |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\line\line_001\tempy\final_overlay_line_001_Favorable, 2011.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\line\line_001\tempy\amplifier_crop_Favorable, 2011_round1.png`

## v_bar

- 结论：PASS
- 样本：synthetic/v_bar_002，sample_id=23a71053293a3b6e
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\vBar_50\chart_config\v_bar_002.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\vBar_50\chart\v_bar_002.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\vBar_50\v_bar_002\v_bar_002_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：Spending in billion Mexican pesos, 2020*，目标总数=5

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/v_bar_002 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\vBar_50\chart_config\v_bar_002.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=5 |
| 原图 | PASS | 1400x1000, variance=5437.11, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\vBar_50\chart\v_bar_002.png |
| grid-with-grid 图 | PASS | 1400x1000, variance=5350.97, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\vBar_50\v_bar_002\v_bar_002_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0013, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| GT tick-pixel 映射进入 grid/feedback prompt | PASS |  |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0032 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心来源 | PASS | 使用上一轮预测或类别轴定位；GT 仅用于记录/指标 |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\v_bar\v_bar_002\tempy\overlay_Spending in billion Mexican pesos, 2020_feedback_grid_with_grid_run1.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\v_bar\v_bar_002\tempy\amplifier_crop_Spending in billion Mexican pesos, 2020_round1_attempt0.png`

## h_bar

- 结论：PASS
- 样本：synthetic/h_bar_001，sample_id=661ec486776ed0d0
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\hBar_50\chart_config\h_bar_001.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\hBar_50\chart\h_bar_001.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\hBar_50\h_bar_001\h_bar_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：Cocoa bean yields, 1961, Vanuatu，目标总数=3

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/h_bar_001 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\hBar_50\chart_config\h_bar_001.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=3 |
| 原图 | PASS | 800x600, variance=4450.40, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\hBar_50\chart\h_bar_001.png |
| grid-with-grid 图 | PASS | 800x600, variance=4328.94, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\hBar_50\h_bar_001\h_bar_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0006, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| GT tick-pixel 映射进入 grid/feedback prompt | PASS |  |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0086 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心来源 | PASS | 使用上一轮预测或类别轴定位；GT 仅用于记录/指标 |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\h_bar\h_bar_001\tempy\overlay_Cocoa bean yields, 1961, Vanuatu_feedback_grid_with_grid_run1.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\h_bar\h_bar_001\tempy\amplifier_crop_Cocoa bean yields, 1961, Vanuatu_round1_attempt0.png`

## pie

- 结论：WARN
- 样本：synthetic/001，sample_id=f3dd047797f5d357
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Pie_50\chart_config\pie_001.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Pie_50\chart\pie_001_no_grid.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Pie_50\pie_001\pie_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：Refused  to use，目标总数=2

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/pie_001_no_grid |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Pie_50\chart_config\pie_001.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=2 |
| 原图 | PASS | 680x680, variance=593.37, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Pie_50\chart\pie_001_no_grid.png |
| grid-with-grid 图 | PASS | 680x680, variance=587.16, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Pie_50\pie_001\pie_001_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0003, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| 圆形图角度网格提示 | WARN | pie/donut 使用 15°角度参考线，不使用笛卡尔 tick-pixel 映射 |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0111 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心来源 | PASS | 使用上一轮预测或类别轴定位；GT 仅用于记录/指标 |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\pie\001\feedback_img\Refused_to_use_feedback.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\pie\001\Refused_to_use_amplifier.png`

## donut

- 结论：PASS
- 样本：synthetic/donut_000，sample_id=1bdc0d1d2be7d3e4
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Donut_50\chart_config\donut_000.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Donut_50\chart\donut_000.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Donut_50\donut_000\donut_000_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：A，目标总数=9

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/donut_000 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Donut_50\chart_config\donut_000.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=9 |
| 原图 | PASS | 500x400, variance=2053.35, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Donut_50\chart\donut_000.png |
| grid-with-grid 图 | PASS | 500x400, variance=2020.12, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Donut_50\donut_000\donut_000_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0007, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| 圆形图角度网格提示 | PASS | pie/donut 使用 15°角度参考线，不使用笛卡尔 tick-pixel 映射 |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0007 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心来源 | PASS | 使用上一轮预测或类别轴定位；GT 仅用于记录/指标 |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\donut\donut_000\feedback_img\A_feedback.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\donut\donut_000\A_amplifier.png`

## radar

- 结论：PASS
- 样本：synthetic/radar_000，sample_id=e15975b5faf4b865
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Radar_50\radar_000.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Radar_50\radar_000.png`
- grid-with-grid：`F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Sy.Dataset\radar_000\radar_000_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png`
- 测试目标：WDULR, A，目标总数=20

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/radar_000 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Radar_50\radar_000.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=20 |
| 原图 | PASS | 600x600, variance=1383.78, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Radar_50\radar_000.png |
| grid-with-grid 图 | PASS | 600x600, variance=1352.85, F:\program\test-demo\backend\evaluation_prediction\results\gt_rendered_grids\Sy.Dataset\radar_000\radar_000_gt_grid_grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1.png |
| grid 样式/来源 | PASS | fallback-rendered, #cccccc pixel ratio=0.0018, style=grid_with_grid_midpoint_encrypted_labels_cccccc_dash_2_2_w1 |
| 三阶段提示词生成 | PASS | missing=[] |
| GT 径向 tick-pixel 映射进入 prompt | PASS |  |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0081 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心来源 | PASS | 使用上一轮预测或类别轴定位；GT 仅用于记录/指标 |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\radar\radar_000\feedback_img\WDULR_A_feedback_round1.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\radar\radar_000\amplifier_img\WDULR_A_amp1.png`

## rose

- 结论：PASS
- 样本：synthetic/rose_000，sample_id=3969a7900c95cec5
- GT JSON：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Rose_50\rose_000.json`
- 原图：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Rose_50\rose_000.png`
- grid-with-grid：`F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Rose_50\rose_000.png`
- 测试目标：LSP，目标总数=9

| 检查项 | 状态 | 说明 |
| --- | --- | --- |
| 样本发现 | PASS | synthetic/rose_000 |
| GT JSON | PASS | F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Rose_50\rose_000.json |
| 模块解耦契约 | PASS | baseline,grid,feedback,amplifier |
| GT loader 和目标枚举 | PASS | targets=9 |
| 原图 | PASS | 1400x1400, variance=2593.21, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Rose_50\rose_000.png |
| grid-with-grid 图 | PASS | 1400x1400, variance=2593.21, F:\program\test-demo\backend\datasets\VisHintPrompt_datasets\Sy.Dataset\Rose_50\rose_000.png |
| grid 样式/来源 | PASS | existing-grid-with-grid, #cccccc pixel ratio=0.0016 |
| 三阶段提示词生成 | PASS | missing=[] |
| GT 径向 tick-pixel 映射进入 prompt | PASS |  |
| feedback 提示引用上一轮预测 | PASS |  |
| amplifier 提示说明局部放大 | PASS |  |
| feedback/amplifier 过程图生成 | PASS | artifacts=2 |
| feedback/amplifier 红色 guide 样式 | PASS | red pixel ratio=0.0045 |
| GT 模式轮次上限 | PASS | baseline/grid=1, feedback=2, amplifier=3 |
| 生成端不使用 GT/RNE 作为提前停止 | PASS |  |
| amplifier 裁剪中心来源 | PASS | 使用上一轮预测或类别轴定位；GT 仅用于记录/指标 |
| 实验入口排除 stacked bar | PASS | supported=['bubble', 'donut', 'h_bar', 'line', 'pie', 'radar', 'rose', 'scatter', 'v_bar'], normalized_stacked={'v_stacked_bar': 'v_stacked_bar', 'h_stacked_bar': 'h_stacked_bar'} |
| modal call 与结构化预测 call_id 关联 | PASS | modal logs, runner records, gt_metric_records.csv, enriched logs |

过程文件：
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\rose\rose_000\feedback_img\LSP_feedback_round1.png`
- `F:\program\test-demo\backend\evaluation_prediction\results\flow_integrity_audit\rose\rose_000\amplifier_img\LSP_amp1.png`

## 复跑命令

```powershell
python backend/evaluation_prediction/gt_experiment_flow_integrity_check.py --source synthetic
```
