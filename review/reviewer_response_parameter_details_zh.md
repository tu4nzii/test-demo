# 审稿回复草稿：参数、可复现性与验证细节

## Comment 1.7

**审稿意见：** Gaussian filter 和 Canny edge detection 的参数是如何选择的？这些参数是否对所有图像固定？性能对这些参数是否敏感？

**回复：** 各类图表内部使用固定参数，不针对单张图像调参。Gaussian filter 和 Canny edge detection 在不同图表家族中的作用不同。

对于直角坐标系图表，当前主流程不使用全图 Gaussian smoothing，也不使用 Canny edge detection。网格线来自低饱和度灰色像素、局部对比度和形态学横/竖线过滤，再结合 OCR/MLLM 轴证据在三套候选网格之间进行选择。该分支中唯一的 Gaussian 操作是 OCR 文本裁剪区域的局部 `(3,3), sigma=0` 平滑。

基于最新代码的验证已经在当前生成 artifacts 上执行。直角系审计共发现 `650` 条仲裁记录，且全部 `650` 条都包含当前流程使用的三套候选来源；同时发现 `1176` 个 final-binding 文件、`650` 个 grid status report 和 `14` 个 failure/exit report。在当前直角系评估集合上，共审计 `325` 个样本，其中 `317` 个成功处理；生成的 final bindings 达到 Tick MAE `0.691 px`、Tick Acc@2px `96.37%`、tick-position MAE `0.849 px`、标签准确率 `96.13%`。

对于 pie/donut，绘图区检测优先使用颜色 mask：HSV 饱和度 `>12`、亮度 `>40`，`(5,5)` 椭圆核形态学处理、连通域过滤和 `minEnclosingCircle`。Hough 圆检测只作为 fallback，参数为 Gaussian `(9,9), sigma=2` 和 `HoughCircles(dp=1.2, param1=50, param2=30)`。

对于 radar/rose，径向圆环检测使用固定 Gaussian `(9,9), sigma=2`，之后进行 Hough 圆检测。第一圈使用 `dp=1.2`、`minDist=100`、`param1=20`、`param2=30`；第二圈使用 `param2=50`。Radar 和 rose 根据图像高度使用不同的固定半径范围。

参数敏感性通过两组边界明确的诊断实验说明。直角系诊断只隔离旧版 Canny/Hough 候选生成器；它不是当前直角系主流程，因此只作为低层候选数量敏感性检查。

| 参数设置 | 平均 Hough 线段候选数 | 相对基线变化 |
| --- | ---: | ---: |
| 无 blur, Canny 20/80, Hough 15 | 181.00 | -0.04% |
| **无 blur, Canny 30/100, Hough 15** | **181.08** | **0.00%** |
| 无 blur, Canny 50/150, Hough 15 | 181.59 | +0.28% |
| 无 blur, Canny 70/210, Hough 15 | 178.87 | -1.22% |
| 3x3 blur, Canny 30/100, Hough 15 | 189.35 | +4.56% |
| 5x5 blur, Canny 30/100, Hough 15 | 188.07 | +3.86% |
| 无 blur, Canny 30/100, Hough 10 | 203.64 | +12.45% |
| 无 blur, Canny 30/100, Hough 20 | 159.53 | -11.90% |

该诊断表明，中等范围内的 Canny 阈值对候选数量影响较小，而 Hough threshold 对候选数量影响更明显。

对于 radar/rose，在 99 张 polar 样本上扫描 Hough `param2`。GT 只用于离线评分。

| HoughCircles `param2` | 样本数 | 圆检测返回率 | 中位圆心误差 | 中位最佳半径误差 | 平均候选圆数量 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 99 | 100.00% | 2.236 px | 1.000 px | 50.83 |
| **30** | **99** | **100.00%** | **2.236 px** | **1.000 px** | **26.87** |
| 40 | 99 | 97.98% | 2.236 px | 2.000 px | 5.60 |
| 50 | 99 | 76.77% | 2.236 px | 2.000 px | 2.21 |

固定使用的 `param2=30` 与 `param2=20` 保持相同的圆检测返回率，同时减少候选数量。完整评估中，直角坐标系图表的 Tick MAE 为 `0.691 px`，Tick Acc@2px 为 `96.37%`，标签准确率为 `96.13%`，图表类型分类准确率为 `100.00%`；极坐标图表类型分类准确率为 `96.64%`。该分类指标中，`bubble` 与 `scatter` 按点图族互认。

当前代码在运行参数 `param2=30` 下的 polar 诊断结果为：`99/99` 个 polar 样本均返回圆检测结果，中位圆心误差 `2.236 px`，中位第一候选半径误差 `2.000 px`，中位最佳半径误差 `1.000 px`，平均候选圆数量 `26.87`。这些诊断中的 GT 只用于离线度量；生成端只使用图像派生的网格、OCR/MLLM 证据和系统生成的几何信息。

## Comment 2.6[2]

**审稿意见：** 部分技术细节仍然描述不足。影响可复现性的关键实现细节没有充分说明，包括图表类型分类的鲁棒性、轴线/tick 提取参数设置，以及 zoom-in verification 机制的具体实现。

**回复：** 实现包含三个关键部分：图表类型分类、网格/轴线/tick 提取，以及数值预测阶段的 zoom-in verification。

**图表类型分类。** 图表类型分类由 MLLM JSON 提取器完成，调用温度为 `temperature=0`。输出类型必须属于注册类型：`rose`、`radar`、`v_bar`、`h_bar`、`line`、`scatter`、`bubble`、`donut` 或 `pie`。如果类型缺失或不受支持，系统显式报错，不回退到默认类别。图表分类准确率中，`bubble` 与 `scatter` 按点图族互认。最新全量评估中的图表类型分类准确率为：整体 `98.95%`，直角坐标系 `100.00%`，极坐标系 `96.64%`。

**网格/轴线/tick 提取。** 直角坐标系图表使用固定 mask 和形态学参数：饱和度 `<=70`，灰度范围 `[95,255]`，局部对比度 `>=7`，`min_line_frac=0.055`，`gap_frac=0.006`，`max_thickness_frac=0.008`，`min_grid_span_frac=0.18`，`min_grid_lines=2`，`cluster_tolerance=3 px`，`grid_thickness=1 px`。OCR 过滤使用 `ocr_min_score=0.45`、检测阈值 `0.35`、box 阈值 `0.60`、unclip ratio `1.15` 和 side limit `960`。

Pie/donut 使用 15 度角度网格，圆形绘图区优先由颜色 mask 检测，必要时使用 Hough fallback。Radar/rose 使用固定 Gaussian-Hough 径向圆环检测，`tick_density=2`，径向 tick 数值由 MLLM 从图像中读取。

当前代码证据文件也确认了直角系的选择机制：每条被审计的仲裁记录都包含三套候选来源，最终评估读取的是系统生成的 `final_bindings`，而不是参考标注。参考 JSON 只在离线评估脚本中用于计算误差，不进入生成端流程。

**Zoom-in verification。** Zoom-in verification 属于数值预测阶段，不属于网格重建阶段。柱状图、散点图和气泡图会围绕预测目标裁剪，并验证目标是否可见；不可见时重试。折线图围绕目标 x 类别和当前 y 估计裁剪，并记录 contains-target 诊断。Radar/rose 同时使用生成后的网格图和原图，提示词包含检测到的 `r_ticks`、`theta_ticks`、`theta_angles` 和颜色提示；目标级预测不完整时使用 whole-chart prompt。Pie/donut 通过三轮裁剪放大细化扇区估计：`pad=15°、grid=5°、zoom=2.0`，`pad=9°、grid=3°、zoom=2.0`，`pad=6°、grid=2°、zoom=3.0`，并使用 LLM contains-sector 检查扇区顺序和可见性。
