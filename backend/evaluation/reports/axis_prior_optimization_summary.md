# 轴先验优化结果

本轮优化只改动 CV 轴/刻度先验，不改变下游 MLLM 数据读取逻辑。

## 改动

- `function_calling/axis/infer_axes.py`
  - 由“最靠下横线 + 最靠左竖线”改为候选线打分。
  - 打分因素包括线长、是否与另一方向候选形成绘图区框线、是否靠近左侧坐标框、是否像图例/页脚等底部装饰线。
  - 保持原有函数接口：`infer_axes_from_lines(lines, image_size, image_gray, angle_tolerance=...)`。

- `function_calling/ticks/detect_ticks.py`
  - 刻度扫描由单侧扫描改为轴线两侧扫描。
  - 相邻像素候选会先压缩成一个刻度线段，避免产生大量重复候选。
  - 灰度阈值从 200 放宽到 230，并允许最多 3 像素前导空隙，以覆盖浅灰网格线和轻微断开的刻度线。

## 指标变化

评估文件：`F:\program\test-demo\backend\evaluation\results\axis_prior_eval_after_optimization.json`

相对于优化前 `backend/evaluation/results/axis_prior_eval_results.json`：

- 轴线返回率：保持 93.57%。
- 几何轴线通过率：18.60% -> 19.30%。
- X 轴中位误差：10 px -> 1 px。
- Y 轴中位误差：保持 29 px。
- X 刻度平均召回：63.19% -> 77.04%。
- Y 刻度平均召回：55.51% -> 55.23%，基本持平。

分类别的主要收益：

- line 的 X 刻度召回：69.60% -> 99.13%。
- v_bar 的 X 刻度召回：85.43% -> 98.12%。
- h_bar 的 Y 刻度召回：88.86% -> 87.19%，轻微下降，主要来自双侧扫描后过滤器对少量水平柱图候选的重排。

## 验证

- `python -m py_compile backend\Grid_generation\function_calling\axis\infer_axes.py backend\Grid_generation\function_calling\ticks\detect_ticks.py`
- 对 `line_149.png`、`scatter_000.png`、`v_bar_063.png`、`h_bar_000.png` 做了轴线抽查，均返回合理的可视坐标框线。
- 端到端 `process_chart` 在进入外部 LLM/API 阶段后因代理连接和 Windows GBK 控制台编码失败中断；中断前 CV 轴线与刻度检测已成功执行。
