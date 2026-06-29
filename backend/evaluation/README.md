# Evaluation Assets

本目录保存离线评估脚本、报告和运行时评估辅助函数。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `scripts/` | 独立评估和批处理脚本 |
| `results/` | 生成的 JSON/CSV/Markdown 指标结果 |
| `reports/` | 人工可读报告、审稿回复、阶段总结 |
| `archive/` | 历史或已替代报告 |
| `metrics.py` | 运行时和离线共用指标函数 |
| `normalizer.py` | GT/预测数据归一化 |
| `service.py` | API 使用的评估服务 |

## 数据边界

- 生成端不使用 GT。
- 离线评估端可以读取 GT 计算指标。
- 自定义上传通常没有 GT，前端展示的是预测结果和处理质量。
- 数据集预览缓存必须来自当前系统链路，而不是 reference 或 GT 直接生成。

## 常用指标

- 数值轴 tick-value MAE。
- 数值轴 tick-value Acc@2px。
- 图例颜色准确率。
- 标签名准确率。
- 图表分类准确率。

tick MAE 和 tick Acc 只计算数值轴，分类轴不纳入。

## 缓存原则

模型缓存键应覆盖：

- 数据集或样本 ID。
- 图片内容 hash。
- prompt 签名。
- 模型名。
- temperature。
- schema/cache 版本。

输入不变时尽量使用缓存，避免重复调用模型。

## 运行提示

具体脚本会随实验阶段变化。运行前先查看脚本参数，并确认是否需要：

- `--cache-only`：只使用已有模型缓存。
- 输出目录是否会覆盖最新结果。
- 是否读取 GT，仅用于指标计算。
