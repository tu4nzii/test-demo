# Prediction Core 归档说明

`reference/prediction_core` 是归档参考代码，不属于当前运行时链路。后端和前端运行时代码不应 import `reference/` 下的模块。

当前运行时评估预测包位于：

```text
backend/evaluation_prediction/
```

## 当前项目边界

- `main` 是直角系网格还原和加密的唯一实现基准。
- 数据集 GT 不允许参与生成运行时网格、tick、label、颜色或预测结果。
- 当前运行时注册表不暴露 `v_stacked_bar` 或 `h_stacked_bar`。
- 活跃模型配置以 `backend/evaluation_prediction/common/model_config.py` 为准。

## 历史参考价值

该目录可用于参考旧版 prompt、parser、geometry、runner 和 evaluation 组织方式，但不能直接接入当前系统。

历史代码覆盖过：

```text
cartesian: v_bar, h_bar, line, scatter, bubble
polar:     pie, donut, rose, radar
```

## 模型说明

当前项目 Gemini 默认 profile 使用 `gemini-2.5-flash-lite`。归档脚本中的旧模型名不代表当前系统配置。
