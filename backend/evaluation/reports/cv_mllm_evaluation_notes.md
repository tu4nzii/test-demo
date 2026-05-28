# CV + MLLM 评估与缓存

## 缓存机制

主流程现在默认启用 MLLM 缓存：

- 刻度值/轴类型缓存：`backend/data/llm_cache/tick_labels`
- 图例颜色缓存：`backend/data/llm_cache/colors`

缓存 key 使用图像绝对路径、文件大小、mtime 和任务版本号生成。命中缓存时不再调用 MLLM；缓存缺失时才调用 MLLM 并保存结果。

## 联合评估脚本

新增脚本：

```powershell
python backend\evaluation\scripts\evaluate_cv_mllm_ticks.py --types line scatter bubble v_bar h_bar --limit 20
```

常用参数：

- `--limit N`：每类最多评估 N 张；`--limit 0` 表示全部。
- `--cache-only`：只读缓存，不在缓存缺失时调用 API。
- `--output PATH`：指定输出 JSON。

输出指标包括：

- `x_pixel_recall` / `y_pixel_recall`：CV 刻度像素位置召回。
- `x_value_recall` / `y_value_recall`：MLLM 刻度值序列召回。
- `x_pair_recall` / `y_pair_recall`：像素位置和刻度值同时匹配的联合召回。
- `cache_hits` / `cache_created` / `cache_misses_no_api`：缓存使用情况。

## 验证

已运行：

```powershell
python -m py_compile backend\Grid_generation\grid_generation.py backend\Grid_generation\function_calling\label\extract_tick_labels_with_llm.py backend\Grid_generation\function_calling\color\extract_chart_colors.py backend\evaluation\scripts\evaluate_cv_mllm_ticks.py
python backend\evaluation\scripts\evaluate_cv_mllm_ticks.py --types line scatter --limit 2 --cache-only --output backend\evaluation\results\cv_mllm_tick_eval_smoke.json

## 缓存机制

`extract_tick_labels_with_llm` 使用 `tick-mllm-v3` 缓存。缓存 key 绑定：

- `dataset_id`
- 图像内容 SHA256
- X/Y 轴提示词与 system prompt 的签名
- MLLM model 与 temperature
- 缓存 schema version

因此，重复评估时只有在数据图像和提示词配置未变化时才会复用之前的 MLLM 原始输出；提示词、模型、temperature、dataset id 或图像内容变化都会自动产生新的 cache miss。评估时建议先用 `--cache-only` 查看缺口，再去掉 `--cache-only` 补齐缺失缓存。
```

第二条命令是缓存只读 smoke test。由于当前测试样本尚无 MLLM 缓存，value/pair 指标为 0，缓存缺失被记录到 `cache_misses_no_api`。
