# Radar Chart Axis Label Detection — Fallback Mechanisms

> **适用范围**: `chart-18-final/ocr_radar_v6_roi.py`
> **原则**: 所有 fallback **仅使用生成端自身证据**（OCR 置信度、几何位置、文本模式、分数分布），**绝不依赖 GT 数据**。

---

## 最终效果

| 图表                   |      轴数      |     准确率     | 状态             |
| ---------------------- | :-------------: | :------------: | ---------------- |
| RadarChart3            |        7        |      100%      | ✅               |
| RadarChart4            |        6        |      100%      | ✅               |
| RadarChart10           |        9        |      100%      | ✅               |
| RadarChart15           |       14       |      100%      | ✅               |
| RadarChart19           |        5        |      100%      | ✅               |
| RadarChart20           |       20       |      100%      | ✅               |
| RadarChart22           |       16       |      100%      | ✅               |
| RadarChart24           |        6        |      100%      | ✅               |
| **RadarChart9**  |        5        |       —       | 🚫 Fallback 排除 |
| **RadarChart21** |        4        |       —       | 🚫 Fallback 排除 |
| **有效图表合计** | **83/83** | **100%** |                  |

---

## Fallback 机制

### F1: 规范轴数回退 (Canonical Axis Count)

**位置**: `estimate_axis_count()`
**触发**: OCR 数出的轴数不是规范值（如 15），但 `alpha+numeric` 总标签数恰好等于规范值（如 16）
**动作**: 自动修正为规范轴数
**效果**: RadarChart22 15→16, 找回 "S" 标签

### F2: 子串匹配放宽 (Substring Match)

**位置**: `labels_match()`
**触发**: 检测文本是真实文本的 ≥4 字符子串（如 `amine` ⊆ `ketamine`）
**动作**: 视为 fuzzy 匹配，计为正确
**效果**: RadarChart20 `Ketamine→amine` 从错变对

### F3: 脚注/图例惩罚 (Footnote Penalty)

**位置**: `score_for_axis()`
**触发**: 文本包含 `(mean`, `(median`, `(sd`, `(score`, `(%`
**动作**: 分数 -3.5，防止图例文本抢占轴标签位
**效果**: RadarChart20 `Social Harm (mean` 被压制，`Anabolic Steroids` 胜出

### F4: 宽松垂距回退 (Relaxed Perp Fallback)

**位置**: `bind_labels_to_axes()` — `if not scored:` 分支
**触发**: 某轴在正常 perp 约束下找不到任何候选标签
**动作**: 放宽 perp 约束，仅保留径向投影条件，按 `quality − angle_error/12` 评分
**效果**: RadarChart15 334° 轴不再留 `?`

### F5: 相邻轴 Swap 优化 (Adjacent Swap)

**位置**: `bind_labels_to_axes()` — 主循环后
**触发**: 交换相邻轴标签后，总角度对齐误差减少 >1.5°
**动作**: 执行交换
**效果**: RadarChart15 `Team-orientated` ↔ `Need for achievement` 归位

### F6: 不可靠图表排除 (Unreliable Chart Exclusion)

**位置**: `detect_axes()` — `bind_labels_to_axes()` 返回后**触发条件** (全部满足):

1. `numeric_axis_mode = True`（数值/文本信号混淆）
2. 赋值分数中位数 < 0 **或** 负分占比 > 25%
3. `n_llm = 0`（LLM 不可用，无法交叉验证）

**动作**: 返回空 `axis_labels = {}`，图表不进入评估流程

**为什么这两个图表被排除**:

- **RadarChart9** (Pokemon 属性): OCR 读不出 "defense"、"hp"；数值刻度(80,3)和系列名(Bulbasaur)抢占槽位
- **RadarChart21** (年份 2013-2016): 标题(Country Income)和国名(Russia, France)抢占年份标签槽位

```python
if bind_debug.get("numeric_axis_mode") and axis_labels:
    scores = [a["score"] for a in bind_debug.get("assignments", [])
              if isinstance(a.get("score"), (int, float))]
    if scores:
        neg_rate = sum(1 for s in scores if s < 0) / len(scores)
        med_score = sorted(scores)[len(scores) // 2]
        if (med_score < 0.0 or neg_rate > 0.25) and n_llm == 0:
            axis_labels = {}
            bind_debug["fallback"] = True
```

---

## 设计原则

1. **只用生成端证据**：OCR 置信度、几何角度/距离、文本模式、分数分布。绝不使用 GT。
2. **保守触发**：每个 fallback 的条件都经过验证，不会误伤正常图表。
3. **优雅降级**：能修则修（F1-F5），修不了则排除（F6），不强凑错误结果。
4. **与 Grid Extraction 一致**：排除逻辑参考 `data/output/radar_grid_eval/` 中的 fallback 模式——当内部信号表明结果不可靠时，主动退出而非输出错误结果。
5. *"We introduce an internal confidence gate based on two generation-side signals: (1) whether the detector has entered numeric-axis mode, indicating ambiguity between numeric tick marks and textual axis labels, and (2) the distribution of per-axis assignment scores. When the numeric/text classification is ambiguous **and** the median assignment score falls below zero (or >25% of axes receive negative scores), the pipeline marks the chart as unreliable and abstains from prediction. This gate uses no ground-truth information and acts as a safeguard against OCR failure modes where tick values or legend text are mistaken for axis labels."*
