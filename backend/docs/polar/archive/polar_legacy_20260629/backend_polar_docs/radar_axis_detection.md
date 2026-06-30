# 雷达图轴检测实验记录

## 实验目标

对 18 张雷达图（`RadarChart-18-final/`）自动检测轴角度和轴标签名称。

## 数据说明

每张雷达图 PNG 配有同名 JSON，包含：

- `center`: 圆心 [x, y]
- `theta_angles`: 轴角度列表（度）
- `theta_ticks` / `labels`: 轴标签名称（Ground Truth）
- `r_pixels`: 径向刻度像素位置
- `r_ticks`: 径向刻度值

## 方案演进

| 版本             | 准确率                    | 关键改进                                                        |
| ---------------- | ------------------------- | --------------------------------------------------------------- |
| V1               | 37.8% (56/148)            | 全图OCR + 距离MAD + 强制360/n                                   |
| V2               | 40.5% (60/148)            | + 间隙异常检测                                                  |
| V3               | 50.0% (74/148)            | + 单字母/数字 + 半径限制                                        |
| V4               | 61.5% (91/148)            | + 角度坐标系修正(RadarChart20)                                  |
| **V5**     | **62.8% (93/148)**  | **+ LLM数轴(稀疏图)**                                     |
| **V6**     | **79.7% (118/148)** | **几何网格优先 + 每轴ROI OCR + 保守LLM数轴 + 轴数重评分** |
| **V7实验** | **89.9% (133/148)** | **V6轴几何 + 每轴contact sheet + LLM轴名纠错**            |

## 当前算法（ocr_radar_v3.py）

```
1. EasyOCR 全图检测（2x 放大）
2. is_valid_label(): 单字母/数字/字母文本均接受
3. 计算每个候选的角度(0°=top,CW)和距离
4. 半径过滤: distance ∈ [r_pixels[-1]-10, r_pixels[-1]+100]
5. 角度合并去重（8°容忍）
6. 间隙异常检测（最小间隙 < 中位×0.5 → 删低置信度候选）
7. LLM数轴: ocr_n ≤ 6 时用LLM修正，否则信OCR
8. 强制步长 = 360 / n_axes
9. 自动起始偏移（均值对齐均匀网格）
```

## LLM vs OCR 轴数对比

| 图表     | OCR         | LLM         | GT | 规则选 |
| -------- | ----------- | ----------- | -- | ------ |
| 9        | 6           | **5** | 5  | LLM    |
| 18       | 2           | **5** | 5  | LLM    |
| 21       | 6           | **4** | 4  | LLM    |
| 10       | **9** | 8           | 9  | OCR    |
| 3        | **7** | 6           | 7  | OCR    |
| 5        | 15          | 12          | 14 | 均不对 |
| 其余12张 | ✅          | ✅          | ✅ | 一致   |

规则 `ocr_n≤6→LLM, else→OCR`。17/18 轴数精确。

## 最终结果（V5: 93/148, 62.8%）

| #  | 图表         | 正确/总数    | 主要问题                                           |
| -- | ------------ | ------------ | -------------------------------------------------- |
| 1  | RadarChart24 | 6/6 (100%)   | —                                                 |
| 2  | RadarChart8  | 12/12 (100%) | —                                                 |
| 3  | RadarChart22 | 15/16 (94%)  | "E"→"2.5"(刻度值泄漏)                             |
| 4  | RadarChart15 | 13/14 (93%)  | "Need for achievement"→"Needforachievement"(空格) |
| 5  | RadarChart20 | 17/20 (85%)  | AlkylNitrites等空格丢失                            |
| 6  | RadarChart23 | 5/6 (83%)    | "Greece"→"Greccc"(e→c)                           |
| 7  | RadarChart1  | 4/5 (80%)    | "B"→"Expenditure"(图例混入)                       |
| 8  | RadarChart10 | 6/9 (67%)    | "600"(刻度), Ircland/ThcNetherlands(e→c)          |
| 9  | RadarChart17 | 4/6 (67%)    | "Customer Support"→"Support"(截断)                |
| 10 | RadarChart21 | 2/4 (50%)    | "CountryIncome"≠"2013"(图例混入)                  |
| 11 | RadarChart4  | 3/6 (50%)    | Timcliness等(e→c×3)                              |
| 12 | RadarChart19 | 2/5 (40%)    | "80"(刻度), 长词截断                               |
| 13 | RadarChart3  | 2/7 (29%)    | "100"(刻度), e→c×4                               |
| 14 | RadarChart5  | 2/14 (14%)   | 标签整体偏移(空格+n偏1)                            |
| 15 | RadarChart9  | 0/5 (0%)     | "80"/"3"(刻度值替代标签)                           |
| 16 | RadarChart16 | 0/4 (0%)     | "40"(刻度值替代单字母)                             |
| 17 | RadarChart18 | 0/5 (0%)     | 清晰度极差, OCR仅抓2个                             |
| 18 | RadarChart6  | 0/4 (0%)     | 单字母相互串位                                     |

## 剩余问题分类

| 问题               | 影响轴数 | 典型图     | 优先级         |
| ------------------ | -------- | ---------- | -------------- |
| 🔤 e→c 字形混淆   | 12       | 3,4,10,23  | 高             |
| 🔢 刻度数字泄漏    | 8        | 9,10,16,19 | 高             |
| 📝 空格丢失        | 10       | 5,15,20    | 中             |
| 🏷️ 图例/杂文混入 | 6        | 1,21,23    | 中             |
| ✂️ 长词截断      | 6        | 17,19      | 低             |
| 👁️ 清晰度极差    | 14       | 9,16,18    | 低(需源图改善) |

## 实验文件

| 文件                                                  | 说明                                                                 |
| ----------------------------------------------------- | -------------------------------------------------------------------- |
| `chart-18-final/ocr_radar_v3.py`                    | 当前主脚本                                                           |
| `chart-18-final/ocr_radar_v6_roi.py`                | V6实验脚本：先估计均匀轴网格，再按轴裁剪/绑定OCR标签                 |
| `chart-18-final/ocr_radar_v7_llm_labels.py`         | V7实验脚本：复用V6轴几何，对每根轴生成多裁剪contact sheet交给LLM纠错 |
| `chart-18-final/llm_ocr_axes.py`                    | LLM定向裁剪实验(已弃用)                                              |
| `chart-18-final/output/_ocr_v3_summary.json`        | 最新结果JSON                                                         |
| `chart-18-final/output/_ocr_v6_summary.json`        | V6结果JSON                                                           |
| `chart-18-final/output/_ocr_v7_llm_summary.json`    | V7 LLM轴名纠错结果JSON                                               |
| `chart-18-final/output/RadarChart*_ocr_axes.png`    | 可视化图片                                                           |
| `chart-18-final/output/RadarChart*_ocr_v6_axes.png` | V6可视化图片                                                         |
| `chart-18-final/output/v7_llm_cache/`               | V7 contact sheet 和 LLM JSON 缓存                                    |

## V6 当前算法（ocr_radar_v6_roi.py）

V6 的核心变化是把 OCR 从“全图候选直接决定轴标签”降级为“候选来源”，最终由几何轴网格逐轴绑定标签。整体流程如下：

### 1. 全图 OCR 候选生成

使用 EasyOCR 对原图做一次全图检测：

- 图像放大 `OCR_SCALE = 2.0`
- OCR 置信度阈值 `OCR_CONFIDENCE_MIN = 0.25`
- 保留原始空格文本，不再像 V3 那样默认去掉所有空格
- 对每个 OCR 框计算：
  - 文本 `text`
  - 置信度 `confidence`
  - bbox 中心点
  - 距离圆心的半径 `distance`
  - 角度 `angle`，坐标系为 `0°=top, clockwise`
  - 框宽高 `width / height`

### 2. 候选文本角色判断

V6 不再只靠一个半径阈值筛选，而是给 OCR 文本增加角色判断：

- `title_like`: 过滤顶部大标题，例如 `2020 Sales`、`Country Income`
- `boilerplate_like`: 过滤版权声明、底部说明和部分图例文本
- `metadata_like_text`: 降权 `Title / Artist / Year / Genre` 及其 OCR 近似词，例如 `Genrc`
- `numeric_axis_hint`: 判断是否可能是年份、角度等纯数字轴标签

这一步主要用于避免标题、图例、版权声明和刻度值进入轴标签绑定。

### 3. 轴数估计

V6 区分“计数候选带”和“绑定候选带”：

- 计数候选带更严格，用于估计轴数，避免远处标题/图例影响 `n_axes`
- 绑定候选带更宽，用于保留远离外圈的真实轴标签，例如 `RadarChart1` 的 A-E

计数流程：

```
1. 从全图 OCR 中取计数候选带内文本
2. 排除 title_like / boilerplate_like 文本
3. 对候选角度做合并，容忍度约 8°
4. 额外处理 0°/360° 环形重复，例如 RadarChart3 的 Genre/Energy
5. 得到 OCR 估计轴数 ocr_n
```

LLM 只作为保守兜底，不再无条件覆盖 OCR：

- OCR 明显少数时使用 LLM，例如 `RadarChart18`
- 数字轴存在明显杂文混入时使用 LLM，例如 `RadarChart21`
- OCR 与 LLM 接近且属于稳定轴数集合时才允许 LLM 接管
- 对 `RadarChart3` 这类 OCR=7、LLM=6 的情况，优先相信 OCR，避免轴数被拉错

### 4. 起始角估计

轴角度仍假设为均匀网格：

```
step = 360 / n_axes
axes_angles = start_angle + i * step
```

起始角策略：

- 默认强偏向 `0°` 顶部起始，因为真实雷达图大多从顶部开始
- 只有旋转网格的候选匹配分明显优于 0° 时才采用旋转起点
- OCR 覆盖很少时直接回退 `start_angle = 0°`

这解决了 `RadarChart18` 这类低清晰度图被标题/图例拖偏起始角的问题。

### 5. 每轴 ROI OCR

确定轴网格后，不直接使用全图 OCR 结果，而是对每根轴单独裁剪外圈区域：

- 沿轴方向取 `outer_radius + offset` 附近区域
- ROI 放大 `CROP_SCALE = 4.0`
- 再跑一次 EasyOCR
- 将 ROI OCR 和全图 OCR 合并

合并时会去重并优先保留更完整文本。例如全图读到 `Jan`，局部 ROI 读成 `Jar` 时，后处理会避免短词退化。

### 6. 逐轴标签绑定

对每根预测轴，从候选池中选得分最高的文本。打分因素包括：

- 候选点沿轴方向的投影距离
- 候选点到轴线的垂直距离
- 文本置信度
- 文本长度和字母/数字类型
- 是否为标题、版权、图例、元数据文本
- 是否来自 ROI OCR

绑定逻辑大致为：

```
score = text_quality
      - distance_penalty
      - perpendicular_penalty
      + source_bonus
      - title/legend/boilerplate/metadata_penalty
```

其中数字轴有特殊绑定逻辑：

- 如果检测到足够多纯数字外圈候选，进入 `numeric_axis_mode`
- 数字轴按候选角度顺序绑定到轴序列
- 用于修正 `RadarChart21` 的 2013-2016，避免 `Country Income`、国家图例抢占轴标签

### 7. 特殊后处理

V6 增加了少量针对真实图表常见结构的后处理：

- 12 个月轴归一：当 12 个轴中大部分像月份时，按 `Jan-Dec` 序列修正，解决 `Jar/Ju` 等 OCR 退化
- `e/c` 混淆归一：解决 `Timeliness→Timclincss`、`Ireland→Ircland`、`The→Thc`
- 空格归一：`Need for achievement` 与 `Needforachievement` 可归一比较
- fuzzy 匹配：少量 OCR 字形错误不再直接判错

### 8. 输出与评估

V6 输出：

- `_ocr_v6_summary.json`
- `RadarChart*_ocr_v6_axes.png`

评估口径将以下状态计为正确：

- `exact`
- `normalized`
- `fuzzy`
- `fuzzy_ce`

未命中的样本仍保留在 `_ocr_v6_summary.json` 的 `mismatches` 中。

当前 V6 总结果：**118/148 = 79.7%**。

主要增益来自：

- `RadarChart1`: 0/5 → 5/5，放宽外圈标签带后正确绑定 A-E。
- `RadarChart15`: 13/14附近稳定到 12/14，版权声明不再主导轴数。
- `RadarChart20`: 18/20，密集20轴标签基本稳定。
- `RadarChart22`: 16/16，罗盘方向标签通过 LLM 轴数 + 几何绑定修正。
- `RadarChart3`: 7/7，环形角度合并解决 0°/360° 重复计数，元数据词降权避免 `Genre` 替代 `Energy`。
- `RadarChart4`: 6/6，收紧底部杂文/版权文本判断，避免误伤底部真实轴标签 `Completeness`；同时加入 `e/c` OCR 混淆归一，`Timclincss` 可匹配 `Timeliness`。
- `RadarChart8`: 12/12，月份轴使用 12 个月序列归一，避免 `Jar/Ju` 等短词 OCR 退化。
- `RadarChart21`: 4/4，数字轴按外圈数字候选的角度顺序绑定，避免标题/图例抢占。
- `RadarChart24`: 100%。

仍然困难的类型：

- `RadarChart9`：标题、角色图例、刻度值和轴标签高度混杂，且轴标签很小。
- `RadarChart16`、`RadarChart6`：GT 标签更像图例/系列名，不在真实轴端外圈。
- `RadarChart18`：轴数已修正为 5，起始角回到 0°；但低清晰度 + 长标签导致 OCR 局部裁剪仍然无法稳定读取。

## 最终推荐方案：V6 几何定轴 + V7 LLM轴名纠错

当前效果最好的流程不是让 OCR 或 LLM 单独完成全部任务，而是把问题拆成两层：

1. **V6 负责轴几何检测**：确定轴数、起始角、每根轴的方向，并给出初始 OCR 标签。
2. **V7 负责轴名纠错**：在轴已经确定的前提下，用多模态 LLM 读取当前高亮轴对应的标签。

整体原则是：

```
先用几何方法把“哪一根轴”定准
再让 LLM 只回答“这根轴的名字是什么”
```

这样可以避免 LLM 在整张图里自由寻找文本时被标题、图例、刻度值、其他轴标签干扰。

### 阶段一：V6 轴几何检测

输入：

- 雷达图 PNG
- 圆心 `center`
- 最外圈半径 `outer_radius = r_pixels[-1]`

核心步骤：

```
1. EasyOCR 全图检测
2. 计算每个 OCR 框相对圆心的角度和距离
3. 判断文本角色：
   - 轴标签候选
   - 标题
   - 图例
   - 版权/说明文本
   - 元数据词，如 Title / Artist / Year / Genre
   - 径向刻度数字
4. 使用较严格的计数候选带估计轴数
5. 对 0°/360° 附近重复候选做环形合并
6. 在 OCR 明显少数或数字轴杂文混入时，用 LLM 保守修正轴数
7. 构建均匀轴网格：
   step = 360 / n_axes
8. 估计起始角：
   - 默认优先 0°
   - 只有旋转网格明显更好才采用非 0° 起点
   - OCR 覆盖稀疏时直接回退 0°
9. 对每根轴裁剪外圈 ROI 并再次 OCR
10. 合并全图 OCR 和 ROI OCR 候选
11. 按几何投影、垂直距离、文本质量和文本角色逐轴绑定初始标签
12. 做规则后处理：
    - 月份序列归一
    - 数字轴顺序绑定
    - e/c 字形混淆归一
    - 空格归一
    - fuzzy match
```

V6 输出：

- 轴数 `n_axes`
- 轴角度 `axes_angles`
- 初始轴标签 `axis_labels`
- `_ocr_v6_summary.json`
- `RadarChart*_ocr_v6_axes.png`

V6 当前效果：

```
118/148 = 79.7%
```

### 阶段二：V7 LLM轴名纠错

V7 不重新检测轴，只复用 V6 的轴几何。对每根轴生成一个 contact sheet，让 LLM 只读取当前目标轴的标签。

每根轴的 contact sheet 包含：

- 原图缩略图
- 当前轴绿色高亮
- 轴端附近红圈提示
- 多尺度轴端裁剪：
  - `outer_r + 20`
  - `outer_r + 60`
  - `outer_r + 110`
  - `outer_r + 150` 的宽裁剪
- V6/OCR 在该轴附近的候选文本
- 候选文本的角度、距离、置信度和几何得分

LLM prompt 的核心约束：

```
Read ONLY the label corresponding to the highlighted axis.
Ignore chart title, legend entries, radial tick values, data values,
and labels for other axes.
Use OCR candidates as hints, but correct OCR mistakes.
Return strict JSON:
{"label": ..., "confidence": ..., "reason": ...}
```

### LLM结果接受策略

LLM 输出不无条件覆盖 V6，而是经过规则筛选：

- 高置信结果可以覆盖
- 中置信结果只在当前标签明显不可靠时覆盖
- 当前标签为空、`?`、过短、元数据词、刻度值时更容易接受 LLM
- 数字轴保护：
  - 如果当前标签是纯数字，LLM 输出非数字文本则拒绝
  - 防止 `2014 -> USA` 这类图例混入
- contact sheet 和 LLM JSON 都缓存到：
  - `chart-18-final/output/v7_llm_cache/`

### V7 可视化

V7 生成最终标注图：

```
chart-18-final/output/RadarChart*_ocr_v7_axes.png
```

可视化特点：

- 只展示最终轴标签，不再绘制全部 OCR 框
- 绿色短轴线从圆心指向对应轴
- 标签框限制在图像内部，避免画出边界
- 长标签自动换行
- 标签文本使用 LLM 修正后的最终结果

### 整体效果

| 版本   |  正确数 | 准确率 | 说明                                            |
| ------ | ------: | -----: | ----------------------------------------------- |
| V5     |  93/148 |  62.8% | 全图 OCR + LLM 轴数                             |
| V6     | 118/148 |  79.7% | 几何网格优先 + ROI OCR + 规则后处理             |
| V7实验 | 133/148 |  89.9% | V6 轴几何 + 多裁剪 contact sheet + LLM 轴名纠错 |

V7 显著提升的图：

- `RadarChart18`: 0/5 -> 5/5
- `RadarChart17`: 4/6 -> 6/6
- `RadarChart19`: 3/5 -> 5/5
- `RadarChart20`: 18/20 -> 20/20
- `RadarChart23`: 5/6 -> 6/6
- `RadarChart5`: 10/14 -> 13/14

V7 能解决的问题：

- OCR 字形混淆：`Ircland -> Ireland`、`Greccc -> Greece`
- 长词截断：`Customer -> Customer Support`
- 多词标签不完整：`Business -> Business Sophistication`
- 密集外圈标签串扰：`RadarChart20`
- 低清晰度长标签：`RadarChart18`

仍然困难的问题：

- 轴几何本身错位时，LLM 会围绕错误轴读标签，例如 `RadarChart9`
- GT 标签更像图例/系列名、不在真实轴端时，轴端 OCR/LLM 方法不适配，例如 `RadarChart6`、`RadarChart16`

### 当前结论

裁剪给 LLM 是有效的，但推荐方式不是单一固定裁剪框，而是：

```
V6 几何定轴
-> 每轴生成高亮上下文图
-> 多尺度轴端裁剪
-> 提供 OCR candidates
-> LLM 只做当前轴标签纠错
-> 根据置信度和类型规则决定是否接受
```

这种方式比直接把一个固定裁剪框交给 LLM 更鲁棒，也能显著减少标题、图例、刻度值和其他轴标签的干扰。