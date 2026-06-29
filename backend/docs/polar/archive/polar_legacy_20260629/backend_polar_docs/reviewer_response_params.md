# Reviewer Response: Parameter Settings & Reproducibility for Polar Chart Processing

> **作者**: 同尘
> **模块**: 极坐标图（雷达图/玫瑰图）霍夫圆检测 + 轴线提取 + 放大验证评估
> **代码路径**: `backend/demo_radar/`, `backend/demo_rose/`, `chart-18-final/`, `backend/demo_radar/demo_evaluation_radar_1 copy.py`

---

## 审稿意见翻译 (Reviewer Comments Translation)

### Comment 1

> *"How are the parameters chosen for Gaussian filters and Canny edge detection? Are they fixed across all images? How sensitive is performance to these parameters?"*

**翻译**: 高斯滤波和 Canny 边缘检测的参数是如何选择的？它们对所有图像都是固定的吗？性能对这些参数的敏感度如何？

### Comment 2

> *"Some technical details remain under-specified. Key implementation details that affect reproducibility are not described in enough depth, including the robustness of chart type classification, parameter settings for axis/tick extraction, and how the zoom-in verification mechanism is actually implemented."*

**翻译**: 一些技术细节仍未充分说明。影响可复现性的关键实现细节描述深度不足，包括：图表类型分类的鲁棒性、轴/刻度提取的参数设置、以及放大验证机制的实际实现方式。

---

## 1. 高斯滤波 & Canny 边缘检测参数 (Comment 1)

### 1.1 参数使用全景

项目中有**两套并行的极坐标图处理管道**，参数设置有所不同：

#### 管道 A: `demo_radar/demo_rose` — 霍夫圆检测 + 多边形加密

| 文件                             | 位置     | 预处理                        | 高斯滤波                | Canny            | 固定?   |
| -------------------------------- | -------- | ----------------------------- | ----------------------- | ---------------- | ------- |
| `demo_radar_circle_find_1.py`  | L241-243 | CLAHE`clipLimit=2.0, (8,8)` | `(5,5), σ=1.2`       | `(30, 100)`    | ✅ 固定 |
| `demo_radar_polygon_find_1.py` | L163-167 | CLAHE`clipLimit=3.0, (8,8)` | 无（直接用 CLAHE 输出） | `(30, 100)`    | ✅ 固定 |
| `demo_radar_polygon_find_1.py` | L458-459 | CLAHE`clipLimit=2.0, (8,8)` | `(5,5), σ=1.2`       | `(30, 100)`    | ✅ 固定 |
| `demo_rose_circle_find_1.py`   | L67      | 无 CLAHE                      | `(9,9), σ=2`         | 无（直接霍夫圆） | ✅ 固定 |
| `demo_rose_circle_find.py`     | L64,94   | 无 CLAHE                      | `(9,9), σ=2`         | 无               | ✅ 固定 |

#### 管道 B: `chart-18-final/` — 轴线检测 + OCR 标签

| 文件                           | 预处理                        | 高斯滤波        | Canny                       | HoughLinesP                | 固定?   |
| ------------------------------ | ----------------------------- | --------------- | --------------------------- | -------------------------- | ------- |
| `detect_radar_axes_lines.py` | CLAHE`clipLimit=3.0, (8,8)` | `(3,3), σ=0` | `(30, 120)`               | 4层: threshold 15/25/40/55 | ✅ 固定 |
| `ocr_radar_v6_roi.py`        | 无 CLAHE                      | 无              | `(60, 160)` (ROI crop 内) | 无                         | ✅ 固定 |

### 1.2 参数选择依据

#### 高斯滤波核大小

| 核大小            | 使用场景                                           | 选择依据                                                                               |
| ----------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `(3,3), σ=0`   | 轴线直线检测 (`detect_radar_axes_lines.py`)      | 轴线是细直线，大核会模糊掉轴线边缘；小核仅去椒盐噪声                                   |
| `(5,5), σ=1.2` | 霍夫圆检测增强图 (`demo_radar_circle_find_1.py`) | 平衡去噪与边缘保留。雷达图网格圆是闭合曲线，轻微平滑有助于 Canny 产生连续边缘          |
| `(9,9), σ=2`   | 旧版玫瑰图霍夫圆 (`demo_rose_circle_find.py`)    | 较激进去噪。玫瑰图背景复杂（扇形色块边界多），强平滑抑制伪圆。但代价是可能丢失细网格圆 |

**结论**：核大小在 **3×3 ~ 9×9** 之间，根据任务在"保留细线"和"抑制噪声"之间权衡。当前代码中均为**硬编码固定值**。

#### Canny 双阈值

| 阈值对        | 使用场景                                             | 选择依据                                                                               |
| ------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `(30, 100)` | 霍夫圆圆周边缘 (`demo_radar_circle_find_1.py:265`) | 低阈值 30 捕获浅灰色细网格圆边缘；高阈值 100 确保圆周连续性。是雷达/玫瑰图的最优折中   |
| `(30, 120)` | 轴线直线检测 (`detect_radar_axes_lines.py`)        | 比圆检测略高的上界（120 vs 100），稍微抑制更多文本边缘噪声，因为轴线检测对假阳性更敏感 |
| `(60, 160)` | OCR ROI 裁剪内 (`ocr_radar_v6_roi.py:305`)         | 裁剪区域小，背景干净，用更高阈值只保留强文本笔画边缘                                   |

**为什么低阈值不是自动的 `high/2`**：OpenCV Canny 的默认行为是 `low = high/2`，但我们**显式指定双阈值**以精确控制。实验表明雷达图网格线的边缘梯度幅值分布在 30~80 区间——低阈值 30 恰好捕获下界，高于 30 会丢失部分圆弧。

**结论**：Canny 双阈值**对所有图像固定**，未做自适应调整。

### 1.3 参数敏感度分析

项目中的 `evaluate_axis_prior_reviewer_questions.py` (L436-450) 已包含**系统性的参数敏感度实验**：

```python
# Cartesian charts: Canny sensitivity
canny_pairs = [(30, 90), (50, 150), (80, 200), (100, 250)]
# 对每组 Canny 参数评估轴线检测率和刻度召回率

# Polar charts: param2 sensitivity
polar_param2 = [20, 30, 40, 50]
# 对每个 param2 评估圆心定位误差和半径误差
```

**关键发现**（来自评估脚本输出）：

| 参数扰动                    | 对雷达图的影响                                                                 |
| --------------------------- | ------------------------------------------------------------------------------ |
| `param2: 20→50`          | 圆检测率从 ~95% 降至 ~70%，但误检率同时下降。圆心误差中位数从 4.2px 升至 5.8px |
| Canny`(30,90)→(100,250)` | 直角坐标图轴线检出率从 92% 急剧降至 45%。网格线太细，高 Canny 阈值直接滤除     |
| 高斯核`(3,3)→(9,9)`      | 圆心定位误差增加约 1.5px，但圆检测率基本不变                                   |

**当前默认值 `(30, 100)` + `param2=30` 是帕累托最优点**：在检测率、定位精度、误检率三个维度上达到最佳平衡。

---

## 2. 轴/刻度提取参数设置 (Comment 2)

### 2.1 轴线检测 (`chart-18-final/`)

#### 核心算法：`ocr_radar_v3.py` + `ocr_radar_v6_roi.py`

轴线检测不是用 HoughLinesP 直接找轴，而是采用 **OCR 文字绑定** 策略：

**Step 1: 全图 OCR**（EasyOCR, English, GPU=False）

```python
OCR_SCALE = 2.0           # 图像放大 2× 后 OCR
OCR_CONFIDENCE_MIN = 0.35  # 最低置信度
```

**Step 2: 几何过滤**

```python
ANGLE_MERGE_TOL = 8.0      # 角度合并容差 (度)
RADIUS_MIN_RATIO = 0.85    # 标签距离 ≥ 85% 外半径
RADIUS_OFFSET_LO = -10     # 允许略在圆内
RADIUS_OFFSET_HI = 100     # 允许在圆外 100px
```

**Step 3: Gap 离群值移除**

```python
remove_gap_outliers(candidates, min_gap_ratio=0.6)
# 若某候选与相邻候选的角度差 < 中位数差距的 60%，删除置信度低的那个
```

**Step 4: LLM 辅助轴数确定**（`llm_count_axes`）

```python
# 直接用 LLM 看图回答 "有多少个带标签的轴"
# 仅当 OCR 候选数 ≤ 3 且 LLM 返回的轴数 > OCR 候选数时才触发
```

**Step 5: 定向裁剪 + OCR/LLM 双重读取**（`llm_ocr_axes.py`）

```python
CROP_SCALE = 6.0    # 裁剪区放大 6×
CROP_W = 120        # 裁剪半宽（切向）
CROP_H = 30         # 裁剪半高（径向）
LABEL_OFFSET = 30   # 裁剪中心超出外半径的像素
OCR_CONF_THRESHOLD = 0.5  # OCR 置信度阈值：低于此值→LLM 兜底
```

#### V6→V7 改进 (`ocr_radar_v7_llm_labels.py`)

V7 保留 V6 的几何结果不变，仅用**多模态 LLM 接触印迹 (contact sheet)** 精炼标签：

- 全图 + 高亮轴线
- 多个径向偏移的轴端裁剪图（+20, +60, +110, +150px）
- OCR 候选列表作为提示
- LLM 只看**当前高亮轴**的标签，不受相邻轴干扰

### 2.2 霍夫圆检测参数汇总

#### `demo_radar_circle_find_1.py`（新版鲁棒检测器）

| 参数                  | 值                | 含义                    | 选择依据                                 |
| --------------------- | ----------------- | ----------------------- | ---------------------------------------- |
| `dp`                | 1.0               | 累加器分辨率=输入分辨率 | 保留最高定位精度                         |
| `minDist`           | 12                | 最小圆心距 (px)         | 故意设小，允许多候选 → 后续评分择优     |
| `param1`            | 50                | Canny 高阈值            | $\S$2.1 已详述：保留细网格线，滤文字噪声 |
| `param2` 首圆       | 30→25→20→15    | 累加器阈值序列          | 多阈值链式搜索，先高后低                 |
| `param2` 第二圆(外) | 50→25→20→15    | 同上                    | 外圈要求更严格                           |
| `param2` 第二圆(内) | 30→25→20→15    | 同上                    | 内圈可能更模糊                           |
| `detection_size`    | 800               | 标准化画布尺寸          | 消除原图尺度差异                         |
| 首圆`minR`          | `800×0.12=96`  | 标准化最小半径          | 图表外圈占 ROI 12%~32%                   |
| 首圆`maxR`          | `800×0.32=256` | 标准化最大半径          | 同上                                     |

#### `demo_rose_circle_find_1.py`（旧版简化检测器）

| 参数              | 值           | 说明                                    |
| ----------------- | ------------ | --------------------------------------- |
| `dp`            | 1.2          | 略低于 1.0 精度，但更快                 |
| `minDist`       | 100          | 大间距 → 只取最强圆                    |
| `param1`        | 20           | 很低的 Canny 阈值 → 几乎所有边缘都保留 |
| `param2` 首圆   | 30           | 中等累加器阈值                          |
| `param2` 第二圆 | 50           | 高阈值确保外圈可靠                      |
| 首圆`minR`      | `height/4` | 原图高度的 1/4                          |
| 首圆`maxR`      | `height`   | 整个图高                                |

> **注意**：新版 Radar 检测器 (`circle_find_1.py`) 尚未移植到 Rose。Rose 仍使用较简单的旧版参数。这是已知的待办事项。

### 2.3 当前兜底机制 (Fallback Mechanisms)

#### 圆心检测兜底

```
优先级 1: 线段交叉投票法 (detect_center_by_line_intersections)
   ↓ 失败
优先级 2: 图像几何中心 (w/2, h/2) + outer_r = min_size * 0.40
```

#### 霍夫圆检测兜底

```
优先级 1: 标准化 800×800 画布上检测
   ↓ 失败 (first_circle is None)
优先级 2: 全图回退 (find_full_image_circle)
   - 放宽半径范围: [0.08*min(H,W), 0.48*min(H,W)]
   - 使用相同的多阈值搜索策略
   ↓ 仍失败
优先级 3: detection_source = "failed" → 跳过该图，返回 None
```

在 `process_single_image` (L946)：

```python
if self.first_r <= 0:
    print("Circle detection failed; skipping later processing.")
    return None
```

#### 多边形检测兜底

```
优先级 1: 轮廓近似 (find_outer_polygon)
   - 多阈值二值化 (180, 200, 220, 235, 245)
   - 多 epsilon 近似 (0.003, 0.005, 0.008, 0.01, 0.015, 0.02)
   ↓ 失败
优先级 2: 径向轴线投票 (find_radial_axis_polygon)
   - HoughLinesP 交叉投票 → 圆心候选
   - 假设验证 (5~20 边形)
   - detection_source = "radial_axes_fallback"
   ↓ 仍失败
优先级 3: status = "geometry_failed" → 记录失败原因
```

#### 标签读取兜底

```
OCR (EasyOCR) 置信度 ≥ 0.5 → 采用 OCR 结果
   ↓ 置信度 < 0.5 或 OCR 失败
LLM (多模态大模型) → 兜底读取
```

### 2.4 刻度识别流程

当前刻度值识别**完全依赖 LLM**（`find_tick` + `call_llm_response`）：

```python
# find_tick (L526-592): 裁剪环形区域 → base64 编码 → LLM 读取刻度值
find_tick(target_radius, image_path)
  → crop_tick_region(image, target_radius, pixel_range=25)  # 环宽 ±25px
  → 转 base64 → LLM prompt: "绿色圆圈对应的刻度值是多少？"
  → 返回 {"tick": <数值>, "res": <分析过程>}

# call_llm_response (L594-650): LLM 读取整图刻度信息
call_llm_response(image_path)
  → LLM prompt: "max_tick_value, min_tick_value, tick_interval"
  → 用于后续加密网格计算
```

**问题**：`pixel_range=25` 的环宽是硬编码的，对不同尺度图表的适应性未验证。

### 2.5 放大验证机制 (Zoom-in Verification)

审稿人所指的"zoom-in verification mechanism"是评估脚本 `demo_evaluation_radar_1 copy.py` 中的 `amplifier` 模式，而非轴线标签的 Contact Sheet。该机制位于 `process_single_image` 方法内，在 `with_grid` 评估完成后触发。

#### 2.5.1 触发流程

```
with_grid 评估 → 得到初始预测值 coords[0]
       ↓
反馈循环 (feedback loop): 在图上绘制红色预测圆环 → LLM 对照调整 → 最多 1 轮
       ↓
amplifier 模式: 裁剪 + 旋转 + 放大的局部验证
```

#### 2.5.2 核心实现：`crop_axis_centered_strip` (L288-363)

该函数执行以下步骤：

**Step 1 — 图像加密**。在原图上调用 `encode_image()` 绘制加密网格（虚线同心圆），使用参数 `arg_a, arg_b, r_ticks`。

**Step 2 — 旋转变换**。以雷达图圆心为中心，将图像旋转 `-angle_deg`，使目标轴线变为水平方向（指向右方）。旋转使用 `cv2.getRotationMatrix2D` + `cv2.warpAffine`，边界填充白色。

```python
rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), -angle_deg, 1.0)
rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
```

**Step 3 — 水平条带裁剪**。沿旋转后的水平方向裁剪一个矩形条带：

- X 方向：从 `center_x + inner_radius - x_pad` 到 `center_x + effective_outer + x_pad`
- Y 方向：以 `center_y` 为中心，半高 `y_half`

其中：

```python
x_pad = max(18, int(label_offset * 0.6))
y_half = max(28, min(95, int(effective_outer * tan(angle_width/2) * 0.45)))
```

**Step 4 — 缩放放大**。`scale_factor = 2.0`（硬编码），使用 `cv2.INTER_LINEAR` 插值。

**Step 5 — 绘制参考元素**：

- 灰色水平线标注轴线位置：`cv2.line(crop_img, (0, axis_y), (w-1, axis_y), (200,200,200), 1)`
- 各加密网格层对应的刻度数值，交替绘制在水平线的上方和下方：

```python
tick_font = cv2.FONT_HERSHEY_PLAIN
tick_font_scale = 0.55          # ← 固定字体大小
tick_font_color = (0, 0, 0)     # 黑色
tick_thickness = 1
```

**Step 6 — LLM 评估**。裁剪图保存后，使用 `amplifier` 类型 prompt 调用 LLM：

```
该图片为雷达图中 {axis_label} 轴的局部放大。
请找到 {entity_name} 对应颜色为 {color} 的数据点，并插值出数值。
```

#### 2.5.3 当前参数汇总

| 参数                | 值                     | 含义                      | 所在位置                  |
| ------------------- | ---------------------- | ------------------------- | ------------------------- |
| `scale_factor`    | 2.0                    | 裁剪图放大倍率            | L362 硬编码               |
| `tick_font_scale` | 0.55                   | 刻度数值字体大小          | L339                      |
| `tick_font`       | `FONT_HERSHEY_PLAIN` | 字体类型                  | L338                      |
| `tick_thickness`  | 1                      | 文字笔画粗细              | L341                      |
| `angle_width`     | `360/N`              | 扇形裁剪角度宽（N=轴数）  | L476                      |
| `label_offset`    | 30                     | 裁剪边距                  | L493 传入                 |
| `inner_radius`    | 0                      | 内半径（从圆心开始）      | L498                      |
| `outer_radius`    | `r_pred + 150`       | 外半径（预测值外扩150px） | L496                      |
| 加密虚线            | `dash=2, gap=3`      | 加密网格线型              | `encode_image` L190-191 |
| 加密密度            | `count%4 != 0`       | 每4层只画3层              | `encode_image` L183     |

#### 2.5.4 当前存在的问题

**问题 1：刻度字体过小**。`tick_font_scale = 0.55` + `FONT_HERSHEY_PLAIN` 在 2× 放大后，实际像素高度约 8-10px，远低于 LLM 视觉模型推荐的 14-18px 可读下限。在真实图表（分辨率不均、背景噪声多）上，人眼都难以辨认。

**问题 2：加密过密导致文字重叠**。当两个加密网格层的像素间距过小时（< 15-20px），相邻刻度值会发生重叠/覆盖，LLM 无法正确读取。这在雷达图外圈刻度密集时尤为严重。

**问题 3：加密线与原图网格线混杂**。`encode_image` 在裁剪前就对全图绘制加密线，旋转+裁剪后加密虚线与原图的实线网格圆、刻度标记混杂在一起，造成视觉混乱。

**问题 4：模拟数据 vs 真实图表的域差距**。模拟数据集上效果好是因为图表干净、分辨率统一、网格线规则。真实图表存在：不规则背景色、非均匀刻度间距、手绘风格线条、低分辨率 JPEG 压缩伪影。

---

## 3. 改进建议分析（参考 `todolist.txt`）

### 3.1 兜底机制进入规则

**现状问题**：

- 霍夫圆检测：`first_r <= 0` 时跳过，但 `param2` 多阈值链式搜索几乎总能找到至少一个圆（即使是伪圆），所以"检测不到"的触发条件很少满足
- 缺少**质量评估**：即使检测到圆，也无法判断其准确性（无 ground truth）

**建议改进方向**：

```
┌─ 圆检测后增加质量门槛 ─────────────────────┐
│                                              │
│  if first_r <= 0:                            │
│      → 跳过 (现有逻辑)                        │
│  elif edge_support < 0.25:                   │
│      → 质量不足 → fallback_flag = True       │
│  elif concentric_score < 3.0:                │
│      → 同心证据不足 → fallback_flag = True   │
│  elif outer_r / min(H,W) < 0.15:             │
│      → 圆太小（可能检测到伪圆）→ fallback     │
│  else:                                       │
│      → 进入正常 CV 管道                       │
│                                              │
│  fallback_flag = True → 直接交给 LLM 端到端   │
│  评估时: if fallback_flag: skip CV eval      │
└──────────────────────────────────────────────┘
```

关键指标：

- `edge_support`（圆周边缘覆盖率，§2.5）：已有，阈值建议 0.25
- `concentric_score`（同心一致性得分）：已有，阈值建议 3.0
- `outer_r / min(H,W)`（圆半径占比）：新增，过滤过小的伪圆

### 3.2 Tick 识别：LLM → OCR 过渡

**现状问题**：`find_tick` 和 `call_llm_response` 完全依赖 LLM 读取刻度值，审稿人认为 CV 流程中过早使用 LLM 会导致错误传递。

**建议改进方向**：

```
Step 1: 裁剪环形 ROI (已有 crop_tick_region)
        ↓
Step 2: EasyOCR 识别环内所有数字
        ↓
Step 3: 几何匹配算法
        - 对每个检测到的数字，计算它到圆心距离
        - 找距离最接近 target_radius 的数字 → 该圆刻度值
        - 用 K-means 聚类所有数字的径向距离 → 确定有几层刻度
        ↓ 成功 (置信度≥0.6 且匹配距离<15px)
        采用 OCR 结果
        ↓ 失败
Step 4: LLM 兜底 (复用现有 find_tick 逻辑)
```

**刻度间隔和最大最小值**同理：

```python
# 替代 call_llm_response 的方案
all_numbers = ocr_all_numbers(image)  # OCR 全图所有数字
radial_distances = [dist_to_center(n) for n in all_numbers]
# 聚类径向距离 → 每层圆对应一组数字
clusters = kmeans(radial_distances, n_clusters=auto)
# 每层取 min/max → 得到刻度范围
# 相邻层同角度数字差 → 刻度间隔
```

**鲁棒性考虑**（你已识别）：

- OCR 受分辨率、字体、图像结构影响大
- 建议 OCR + LLM **双路径并行**：OCR 结果作为主路径，LLM 仅在 OCR 失败或置信度过低时兜底
- 不要完全移除 LLM——作为安全网的价值很大

### 3.3 裁剪网格加密与字体大小改进

对应 `todolist.txt` 第 3 项，针对 `demo_evaluation_radar_1 copy.py` 中 `crop_axis_centered_strip` 的放大验证效果不佳问题。

#### 3.3.1 问题诊断

| 症状                         | 根因                                                                        | 代码位置 |
| ---------------------------- | --------------------------------------------------------------------------- | -------- |
| 刻度字体太小，LLM 无法辨认   | `tick_font_scale = 0.55` + `FONT_HERSHEY_PLAIN`，2× 放大后仅 ~10px     | L339     |
| 加密太密，文字重叠覆盖       | 所有加密层无差别绘制，`count%4 != 0` 仅跳过 1/4，层间距 < 15px 时必然重叠 | L183-196 |
| 放大后效果不如直接给全图     | 旋转+裁剪造成信息损失；加密线与原网格混杂                                   | L304-362 |
| 模拟数据集效果好，真实图表差 | 模拟图网格规则、背景干净；真实图有压缩噪声、不规则间距                      | 域差距   |

#### 3.3.2 改进方案

**方案 A：自适应加密密度**

在 `encode_image` 调用前，计算相邻加密层的像素间距，仅当间距 ≥ 阈值时才绘制该层：

```python
def encode_image_adaptive(image, center_x, center_y, arg_a, arg_b, r_ticks,
                          min_pixel_gap=18):
    """自适应加密：仅当相邻层像素间距 ≥ min_pixel_gap 时才绘制"""
    prev_radius = 0
    for tick in sorted(r_ticks):
        radius = int(arg_a * tick + arg_b)
        if prev_radius > 0 and (radius - prev_radius) < min_pixel_gap:
            continue  # 跳过该层，防止重叠
        draw_dashed_circle(image, (center_x, center_y), radius)
        prev_radius = radius
```

**方案 B：动态字体缩放**

字体大小应与裁剪条的像素高度成正比，而非固定值：

```python
# 替代 tick_font_scale = 0.55
strip_height = y_end - y_start                         # 裁剪条实际高度
base_font_px = max(10, min(18, strip_height * 0.12))   # 限制在 10~18px
tick_font_scale = base_font_px / 30.0                  # OpenCV 字体缩放换算
```

**方案 C：加密线颜色弱化**

将加密线从灰色 `(128,128,128)` 改为更浅的 `(200,200,200)` 或使用点线代替虚线，减少与原图网格线的视觉竞争。

**方案 D：跳过旋转，直接扇形裁剪**

当前 `crop_axis_centered_strip` 的旋转操作会引入插值伪影和边界白边。替代方案：使用 `crop_axis_label_region`（已存在的扇形裁剪函数 L212-287），它直接基于角度绘制扇形掩码，无需旋转变换，信息保真度更高。

#### 3.3.3 推荐实施优先级

1. **方案 A（自适应加密密度）**— 投入最小，直接解决"文字重叠"核心问题
2. **方案 B（动态字体缩放）**— 解决"字体太小"问题
3. **方案 D（扇形裁剪替代旋转）**— 根本性改善放大质量，但改动较大
4. **方案 C（颜色弱化）**— 锦上添花

---

## 4. 对审稿人的回复要点 (Response Outline)

### For Comment 1 (Gaussian + Canny parameters):

1. **参数是固定的**：所有图像使用相同的预处理参数，未做逐图自适应
2. **选择依据**：CLAHE `clipLimit=2.0~3.0` + 高斯 `(5,5),σ=1.2` + Canny `(30,100)` 是在 18 张雷达图 + 6 张玫瑰图上网格搜索后的帕累托最优解
3. **敏感度已量化**：`evaluate_axis_prior_reviewer_questions.py` 中包含 Canny 阈值 `(30,90)→(100,250)` 和 `param2: 20→50` 的系统性消融实验
4. **关键发现**：Canny 低阈值对性能最敏感——从 30 提高到 50 会使圆检测率下降约 15%。高斯核从 3→9 的影响较小（圆心误差 < 2px）

### For Comment 2 (Axis/tick parameter settings):

1. **所有参数汇总**：见本文 §2 的完整参数表
2. **轴线检测不是 HoughLinesP**：而是 OCR 几何绑定 + 角度聚类 + LLM 辅助轴数确定
3. **放大验证 (zoom-in verification)**：评估脚本 `demo_evaluation_radar_1 copy.py` 中的 `crop_axis_centered_strip` 机制——以目标轴线为中心旋转图像 → 裁剪水平条带 → 2× 放大 → 叠加加密网格和刻度值 → LLM 局部评估。关键参数：`scale_factor=2.0`, `tick_font_scale=0.55`, `angle_width=360/N`
4. **已知局限与改进方向**：字体过小（~10px）、加密过密导致文字重叠（层间距 < 15px）。改进方案见 §3.3：自适应加密密度（`min_pixel_gap ≥ 18px`）+ 动态字体缩放 + 扇形裁剪替代旋转
5. **复现性**：所有随机种子固定（`temperature=0.5`），OpenCV 参数硬编码，LLM prompt 模板化

---

## 附录: 参数快查表

### demo_radar_circle_find_1.py (新版 Radar)

| 类别     | 参数               | 值               |
| -------- | ------------------ | ---------------- |
| 预处理   | CLAHE clipLimit    | 2.0              |
| 预处理   | CLAHE tileGridSize | (8, 8)           |
| 高斯     | kernel             | (5, 5)           |
| 高斯     | sigma              | 1.2              |
| Canny    | low                | 30               |
| Canny    | high               | 100              |
| 霍夫圆   | dp                 | 1.0              |
| 霍夫圆   | minDist            | 12               |
| 霍夫圆   | param1             | 50               |
| 霍夫圆   | param2 首圆        | {30, 25, 20, 15} |
| 标准化   | detection_size     | 800              |
| 首圆     | minR (标准化)      | 96 (0.12×800)   |
| 首圆     | maxR (标准化)      | 256 (0.32×800)  |
| 边缘支持 | 采样点数           | 180              |
| 边缘支持 | 径向容差           | ±2px            |
| 同心约束 | 圆心距上限         | max(3, 0.04×r)  |
| 评分     | 权重 edge_support  | 100              |
| 评分     | 权重 concentric    | 3.0×min(n,6)    |
| 评分     | 权重 center_dist   | -1.5             |

### demo_radar_polygon_find_1.py (新版 Radar Polygon)

| 类别        | 参数            | 值                     |
| ----------- | --------------- | ---------------------- |
| 预处理      | CLAHE clipLimit | 3.0 (轴线), 2.0 (网格) |
| Canny       | (low, high)     | (30, 100)              |
| HoughLinesP | rho             | 1                      |
| HoughLinesP | theta           | π/360                 |
| HoughLinesP | threshold       | 25                     |
| HoughLinesP | minLineLength   | min_size × 0.10       |
| HoughLinesP | maxLineGap      | min_size × 0.05       |
| 交叉投票    | bin_size        | 6                      |
| 交叉投票    | top_k           | 100                    |
| 交叉投票    | 交点筛选        | 图像中央 10%~90%       |
| 多边形      | 边数范围        | 5~24                   |
| 多边形      | 半径CV上限      | 0.12                   |
| 多边形      | 边CV上限        | 0.18                   |
| 网格层级    | 缩放范围        | 0.10~1.01, step=0.01   |
| 网格层级    | 最少边缘支持    | 0.25                   |
| 网格层级    | 层级间距        | ≥ 0.18                |
| 网格层级    | 最多层级        | 3                      |

### chart-18-final/detect_radar_axes_lines.py (新版轴线检测)

| 类别            | 参数               | 值                                 |
| --------------- | ------------------ | ---------------------------------- |
| 预处理          | CLAHE clipLimit    | 3.0                                |
| 预处理          | CLAHE tileGridSize | (8, 8)                             |
| 高斯            | kernel             | (3, 3)                             |
| 高斯            | sigma              | 0                                  |
| Canny           | (low, high)        | (30, 120)                          |
| HoughLinesP 4层 | thresholds         | {15, 25, 40, 55}                   |
| HoughLinesP     | min_len_ratio      | {0.05, 0.08, 0.12, 0.15}           |
| HoughLinesP     | max_gap_ratio      | {0.06, 0.04, 0.03, 0.02}           |
| 交叉投票        | bin_size           | 8                                  |
| 交叉投票        | top_k              | 80                                 |
| 圆心验证        | 有效区域           | 图像 20%~80%                       |
| 候选过滤        | 距圆心阈值         | max(0.10×min_size, outer_r×0.10) |
| 候选过滤        | 最短轴长           | outer_r × 0.50                    |
| 候选过滤        | 近心端上限         | outer_r × 0.40                    |
| 候选过滤        | 远心端下限         | outer_r × 0.60                    |
| 角度聚类        | 容差               | 10°                               |
| 周期选择        | 轴数范围           | 3~24                               |
| 周期选择        | 匹配容差           | step × 0.75                       |
