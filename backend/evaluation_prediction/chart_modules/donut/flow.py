# -*- coding: utf-8 -*-
"""
donut_only_experiment.py — **专为 donut / pie / ring charts 设计**的最小可运行批量实验脚本。

▶ 依赖
   * Python ≥ 3.9
   * pip install pandas matplotlib pillow aiohttp
▶ 使用
   1. 把每个图表的元数据 JSON 放进 ``chart_configs/`` 文件夹，示例：
      ```json
      {
        "chart_id": "chart01",
        "chart_type": "donut",              // 必须 = "donut"
        "image_paths": {
          "no_grid": "charts/chart01.png",
          "with_grid": "charts/chart01_grid.png"
        },
        "data_points": {
          "Samsung Mobile": 20,
          "Redmi Mobile"  : 35,
          "Nokia Mobile"  : 10,
          "LG"            :  5,
          "Oppo"          : 15,
          "VIVO"          : 15
        }
      }
      ```
   2. 运行：
      ```bash
      python donut_only_experiment.py --chart-ids chart01
      ```
   3. 结果与评估 CSV/PNG 会写入 ``results/<chart_id>/``

与原 bar/line 脚本相比，删除了所有 Cartesian‑axis 逻辑，保留 **baseline** 与 **grid** 两种 Prompt‑Image 组合。
"""
from __future__ import annotations
import os, asyncio, argparse
from typing import Dict, List
from PIL import Image
import pandas as pd
from ...common.runtime import get_repeat_times
from .angle_grid import draw_angle_grid_30deg
from .evaluation import compute_mae, compute_relative_error, save_summary_and_plot
from .data import load_chart_configs
from .model import call_llm_once
from .prompts import generate_prompt
from .visual import draw_angle_feedback, crop_sector_for_amplifier


# ────────────────────────────────────────────────────────────
# 2) DATA / CONFIG LOADING
# ────────────────────────────────────────────────────────────
EXPERIMENT_TYPES = [
    ("baseline", "no_grid"),
    ("grid", "with_grid"),
    # ("feedback", "with_grid"),
    ("amplifier", "with_grid"),
]
REPEAT_TIMES = get_repeat_times()
MAX_ATTEMPTS = 10



DATASET_CONFIGS = load_chart_configs()


# ────────────────────────────────────────────────────────────
# 3) PROMPT GENERATION (DONUT ONLY)
# ────────────────────────────────────────────────────────────



async def run_dataset_segmentwise_feedback(
    cfg: Dict, records: List[Dict],
    repeat_rounds: int = 3,
    use_amplifier: bool = True,
    amp_img_type: str = "with_grid"
):
    chart_id = cfg["chart_id"]
    img_path = cfg["image_paths"][amp_img_type]
    print(f"[AMPLIFIER] src image = {img_path}")

    # === 基本图参数 ===
    with Image.open(img_path) as img:
        w, h = img.size

    circle_center = tuple(cfg.get("center", (w // 2, h // 2)))
    inner_radius = int((cfg.get("r_pixels") or [None, min(w, h) // 4])[1])
    outer_radius = int(cfg.get("outer_radius", min(w, h) // 2))

    # ============================================================
    # 对每一个扇区（point）执行 grid → feedback → feedback → amplifier
    # ============================================================
    for point in cfg["data_points"].keys():

        last_pred = None
        img_to_use = img_path   # 每个 point 第一轮使用完整图

        # ============================================================
        # PHASE 1 — GRID + FEEDBACK（3轮）
        # ============================================================
        for round_idx in range(repeat_rounds):

            if last_pred is None:
                prompt_type_this_round = "grid"
            else:
                prompt_type_this_round = "feedback"

            prompt = generate_prompt(
                item_name=point,
                prompt_type=prompt_type_this_round,
                prev_angle=last_pred
            )

            print(
                f"\n📌 Prompt type = [{prompt_type_this_round}] for [{point}] – "
                f"Image: {img_to_use} | Round {round_idx + 1}\n{prompt}\n"
            )

            pred = await call_llm_once(prompt, img_to_use)

            # ---- 预测失败 ----
            if not (
                pred and isinstance(pred, dict)
                and "start_angle" in pred
                and "end_angle" in pred
            ):
                print(f"❌ 预测失败 [{chart_id} – {point} – round {round_idx + 1}]")
                break

            last_pred = pred

            # ---- 记录反馈结果 ----
            is_final_feedback = (
                prompt_type_this_round == "feedback"
                and round_idx == repeat_rounds - 1
            )

            pred_pct = ((pred["end_angle"] - pred["start_angle"] + 360) % 360) / 360
            gt_pct = cfg["data_points"][point]

            rec = {
                "chart_id": chart_id,
                "point": point,
                "prompt_type": prompt_type_this_round,
                "image_type": "with_grid",
                "gt": gt_pct,
                "pred": pred,
                "round_index": round_idx + 1,
            }

            if is_final_feedback:
                rec["pred_pct"] = pred_pct
                rec["mae"] = compute_mae(pred_pct, gt_pct)
                rec["rel_err"] = compute_relative_error(pred_pct, gt_pct)

            records.append(rec)

            # ---- 绘制 feedback 图，用于下一轮 ----
            if round_idx < repeat_rounds - 1:

                filename = f"{point}_feedback_round{round_idx + 1}.png"
                feedback_img_path = os.path.join("results_Pixtral", chart_id, filename)

                try:
                    img_to_use = draw_angle_feedback(
                        image_path=img_path,
                        angle_deg=[pred["start_angle"], pred["end_angle"]],
                        output_path=feedback_img_path,
                        circle_center=circle_center,
                        inner_radius=inner_radius,
                    )
                    print(f"[🖍️] Round {round_idx + 1} feedback drawn → {img_to_use}")

                except Exception as e:
                    print(f"[⚠️] 绘制反馈失败：{e}")
                    break

        # ============================================================
        # PHASE 2 — AMPLIFIER（只执行一次，但包含 3 轮）
        # ============================================================
        if use_amplifier and last_pred is not None:

            amp_pred = last_pred  # 初始 amplifier 预测来自 feedback 最后一轮
            amp_rounds = 3

            for amp_round_idx in range(amp_rounds):

                current_amp_round = amp_round_idx + 1

                # ---- 裁剪放大扇区 ----
                img_to_use, drawn_angles, angle_order_hint = crop_sector_for_amplifier(
                    image_path=cfg["image_paths"]["no_grid"],
                    centre=circle_center,
                    inner_r=inner_radius,     # FIXED
                    outer_r=outer_radius,     # FIXED
                    feedback_angles=amp_pred,
                    chart_id=chart_id,
                    point_name=point,
                    save_suffix=f"_amp{current_amp_round}",
                    amp_round=current_amp_round,
                )

                # ---- amplifier prompt ----
                amp_prompt = generate_prompt(
                    item_name=point,
                    prompt_type="amplifier",
                    prev_angle=amp_pred,
                    drawn_angles=drawn_angles,
                    angle_order_hint=angle_order_hint
                )

                print(
                    f"\n📌 Amplifier Prompt (Round {current_amp_round}) "
                    f"for [{point}] – Image: {img_to_use}\n{amp_prompt}\n"
                )

                amp_pred_new = await call_llm_once(amp_prompt, img_to_use)

                if not (
                    amp_pred_new
                    and "start_angle" in amp_pred_new
                    and "end_angle" in amp_pred_new
                ):
                    print(f"⚠️ amplifier round {current_amp_round} 预测失败 [{chart_id} – {point}]")
                    break

                amp_pred = amp_pred_new

                records.append({
                    "chart_id": chart_id,
                    "point": point,
                    "prompt_type": "amplifier",
                    "image_type": "no_grid",
                    "gt": cfg["data_points"][point],
                    "pred": amp_pred,
                    "round_index": f"amp{current_amp_round}",
                })

            # ---- 最终 amplifier pct 记录 ----
            start = float(amp_pred["start_angle"])
            end = float(amp_pred["end_angle"])
            pred_pct = ((end - start + 360) % 360) / 360
            gt_pct = cfg["data_points"][point]

            records.append({
                "chart_id": chart_id,
                "point": point,
                "prompt_type": "amplifier_pct",
                "image_type": "with_grid",
                "gt": gt_pct,
                "pred": pred_pct,
                "mae": compute_mae(pred_pct, gt_pct),
                "rel_err": compute_relative_error(pred_pct, gt_pct),
                "round": "final"
            })


    # ============================================================
    # ❗ 删除你原来重复写 amplifier_pct 的第二段（已完全整合）
    # ============================================================



# ────────────────────────────────────────────────────────────
# 5) METRICS
# ────────────────────────────────────────────────────────────

# 6) EXPERIMENT CORE
# ────────────────────────────────────────────────────────────
ENABLE_COLOR_FEEDBACK = True
MAX_COLOR_FEEDBACK_ROUNDS = 4

async def run_dataset(cfg: Dict, records: List[Dict]):
    chart_id = cfg["chart_id"]

    for prompt_type, img_type in EXPERIMENT_TYPES:
        img_path = cfg["image_paths"][img_type]

        if prompt_type == "amplifier":
            await run_dataset_segmentwise_feedback(
                cfg, records,
                repeat_rounds=REPEAT_TIMES,
                use_amplifier=True,
                amp_img_type="with_grid"  # ✅ 显式指定放大阶段用 with_grid
            )
            continue

        if prompt_type == "amplifier":
            await run_dataset_segmentwise_feedback(cfg, records, repeat_rounds=REPEAT_TIMES)
        #     continue

        elif prompt_type == "feedback":
            await run_dataset_segmentwise_feedback(cfg, records, repeat_rounds=REPEAT_TIMES, use_amplifier=False)

        elif prompt_type == "grid":
            for round_idx in range(REPEAT_TIMES):
                angle_dict = {}  # 保存每个扇区的 {"start_angle": ..., "end_angle": ...}
                pred_pcts = {}  # 保存每个扇区的占比预测结果

                for point in cfg["data_points"].keys():
                    prompt = generate_prompt(point, prompt_type)
                    print(f"\n📈 GRID Prompt for [{point}] – Image: {img_path} | Round {round_idx + 1}\n{prompt}\n")

                    pred = await call_llm_once(prompt=prompt, image_path=img_path)

                    # 检查预测格式是否为 dict 且包含 start/end
                    if (
                            pred is not None
                            and isinstance(pred, dict)
                            and "start_angle" in pred
                            and "end_angle" in pred
                    ):
                        start_angle = pred["start_angle"]
                        end_angle = pred["end_angle"]
                        angle_diff = (end_angle - start_angle + 360) % 360
                        pred_pct = angle_diff / 360

                        angle_dict[point] = pred  # 存下角度
                        pred_pcts[point] = pred_pct
                    else:
                        print(f"❌ 格式错误或缺失字段: {pred}")
                        break

                # 如果预测完整
                if len(pred_pcts) == len(cfg["data_points"]):
                    for point, pred_pct in pred_pcts.items():
                        gt_pct = cfg["data_points"][point]
                        records.append({
                            "chart_id": chart_id,
                            "point": point,
                            "prompt_type": prompt_type,
                            "image_type": img_type,
                            "gt": gt_pct,
                            "pred": pred_pct,
                            "mae": compute_mae(pred_pct, gt_pct),
                            "rel_err": compute_relative_error(pred_pct, gt_pct)
                        })
                    print(f"✅ Success (Round {round_idx + 1}): {angle_dict}")
                else:
                    print(f"❌ Failed: incomplete prediction in Round {round_idx + 1}")


        elif prompt_type == "grid_0":
            for point, gt in cfg["data_points"].items():
                valid = 0
                attempts = 0
                while valid < REPEAT_TIMES and attempts < MAX_ATTEMPTS:
                    attempts += 1
                    prompt = generate_prompt(point, prompt_type, cfg["theta_ticks"])
                    pred = await call_llm_once(prompt=prompt, image_path=img_path)
                    print(f"\n📈 grid_0 Prompt for [{point}] – Image: {img_path}\n{prompt}\n")
                    if pred is None:
                        continue
                    records.append({
                        "chart_id": chart_id,
                        "point": point,
                        "prompt_type": prompt_type,
                        "image_type": img_type,
                        "gt": gt,
                        "pred": pred,
                        "mae": compute_mae(pred, gt),
                        "rel_err": compute_relative_error(pred, gt)
                    })
                    print(f"✅ Success: attempt {attempts} ({valid + 1}/{REPEAT_TIMES})")
                    valid += 1

        elif prompt_type == "baseline":
            for point, gt in cfg["data_points"].items():
                valid = 0
                attempts = 0
                while valid < REPEAT_TIMES and attempts < MAX_ATTEMPTS:
                    attempts += 1
                    prompt = generate_prompt(point, prompt_type, cfg["theta_ticks"])
                    pred = await call_llm_once(prompt=prompt, image_path=img_path)
                    print(f"\n📈 BASELINE Prompt for [{point}] – Image: {img_path}\n{prompt}\n")

                    if pred is None:
                        continue

                    # ✅ 除以100换成小数比例
                    pred = pred / 100

                    records.append({
                        "chart_id": chart_id,
                        "point": point,
                        "prompt_type": prompt_type,
                        "image_type": img_type,
                        "gt": gt,
                        "pred": pred,
                        "mae": compute_mae(pred, gt),
                        "rel_err": compute_relative_error(pred, gt)
                    })

                    print(f"✅ Success: attempt {attempts} ({valid + 1}/{REPEAT_TIMES})")
                    valid += 1

        else:
            raise ValueError(f"❌ Unsupported prompt_type: '{prompt_type}' – please check EXPERIMENT_TYPES list.")


async def run_experiment(batch_size: int | None = None, chart_ids: List[str] | None = None):
    datasets = [d for d in DATASET_CONFIGS if (not chart_ids or d["chart_id"] in chart_ids)]
    if not datasets:
        print("❌ No donut chart configs found.")
        return
    records: List[Dict] = []

    for cfg in datasets:
        print(f"🌀 Generating angle grid overlay for {cfg['chart_id']}")
        grid_img_path = draw_angle_grid_30deg(
            cfg,
            img_type="no_grid",  # 原图来源
            output_suffix="_with_grid",  # 控制保存路径
            inner_radius=cfg.get("r_pixels")[1],
            line_color=(0, 0, 0, 255)  # 黑色辅助线
        )
        cfg["image_paths"]["with_grid"] = grid_img_path  # ⬅️ 更新路径

    async def run_batch(batch: List[Dict]):
        await asyncio.gather(*[run_dataset(ds, records) for ds in batch])

    if batch_size:
        for i in range(0, len(datasets), batch_size):
            await run_batch(datasets[i:i + batch_size])
    else:
        await run_batch(datasets)

    if not records:
        print("⚠️  No successful predictions.")
        return

    df = pd.DataFrame(records)
    for cid, g in df.groupby("chart_id"):
        out_dir = os.path.join("results_Pixtral", cid)
        os.makedirs(out_dir, exist_ok=True)
        g.to_csv(os.path.join(out_dir, "experiment_results.csv"), index=False)
        save_summary_and_plot(g, out_dir, cid)
        print(f"✅  Saved results & plots for {cid}")

# ────────────────────────────────────────────────────────────
# 7) CLI
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Donut chart batch evaluator")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--chart-ids", nargs="+", default=None)
    args = parser.parse_args()

    asyncio.run(run_experiment(
        batch_size=args.batch_size,
        chart_ids=args.chart_ids
        # chart_ids=["donut_166", "donut_170", "donut_171"]
    ))
