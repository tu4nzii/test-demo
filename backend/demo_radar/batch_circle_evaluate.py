"""
批量霍夫圆检测评估 (排除多边形图表)
======================================
排除: RadarChart 1, 5, 6, 8, 16, 17, 18, 23
评估指标: 圆心误差, 半径误差, 检测成功率
"""

import sys, json, math
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # test-demo root
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
circle_path = Path(__file__).resolve().parent / "demo_radar_circle_find_1.py"
spec = importlib.util.spec_from_file_location("circle_detect", circle_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

RADAR_DIR = PROJECT_ROOT / "backend" / "real" / "RadarChart-18 & RoseChart-6" / "RadarChart-18-final"
EXCLUDE_NUMBERS = {1, 5, 6, 8, 16, 17, 18, 23}


def main():
    # 收集所有雷达图
    all_pngs = sorted(RADAR_DIR.glob("RadarChart*.png"))
    pngs = []
    for p in all_pngs:
        num_str = p.stem.replace("RadarChart", "")
        try:
            num = int(num_str)
        except ValueError:
            continue
        if num not in EXCLUDE_NUMBERS:
            pngs.append(p)

    print(f"评估集: {len(pngs)} 张 (排除 {EXCLUDE_NUMBERS})")
    print("=" * 70)

    results = []
    for png in pngs:
        name = png.name
        json_path = png.with_suffix(".json")

        # 读取 ground truth
        if not json_path.exists():
            print(f"  {name}: 无 JSON, 跳过")
            continue
        with open(json_path) as f:
            gt = json.load(f)

        gt_center = np.array(gt["center"], dtype=float)
        gt_r_pixels = sorted(gt.get("r_pixels", []))

        # 霍夫圆检测
        encoder = mod.RadarChartEncoder()
        ring_mask = encoder.visualize_ring_mask(str(png))
        encoder.second_circle_find(ring_mask)

        if encoder.first_r <= 0:
            results.append({
                "name": name, "detected": False,
                "center_error": None, "r1_error": None, "r2_error": None,
                "source": encoder.detection_source,
            })
            print(f"  {name:25s} | NOT FOUND (source={encoder.detection_source})")
            continue

        # 圆心误差
        pred_center = np.array(encoder.coords, dtype=float)
        center_err = float(np.linalg.norm(pred_center - gt_center))

        # 半径误差: 找最接近 ground truth 的匹配
        if gt_r_pixels:
            r1_err = min(abs(encoder.first_r - r) for r in gt_r_pixels)
            if encoder.second_r > 0:
                r2_err = min(abs(encoder.second_r - r) for r in gt_r_pixels)
            else:
                r2_err = None
        else:
            r1_err = None
            r2_err = None

        results.append({
            "name": name, "detected": True,
            "center_error": center_err,
            "r1_error": r1_err, "r2_error": r2_err,
            "pred_center": encoder.coords,
            "pred_r1": encoder.first_r, "pred_r2": encoder.second_r,
            "gt_center": gt_center.tolist(),
            "gt_r_pixels": gt_r_pixels,
            "source": encoder.detection_source,
            "edge_support": round(encoder.last_edge_support, 3),
        })

        r2_str = f"r2_err={r2_err:5.1f}" if r2_err is not None else "r2=N/A"
        print(f"  {name:25s} | center_err={center_err:5.1f}px  "
              f"r1_err={r1_err:5.1f}px  {r2_str}  "
              f"edge={encoder.last_edge_support:.3f}  src={encoder.detection_source}")

    # ── 汇总统计 ──
    detected = [r for r in results if r["detected"]]
    not_found = [r for r in results if not r["detected"]]

    center_errs = [r["center_error"] for r in detected if r["center_error"] is not None]
    r1_errs = [r["r1_error"] for r in detected if r["r1_error"] is not None]
    r2_errs = [r["r2_error"] for r in detected if r["r2_error"] is not None]

    print(f"\n{'='*70}")
    print(f"评估汇总 ({len(results)} 张)")
    print(f"{'='*70}")
    print(f"  检测成功:  {len(detected)}/{len(results)} ({100*len(detected)/len(results):.1f}%)")
    print(f"  未检测到:  {len(not_found)}/{len(results)}")
    if center_errs:
        print(f"  圆心误差:  均值={np.mean(center_errs):.1f}px  "
              f"中位数={np.median(center_errs):.1f}px  "
              f"最大={np.max(center_errs):.1f}px")
    if r1_errs:
        print(f"  r1 半径误差: 均值={np.mean(r1_errs):.1f}px  "
              f"中位数={np.median(r1_errs):.1f}px  "
              f"最大={np.max(r1_errs):.1f}px")
    if r2_errs:
        print(f"  r2 半径误差: 均值={np.mean(r2_errs):.1f}px  "
              f"中位数={np.median(r2_errs):.1f}px  "
              f"最大={np.max(r2_errs):.1f}px")

    if not_found:
        print(f"\n  未检测到的图表:")
        for r in not_found:
            print(f"    {r['name']}: source={r['source']}")

    # 保存详细结果
    out_path = PROJECT_ROOT / "data" / "output" / "radar" / "hough_circle_eval_exclude_polygon.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "exclude": sorted(EXCLUDE_NUMBERS),
            "total": len(results),
            "detected": len(detected),
            "detection_rate": round(100*len(detected)/len(results), 1),
            "center_error_mean": round(float(np.mean(center_errs)), 1) if center_errs else None,
            "center_error_median": round(float(np.median(center_errs)), 1) if center_errs else None,
            "r1_error_mean": round(float(np.mean(r1_errs)), 1) if r1_errs else None,
            "r1_error_median": round(float(np.median(r1_errs)), 1) if r1_errs else None,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果: {out_path}")


if __name__ == "__main__":
    main()
