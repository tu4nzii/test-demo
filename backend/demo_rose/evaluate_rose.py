"""
Evaluate rose chart axis label detection — real & synthetic.

Usage:
  python evaluate_rose.py --dataset all
  python evaluate_rose.py --dataset synthetic --limit 10
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.demo_rose.detect_rose_axes import detect_rose
from backend.demo_radar.detect_radar_axes import evaluate, init_ocr

REAL_ROSE_DIR = (
    Path(__file__).resolve().parent.parent
    / "real" / "RadarChart-18 & RoseChart-6" / "RoseChart-6"
)
SYNTH_ROSE_DIR = Path(__file__).resolve().parent.parent / "real" / "rose"


def parse_json_field(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def read_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8-sig")
    idx = raw.find("{")
    if idx > 0:
        raw = raw[idx:]
    return json.loads(raw)


def evaluate_chart(json_path: Path, reader=None, use_llm: bool = True) -> dict:
    gt = read_json(json_path)
    name = json_path.stem

    center_raw = gt.get("center")
    if isinstance(center_raw, dict):
        center = (float(center_raw["x"]), float(center_raw["y"]))
    elif isinstance(center_raw, (list, tuple)):
        center = tuple(map(float, center_raw))
    else:
        return {"name": name, "skipped": True, "reason": "no center"}

    r_pixels = [float(v) for v in parse_json_field(gt.get("r_pixels", []))]
    if not r_pixels:
        return {"name": name, "skipped": True, "reason": "no r_pixels"}

    gt_labels = [str(v) for v in parse_json_field(gt.get("theta_ticks", gt.get("labels", [])))]
    gt_angles = [float(v) for v in parse_json_field(gt.get("theta_angles", []))]
    if not gt_angles:
        return {"name": name, "skipped": True, "reason": "no angles"}

    # Synthetic charts use math convention (0°=right, CCW).  Convert to
    # code convention (0°=top, CW) for evaluation: angle_code = (90 - angle_math) % 360.
    # Detect by checking if the first label is near 0° (math convention starts at right).
    if gt_angles and gt_angles[0] < 5:
        gt_angles = [(90.0 - a) % 360 for a in gt_angles]

    png = json_path.with_suffix(".png")
    if not png.exists():
        return {"name": name, "skipped": True, "reason": "no image"}

    axis_labels, debug = detect_rose(str(png), center, float(r_pixels[-1]), use_llm=use_llm, reader=reader)

    if debug.get("fallback") or not axis_labels:
        return {
            "name": name, "correct": 0, "total": len(gt_angles),
            "accuracy": 0.0, "fallback": True,
            "fallback_reason": debug.get("fallback_reason", "no_labels"),
            "mismatches": [],
        }

    correct, _, details = evaluate(axis_labels, gt_labels, gt_angles)
    pct = round(100 * correct / max(len(gt_angles), 1), 1)

    mismatches = [
        {"angle": a, "gt": g, "detected": d, "status": s}
        for a, g, d, s in details
        if s not in ("exact", "normalized", "fuzzy", "fuzzy_ce", "fuzzy_sub")
    ]

    return {
        "name": name, "correct": correct, "total": len(gt_angles),
        "accuracy": pct, "fallback": False, "fallback_reason": "",
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["real", "synthetic", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", type=str)
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM refinement")
    args = parser.parse_args()
    use_llm = not args.no_llm

    selected = []
    if args.dataset in ("real", "all"):
        selected.append(("real", REAL_ROSE_DIR))
    if args.dataset in ("synthetic", "all"):
        selected.append(("synthetic", SYNTH_ROSE_DIR))

    all_results = []
    reader = init_ocr()  # load once, reuse across all charts
    for dataset, directory in selected:
        jsons = sorted(directory.glob("*.json"))
        if args.only:
            needle = args.only.lower()
            jsons = [j for j in jsons if needle in j.stem.lower()]
        if args.limit:
            jsons = jsons[:args.limit]

        print(f"[{dataset}] {len(jsons)} charts")
        for jp in jsons:
            r = evaluate_chart(jp, reader=reader, use_llm=use_llm)
            r["dataset"] = dataset
            all_results.append(r)
            if r.get("skipped"):
                print(f"  {r['name']:30s} SKIP")
            elif r["fallback"]:
                print(f"  {r['name']:30s} FALLBACK ({r['fallback_reason'][:50]})")
            else:
                print(f"  {r['name']:30s} {r['correct']:2d}/{r['total']:2d} ({r['accuracy']:.0f}%)")

    evaluated = [r for r in all_results if not r.get("skipped") and not r["fallback"]]
    fallback = [r for r in all_results if r.get("fallback")]
    total_c = sum(r["correct"] for r in evaluated)
    total_t = sum(r["total"] for r in evaluated)

    print(f"\nEvaluated: {len(evaluated)}, Fallback: {len(fallback)}")
    if evaluated:
        print(f"Accuracy: {total_c}/{total_t} ({100*total_c/max(total_t,1):.1f}%)")


if __name__ == "__main__":
    main()
