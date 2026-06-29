import json
import math
import os
import traceback

from demo_donut_circle_find_1 import DonutCircleDetector


INPUT_DIR = os.path.join(
    "backend",
    "real",
    "PieChart-11 & DonutChart-14",
    "DonutChart-14-final",
)
GROUND_TRUTH_DIR = os.path.join("backend", "real", "donut")
OUTPUT_DIR = os.path.join("data", "output", "donut", "all_tests")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []
    for image_name in sorted(
        name for name in os.listdir(INPUT_DIR) if name.lower().endswith(".png")
    ):
        name = os.path.splitext(image_name)[0]
        image_path = os.path.join(INPUT_DIR, image_name)
        truth_path = os.path.join(GROUND_TRUTH_DIR, f"{name}.json")
        item = {"chart": name, "image_path": image_path}
        try:
            result = DonutCircleDetector().process_single_image(image_path, OUTPUT_DIR)
            item.update({"status": "success", "result": result})
            if os.path.exists(truth_path):
                with open(truth_path, encoding="utf-8") as file:
                    truth = json.load(file)
                item["ground_truth"] = {
                    "center": truth["center"],
                    "r_pixels": truth["r_pixels"],
                }
                item["center_error"] = math.hypot(
                    result["center"][0] - truth["center"][0],
                    result["center"][1] - truth["center"][1],
                )
                item["inner_radius_error"] = abs(
                    result["inner_radius"] - truth["r_pixels"][0]
                )
                item["outer_radius_error"] = abs(
                    result["outer_radius"] - truth["r_pixels"][1]
                )
        except Exception as error:
            for suffix in (
                "_donut_circle.json",
                "_donut_circle_detection.png",
            ):
                stale_path = os.path.join(OUTPUT_DIR, f"{name}{suffix}")
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            item.update(
                {
                    "status": "skipped",
                    "reason": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        summary.append(item)
        print(
            f"{name}: {item['status']}, "
            f"center_error={item.get('center_error')}, "
            f"inner_error={item.get('inner_radius_error')}, "
            f"outer_error={item.get('outer_radius_error')}"
        )

    successful = [item for item in summary if item["status"] == "success"]
    aggregate = {
        "total": len(summary),
        "successful": len(successful),
        "skipped": len(summary) - len(successful),
        "mean_center_error": (
            sum(item["center_error"] for item in successful) / len(successful)
            if successful
            else None
        ),
        "max_center_error": (
            max(item["center_error"] for item in successful)
            if successful
            else None
        ),
        "mean_inner_radius_error": (
            sum(item["inner_radius_error"] for item in successful) / len(successful)
            if successful
            else None
        ),
        "max_inner_radius_error": (
            max(item["inner_radius_error"] for item in successful)
            if successful
            else None
        ),
        "mean_outer_radius_error": (
            sum(item["outer_radius_error"] for item in successful) / len(successful)
            if successful
            else None
        ),
        "max_outer_radius_error": (
            max(item["outer_radius_error"] for item in successful)
            if successful
            else None
        ),
    }
    export = {"aggregate": aggregate, "results": summary}
    summary_path = os.path.join(OUTPUT_DIR, "donut_circle_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(export, file, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
