import json
import math
import os
import traceback

from demo_pie_circle_find_1 import PieCircleDetector


INPUT_DIR = os.path.join(
    "backend",
    "real",
    "PieChart-11 & DonutChart-14",
    "PieChart-11-final",
)
GROUND_TRUTH_DIR = os.path.join("backend", "real", "pie")
OUTPUT_DIR = os.path.join("data", "output", "pie", "all_tests")


def load_ground_truth(name):
    path = os.path.join(GROUND_TRUTH_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return {
        "center": data.get("center"),
        "radius": data.get("r_pixels"),
        "path": path,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []
    image_names = sorted(
        name
        for name in os.listdir(INPUT_DIR)
        if name.lower().endswith(".png")
    )

    for image_name in image_names:
        image_path = os.path.join(INPUT_DIR, image_name)
        name = os.path.splitext(image_name)[0]
        item = {"chart": name, "image_path": image_path}
        try:
            result = PieCircleDetector().process_single_image(image_path, OUTPUT_DIR)
            item.update({"status": "success", "result": result})
            truth = load_ground_truth(name)
            if truth is not None and truth["center"] and truth["radius"] is not None:
                center_error = math.hypot(
                    result["center"]["x"] - truth["center"][0],
                    result["center"]["y"] - truth["center"][1],
                )
                radius_error = abs(result["radius"] - truth["radius"])
                item["ground_truth"] = truth
                item["center_error"] = center_error
                item["radius_error"] = radius_error
        except Exception as error:
            item.update(
                {
                    "status": "failed",
                    "reason": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
        summary.append(item)
        print(
            f"{name}: {item['status']}, "
            f"center_error={item.get('center_error')}, "
            f"radius_error={item.get('radius_error')}"
        )

    successful = [item for item in summary if item["status"] == "success"]
    evaluated = [item for item in successful if "center_error" in item]
    aggregate = {
        "total": len(summary),
        "successful": len(successful),
        "failed": len(summary) - len(successful),
        "evaluated": len(evaluated),
        "mean_center_error": (
            sum(item["center_error"] for item in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        "max_center_error": (
            max(item["center_error"] for item in evaluated)
            if evaluated
            else None
        ),
        "mean_radius_error": (
            sum(item["radius_error"] for item in evaluated) / len(evaluated)
            if evaluated
            else None
        ),
        "max_radius_error": (
            max(item["radius_error"] for item in evaluated)
            if evaluated
            else None
        ),
    }
    export = {"aggregate": aggregate, "results": summary}
    summary_path = os.path.join(OUTPUT_DIR, "pie_circle_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(export, file, ensure_ascii=False, indent=2)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
