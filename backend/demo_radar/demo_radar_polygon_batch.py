import json
import os
import traceback

import cv2

try:
    from .demo_radar_polygon_find_1 import RadarPolygonEncoder
except ImportError:
    from demo_radar_polygon_find_1 import RadarPolygonEncoder


POLYGON_RADAR_NUMBERS = (1, 5, 6, 8, 16, 17, 18, 23)
INPUT_DIR = os.path.join(
    "backend",
    "real",
    "RadarChart-18 & RoseChart-6",
    "RadarChart-18-final",
)
OUTPUT_DIR = os.path.join("data", "output", "radar_polygon", "all_tests")


def save_diagnostic_image(image, output_path, message, color=(0, 0, 255)):
    result = image.copy()
    font_scale = max(0.45, min(image.shape[:2]) / 900)
    thickness = max(1, int(round(font_scale * 2)))
    cv2.rectangle(result, (0, 0), (image.shape[1], 38), (255, 255, 255), -1)
    cv2.putText(
        result,
        message,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    cv2.imwrite(output_path, result)


def process_with_diagnostics(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    if image is None:
        return {
            "chart": name,
            "status": "image_read_failed",
            "reason": f"Unable to read image: {image_path}",
        }

    encoder = RadarPolygonEncoder()
    polygon = encoder.find_outer_polygon(image)
    detection_path = os.path.join(output_dir, f"{name}_polygon_detection.png")
    encode_path = os.path.join(output_dir, f"{name}_polygon_encode.png")
    json_path = os.path.join(output_dir, f"{name}_polygon_result.json")

    if polygon is None:
        reason = "No sufficiently regular outer polygon candidate."
        save_diagnostic_image(image, detection_path, reason)
        save_diagnostic_image(
            image,
            encode_path,
            "PREVIEW ONLY: polygon geometry was not detected",
            (0, 165, 255),
        )
        result = {
            "chart": name,
            "status": "geometry_failed",
            "success": False,
            "reason": reason,
            "detection_path": detection_path,
            "encode_path": encode_path,
        }
    else:
        cv2.imwrite(detection_path, encoder.visualize_detection(image, polygon))
        try:
            export = encoder.process_single_image(image_path, output_dir)
            result = {
                "chart": name,
                "status": "success",
                "success": True,
                "result": export,
            }
        except Exception as error:
            cv2.imwrite(
                encode_path,
                encoder.encrypt_polygon_grid(image, polygon, relation=None),
            )
            result = {
                "chart": name,
                "status": "partial_geometry_only",
                "success": False,
                "reason": str(error),
                "traceback": traceback.format_exc(),
                "center": [float(value) for value in polygon["center"]],
                "sides": int(polygon["sides"]),
                "detection_source": polygon["detection_source"],
                "axis_matches": polygon.get("axis_matches"),
                "axis_error_degrees": polygon.get("axis_error_degrees"),
                "outer_edge_support": polygon.get("outer_edge_support"),
                "corrected_vertex_indices": [
                    int(value) for value in polygon["corrected_vertex_indices"]
                ],
                "detection_path": detection_path,
                "encode_path": encode_path,
                "encode_note": (
                    "Preview only: uses equal subdivision because a reliable "
                    "tick-radius relation was not available."
                ),
            }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []
    for number in POLYGON_RADAR_NUMBERS:
        image_path = os.path.join(INPUT_DIR, f"RadarChart{number}.png")
        result = process_with_diagnostics(image_path, OUTPUT_DIR)
        summary.append(result)
        print(f"{result['chart']}: {result['status']}")

    summary_path = os.path.join(OUTPUT_DIR, "polygon_all_tests_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
