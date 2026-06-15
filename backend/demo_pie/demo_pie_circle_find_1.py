import argparse
import json
import math
import os

import cv2
import numpy as np


class PieCircleDetector:
    def __init__(self, detection_size=800):
        self.detection_size = detection_size

    @staticmethod
    def _circle_is_safe(circle, image_shape, margin=2):
        height, width = image_shape[:2]
        center_x, center_y, radius = circle
        return (
            radius > 0
            and center_x - radius >= margin
            and center_y - radius >= margin
            and center_x + radius < width - margin
            and center_y + radius < height - margin
        )

    @staticmethod
    def _foreground_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturated = hsv[:, :, 1] >= 35
        non_white = gray < 245
        mask = np.uint8(saturated & non_white) * 255
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _prepare_detection_image(self, image):
        height, width = image.shape[:2]
        mask = self._foreground_mask(image)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if component_count <= 1:
            return None

        component_index = max(
            range(1, component_count),
            key=lambda index: stats[index, cv2.CC_STAT_AREA],
        )
        component_mask = np.uint8(labels == component_index) * 255
        x, y, component_width, component_height, _ = [
            int(value) for value in stats[component_index]
        ]
        side = int(round(max(component_width, component_height) * 1.35))
        side = min(max(side, component_width, component_height), width, height)
        center_x = x + component_width / 2
        center_y = y + component_height / 2
        x0 = max(0, min(int(round(center_x - side / 2)), width - side))
        y0 = max(0, min(int(round(center_y - side / 2)), height - side))
        x1 = x0 + side
        y1 = y0 + side

        roi = image[y0:y1, x0:x1]
        scale = self.detection_size / side
        normalized = cv2.resize(
            roi,
            (self.detection_size, self.detection_size),
            interpolation=cv2.INTER_AREA,
        )
        return {
            "image": normalized,
            "component_mask": component_mask,
            "roi": (x0, y0, x1, y1),
            "scale": scale,
        }

    @staticmethod
    def _enhance_gray(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)
        return cv2.GaussianBlur(enhanced, (5, 5), 1.2)

    @staticmethod
    def _circle_edge_support(edges, circle):
        center_x, center_y, radius = circle
        supported = 0
        sample_count = 360
        for angle in np.linspace(0, 2 * math.pi, sample_count, endpoint=False):
            found = False
            for radius_offset in (-2, -1, 0, 1, 2):
                x = int(round(center_x + (radius + radius_offset) * math.cos(angle)))
                y = int(round(center_y + (radius + radius_offset) * math.sin(angle)))
                if (
                    0 <= x < edges.shape[1]
                    and 0 <= y < edges.shape[0]
                    and edges[y, x] > 0
                ):
                    found = True
                    break
            supported += int(found)
        return supported / sample_count

    @staticmethod
    def _map_circle_to_original(circle, transform):
        center_x, center_y, radius = [float(value) for value in circle]
        x0, y0, _, _ = transform["roi"]
        scale = transform["scale"]
        return (
            (center_x / scale) + x0,
            (center_y / scale) + y0,
            radius / scale,
        )

    def _hough_candidates(self, transform):
        normalized = transform["image"]
        enhanced = self._enhance_gray(normalized)
        edges = cv2.Canny(enhanced, 30, 100)
        candidates = []
        seen = set()
        for param2 in (50, 35, 25, 20, 15):
            circles = cv2.HoughCircles(
                enhanced,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=20,
                param1=50,
                param2=param2,
                minRadius=int(self.detection_size * 0.30),
                maxRadius=int(self.detection_size * 0.53),
            )
            if circles is None:
                continue
            for circle in np.around(circles[0]).astype(int):
                key = tuple(circle)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "normalized_circle": tuple(float(value) for value in circle),
                        "edge_support": self._circle_edge_support(edges, circle),
                        "source": "hough",
                    }
                )
        return candidates

    def _component_candidate(self, transform):
        contours, _ = cv2.findContours(
            transform["component_mask"],
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        x0, y0, _, _ = transform["roi"]
        scale = transform["scale"]
        normalized_circle = (
            (center_x - x0) * scale,
            (center_y - y0) * scale,
            radius * scale,
        )
        edges = cv2.Canny(self._enhance_gray(transform["image"]), 30, 100)
        return {
            "normalized_circle": normalized_circle,
            "edge_support": self._circle_edge_support(edges, normalized_circle),
            "source": "colored_component_fallback",
        }

    def detect(self, image):
        transform = self._prepare_detection_image(image)
        if transform is None:
            raise ValueError("Unable to isolate the main colored pie region.")

        candidates = self._hough_candidates(transform)
        component_candidate = self._component_candidate(transform)
        if component_candidate is not None:
            candidates.append(component_candidate)

        scored = []
        for candidate in candidates:
            mapped = self._map_circle_to_original(
                candidate["normalized_circle"],
                transform,
            )
            clipping_tolerance = max(3, mapped[2] * 0.03)
            if not self._circle_is_safe(
                mapped,
                image.shape,
                margin=-clipping_tolerance,
            ):
                continue
            center_x, center_y, radius = mapped
            mask = transform["component_mask"]
            sample_points = []
            for angle in np.linspace(0, 2 * math.pi, 180, endpoint=False):
                x = int(round(center_x + radius * 0.75 * math.cos(angle)))
                y = int(round(center_y + radius * 0.75 * math.sin(angle)))
                if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                    sample_points.append(mask[y, x] > 0)
            fill_support = float(np.mean(sample_points)) if sample_points else 0.0
            score = candidate["edge_support"] * 140 + fill_support * 20
            if candidate["source"] == "colored_component_fallback":
                score += 8
            scored.append(
                {
                    "circle": mapped,
                    "source": candidate["source"],
                    "edge_support": candidate["edge_support"],
                    "fill_support": fill_support,
                    "score": score,
                }
            )

        if not scored:
            raise ValueError("Unable to find a safe pie circle.")
        best = max(scored, key=lambda item: item["score"])
        best["transform"] = transform
        return best

    def process_single_image(self, image_path, output_dir):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        os.makedirs(output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(image_path))[0]
        result = self.detect(image)
        center_x, center_y, radius = result["circle"]

        marked = image.copy()
        center = (int(round(center_x)), int(round(center_y)))
        cv2.circle(marked, center, int(round(radius)), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(marked, center, 3, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(
            marked,
            f"center=({center[0]}, {center[1]}), radius={radius:.2f}",
            (5, max(18, image.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        detection_path = os.path.join(output_dir, f"{name}_pie_circle_detection.png")
        json_path = os.path.join(output_dir, f"{name}_pie_circle.json")
        normalized_path = os.path.join(output_dir, f"{name}_pie_normalized.png")
        cv2.imwrite(detection_path, marked)
        cv2.imwrite(normalized_path, result["transform"]["image"])

        export = {
            "image_path": image_path,
            "center": {
                "x": float(center_x),
                "y": float(center_y),
            },
            "center_array": [
                int(round(center_x)),
                int(round(center_y)),
            ],
            "radius": float(radius),
            "r_pixels": int(round(radius)),
            "detection_source": result["source"],
            "edge_support": float(result["edge_support"]),
            "fill_support": float(result["fill_support"]),
            "score": float(result["score"]),
            "roi": [int(value) for value in result["transform"]["roi"]],
            "detection_path": detection_path,
            "normalized_path": normalized_path,
        }
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(export, file, ensure_ascii=False, indent=2)
        return export


def main():
    parser = argparse.ArgumentParser(description="Detect a pie chart center and radius.")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=os.path.join(
            "backend",
            "real",
            "PieChart-11 & DonutChart-14",
            "PieChart-11-final",
            "PieChart2.png",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "output", "pie"),
    )
    args = parser.parse_args()
    result = PieCircleDetector().process_single_image(args.image_path, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
