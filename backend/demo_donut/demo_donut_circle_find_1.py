import argparse
import json
import math
import os

import cv2
import numpy as np


class DonutCircleDetector:
    def __init__(self, sample_count=360):
        self.sample_count = sample_count

    @staticmethod
    def _colored_mask(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.uint8((hsv[:, :, 1] >= 20) & (gray < 250)) * 255
        kernel_size = max(3, int(round(min(image.shape[:2]) * 0.012)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), np.uint8),
        )

    @staticmethod
    def _white_hole_candidates(image):
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white = np.uint8(gray >= 248) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
        candidates = []
        for index in range(1, count):
            x, y, component_width, component_height, area = [
                int(value) for value in stats[index]
            ]
            if (
                x == 0
                or y == 0
                or x + component_width >= width
                or y + component_height >= height
                or area < min(height, width) ** 2 * 0.002
            ):
                continue
            mask = np.uint8(labels == index) * 255
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(
                max(contours, key=cv2.contourArea)
            )
            aspect_ratio = component_width / max(1, component_height)
            if 0.65 <= aspect_ratio <= 1.35:
                candidates.append(
                    {
                        "center": np.array([center_x, center_y], dtype=np.float64),
                        "radius_hint": float(radius),
                        "source": "white_hole",
                        "area": area,
                    }
                )
        return candidates

    @staticmethod
    def _colored_contour_candidates(mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            candidates.append(
                {
                    "center": np.array([center_x, center_y], dtype=np.float64),
                    "radius_hint": float(radius),
                    "source": "colored_contour",
                    "area": float(area),
                }
            )
        return candidates

    @staticmethod
    def _hough_center_candidates(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)
        enhanced = cv2.GaussianBlur(enhanced, (5, 5), 1.2)
        short_side = min(image.shape[:2])
        candidates = []
        for param2 in (50, 35, 25, 20):
            circles = cv2.HoughCircles(
                enhanced,
                cv2.HOUGH_GRADIENT,
                dp=1.0,
                minDist=max(8, int(short_side * 0.04)),
                param1=50,
                param2=param2,
                minRadius=max(3, int(short_side * 0.05)),
                maxRadius=max(5, int(short_side * 0.48)),
            )
            if circles is None:
                continue
            for center_x, center_y, radius in circles[0]:
                candidates.append(
                    {
                        "center": np.array([center_x, center_y], dtype=np.float64),
                        "radius_hint": float(radius),
                        "source": "hough",
                        "area": 0.0,
                    }
                )
        return candidates

    def _circle_occupancy(self, mask, center, radius):
        height, width = mask.shape[:2]
        values = []
        for angle in np.linspace(
            0,
            2 * math.pi,
            self.sample_count,
            endpoint=False,
        ):
            x = int(round(center[0] + radius * math.cos(angle)))
            y = int(round(center[1] + radius * math.sin(angle)))
            values.append(
                mask[y, x] > 0
                if 0 <= x < width and 0 <= y < height
                else False
            )
        return float(np.mean(values))

    def _scan_radial_structure(self, mask, center):
        height, width = mask.shape[:2]
        max_radius = int(
            max(
                math.hypot(center[0], center[1]),
                math.hypot(width - center[0], center[1]),
                math.hypot(center[0], height - center[1]),
                math.hypot(width - center[0], height - center[1]),
            )
        )
        scans = []
        for radius in range(3, max_radius - 4):
            inner_occupancy = self._circle_occupancy(mask, center, radius - 3)
            outer_occupancy = self._circle_occupancy(mask, center, radius + 3)
            scans.append(
                {
                    "radius": radius,
                    "inside": inner_occupancy,
                    "outside": outer_occupancy,
                    "inner_transition": outer_occupancy - inner_occupancy,
                    "outer_transition": inner_occupancy - outer_occupancy,
                }
            )

        inner_candidates = [
            item for item in scans
            if item["inner_transition"] >= 0.35 and item["outside"] >= 0.45
        ]
        if not inner_candidates:
            return None
        maximum_inner_transition = max(
            item["inner_transition"] for item in inner_candidates
        )
        strongest_inner = [
            item for item in inner_candidates
            if item["inner_transition"] >= maximum_inner_transition - 0.02
        ]
        inner = strongest_inner[len(strongest_inner) // 2]
        outer_candidates = [
            item for item in scans
            if (
                item["radius"] >= inner["radius"] * 1.15
                and item["outer_transition"] >= 0.35
                and item["inside"] >= 0.45
            )
        ]
        if not outer_candidates:
            return None
        maximum_outer_transition = max(
            item["outer_transition"] for item in outer_candidates
        )
        strongest_outer = [
            item for item in outer_candidates
            if item["outer_transition"] >= maximum_outer_transition - 0.02
        ]
        outer = strongest_outer[len(strongest_outer) // 2]
        return inner, outer

    def _outer_radius_consistency(self, mask, center, inner_radius, outer_radius):
        height, width = mask.shape[:2]
        matching_angles = 0
        valid_angles = 0
        for angle in np.linspace(
            0,
            2 * math.pi,
            self.sample_count,
            endpoint=False,
        ):
            colored_radii = []
            for radius in range(
                max(1, int(inner_radius * 0.75)),
                max(2, int(outer_radius * 1.35)),
            ):
                x = int(round(center[0] + radius * math.cos(angle)))
                y = int(round(center[1] + radius * math.sin(angle)))
                if (
                    0 <= x < width
                    and 0 <= y < height
                    and mask[y, x] > 0
                ):
                    colored_radii.append(radius)
            if not colored_radii:
                continue
            valid_angles += 1
            if abs(max(colored_radii) - outer_radius) <= outer_radius * 0.08:
                matching_angles += 1
        return matching_angles / max(1, valid_angles)

    def detect(self, image):
        mask = self._colored_mask(image)
        candidates = (
            self._white_hole_candidates(image)
            + self._colored_contour_candidates(mask)
        )

        unique_candidates = []
        for candidate in candidates:
            if any(
                np.linalg.norm(candidate["center"] - existing["center"]) < 2
                for existing in unique_candidates
            ):
                continue
            unique_candidates.append(candidate)

        results = []
        for candidate in unique_candidates:
            structure = self._scan_radial_structure(mask, candidate["center"])
            if structure is None:
                continue
            inner, outer = structure
            center = candidate["center"]
            outer_consistency = self._outer_radius_consistency(
                mask,
                center,
                inner["radius"],
                outer["radius"],
            )
            center_offset_penalty = 0.0
            if candidate["source"] == "white_hole":
                center_offset_penalty = abs(
                    candidate["radius_hint"] - inner["radius"]
                ) * 0.2
            score = (
                inner["inner_transition"] * 100
                + outer["outer_transition"] * 100
                + outer_consistency * 20
                - center_offset_penalty
            )
            results.append(
                {
                    "center": center,
                    "inner_radius": float(inner["radius"]),
                    "outer_radius": float(outer["radius"]),
                    "inner_transition": float(inner["inner_transition"]),
                    "outer_transition": float(outer["outer_transition"]),
                    "outer_radius_consistency": float(outer_consistency),
                    "source": candidate["source"],
                    "score": float(score),
                }
            )

        if not results:
            raise ValueError("Unable to find reliable concentric donut boundaries.")

        best = max(results, key=lambda item: item["score"])
        if (
            best["inner_transition"] < 0.65
            or best["outer_transition"] < 0.65
            or best["outer_radius_consistency"] < 0.85
        ):
            raise ValueError(
                "Outer ring appears exploded or does not form a reliable common circle."
            )
        return best

    def process_single_image(self, image_path, output_dir):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        os.makedirs(output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(image_path))[0]
        result = self.detect(image)

        center = tuple(np.round(result["center"]).astype(int))
        inner_radius = int(round(result["inner_radius"]))
        outer_radius = int(round(result["outer_radius"]))
        marked = image.copy()
        cv2.circle(marked, center, outer_radius, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(marked, center, inner_radius, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(marked, center, 3, (255, 0, 0), -1, cv2.LINE_AA)

        detection_path = os.path.join(
            output_dir,
            f"{name}_donut_circle_detection.png",
        )
        json_path = os.path.join(output_dir, f"{name}_donut_circle.json")
        cv2.imwrite(detection_path, marked)
        export = {
            "image_path": image_path,
            "center": [float(value) for value in result["center"]],
            "center_array": [int(value) for value in center],
            "inner_radius": result["inner_radius"],
            "outer_radius": result["outer_radius"],
            "r_pixels": [inner_radius, outer_radius],
            "detection_source": result["source"],
            "inner_transition": result["inner_transition"],
            "outer_transition": result["outer_transition"],
            "outer_radius_consistency": result["outer_radius_consistency"],
            "score": result["score"],
            "detection_path": detection_path,
        }
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(export, file, ensure_ascii=False, indent=2)
        return export


def main():
    parser = argparse.ArgumentParser(
        description="Detect concentric inner and outer radii of a doughnut chart.",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=os.path.join(
            "backend",
            "real",
            "PieChart-11 & DonutChart-14",
            "DonutChart-14-final",
            "DonutChart2.png",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("data", "output", "donut"),
    )
    args = parser.parse_args()
    result = DonutCircleDetector().process_single_image(args.image_path, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
