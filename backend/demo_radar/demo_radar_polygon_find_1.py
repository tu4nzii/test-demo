import json
import math
import os

import cv2
import numpy as np

try:
    from .demo_radar_circle_find_1 import RadarChartEncoder
except ImportError:
    from demo_radar_circle_find_1 import RadarChartEncoder


class RadarPolygonEncoder:
    def __init__(self, base_level_count=6, density=2):
        self.base_level_count = base_level_count
        self.density = density

    def _polygon_center(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        initial_center = vertices.mean(axis=0)
        angles = np.arctan2(
            vertices[:, 1] - initial_center[1],
            vertices[:, 0] - initial_center[0],
        )
        ordered = vertices[np.argsort(angles)]

        if len(ordered) % 2 == 0:
            half = len(ordered) // 2
            opposite_midpoints = (ordered[:half] + ordered[half:]) / 2.0
            center = np.median(opposite_midpoints, axis=0)
        else:
            center = initial_center
        return center, ordered

    def _regular_center_score(self, vertices, center):
        vectors = vertices - center
        radii = np.linalg.norm(vectors, axis=1)
        median_radius = float(np.median(radii))
        if median_radius <= 0:
            return float("inf")

        angles = np.sort(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi))
        angle_gaps = np.diff(np.append(angles, angles[0] + 2 * np.pi))
        expected_gap = 2 * np.pi / len(vertices)
        angle_errors = np.abs(angle_gaps - expected_gap) / expected_gap
        radius_errors = np.abs(radii - median_radius) / median_radius

        # One displaced vertex affects two neighboring angle gaps and one radius.
        # Trim those largest errors so the remaining regular vertices determine
        # the center instead of allowing one bad contour point to pull it away.
        kept_angle_errors = np.sort(angle_errors)[: max(1, len(vertices) - 2)]
        kept_radius_errors = np.sort(radius_errors)[: max(1, len(vertices) - 1)]
        return float(
            np.mean(kept_angle_errors ** 2)
            + np.mean(kept_radius_errors ** 2)
        )

    def _refine_regular_center(self, vertices, initial_center):
        vertices = np.asarray(vertices, dtype=np.float64)
        best_center = np.asarray(initial_center, dtype=np.float64)
        best_score = self._regular_center_score(vertices, best_center)
        if best_score < 1e-4:
            return best_center, best_score

        radius = float(np.median(np.linalg.norm(vertices - best_center, axis=1)))
        search_radius = min(12.0, max(3.0, radius * 0.04))

        stages = [
            (search_radius, 1.0),
            (1.0, 0.25),
            (0.25, 0.05),
        ]
        for span, step in stages:
            stage_origin = best_center.copy()
            offsets = np.arange(-span, span + step * 0.5, step)
            for x_offset in offsets:
                for y_offset in offsets:
                    candidate = stage_origin + (x_offset, y_offset)
                    score = self._regular_center_score(vertices, candidate)
                    if score < best_score:
                        best_score = score
                        best_center = candidate
        return best_center, best_score

    def _regularize_vertices(self, vertices, center, max_angle_error_degrees=1.0):
        vertices = np.asarray(vertices, dtype=np.float64)
        center = np.asarray(center, dtype=np.float64)
        side_count = len(vertices)
        angle_step = 2 * np.pi / side_count

        vectors = vertices - center
        detected_angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
        phase_samples = detected_angles - np.arange(side_count) * angle_step
        phase = float(np.median(phase_samples))
        radius = float(np.median(np.linalg.norm(vectors, axis=1)))

        regular_angles = phase + np.arange(side_count) * angle_step
        regular_vertices = center + radius * np.column_stack(
            (np.cos(regular_angles), np.sin(regular_angles))
        )

        detected_gaps = np.diff(
            np.append(detected_angles, detected_angles[0] + 2 * np.pi)
        )
        gap_errors_degrees = np.abs(
            np.degrees(detected_gaps - angle_step)
        )
        vertex_angle_errors_degrees = np.abs(
            np.degrees(
                np.arctan2(
                    np.sin(detected_angles - regular_angles),
                    np.cos(detected_angles - regular_angles),
                )
            )
        )
        correction_distances = np.linalg.norm(
            vertices - regular_vertices,
            axis=1,
        )
        corrected_indices = np.flatnonzero(
            (vertex_angle_errors_degrees > max_angle_error_degrees)
            | (correction_distances > max(2.0, radius * 0.02))
        )
        return {
            "vertices": regular_vertices,
            "radius": radius,
            "phase_degrees": float(np.degrees(phase) % 360),
            "detected_gap_errors_degrees": gap_errors_degrees,
            "vertex_angle_errors_degrees": vertex_angle_errors_degrees,
            "correction_distances": correction_distances,
            "corrected_indices": corrected_indices,
            "max_final_angle_error_degrees": 0.0,
        }

    def _candidate_score(self, contour, vertices, image_area):
        if not cv2.isContourConvex(vertices.astype(np.int32)):
            return None

        center, ordered = self._polygon_center(vertices)
        radii = np.linalg.norm(ordered - center, axis=1)
        edges = np.linalg.norm(ordered - np.roll(ordered, -1, axis=0), axis=1)
        if np.mean(radii) <= 0 or np.mean(edges) <= 0:
            return None

        radius_cv = float(np.std(radii) / np.mean(radii))
        edge_cv = float(np.std(edges) / np.mean(edges))
        area_ratio = cv2.contourArea(contour) / image_area
        if radius_cv > 0.12 or edge_cv > 0.18:
            return None

        score = area_ratio * 100.0 - radius_cv * 120.0 - edge_cv * 80.0
        return score, center, ordered, radii

    @staticmethod
    def _cross_2d(first, second):
        return first[0] * second[1] - first[1] * second[0]

    def _find_radial_axis_polygon(self, image):
        height, width = image.shape[:2]
        minimum_size = min(height, width)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8),
        ).apply(gray)
        edges = cv2.Canny(enhanced, 30, 100)
        raw_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 360,
            threshold=25,
            minLineLength=int(minimum_size * 0.10),
            maxLineGap=int(minimum_size * 0.05),
        )
        if raw_lines is None:
            return None

        lines = []
        for raw_line in raw_lines[:, 0]:
            start = raw_line[:2].astype(np.float64)
            end = raw_line[2:].astype(np.float64)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length >= minimum_size * 0.10:
                lines.append((start, end, vector, length))

        intersection_bins = {}
        for index, (start, _, vector, _) in enumerate(lines):
            for other_start, _, other_vector, _ in lines[index + 1:]:
                denominator = self._cross_2d(vector, other_vector)
                if abs(denominator) < 1e-6:
                    continue
                amount = self._cross_2d(
                    other_start - start,
                    other_vector,
                ) / denominator
                intersection = start + amount * vector
                if not (
                    width * 0.10 < intersection[0] < width * 0.90
                    and height * 0.10 < intersection[1] < height * 0.90
                ):
                    continue
                key = tuple(np.round(intersection / 6).astype(int))
                intersection_bins.setdefault(key, []).append(intersection)

        center_candidates = [
            np.median(points, axis=0)
            for points in sorted(
                intersection_bins.values(),
                key=len,
                reverse=True,
            )[:100]
        ]
        hypotheses = []
        for center in center_candidates:
            directions = []
            for start, end, vector, length in lines:
                distance = abs(self._cross_2d(vector, center - start)) / length
                projection = np.dot(center - start, vector) / (length * length)
                if distance > 4 or not -0.18 <= projection <= 1.18:
                    continue
                farther = (
                    start
                    if np.linalg.norm(start - center) > np.linalg.norm(end - center)
                    else end
                )
                if np.linalg.norm(farther - center) >= minimum_size * 0.08:
                    directions.append(
                        math.atan2(
                            farther[1] - center[1],
                            farther[0] - center[0],
                        ) % (2 * np.pi)
                    )

            for side_count in range(5, 21):
                step = 2 * np.pi / side_count
                for phase in np.arange(0, step, np.deg2rad(2)):
                    errors = []
                    for target_index in range(side_count):
                        target = phase + target_index * step
                        error = min(
                            (
                                abs(
                                    (direction - target + np.pi)
                                    % (2 * np.pi)
                                    - np.pi
                                )
                                for direction in directions
                            ),
                            default=np.pi,
                        )
                        if error <= np.deg2rad(5):
                            errors.append(error)
                    if len(errors) < max(5, math.ceil(side_count * 0.75)):
                        continue
                    hypotheses.append(
                        {
                            "center": center,
                            "sides": side_count,
                            "phase": phase,
                            "axis_matches": len(errors),
                            "axis_error_degrees": float(
                                np.degrees(np.mean(errors))
                            ),
                        }
                    )

        best = None
        for hypothesis in sorted(
            hypotheses,
            key=lambda item: (
                item["axis_matches"] / item["sides"] * 10
                + item["axis_matches"] * 0.2
                - item["axis_error_degrees"] * 0.2
            ),
            reverse=True,
        )[:40]:
            center = hypothesis["center"]
            angles = (
                hypothesis["phase"]
                + np.arange(hypothesis["sides"])
                * (2 * np.pi / hypothesis["sides"])
            )
            unit_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
            radius_samples = []
            for radius in np.arange(
                minimum_size * 0.12,
                minimum_size * 0.48,
                4.0,
            ):
                vertices = center + radius * unit_vertices
                if (
                    np.min(vertices[:, 0]) < 1
                    or np.max(vertices[:, 0]) >= width - 1
                    or np.min(vertices[:, 1]) < 1
                    or np.max(vertices[:, 1]) >= height - 1
                ):
                    continue
                support = self.polygon_edge_support(edges, vertices)
                radius_samples.append((support, radius))
            if not radius_samples:
                continue
            maximum_support = max(item[0] for item in radius_samples)
            reliable_samples = [
                item
                for item in radius_samples
                if item[0] >= max(0.12, maximum_support * 0.35)
            ]
            support, radius = max(reliable_samples, key=lambda item: item[1])
            score = (
                support * 100
                + hypothesis["axis_matches"] / hypothesis["sides"] * 60
                + hypothesis["axis_matches"] * 5
                - hypothesis["sides"] * 1.5
                - hypothesis["axis_error_degrees"] * 2
            )
            if support < 0.12 or (best is not None and score <= best["score"]):
                continue
            vertices = center + radius * unit_vertices
            best = {
                "score": score,
                "center": center,
                "initial_center": center.copy(),
                "center_refinement_shift": 0.0,
                "regular_center_score": 0.0,
                "vertices": vertices,
                "detected_vertices": vertices.copy(),
                "radius": float(radius),
                "radius_std": 0.0,
                "sides": hypothesis["sides"],
                "regular_phase_degrees": float(
                    np.degrees(hypothesis["phase"]) % 360
                ),
                "detected_gap_errors_degrees": np.zeros(hypothesis["sides"]),
                "vertex_angle_errors_degrees": np.zeros(hypothesis["sides"]),
                "vertex_correction_distances": np.zeros(hypothesis["sides"]),
                "corrected_vertex_indices": np.array([], dtype=int),
                "max_final_angle_error_degrees": 0.0,
                "detection_source": "radial_axes_fallback",
                "axis_matches": hypothesis["axis_matches"],
                "axis_error_degrees": hypothesis["axis_error_degrees"],
                "outer_edge_support": float(support),
            }
        return best

    def find_outer_polygon(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_area = image.shape[0] * image.shape[1]
        candidates = []

        for threshold in (180, 200, 220, 235, 245):
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if area < image_area * 0.05 or perimeter <= 0:
                    continue
                for epsilon_ratio in (0.003, 0.005, 0.008, 0.01, 0.015, 0.02):
                    polygon = cv2.approxPolyDP(
                        contour,
                        epsilon_ratio * perimeter,
                        True,
                    ).reshape(-1, 2)
                    if not 5 <= len(polygon) <= 24:
                        continue
                    result = self._candidate_score(contour, polygon, image_area)
                    if result is None:
                        continue
                    score, center, ordered, radii = result
                    candidates.append(
                        {
                            "score": score,
                            "center": center,
                            "vertices": ordered,
                            "radius": float(np.median(radii)),
                            "radius_std": float(np.std(radii)),
                            "sides": len(ordered),
                        }
                    )
                    break

        if not candidates:
            return self._find_radial_axis_polygon(image)
        best = max(candidates, key=lambda item: item["score"])
        initial_center = best["center"].copy()
        refined_center, regular_center_score = self._refine_regular_center(
            best["vertices"],
            initial_center,
        )
        detected_vertices = best["vertices"].copy()
        regularized = self._regularize_vertices(
            detected_vertices,
            refined_center,
        )
        best.update(
            {
                "initial_center": initial_center,
                "center": refined_center,
                "center_refinement_shift": float(
                    np.linalg.norm(refined_center - initial_center)
                ),
                "regular_center_score": regular_center_score,
                "detected_vertices": detected_vertices,
                "vertices": regularized["vertices"],
                "radius": regularized["radius"],
                "radius_std": 0.0,
                "regular_phase_degrees": regularized["phase_degrees"],
                "detected_gap_errors_degrees": regularized[
                    "detected_gap_errors_degrees"
                ],
                "vertex_angle_errors_degrees": regularized[
                    "vertex_angle_errors_degrees"
                ],
                "vertex_correction_distances": regularized[
                    "correction_distances"
                ],
                "corrected_vertex_indices": regularized["corrected_indices"],
                "max_final_angle_error_degrees": regularized[
                    "max_final_angle_error_degrees"
                ],
                "detection_source": "closed_contour",
            }
        )
        return best

    def scaled_vertices(self, vertices, center, scale):
        return center + (vertices - center) * scale

    def polygon_edge_support(self, edges, vertices):
        supported = 0
        sample_count = 0
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            length = float(np.linalg.norm(end - start))
            for ratio in np.linspace(0, 1, max(2, int(length / 2))):
                point = start + (end - start) * ratio
                found = False
                for y_offset in range(-2, 3):
                    for x_offset in range(-2, 3):
                        x = int(round(point[0] + x_offset))
                        y = int(round(point[1] + y_offset))
                        if (
                            0 <= x < edges.shape[1]
                            and 0 <= y < edges.shape[0]
                            and edges[y, x] > 0
                        ):
                            found = True
                            break
                    if found:
                        break
                supported += int(found)
                sample_count += 1
        return supported / max(1, sample_count)

    def find_grid_levels(self, image, polygon):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        edges = cv2.Canny(cv2.GaussianBlur(enhanced, (5, 5), 1.2), 30, 100)

        samples = []
        for scale in np.arange(0.10, 1.011, 0.01):
            vertices = self.scaled_vertices(
                polygon["vertices"],
                polygon["center"],
                float(scale),
            )
            samples.append(
                {
                    "scale": float(scale),
                    "radius": float(polygon["radius"] * scale),
                    "support": self.polygon_edge_support(edges, vertices),
                    "vertices": vertices,
                }
            )

        peaks = []
        for index in range(1, len(samples) - 1):
            current = samples[index]
            if (
                current["support"] >= samples[index - 1]["support"]
                and current["support"] >= samples[index + 1]["support"]
            ):
                peaks.append(current)

        selected = []
        # Tick labels are normally placed near the outer radial axis, so prefer
        # reliable outer grid levels instead of fully supported tiny center grids.
        outer_peaks = [item for item in peaks if item["scale"] >= 0.45]
        peak_groups = []
        for peak in sorted(outer_peaks, key=lambda item: item["scale"]):
            if (
                peak_groups
                and peak["scale"] - peak_groups[-1][-1]["scale"] <= 0.031
                and abs(peak["support"] - peak_groups[-1][-1]["support"]) <= 0.08
            ):
                peak_groups[-1].append(peak)
            else:
                peak_groups.append([peak])

        collapsed_peaks = []
        for group in peak_groups:
            best_support = max(item["support"] for item in group)
            strongest = [
                item for item in group
                if item["support"] >= best_support - 0.01
            ]
            collapsed_peaks.append(strongest[(len(strongest) - 1) // 2])

        for candidate in sorted(collapsed_peaks, key=lambda item: item["scale"], reverse=True):
            if candidate["support"] < 0.25:
                continue
            # Keep marked polygons far apart so the tick reader can distinguish
            # their labels. A third level provides a fallback for one bad read.
            if all(abs(candidate["scale"] - item["scale"]) >= 0.18 for item in selected):
                selected.append(candidate)
            if len(selected) >= 3:
                break
        return sorted(selected, key=lambda item: item["radius"], reverse=True)

    def recognize_level_ticks(self, image_path, image, polygon, levels, output_dir, name):
        tick_reader = RadarChartEncoder()
        tick_reader.coords = [
            int(round(polygon["center"][0])),
            int(round(polygon["center"][1])),
        ]
        recognized = []
        for index, level in enumerate(levels):
            marked = image.copy()
            vertices = np.round(level["vertices"]).astype(int)
            cv2.polylines(marked, [vertices], True, (0, 255, 0), 2, cv2.LINE_AA)
            marked_path = os.path.join(output_dir, f"{name}_tick_level_{index + 1}.png")
            cv2.imwrite(marked_path, marked)
            response = tick_reader.find_tick(
                int(round(level["radius"])),
                marked_path,
            )
            tick = response.get("tick") if isinstance(response, dict) else None
            recognized.append(
                {
                    **level,
                    "tick": tick,
                    "tick_reason": response.get("res") if isinstance(response, dict) else None,
                    "marked_path": marked_path,
                }
            )
        return recognized

    def fit_tick_radius(self, levels):
        valid = [level for level in levels if level.get("tick") is not None]
        if len(valid) < 2:
            raise ValueError("At least two polygon levels with recognized ticks are required.")

        candidates = []
        for first_index, first in enumerate(valid):
            for second in valid[first_index + 1:]:
                tick_delta = second["tick"] - first["tick"]
                if tick_delta == 0:
                    continue
                a = (second["radius"] - first["radius"]) / tick_delta
                if a <= 0:
                    continue
                b = first["radius"] - a * first["tick"]
                maximum_radius = max(first["radius"], second["radius"])
                if abs(b) > maximum_radius * 0.20:
                    continue
                radius_span = abs(first["radius"] - second["radius"])
                # Radar scales normally originate close to radius zero. Prefer
                # a wide pair and penalize relations with an implausible offset.
                score = radius_span - abs(b) * 2.0
                candidates.append((score, first, second, a, b))

        if not candidates:
            raise ValueError(
                "No recognized polygon tick pair increases together with radius."
            )

        _, first, second, a, b = max(candidates, key=lambda item: item[0])
        tick_values = sorted({float(level["tick"]) for level in valid})
        tick_differences = [
            second - first
            for first, second in zip(tick_values, tick_values[1:])
            if second > first
        ]
        return {
            "a": float(a),
            "b": float(b),
            "tick_interval": float(min(tick_differences)),
            "max_tick": float(max(tick_values)),
        }

    def draw_dashed_polygon(self, image, vertices, color=(128, 128, 128),
                            thickness=1, dash_length=5, gap_length=4):
        vertices = np.asarray(vertices, dtype=np.float64)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length <= 0:
                continue
            direction = vector / length
            position = 0.0
            while position < length:
                dash_end = min(position + dash_length, length)
                p1 = tuple(np.round(start + direction * position).astype(int))
                p2 = tuple(np.round(start + direction * dash_end).astype(int))
                cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)
                position += dash_length + gap_length

    def _ray_polygon_intersection(self, vertices, center, direction):
        direction = np.asarray(direction, dtype=np.float64)
        distances = []
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge = end - start
            denominator = self._cross_2d(direction, edge)
            if abs(denominator) < 1e-8:
                continue
            offset = start - center
            distance = self._cross_2d(offset, edge) / denominator
            edge_amount = self._cross_2d(offset, direction) / denominator
            if distance >= 0 and 0 <= edge_amount <= 1:
                distances.append(distance)
        if not distances:
            return None
        return center + min(distances) * direction

    @staticmethod
    def _format_tick_value(tick):
        rounded = round(float(tick))
        if abs(float(tick) - rounded) < 1e-6:
            return str(int(rounded))
        return f"{float(tick):.2f}".rstrip("0").rstrip(".")

    def draw_four_position_tick_labels(
        self,
        image,
        vertices,
        center,
        tick,
        font_scale,
        label_offset,
    ):
        text = self._format_tick_value(tick)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size
        directions = {
            "top": np.array([0.0, -1.0]),
            "right": np.array([1.0, 0.0]),
            "bottom": np.array([0.0, 1.0]),
            "left": np.array([-1.0, 0.0]),
        }

        for position, direction in directions.items():
            intersection = self._ray_polygon_intersection(
                vertices,
                center,
                direction,
            )
            if intersection is None:
                continue
            x, y = np.round(intersection).astype(int)
            if position == "top":
                origin = (x - text_width // 2, y - label_offset)
            elif position == "right":
                origin = (x + label_offset, y + text_height // 2)
            elif position == "bottom":
                origin = (
                    x - text_width // 2,
                    y + text_height + label_offset,
                )
            else:
                origin = (
                    x - text_width - label_offset,
                    y + text_height // 2,
                )

            origin = (
                max(1, min(image.shape[1] - text_width - 1, origin[0])),
                max(text_height + 1, min(image.shape[0] - baseline - 1, origin[1])),
            )
            cv2.putText(
                image,
                text,
                origin,
                font,
                font_scale,
                (255, 255, 255),
                thickness + 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                text,
                origin,
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

    def encrypt_polygon_grid(self, image, polygon, relation=None):
        result = image.copy()
        center = polygon["center"]
        vertices = polygon["vertices"]
        tick_scales = []
        if relation is None:
            total_levels = self.base_level_count * self.density
            scales = [
                level / total_levels
                for level in range(1, total_levels)
                if level % self.density != 0
            ]
        else:
            half_interval = relation["tick_interval"] / self.density
            tick = relation["max_tick"] - half_interval
            scales = []
            while tick > 0:
                radius = relation["a"] * tick + relation["b"]
                scale = radius / polygon["radius"]
                if 0 < scale < 1.02:
                    scales.append(scale)
                    tick_scales.append((tick, scale))
                tick -= relation["tick_interval"]

        for scale in scales:
            self.draw_dashed_polygon(
                result,
                self.scaled_vertices(vertices, center, scale),
            )
        if relation is not None and tick_scales:
            interval_radius = abs(
                relation["a"] * relation["tick_interval"] / self.density
            )
            font_scale = max(
                0.25,
                min(0.65, min(image.shape[:2]) / 900, interval_radius / 55),
            )
            label_offset = max(3, int(round(font_scale * 8)))
            for tick, scale in tick_scales:
                self.draw_four_position_tick_labels(
                    result,
                    self.scaled_vertices(vertices, center, scale),
                    center,
                    tick,
                    font_scale,
                    label_offset,
                )
        return result

    def visualize_detection(self, image, polygon):
        result = image.copy()
        center = tuple(np.round(polygon["center"]).astype(int))
        vertices = np.round(polygon["vertices"]).astype(int)
        detected_vertices = np.round(polygon["detected_vertices"]).astype(int)
        cv2.polylines(result, [vertices], True, (0, 255, 0), 2, cv2.LINE_AA)
        for detected, corrected in zip(detected_vertices, vertices):
            if np.linalg.norm(detected - corrected) > 2:
                cv2.line(
                    result,
                    tuple(detected),
                    tuple(corrected),
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.circle(result, tuple(detected), 3, (0, 0, 255), -1)
        for index, vertex in enumerate(vertices):
            point = tuple(vertex)
            cv2.circle(result, point, 4, (0, 255, 0), -1)
            cv2.putText(
                result,
                str(index),
                (point[0] + 4, point[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.circle(result, center, 5, (255, 0, 0), -1)
        return result

    def process_single_image(self, image_path, output_dir):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        polygon = self.find_outer_polygon(image)
        if polygon is None:
            raise ValueError("Unable to find a regular radar polygon.")

        os.makedirs(output_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(image_path))[0]
        levels = self.find_grid_levels(image, polygon)
        if len(levels) < 2:
            raise ValueError("Unable to find two reliable polygon grid levels.")
        recognized_levels = self.recognize_level_ticks(
            image_path,
            image,
            polygon,
            levels,
            output_dir,
            name,
        )
        relation = self.fit_tick_radius(recognized_levels)
        detection_path = os.path.join(output_dir, f"{name}_polygon_detection.png")
        encode_path = os.path.join(output_dir, f"{name}_polygon_encode.png")
        json_path = os.path.join(output_dir, f"{name}_polygon.json")

        cv2.imwrite(detection_path, self.visualize_detection(image, polygon))
        cv2.imwrite(encode_path, self.encrypt_polygon_grid(image, polygon, relation))

        export = {
            "detection_source": polygon["detection_source"],
            "center": [float(value) for value in polygon["center"]],
            "initial_center": [
                float(value) for value in polygon["initial_center"]
            ],
            "center_refinement_shift": float(polygon["center_refinement_shift"]),
            "regular_center_score": float(polygon["regular_center_score"]),
            "sides": int(polygon["sides"]),
            "outer_radius": float(polygon["radius"]),
            "radius_std": float(polygon["radius_std"]),
            "vertices": [
                [float(value) for value in vertex]
                for vertex in polygon["vertices"]
            ],
            "detected_vertices": [
                [float(value) for value in vertex]
                for vertex in polygon["detected_vertices"]
            ],
            "regular_phase_degrees": float(polygon["regular_phase_degrees"]),
            "detected_gap_errors_degrees": [
                float(value) for value in polygon["detected_gap_errors_degrees"]
            ],
            "vertex_angle_errors_degrees": [
                float(value) for value in polygon["vertex_angle_errors_degrees"]
            ],
            "vertex_correction_distances": [
                float(value) for value in polygon["vertex_correction_distances"]
            ],
            "corrected_vertex_indices": [
                int(value) for value in polygon["corrected_vertex_indices"]
            ],
            "max_final_angle_error_degrees": float(
                polygon["max_final_angle_error_degrees"]
            ),
            "axis_matches": polygon.get("axis_matches"),
            "axis_error_degrees": polygon.get("axis_error_degrees"),
            "outer_edge_support": polygon.get("outer_edge_support"),
            "grid_levels": [
                {
                    "scale": float(level["scale"]),
                    "radius": float(level["radius"]),
                    "support": float(level["support"]),
                    "tick": level["tick"],
                    "tick_reason": level["tick_reason"],
                    "marked_path": level["marked_path"],
                }
                for level in recognized_levels
            ],
            "tick_radius_relation": relation,
            "detection_path": detection_path,
            "encode_path": encode_path,
        }
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(export, file, ensure_ascii=False, indent=2)
        return export


if __name__ == "__main__":
    encoder = RadarPolygonEncoder()
    result = encoder.process_single_image(
        os.path.join(
            "backend",
            "real",
            "RadarChart-18 & RoseChart-6",
            "RadarChart-18-final",
            "RadarChart1.png",
        ),
        os.path.join("data", "output", "radar_polygon"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
