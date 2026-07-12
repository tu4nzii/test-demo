from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np


class PaddleEasyOCRAdapter:
    """Expose PaddleOCR through the EasyOCR readtext() shape used by polar code."""

    def __init__(self, lang: str = "en") -> None:
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_use_onednn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        from paddleocr import PaddleOCR

        init_attempts = [
            {
                "lang": lang,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "text_detection_model_name": "PP-OCRv5_mobile_det",
                "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
                "text_det_thresh": 0.30,
                "text_det_box_thresh": 0.50,
                "text_det_unclip_ratio": 1.25,
                "text_det_limit_side_len": 960,
                "text_det_limit_type": "max",
                "return_word_box": False,
            },
            {
                "use_angle_cls": False,
                "lang": lang,
            },
            {"lang": lang},
        ]
        last_error: Exception | None = None
        for kwargs in init_attempts:
            try:
                self.engine = PaddleOCR(**kwargs)
                return
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                if "Unknown argument" in str(exc) or "deprecated" in str(exc):
                    last_error = exc
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("PaddleOCR init failed")

    def readtext(self, image: np.ndarray | str) -> list[tuple[list[list[float]], str, float]]:
        if isinstance(image, str):
            raw_image = cv2.imread(image)
            if raw_image is None:
                return []
            image = raw_image
        if image is None or getattr(image, "size", 0) == 0:
            return []
        raw = self._predict_raw(image)
        items = _collect_new_paddleocr_results(raw)
        if not items:
            items = _collect_old_paddleocr_results(raw)
        output: list[tuple[list[list[float]], str, float]] = []
        for item in items:
            box = item.get("box")
            text = str(item.get("text", "") or "")
            score = float(item.get("score", 0.0) or 0.0)
            if text and _is_bbox(box):
                output.append((_normalize_box(box), text, score))
        return output

    def _predict_raw(self, image: np.ndarray) -> Any:
        if hasattr(self.engine, "predict"):
            return self.engine.predict(image)
        if hasattr(self.engine, "ocr"):
            try:
                return self.engine.ocr(image, cls=True)
            except TypeError:
                return self.engine.ocr(image)
        raise RuntimeError("PaddleOCR object has no predict/ocr method")


def _is_bbox(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 2 and value.shape[0] == 4 and value.shape[1] >= 2
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    for point in value:
        if not isinstance(point, (list, tuple, np.ndarray)) or len(point) < 2:
            return False
        if not all(isinstance(coord, (int, float, np.integer, np.floating)) for coord in point[:2]):
            return False
    return True


def _normalize_box(box: Any) -> list[list[float]]:
    points = np.array([[float(point[0]), float(point[1])] for point in box], dtype=np.float32)
    return points.tolist()


def _parse_old_ocr_line(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2 or not _is_bbox(item[0]):
        return None
    payload = item[1]
    text = ""
    score = 0.0
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        text = str(payload[0])
        try:
            score = float(payload[1])
        except (TypeError, ValueError):
            score = 0.0
    elif isinstance(payload, str):
        text = payload
        score = 1.0
    if not text:
        return None
    return {"text": text, "score": score, "box": _normalize_box(item[0])}


def _collect_old_paddleocr_results(raw: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if raw is None:
        return results
    parsed = _parse_old_ocr_line(raw)
    if parsed is not None:
        return [parsed]
    if isinstance(raw, (list, tuple)):
        for item in raw:
            results.extend(_collect_old_paddleocr_results(item))
    return results


def _collect_new_paddleocr_results(raw: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]

    for item in raw:
        data = None
        if isinstance(item, dict) or hasattr(item, "get"):
            data = item
        elif hasattr(item, "json"):
            try:
                data = item.json() if callable(item.json) else item.json
            except Exception:
                data = None
        elif hasattr(item, "to_dict"):
            try:
                data = item.to_dict()
            except Exception:
                data = None
        if not isinstance(data, dict):
            continue

        texts = data.get("rec_texts") or data.get("texts") or []
        scores = data.get("rec_scores") or data.get("scores") or [1.0] * len(texts)
        boxes = data.get("dt_polys") or data.get("rec_polys") or data.get("boxes") or []
        for text, score, box in zip(texts, scores, boxes):
            if not text or not _is_bbox(box):
                continue
            results.append({"text": str(text), "score": float(score), "box": _normalize_box(box)})
    return results


def init_paddle_ocr_reader(lang: str = "en") -> PaddleEasyOCRAdapter:
    return PaddleEasyOCRAdapter(lang=lang)
