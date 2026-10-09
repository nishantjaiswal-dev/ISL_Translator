from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

import config
from modules.hand_tracker import BoundingBox, HandDetection


class FeatureExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImagePreprocessorConfig:
    target_size: tuple[int, int]
    channels: int = 1
    normalization: str = config.DEFAULT_IMAGE_NORMALIZATION


class FeatureExtractor:
    def __init__(
        self,
        model_input_shape: tuple[int | None, ...] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger
        self.config = self._build_config(model_input_shape)

    def extract(self, frame, detection: HandDetection) -> np.ndarray:
        crop = self._crop_hand(frame, detection.bbox)
        if getattr(config, "MASK_HAND_BACKGROUND", False) and detection.landmarks:
            crop = self._apply_hand_mask(crop, frame.shape[:2], detection)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.config.target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.expand_dims(normalized, axis=-1)
        normalized = np.expand_dims(normalized, axis=0)
        return normalized

    def _apply_hand_mask(self, crop: np.ndarray, frame_shape: tuple[int, int], detection: HandDetection) -> np.ndarray:
        frame_h, frame_w = frame_shape
        bbox = detection.bbox
        crop_h, crop_w = crop.shape[:2]
        pts = []
        for lm in detection.landmarks:
            px = int(lm.x * frame_w) - bbox.x_min
            py = int(lm.y * frame_h) - bbox.y_min
            px = max(0, min(crop_w - 1, px))
            py = max(0, min(crop_h - 1, py))
            pts.append([px, py])

        if len(pts) < 3:
            return crop

        pts_arr = np.array(pts, dtype=np.int32)
        hull = cv2.convexHull(pts_arr)
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        dilation = getattr(config, "HAND_MASK_DILATION", 20)
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
            mask = cv2.dilate(mask, kernel)

        return cv2.bitwise_and(crop, crop, mask=mask)

    def _build_config(self, model_input_shape: tuple[int | None, ...] | None) -> ImagePreprocessorConfig:
        shape = self._normalize_input_shape(model_input_shape)
        if shape is None:
            shape = config.DEFAULT_MODEL_INPUT_SHAPE

        if len(shape) != 3:
            raise FeatureExtractionError(f"Expected image input shape with 3 dimensions, got {shape}")

        height, width, channels = shape
        if channels != 1:
            raise FeatureExtractionError(
                f"This application is configured for grayscale input, but the model expects {shape}."
            )
        if height is None or width is None:
            height, width = config.MODEL_INPUT_SIZE
        return ImagePreprocessorConfig(target_size=(int(width), int(height)), channels=int(channels))

    def _normalize_input_shape(self, model_input_shape: tuple[int | None, ...] | None) -> tuple[int | None, ...] | None:
        if model_input_shape is None:
            return None
        shape = tuple(model_input_shape)
        if len(shape) == 4:
            return shape[1:]
        return shape

    def _crop_hand(self, frame, bbox: BoundingBox) -> np.ndarray:
        height, width = frame.shape[:2]
        x_min = max(0, min(width - 1, bbox.x_min))
        y_min = max(0, min(height - 1, bbox.y_min))
        x_max = max(0, min(width, bbox.x_max))
        y_max = max(0, min(height, bbox.y_max))
        if x_max <= x_min or y_max <= y_min:
            raise FeatureExtractionError("Invalid hand crop bounds")

        crop = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            raise FeatureExtractionError("Empty hand crop")
        return crop
