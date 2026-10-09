from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import config


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class HandDetection:
    bbox: BoundingBox
    landmarks: tuple[LandmarkPoint, ...]
    handedness: str
    score: float


@dataclass(frozen=True)
class TrackingResult:
    detections: tuple[HandDetection, ...]
    raw_results: object | None = None


class HandTracker:
    def __init__(
        self,
        max_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.logger = logger
        self._mp = None
        self._hands = None
        self._drawing = None
        self._styles = None

    def start(self) -> None:
        import mediapipe as mp

        self._mp = mp
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            model_complexity=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        if self.logger:
            self.logger.info("MediaPipe Hands initialized")

    def process(self, frame) -> TrackingResult:
        if self._hands is None:
            raise RuntimeError("HandTracker.start() must be called before process()")

        import cv2

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        detections: list[HandDetection] = []
        multi_landmarks = results.multi_hand_landmarks or []
        multi_handedness = results.multi_handedness or []
        for index, hand_landmarks in enumerate(multi_landmarks):
            handedness = "Unknown"
            score = 0.0
            if index < len(multi_handedness):
                classification = multi_handedness[index].classification[0]
                handedness = self._normalize_handedness(classification.label)
                score = float(classification.score)

            landmark_points: list[LandmarkPoint] = []
            x_coords: list[int] = []
            y_coords: list[int] = []
            for point in hand_landmarks.landmark:
                px = int(point.x * width)
                py = int(point.y * height)
                x_coords.append(px)
                y_coords.append(py)
                landmark_points.append(LandmarkPoint(float(point.x), float(point.y), float(point.z)))

            if not x_coords or not y_coords:
                continue

            padding = config.HAND_PADDING_PIXELS
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(width - 1, max(x_coords) + padding)
            y_max = min(height - 1, max(y_coords) + padding)
            if x_max <= x_min or y_max <= y_min:
                continue

            detections.append(
                HandDetection(
                    bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    landmarks=tuple(landmark_points),
                    handedness=handedness,
                    score=score,
                )
            )

        return TrackingResult(detections=tuple(detections), raw_results=results)

    def _normalize_handedness(self, label: str) -> str:
        if not config.NORMALIZE_HANDEDNESS:
            return label
        normalized = str(label).strip().lower()
        if normalized == "left":
            return "Right"
        if normalized == "right":
            return "Left"
        return label

    def draw(self, frame, result: TrackingResult):
        if self._drawing is None or self._mp is None or result.raw_results is None:
            return frame

        import cv2

        landmarks: Sequence = result.raw_results.multi_hand_landmarks or []
        for detection, hand_landmarks in zip(result.detections, landmarks):
            self._drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self._mp.solutions.hands.HAND_CONNECTIONS,
                self._styles.get_default_hand_landmarks_style() if self._styles else None,
                self._styles.get_default_hand_connections_style() if self._styles else None,
            )
            bbox = detection.bbox
            cv2.rectangle(frame, (bbox.x_min, bbox.y_min), (bbox.x_max, bbox.y_max), (79, 214, 172), 2)
            label = f"{detection.handedness} {detection.score:.2f}"
            cv2.putText(
                frame,
                label,
                (bbox.x_min, max(24, bbox.y_min - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (79, 214, 172),
                2,
                cv2.LINE_AA,
            )
        return frame

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            if self.logger:
                self.logger.info("MediaPipe Hands closed")
