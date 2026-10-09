from __future__ import annotations

import logging
import time
from dataclasses import dataclass


class CameraError(RuntimeError):
    pass


@dataclass
class Camera:
    index: int
    width: int
    height: int
    target_fps: int
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        self._capture = None
        self._last_frame_time = None
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def is_open(self) -> bool:
        return bool(self._capture is not None and self._capture.isOpened())

    def open(self) -> None:
        import cv2

        self._capture = cv2.VideoCapture(self.index)
        if not self._capture.isOpened():
            raise CameraError(f"Unable to open camera index {self.index}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)
        if self.logger:
            self.logger.info("Camera initialized at index %s", self.index)

    def read(self):
        if self._capture is None:
            raise CameraError("Camera is not open")

        ok, frame = self._capture.read()
        now = time.perf_counter()
        if self._last_frame_time is not None:
            elapsed = now - self._last_frame_time
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                self._fps = instantaneous_fps if self._fps == 0 else (0.9 * self._fps + 0.1 * instantaneous_fps)
        self._last_frame_time = now
        return ok, frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            if self.logger:
                self.logger.info("Camera released")
