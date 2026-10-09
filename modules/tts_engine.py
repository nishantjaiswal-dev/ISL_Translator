from __future__ import annotations

import logging
import queue
import threading


class TTSEngine:
    def __init__(self, rate: int, volume: float, logger: logging.Logger | None = None) -> None:
        self.rate = rate
        self.volume = volume
        self.logger = logger
        self._engine = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._start()

    def speak(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned and self._thread is not None:
            self._queue.put(cleaned)

    def stop(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def _start(self) -> None:
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
        except Exception as exc:
            if self.logger:
                self.logger.warning("Text-to-speech unavailable: %s", exc)
            return

        self._thread = threading.Thread(target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                if self.logger:
                    self.logger.warning("Text-to-speech failed: %s", exc)
