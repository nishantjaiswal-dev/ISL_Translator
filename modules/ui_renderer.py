from __future__ import annotations


class UIRenderer:
    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self._ui_ready = False

    def create_window(self) -> None:
        try:
            import cv2

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            self._ui_ready = True
        except Exception:
            self._ui_ready = False

    def render(
        self,
        frame,
        fps: float,
        prediction: str,
        stable_prediction: str,
        confidence: float,
        current_word: str,
        sentence: str,
        tracking_status: str,
        model_status: str,
        last_action: str,
    ):
        import cv2

        canvas = frame.copy()
        height, width = canvas.shape[:2]
        panel_width = min(430, max(330, width // 3))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, height), (20, 24, 31), -1)
        cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)

        self._title(canvas, "ISL Translator", 24, 44)
        y = 88
        y = self._metric(canvas, "Prediction", prediction or "-", y, accent=(79, 214, 172))
        y = self._metric(canvas, "Stable", stable_prediction or "-", y, accent=(96, 165, 250))
        y = self._metric(canvas, "Confidence", f"{confidence * 100:.1f}%", y)
        y = self._metric(canvas, "FPS", f"{fps:.1f}", y)
        y = self._status(canvas, "Tracking", tracking_status, y)
        y = self._status(canvas, "Model", model_status, y)

        y += 8
        y = self._block(canvas, "Current Word", current_word or "-", y, panel_width - 34)
        y = self._block(canvas, "Sentence", sentence.strip() or "-", y, panel_width - 34)
        if last_action:
            self._small(canvas, f"Last: {last_action}", 22, min(height - 24, y + 16), (195, 202, 212))

        self._footer(canvas, width, height)
        return canvas

    def show(self, frame) -> int:
        if not self._ui_ready:
            return -1
        try:
            import cv2

            cv2.imshow(self.window_name, frame)
            return cv2.waitKey(1) & 0xFF
        except Exception:
            self._ui_ready = False
            return -1

    def close(self) -> None:
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass
        self._ui_ready = False

    def _title(self, image, text: str, x: int, y: int) -> None:
        import cv2

        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.85, (245, 247, 250), 2, cv2.LINE_AA)

    def _metric(self, image, label: str, value: str, y: int, accent=(226, 232, 240)) -> int:
        self._small(image, label.upper(), 24, y, (148, 163, 184))
        import cv2

        cv2.putText(image, value[:24], (24, y + 34), cv2.FONT_HERSHEY_DUPLEX, 0.82, accent, 2, cv2.LINE_AA)
        return y + 72

    def _status(self, image, label: str, value: str, y: int) -> int:
        color = (79, 214, 172) if value.lower() in {"tracking", "loaded"} else (250, 204, 21)
        self._small(image, label.upper(), 24, y, (148, 163, 184))
        import cv2

        cv2.circle(image, (29, y + 29), 6, color, -1)
        cv2.putText(image, value, (44, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (226, 232, 240), 2, cv2.LINE_AA)
        return y + 58

    def _block(self, image, label: str, value: str, y: int, max_width: int) -> int:
        self._small(image, label.upper(), 24, y, (148, 163, 184))
        lines = self._wrap(value, max_width=max_width, font_scale=0.58)
        import cv2

        line_y = y + 30
        for line in lines[:4]:
            cv2.putText(image, line, (24, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (241, 245, 249), 2, cv2.LINE_AA)
            line_y += 28
        return line_y + 14

    def _small(self, image, text: str, x: int, y: int, color) -> None:
        import cv2

        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    def _footer(self, image, width: int, height: int) -> None:
        import cv2

        text = "Q quit   S speak   Enter commit   Space word-space   Backspace delete   C clear"
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = max(16, width - size[0] - 20)
        y = height - 18
        cv2.rectangle(image, (x - 10, y - 22), (width - 10, y + 8), (15, 23, 42), -1)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1, cv2.LINE_AA)

    def _wrap(self, text: str, max_width: int, font_scale: float) -> list[str]:
        import cv2

        words = text.split()
        if not words:
            return ["-"]
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0][0]
            if width <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or ["-"]
