from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import config
from modules.camera import Camera
from modules.classifier import GestureClassifier, ModelLoadError, ModelNotFoundError
from modules.confidence_filter import ConfidenceFilter
from modules.feature_extractor import FeatureExtractionError, FeatureExtractor
from modules.hand_tracker import HandTracker
from modules.logger import configure_logging
from modules.prediction_buffer import PredictionBuffer
from modules.spell_corrector import SpellCorrector
from modules.tts_engine import TTSEngine
from modules.ui_renderer import UIRenderer
from modules.word_builder import WordBuilder


@dataclass
class RuntimeState:
    prediction: str = ""
    stable_prediction: str = ""
    confidence: float = 0.0
    tracking_status: str = "Starting"
    model_status: str = "Not loaded"
    last_action: str = ""


class ISLTranslatorApplication:
    def __init__(self) -> None:
        self.logger = configure_logging()
        self.state = RuntimeState()
        self.camera = Camera(
            index=config.CAMERA_INDEX,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            target_fps=config.TARGET_FPS,
            logger=self.logger,
        )
        self.tracker = HandTracker(
            max_hands=config.MAX_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            logger=self.logger,
        )
        self.classifier = GestureClassifier(
            model_path=config.MODEL_PATH,
            labels_path=config.LABELS_PATH,
            model_info_path=config.MODEL_INFO_PATH,
            model_dir=config.MODEL_DIR,
            logger=self.logger,
        )
        self.extractor: FeatureExtractor | None = None
        self.confidence_filter = ConfidenceFilter(config.CONFIDENCE_THRESHOLD)
        self.prediction_buffer = PredictionBuffer(
            max_size=config.PREDICTION_BUFFER_SIZE,
            min_votes=config.PREDICTION_MIN_VOTES,
        )
        self.word_builder = WordBuilder(
            hold_time_seconds=config.LETTER_HOLD_TIME,
            delete_labels=config.CONTROL_DELETE_LABELS,
            space_labels=config.CONTROL_SPACE_LABELS,
            clear_labels=config.CONTROL_CLEAR_LABELS,
            ignore_labels=config.CONTROL_IGNORE_LABELS,
        )
        self.spell_corrector = SpellCorrector(logger=self.logger)
        self.tts = TTSEngine(rate=config.TTS_RATE, volume=config.TTS_VOLUME, logger=self.logger)
        self.ui = UIRenderer(config.WINDOW_NAME)

    def start(self) -> int:
        self.logger.info("Starting ISL Translator")
        try:
            self.classifier.load()
            self.extractor = FeatureExtractor(self.classifier.metadata.input_shape, logger=self.logger)
            self.state.model_status = f"Loaded model ({len(self.classifier.metadata.labels)} classes)"
            self.logger.info(
                "Model ready with input_shape=%s output_shape=%s label_count=%s",
                self.classifier.metadata.input_shape,
                self.classifier.metadata.output_shape,
                len(self.classifier.metadata.labels),
            )
        except ModelNotFoundError as exc:
            self.state.model_status = "Model missing"
            self.logger.error("Model file missing: %s", exc)
            self.extractor = None
        except ModelLoadError as exc:
            self.state.model_status = "Model load failed"
            self.logger.error("Model failed to load: %s", exc)
            self.extractor = None

        try:
            self.camera.open()
            self.tracker.start()
            self.ui.create_window()
            self._loop()
            return 0
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 0
        except Exception:
            self.logger.exception("Fatal application error")
            return 1
        finally:
            self.shutdown()

    def _loop(self) -> None:
        while True:
            ok, frame = self.camera.read()
            if not ok or frame is None:
                self.state.tracking_status = "Camera unavailable"
                self.logger.warning("Camera frame unavailable")
                time.sleep(0.05)
                continue

            raw_frame = frame.copy()
            detections = []
            try:
                tracking = self.tracker.process(raw_frame)
                detections = tracking.detections
                frame = self.tracker.draw(frame, tracking)
                self.state.tracking_status = "Tracking" if detections else "No hand"
            except Exception as exc:
                self.state.tracking_status = "Tracking error"
                self.logger.warning("MediaPipe processing failed: %s", exc)

            if detections and self.classifier.is_loaded and self.extractor is not None:
                self._predict(raw_frame, detections[0])
            else:
                self.prediction_buffer.clear()
                self.word_builder.reset_hold()
                if not self.classifier.is_loaded:
                    self.state.prediction = ""
                    self.state.stable_prediction = ""
                    self.state.confidence = 0.0

            rendered = self.ui.render(
                frame=frame,
                fps=self.camera.fps,
                prediction=self.state.prediction,
                stable_prediction=self.state.stable_prediction,
                confidence=self.state.confidence,
                current_word=self.word_builder.current_word,
                sentence=self.word_builder.sentence,
                tracking_status=self.state.tracking_status,
                model_status=self.state.model_status,
                last_action=self.state.last_action,
            )
            key = self.ui.show(rendered)
            if self._handle_key(key):
                break

    def _predict(self, frame, detection) -> None:
        assert self.extractor is not None
        try:
            features = self.extractor.extract(frame=frame, detection=detection)
            prediction = self.classifier.predict(features)
        except FeatureExtractionError as exc:
            self.state.tracking_status = "Feature error"
            self.logger.warning("Feature extraction failed: %s", exc)
            return
        except Exception as exc:
            self.state.model_status = "Inference error"
            self.logger.warning("Inference failed: %s", exc)
            return

        self.state.prediction = prediction.label
        self.state.confidence = prediction.confidence

        if not self.confidence_filter.accepts(prediction.confidence):
            self.state.stable_prediction = ""
            self.prediction_buffer.clear()
            self.word_builder.reset_hold()
            return

        stable = self.prediction_buffer.add(prediction.label)
        self.state.stable_prediction = stable or ""
        if stable:
            action = self.word_builder.observe(stable, prediction.confidence)
            if action:
                self.state.last_action = action
                self.logger.info("Accepted gesture %s: %s", stable, action)

    def _handle_key(self, key: int) -> bool:
        if key < 0:
            return False
        if key == config.KEY_QUIT:
            return True
        if key == config.KEY_SPEAK:
            text = self.word_builder.text_for_speech()
            if text:
                self.tts.speak(text)
                self.state.last_action = "Spoke text"
            return False
        if key == config.KEY_CLEAR:
            self.word_builder.clear_all()
            self.prediction_buffer.clear()
            self.state.last_action = "Cleared text"
            return False
        if key == config.KEY_SPACE:
            accepted = self.word_builder.accept_space(self.spell_corrector)
            self.state.last_action = accepted or "Added space"
            return False
        if key == config.KEY_ENTER:
            accepted = self.word_builder.commit_word(self.spell_corrector)
            self.state.last_action = accepted or "Committed word"
            return False
        if key == config.KEY_BACKSPACE:
            self.word_builder.delete_last()
            self.state.last_action = "Deleted last character"
            return False
        return False

    def shutdown(self) -> None:
        self.logger.info("Shutting down ISL Translator")
        self.tts.stop()
        self.tracker.close()
        self.camera.release()
        self.ui.close()


def main() -> int:
    return ISLTranslatorApplication().start()


if __name__ == "__main__":
    sys.exit(main())

