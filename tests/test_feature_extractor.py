import numpy as np

from modules.feature_extractor import FeatureExtractor
from modules.hand_tracker import BoundingBox, HandDetection


def test_feature_extractor_builds_cnn_tensor():
    extractor = FeatureExtractor(model_input_shape=(None, 128, 128, 1))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detection = HandDetection(
        bbox=BoundingBox(20, 20, 140, 140),
        landmarks=(),
        handedness="Right",
        score=0.9,
    )
    tensor = extractor.extract(frame=frame, detection=detection)
    assert tensor.shape == (1, 128, 128, 1)
    assert tensor.dtype == np.float32


def test_feature_extractor_masks_hand_landmarks():
    from modules.hand_tracker import LandmarkPoint

    extractor = FeatureExtractor(model_input_shape=(None, 128, 128, 1))
    frame = np.full((200, 200, 3), 180, dtype=np.uint8)  # bright frame
    landmarks = (
        LandmarkPoint(0.2, 0.2, 0.0),
        LandmarkPoint(0.5, 0.2, 0.0),
        LandmarkPoint(0.5, 0.5, 0.0),
        LandmarkPoint(0.2, 0.5, 0.0),
    )
    detection = HandDetection(
        bbox=BoundingBox(20, 20, 140, 140),
        landmarks=landmarks,
        handedness="Right",
        score=0.95,
    )
    tensor = extractor.extract(frame=frame, detection=detection)
    assert tensor.shape == (1, 128, 128, 1)
    # Ensure pixels outside the dilated mask were set to zero (black)
    assert np.any(tensor == 0.0)
