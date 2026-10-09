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
