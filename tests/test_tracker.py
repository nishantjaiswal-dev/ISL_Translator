from modules.hand_tracker import BoundingBox, HandDetection, HandTracker, TrackingResult


def test_tracking_result_data_shape():
    detection = HandDetection(
        bbox=BoundingBox(10, 20, 100, 120),
        landmarks=(),
        handedness="Right",
        score=0.9,
    )
    result = TrackingResult(detections=(detection,))
    assert result.detections[0].bbox.x_min == 10
    assert result.detections[0].handedness == "Right"


def test_handedness_is_normalized_to_user_view():
    tracker = HandTracker(max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    assert tracker._normalize_handedness("Left") == "Right"
    assert tracker._normalize_handedness("Right") == "Left"
