from modules.confidence_filter import ConfidenceFilter
from modules.prediction_buffer import PredictionBuffer


def test_confidence_and_buffer_pipeline():
    confidence_filter = ConfidenceFilter(0.8)
    buffer = PredictionBuffer(max_size=5, min_votes=3)

    accepted = []
    for label, confidence in [("A", 0.91), ("A", 0.92), ("B", 0.95), ("A", 0.93)]:
        if confidence_filter.accepts(confidence):
            accepted.append(buffer.add(label))

    assert accepted[-1] == "A"
