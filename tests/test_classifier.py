from pathlib import Path

import config
from modules.classifier import GestureClassifier


def test_classifier_loads_label_formats(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text('{"0": "A", "1": "B"}', encoding="utf-8")
    classifier = GestureClassifier(
        model_path=tmp_path / "missing.h5",
        labels_path=labels,
        model_info_path=tmp_path / "missing.json",
        model_dir=tmp_path,
    )
    assert classifier._load_labels() == {0: "A", 1: "B"}


def test_project_labels_json_has_35_labels():
    classifier = GestureClassifier(
        model_path=config.MODEL_PATH,
        labels_path=config.LABELS_PATH,
        model_info_path=config.MODEL_INFO_PATH,
        model_dir=config.MODEL_DIR,
    )
    labels = classifier._load_labels(strict=False)
    assert len(labels) == 35
    assert labels[0] == "1"
    assert labels[34] == "Z"
