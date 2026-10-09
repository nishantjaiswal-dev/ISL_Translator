# -*- coding: utf-8 -*-
"""
evaluate_system.py — Offline, full-pipeline evaluation harness for the ISL Translator.

WHY THIS EXISTS
----------------
`isl_model.h5`'s training-time validation accuracy tells you how good the *raw CNN* is
on clean, pre-cropped dataset images. It does NOT tell you how the *deployed system*
performs, because the live app adds several extra stages the model never sees during
training: MediaPipe hand detection + cropping, a 0.80 confidence gate, and 7-frame
majority-vote temporal smoothing. Any of those can raise OR lower real-world accuracy.

This script runs the *actual production modules* (GestureClassifier, FeatureExtractor,
HandTracker, ConfidenceFilter, PredictionBuffer — imported directly from modules/,
not reimplemented) against a static, labeled image folder, so you get system-level
metrics suitable for citing in a paper's Results section, without touching a webcam.

WHAT IT PRODUCES (in --output-dir)
-----------------------------------
- results.json          machine-readable metrics bundle
- predictions.csv        per-image raw log (path, true/pred label, confidence, latency)
- confusion_matrix.png   heatmap over all classes
- threshold_sweep.csv/png  accuracy-vs-coverage trade-off across confidence thresholds
- summary_report.md      human-readable report with a ready-to-paste per-class table

USAGE
-----
    python evaluation/evaluate_system.py --test-dir /path/to/test_images --mode pipeline
    python evaluation/evaluate_system.py --test-dir /path/to/test_images --mode direct --simulate-buffer

MODES
-----
    pipeline  (recommended) MediaPipe detection -> crop -> classify, exactly like the
              live app. Reports a hand-detection rate too, since MediaPipe can fail to
              find a hand at all — that's a real system failure mode, not captured by
              raw model accuracy.
    direct    Skip detection, classify the whole image directly. Matches the
              distribution isl_model.h5 was TRAINED on (ImageDataGenerator over whole
              images), so this isolates raw model quality from detection-stage effects.

DATASET LAYOUT EXPECTED (same convention as training_mobilenetv2_colab.py)
----------------------------------------------------------------------------
    test_dir/
        A/   img001.jpg  img002.jpg  ...
        B/   ...
        1/   ...
        ...

IMPORTANT — test-set integrity for a paper result
---------------------------------------------------
Point --test-dir at images the model never saw during training OR validation.
The training script only ever did a train/val split (val_split=0.2) on ONE folder —
there is no held-out test set yet. Re-using that same folder here would be evaluating
on data the model has already been validated against, which a reviewer will flag.
See evaluation/README.md for how to carve out a clean 3-way split, or how to build an
independent test set from fresh webcam captures (also closes the domain gap between
clean Kaggle images and real MediaPipe-cropped hands).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config  # noqa: E402
from modules.classifier import GestureClassifier  # noqa: E402
from modules.feature_extractor import FeatureExtractor  # noqa: E402
from modules.hand_tracker import HandTracker  # noqa: E402
from modules.prediction_buffer import PredictionBuffer  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SampleResult:
    path: str
    true_label: str
    pred_label: Optional[str]
    confidence: float
    detected: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Dataset discovery — pure, no ML dependencies, safe to unit test directly.
# ---------------------------------------------------------------------------

def discover_dataset(test_dir: Path) -> list[tuple[Path, str]]:
    """Walk a class-subfolder directory, return (image_path, true_label) pairs."""
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((img_path, label))

    if not samples:
        raise ValueError(f"No labeled images found under {test_dir}")
    return samples


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def preprocess_direct(frame_bgr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Mirrors FeatureExtractor.extract() but skips the hand crop (whole image)."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = np.expand_dims(normalized, axis=-1)
    normalized = np.expand_dims(normalized, axis=0)
    return normalized


def jitter_frame(frame_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Small rotation/scale/translate/brightness jitter to simulate natural hand
    micro-movement across consecutive frames of a held gesture."""
    import cv2

    height, width = frame_bgr.shape[:2]
    angle = rng.uniform(-5, 5)
    scale = rng.uniform(0.97, 1.03)
    tx = rng.uniform(-3, 3)
    ty = rng.uniform(-3, 3)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    rotated = cv2.warpAffine(frame_bgr, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    brightness = rng.uniform(0.85, 1.15)
    jittered = np.clip(rotated.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return jittered


# ---------------------------------------------------------------------------
# Model / pipeline loading (real deps: tensorflow via classifier, mediapipe via tracker)
# ---------------------------------------------------------------------------

def load_classifier_and_extractor() -> tuple[GestureClassifier, FeatureExtractor]:
    classifier = GestureClassifier(
        model_path=config.MODEL_PATH,
        labels_path=config.LABELS_PATH,
        model_info_path=config.MODEL_INFO_PATH,
        model_dir=config.MODEL_DIR,
    )
    classifier.load()
    extractor = FeatureExtractor(classifier.metadata.input_shape)
    return classifier, extractor


def load_tracker() -> HandTracker:
    tracker = HandTracker(
        max_hands=config.MAX_HANDS,
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    tracker.start()
    return tracker


# ---------------------------------------------------------------------------
# Per-sample inference
# ---------------------------------------------------------------------------

def classify_direct(classifier: GestureClassifier, extractor: FeatureExtractor, frame_bgr: np.ndarray):
    features = preprocess_direct(frame_bgr, extractor.config.target_size)
    t0 = time.perf_counter()
    prediction = classifier.predict(features)
    latency_ms = (time.perf_counter() - t0) * 1000
    return prediction.label, prediction.confidence, latency_ms, True


def classify_pipeline(classifier: GestureClassifier, extractor: FeatureExtractor, tracker: HandTracker, frame_bgr: np.ndarray):
    tracking_result = tracker.process(frame_bgr)
    if not tracking_result.detections:
        return None, 0.0, 0.0, False
    detection = max(tracking_result.detections, key=lambda d: d.score)
    features = extractor.extract(frame_bgr, detection)
    t0 = time.perf_counter()
    prediction = classifier.predict(features)
    latency_ms = (time.perf_counter() - t0) * 1000
    return prediction.label, prediction.confidence, latency_ms, True


def run_evaluation(
    samples: list[tuple[Path, str]],
    classifier: GestureClassifier,
    extractor: FeatureExtractor,
    tracker: Optional[HandTracker],
    mode: str,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 25,
    progress_every: int = 50,
) -> list[SampleResult]:
    """Runs inference over every sample.

    Prints periodic progress (count / elapsed / ETA) so long CPU runs don't look
    hung, and — if checkpoint_path is given — appends each result to a CSV as it
    goes, so a Ctrl+C or crash partway through doesn't lose everything already
    computed.
    """
    import cv2

    total = len(samples)
    results: list[SampleResult] = []
    start_time = time.perf_counter()

    checkpoint_file = None
    checkpoint_writer = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file = open(checkpoint_path, "w", newline="", encoding="utf-8")
        checkpoint_writer = csv.writer(checkpoint_file)
        checkpoint_writer.writerow(["path", "true_label", "pred_label", "confidence", "detected", "correct", "latency_ms"])

    try:
        for i, (path, true_label) in enumerate(samples, start=1):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"  [warn] could not read image, skipping: {path}")
                continue

            if mode == "direct":
                label, confidence, latency_ms, detected = classify_direct(classifier, extractor, frame)
            else:
                label, confidence, latency_ms, detected = classify_pipeline(classifier, extractor, tracker, frame)

            result = SampleResult(str(path), true_label, label, confidence, detected, latency_ms)
            results.append(result)

            if checkpoint_writer is not None:
                correct = bool(result.pred_label is not None and result.pred_label == result.true_label)
                checkpoint_writer.writerow(
                    [result.path, result.true_label, result.pred_label, f"{result.confidence:.4f}", result.detected, correct, f"{result.latency_ms:.3f}"]
                )
                if i % checkpoint_every == 0:
                    checkpoint_file.flush()

            if i % progress_every == 0 or i == total:
                elapsed = time.perf_counter() - start_time
                rate = i / elapsed if elapsed > 0 else 0.0
                remaining = (total - i) / rate if rate > 0 else 0.0
                print(
                    f"  [{i}/{total}] {i / total * 100:.1f}%  "
                    f"elapsed={_format_duration(elapsed)}  eta={_format_duration(remaining)}  "
                    f"({rate:.1f} img/s)"
                )
    finally:
        if checkpoint_file is not None:
            checkpoint_file.close()

    return results


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Metrics — pure functions over SampleResult lists, unit-testable without ML deps.
# ---------------------------------------------------------------------------

def compute_metrics(results: list[SampleResult], threshold: float) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    n_total = len(results)
    detected = [r for r in results if r.detected and r.pred_label is not None]
    n_detected = len(detected)
    detection_rate = (n_detected / n_total) if n_total else 0.0

    y_true = [r.true_label for r in detected]
    y_pred = [r.pred_label for r in detected]
    raw_accuracy = accuracy_score(y_true, y_pred) if detected else 0.0

    accepted = [r for r in detected if r.confidence >= threshold]
    coverage = (len(accepted) / n_detected) if n_detected else 0.0
    accuracy_at_threshold = (
        accuracy_score([r.true_label for r in accepted], [r.pred_label for r in accepted])
        if accepted else 0.0
    )

    labels_sorted = sorted(set(y_true) | set(y_pred))
    report = (
        classification_report(y_true, y_pred, labels=labels_sorted, output_dict=True, zero_division=0)
        if detected else {}
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted).tolist() if detected else []

    latencies = [r.latency_ms for r in detected]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    fps_estimate = (1000.0 / mean_latency) if mean_latency > 0 else 0.0

    return {
        "n_total_images": n_total,
        "n_detected": n_detected,
        "detection_rate": detection_rate,
        "raw_accuracy": raw_accuracy,
        "confidence_threshold": threshold,
        "coverage_at_threshold": coverage,
        "accuracy_at_threshold": accuracy_at_threshold,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": labels_sorted,
        "mean_latency_ms": mean_latency,
        "p95_latency_ms": p95_latency,
        "estimated_fps": fps_estimate,
    }


def sweep_thresholds(results: list[SampleResult], thresholds: list[float]) -> list[dict]:
    detected = [r for r in results if r.detected and r.pred_label is not None]
    sweep = []
    for t in thresholds:
        accepted = [r for r in detected if r.confidence >= t]
        coverage = (len(accepted) / len(detected)) if detected else 0.0
        if accepted:
            correct = sum(1 for r in accepted if r.pred_label == r.true_label)
            accuracy_if_accepted = correct / len(accepted)
        else:
            accuracy_if_accepted = 0.0
        sweep.append({"threshold": round(float(t), 2), "coverage": coverage, "accuracy_if_accepted": accuracy_if_accepted})
    return sweep


def simulate_temporal_smoothing(
    samples: list[tuple[Path, str]],
    classifier: GestureClassifier,
    extractor: FeatureExtractor,
    tracker: Optional[HandTracker],
    mode: str,
    n_frames: int,
    min_votes: int,
    sample_size: int,
    rng: random.Random,
) -> dict:
    import cv2

    chosen = rng.sample(samples, min(sample_size, len(samples)))
    n = 0
    single_correct = 0
    stabilized_correct = 0
    stabilized_total = 0

    for path, true_label in chosen:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        n += 1
        buffer = PredictionBuffer(max_size=n_frames, min_votes=min_votes)
        stable_label = None
        first_label = None

        for i in range(n_frames):
            simulated_frame = frame if i == 0 else jitter_frame(frame, rng)
            if mode == "direct":
                label, _, _, detected = classify_direct(classifier, extractor, simulated_frame)
            else:
                label, _, _, detected = classify_pipeline(classifier, extractor, tracker, simulated_frame)
            if not detected:
                continue
            if first_label is None:
                first_label = label
            resolved = buffer.add(label)
            if resolved is not None:
                stable_label = resolved

        if first_label == true_label:
            single_correct += 1
        if stable_label is not None:
            stabilized_total += 1
            if stable_label == true_label:
                stabilized_correct += 1

    return {
        "n_samples": n,
        "single_frame_accuracy": (single_correct / n) if n else 0.0,
        "stabilized_accuracy_of_resolved": (stabilized_correct / stabilized_total) if stabilized_total else 0.0,
        "stabilization_resolve_rate": (stabilized_total / n) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: list[list[int]], labels: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(cm)
    side = max(6, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("ISL Gesture Classifier — Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_threshold_sweep(sweep: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = [row["threshold"] for row in sweep]
    coverage = [row["coverage"] for row in sweep]
    accuracy = [row["accuracy_if_accepted"] for row in sweep]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(thresholds, accuracy, "o-", color="#2b6cb0", label="Accuracy (accepted predictions)")
    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Accuracy", color="#2b6cb0")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverage, "s--", color="#c05621", label="Coverage")
    ax2.set_ylabel("Coverage (fraction of predictions accepted)", color="#c05621")
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Confidence Threshold Trade-off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary_markdown(path: Path, metrics: dict, sweep: list[dict], smoothing: dict, mode: str) -> None:
    lines: list[str] = []
    lines.append(f"# ISL Translator — System Evaluation Report ({mode} mode)\n")
    lines.append(f"- Total test images: **{metrics['n_total_images']}**")
    if mode == "pipeline":
        lines.append(
            f"- Hand-detection rate (MediaPipe found a hand): "
            f"**{metrics['detection_rate'] * 100:.2f}%** ({metrics['n_detected']}/{metrics['n_total_images']})"
        )
    lines.append(f"- Raw classification accuracy (all detected images, no confidence gate): **{metrics['raw_accuracy'] * 100:.2f}%**")
    lines.append(f"- Confidence threshold evaluated: **{metrics['confidence_threshold']}**")
    lines.append(f"- Coverage at threshold (fraction of predictions the app would accept): **{metrics['coverage_at_threshold'] * 100:.2f}%**")
    lines.append(f"- Accuracy of accepted predictions (what the user actually sees): **{metrics['accuracy_at_threshold'] * 100:.2f}%**")
    lines.append(f"- Mean inference latency: **{metrics['mean_latency_ms']:.2f} ms** (p95: {metrics['p95_latency_ms']:.2f} ms)")
    lines.append(f"- Estimated throughput: **{metrics['estimated_fps']:.1f} FPS** (classifier inference only, excludes capture/detection I/O)")

    if smoothing:
        lines.append("")
        lines.append("## Temporal smoothing ablation")
        lines.append(
            f"- Sampled {smoothing['n_samples']} images, each replayed across "
            f"{config.PREDICTION_BUFFER_SIZE} jittered simulated frames"
        )
        lines.append(f"- Single-frame accuracy (no smoothing): **{smoothing['single_frame_accuracy'] * 100:.2f}%**")
        lines.append(
            f"- Stabilized accuracy after {config.PREDICTION_MIN_VOTES}-of-"
            f"{config.PREDICTION_BUFFER_SIZE} majority vote (of gestures that resolved to a stable label): "
            f"**{smoothing['stabilized_accuracy_of_resolved'] * 100:.2f}%**"
        )
        lines.append(f"- Buffer resolve rate (fraction reaching any stable vote): **{smoothing['stabilization_resolve_rate'] * 100:.2f}%**")

    lines.append("")
    lines.append("## Per-class metrics")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1-score | Support |")
    lines.append("|---|---|---|---|---|")
    report = metrics.get("classification_report", {})
    for label, stats in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(stats, dict):
            continue
        lines.append(f"| {label} | {stats['precision']:.3f} | {stats['recall']:.3f} | {stats['f1-score']:.3f} | {int(stats['support'])} |")
    if "macro avg" in report:
        m = report["macro avg"]
        lines.append(f"| **Macro avg** | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {int(m['support'])} |")
    if "weighted avg" in report:
        w = report["weighted avg"]
        lines.append(f"| **Weighted avg** | {w['precision']:.3f} | {w['recall']:.3f} | {w['f1-score']:.3f} | {int(w['support'])} |")

    if sweep:
        lines.append("")
        lines.append("## Confidence threshold sweep")
        lines.append("")
        lines.append("| Threshold | Coverage | Accuracy (accepted) |")
        lines.append("|---|---|---|")
        for row in sweep:
            lines.append(f"| {row['threshold']:.2f} | {row['coverage'] * 100:.2f}% | {row['accuracy_if_accepted'] * 100:.2f}% |")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, results: list[SampleResult], metrics: dict, sweep: list[dict], smoothing: dict, mode: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "confidence", "detected", "correct", "latency_ms"])
        for r in results:
            correct = bool(r.pred_label is not None and r.pred_label == r.true_label)
            writer.writerow([r.path, r.true_label, r.pred_label, f"{r.confidence:.4f}", r.detected, correct, f"{r.latency_ms:.3f}"])

    payload = {"mode": mode, "metrics": metrics, "threshold_sweep": sweep, "temporal_smoothing": smoothing}
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if metrics.get("confusion_matrix"):
        _plot_confusion_matrix(metrics["confusion_matrix"], metrics["confusion_matrix_labels"], output_dir / "confusion_matrix.png")

    if sweep:
        with open(output_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["threshold", "coverage", "accuracy_if_accepted"])
            for row in sweep:
                writer.writerow([row["threshold"], f"{row['coverage']:.4f}", f"{row['accuracy_if_accepted']:.4f}"])
        _plot_threshold_sweep(sweep, output_dir / "threshold_sweep.png")

    _write_summary_markdown(output_dir / "summary_report.md", metrics, sweep, smoothing, mode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline, full-pipeline evaluation for the ISL Translator (no camera required)."
    )
    parser.add_argument("--test-dir", required=True, type=Path, help="Class-subfolder directory of labeled test images")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"), help="Where to write metrics/plots")
    parser.add_argument(
        "--mode", choices=["pipeline", "direct"], default="pipeline",
        help="'pipeline' = MediaPipe detect+crop, matches the deployed app (default). "
             "'direct' = classify whole image, matches the training distribution.",
    )
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD, help="Confidence threshold to report (default: app's configured threshold)")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the confidence threshold sweep")
    parser.add_argument("--simulate-buffer", action="store_true", help="Also run the temporal-smoothing ablation (slower, re-runs inference per sample)")
    parser.add_argument("--buffer-sample-size", type=int, default=150, help="Number of images to sample for the temporal-smoothing ablation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for jitter and sampling")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate a random subset of this many images (smoke-test before committing to a full run)",
    )
    parser.add_argument(
        "--progress-every", type=int, default=50,
        help="Print a progress line every N images (default 50)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    print("Discovering test set...")
    samples = discover_dataset(args.test_dir)
    n_classes = len(set(label for _, label in samples))
    print(f"Found {len(samples)} labeled images across {n_classes} classes.")

    if args.limit is not None and args.limit < len(samples):
        samples = rng.sample(samples, args.limit)
        print(f"--limit set: evaluating a random subset of {len(samples)} images instead.")

    print("Loading classifier (this loads isl_model.h5)...")
    classifier, extractor = load_classifier_and_extractor()
    print(
        f"Model loaded via source='{classifier.metadata.source}', "
        f"input_shape={classifier.metadata.input_shape}, {len(classifier.metadata.labels)} labels."
    )

    tracker: Optional[HandTracker] = None
    if args.mode == "pipeline":
        print("Starting MediaPipe hand tracker...")
        tracker = load_tracker()

    checkpoint_path = args.output_dir / "predictions_checkpoint.csv"
    print(f"Running evaluation in '{args.mode}' mode over {len(samples)} images...")
    print(f"(Live progress below. Partial results are being written to {checkpoint_path} as it goes,")
    print(" so if this gets interrupted, nothing already computed is lost.)")
    results = run_evaluation(
        samples, classifier, extractor, tracker, args.mode,
        checkpoint_path=checkpoint_path,
        progress_every=args.progress_every,
    )

    print("Computing metrics...")
    metrics = compute_metrics(results, args.threshold)

    sweep: list[dict] = []
    if not args.no_sweep:
        thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]
        sweep = sweep_thresholds(results, thresholds)

    smoothing: dict = {}
    if args.simulate_buffer:
        print(f"Simulating temporal smoothing over {args.buffer_sample_size} sampled images...")
        smoothing = simulate_temporal_smoothing(
            samples, classifier, extractor, tracker, args.mode,
            n_frames=config.PREDICTION_BUFFER_SIZE,
            min_votes=config.PREDICTION_MIN_VOTES,
            sample_size=args.buffer_sample_size,
            rng=rng,
        )

    if tracker is not None:
        tracker.close()

    print(f"Writing results to {args.output_dir} ...")
    write_outputs(args.output_dir, results, metrics, sweep, smoothing, args.mode)

    print("\n=== SUMMARY ===")
    if args.mode == "pipeline":
        print(f"Detection rate:        {metrics['detection_rate'] * 100:.2f}%")
    print(f"Raw accuracy:          {metrics['raw_accuracy'] * 100:.2f}%")
    print(f"Accuracy @ threshold:  {metrics['accuracy_at_threshold'] * 100:.2f}%  (coverage {metrics['coverage_at_threshold'] * 100:.2f}%)")
    print(f"Mean latency:          {metrics['mean_latency_ms']:.2f} ms  (~{metrics['estimated_fps']:.1f} FPS)")
    if smoothing:
        print(f"Single-frame acc:      {smoothing['single_frame_accuracy'] * 100:.2f}%")
        print(f"Stabilized acc:        {smoothing['stabilized_accuracy_of_resolved'] * 100:.2f}%")
    print(f"\nFull report written to: {args.output_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
