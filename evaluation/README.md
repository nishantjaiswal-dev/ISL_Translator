# System Evaluation Harness

`evaluate_system.py` measures the **deployed pipeline's** accuracy — detection,
cropping, classification, confidence gating, and temporal smoothing — against a
static labeled image folder. No webcam needed. Use this to generate the numbers,
tables, and figures for your paper's Results section.

## 1. First, build a real held-out test set (do this before anything else)

`train_mobilenetv2_colab.py` only ever did a **train/validation** split
(`val_split=0.2`) on one folder — the model's reported `val_accuracy` already
came from images pulled out of that same pool during training. If you point this
evaluation script at that same folder again, you are testing on data the model
selection process already touched. A reviewer will ask about this, so avoid it now.

Two options, in order of rigor:

**Option A — Independent test split (fastest, still valid).**
Re-run `make_test_split.py` (included in this folder) against your original
Kaggle download *before* you ever retrain, using a different `seed`/fraction than
training's `val_split`, and set it aside. Never let the trainer see it.

```bash
python evaluation/make_test_split.py \
    --source /path/to/kagglehub/Indian \
    --dest   /path/to/isl_test_holdout \
    --fraction 0.15 --seed 123
```

**Option B — Independent webcam-captured test set (stronger claim, more work).**
Capture ~15-20 fresh images per class yourself, through the same conditions a
real user would sign in (varied lighting/background), and label them into
`class_name/` folders. This also closes a real gap worth mentioning in your
paper: `isl_model.h5` was trained on clean, isolated Kaggle images, but the app
classifies **MediaPipe-cropped webcam frames** — a different visual distribution
(background clutter, motion blur, arbitrary hand scale/rotation). A test set
built from actual webcam captures measures real-world accuracy, not just held-out
accuracy on the training distribution. `--mode pipeline` is designed for exactly
this: it runs your images through MediaPipe first, same as the live app.

## 2. Run the evaluation

```bash
pip install scikit-learn matplotlib   # if not already installed

# Recommended: full pipeline (detection + crop + classify), matches the live app
python evaluation/evaluate_system.py \
    --test-dir /path/to/isl_test_holdout \
    --mode pipeline \
    --simulate-buffer

# Also worth reporting separately: raw model accuracy, no detection stage
python evaluation/evaluate_system.py \
    --test-dir /path/to/isl_test_holdout \
    --mode direct \
    --output-dir evaluation/results_direct
```

`--simulate-buffer` additionally jitters each test image across 7 simulated
frames and runs them through the real `PredictionBuffer`, so you can report a
single-frame-vs-temporal-smoothing ablation. It re-runs inference per sample, so
it's slower — use `--buffer-sample-size` to cap how many images it covers.

## 3. What you get, and where it maps in a paper

| Output file | Paper use |
|---|---|
| `summary_report.md` | Pull headline numbers + the per-class table straight into your Results section |
| `confusion_matrix.png` | Drop in as a figure |
| `threshold_sweep.png` / `.csv` | Justifies *why* `CONFIDENCE_THRESHOLD = 0.80` was chosen — a coverage-vs-accuracy trade-off curve reviewers like seeing |
| `results.json` | Full machine-readable metrics if you want to compute anything extra |
| `predictions.csv` | Per-image raw log, useful for error analysis / picking failure-case examples to discuss |

Two numbers matter most for your Results section and are easy to conflate —
report both, and say clearly which is which:
- **Raw accuracy** — the classifier's accuracy on every detected frame, no
  confidence gate. This is closest to a "model accuracy" you'd compare against
  other ISL papers.
- **Accuracy at threshold** — accuracy of only the predictions the *app* would
  actually surface to the user (confidence ≥ 0.80). This is the honest
  "what does a user experience" number, and it should be reported alongside its
  **coverage** (what fraction of gestures actually clear the bar) — a system
  that's 99% accurate but only accepts 40% of attempts is a different claim than
  one that's 90% accurate and accepts 95%.
