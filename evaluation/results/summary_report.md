# ISL Translator — System Evaluation Report (pipeline mode)

- Total test images: **6410**
- Hand-detection rate (MediaPipe found a hand): **99.78%** (6396/6410)
- Raw classification accuracy (all detected images, no confidence gate): **95.54%**
- Confidence threshold evaluated: **0.8**
- Coverage at threshold (fraction of predictions the app would accept): **98.45%**
- Accuracy of accepted predictions (what the user actually sees): **96.35%**
- Mean inference latency: **122.07 ms** (p95: 213.77 ms)
- Estimated throughput: **8.2 FPS** (classifier inference only, excludes capture/detection I/O)

## Temporal smoothing ablation
- Sampled 150 images, each replayed across 7 jittered simulated frames
- Single-frame accuracy (no smoothing): **94.00%**
- Stabilized accuracy after 4-of-7 majority vote (of gestures that resolved to a stable label): **97.32%**
- Buffer resolve rate (fraction reaching any stable vote): **99.33%**

## Per-class metrics

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 | 180 |
| 2 | 1.000 | 1.000 | 1.000 | 180 |
| 3 | 1.000 | 1.000 | 1.000 | 180 |
| 4 | 0.994 | 1.000 | 0.997 | 180 |
| 5 | 1.000 | 1.000 | 1.000 | 180 |
| 6 | 1.000 | 1.000 | 1.000 | 180 |
| 7 | 1.000 | 1.000 | 1.000 | 180 |
| 8 | 1.000 | 1.000 | 1.000 | 180 |
| 9 | 1.000 | 1.000 | 1.000 | 180 |
| A | 1.000 | 1.000 | 1.000 | 180 |
| B | 0.933 | 1.000 | 0.965 | 180 |
| C | 0.633 | 1.000 | 0.775 | 217 |
| D | 1.000 | 1.000 | 1.000 | 180 |
| E | 1.000 | 1.000 | 1.000 | 180 |
| F | 0.845 | 1.000 | 0.916 | 180 |
| G | 0.964 | 0.911 | 0.937 | 179 |
| H | 0.978 | 1.000 | 0.989 | 180 |
| I | 0.995 | 0.995 | 0.995 | 205 |
| J | 0.984 | 1.000 | 0.992 | 180 |
| K | 1.000 | 1.000 | 1.000 | 180 |
| L | 0.994 | 1.000 | 0.997 | 180 |
| M | 1.000 | 1.000 | 1.000 | 180 |
| N | 1.000 | 0.400 | 0.571 | 180 |
| O | 0.986 | 0.981 | 0.984 | 214 |
| P | 1.000 | 0.989 | 0.994 | 180 |
| Q | 1.000 | 0.918 | 0.957 | 171 |
| R | 1.000 | 1.000 | 1.000 | 180 |
| S | 0.915 | 0.961 | 0.938 | 180 |
| T | 1.000 | 1.000 | 1.000 | 180 |
| U | 1.000 | 1.000 | 1.000 | 180 |
| V | 0.704 | 0.958 | 0.812 | 191 |
| W | 1.000 | 1.000 | 1.000 | 179 |
| X | 1.000 | 1.000 | 1.000 | 180 |
| Y | 0.982 | 0.306 | 0.466 | 180 |
| Z | 1.000 | 1.000 | 1.000 | 180 |
| **Macro avg** | 0.969 | 0.955 | 0.951 | 6396 |
| **Weighted avg** | 0.967 | 0.955 | 0.950 | 6396 |

## Confidence threshold sweep

| Threshold | Coverage | Accuracy (accepted) |
|---|---|---|
| 0.50 | 99.84% | 95.65% |
| 0.55 | 99.58% | 95.76% |
| 0.60 | 99.44% | 95.82% |
| 0.65 | 99.25% | 95.92% |
| 0.70 | 99.02% | 96.02% |
| 0.75 | 98.72% | 96.18% |
| 0.80 | 98.45% | 96.35% |
| 0.85 | 98.03% | 96.54% |
| 0.90 | 97.65% | 96.72% |
| 0.95 | 96.95% | 96.95% |