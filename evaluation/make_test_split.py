# -*- coding: utf-8 -*-
"""
make_test_split.py — Carve an independent, untouched test split out of the raw
ISL dataset, BEFORE training. Copies (does not move) a random fraction of each
class's images into a separate folder with the same class-subfolder layout that
both train_mobilenetv2_colab.py and evaluate_system.py expect.

Run this once, right after downloading the Kaggle dataset and before you ever
call train_mobilenetv2_colab.py on it. Keep the test folder somewhere the
trainer never sees, and use a different seed than the trainer's val_split.

Usage:
    python evaluation/make_test_split.py \
        --source /path/to/kagglehub/Indian \
        --dest   /path/to/isl_test_holdout \
        --fraction 0.15 --seed 123
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an independent held-out test split.")
    parser.add_argument("--source", required=True, type=Path, help="Raw dataset folder (class subfolders)")
    parser.add_argument("--dest", required=True, type=Path, help="Where to write the held-out test set")
    parser.add_argument("--fraction", type=float, default=0.15, help="Fraction of each class to hold out (default 0.15)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (use something different from training's val_split seed)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying (removes them from --source so training truly can't see them)")
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Source not found: {args.source}")

    rng = random.Random(args.seed)
    args.dest.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for class_dir in sorted(p for p in args.source.iterdir() if p.is_dir()):
        images = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            continue
        rng.shuffle(images)
        n_holdout = max(1, int(len(images) * args.fraction))
        holdout = images[:n_holdout]

        out_dir = args.dest / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in holdout:
            target = out_dir / img_path.name
            if args.move:
                shutil.move(str(img_path), str(target))
            else:
                shutil.copy2(str(img_path), str(target))
            total_copied += 1

        print(f"{class_dir.name}: {n_holdout}/{len(images)} images -> {out_dir}")

    action = "moved" if args.move else "copied"
    print(f"\nDone. {action} {total_copied} images total to {args.dest}")
    if not args.move:
        print(
            "NOTE: files were only copied, so --source still contains them. "
            "Make sure train_mobilenetv2_colab.py points at a source that EXCLUDES "
            "these held-out images, or re-run with --move."
        )


if __name__ == "__main__":
    main()
