"""
TopoVein — File 2: CLAHE Preprocessing Pipeline
=================================================
Task: Implement CLAHE preprocessing + Gaussian denoise for entire dataset.

Reads dataset_index.csv produced by 01_explore_dataset.py
Processes EVERY image through:
    Grayscale → ROI Crop → CLAHE → Gaussian Denoise

Saves preprocessed images mirroring the original folder structure
under OUTPUT_ROOT so nothing in the original dataset is modified.

Usage:
    python 02_clahe_pipeline.py

Prerequisites:
    Run 01_explore_dataset.py first to generate dataset_index.csv

Output structure:
    preprocessed/
      1st_session/
        001_1/
          01_clahe.png
          02_clahe.png
          ...
      2nd_session/
        ...
"""

import os
import cv2
import csv
import time
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
CSV_INDEX      = "dataset_index.csv"          # from 01_explore_dataset.py
OUTPUT_ROOT    = "preprocessed"               # where processed images are saved
LOG_FILE       = "02_clahe_log.txt"

# CLAHE parameters (tuned for NIR finger vein images)
CLAHE_CLIP_LIMIT   = 2.5      # 2.0–3.0: higher = more contrast, more noise risk
CLAHE_TILE_SIZE    = (8, 8)   # smaller = more local equalization

# Gaussian blur parameters
BLUR_KERNEL        = (5, 5)   # increase to (7,7) if too much noise remains
BLUR_SIGMA         = 0        # 0 = auto-computed from kernel size

# Skip already-processed files (set False to reprocess everything)
SKIP_EXISTING      = True


# ─────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────

def load_csv_index(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found.\n"
            f"Run 01_explore_dataset.py first."
        )
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row["status"] == "OK"]


def roi_crop(gray):
    """
    Remove the large black border around the finger.
    Threshold at 15 to find any non-black pixel, then crop
    the bounding box with 10px padding.
    """
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    coords  = cv2.findNonZero(mask)
    if coords is None:
        return gray  # image is all black — return as-is

    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)
    return gray[y:y+h, x:x+w]


def apply_clahe(gray_roi):
    """
    CLAHE: Contrast Limited Adaptive Histogram Equalization.
    Divides image into tileGridSize tiles, equalizes each independently.
    clipLimit prevents noise amplification in uniform regions.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE
    )
    return clahe.apply(gray_roi)


def apply_gaussian_denoise(enhanced):
    """
    Mild Gaussian blur to suppress NIR shot noise and speckle.
    Applied AFTER CLAHE so the blur doesn't counteract the enhancement.
    """
    return cv2.GaussianBlur(enhanced, BLUR_KERNEL, sigmaX=BLUR_SIGMA)


def preprocess_one(img_path):
    """
    Full preprocessing chain for one image.
    Returns the denoised image or None on failure.
    """
    raw = cv2.imread(img_path)
    if raw is None:
        return None, "imread failed"

    gray     = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    roi      = roi_crop(gray)
    enhanced = apply_clahe(roi)
    denoised = apply_gaussian_denoise(enhanced)

    return denoised, "OK"


def get_output_path(record, output_root):
    """
    Mirror the original folder structure under output_root.
    e.g.  1st_session/001_1/01.jpg  →  preprocessed/1st_session/001_1/01_clahe.png
    """
    session  = record["session"]
    folder   = f"{record['subject_id']}_{record['finger_id']}"
    stem     = Path(record["image_name"]).stem          # "01"
    out_dir  = Path(output_root) / session / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{stem}_clahe.png")


# ─────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────

def run_batch(csv_path=CSV_INDEX, output_root=OUTPUT_ROOT):
    records = load_csv_index(csv_path)
    total   = len(records)

    print("=" * 60)
    print("  TopoVein — File 2: CLAHE Batch Preprocessing")
    print("=" * 60)
    print(f"  Total images to process : {total}")
    print(f"  Output root             : {output_root}")
    print(f"  CLAHE clipLimit         : {CLAHE_CLIP_LIMIT}")
    print(f"  CLAHE tileGridSize      : {CLAHE_TILE_SIZE}")
    print(f"  Gaussian kernel         : {BLUR_KERNEL}")
    print(f"  Skip existing           : {SKIP_EXISTING}")
    print()

    success = 0
    skipped = 0
    failed  = []
    log_lines = []
    t_start = time.time()

    for i, record in enumerate(records, 1):
        img_path = record["path"]
        out_path = get_output_path(record, output_root)

        # Progress indicator every 10 images
        if i % 10 == 0 or i == 1 or i == total:
            elapsed = time.time() - t_start
            eta     = (elapsed / i) * (total - i) if i > 1 else 0
            print(f"  [{i:>4}/{total}]  ETA {eta:.0f}s  →  {Path(img_path).name}")

        if SKIP_EXISTING and os.path.exists(out_path):
            skipped += 1
            log_lines.append(f"SKIP  {img_path}")
            continue

        result, status = preprocess_one(img_path)

        if result is None:
            failed.append(img_path)
            log_lines.append(f"FAIL  {img_path}  reason={status}")
            continue

        cv2.imwrite(out_path, result)
        success += 1
        log_lines.append(f"OK    {out_path}")

    elapsed = time.time() - t_start

    # Write log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Final summary
    print()
    print("=" * 60)
    print("  BATCH COMPLETE")
    print("=" * 60)
    print(f"  Processed successfully : {success}")
    print(f"  Skipped (existing)     : {skipped}")
    print(f"  Failed                 : {len(failed)}")
    print(f"  Time elapsed           : {elapsed:.1f}s")
    print(f"  Log saved              : {LOG_FILE}")
    print(f"  Output folder          : {output_root}/")
    print()

    if failed:
        print("  FAILED FILES:")
        for f in failed:
            print(f"    ✗ {f}")

    return success, failed


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    run_batch()