"""
TopoVein — File 3: Otsu Binarization Pipeline
===============================================
Task: Gaussian denoise + Otsu binarization pipeline.

Reads CLAHE-preprocessed images from the 'preprocessed/' folder
(produced by 02_clahe_pipeline.py) and applies:

    CLAHE output → Otsu Binarization → Morphological Cleanup → Binary image

Why Otsu for this task?
    The assignment specifically asks for Otsu binarization. Otsu's method
    finds the single global threshold that maximises the inter-class variance
    between background and vein pixels. It works well when CLAHE has already
    normalized the histogram to have a clear bimodal distribution (one peak
    for tissue, one for veins).

    Note: If results are poor (veins broken up), switch to
    ADAPTIVE_THRESH_GAUSSIAN_C in the binarize() function — this handles
    spatially uneven illumination better for some finger images.

Usage:
    python 03_binarize_otsu.py

Prerequisites:
    Run 02_clahe_pipeline.py first.

Output:
    binary/
      1st_session/
        001_1/
          01_binary.png
          ...
"""

import os
import cv2
import glob
import time
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PREPROCESSED_ROOT = "preprocessed"   # output of 02_clahe_pipeline.py
BINARY_ROOT       = "binary"         # where binary images are saved
LOG_FILE          = "03_binarize_log.txt"

# Morphological cleanup kernel size
MORPH_KERNEL_SIZE = (3, 3)

SKIP_EXISTING = True


# ─────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────

def otsu_binarize(denoised):
    """
    Otsu's automatic global thresholding.

    cv2.threshold with THRESH_OTSU automatically finds the threshold T
    that minimises intra-class pixel variance. Returns a binary image
    where veins = 255 (white) and background = 0 (black).

    THRESH_BINARY_INV is used because in NIR images, veins absorb IR
    light and appear DARKER than surrounding tissue. Inverting makes
    veins white, which is the convention for skeletonization input.

    The returned `otsu_thresh` value is printed for quality analysis.
    """
    otsu_thresh, binary = cv2.threshold(
        denoised,
        0,                          # ignored when THRESH_OTSU is set
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary, otsu_thresh


def morphological_cleanup(binary):
    """
    Remove noise artefacts from the binary vein mask:

    1. Opening (erode then dilate):
       Removes small isolated white blobs (shot noise, skin pores)
       that are thinner than the structuring element.

    2. Closing (dilate then erode):
       Fills small black holes inside thick vein segments so the
       vein mask is solid — important for clean skeletonization.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def compute_quality_metrics(binary):
    """
    Quick quality checks on the binary image before skeletonization:

    vein_ratio   : fraction of white pixels — should be 0.03–0.20
                   (too low = missed veins, too high = over-binarized)
    connectivity : number of connected components — expect 1–10
                   (too many = fragmented, noisy binary)
    """
    total_px   = binary.size
    white_px   = np.sum(binary == 255)
    vein_ratio = white_px / total_px

    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary)
    connectivity = num_labels - 1  # subtract background label

    quality = "GOOD"
    if vein_ratio < 0.02:
        quality = "WARN: very few vein pixels — possible under-binarization"
    elif vein_ratio > 0.35:
        quality = "WARN: too many white pixels — possible over-binarization"
    elif connectivity > 50:
        quality = "WARN: highly fragmented — morphological cleanup may help"

    return {
        "vein_ratio":    round(vein_ratio, 4),
        "white_px":      int(white_px),
        "total_px":      int(total_px),
        "components":    connectivity,
        "quality":       quality,
    }


def binarize_one(clahe_img_path):
    """
    Full binarization chain for one preprocessed image.
    Returns (binary_image, otsu_threshold, quality_metrics) or (None, -1, {}).
    """
    img = cv2.imread(clahe_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, -1, {}

    binary, thresh = otsu_binarize(img)
    cleaned        = morphological_cleanup(binary)
    metrics        = compute_quality_metrics(cleaned)

    return cleaned, thresh, metrics


def get_output_path(clahe_path, binary_root):
    """
    preprocessed/1st_session/001_1/01_clahe.png
    → binary/1st_session/001_1/01_binary.png
    """
    p        = Path(clahe_path)
    # Relative path under preprocessed root
    parts    = p.parts
    try:
        pre_idx = next(i for i, pt in enumerate(parts) if pt == "preprocessed")
        rel     = Path(*parts[pre_idx+1:])
    except StopIteration:
        rel = p.name

    stem    = p.stem.replace("_clahe", "")
    out_dir = Path(binary_root) / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{stem}_binary.png")


# ─────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────

def run_batch(preprocessed_root=PREPROCESSED_ROOT, binary_root=BINARY_ROOT):
    # Find all CLAHE images
    pattern = os.path.join(preprocessed_root, "**", "*_clahe.png")
    all_files = sorted(glob.glob(pattern, recursive=True))

    if not all_files:
        print(f"No *_clahe.png files found under '{preprocessed_root}'")
        print("Run 02_clahe_pipeline.py first.")
        return

    total = len(all_files)

    print("=" * 60)
    print("  TopoVein — File 3: Otsu Binarization Batch")
    print("=" * 60)
    print(f"  Input images   : {total}")
    print(f"  Method         : Otsu global threshold + morph cleanup")
    print(f"  Output root    : {binary_root}/")
    print()

    success       = 0
    skipped       = 0
    failed        = []
    warn_quality  = []
    log_lines     = []
    otsu_values   = []
    t_start       = time.time()

    for i, clahe_path in enumerate(all_files, 1):
        out_path = get_output_path(clahe_path, binary_root)

        if i % 10 == 0 or i == 1 or i == total:
            elapsed = time.time() - t_start
            eta     = (elapsed / i) * (total - i) if i > 1 else 0
            print(f"  [{i:>4}/{total}]  ETA {eta:.0f}s  →  {Path(clahe_path).name}")

        if SKIP_EXISTING and os.path.exists(out_path):
            skipped += 1
            log_lines.append(f"SKIP  {clahe_path}")
            continue

        binary, thresh, metrics = binarize_one(clahe_path)

        if binary is None:
            failed.append(clahe_path)
            log_lines.append(f"FAIL  {clahe_path}")
            continue

        cv2.imwrite(out_path, binary)
        otsu_values.append(thresh)
        success += 1

        quality_flag = metrics.get("quality", "")
        if "WARN" in quality_flag:
            warn_quality.append((clahe_path, metrics))

        log_lines.append(
            f"OK  otsu={thresh:>3}  vein%={metrics['vein_ratio']*100:.1f}%  "
            f"comp={metrics['components']:>3}  [{quality_flag}]  {out_path}"
        )

    elapsed = time.time() - t_start

    # Summary stats for Otsu thresholds
    print()
    print("=" * 60)
    print("  BATCH COMPLETE")
    print("=" * 60)
    print(f"  Processed    : {success}")
    print(f"  Skipped      : {skipped}")
    print(f"  Failed       : {len(failed)}")
    print(f"  Quality warns: {len(warn_quality)}")
    print(f"  Time elapsed : {elapsed:.1f}s")

    if otsu_values:
        print()
        print("  OTSU THRESHOLD STATISTICS")
        print(f"    Mean   : {np.mean(otsu_values):.1f}")
        print(f"    Std    : {np.std(otsu_values):.1f}")
        print(f"    Min    : {min(otsu_values)}")
        print(f"    Max    : {max(otsu_values)}")
        print("    (Large std dev = inconsistent illumination across dataset)")

    if warn_quality:
        print()
        print("  QUALITY WARNINGS (first 10):")
        for path, m in warn_quality[:10]:
            print(f"    ✗ {Path(path).name}  {m['quality']}")
            print(f"      vein_ratio={m['vein_ratio']}  components={m['components']}")

    # Save log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n  Log saved → {LOG_FILE}")

    return success, warn_quality


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    run_batch()