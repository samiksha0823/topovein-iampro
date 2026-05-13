"""
TopoVein — File 4: Binary Output Quality Validator
====================================================
Task: Validate binary output quality for skeletonization.

Reads ALL binary images from the 'binary/' folder and scores each one
on three metrics, then produces:

  1. Console report with pass/fail counts
  2. quality_report.csv — per-image scores (for Rishu's documentation)
  3. quality_samples/ — a visual grid of GOOD, WARN, and FAIL examples
     so the team can visually inspect preprocessing quality

Metrics used:
  ┌───────────────┬──────────────────────────────────────────────────┐
  │ Metric        │ What it checks                                   │
  ├───────────────┼──────────────────────────────────────────────────┤
  │ vein_ratio    │ % of white pixels (0.02–0.30 expected for veins) │
  │ components    │ # connected regions (1–15 expected)              │
  │ skeleton_len  │ # pixels in skeleton (proxy for vein complexity) │
  └───────────────┴──────────────────────────────────────────────────┘

Usage:
    python 04_validate_quality.py

Prerequisites:
    Run 03_binarize_otsu.py first.

Output:
    quality_report.csv
    quality_samples/good_sample.png
    quality_samples/warn_sample.png
    quality_samples/fail_sample.png
"""

import os
import cv2
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
BINARY_ROOT    = "results/binary"
REPORT_CSV     = "results/quality_report.csv"
SAMPLES_DIR    = "results/quality_samples"

# Quality thresholds — adjust based on FV-USM image characteristics
VEIN_RATIO_MIN  = 0.02   # below this = under-binarized (too few veins)
VEIN_RATIO_MAX  = 0.30   # above this = over-binarized  (background noise)
COMPONENT_MAX   = 30     # above this = too fragmented
SKELETON_MIN    = 100    # below this = too few vein pixels after thinning


# ─────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────

def compute_vein_ratio(binary):
    """Fraction of white pixels = proportion of image classified as vein."""
    return float(np.sum(binary == 255)) / binary.size


def compute_components(binary):
    """
    Count connected white regions (connected components).
    Background (label 0) is excluded.
    Ideal: 1–15 distinct vein network segments.
    Too many components = fragmented binary (noise not removed).
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    # Filter out tiny components (< 20 pixels) — they are noise
    min_size = 20
    valid = sum(1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_size)
    return valid


def compute_skeleton_length(binary):
    """
    Skeletonize the binary and count skeleton pixels.
    This is a fast proxy for vein network complexity.
    Too few skeleton pixels → vein not captured properly.
    """
    bool_img  = binary > 0
    skeleton  = skeletonize(bool_img)
    return int(skeleton.sum())


def score_image(binary):
    """
    Compute all metrics and assign a PASS / WARN / FAIL grade.

    Returns a dict with all metric values and a 'grade' field.
    """
    vein_ratio    = compute_vein_ratio(binary)
    components    = compute_components(binary)
    skeleton_len  = compute_skeleton_length(binary)

    issues = []
    if vein_ratio < VEIN_RATIO_MIN:
        issues.append(f"low_vein_ratio={vein_ratio:.3f}")
    if vein_ratio > VEIN_RATIO_MAX:
        issues.append(f"high_vein_ratio={vein_ratio:.3f}")
    if components > COMPONENT_MAX:
        issues.append(f"fragmented_components={components}")
    if skeleton_len < SKELETON_MIN:
        issues.append(f"short_skeleton={skeleton_len}px")

    if len(issues) == 0:
        grade = "PASS"
    elif len(issues) == 1:
        grade = "WARN"
    else:
        grade = "FAIL"

    return {
        "vein_ratio":   round(vein_ratio, 4),
        "components":   components,
        "skeleton_len": skeleton_len,
        "grade":        grade,
        "issues":       "; ".join(issues),
    }


# ─────────────────────────────────────────
# VISUAL SAMPLE GENERATOR
# ─────────────────────────────────────────

def save_sample_grid(records_by_grade, preprocessed_root="preprocessed"):
    """
    For each grade (PASS, WARN, FAIL), create a visual grid showing:
      - The CLAHE-preprocessed image
      - The binary output
    Side-by-side for up to 6 examples per grade.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    for grade, records in records_by_grade.items():
        if not records:
            continue

        sample    = records[:6]   # max 6 examples per grid
        n         = len(sample)
        fig, axes = plt.subplots(n, 2, figsize=(8, n * 2.5))

        if n == 1:
            axes = [axes]

        fig.suptitle(f"Quality Grade: {grade}  ({len(records)} images total)",
                     fontsize=13, fontweight="bold", y=1.01)

        for row, rec in enumerate(sample):
            binary_path = rec["path"]
            binary_img  = cv2.imread(binary_path, cv2.IMREAD_GRAYSCALE)

            # Try to find matching CLAHE image
            clahe_path = binary_path.replace("binary", "preprocessed") \
                                    .replace("_binary.png", "_clahe.png")
            clahe_img  = cv2.imread(clahe_path, cv2.IMREAD_GRAYSCALE) \
                         if os.path.exists(clahe_path) else None

            ax_clahe, ax_bin = axes[row][0], axes[row][1]

            if clahe_img is not None:
                ax_clahe.imshow(clahe_img, cmap="gray")
                ax_clahe.set_title(f"CLAHE\n{Path(binary_path).name}", fontsize=8)
            else:
                ax_clahe.text(0.5, 0.5, "CLAHE not found",
                              ha="center", va="center", transform=ax_clahe.transAxes)
            ax_clahe.axis("off")

            if binary_img is not None:
                ax_bin.imshow(binary_img, cmap="gray")
                ax_bin.set_title(
                    f"Binary  vein%={rec['vein_ratio']*100:.1f}%\n"
                    f"comp={rec['components']}  skel={rec['skeleton_len']}px",
                    fontsize=8
                )
            else:
                ax_bin.text(0.5, 0.5, "Binary not found",
                            ha="center", va="center", transform=ax_bin.transAxes)
            ax_bin.axis("off")

        plt.tight_layout()
        out_path = os.path.join(SAMPLES_DIR, f"{grade.lower()}_samples.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Sample grid saved → {out_path}")


# ─────────────────────────────────────────
# BATCH VALIDATOR
# ─────────────────────────────────────────

def run_validation(binary_root=BINARY_ROOT):
    pattern   = os.path.join(binary_root, "**", "*_binary.png")
    all_files = sorted(glob.glob(pattern, recursive=True))

    if not all_files:
        print(f"No *_binary.png files found under '{binary_root}'")
        print("Run 03_binarize_otsu.py first.")
        return

    total = len(all_files)

    print("=" * 60)
    print("  TopoVein — File 4: Quality Validation")
    print("=" * 60)
    print(f"  Images to validate : {total}")
    print()

    all_records = []
    by_grade    = {"PASS": [], "WARN": [], "FAIL": []}

    for i, fpath in enumerate(all_files, 1):
        if i % 20 == 0 or i == 1 or i == total:
            print(f"  [{i:>4}/{total}]  {Path(fpath).name}")

        binary = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if binary is None:
            rec = {"path": fpath, "grade": "FAIL", "issues": "imread failed",
                   "vein_ratio": -1, "components": -1, "skeleton_len": -1}
        else:
            metrics = score_image(binary)
            rec     = {"path": fpath, **metrics}

        # Parse subject/session from path parts
        parts = Path(fpath).parts
        rec["session"]    = parts[-3] if len(parts) >= 3 else "?"
        rec["subject_id"] = parts[-2] if len(parts) >= 2 else "?"
        rec["image_name"] = parts[-1]

        all_records.append(rec)
        by_grade[rec["grade"]].append(rec)

    # ── Console report ──
    print()
    print("=" * 60)
    print("  VALIDATION RESULTS")
    print("=" * 60)
    print(f"  PASS : {len(by_grade['PASS']):>5}  ({len(by_grade['PASS'])/total*100:.1f}%)")
    print(f"  WARN : {len(by_grade['WARN']):>5}  ({len(by_grade['WARN'])/total*100:.1f}%)")
    print(f"  FAIL : {len(by_grade['FAIL']):>5}  ({len(by_grade['FAIL'])/total*100:.1f}%)")
    print()

    ok_records = [r for r in all_records if r["vein_ratio"] != -1]
    if ok_records:
        ratios = [r["vein_ratio"] for r in ok_records]
        comps  = [r["components"] for r in ok_records]
        skels  = [r["skeleton_len"] for r in ok_records]
        print("  AGGREGATE METRICS")
        print(f"    Vein ratio  — mean={np.mean(ratios):.3f}  std={np.std(ratios):.3f}  "
              f"min={min(ratios):.3f}  max={max(ratios):.3f}")
        print(f"    Components  — mean={np.mean(comps):.1f}  std={np.std(comps):.1f}  "
              f"min={min(comps)}  max={max(comps)}")
        print(f"    Skeleton px — mean={np.mean(skels):.0f}  std={np.std(skels):.0f}  "
              f"min={min(skels)}  max={max(skels)}")

    # ── FAIL examples ──
    if by_grade["FAIL"]:
        print()
        print(f"  FAIL EXAMPLES (first 5):")
        for rec in by_grade["FAIL"][:5]:
            print(f"    ✗ {rec['image_name']}  issues: {rec['issues']}")

    # ── Save CSV report ──
    fieldnames = ["path", "session", "subject_id", "image_name",
                  "grade", "vein_ratio", "components", "skeleton_len", "issues"]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n  Report saved → {REPORT_CSV}")

    # ── Visual samples ──
    print("\n  Generating visual sample grids...")
    save_sample_grid(by_grade)

    print()
    print("  INTERPRETATION GUIDE")
    print("  ─────────────────────────────────────────────")
    print("  PASS → image is ready for skeletonization (Phase 2)")
    print("  WARN → usable but check quality_samples/warn_samples.png")
    print("         Try adjusting CLAHE clipLimit or Otsu parameters")
    print("  FAIL → recheck binarization parameters for this subject/session")
    print("         Consider using adaptive threshold for these images")
    print()
    print("  Hand the quality_report.csv to Rishu for documentation.")
    print("  PASS images feed into Harsh's Phase 2 graph construction.")

    return all_records, by_grade


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    run_validation()