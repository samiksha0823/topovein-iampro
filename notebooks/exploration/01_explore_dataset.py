"""
TopoVein — File 1: Dataset Explorer
=====================================
Task: Dataset exploration — understand FV-USM folder structure.

It scans the entire FV-USM dataset and prints:
  - How many subjects exist
  - How many sessions
  - How many fingers per subject
  - How many images per finger
  - Image sizes and any corrupted files

Usage:
    python 01_explore_dataset.py

Output:
    dataset_report.txt  — full text report
    dataset_index.csv   — one row per image (for batch processing)
"""

import os
import cv2
import csv
import json
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────
# CONFIGURE THIS — point to our dataset root
# ─────────────────────────────────────────
DATASET_ROOT = r"E:\topoVein-Iampro\Published_database_FV-USM_Dec2013\Published_database_FV-USM_Dec2013"
OUTPUT_REPORT = "dataset_report.txt"
OUTPUT_CSV    = "dataset_index.csv"
OUTPUT_JSON   = "dataset_stats.json"


def explore_dataset(root):
    """
    Walk the FV-USM folder tree and collect metadata for every image.

    FV-USM structure:
        <root>/
          <session>/          e.g. 1st_session, 2nd_session
            raw_data/
              <subject_finger>/   e.g. 001_1, 001_2, 002_1 ...
                01.jpg ... 06.jpg
    """
    records   = []   # one dict per image
    errors    = []   # corrupted or unreadable files
    stats     = defaultdict(set)

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found:\n  {root}")

    print(f"\nScanning: {root}\n{'─'*55}")

    for session_dir in sorted(root_path.iterdir()):
        if not session_dir.is_dir():
            continue
        session_name = session_dir.name

        raw_data_dir = session_dir / "raw_data"
        if not raw_data_dir.exists():
            # try direct children
            raw_data_dir = session_dir

        for subject_finger_dir in sorted(raw_data_dir.iterdir()):
            if not subject_finger_dir.is_dir():
                continue

            folder_name = subject_finger_dir.name  # e.g. "001_1"

            # Parse subject ID and finger ID from folder name
            parts = folder_name.split("_")
            if len(parts) >= 2:
                subject_id = parts[0]           # "001"
                finger_id  = parts[1]           # "1"
            else:
                subject_id = folder_name
                finger_id  = "?"

            stats["sessions"].add(session_name)
            stats["subjects"].add(subject_id)
            stats["fingers"].add(f"{subject_id}_{finger_id}")

            image_files = sorted(subject_finger_dir.glob("*.jpg")) + \
                          sorted(subject_finger_dir.glob("*.png")) + \
                          sorted(subject_finger_dir.glob("*.bmp"))

            for img_path in image_files:
                img = cv2.imread(str(img_path))

                if img is None:
                    errors.append(str(img_path))
                    records.append({
                        "path":       str(img_path),
                        "session":    session_name,
                        "subject_id": subject_id,
                        "finger_id":  finger_id,
                        "image_name": img_path.name,
                        "height":     -1,
                        "width":      -1,
                        "channels":   -1,
                        "size_bytes": img_path.stat().st_size,
                        "status":     "CORRUPTED",
                    })
                    continue

                h, w = img.shape[:2]
                ch   = img.shape[2] if len(img.shape) == 3 else 1

                stats["heights"].add(h)
                stats["widths"].add(w)
                stats["total_images"] = stats.get("total_images", 0) + 1

                records.append({
                    "path":       str(img_path),
                    "session":    session_name,
                    "subject_id": subject_id,
                    "finger_id":  finger_id,
                    "image_name": img_path.name,
                    "height":     h,
                    "width":      w,
                    "channels":   ch,
                    "size_bytes": img_path.stat().st_size,
                    "status":     "OK",
                })

    return records, errors, stats


def print_report(records, errors, stats):
    lines = []
    lines.append("=" * 60)
    lines.append("  TopoVein — FV-USM Dataset Exploration Report")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    ok_records = [r for r in records if r["status"] == "OK"]
    lines.append("SUMMARY")
    lines.append("─" * 40)
    lines.append(f"  Total sessions  : {len(stats['sessions'])}")
    lines.append(f"  Total subjects  : {len(stats['subjects'])}")
    lines.append(f"  Unique fingers  : {len(stats['fingers'])}")
    lines.append(f"  Total images    : {len(ok_records)}")
    lines.append(f"  Corrupted files : {len(errors)}")
    lines.append("")

    # Image dimensions
    lines.append("IMAGE DIMENSIONS")
    lines.append("─" * 40)
    if ok_records:
        heights = [r["height"] for r in ok_records]
        widths  = [r["width"]  for r in ok_records]
        lines.append(f"  Height range : {min(heights)} – {max(heights)} px")
        lines.append(f"  Width  range : {min(widths)}  – {max(widths)} px")
        if len(set(heights)) == 1 and len(set(widths)) == 1:
            lines.append(f"  All images   : SAME SIZE ✓  ({widths[0]} × {heights[0]})")
        else:
            lines.append(f"  WARNING: images have different sizes — resize needed before batch processing")
    lines.append("")

    # Per-session breakdown
    lines.append("PER-SESSION BREAKDOWN")
    lines.append("─" * 40)
    sessions = sorted(set(r["session"] for r in ok_records))
    for s in sessions:
        sess_imgs = [r for r in ok_records if r["session"] == s]
        subjs     = set(r["subject_id"] for r in sess_imgs)
        lines.append(f"  {s:<20} : {len(sess_imgs):>4} images  |  {len(subjs)} subjects")
    lines.append("")

    # Per-subject sample
    lines.append("PER-SUBJECT SAMPLE (first 10 subjects)")
    lines.append("─" * 40)
    subjects = sorted(set(r["subject_id"] for r in ok_records))[:10]
    for subj in subjects:
        subj_imgs   = [r for r in ok_records if r["subject_id"] == subj]
        fingers     = set(r["finger_id"] for r in subj_imgs)
        sessions_s  = set(r["session"]   for r in subj_imgs)
        lines.append(f"  Subject {subj} : {len(subj_imgs):>3} images | fingers={sorted(fingers)} | sessions={len(sessions_s)}")
    lines.append("")

    # Errors
    if errors:
        lines.append("CORRUPTED / UNREADABLE FILES")
        lines.append("─" * 40)
        for e in errors:
            lines.append(f"  ✗ {e}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("Files saved:")
    lines.append(f"  {OUTPUT_CSV}   — full image index for batch processing")
    lines.append(f"  {OUTPUT_JSON}  — stats summary")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved → {OUTPUT_REPORT}")

    return report_text


def save_csv_index(records):
    """Save full image index as CSV — used by preprocessing scripts."""
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV index saved → {OUTPUT_CSV}  ({len(records)} rows)")


def save_json_stats(stats):
    """Save stats as JSON for later use."""
    serializable = {k: list(v) if isinstance(v, set) else v
                    for k, v in stats.items()}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"JSON stats saved → {OUTPUT_JSON}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    records, errors, stats = explore_dataset(DATASET_ROOT)

    if not records:
        print("No images found. Check DATASET_ROOT path.")
    else:
        print_report(records, errors, stats)
        save_csv_index(records)
        save_json_stats(stats)