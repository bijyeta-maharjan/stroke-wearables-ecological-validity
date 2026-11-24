"""
01_compute_measures.py
======================

PURPOSE:
    Load accelerometer data for all subjects and compute the 7 upper limb
    performance measures for both in-lab and real-world contexts.

    This is the first step in the pipeline. All subsequent scripts read the
    output CSV produced here instead of re-loading raw data.

INPUTS:
    For each subject in INCLUDED_SUBJECTS:
        {INLAB_DIR}/{subject_id}/processed.csv    — in-lab accelerometer data
        {RW_DIR}/{subject_id}/combined.csv         — real-world accelerometer data
    clinical_scores.csv                            — clinical assessment scores

OUTPUTS:
    results/tables/all_measures.csv
        One row per subject. Columns:
          subject, FMA, MAL-AS, MAL-HW, WMFT-TIME, WMFT-FAS,
          lab_affected_intensity_avg, lab_unaffected_intensity_avg, ...,
          rw_affected_intensity_avg,  rw_unaffected_intensity_avg,  ...,
          lab_n_epochs, rw_n_epochs

KEY DECISIONS:
    - We use wrist sensors only (not finger sensors) for all measures.
    - Real-world data is NOT filtered here — raw measures are computed.
      Filtering for the threshold sweep is done in 02_threshold_sweep.py.
    - Alpha thresholds (0.003328 g) from the lab's original pipeline are used
      for the use_duration and use_ratio measures.

Author: Bijju
Project: Lab vs Home Arm Use After Stroke
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Import from parent directory ──────────────────────────────────────────────
# We add the repository root to the path so we can import config.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INCLUDED_SUBJECTS, CLINICAL, CLINICAL_COLS,
    MEASURES, TABLES_DIR,
    USE_DURATION_ALPHA, USE_RATIO_ALPHA,
)
from scripts.utils import load_subject_data, compute_measures


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

# Create output directory if it doesn't exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 01: COMPUTE PERFORMANCE MEASURES")
print("=" * 60)
print(f"\nSubjects: {len(INCLUDED_SUBJECTS)}")
print(f"Measures: {len(MEASURES)}")
print(f"Use-duration alpha threshold: {USE_DURATION_ALPHA} g")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CLINICAL SCORES
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading clinical scores...")
clinical_df = pd.read_csv(CLINICAL)
# Strip any accidental whitespace from the subject ID column
clinical_df["SubjID"] = clinical_df["SubjID"].str.strip()
print(f"  Loaded clinical data for {len(clinical_df)} subjects")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP: LOAD DATA AND COMPUTE MEASURES FOR EACH SUBJECT
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading accelerometer data and computing measures...")
print("-" * 60)

rows = []
skipped = []

for subject_id in INCLUDED_SUBJECTS:

    # ── Clinical scores for this subject ──────────────────────────────────────
    # We always include the subject regardless of whether clinical data exists.
    # Missing clinical scores will appear as NaN in the output and are simply
    # excluded from the clinical correlation analysis (step 04) on a per-score basis.
    clin_match = clinical_df[clinical_df["SubjID"] == subject_id]
    if len(clin_match) == 0:
        print(f"  {subject_id}: NOTE — no clinical data found, sensor measures will still be computed")
        clin_row = None
    else:
        clin_row = clin_match.iloc[0]  # convert single-row DataFrame to Series

    # ── Load lab and real-world accelerometer data ────────────────────────────
    try:
        data = load_subject_data(subject_id)
    except FileNotFoundError as e:
        print(f"  {subject_id}: WARNING — {e}, skipping")
        skipped.append(subject_id)
        continue

    print(f"  {subject_id}: lab={data['lab_n_epochs']:,} epochs | "
          f"rw={data['rw_n_epochs']:,} epochs")

    # ── Compute measures for lab data ─────────────────────────────────────────
    # Lab data: structured clinical tasks — no filtering needed
    lab_measures = compute_measures(
        affected_acc    = data["lab_affected"],
        unaffected_acc  = data["lab_unaffected"],
        use_duration_alpha = USE_DURATION_ALPHA,
        use_ratio_alpha    = USE_RATIO_ALPHA,
    )

    # ── Compute measures for real-world data ──────────────────────────────────
    # Real-world data: unfiltered (contains rest + active epochs)
    # Filtering is explored separately in 02_threshold_sweep.py
    rw_measures = compute_measures(
        affected_acc    = data["rw_affected"],
        unaffected_acc  = data["rw_unaffected"],
        use_duration_alpha = USE_DURATION_ALPHA,
        use_ratio_alpha    = USE_RATIO_ALPHA,
    )

    # ── Assemble row for this subject ─────────────────────────────────────────
    row = {"subject": subject_id}

    # Clinical scores — NaN if not available for this subject
    for col in CLINICAL_COLS:
        if clin_row is not None:
            row[col] = clin_row.get(col, np.nan)
        else:
            row[col] = np.nan

    # Lab measures (prefixed with "lab_")
    for measure_name, value in lab_measures.items():
        row[f"lab_{measure_name}"] = value

    # Real-world measures (prefixed with "rw_")
    for measure_name, value in rw_measures.items():
        row[f"rw_{measure_name}"] = value

    # Recording lengths (useful for quality checking)
    row["lab_n_epochs"] = data["lab_n_epochs"]
    row["rw_n_epochs"]  = data["rw_n_epochs"]

    rows.append(row)

# ── Build output DataFrame ────────────────────────────────────────────────────
measures_df = pd.DataFrame(rows)
print("-" * 60)
print(f"\nSuccessfully processed: {len(measures_df)} subjects")
if skipped:
    print(f"Skipped:               {len(skipped)} subjects → {skipped}")


# ══════════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(f"\n{'Measure':<40} {'Lab Mean':>10} {'Lab SD':>10} {'RW Mean':>10} {'RW SD':>10}")
print("-" * 82)

for measure in MEASURES:
    lab_vals = measures_df[f"lab_{measure}"].dropna()
    rw_vals  = measures_df[f"rw_{measure}"].dropna()
    print(f"{measure:<40} {lab_vals.mean():>10.4f} {lab_vals.std():>10.4f} "
          f"{rw_vals.mean():>10.4f} {rw_vals.std():>10.4f}")

# ── Quick sanity check: lab vs RW means ──────────────────────────────────────
# We expect lab intensity to be MUCH higher than RW intensity because:
#   - Lab data is all active (structured tasks)
#   - RW data includes long rest periods (sleep, sitting, etc.)
# This difference motivates the threshold sweep in step 02.
print("\nNote on Lab vs RW intensity difference:")
lab_aff_mean = measures_df["lab_affected_intensity_avg"].mean()
rw_aff_mean  = measures_df["rw_affected_intensity_avg"].mean()
print(f"  Lab affected intensity mean : {lab_aff_mean:.4f} g")
print(f"  RW  affected intensity mean : {rw_aff_mean:.4f} g")
print(f"  Ratio (lab/rw)              : {lab_aff_mean/rw_aff_mean:.1f}x")
print("  → This large difference is expected and motivates the threshold sweep.")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

out_path = TABLES_DIR / "all_measures.csv"
measures_df.to_csv(out_path, index=False)

print("\n" + "=" * 60)
print("OUTPUT SAVED")
print("=" * 60)
print(f"  {out_path}")
print(f"  Shape: {measures_df.shape[0]} rows × {measures_df.shape[1]} columns")
