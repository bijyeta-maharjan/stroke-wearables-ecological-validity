"""
02_threshold_sweep.py
=====================

PURPOSE:
    Find the activity threshold that makes lab and real-world intensity means
    comparable while preserving strong correlation between them.

THE PROBLEM:
    In-lab data is recorded during structured, continuous upper limb tasks.
    Participants are always active — there are no rest or non-wear periods.

    Real-world data is recorded over multiple days. It includes:
      - Active periods (functional tasks, reaching, carrying)
      - Rest periods (sleeping, watching TV, sitting at a desk)
      - Possible non-wear periods (sensor removed, charging)

    Because of all this inactive time, the RAW real-world mean acceleration
    is much lower than the lab mean — even for the same participant doing the
    same types of movements. This makes the two contexts non-comparable.

THE SOLUTION:
    Filter out inactive epochs from the real-world data before comparing
    intensities. An epoch is considered "inactive" if BOTH wrists are below
    an activity threshold T.

    We sweep T from SWEEP_START to SWEEP_STOP and at each step:
      1. Filter real-world data: keep epochs where (affected > T) OR (unaffected > T)
      2. Compute affected_intensity_avg for filtered RW data
      3. Run a paired t-test: lab mean vs filtered RW mean (across 18 subjects)
      4. Compute Pearson correlation: lab vs filtered RW
      5. Record t-test p-value and R²

    The OPTIMAL threshold satisfies:
      - T-test p >= 0.05 (means are statistically comparable — no significant bias)
      - R² is high (subjects still rank similarly in lab and real-world)

INPUTS:
    results/tables/all_measures.csv  — but we re-load raw data here because
                                        we need to apply the threshold filter
                                        to individual epochs, not to means

OUTPUTS:
    results/tables/threshold_sweep_results.csv
        Columns: threshold, pearson_r, r_squared, p_correlation, t_stat, p_ttest, means_comparable

Author: Bijju
Project: AHHA Lab – Stroke Rehabilitation Ecological Validity Study
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    INCLUDED_SUBJECTS,
    SWEEP_START, SWEEP_STOP, SWEEP_STEP,
    USE_DURATION_ALPHA, USE_RATIO_ALPHA,
    ALPHA_SIGNIFICANCE, TABLES_DIR, FIGURES_DIR,
)
from scripts.utils import load_subject_data, compute_measures, filter_inactive_epochs


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 02: THRESHOLD SWEEP")
print("=" * 60)
print(f"\nSweep range: {SWEEP_START}g to {SWEEP_STOP}g in steps of {SWEEP_STEP}g")
print(f"Target: p(t-test) >= {ALPHA_SIGNIFICANCE} AND high R²")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD RAW ACCELEROMETER DATA FOR ALL SUBJECTS
# ══════════════════════════════════════════════════════════════════════════════
# We need epoch-level data (not the pre-computed means) so that we can apply
# different threshold filters at each sweep step.

print("\nLoading raw accelerometer data for all subjects...")
subject_data = {}   # { subject_id: { "lab_affected", "lab_unaffected", ... } }

for subject_id in INCLUDED_SUBJECTS:
    try:
        data = load_subject_data(subject_id)
        subject_data[subject_id] = data
        print(f"  {subject_id}: OK")
    except FileNotFoundError as e:
        print(f"  {subject_id}: SKIP — {e}")

subjects = list(subject_data.keys())
print(f"\nLoaded {len(subjects)} subjects for sweep")


# ══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE LAB AFFECTED INTENSITY (does not change with threshold)
# ══════════════════════════════════════════════════════════════════════════════
# Lab data is never filtered — it doesn't have inactive periods.
# So the lab intensity is the same at every sweep step; compute it once.

lab_aff_intensities = np.array([
    np.mean(subject_data[s]["lab_affected"])
    for s in subjects
])

print(f"\nLab affected intensity (fixed): mean = {lab_aff_intensities.mean():.4f} g")


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════════════════════

# Build the array of thresholds to test
thresholds = np.arange(SWEEP_START, SWEEP_STOP + SWEEP_STEP / 2, SWEEP_STEP)

print(f"\nRunning sweep over {len(thresholds)} thresholds...")
print("-" * 60)

sweep_results = []

for T in thresholds:

    # For each subject: filter RW data at this threshold and compute affected intensity
    rw_aff_filtered_means = []

    for s in subjects:
        aff_rw   = subject_data[s]["rw_affected"]
        unaff_rw = subject_data[s]["rw_unaffected"]

        # Filter: keep epochs where EITHER wrist is above threshold T
        # This removes rest/sleep epochs while keeping asymmetric movement
        filtered = filter_inactive_epochs(aff_rw, unaff_rw, threshold=T)

        # Compute affected intensity on the filtered data
        aff_filtered = filtered["affected_filtered"]

        if len(aff_filtered) < 10:
            # Too few epochs after filtering — treat as missing
            rw_aff_filtered_means.append(np.nan)
        else:
            rw_aff_filtered_means.append(np.mean(aff_filtered))

    rw_aff_filtered_means = np.array(rw_aff_filtered_means)

    # Only use subjects where both lab and filtered RW values are available
    valid_mask = ~np.isnan(rw_aff_filtered_means)
    n_valid = valid_mask.sum()

    if n_valid < 5:
        # Too few subjects — skip this threshold
        continue

    lab_vals = lab_aff_intensities[valid_mask]
    rw_vals  = rw_aff_filtered_means[valid_mask]

    # ── Pearson correlation: do subjects rank the same in lab and RW? ─────────
    r_corr, p_corr = pearsonr(lab_vals, rw_vals)
    r_squared = r_corr ** 2

    # ── Paired t-test: are the lab and RW means statistically the same? ───────
    # We WANT this to be non-significant — it means the filter has made
    # the two contexts comparable (no systematic bias in the means).
    t_stat, p_ttest = ttest_rel(lab_vals, rw_vals)

    means_comparable = p_ttest >= ALPHA_SIGNIFICANCE  # True if we want to keep this T

    sweep_results.append({
        "threshold":          round(T, 6),
        "n_subjects":         n_valid,
        "pearson_r":          round(r_corr, 4),
        "r_squared":          round(r_squared, 4),
        "p_correlation":      round(p_corr, 4),
        "t_stat":             round(t_stat, 4),
        "p_ttest":            round(p_ttest, 4),
        "means_comparable":   means_comparable,  # True = means not significantly different
        "lab_mean":           round(lab_vals.mean(), 5),
        "rw_filtered_mean":   round(rw_vals.mean(), 5),
    })

sweep_df = pd.DataFrame(sweep_results)
print(f"Sweep complete. {len(sweep_df)} threshold values evaluated.")


# ══════════════════════════════════════════════════════════════════════════════
# IDENTIFY OPTIMAL THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════
# Optimal = means are comparable (t-test NS) AND correlation is maximised

comparable_df = sweep_df[sweep_df["means_comparable"]]

if len(comparable_df) == 0:
    print("\nWARNING: No threshold found where t-test is non-significant!")
    print("         Reporting threshold with highest R² overall.")
    optimal_row = sweep_df.loc[sweep_df["r_squared"].idxmax()]
else:
    # Among thresholds where means are comparable, pick the one with highest R²
    optimal_row = comparable_df.loc[comparable_df["r_squared"].idxmax()]

T_opt = optimal_row["threshold"]


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SWEEP RESULTS (showing every 5th threshold)")
print("=" * 60)
print(f"\n{'Threshold (g)':>14} {'R²':>8} {'p(corr)':>10} {'p(t-test)':>10} {'Comparable':>12}")
print("-" * 58)

for _, row in sweep_df.iloc[::5].iterrows():
    flag = " ← comparable" if row["means_comparable"] else ""
    print(f"{row['threshold']:>14.4f} {row['r_squared']:>8.3f} "
          f"{row['p_correlation']:>10.4f} {row['p_ttest']:>10.4f} "
          f"{'YES' if row['means_comparable'] else 'no':>12}{flag}")

print("\n" + "=" * 60)
print("OPTIMAL THRESHOLD")
print("=" * 60)
print(f"""
  Threshold T = {T_opt:.4f} g  ({T_opt*1000:.2f} mg)

  At this threshold:
    R²          = {optimal_row['r_squared']:.3f}
    p(corr)     = {optimal_row['p_correlation']:.4f}
    t-stat      = {optimal_row['t_stat']:.3f}
    p(t-test)   = {optimal_row['p_ttest']:.4f}
    Lab mean    = {optimal_row['lab_mean']:.5f} g
    RW mean     = {optimal_row['rw_filtered_mean']:.5f} g

  Interpretation:
    The paired t-test is {'non-significant' if optimal_row['means_comparable'] else 'STILL SIGNIFICANT'} —
    lab and filtered RW means are {'statistically comparable' if optimal_row['means_comparable'] else 'NOT comparable'}.
    Pearson R² = {optimal_row['r_squared']:.3f}, meaning subjects retain their relative
    ranking across contexts.
""")


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE PER-SUBJECT FILTERING STATISTICS AT OPTIMAL THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════
# Now that we know the optimal threshold, compute how much data was removed
# from each subject's real-world recording at that threshold.

print("\nComputing per-subject filtering statistics at optimal threshold...")
filter_stats = []

for s in subjects:
    aff_rw   = subject_data[s]["rw_affected"]
    unaff_rw = subject_data[s]["rw_unaffected"]
    filtered = filter_inactive_epochs(aff_rw, unaff_rw, threshold=T_opt)
    filter_stats.append({
        "subject":     s,
        "n_original":  filtered["n_original"],
        "n_filtered":  filtered["n_filtered"],
        "n_removed":   filtered["n_removed"],
        "pct_removed": filtered["pct_removed"],
    })
    print(f"  {s}: {filtered['pct_removed']:.1f}% removed "
          f"({filtered['n_removed']:,} / {filtered['n_original']:,} epochs)")

filter_df = pd.DataFrame(filter_stats)
print(f"\n  Mean % removed: {filter_df['pct_removed'].mean():.1f}%")
print(f"  Range: {filter_df['pct_removed'].min():.1f}% – {filter_df['pct_removed'].max():.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE: THRESHOLD SWEEP + FILTERING STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
# Two-panel figure:
#   Panel A (top): The sweep curve showing R² and t-test p-value vs threshold.
#                  Shaded region = thresholds where means are statistically
#                  comparable (t-test NS). Optimal threshold marked.
#   Panel B (bottom): Per-subject bar chart of % real-world data removed at
#                     the optimal threshold, sorted by % removed.

print("\nGenerating threshold sweep + filtering statistics figure...")

fig = plt.figure(figsize=(12, 9))
gs  = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.45)

ax_sweep = fig.add_subplot(gs[0])
ax_filt  = fig.add_subplot(gs[1])

# ── Panel A: Sweep curve ──────────────────────────────────────────────────────
t_arr  = sweep_df["threshold"].values
r2_arr = sweep_df["r_squared"].values
p_arr  = sweep_df["p_ttest"].values

# R² line
ax_sweep.plot(t_arr * 1000, r2_arr,
              color="#4878CF", lw=2, label="Pearson R² (lab vs filtered RW)")

# t-test p-value line (secondary axis)
ax2 = ax_sweep.twinx()
ax2.plot(t_arr * 1000, p_arr,
         color="#F5A623", lw=1.5, ls="--", alpha=0.8,
         label="Paired t-test p-value")
ax2.axhline(ALPHA_SIGNIFICANCE, color="#F5A623", lw=1, ls=":",
            alpha=0.6, label=f"α = {ALPHA_SIGNIFICANCE}")
ax2.set_ylabel("Paired t-test p-value", color="#F5A623", fontsize=9)
ax2.tick_params(axis="y", labelcolor="#F5A623", labelsize=8)
ax2.set_ylim(bottom=0)

# Shade the "comparable" region (t-test NS)
comparable_mask = sweep_df["means_comparable"].values
for i in range(len(t_arr) - 1):
    if comparable_mask[i]:
        ax_sweep.axvspan(t_arr[i] * 1000, t_arr[i + 1] * 1000,
                         alpha=0.12, color="#5BAD6F", zorder=0)

# Mark optimal threshold
ax_sweep.axvline(T_opt * 1000, color="#D65F5F", lw=2, ls="--",
                 label=f"Optimal T = {T_opt*1000:.2f} mg  (R² = {optimal_row['r_squared']:.3f})")

ax_sweep.set_xlabel("Activity Threshold T (mg)", fontsize=10)
ax_sweep.set_ylabel("Pearson R²", color="#4878CF", fontsize=10)
ax_sweep.tick_params(axis="y", labelcolor="#4878CF", labelsize=8)
ax_sweep.set_title(
    "A.  Threshold Sweep: R² and t-test p-value vs Activity Threshold\n"
    "    Green shading = thresholds where lab and real-world means are statistically comparable (t-test p ≥ 0.05)",
    fontsize=10, fontweight="bold", pad=8
)
ax_sweep.set_ylim(0, 1)

# Combined legend from both axes
lines1, labels1 = ax_sweep.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_sweep.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
ax_sweep.spines[["top"]].set_visible(False)
ax2.spines[["top"]].set_visible(False)

# ── Panel B: Per-subject % removed ───────────────────────────────────────────
# Sort by % removed so it's easy to see which subjects had most inactive time
filter_sorted = filter_df.sort_values("pct_removed", ascending=False).reset_index(drop=True)

# Colour bars by % removed: dark = more filtered, light = less filtered
colors = plt.cm.YlOrRd(filter_sorted["pct_removed"].values / 100)

bars = ax_filt.bar(
    range(len(filter_sorted)),
    filter_sorted["pct_removed"],
    color=colors, edgecolor="white", linewidth=0.5, zorder=3
)

# Add % label on each bar
for idx, (bar, pct) in enumerate(zip(bars, filter_sorted["pct_removed"])):
    ax_filt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{pct:.0f}%",
                 ha="center", va="bottom", fontsize=7.5, color="#333333")

# Mean line
mean_pct = filter_df["pct_removed"].mean()
ax_filt.axhline(mean_pct, color="#4878CF", lw=1.5, ls="--",
                label=f"Mean = {mean_pct:.1f}%", zorder=4)

ax_filt.set_xticks(range(len(filter_sorted)))
ax_filt.set_xticklabels(filter_sorted["subject"], rotation=45, ha="right", fontsize=8)
ax_filt.set_ylabel("Real-World Data Removed (%)", fontsize=10)
ax_filt.set_title(
    f"B.  Per-Subject Filtering at Optimal Threshold (T = {T_opt*1000:.2f} mg)\n"
    "    Epochs removed = both wrists below threshold (inactive / non-wear)",
    fontsize=10, fontweight="bold", pad=8
)
ax_filt.set_ylim(0, max(filter_sorted["pct_removed"]) * 1.15)
ax_filt.legend(fontsize=9)
ax_filt.grid(axis="y", alpha=0.3, linewidth=0.5)
ax_filt.spines[["top", "right"]].set_visible(False)

fig.suptitle(
    "Threshold Sweep Analysis — Finding Optimal Activity Threshold\n"
    "for Filtering Real-World Accelerometer Data",
    fontsize=12, fontweight="bold", y=1.01
)

fig_path = FIGURES_DIR / "fig_threshold_sweep_and_filtering.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

out_path = TABLES_DIR / "threshold_sweep_results.csv"
sweep_df.to_csv(out_path, index=False)

filter_stats_path = TABLES_DIR / "filtering_statistics.csv"
filter_df.to_csv(filter_stats_path, index=False)

print("\n" + "=" * 60)
print("OUTPUTS SAVED")
print("=" * 60)
print(f"  {out_path}")
print(f"  {filter_stats_path}")
print(f"  {fig_path}")
print(f"  Shape: {sweep_df.shape[0]} rows × {sweep_df.shape[1]} columns")
