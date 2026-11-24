"""
03_lab_rw_correlations.py
=========================

PURPOSE:
    Compute Pearson correlations between laboratory and real-world upper limb
    performance measures, and visualise the key finding about ecological validity.

    The central question: does a subject's performance rank in the lab predict
    their rank in real-world daily life?

INPUTS:
    results/tables/all_measures.csv   (produced by 01_compute_measures.py)

OUTPUTS:
    results/tables/lab_rw_correlations.csv
        Pearson r, R², p-value, and N for each of the 7 measures.

    results/figures/fig1_lab_rw_scatterplots.png
        2-panel figure:
          Panel A: Affected limb — lab intensity vs RW intensity
          Panel B: Unaffected limb — lab intensity vs RW intensity
        This figure is the central visual for the ecological validity finding.

    results/figures/fig1b_lab_rw_violins.png
        Violin plots comparing the distribution of each measure in lab vs
        real-world, with paired t-test significance annotations.

KEY FINDING:
    Affected limb shows strong Lab–RW correlation (R² ≈ 0.63–0.85):
        → Lab accurately predicts real-world function for the impaired limb
        → Good ecological validity for the affected side

    Unaffected limb shows weak Lab–RW correlation (R² ≈ 0.17–0.52):
        → Lab does NOT predict real-world function for the unaffected limb
        → The unaffected limb adapts its role based on context
        → In the lab (constrained tasks) it behaves differently than at home

Author: Bijju
Project: Lab vs Home Arm Use After Stroke
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr, ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MEASURES, MEASURE_LABELS, TABLES_DIR, FIGURES_DIR


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 03: LAB vs REAL-WORLD CORRELATIONS")
print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

measures_path = TABLES_DIR / "all_measures.csv"
if not measures_path.exists():
    raise FileNotFoundError(
        f"Cannot find {measures_path}\n"
        "Please run 01_compute_measures.py first."
    )

df = pd.read_csv(measures_path)
print(f"\nLoaded measures for {len(df)} subjects")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: SIGNIFICANCE STARS
# ══════════════════════════════════════════════════════════════════════════════

def sig_stars(p: float) -> str:
    """Return significance stars for a given p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE PEARSON CORRELATIONS FOR ALL 7 MEASURES
# ══════════════════════════════════════════════════════════════════════════════
# For each measure: correlate lab values vs real-world values across subjects.
# We use Pearson r (not Spearman) because we are interested in linear association
# and the measures are roughly continuous.

print("\n" + "=" * 60)
print("PEARSON CORRELATIONS: Lab vs Real-World")
print("=" * 60)
print(f"\n{'Measure':<40} {'r':>8} {'R²':>8} {'p':>10} {'sig':>6} {'N':>4}")
print("-" * 78)

corr_rows = []

for measure in MEASURES:
    lab_col = f"lab_{measure}"
    rw_col  = f"rw_{measure}"

    # Drop subjects where either value is missing
    valid = df[lab_col].notna() & df[rw_col].notna()
    n = valid.sum()

    if n < 3:
        print(f"{measure:<40} {'—':>8} {'—':>8} {'—':>10} {'—':>6} {n:>4}  (too few)")
        continue

    lab_vals = df.loc[valid, lab_col].values
    rw_vals  = df.loc[valid, rw_col].values

    r, p = pearsonr(lab_vals, rw_vals)
    r2   = r ** 2
    sig  = sig_stars(p)

    print(f"{measure:<40} {r:>8.3f} {r2:>8.3f} {p:>10.4f} {sig:>6} {n:>4}")

    corr_rows.append({
        "measure":   measure,
        "pearson_r": round(r, 4),
        "r_squared": round(r2, 4),
        "p_value":   round(p, 4),
        "sig":       sig,
        "n":         n,
    })

corr_df = pd.DataFrame(corr_rows)


# ══════════════════════════════════════════════════════════════════════════════
# KEY FINDING SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("KEY FINDING: Affected vs Unaffected Ecological Validity")
print("=" * 60)

# Extract affected and unaffected intensity correlations
aff_row   = corr_df[corr_df["measure"] == "affected_intensity_avg"]
unaff_row = corr_df[corr_df["measure"] == "unaffected_intensity_avg"]

if len(aff_row) > 0 and len(unaff_row) > 0:
    r2_aff   = aff_row.iloc[0]["r_squared"]
    r2_unaff = unaff_row.iloc[0]["r_squared"]
    print(f"""
  AFFECTED LIMB:
    Lab–RW Pearson R² = {r2_aff:.3f}
    → {'GOOD ecological validity' if r2_aff >= 0.5 else 'Moderate ecological validity'}:
      lab performance explains {r2_aff*100:.0f}% of variance in real-world performance.

  UNAFFECTED LIMB:
    Lab–RW Pearson R² = {r2_unaff:.3f}
    → {'POOR ecological validity' if r2_unaff < 0.5 else 'Moderate ecological validity'}:
      lab performance explains only {r2_unaff*100:.0f}% of variance in real-world performance.

  ECOLOGICAL VALIDITY GAP:
    ΔR² = {r2_aff - r2_unaff:.3f}

  INTERPRETATION:
    The affected limb shows good ecological validity — how well a patient
    uses their stroke-affected arm in the clinic reflects how they use it at home.

    The unaffected limb does NOT show good ecological validity — the less-affected
    arm is used very differently at home than in the structured lab setting. This
    may reflect compensatory strategies that only emerge in real-world contexts.
""")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: SCATTERPLOTS — AFFECTED vs UNAFFECTED LAB–RW CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
# Two panels side by side:
#   Panel A (left):  Affected wrist — lab vs RW intensity
#   Panel B (right): Unaffected wrist — lab vs RW intensity
#
# The visual contrast in R² between the two panels IS the main finding.

print("\nGenerating Figure 1: Lab vs Real-World scatterplots...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle(
    "Lab vs Real-World Upper Limb Intensity  (N=18)\n"
    "Pearson correlation — one point per participant",
    fontsize=12, fontweight="bold", y=1.01
)

panel_specs = [
    {
        "ax":        axes[0],
        "measure":   "affected_intensity_avg",
        "title":     "A.  Affected Wrist",
        "subtitle":  "(stroke-impaired limb)",
        "color":     "#D65F5F",   # red — affected side
    },
    {
        "ax":        axes[1],
        "measure":   "unaffected_intensity_avg",
        "title":     "B.  Unaffected Wrist",
        "subtitle":  "(less-impaired limb)",
        "color":     "#4878CF",   # blue — unaffected side
    },
]

for spec in panel_specs:
    ax      = spec["ax"]
    measure = spec["measure"]

    # ── Extract paired values ─────────────────────────────────────────────────
    lab_col = f"lab_{measure}"
    rw_col  = f"rw_{measure}"
    valid   = df[lab_col].notna() & df[rw_col].notna()
    x = df.loc[valid, lab_col].values   # lab values (x-axis)
    y = df.loc[valid, rw_col].values    # real-world values (y-axis)

    # ── Pearson r and R² ──────────────────────────────────────────────────────
    r, p = pearsonr(x, y)
    r2   = r ** 2
    sig  = sig_stars(p)

    # ── Scatter plot ──────────────────────────────────────────────────────────
    ax.scatter(x, y,
               color=spec["color"], s=60,
               edgecolors="white", linewidths=0.6,
               zorder=3, label="Participant")

    # ── Regression (OLS) line ─────────────────────────────────────────────────
    # Shows the direction and strength of the linear trend
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color="#222222", lw=1.8, ls="-", zorder=2, label="Regression line")

    # ── Identity line (y = x) ─────────────────────────────────────────────────
    # If lab perfectly predicted RW, all points would lie on this line.
    # Deviations from the identity line show systematic lab–RW differences.
    lo = min(x.min(), y.min()) * 0.95
    hi = max(x.max(), y.max()) * 1.05
    ax.plot([lo, hi], [lo, hi],
            color="#888888", lw=1.2, ls="--", zorder=1, alpha=0.6,
            label="Identity (x=y)")

    # ── Annotation: R² and p-value ────────────────────────────────────────────
    annotation = f"R² = {r2:.3f}\np = {p:.3f} {sig}"
    ax.text(0.05, 0.95, annotation,
            transform=ax.transAxes,
            fontsize=10, va="top", ha="left", color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))

    # ── Labels and formatting ─────────────────────────────────────────────────
    ax.set_xlabel("Lab Intensity (g)", fontsize=10)
    ax.set_ylabel("Real-World Intensity (g)", fontsize=10)
    ax.set_title(f"{spec['title']}\n{spec['subtitle']}",
                 fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)

# Add legend to the right panel only
axes[1].legend(fontsize=8, loc="lower right", frameon=True)

plt.tight_layout()

fig_path = FIGURES_DIR / "fig1_lab_rw_scatterplots.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1B: VIOLIN PLOTS — LAB vs REAL-WORLD DISTRIBUTIONS PER MEASURE
# ══════════════════════════════════════════════════════════════════════════════
# For each of the 7 measures: a violin showing the distribution of values
# across 18 subjects in the Lab context vs the Real-World context.
#
# Overlaid on each violin:
#   - Individual subject points (paired lines connecting lab to RW)
#   - Median marker
#   - Wilcoxon signed-rank test result (non-parametric paired test)
#     We use Wilcoxon rather than a paired t-test because:
#       (a) n=18 is small, (b) some measures are not normally distributed.
#
# This plot shows not just whether the means differ, but HOW the distributions
# shift between contexts — wide vs narrow, skewed, etc.

print("\nGenerating Figure 1b: Violin plots (Lab vs Real-World)...")

# Number of columns in the figure grid
N_COLS = 4
N_ROWS = int(np.ceil(len(MEASURES) / N_COLS))

fig_v, axes_v = plt.subplots(N_ROWS, N_COLS,
                              figsize=(N_COLS * 3.5, N_ROWS * 4.2))
axes_v = np.array(axes_v).flatten()

fig_v.suptitle(
    "Lab vs Real-World: Distribution Comparison (N=18)\n"
    "Wilcoxon signed-rank test  |  Lines connect the same participant",
    fontsize=12, fontweight="bold"
)

LAB_COLOR = "#4878CF"    # blue for lab
RW_COLOR  = "#D65F5F"    # red for real-world

for idx, measure in enumerate(MEASURES):
    ax = axes_v[idx]

    lab_col = f"lab_{measure}"
    rw_col  = f"rw_{measure}"

    # Paired values (drop subjects missing either value)
    valid   = df[lab_col].notna() & df[rw_col].notna()
    lab_vals = df.loc[valid, lab_col].values
    rw_vals  = df.loc[valid, rw_col].values
    n = len(lab_vals)

    # ── Violin plots ──────────────────────────────────────────────────────────
    # Position 1 = Lab, position 2 = Real-World
    parts = ax.violinplot(
        [lab_vals, rw_vals],
        positions=[1, 2],
        widths=0.55,
        showmedians=True,
        showextrema=True,
    )

    # Colour the violin bodies
    for pc, color in zip(parts["bodies"], [LAB_COLOR, RW_COLOR]):
        pc.set_facecolor(color)
        pc.set_edgecolor("white")
        pc.set_alpha(0.6)

    # Style median lines and extrema bars
    for partname in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[partname].set_edgecolor("#333333")
        parts[partname].set_linewidth(1.2)

    # ── Individual subject points + connecting lines ───────────────────────────
    # Jitter x positions slightly so overlapping points are visible
    rng    = np.random.default_rng(seed=42)
    jitter = rng.uniform(-0.06, 0.06, size=n)

    for i in range(n):
        x_lab = 1 + jitter[i]
        x_rw  = 2 + jitter[i]
        # Thin grey connecting line shows direction of change per subject
        ax.plot([x_lab, x_rw], [lab_vals[i], rw_vals[i]],
                color="#aaaaaa", lw=0.8, alpha=0.5, zorder=2)
        # Subject dots
        ax.scatter(x_lab, lab_vals[i],
                   color=LAB_COLOR, s=22, zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter(x_rw, rw_vals[i],
                   color=RW_COLOR,  s=22, zorder=3, edgecolors="white", linewidths=0.4)

    # ── Wilcoxon signed-rank test ─────────────────────────────────────────────
    # Non-parametric paired test: does the measure differ between lab and RW?
    # We expect it to differ (lab has higher intensity due to structured tasks).
    try:
        _, p_wilcox = wilcoxon(lab_vals, rw_vals)
        sig_label   = sig_stars(p_wilcox)
        test_str    = f"Wilcoxon p = {p_wilcox:.3f} {sig_label}"
    except Exception:
        # wilcoxon fails if all differences are zero
        test_str = "p = N/A"

    # Place the p-value annotation above the plot
    y_max = max(np.max(lab_vals), np.max(rw_vals))
    y_pad = (y_max - min(np.min(lab_vals), np.min(rw_vals))) * 0.08
    ax.text(1.5, y_max + y_pad,
            test_str,
            ha="center", va="bottom", fontsize=8,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    # ── Labels and formatting ─────────────────────────────────────────────────
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Lab", "Real-World"], fontsize=9)
    ax.set_title(MEASURE_LABELS[measure].replace("\n", " "),
                 fontsize=9, fontweight="bold", pad=6)
    ax.set_ylabel("Value", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

# Hide unused subplots (if number of measures doesn't fill the grid)
for idx in range(len(MEASURES), len(axes_v)):
    axes_v[idx].set_visible(False)

# Legend
lab_patch = mpatches.Patch(color=LAB_COLOR, alpha=0.7, label="Lab")
rw_patch  = mpatches.Patch(color=RW_COLOR,  alpha=0.7, label="Real-World")
fig_v.legend(handles=[lab_patch, rw_patch],
             loc="lower right", fontsize=9,
             bbox_to_anchor=(0.98, 0.01), frameon=True)

plt.tight_layout()

fig_v_path = FIGURES_DIR / "fig1b_lab_rw_violins.png"
fig_v.savefig(fig_v_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_v_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE CORRELATION TABLE
# ══════════════════════════════════════════════════════════════════════════════

out_path = TABLES_DIR / "lab_rw_correlations.csv"
corr_df.to_csv(out_path, index=False)

print("\n" + "=" * 60)
print("OUTPUTS SAVED")
print("=" * 60)
print(f"  {out_path}")
print(f"  {fig_path}")
print(f"  {fig_v_path}")
