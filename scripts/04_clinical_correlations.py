"""
04_clinical_correlations.py
===========================

PURPOSE:
    Correlate sensor-based upper limb performance measures with standard
    clinical assessments (FMA, MAL, WMFT) to assess construct validity.

    We examine correlations separately for:
      - Lab context (measures from in-lab sessions)
      - Real-world context (measures from multi-day daily life recording)

    This answers: "Which context better captures clinically-relevant function?"

INPUTS:
    results/tables/all_measures.csv   (from 01_compute_measures.py)

OUTPUTS:
    results/tables/clinical_correlations.csv
        Long-format table: one row per (setting × measure × clinical_score) triple.
        Columns: setting, measure, clinical, n, spearman_r, p_value, sig

    results/figures/fig2_clinical_correlations.png
        Heatmap showing Spearman r values for all measure × clinical score pairs,
        with Lab (left) and Real-World (right) side by side.

WHY SPEARMAN (not Pearson)?
    Clinical scores (FMA, MAL, WMFT) are ordinal or bounded scales.
    Many are not normally distributed.
    Spearman rank correlation is more robust for these variables.

CLINICAL SCORE NOTES:
    FMA (Fugl-Meyer Assessment):
        - Higher = better motor function (0–66 for upper extremity)
        - We expect positive r with activity measures

    MAL-AS (Motor Activity Log — Amount of Use):
        - Self-reported arm use frequency in daily life (0–5)
        - Higher = more arm use. Expect positive r.

    MAL-HW (Motor Activity Log — How Well):
        - Self-reported quality of movement (0–5)
        - Higher = better. Expect positive r.

    WMFT-TIME (Wolf Motor Function Test — Time):
        - Time to complete tasks (seconds). LOWER = better function.
        - Expect NEGATIVE r (faster = more movement capacity)

    WMFT-FAS (Wolf Motor Function Test — Functional Ability Scale):
        - Quality of movement during tasks (0–75). Higher = better.
        - Expect positive r.

Author: Bijju
Project: AHHA Lab – Stroke Rehabilitation Ecological Validity Study
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MEASURES, MEASURE_LABELS, CLINICAL_COLS, TABLES_DIR, FIGURES_DIR


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 04: CLINICAL CORRELATIONS")
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

# Identify which clinical columns are actually present in the loaded data
available_clinical = [col for col in CLINICAL_COLS if col in df.columns]
missing_clinical   = [col for col in CLINICAL_COLS if col not in df.columns]

if missing_clinical:
    print(f"  Note: These clinical columns not found in data: {missing_clinical}")
print(f"  Using {len(available_clinical)} clinical scores: {available_clinical}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def sig_stars(p: float) -> str:
    """Return significance annotation for a p-value."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE SPEARMAN CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
# For each combination of:
#   - Setting (Lab, Real-World)
#   - Sensor measure (7 measures)
#   - Clinical score (11 scores)
# compute Spearman r and p-value.

settings = {
    "Lab":        "lab",
    "Real-World": "rw",
}

print("\nComputing Spearman correlations...")
corr_records = []

for setting_label, prefix in settings.items():
    for measure in MEASURES:
        sensor_col = f"{prefix}_{measure}"

        if sensor_col not in df.columns:
            print(f"  WARNING: Column '{sensor_col}' not found, skipping")
            continue

        for clinical_col in available_clinical:
            # Only use subjects where BOTH the sensor measure and clinical score are available
            valid = df[sensor_col].notna() & df[clinical_col].notna()
            n = valid.sum()

            if n < 3:
                # Too few subjects to compute a meaningful correlation
                r, p = np.nan, np.nan
                sig  = ""
            else:
                r, p = spearmanr(df.loc[valid, sensor_col], df.loc[valid, clinical_col])
                sig  = sig_stars(p)

            corr_records.append({
                "setting":    setting_label,
                "measure":    measure,
                "clinical":   clinical_col,
                "n":          n,
                "spearman_r": round(r, 3) if not np.isnan(r) else np.nan,
                "p_value":    round(p, 4) if not np.isnan(p) else np.nan,
                "sig":        sig,
            })

corr_df = pd.DataFrame(corr_records)
print(f"  Computed {len(corr_df)} correlation pairs")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS: ONE TABLE PER CLINICAL SCORE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SPEARMAN CORRELATIONS BY CLINICAL SCORE")
print("Significance: * p<.05  ** p<.01  *** p<.001")
print("=" * 60)

for clinical_col in available_clinical:
    sub = corr_df[corr_df["clinical"] == clinical_col]
    print(f"\n── {clinical_col} ──")
    print(f"  {'Measure':<40} {'Lab r':>10} {'RW r':>10}")
    print("  " + "-" * 62)

    for measure in MEASURES:
        lab_row = sub[(sub["setting"] == "Lab") & (sub["measure"] == measure)]
        rw_row  = sub[(sub["setting"] == "Real-World") & (sub["measure"] == measure)]

        if len(lab_row) == 0 or len(rw_row) == 0:
            continue

        lab_r   = lab_row.iloc[0]["spearman_r"]
        lab_sig = lab_row.iloc[0]["sig"]
        rw_r    = rw_row.iloc[0]["spearman_r"]
        rw_sig  = rw_row.iloc[0]["sig"]

        lab_str = f"{lab_r:+.3f}{lab_sig}" if not np.isnan(lab_r) else "—"
        rw_str  = f"{rw_r:+.3f}{rw_sig}"  if not np.isnan(rw_r)  else "—"

        print(f"  {measure:<40} {lab_str:>10} {rw_str:>10}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: CLINICAL CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
# Heatmap with:
#   Rows    = sensor measures (7 measures)
#   Columns = clinical scores
#   Color   = Spearman r (diverging: red=positive, blue=negative)
#   Text    = r value, with * for significant
#
# Two side-by-side heatmaps: Lab (left) and Real-World (right).
# Comparing them shows whether lab or real-world better captures clinical status.

print("\nGenerating Figure 2: Clinical correlation heatmap...")

def build_r_matrix(setting_label: str) -> tuple:
    """
    Build a (n_measures × n_clinical) matrix of Spearman r values for one setting.

    Returns
    -------
    r_matrix : np.ndarray, shape (n_measures, n_clinical)
    p_matrix : np.ndarray, shape (n_measures, n_clinical)
    """
    sub = corr_df[corr_df["setting"] == setting_label]
    r_matrix = np.full((len(MEASURES), len(available_clinical)), np.nan)
    p_matrix = np.full((len(MEASURES), len(available_clinical)), np.nan)

    for i, measure in enumerate(MEASURES):
        for j, clinical_col in enumerate(available_clinical):
            cell = sub[(sub["measure"] == measure) & (sub["clinical"] == clinical_col)]
            if len(cell) > 0:
                r_matrix[i, j] = cell.iloc[0]["spearman_r"]
                p_matrix[i, j] = cell.iloc[0]["p_value"]

    return r_matrix, p_matrix


r_lab, p_lab   = build_r_matrix("Lab")
r_rw,  p_rw    = build_r_matrix("Real-World")

# ── Axis labels ───────────────────────────────────────────────────────────────
# Shorten clinical column names for display (long names would overlap)
clinical_display = {
    "FMA":                       "FMA",
    "FMA-UE+COORDINATION/SPEED": "FMA-UE\n+Coord",
    "FMA-WRIST+HAND":            "FMA-Wrist\n+Hand",
    "MAL-AS":                    "MAL-AS",
    "MAL-HW":                    "MAL-HW",
    "WMFT-TIME":                 "WMFT-TIME",
    "WMFT-FAS":                  "WMFT-FAS",
    "FMA-UE":                    "FMA-UE",
    "FMA-COORDINATION/SPEED":    "FMA-Coord",
    "FMA-WRIST":                 "FMA-Wrist",
    "FMA-HAND":                  "FMA-Hand",
}
x_labels = [clinical_display.get(c, c) for c in available_clinical]
y_labels  = [MEASURE_LABELS[m].replace("\n", " ") for m in MEASURES]

# ── Build figure ──────────────────────────────────────────────────────────────
fig, (ax_lab, ax_rw) = plt.subplots(
    1, 2,
    figsize=(max(14, len(available_clinical) * 1.2), 5),
    sharey=True
)
fig.suptitle(
    "Spearman Correlations: Sensor Measures × Clinical Scores\n"
    "* p<.05   ** p<.01   *** p<.001",
    fontsize=12, fontweight="bold"
)

cmap = plt.cm.RdBu           # diverging: blue=negative r, red=positive r
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)  # centre at 0


def draw_heatmap(ax, r_matrix, p_matrix, title):
    """Draw one heatmap panel."""
    im = ax.imshow(r_matrix, cmap=cmap, norm=norm, aspect="auto")

    # Axis ticks and labels
    ax.set_xticks(range(len(available_clinical)))
    ax.set_xticklabels(x_labels, fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(len(MEASURES)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Annotate cells with r value and significance stars
    for i in range(len(MEASURES)):
        for j in range(len(available_clinical)):
            r = r_matrix[i, j]
            p = p_matrix[i, j]
            if np.isnan(r):
                continue
            sig  = sig_stars(p) if not np.isnan(p) else ""
            # Use white text on dark cells for readability
            text_color = "white" if abs(r) > 0.55 else "black"
            ax.text(j, i, f"{r:+.2f}{sig}",
                    ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")
    return im


im1 = draw_heatmap(ax_lab, r_lab, p_lab, "Lab Context")
im2 = draw_heatmap(ax_rw,  r_rw,  p_rw,  "Real-World Context")

# Add a single shared colorbar on the right
plt.colorbar(im2, ax=[ax_lab, ax_rw],
             orientation="vertical",
             fraction=0.015, pad=0.02,
             label="Spearman r")

plt.tight_layout()

fig_path = FIGURES_DIR / "fig2_clinical_correlations.png"
fig.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {fig_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE CORRELATION TABLE
# ══════════════════════════════════════════════════════════════════════════════

out_path = TABLES_DIR / "clinical_correlations.csv"
corr_df.to_csv(out_path, index=False)

print("\n" + "=" * 60)
print("OUTPUTS SAVED")
print("=" * 60)
print(f"  {out_path}")
print(f"  {fig_path}")
