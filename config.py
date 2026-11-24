"""
config.py
=========

Central configuration for the lab-vs-home arm-use analysis pipeline.

Paths are relative to this repository. By default the pipeline reads from
`data/`, which is git-ignored. Participant data is NOT included in this
repository; run `python tools/make_synthetic_data.py` to create a synthetic
dataset with the same file layout, or point DATA_DIR at your own data:

    export ARMUSE_DATA_DIR=/path/to/your/data

Expected layout under DATA_DIR:
    in-lab/<subject_id>/processed.csv       epoch-level wrist/finger acceleration (g)
    real-world/<subject_id>/combined.csv    epoch-level wrist/finger acceleration (g)
    clinical_scores.csv                     one row per subject (column `SubjID`)
"""

import os
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ARMUSE_DATA_DIR", REPO_DIR / "data"))

INLAB_DIR = DATA_DIR / "in-lab"
RW_DIR    = DATA_DIR / "real-world"
CLINICAL  = DATA_DIR / "clinical_scores.csv"

RESULTS_DIR = REPO_DIR / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


# ══════════════════════════════════════════════════════════════════════════════
# INCLUDED SUBJECTS
# ══════════════════════════════════════════════════════════════════════════════
#
# Subjects are discovered automatically: any subject with BOTH an in-lab
# recording and a real-world recording is included. In the original study,
# participants were excluded for a missing wrist sensor, empty real-world
# recordings, or a missing in-lab session, leaving N = 18.

def _discover_subjects():
    if not INLAB_DIR.exists() or not RW_DIR.exists():
        return []
    lab = {p.parent.name for p in INLAB_DIR.glob("*/processed.csv")}
    rw  = {p.parent.name for p in RW_DIR.glob("*/combined.csv")}
    return sorted(lab & rw)

INCLUDED_SUBJECTS = _discover_subjects()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Activity threshold for use_duration and use_ratio measures.
# Epochs where acceleration > alpha are counted as "active" arm use.
USE_DURATION_ALPHA = 0.003328   # g
USE_RATIO_ALPHA    = 0.003328   # g (same threshold, kept separate for clarity)

# Threshold sweep (02_threshold_sweep.py): find the threshold where lab and
# real-world means become comparable while correlation stays strong.
SWEEP_START = 0.001    # g
SWEEP_STOP  = 0.010    # g (inclusive)
SWEEP_STEP  = 0.0005   # g

ALPHA_SIGNIFICANCE = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# MEASURE NAMES
# ══════════════════════════════════════════════════════════════════════════════

MEASURES = [
    "affected_intensity_avg",
    "unaffected_intensity_avg",
    "bilateral_intensity_avg",
    "intensity_ratio",
    "affected_use_duration_percentage",
    "unaffected_use_duration_percentage",
    "use_ratio",
]

MEASURE_LABELS = {
    "affected_intensity_avg":              "Affected\nIntensity Avg",
    "unaffected_intensity_avg":            "Unaffected\nIntensity Avg",
    "bilateral_intensity_avg":             "Bilateral\nIntensity Avg",
    "intensity_ratio":                     "Intensity\nRatio",
    "affected_use_duration_percentage":    "Affected Use\nDuration (%)",
    "unaffected_use_duration_percentage":  "Unaffected Use\nDuration (%)",
    "use_ratio":                           "Use Ratio",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL SCORE COLUMNS
# ══════════════════════════════════════════════════════════════════════════════
#
# Score directions:
#   FMA and subscores : higher = better motor function
#   MAL-AS, MAL-HW    : higher = more/better arm use (self-reported)
#   WMFT-FAS          : higher = better functional ability
#   WMFT-TIME         : LOWER  = better (faster task completion)

CLINICAL_COLS = [
    "FMA",                        # Fugl-Meyer Assessment, total
    "FMA-UE+COORDINATION/SPEED",
    "FMA-WRIST+HAND",
    "MAL-AS",                     # Motor Activity Log, Amount of Use
    "MAL-HW",                     # Motor Activity Log, How Well
    "WMFT-TIME",                  # Wolf Motor Function Test, time
    "WMFT-FAS",                   # Wolf Motor Function Test, functional ability
    "FMA-UE",
    "FMA-COORDINATION/SPEED",
    "FMA-WRIST",
    "FMA-HAND",
]
