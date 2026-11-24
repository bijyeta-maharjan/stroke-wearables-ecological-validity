"""
utils.py
========

Shared helper functions used across all analysis scripts.

Functions:
  - load_subject_data()        — Load lab and real-world accelerometer data for one subject
  - compute_measures()         — Compute the 7 core performance measures
  - filter_inactive_epochs()   — Remove sedentary/non-wear epochs using an activity threshold

Author: Bijju
Project: AHHA Lab – Stroke Rehabilitation Ecological Validity Study
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Allow importing config.py from the repository root
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import INLAB_DIR, RW_DIR


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_subject_data(subject_id: str) -> dict:
    """
    Load in-lab and real-world accelerometer data for a single subject.

    The data files are pre-processed epoch-level CSVs where each row is one
    epoch (time window) and columns are acceleration magnitude (in g) for each
    sensor placement.

    Parameters
    ----------
    subject_id : str
        The subject identifier, e.g. "SYN01"

    Returns
    -------
    dict with keys:
        "lab_affected"    : np.ndarray — affected wrist acceleration (g), in-lab
        "lab_unaffected"  : np.ndarray — unaffected wrist acceleration (g), in-lab
        "rw_affected"     : np.ndarray — affected wrist acceleration (g), real-world
        "rw_unaffected"   : np.ndarray — unaffected wrist acceleration (g), real-world
        "lab_n_epochs"    : int — number of epochs in lab recording
        "rw_n_epochs"     : int — number of epochs in real-world recording

    Raises
    ------
    FileNotFoundError
        If either the lab or real-world data file is missing for this subject.

    Example
    -------
    >>> data = load_subject_data("SYN01")
    >>> print(data["lab_n_epochs"])   # e.g. 12480
    >>> print(data["rw_n_epochs"])    # e.g. 86400
    """

    lab_path = INLAB_DIR / subject_id / "processed.csv"
    rw_path  = RW_DIR    / subject_id / "combined.csv"

    if not lab_path.exists():
        raise FileNotFoundError(f"Missing lab data: {lab_path}")
    if not rw_path.exists():
        raise FileNotFoundError(f"Missing real-world data: {rw_path}")

    # Load and normalise column names (strip whitespace, lowercase)
    lab_df = pd.read_csv(lab_path)
    rw_df  = pd.read_csv(rw_path)
    lab_df.columns = [c.strip().lower() for c in lab_df.columns]
    rw_df.columns  = [c.strip().lower() for c in rw_df.columns]

    # Extract wrist columns only (we focus on wrist, not finger, sensors)
    lab_aff   = lab_df["affected wrist"].to_numpy(dtype=float)
    lab_unaff = lab_df["less affected wrist"].to_numpy(dtype=float)
    rw_aff    = rw_df["affected wrist"].to_numpy(dtype=float)
    rw_unaff  = rw_df["less affected wrist"].to_numpy(dtype=float)

    return {
        "lab_affected":   lab_aff,
        "lab_unaffected": lab_unaff,
        "rw_affected":    rw_aff,
        "rw_unaffected":  rw_unaff,
        "lab_n_epochs":   len(lab_aff),
        "rw_n_epochs":    len(rw_aff),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE MEASURE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_measures(affected_acc: np.ndarray,
                     unaffected_acc: np.ndarray,
                     use_duration_alpha: float,
                     use_ratio_alpha: float) -> dict:
    """
    Compute the 7 core upper limb performance measures.

    These measures follow the computation methods from the lab's original pipeline.
    Two categories:

    INTENSITY MEASURES (computed on raw data, no threshold filtering):
        These capture the average movement amplitude.

    DURATION MEASURES (require an activity threshold 'alpha'):
        These count how often each limb is "active" (above the threshold),
        reflecting the proportion of time the limb is used.

    Parameters
    ----------
    affected_acc : np.ndarray
        Epoch-level acceleration magnitudes (g) for the affected wrist.
    unaffected_acc : np.ndarray
        Epoch-level acceleration magnitudes (g) for the less-affected wrist.
    use_duration_alpha : float
        Threshold (g) for counting "active" epochs in use_duration measures.
        Epochs where acceleration > alpha are considered "active" arm use.
    use_ratio_alpha : float
        Threshold (g) for counting "active" epochs in the use_ratio measure.
        (Kept separate from use_duration_alpha for flexibility, though in
         practice both are set to the same value: 0.003328 g from the lab's original pipeline.)

    Returns
    -------
    dict with 7 keys:
        "affected_intensity_avg"              : float
        "unaffected_intensity_avg"            : float
        "bilateral_intensity_avg"             : float
        "intensity_ratio"                     : float
        "affected_use_duration_percentage"    : float
        "unaffected_use_duration_percentage"  : float
        "use_ratio"                           : float

    Example
    -------
    >>> measures = compute_measures(lab_aff, lab_unaff, alpha=0.003328, alpha2=0.003328)
    >>> print(measures["affected_intensity_avg"])   # e.g. 0.187 g
    """

    # ── INTENSITY AVERAGES ──────────────────────────────────────────────────
    # Simple mean of all epoch-level acceleration values.
    # No threshold is applied — we want the average over the entire recording.

    int_aff  = np.mean(affected_acc)     # mean affected wrist acceleration
    int_unaff = np.mean(unaffected_acc)  # mean unaffected wrist acceleration
    int_bilateral = int_aff + int_unaff  # bilateral = sum of both limbs

    # ── INTENSITY RATIO ─────────────────────────────────────────────────────
    # Log-based asymmetry measure:
    #   intensity_ratio = mean(log(affected)) - mean(log(unaffected))
    #
    # Log transform compresses the right-skewed acceleration distribution
    # and makes the ratio interpretable on an additive scale.
    #   0  = symmetric use (both limbs equally active)
    #  <0  = affected limb less active than unaffected (common in stroke)
    #
    # We filter out non-positive values before taking log (log(0) = -inf).
    aff_pos   = affected_acc[affected_acc > 0]
    unaff_pos = unaffected_acc[unaffected_acc > 0]

    if len(aff_pos) > 0 and len(unaff_pos) > 0:
        int_ratio = np.mean(np.log(aff_pos)) - np.mean(np.log(unaff_pos))
    else:
        # Not enough positive values — log ratio undefined
        int_ratio = np.nan

    # ── USE DURATION PERCENTAGES ─────────────────────────────────────────────
    # For each limb: count the epochs where acceleration exceeds the threshold
    # and express this as a proportion of total epochs.
    #
    # This measures how OFTEN the limb is used, not how intensely.
    n_aff   = len(affected_acc)
    n_unaff = len(unaffected_acc)

    n_active_aff   = np.sum(affected_acc   > use_duration_alpha)
    n_active_unaff = np.sum(unaffected_acc > use_duration_alpha)

    dur_pct_aff   = n_active_aff   / n_aff   if n_aff   > 0 else np.nan
    dur_pct_unaff = n_active_unaff / n_unaff if n_unaff > 0 else np.nan

    # ── USE RATIO ────────────────────────────────────────────────────────────
    # Log-based ratio of active epoch counts between limbs:
    #   use_ratio = log(count_affected_active) - log(count_unaffected_active)
    #
    # Parallels intensity_ratio but for duration rather than magnitude.
    # Values near 0 = symmetric; negative = affected limb used less often.
    n_active_aff_ratio   = np.sum(affected_acc   > use_ratio_alpha)
    n_active_unaff_ratio = np.sum(unaffected_acc > use_ratio_alpha)

    if n_active_aff_ratio > 0 and n_active_unaff_ratio > 0:
        use_ratio = np.log(n_active_aff_ratio) - np.log(n_active_unaff_ratio)
    else:
        # One or both limbs had zero active epochs — ratio undefined
        use_ratio = np.nan

    return {
        "affected_intensity_avg":             int_aff,
        "unaffected_intensity_avg":           int_unaff,
        "bilateral_intensity_avg":            int_bilateral,
        "intensity_ratio":                    int_ratio,
        "affected_use_duration_percentage":   dur_pct_aff,
        "unaffected_use_duration_percentage": dur_pct_unaff,
        "use_ratio":                          use_ratio,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EPOCH FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def filter_inactive_epochs(affected: np.ndarray,
                           unaffected: np.ndarray,
                           threshold: float) -> dict:
    """
    Remove sedentary or non-wear epochs from accelerometer data.

    WHY WE FILTER:
        Real-world recordings span multiple days and include periods when the
        participant is sleeping, at rest, or not wearing the sensors. These
        inactive epochs pull the real-world mean intensity down dramatically,
        making lab vs real-world means non-comparable (lab is always active
        since tasks are structured and continuous).

        We keep an epoch if EITHER wrist shows activity above the threshold —
        this ensures we don't accidentally discard epochs where only one limb
        is moving (common in stroke survivors who heavily favour one limb).

    Parameters
    ----------
    affected : np.ndarray
        Epoch-level acceleration magnitudes (g) for affected wrist.
    unaffected : np.ndarray
        Epoch-level acceleration magnitudes (g) for unaffected wrist.
    threshold : float
        Activity threshold (g). Epochs where BOTH limbs are below this value
        are removed (considered inactive/non-wear).

    Returns
    -------
    dict with keys:
        "affected_filtered"   : np.ndarray — filtered affected wrist data
        "unaffected_filtered" : np.ndarray — filtered unaffected wrist data
        "n_original"          : int — number of epochs before filtering
        "n_filtered"          : int — number of epochs after filtering
        "n_removed"           : int — number of epochs removed
        "pct_removed"         : float — percentage of epochs removed

    Example
    -------
    >>> result = filter_inactive_epochs(rw_aff, rw_unaff, threshold=0.005)
    >>> print(f"Removed {result['pct_removed']:.1f}% of epochs")
    """

    # Keep epoch if EITHER wrist is active (above threshold)
    # — don't require BOTH to be active, since stroke survivors often
    # move mainly one arm at a time
    active_mask = (affected > threshold) | (unaffected > threshold)

    n_original = len(affected)
    n_filtered = int(np.sum(active_mask))
    n_removed  = n_original - n_filtered
    pct_removed = (n_removed / n_original) * 100 if n_original > 0 else 0.0

    return {
        "affected_filtered":   affected[active_mask],
        "unaffected_filtered": unaffected[active_mask],
        "n_original":          n_original,
        "n_filtered":          n_filtered,
        "n_removed":           n_removed,
        "pct_removed":         pct_removed,
    }
