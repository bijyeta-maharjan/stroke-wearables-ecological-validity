"""
make_synthetic_data.py
======================

Generates a fully SYNTHETIC dataset with the same file layout and column names
as the study data, so the analysis pipeline can be run end-to-end without
access to participant data.

No values here are derived from real participants. Subjects are named
SYN01..SYN18 and all numbers come from a seeded random generator. The
generator loosely builds in the study's qualitative pattern (lab activity of
the affected arm tracks home activity; the unaffected arm much less so) so the
figures are meaningful as a demo, but results will NOT match the real study.

Usage (from the repository root):
    python tools/make_synthetic_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, INLAB_DIR, RW_DIR, CLINICAL

SEED = 42
N_SUBJECTS = 18
LAB_EPOCHS = 2_000      # structured clinic session
RW_EPOCHS = 30_000      # multi-day home recording (includes rest/sleep)
COLS = ["affected wrist", "less affected wrist", "affected finger", "less affected finger"]

rng = np.random.default_rng(SEED)


def active_epochs(n, level):
    """Log-normal 'movement' magnitudes (g) centred on `level`."""
    return rng.lognormal(mean=np.log(level), sigma=0.6, size=n)


def make_recording(n, aff_level, unaff_level, rest_fraction):
    """Mix of rest epochs (near-zero) and active epochs for each sensor."""
    rest = rng.random(n) < rest_fraction
    out = {}
    for col, level in zip(COLS, [aff_level, unaff_level, aff_level * 1.3, unaff_level * 1.3]):
        x = active_epochs(n, level)
        x[rest] = rng.uniform(0.0, 0.002, rest.sum())
        out[col] = x
    return pd.DataFrame(out)


def main():
    INLAB_DIR.mkdir(parents=True, exist_ok=True)
    RW_DIR.mkdir(parents=True, exist_ok=True)

    clinical_rows = []
    for i in range(1, N_SUBJECTS + 1):
        sid = f"SYN{i:02d}"

        # Latent motor ability of the affected arm (0 = severe, 1 = mild)
        ability = rng.uniform(0.05, 1.0)

        # Affected arm: lab and home both driven by ability -> strong agreement
        lab_aff = 0.01 + 0.04 * ability
        rw_aff = (0.004 + 0.02 * ability) * rng.lognormal(0, 0.15)

        # Unaffected arm: home use driven by lifestyle, not impairment -> weak agreement
        lab_unaff = 0.045 * rng.lognormal(0, 0.15)
        rw_unaff = 0.02 * rng.lognormal(0, 0.35)

        for d in (INLAB_DIR / sid, RW_DIR / sid):
            d.mkdir(parents=True, exist_ok=True)
        make_recording(LAB_EPOCHS, lab_aff, lab_unaff, rest_fraction=0.05) \
            .to_csv(INLAB_DIR / sid / "processed.csv", index=False)
        make_recording(RW_EPOCHS, rw_aff, rw_unaff, rest_fraction=rng.uniform(0.5, 0.7)) \
            .to_csv(RW_DIR / sid / "combined.csv", index=False)

        noise = lambda s: rng.normal(0, s)
        fma_ue = np.clip(round(10 + 50 * ability + noise(4)), 0, 66)
        clinical_rows.append({
            "SubjID": sid,
            "FMA": fma_ue,
            "FMA-UE+COORDINATION/SPEED": np.clip(round(fma_ue * 0.6 + noise(2)), 0, 42),
            "FMA-WRIST+HAND": np.clip(round(fma_ue * 0.37 + noise(2)), 0, 24),
            "MAL-AS": round(float(np.clip(0.3 + 4 * ability + noise(0.6), 0, 5)), 2),
            "MAL-HW": round(float(np.clip(0.2 + 3.5 * ability + noise(0.6), 0, 5)), 2),
            "WMFT-TIME": round(float(np.clip(60 - 55 * ability + noise(5), 1, 120)), 2),
            "WMFT-FAS": round(float(np.clip(1 + 3.5 * ability + noise(0.3), 0, 5)), 2),
            "FMA-UE": np.clip(round(fma_ue * 0.55 + noise(2)), 0, 36),
            "FMA-COORDINATION/SPEED": np.clip(round(2 + 4 * ability + noise(0.5)), 0, 6),
            "FMA-WRIST": np.clip(round(10 * ability + noise(1)), 0, 10),
            "FMA-HAND": np.clip(round(14 * ability + noise(1)), 0, 14),
        })

    pd.DataFrame(clinical_rows).to_csv(CLINICAL, index=False)
    print(f"Synthetic dataset for {N_SUBJECTS} subjects written to {DATA_DIR}")


if __name__ == "__main__":
    main()
