"""
run_all.py
==========

Runs the complete analysis pipeline in sequence.

Scripts are executed in order:
  01_compute_measures.py    — Load data, compute 7 measures for all subjects
  02_threshold_sweep.py     — Find optimal activity threshold
  03_lab_rw_correlations.py — Lab vs Real-World correlations + Figure 1
  04_clinical_correlations.py — Clinical correlations + Figure 2

Run from the repository root:
    python run_all.py

Author: Bijju
"""

import subprocess
import sys
import time
from pathlib import Path

# ── Locate the scripts directory ──────────────────────────────────────────────
REPO_DIR   = Path(__file__).parent
SCRIPTS_DIR = REPO_DIR / "scripts"

PIPELINE = [
    ("01_compute_measures.py",    "Load data and compute performance measures"),
    ("02_threshold_sweep.py",     "Find optimal activity threshold"),
    ("03_lab_rw_correlations.py", "Lab vs Real-World correlations + Figure 1"),
    ("04_clinical_correlations.py","Clinical correlations + Figure 2"),
]

# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("AHHA LAB — ECOLOGICAL VALIDITY ANALYSIS PIPELINE")
print("=" * 60)
print(f"\nRunning {len(PIPELINE)} scripts in sequence...\n")

total_start = time.time()

for i, (script_name, description) in enumerate(PIPELINE, start=1):
    script_path = SCRIPTS_DIR / script_name

    print(f"[{i}/{len(PIPELINE)}] {script_name}")
    print(f"        {description}")
    print("-" * 60)

    step_start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_DIR),   # run from the repo root so imports work
        capture_output=False, # let output print to terminal in real-time
    )
    elapsed = time.time() - step_start

    if result.returncode != 0:
        print(f"\n[FAILED] {script_name} exited with code {result.returncode}")
        print("Pipeline stopped. Fix the error above and re-run.")
        sys.exit(1)

    print(f"\n  Completed in {elapsed:.1f}s\n")

total_elapsed = time.time() - total_start

print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"\nTotal time: {total_elapsed:.1f}s")
print("\nOutputs written to:")
print(f"  {REPO_DIR}/results/tables/all_measures.csv")
print(f"  {REPO_DIR}/results/tables/threshold_sweep_results.csv")
print(f"  {REPO_DIR}/results/tables/lab_rw_correlations.csv")
print(f"  {REPO_DIR}/results/tables/clinical_correlations.csv")
print(f"  {REPO_DIR}/results/figures/fig1_lab_rw_scatterplots.png")
print(f"  {REPO_DIR}/results/figures/fig2_clinical_correlations.png")
