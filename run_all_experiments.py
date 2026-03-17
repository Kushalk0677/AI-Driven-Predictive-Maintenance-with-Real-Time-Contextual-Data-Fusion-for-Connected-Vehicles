"""
run_all_experiments.py
======================
Runs all 7 experiments in sequence and saves results to results/

Usage:
    python run_all_experiments.py

Requirements:
    pip install vehiclepm lightgbm xgboost shap matplotlib

Note: Exp 3 requires ai4i2020.csv in data/raw/
      Download from: https://archive.ics.uci.edu/dataset/601/
"""

import os
import sys
import time
import subprocess

EXPERIMENTS = [
    ("Exp 1 — Feature Group Ablation Study",      "experiments/exp1_ablation_study.py"),
    ("Exp 2 — Multi-Model Classification",         "experiments/exp2_classification_benchmark.py"),
    ("Exp 3 — AI4I 2020 Real-World Benchmark",     "experiments/exp3_ai4i_benchmark.py"),
    ("Exp 4 — Noise Sensitivity Analysis",         "experiments/exp4_noise_sensitivity.py"),
    ("Exp 5 — SHAP Feature Importance",            "experiments/exp5_shap_importance.py"),
    ("Exp 6 — Probability Calibration",            "experiments/exp6_calibration.py"),
    ("Exp 7 — Service Time Regression",            "experiments/exp7_regression.py"),
]

os.makedirs("results", exist_ok=True)

print("=" * 60)
print("  vehiclepm — Full Experiment Suite")
print("  AI-Driven Predictive Maintenance")
print("  Khemani & Qureshi (2026) arXiv:2603.13343")
print("=" * 60)
print()

total_start = time.time()

for name, script in EXPERIMENTS:
    print(f"▶ {name}")
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"  {status} Done in {elapsed:.1f}s\n")

total = time.time() - total_start
print("=" * 60)
print(f"  All experiments complete in {total:.0f}s")
print(f"  Results saved to: results/")
print("=" * 60)
