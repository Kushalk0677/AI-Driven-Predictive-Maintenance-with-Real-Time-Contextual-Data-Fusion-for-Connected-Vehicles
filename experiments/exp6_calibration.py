"""
=============================================================================
EXP 6 — Probability Calibration Analysis
=============================================================================
Evaluates reliability of LightGBM probability outputs before and after
Platt scaling. Critical for safety-critical alert threshold selection.

Outputs:
    results/exp6_calibration_results.csv
    results/exp6_calibration_plot.png
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset
from vehiclepm.features import build_feature_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("EXP 6 — Probability Calibration Analysis")
print("=" * 60)

# Data
df = generate_synthetic_dataset(n_samples=2000, noise_sigma=1.0, random_state=42)
X = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
y = df["maintenance_needed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Uncalibrated
clf_raw = VehiclePMClassifier(model_type="lightgbm")
clf_raw.fit(X_train, y_train)
probs_raw = clf_raw.predict_proba(X_test)[:, 1]
brier_raw = brier_score_loss(y_test, probs_raw)

# Calibrated (Platt scaling)
clf_cal = VehiclePMClassifier(model_type="lightgbm", calibrate=True)
clf_cal.fit(X_train, y_train)
probs_cal = clf_cal.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, probs_cal)

print(f"\n  Brier score (uncalibrated): {brier_raw:.4f}")
print(f"  Brier score (calibrated):   {brier_cal:.4f}")
print(f"  Improvement:                {brier_raw - brier_cal:.4f}")

results = pd.DataFrame([
    {"Model": "LightGBM (uncalibrated)", "Brier Score": round(brier_raw, 4)},
    {"Model": "LightGBM + Platt Scaling", "Brier Score": round(brier_cal, 4)},
])
results.to_csv("results/exp6_calibration_results.csv", index=False)
print("\n  → Saved: results/exp6_calibration_results.csv")

# Calibration curves
frac_raw, mean_raw = calibration_curve(y_test, probs_raw, n_bins=10)
frac_cal, mean_cal = calibration_curve(y_test, probs_cal, n_bins=10)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Experiment 6: Probability Calibration Analysis\n"
             "Closer alignment to diagonal = better-calibrated alert probabilities",
             fontsize=11, fontweight="bold")

for ax, frac, mean, brier, title in [
    (ax1, frac_raw, mean_raw, brier_raw,
     f"LightGBM (uncalibrated)\nBrier = {brier_raw:.4f}"),
    (ax2, frac_cal, mean_cal, brier_cal,
     f"LightGBM + Platt Scaling\nBrier = {brier_cal:.4f}"),
]:
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly calibrated")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.plot(mean, frac, "o-", color="#D73027", lw=2.5, markersize=7,
            label=f"LightGBM (Brier={brier:.4f})")
    ax.fill_between(mean, frac, mean, alpha=0.15, color="#D73027",
                    label="Calibration gap")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/exp6_calibration_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: results/exp6_calibration_plot.png")

print("\n" + "=" * 60)
print("EXP 6 COMPLETE")
print("=" * 60)
