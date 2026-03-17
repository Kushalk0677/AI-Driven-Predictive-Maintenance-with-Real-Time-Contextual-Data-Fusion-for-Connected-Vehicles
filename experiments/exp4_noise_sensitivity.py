"""
=============================================================================
EXP 4 — Model Robustness: Noise Sensitivity Analysis
=============================================================================
Characterises how model performance degrades as sensor noise increases
from sigma=0 (clean) to sigma=3.0 (heavy noise).

Replaces speculative '15-25% degradation' estimates with measured curves.

Outputs:
    results/exp4_noise_results.csv
    results/exp4_noise_plot.png
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset, noise_sensitivity_analysis
from vehiclepm.features import build_feature_matrix

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("results", exist_ok=True)

# =============================================================================
# 1. GENERATE DATA
# =============================================================================

print("=" * 60)
print("EXP 4 — Noise Sensitivity Analysis")
print("=" * 60)
print("\nGenerating synthetic dataset (n=2000)...")

df = generate_synthetic_dataset(n_samples=2000, noise_sigma=1.0, random_state=42)
X = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
y = df["maintenance_needed"]

# =============================================================================
# 2. RUN NOISE SENSITIVITY
# =============================================================================

print("Running noise sensitivity analysis (5 seeds × 7 sigma levels)...")

clf = VehiclePMClassifier(model_type="lightgbm", n_splits=5)
sigma_levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

results = noise_sensitivity_analysis(
    X, y, clf,
    sigma_levels=sigma_levels,
    n_seeds=5,
    n_splits=5,
)

# Add interpretation
interpretations = {
    0.0: "Clean simulation",
    0.25: "Minimal noise",
    0.5: "Low noise",
    1.0: "Moderate noise (baseline σ)",
    1.5: "High noise",
    2.0: "Very high noise",
    3.0: "Extreme noise",
}
results["Interpretation"] = results["sigma"].map(interpretations)

print("\n" + results.to_string(index=False))
results.to_csv("results/exp4_noise_results.csv", index=False)
print("\n  → Saved: results/exp4_noise_results.csv")

# =============================================================================
# 3. PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Experiment 4: Model Robustness to Increasing Sensor Noise\n"
             "(σ=0: clean simulation; σ=3: heavy noise approximating real-world conditions)",
             fontsize=12, fontweight="bold")

sigmas   = results["sigma"].tolist()
f1_means = results["f1_mean"].tolist()
f1_stds  = results["f1_std"].tolist()
auc_means = results["auc_mean"].tolist()

ax.fill_between(sigmas,
                [f - s for f, s in zip(f1_means, f1_stds)],
                [f + s for f, s in zip(f1_means, f1_stds)],
                alpha=0.2, color="#4393C3", label="±1 std")
ax.plot(sigmas, f1_means, "o-", color="#4393C3", lw=2.5,
        markersize=7, label="F1 (Macro)", zorder=5)
ax.plot(sigmas, auc_means, "s--", color="#D73027", lw=2,
        markersize=6, label="AUC-ROC", zorder=4)

ax.axhline(0.90, color="grey", linestyle=":", lw=1.5, alpha=0.7)
ax.text(2.7, 0.905, "F1 = 0.90 threshold", fontsize=9, color="grey")

ax.set_xlabel("Noise Scale Factor (σ)")
ax.set_ylabel("Score")
ax.set_ylim(0.60, 1.02)
ax.set_xticks(sigmas)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

# Annotate key points
for sigma, f1, std in zip(sigmas, f1_means, f1_stds):
    ax.annotate(f"{f1:.3f}",
                xy=(sigma, f1), xytext=(0, 12),
                textcoords="offset points", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("results/exp4_noise_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: results/exp4_noise_plot.png")

print("\n" + "=" * 60)
print("EXP 4 COMPLETE")
print(f"  F1 at σ=0.5:  {results.loc[results['sigma']==0.5, 'f1_mean'].values[0]:.3f}")
print(f"  F1 at σ=1.0:  {results.loc[results['sigma']==1.0, 'f1_mean'].values[0]:.3f}")
print(f"  F1 at σ=2.0:  {results.loc[results['sigma']==2.0, 'f1_mean'].values[0]:.3f}")
print("=" * 60)
