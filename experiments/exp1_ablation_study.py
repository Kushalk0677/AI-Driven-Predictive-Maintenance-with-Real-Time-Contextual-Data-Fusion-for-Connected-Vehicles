"""
=============================================================================
EXP 1 — Feature Group Ablation Study
=============================================================================
Demonstrates that V2X contextual features contribute meaningful predictive
signal beyond internal mechanical state alone.

Outputs:
    results/exp1_ablation_results.csv
    results/exp1_ablation_plot.png
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset
from vehiclepm.features import build_feature_matrix
from vehiclepm.evaluation.ablation import run_ablation_study

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("results", exist_ok=True)

# =============================================================================
# 1. GENERATE DATA
# =============================================================================

print("=" * 60)
print("EXP 1 — Feature Group Ablation Study")
print("=" * 60)
print("\nGenerating synthetic dataset (n=2000)...")

df = generate_synthetic_dataset(n_samples=2000, noise_sigma=1.0, random_state=42)
print(f"  Shape: {df.shape}")
print(f"  Failure rate: {df['maintenance_needed'].mean():.1%}")

# =============================================================================
# 2. RUN ABLATION
# =============================================================================

print("\nRunning ablation study (5-fold CV, LightGBM)...")
clf = VehiclePMClassifier(model_type="lightgbm", n_splits=5)
results = run_ablation_study(df, target_col="maintenance_needed", classifier=clf,
                              drop_cols=["risk_score"])

print("\n" + results.to_string(index=False))
results.to_csv("results/exp1_ablation_results.csv", index=False)
print("\n  → Saved: results/exp1_ablation_results.csv")

# =============================================================================
# 3. PLOT
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Experiment 1: Feature Group Ablation Study\n"
             "(5-fold CV, LightGBM, n=2000 synthetic samples)",
             fontsize=12, fontweight="bold")

configs   = results["configuration"].tolist()
f1_means  = results["f1_mean"].tolist()
f1_stds   = results["f1_std"].tolist()
f1_drops  = results["f1_drop"].tolist()

colors = ["#2166AC" if c == "All Features (Full Model)" else "#92C5DE" for c in configs]

# Panel A — absolute F1
ax1.barh(configs[::-1], f1_means[::-1], xerr=f1_stds[::-1],
         color=colors[::-1], alpha=0.88,
         error_kw={"elinewidth": 1.5, "capsize": 4})
ax1.set_xlabel("Macro F1-Score (5-fold CV)")
ax1.set_title("(a) Absolute Macro F1-Score per Configuration")
ax1.set_xlim(0.60, 1.0)
for i, (v, e) in enumerate(zip(f1_means[::-1], f1_stds[::-1])):
    ax1.text(v + e + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax1.grid(axis="x", alpha=0.3)

# Panel B — F1 drop
drop_colors = []
for d in f1_drops[::-1]:
    if d == 0.0:
        drop_colors.append("#2166AC")
    elif abs(d) > 0.02:
        drop_colors.append("#D73027")
    elif abs(d) > 0.005:
        drop_colors.append("#F4A582")
    else:
        drop_colors.append("#92C5DE")

ax2.barh(configs[::-1], f1_drops[::-1], color=drop_colors, alpha=0.88)
ax2.set_xlabel("F1 Drop vs. Full Model")
ax2.set_title("(b) F1 Drop When Each Feature Group Removed")
ax2.axvline(0, color="black", lw=0.8)
for i, v in enumerate(f1_drops[::-1]):
    if v != 0:
        ax2.text(v - 0.001, i, f"{v:.3f}", va="center", ha="right", fontsize=9)
ax2.grid(axis="x", alpha=0.3)

large   = mpatches.Patch(color="#D73027", label="Large drop (>0.02)")
medium  = mpatches.Patch(color="#F4A582", label="Moderate drop (>0.005)")
small   = mpatches.Patch(color="#92C5DE", label="Small drop")
ax2.legend(handles=[large, medium, small], fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("results/exp1_ablation_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: results/exp1_ablation_plot.png")

print("\n" + "=" * 60)
print("EXP 1 COMPLETE")
print(f"  V2X removal F1 drop: {results.loc[results['configuration'].str.contains('Environmental'), 'f1_drop'].values[0]:.3f}")
print(f"  Internal only F1 drop: {results.loc[results['configuration'].str.contains('Internal Only'), 'f1_drop'].values[0]:.3f}")
print("=" * 60)
