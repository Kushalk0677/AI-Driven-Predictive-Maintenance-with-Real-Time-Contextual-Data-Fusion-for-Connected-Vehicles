"""
=============================================================================
EXP 5 — SHAP Feature Importance Analysis
=============================================================================
Confirms that V2X-sourced contextual features rank among the top predictors
using SHAP TreeExplainer.

Outputs:
    results/exp5_shap_plot.png
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
from vehiclepm.interpretability.shap_analysis import compute_shap_importance

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("EXP 5 — SHAP Feature Importance Analysis")
print("=" * 60)

# Generate and train
df = generate_synthetic_dataset(n_samples=2000, noise_sigma=1.0, random_state=42)
X = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
y = df["maintenance_needed"]

clf = VehiclePMClassifier(model_type="lightgbm")
clf.fit(X, y)

print("\nComputing SHAP values...")
importance_df = compute_shap_importance(clf.get_base_model(), X, max_display=15, plot=False)

print("\nTop 15 Features by Mean |SHAP Value|:")
print(importance_df.head(15).to_string(index=False))

# Identify which group each feature belongs to
contextual = ["weather_cond_Rain", "weather_cond_Snow", "weather_cond_Clear",
              "road_roughness", "ambient_temp", "traffic_density",
              "monthly_precipitation", "road_type_Urban", "road_type_Highway",
              "road_type_Rural", "traffic_road_impact", "brake_stress_idx",
              "engine_thermal_load", "engine_battery_ratio"]

top15 = importance_df.head(15)
top15["group"] = top15["feature"].apply(
    lambda f: "Contextual/Interaction" if f in contextual else "Internal/Driver"
)

n_contextual = (top15["group"] == "Contextual/Interaction").sum()
print(f"\nContextual/Interaction features in top 15: {n_contextual}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Experiment 5: SHAP Feature Importance (LightGBM Classifier)\n"
             "Colour = feature group: blue=internal/driver, orange=contextual/interaction",
             fontsize=11, fontweight="bold")

# Bar chart
colors_bar = ["#F4A582" if f in contextual else "#4393C3"
              for f in top15["feature"][::-1]]
ax1.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1],
         color=colors_bar, alpha=0.88)
ax1.set_xlabel("Mean |SHAP Value|")
ax1.set_title("(a) Top 15 Features by Mean |SHAP|")
ax1.grid(axis="x", alpha=0.3)

import matplotlib.patches as mpatches
p1 = mpatches.Patch(color="#4393C3", label="Internal / Driver")
p2 = mpatches.Patch(color="#F4A582", label="Contextual / Interaction (V2X)")
ax1.legend(handles=[p1, p2], fontsize=9)

# Feature group summary
groups = top15.groupby("group")["mean_abs_shap"].sum()
ax2.bar(groups.index, groups.values,
        color=["#4393C3", "#F4A582"], alpha=0.88, width=0.5)
ax2.set_title("(b) Total SHAP Importance by Feature Group")
ax2.set_ylabel("Sum of Mean |SHAP Values| (Top 15)")
for i, (g, v) in enumerate(groups.items()):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/exp5_shap_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  → Saved: results/exp5_shap_plot.png")

print("\n" + "=" * 60)
print("EXP 5 COMPLETE")
print(f"  Contextual features in top 15: {n_contextual}/15")
print("=" * 60)
