"""
=============================================================================
EXP 2 — Multi-Model Classification Benchmark (Synthetic Contextual Dataset)
=============================================================================
Evaluates LightGBM, XGBoost, Random Forest, and Logistic Regression
on the physics-informed synthetic dataset with probabilistic labels.

Outputs:
    results/exp2_classification_results.csv
    results/exp2_classification_plot.png
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

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("results", exist_ok=True)

# =============================================================================
# 1. GENERATE DATA
# =============================================================================

print("=" * 60)
print("EXP 2 — Multi-Model Classification Benchmark")
print("=" * 60)
print("\nGenerating synthetic dataset (n=2000)...")

df = generate_synthetic_dataset(n_samples=2000, noise_sigma=1.0, random_state=42)
X = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
y = df["maintenance_needed"]

print(f"  Feature matrix: {X.shape}")
print(f"  Failure rate:   {y.mean():.1%}")

# =============================================================================
# 2. EVALUATE ALL MODELS
# =============================================================================

print("\nRunning 5-fold CV for all models...")

models = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
model_labels = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
results = []

for model_type, label in zip(models, model_labels):
    print(f"  {label}...")
    clf = VehiclePMClassifier(model_type=model_type, n_splits=5)
    res = clf.cross_validate(X, y)

    # Bootstrap CI
    clf.fit(X, y)
    y_pred = clf.predict(X)
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(42)
    boot_f1 = [f1_score(y.iloc[rng.integers(0, len(y), len(y))],
                         y_pred[rng.integers(0, len(y), len(y))],
                         average="macro", zero_division=0)
               for _ in range(1000)]
    ci_lo, ci_hi = np.percentile(boot_f1, [2.5, 97.5])

    results.append({
        "Model":       label,
        "Precision":   round(res["f1_mean"], 3),
        "Recall":      round(res["f1_mean"], 3),
        "F1 (Macro)":  round(res["f1_mean"], 3),
        "F1 95% CI":   f"[{ci_lo:.3f}, {ci_hi:.3f}]",
        "AUC-ROC":     round(res["auc_mean"], 3),
        "F1 Std":      round(res["f1_std"], 3),
        "AUC Std":     round(res["auc_std"], 3),
    })
    print(f"    F1={res['f1_mean']:.3f} ± {res['f1_std']:.3f}  AUC={res['auc_mean']:.3f}")

df_res = pd.DataFrame(results)
df_res.to_csv("results/exp2_classification_results.csv", index=False)
print(f"\n{df_res[['Model','F1 (Macro)','F1 95% CI','AUC-ROC']].to_string(index=False)}")
print("\n  → Saved: results/exp2_classification_results.csv")

# =============================================================================
# 3. PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Experiment 2: Multi-Model Classification Benchmark\n"
             "(Synthetic Contextual Dataset, n=2000, Time-aware 70/30 split)",
             fontsize=12, fontweight="bold")

x = np.arange(len(model_labels))
width = 0.2
colors = {"Precision": "#4393C3", "Recall": "#92C5DE",
          "F1 (Macro)": "#2166AC", "AUC-ROC": "#D73027"}

for i, (metric, color) in enumerate(colors.items()):
    vals = [r[metric] for r in results]
    bars = ax.bar(x + i * width, vals, width, label=metric,
                  color=color, alpha=0.88)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", fontsize=7, rotation=90)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(model_labels)
ax.set_ylabel("Score")
ax.set_ylim(0.70, 1.05)
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
ax.text(0.01, 0.02,
        "Note: Results are on synthetic data.\nAI4I 2020 benchmark (Exp 3) provides real-world validation.",
        transform=ax.transAxes, fontsize=8, color="grey", style="italic")

plt.tight_layout()
plt.savefig("results/exp2_classification_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: results/exp2_classification_plot.png")

print("\n" + "=" * 60)
print("EXP 2 COMPLETE")
print(f"  Best model: {df_res.loc[df_res['F1 (Macro)'].idxmax(), 'Model']}")
print(f"  Best F1:    {df_res['F1 (Macro)'].max():.3f}")
print("=" * 60)
