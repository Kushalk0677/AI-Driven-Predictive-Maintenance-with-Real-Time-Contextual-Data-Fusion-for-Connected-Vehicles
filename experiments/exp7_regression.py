"""
=============================================================================
EXP 7 — Service Time Regression
=============================================================================
Predicts days until next service using LightGBM, XGBoost, Random Forest.

Note: High R² reflects simulation-domain performance only.
Real-world service time prediction requires field validation.

Outputs:
    results/exp7_regression_results.csv
    results/exp7_regression_plot.png
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

from vehiclepm.data.synthetic import generate_synthetic_dataset
from vehiclepm.features import build_feature_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("Install lightgbm and xgboost: pip install lightgbm xgboost")

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("EXP 7 — Service Time Regression")
print("=" * 60)

# Generate synthetic data with service time target
df = generate_synthetic_dataset(n_samples=1500, noise_sigma=1.0, random_state=42)

# Engineer service time from risk score (0–365 days, inverse of risk)
df["days_to_service"] = np.clip(
    365 * (1 - (df["risk_score"] - df["risk_score"].min()) /
           (df["risk_score"].max() - df["risk_score"].min() + 1e-6)),
    0, 365
).round(1)

X = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed", "days_to_service"]))
y = df["days_to_service"]

print(f"\n  Feature matrix: {X.shape}")
print(f"  Target range:   {y.min():.0f}–{y.max():.0f} days")

# Time-aware 70/30 split
split = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

MODELS = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "XGBoost":       XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                   random_state=42, verbosity=0),
    "LightGBM":      LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                    num_leaves=31, random_state=42, verbose=-1),
}

results = []
trained = {}

for name, model in MODELS.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    results.append({"Model": name, "RMSE (days)": round(rmse, 2),
                    "MAE (days)": round(mae, 2), "R²": round(r2, 4)})
    trained[name] = (model, y_pred)
    print(f"  {name}: RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")

df_res = pd.DataFrame(results)
df_res.to_csv("results/exp7_regression_results.csv", index=False)
print(f"\n{df_res.to_string(index=False)}")
print("\n  → Saved: results/exp7_regression_results.csv")
print("\n  ⚠️  Note: R² reflects simulation-domain performance only.")
print("       Real-world generalisation requires field validation.")

# Plot — best model
best_name = df_res.loc[df_res["R²"].idxmax(), "Model"]
_, y_pred_best = trained[best_name]
residuals = y_test.values - y_pred_best

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Experiment 7: Service Time Regression — {best_name}\n"
             "Note: R² reflects simulation-domain performance only",
             fontsize=11, fontweight="bold")

ax1.scatter(y_test, y_pred_best, alpha=0.4, s=15, color="#4393C3")
lims = [0, 365]
ax1.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax1.set_xlabel("Actual Days Until Service")
ax1.set_ylabel("Predicted Days Until Service")
ax1.set_title("(a) Actual vs. Predicted")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.hist(residuals, bins=40, color="#4393C3", alpha=0.8, edgecolor="white")
ax2.axvline(0, color="red", lw=2, linestyle="--")
ax2.set_xlabel("Residual (Predicted − Actual, days)")
ax2.set_ylabel("Count")
ax2.set_title(f"(b) Residual Distribution\n(MAE={df_res.loc[df_res['Model']==best_name,'MAE (days)'].values[0]} days)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/exp7_regression_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: results/exp7_regression_plot.png")

print("\n" + "=" * 60)
print("EXP 7 COMPLETE")
print("=" * 60)
