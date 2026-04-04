"""
train_synthetic.py  (v3)
========================
Code 1 — Trains the synthetic base model on drive-level summaries.

What's new in v3 vs v2:
  ─────────────────────────────────────────────────────────
  SERVICE EVENT SIMULATION
  ─────────────────────────────────────────────────────────
  ~30% of synthetic vehicles now receive a service event mid-sequence.
  After a service, wear components reset to near-new values and
  days_to_service correctly jumps back up. This teaches the base model
  what post-service patterns look like — critical for the fine-tuned model
  to generalise correctly when real service logs reset wear state.

  Without this, a base model trained purely on monotonic degradation will
  have zero representation of "healthy car after service" and will wrongly
  predict high risk for any car whose wear was recently reset.

  ─────────────────────────────────────────────────────────
  NEW FEATURE: service_applied
  ─────────────────────────────────────────────────────────
  Each drive-summary row carries a binary service_applied flag.
  The model learns: when service_applied=1, subsequent wear is healthy
  → lower risk, higher days_to_service.

Usage:
    python train_synthetic_v3.py

Output:
    models/synthetic/classifier.pkl
    models/synthetic/regressor.pkl
    models/synthetic/columns.json
    models/synthetic/report.txt
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.over_sampling import SMOTE

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "models", "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Generation parameters ─────────────────────────────────────────────────────
N_VEHICLES   = 400
N_DRIVES     = 14
RANDOM_SEED  = 42
SERVICE_PROB = 0.30   # fraction of vehicles that get a mid-sequence service

# ── Wear model — must match finetune_car_v3.py exactly ───────────────────────
WEAR_PER_KM = {
    "brake_thickness": 0.0012,
    "tire_tread":      0.0006,
    "oil_degradation": 0.00015,
}
THRESH = {
    "brake_thickness": 2.0,
    "tire_tread":      1.6,
    "oil_degradation": 0.85,
}

# Service reset defaults (mirrors service_log_utils.SERVICE_RESETS)
SERVICE_RESETS = {
    "oil_change":        {"oil_degradation": 0.05},
    "brake_replacement": {"brake_thickness": 9.0},
    "tire_replacement":  {"tire_tread":      7.5},
    "full_service":      {"brake_thickness": 9.0, "tire_tread": 7.5, "oil_degradation": 0.05},
}

# ── Feature columns ───────────────────────────────────────────────────────────
BASE_FCOLS = [
    "engine_temp", "fuel_level", "battery_health",
    "brake_thickness", "tire_tread", "oil_degradation",
    "mileage", "vehicle_age",
    "hard_braking_freq", "accel_variance", "idle_ratio",
    "ambient_temp", "road_roughness", "monthly_precipitation",
    "traffic_density", "wind_speed", "humidity", "pressure_hpa",
    "driving_style_enc", "weather_condition_enc", "road_type_enc",
    "engine_thermal_load", "brake_stress_idx",
    "traffic_road_impact", "engine_battery_ratio",
]
EXTRA_FCOLS = [
    "km_driven", "stress", "aggressive_frac", "stopgo_frac",
    "brake_wear_pct", "tire_wear_pct", "oil_wear_pct", "composite_wear_idx",
    "service_applied",   # ← v3: service event flag
]
ALL_FCOLS = BASE_FCOLS + EXTRA_FCOLS   # 34 total (was 33 in v2)


# ── Synthetic fleet generator ─────────────────────────────────────────────────

def generate_vehicle_fleet(n_vehicles, n_drives, seed=42):
    """
    Generate a synthetic fleet with optional mid-sequence service events.

    v3 change: ~30% of vehicles are serviced at a random drive in the
    second half of their sequence. Service resets wear components and the
    labels correctly recover — teaching the model what a healthy car looks
    like right after service.
    """
    rng  = np.random.RandomState(seed)
    rows = []

    REGIONS = {
        "india_urban":   dict(temp_mu=30, temp_sig=5,  roughness_mu=3.5, roughness_sig=1.5, precip_mu=40),
        "india_highway": dict(temp_mu=32, temp_sig=4,  roughness_mu=2.0, roughness_sig=0.8, precip_mu=30),
        "europe":        dict(temp_mu=14, temp_sig=8,  roughness_mu=1.5, roughness_sig=0.5, precip_mu=60),
        "us_urban":      dict(temp_mu=18, temp_sig=10, roughness_mu=2.5, roughness_sig=1.0, precip_mu=50),
        "middle_east":   dict(temp_mu=38, temp_sig=6,  roughness_mu=2.0, roughness_sig=0.8, precip_mu=5),
    }
    REGION_KEYS     = list(REGIONS.keys())
    WEATHER_OPTIONS = ["Clear", "Clear", "Clear", "Clouds", "Clouds", "Rain", "Haze"]
    WEATHER_ENC     = {"Clear":0, "Clouds":0, "Haze":0, "Rain":1, "Snow":2, "Fog":3}
    ROAD_OPTIONS    = ["Urban", "Urban", "Urban", "Highway", "Highway", "Rural"]
    ROAD_ENC        = {"Urban":0, "Highway":1, "Rural":2}
    STYLE_ENC       = {"Smooth":0, "Aggressive":1, "Stop-and-Go":2}
    SERVICE_TYPES   = list(SERVICE_RESETS.keys())

    for v in range(n_vehicles):
        # ── Vehicle parameters ─────────────────────────────────────────────
        age       = rng.randint(1, 16)
        annual_km = rng.uniform(8000, 30000)
        mileage_0 = age * annual_km * rng.uniform(0.7, 1.3)

        wear_tier = rng.choice(["fresh", "mid", "worn"], p=[0.40, 0.35, 0.25])
        if wear_tier == "fresh":
            brake_0, tire_0, oil_0 = rng.uniform(7.0, 10.0), rng.uniform(5.5, 8.0), rng.uniform(0.0, 0.25)
        elif wear_tier == "mid":
            brake_0, tire_0, oil_0 = rng.uniform(4.0, 7.0),  rng.uniform(3.0, 5.5), rng.uniform(0.25, 0.60)
        else:
            brake_0, tire_0, oil_0 = rng.uniform(2.1, 4.0),  rng.uniform(1.7, 3.0), rng.uniform(0.60, 0.84)

        brake_0 = float(np.clip(brake_0 + rng.normal(0, 0.3), 2.1, 12.0))
        tire_0  = float(np.clip(tire_0  + rng.normal(0, 0.2), 1.7, 10.0))
        oil_0   = float(np.clip(oil_0   + rng.normal(0, 0.03), 0.0, 0.84))

        battery_base = max(0.50, 1.0 - (age / 15) * 0.5 + rng.normal(0, 0.05))
        engine_base  = rng.uniform(83, 93)
        vehicle_age  = age
        dom_style    = rng.choice(["Smooth", "Aggressive", "Mixed", "Urban"],
                                   p=[0.30, 0.25, 0.25, 0.20])
        region_key   = rng.choice(REGION_KEYS)
        region       = REGIONS[region_key]

        # ── v3: Service event scheduling ───────────────────────────────────
        has_service    = rng.random() < SERVICE_PROB
        service_drive  = rng.randint(n_drives // 2, n_drives) if has_service else -1
        service_type   = rng.choice(SERVICE_TYPES) if has_service else None

        wear = {
            "brake_thickness": float(np.clip(brake_0, 2.1, 12.0)),
            "tire_tread":      float(np.clip(tire_0,  1.7, 10.0)),
            "oil_degradation": float(np.clip(oil_0,   0.0, 0.84)),
        }

        total_km        = 0.0
        total_stress_km = 0.0
        mileage_cur     = mileage_0

        for d in range(n_drives):
            # ── v3: Apply service before this drive if scheduled ───────────
            service_applied = 0
            if d == service_drive and service_type is not None:
                resets = SERVICE_RESETS[service_type]
                for comp, val in resets.items():
                    wear[comp] = val
                service_applied = 1

            # ── Drive parameters ───────────────────────────────────────────
            road_type = rng.choice(ROAD_OPTIONS)
            weather   = rng.choice(WEATHER_OPTIONS)
            km        = (rng.uniform(20, 80) if road_type == "Highway"
                         else rng.uniform(5, 30) if road_type == "Rural"
                         else rng.uniform(3, 20))

            if dom_style == "Smooth":
                aggressive_frac, stopgo_frac = rng.beta(1, 5), rng.beta(1, 4)
            elif dom_style == "Aggressive":
                aggressive_frac, stopgo_frac = rng.beta(3, 2), rng.beta(1, 4)
            elif dom_style == "Urban":
                aggressive_frac, stopgo_frac = rng.beta(1, 4), rng.beta(3, 2)
            else:   # Mixed
                aggressive_frac, stopgo_frac = rng.beta(2, 3), rng.beta(2, 3)

            total       = aggressive_frac + stopgo_frac
            if total > 1.0:
                aggressive_frac /= total; stopgo_frac /= total
            smooth_frac = 1.0 - aggressive_frac - stopgo_frac

            style_mult   = 1.0 + aggressive_frac * 0.5 + stopgo_frac * 0.3
            roughness    = max(0.5, rng.normal(region["roughness_mu"], region["roughness_sig"]))
            traffic      = np.clip(rng.normal(60 if road_type == "Urban" else 30, 20), 0, 100)
            hard_braking = np.clip(rng.normal(0.3 + aggressive_frac * 0.8, 0.2), 0, 2.5)
            road_mult    = 1.0 + (roughness - 2.0) * 0.06
            brake_mult   = 1.0 + hard_braking * 0.15
            traffic_mult = 1.0 + (traffic - 50) * 0.003
            stress       = max(0.8, style_mult * road_mult * brake_mult * traffic_mult)

            total_km        += km
            total_stress_km += km * stress
            mileage_cur     += km

            wear["brake_thickness"] = max(0.1, wear["brake_thickness"]
                                          - WEAR_PER_KM["brake_thickness"] * km * stress)
            wear["tire_tread"]      = max(0.1, wear["tire_tread"]
                                          - WEAR_PER_KM["tire_tread"] * km * stress)
            wear["oil_degradation"] = min(1.0, wear["oil_degradation"]
                                          + WEAR_PER_KM["oil_degradation"] * km * stress)

            n_days_elapsed  = max(1, d + 1)
            daily_km_avg    = max(5.0, total_km / n_days_elapsed)
            avg_stress      = max(1.0, total_stress_km / max(total_km, 1.0))

            days_brake = max(0,
                (wear["brake_thickness"] - THRESH["brake_thickness"])
                / (WEAR_PER_KM["brake_thickness"] * avg_stress * daily_km_avg)
            ) if wear["brake_thickness"] > THRESH["brake_thickness"] else 0
            days_tire  = max(0,
                (wear["tire_tread"] - THRESH["tire_tread"])
                / (WEAR_PER_KM["tire_tread"] * avg_stress * daily_km_avg)
            ) if wear["tire_tread"] > THRESH["tire_tread"] else 0
            days_oil   = max(0,
                (THRESH["oil_degradation"] - wear["oil_degradation"])
                / (WEAR_PER_KM["oil_degradation"] * avg_stress * daily_km_avg)
            ) if wear["oil_degradation"] < THRESH["oil_degradation"] else 0

            days_to_service = round(max(1, min(365, min(days_brake, days_tire, days_oil))), 1)

            ambient_temp     = float(np.clip(rng.normal(region["temp_mu"], region["temp_sig"]), -10, 55))
            precip           = max(0, rng.normal(region["precip_mu"], 15))
            wind_speed       = max(0, rng.normal(5, 3))
            humidity         = float(np.clip(rng.normal(60, 20), 10, 100))
            pressure_hpa     = float(np.clip(rng.normal(1010, 10), 960, 1060))
            accel_variance   = max(0.1, rng.normal(1.0 + aggressive_frac * 1.5, 0.5))
            idle_ratio       = float(np.clip(rng.normal(0.1 + stopgo_frac * 0.4, 0.05), 0, 0.8))
            fuel_level       = float(np.clip(rng.normal(0.5, 0.2), 0.05, 1.0))
            battery_health   = float(np.clip(battery_base + rng.normal(0, 0.03), 0.2, 1.0))
            engine_temp      = float(np.clip(engine_base + rng.normal(0, 3)
                                             + (ambient_temp - 25) * 0.2
                                             + traffic * 0.05, 60, 130))

            bt                   = wear["brake_thickness"]
            engine_thermal_load  = engine_temp * 0.5 + traffic * 0.3 + ambient_temp * 0.2
            brake_stress_idx     = hard_braking * 0.4 + roughness * 0.4 + (1 / max(0.1, bt)) * 0.2
            traffic_road_impact  = traffic * roughness
            engine_battery_ratio = engine_temp / max(0.1, battery_health)
            brake_wear_pct       = float(np.clip(1 - (bt - THRESH["brake_thickness"]) / (8.0 - THRESH["brake_thickness"]), 0, 1))
            tire_wear_pct        = float(np.clip(1 - (wear["tire_tread"] - THRESH["tire_tread"]) / (8.0 - THRESH["tire_tread"]), 0, 1))
            oil_wear_pct         = float(np.clip(wear["oil_degradation"] / THRESH["oil_degradation"], 0, 1))
            composite_wear       = brake_wear_pct * 0.4 + tire_wear_pct * 0.35 + oil_wear_pct * 0.25

            dominant_style  = ("Aggressive" if aggressive_frac > 0.5
                               else "Stop-and-Go" if stopgo_frac > 0.5
                               else "Smooth")

            rows.append({
                # Base features
                "engine_temp":           round(engine_temp, 2),
                "fuel_level":            round(fuel_level, 3),
                "battery_health":        round(battery_health, 3),
                "brake_thickness":       round(wear["brake_thickness"], 4),
                "tire_tread":            round(wear["tire_tread"], 4),
                "oil_degradation":       round(wear["oil_degradation"], 4),
                "mileage":               round(mileage_cur, 1),
                "vehicle_age":           vehicle_age,
                "hard_braking_freq":     round(hard_braking, 3),
                "accel_variance":        round(accel_variance, 3),
                "idle_ratio":            round(idle_ratio, 3),
                "ambient_temp":          round(ambient_temp, 1),
                "road_roughness":        round(roughness, 3),
                "monthly_precipitation": round(precip, 1),
                "traffic_density":       round(traffic, 1),
                "wind_speed":            round(wind_speed, 1),
                "humidity":              round(humidity, 1),
                "pressure_hpa":          round(pressure_hpa, 1),
                "driving_style_enc":     STYLE_ENC.get(dominant_style, 0),
                "weather_condition_enc": WEATHER_ENC.get(weather, 0),
                "road_type_enc":         ROAD_ENC.get(road_type, 0),
                "engine_thermal_load":   round(engine_thermal_load, 3),
                "brake_stress_idx":      round(brake_stress_idx, 3),
                "traffic_road_impact":   round(traffic_road_impact, 3),
                "engine_battery_ratio":  round(engine_battery_ratio, 3),
                # Extra drive-level features
                "km_driven":             round(km, 2),
                "stress":                round(stress, 3),
                "aggressive_frac":       round(aggressive_frac, 3),
                "stopgo_frac":           round(stopgo_frac, 3),
                "brake_wear_pct":        round(brake_wear_pct, 4),
                "tire_wear_pct":         round(tire_wear_pct, 4),
                "oil_wear_pct":          round(oil_wear_pct, 4),
                "composite_wear_idx":    round(composite_wear, 4),
                "service_applied":       service_applied,   # ← v3
                # Labels
                "days_to_service":       days_to_service,
                "maintenance_needed":    int(days_to_service <= 30),
                # Metadata
                "vehicle_id":            v,
                "drive_id":              d,
                "region":                region_key,
                "wear_tier":             wear_tier,
                "service_type":          service_type or "none",
            })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  train_synthetic.py v3 — Service-Event-Aware Base Model")
print(f"  Output: {OUT_DIR}")
print("=" * 60)

print(f"\n[1/4] Generating {N_VEHICLES} vehicles × {N_DRIVES} drives "
      f"= {N_VEHICLES * N_DRIVES:,} drive samples "
      f"(~{int(N_VEHICLES * SERVICE_PROB)} vehicles with service events)...")

df = generate_vehicle_fleet(N_VEHICLES, N_DRIVES, seed=RANDOM_SEED)

print(f"  Generated: {len(df):,} rows")
print(f"  Failure rate (<=30 days): {df['maintenance_needed'].mean():.1%}")
print(f"  days_to_service: {df['days_to_service'].min():.0f} – "
      f"{df['days_to_service'].max():.0f}  mean={df['days_to_service'].mean():.0f}")
print(f"  Drives with service_applied=1: {df['service_applied'].sum()} "
      f"({df['service_applied'].mean():.1%})")
print(f"  Service type distribution:\n"
      + "\n".join(f"    {k}: {v}" for k, v in
                  df[df["service_applied"] == 1]["service_type"].value_counts().items()))

print(f"\n[2/4] Building feature matrix...")
missing = [c for c in ALL_FCOLS if c not in df.columns]
if missing:
    print(f"  ⚠️  Missing columns: {missing}")
    sys.exit(1)

X     = df[ALL_FCOLS].apply(pd.to_numeric, errors="coerce").fillna(0)
y_cls = df["maintenance_needed"].astype(int)
y_reg = df["days_to_service"]

print(f"  Features: {len(ALL_FCOLS)}  Samples: {len(X):,}")

print(f"\n[3/4] Training classifier (5-fold stratified CV)...")
clf = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    max_depth=8, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, random_state=42, verbose=-1,
    class_weight="balanced"
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1, cv_auc = [], []

for fold, (tr, te) in enumerate(skf.split(X, y_cls)):
    Xt, Xe = X.iloc[tr], X.iloc[te]
    yt, ye = y_cls.iloc[tr], y_cls.iloc[te]
    if yt.sum() >= 5:
        sm  = SMOTE(random_state=42, k_neighbors=min(5, int(yt.sum()) - 1))
        Xt, yt = sm.fit_resample(Xt, yt)
    clf.fit(Xt, yt)
    f1  = f1_score(ye, clf.predict(Xe), average="macro", zero_division=0)
    auc = roc_auc_score(ye, clf.predict_proba(Xe)[:, 1])
    cv_f1.append(f1); cv_auc.append(auc)
    print(f"  Fold {fold+1}: F1={f1:.3f}  AUC={auc:.3f}")

print(f"  CV — F1: {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}  "
      f"AUC: {np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}")
clf.fit(X, y_cls)

print(f"\n[4/4] Training regressor (5-fold CV)...")
reg = LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    max_depth=8, subsample=0.8, random_state=42, verbose=-1
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae, cv_r2 = [], []

for fold, (tr, te) in enumerate(kf.split(X)):
    reg.fit(X.iloc[tr], y_reg.iloc[tr])
    preds = reg.predict(X.iloc[te])
    mae   = mean_absolute_error(y_reg.iloc[te], preds)
    r2    = r2_score(y_reg.iloc[te], preds)
    cv_mae.append(mae); cv_r2.append(r2)
    print(f"  Fold {fold+1}: MAE={mae:.1f} days  R²={r2:.4f}")

print(f"  CV — MAE: {np.mean(cv_mae):.1f} ± {np.std(cv_mae):.1f} days  "
      f"R²: {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
reg.fit(X, y_reg)

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n  Saving to {OUT_DIR}...")

with open(os.path.join(OUT_DIR, "classifier.pkl"), "wb") as f: pickle.dump(clf, f)
with open(os.path.join(OUT_DIR, "regressor.pkl"),  "wb") as f: pickle.dump(reg, f)

meta = {
    "feature_cols":  ALL_FCOLS,
    "base_fcols":    BASE_FCOLS,
    "extra_fcols":   EXTRA_FCOLS,
    "n_vehicles":    N_VEHICLES,
    "n_drives":      N_DRIVES,
    "n_samples":     len(df),
    "failure_rate":  round(float(y_cls.mean()), 4),
    "days_min":      float(y_reg.min()),
    "days_max":      float(y_reg.max()),
    "days_mean":     round(float(y_reg.mean()), 1),
    "model_type":    "drive_level_v3_service_aware",
    "service_event_pct": round(float(df["service_applied"].mean()), 4),
    "date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(os.path.join(OUT_DIR, "columns.json"), "w") as f:
    json.dump(meta, f, indent=2)

report = (
    f"Synthetic Base Model v3 — Service-Event-Aware\n"
    f"Date: {datetime.now()}\n"
    f"Vehicles: {N_VEHICLES}  Drives/vehicle: {N_DRIVES}  "
    f"Total samples: {len(df):,}\n"
    f"Service events simulated: {df['service_applied'].sum()} "
    f"({df['service_applied'].mean():.1%} of drives)\n"
    f"Features: {len(ALL_FCOLS)}\n\n"
    f"Classifier (5-fold CV):\n"
    f"  F1:  {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}\n"
    f"  AUC: {np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}\n\n"
    f"Regressor (5-fold CV):\n"
    f"  MAE: {np.mean(cv_mae):.1f} ± {np.std(cv_mae):.1f} days\n"
    f"  R²:  {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}\n"
)
with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
    f.write(report)

print(f"\n  ✅ Saved to: {OUT_DIR}")
print(f"  Classifier  F1: {np.mean(cv_f1):.3f}  AUC: {np.mean(cv_auc):.3f}")
print(f"  Regressor  MAE: {np.mean(cv_mae):.1f} days  R²: {np.mean(cv_r2):.4f}")
print("=" * 60)
