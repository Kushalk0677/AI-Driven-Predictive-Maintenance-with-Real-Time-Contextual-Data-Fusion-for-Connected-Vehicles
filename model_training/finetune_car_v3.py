"""
finetune_car.py  (v3)
=====================
Code 2 — Fine-tunes the synthetic base model per car.

What's new in v3 vs v2:
  ─────────────────────────────────────────────────────────
  SERVICE LOG SUPPORT
  ─────────────────────────────────────────────────────────
  Drop a service CSV (or JSON) in  data/service_logs/  for each car.
  The wear accumulation now resets at service events, so a car that had
  its oil changed or brakes replaced no longer gets stuck at days=1 forever.

  See  service_log_utils.py  for the full format description.

  ─────────────────────────────────────────────────────────
  FAULT-TOLERANT INITIAL WEAR
  ─────────────────────────────────────────────────────────
  If the first CSV already shows a component past its threshold (e.g., a
  fault code caused abnormal readings), a warning is printed and the state
  is recorded as-is.  Once a service log resets that component, subsequent
  drives immediately recover to a healthy label range.

  ─────────────────────────────────────────────────────────
  NEW FEATURE: service_applied
  ─────────────────────────────────────────────────────────
  Each drive-summary row now carries a binary  service_applied  column and
  a  services_done  string column.  The model can learn that high wear
  followed by a service event produces healthy future predictions.

Usage:
    python finetune_car_v3.py --list
    python finetune_car_v3.py --car "Safari Storme EX"
    python finetune_car_v3.py --all

Output:
    models/finetuned/<Car_Name>/classifier.pkl
    models/finetuned/<Car_Name>/regressor.pkl
    models/finetuned/<Car_Name>/report.txt
    models/finetuned/<Car_Name>/drive_summary.csv
"""

import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (f1_score, roc_auc_score,
                             mean_absolute_error, r2_score)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier, LGBMRegressor
from imblearn.over_sampling import SMOTE

# ── Import service-log helpers ────────────────────────────────────────────────
try:
    from service_log_utils import (load_service_log, get_drive_date,
                                   apply_pending_services)
except ImportError:
    print("  ⚠️  service_log_utils.py not found — service log support disabled.")
    def load_service_log(*a, **k): return []
    def get_drive_date(p): return None
    def apply_pending_services(wear, evs, *a, **k): return wear, []

MIN_DRIVES_FOR_LGBM_REG = 20

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(HERE, "data", "drives")
SERVICE_DIR = os.path.join(HERE, "data", "service_logs")
FT_DIR      = os.path.join(HERE, "models", "finetuned")
SYN_DIR     = os.path.join(HERE, "models", "synthetic")
os.makedirs(FT_DIR,        exist_ok=True)
os.makedirs(SERVICE_DIR,   exist_ok=True)

# ── Encodings ─────────────────────────────────────────────────────────────────
STYLE_MAP   = {"Smooth":0, "Aggressive":1, "Stop-and-Go":2}
WEATHER_MAP = {"Clear":0, "Clouds":0, "Cloud":0, "Cloudy":0, "Haze":0,
               "Rain":1, "Drizzle":1, "Thunderstorm":1,
               "Snow":2, "Mist":3, "Fog":3}
ROAD_MAP    = {"Urban":0, "Highway":1, "Rural":2}

# ── Wear model ────────────────────────────────────────────────────────────────
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
DEFAULT_DAILY_KM = 30.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_base():
    for clf_p, reg_p, col_p in [
        (os.path.join(SYN_DIR, "classifier.pkl"),
         os.path.join(SYN_DIR, "regressor.pkl"),
         os.path.join(SYN_DIR, "columns.json")),
        (os.path.join(HERE, "models", "base_classifier.pkl"),
         os.path.join(HERE, "models", "base_regressor.pkl"),
         os.path.join(HERE, "models", "base_columns.json")),
    ]:
        if all(os.path.exists(p) for p in [clf_p, reg_p, col_p]):
            with open(clf_p, "rb") as f: clf = pickle.load(f)
            with open(reg_p, "rb") as f: reg = pickle.load(f)
            with open(col_p)       as f: meta = json.load(f)
            print(f"  Base model: {os.path.dirname(clf_p)}")
            return clf, reg, meta
    print("  ❌ No base model found. Run train_synthetic_v3.py first.")
    sys.exit(1)


def get_car_name(df, path):
    if "vehicle_name" in df.columns:
        n = str(df["vehicle_name"].iloc[0]).strip()
        if n and n != "nan":
            return n
    parts = os.path.basename(path).replace(".csv", "").split("_")
    return " ".join(parts[2:]) if len(parts) > 2 else "Unknown"


def group_by_car():
    groups = defaultdict(list)
    for f in sorted(glob(os.path.join(DATA_DIR, "*.csv"))):
        try:
            df_tmp = pd.read_csv(f, nrows=1)
            if len(df_tmp) == 0:
                continue
            groups[get_car_name(df_tmp, f)].append(f)
        except Exception:
            pass
    return groups


def to_numeric(df):
    skip = {"driving_style", "weather_condition", "road_type", "vehicle_name",
            "timestamp", "city", "trouble_codes", "severity", "needs_maintenance",
            "weather_raw", "service_event", "date", "actual_daily_km", "source_file"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def compute_stress(row):
    style_mult  = (1.0
                   + row.get("aggressive_frac", 0) * 0.5
                   + row.get("stopgo_frac",     0) * 0.3)
    road_mult   = 1.0 + (row.get("avg_road_roughness", 2.0) - 2.0) * 0.06
    brake_mult  = 1.0 + row.get("avg_hard_braking", 0.5) * 0.15
    traffic_mult= 1.0 + (row.get("avg_traffic_density", 50) - 50) * 0.003
    return max(0.8, style_mult * road_mult * brake_mult * traffic_mult)


def aggregate_drive(df, source_file):
    df = df.copy()
    df = to_numeric(df)

    mileage   = df["mileage"].clip(lower=0)
    km_driven = float(mileage.iloc[-1] - mileage.iloc[0])
    if km_driven <= 0:
        speed = df["speed"].clip(lower=0)
        dt    = df["elapsed_seconds"].diff().fillna(2.0)
        km_driven = float((speed * dt / 3600).sum())
    km_driven = max(1.0, km_driven)

    style_counts    = df["driving_style"].fillna("Smooth").value_counts(normalize=True)
    aggressive_frac = float(style_counts.get("Aggressive",  0))
    stopgo_frac     = float(style_counts.get("Stop-and-Go", 0))
    smooth_frac     = float(style_counts.get("Smooth",      0))
    dominant_style  = style_counts.index[0] if len(style_counts) > 0 else "Smooth"

    def safe_mean(col, default=0.0):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            return float(v.mean()) if v.notna().any() else default
        return default

    def safe_last(col, default=0.0):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(v.iloc[-1]) if len(v) > 0 else default
        return default

    def safe_first(col, default=0.0):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(v.iloc[0]) if len(v) > 0 else default
        return default

    weather_mode = (df["weather_condition"].fillna("Clear").mode()[0]
                    if "weather_condition" in df.columns else "Clear")
    road_mode    = (df["road_type"].fillna("Urban").mode()[0]
                    if "road_type" in df.columns else "Urban")

    row = {
        "source_file":           source_file,
        "km_driven":             round(km_driven, 2),
        "engine_temp":           safe_mean("engine_temp", 85),
        "fuel_level":            safe_mean("fuel_level", 0.5),
        "battery_health":        safe_mean("battery_health", 0.8),
        "mileage":               safe_last("mileage", 0),
        "vehicle_age":           safe_first("vehicle_age", 5),
        "brake_thickness":       safe_last("brake_thickness", 6.0),
        "tire_tread":            safe_last("tire_tread", 5.0),
        "oil_degradation":       safe_last("oil_degradation", 0.3),
        "hard_braking_freq":     safe_mean("hard_braking_freq", 0.5),
        "accel_variance":        safe_mean("accel_variance", 1.0),
        "idle_ratio":            safe_mean("idle_ratio", 0.1),
        "aggressive_frac":       round(aggressive_frac, 3),
        "stopgo_frac":           round(stopgo_frac, 3),
        "smooth_frac":           round(smooth_frac, 3),
        "ambient_temp":          safe_mean("ambient_temp", 28),
        "road_roughness":        safe_mean("road_roughness", 2.5),
        "monthly_precipitation": safe_mean("monthly_precipitation", 20),
        "traffic_density":       safe_mean("traffic_density", 50),
        "wind_speed":            safe_mean("wind_speed", 4),
        "humidity":              safe_mean("humidity", 50),
        "pressure_hpa":          safe_mean("pressure_hpa", 1010),
        "driving_style_enc":     STYLE_MAP.get(dominant_style, 0),
        "weather_condition_enc": WEATHER_MAP.get(weather_mode, 0),
        "road_type_enc":         ROAD_MAP.get(road_mode, 0),
        "avg_road_roughness":    safe_mean("road_roughness", 2.5),
        "avg_hard_braking":      safe_mean("hard_braking_freq", 0.5),
        "avg_traffic_density":   safe_mean("traffic_density", 50),
    }
    return row


# ── Cumulative wear tracking (v3 — service-log aware) ─────────────────────────

def build_drive_summary(car_name: str, csv_files: list) -> pd.DataFrame:
    """
    Process drives in chronological order carrying wear state forward.

    NEW in v3:
      • Loads the car's service log from data/service_logs/
      • Before each drive, applies any service events that occurred between
        the previous drive date and this drive date, resetting wear components
      • Records service_applied and services_done in the output row so
        training knows when a reset happened
      • Warns if initial wear is already past threshold (bad sensor / fault)
    """
    sorted_files = sorted(csv_files)

    # ── Load service log ───────────────────────────────────────────────────
    service_events = load_service_log(car_name, SERVICE_DIR, verbose=True)

    # ── Read initial wear state from first file ────────────────────────────
    df0 = pd.read_csv(sorted_files[0], low_memory=False)
    df0 = to_numeric(df0)

    def first_valid(df, col, default):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            return float(v.iloc[0]) if len(v) > 0 else default
        return default

    wear = {
        "brake_thickness": first_valid(df0, "brake_thickness", 8.0),
        "tire_tread":      first_valid(df0, "tire_tread",      6.0),
        "oil_degradation": first_valid(df0, "oil_degradation", 0.1),
        "battery_health":  first_valid(df0, "battery_health",  0.8),
    }

    # Warn if initial state is already past threshold
    for comp, thresh in THRESH.items():
        val = wear.get(comp, 0)
        is_worn = (val <= thresh) if comp != "oil_degradation" else (val >= thresh)
        if is_worn:
            print(f"     ⚠️  Initial {comp} ({val:.3f}) already at/past threshold "
                  f"({thresh}) — add a service log entry to reset after service")

    rows            = []
    total_km        = 0.0
    total_stress_km = 0.0
    prev_date       = None   # date of previous drive (for service event windowing)

    for i, fpath in enumerate(sorted_files):
        df = pd.read_csv(fpath, low_memory=False)
        if len(df) == 0:
            print(f"     ⚠️  {os.path.basename(fpath)} empty — skip")
            continue

        curr_date = get_drive_date(fpath)   # pd.Timestamp or None

        # ── Apply service events between last drive and this drive ─────────
        wear, applied_services = apply_pending_services(
            wear, service_events, prev_date, curr_date, drive_index=i
        )
        if applied_services:
            print(f"     🔧 Service applied before drive {i+1} "
                  f"({os.path.basename(fpath)[:40]}): "
                  f"{', '.join(applied_services)}")
            print(f"        Wear reset → brake: {wear['brake_thickness']:.3f}mm  "
                  f"tread: {wear['tire_tread']:.3f}mm  "
                  f"oil: {wear['oil_degradation']:.3f}  "
                  f"battery: {wear['battery_health']:.3f}")

        # ── Aggregate this drive ───────────────────────────────────────────
        row = aggregate_drive(df, os.path.basename(fpath))

        stress = compute_stress(row)
        km     = row["km_driven"]

        total_km        += km
        total_stress_km += km * stress

        # Degrade wear for this drive (from the post-service state)
        wear["brake_thickness"] = max(
            0.1, wear["brake_thickness"] - WEAR_PER_KM["brake_thickness"] * km * stress
        )
        wear["tire_tread"] = max(
            0.1, wear["tire_tread"] - WEAR_PER_KM["tire_tread"] * km * stress
        )
        wear["oil_degradation"] = min(
            1.0, wear["oil_degradation"] + WEAR_PER_KM["oil_degradation"] * km * stress
        )

        # Override per-component values in the row with cumulative wear
        row["brake_thickness"] = round(wear["brake_thickness"], 4)
        row["tire_tread"]      = round(wear["tire_tread"],      4)
        row["oil_degradation"] = round(wear["oil_degradation"], 4)
        # battery_health is not degraded by km in this model — keep CSV value

        # ── Rolling daily_km average ───────────────────────────────────────
        n_days_elapsed = max(1, i + 1)
        daily_km_avg   = max(5.0, total_km / n_days_elapsed)
        avg_stress     = max(1.0, total_stress_km / max(total_km, 1.0))

        # ── Compute days_to_service from cumulative wear ───────────────────
        days_brake = max(0,
            (wear["brake_thickness"] - THRESH["brake_thickness"])
            / (WEAR_PER_KM["brake_thickness"] * avg_stress * daily_km_avg)
        ) if wear["brake_thickness"] > THRESH["brake_thickness"] else 0

        days_tire = max(0,
            (wear["tire_tread"] - THRESH["tire_tread"])
            / (WEAR_PER_KM["tire_tread"] * avg_stress * daily_km_avg)
        ) if wear["tire_tread"] > THRESH["tire_tread"] else 0

        days_oil = max(0,
            (THRESH["oil_degradation"] - wear["oil_degradation"])
            / (WEAR_PER_KM["oil_degradation"] * avg_stress * daily_km_avg)
        ) if wear["oil_degradation"] < THRESH["oil_degradation"] else 0

        days_to_service = round(max(1, min(365, min(days_brake, days_tire, days_oil))), 1)

        row["days_to_service"]  = days_to_service
        row["failure_binary"]   = int(days_to_service <= 30)
        row["stress"]           = round(stress, 3)
        row["avg_stress"]       = round(avg_stress, 3)
        row["daily_km_avg"]     = round(daily_km_avg, 2)
        row["total_km"]         = round(total_km, 2)
        row["drive_index"]      = i
        # ── Service log fields (v3) ────────────────────────────────────────
        row["service_applied"]  = int(bool(applied_services))
        row["services_done"]    = "|".join(applied_services)
        row["drive_date"]       = str(curr_date.date()) if curr_date else ""

        rows.append(row)

        srv_tag = f" 🔧{'+'.join(applied_services)}" if applied_services else ""
        print(f"     Drive {i+1:>2}: {os.path.basename(fpath)[:40]:<40} "
              f"km={km:5.1f}  stress={stress:.2f}  "
              f"brake={wear['brake_thickness']:.3f}mm  "
              f"days={days_to_service:.0f}{srv_tag}")

        prev_date = curr_date

    return pd.DataFrame(rows)


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_drive_features(df):
    df = df.copy()
    df["engine_thermal_load"]  = (df["engine_temp"] * 0.5
                                  + df["avg_traffic_density"] * 0.3
                                  + df["ambient_temp"] * 0.2)
    df["brake_stress_idx"]     = (df["hard_braking_freq"] * 0.4
                                  + df["avg_road_roughness"] * 0.4
                                  + (1 / df["brake_thickness"].clip(lower=0.1)) * 0.2)
    df["traffic_road_impact"]  = df["avg_traffic_density"] * df["avg_road_roughness"]
    df["engine_battery_ratio"] = (df["engine_temp"]
                                  / df["battery_health"].clip(lower=0.1))
    df["brake_wear_pct"] = (1 - (df["brake_thickness"] - THRESH["brake_thickness"])
                            / (8.0 - THRESH["brake_thickness"])).clip(0, 1)
    df["tire_wear_pct"]  = (1 - (df["tire_tread"] - THRESH["tire_tread"])
                            / (8.0 - THRESH["tire_tread"])).clip(0, 1)
    df["oil_wear_pct"]   = (df["oil_degradation"] / THRESH["oil_degradation"]).clip(0, 1)
    df["composite_wear_idx"] = (df["brake_wear_pct"] * 0.4
                                + df["tire_wear_pct"]  * 0.35
                                + df["oil_wear_pct"]   * 0.25)
    return df


def make_X(df, fcols):
    X = pd.DataFrame(index=df.index)
    for c in fcols:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0
    return X.fillna(0)


# ── Fine-tune ─────────────────────────────────────────────────────────────────

def finetune(car_name, csv_files, base_clf, base_reg, meta):
    safe    = car_name.replace(" ", "_").replace("/", "_")
    car_dir = os.path.join(FT_DIR, safe)
    os.makedirs(car_dir, exist_ok=True)

    print(f"\n  🚗 {car_name}  ({len(csv_files)} drive file(s))")

    drive_df = build_drive_summary(car_name, csv_files)

    if len(drive_df) == 0:
        print("     ❌ No valid drives — skip")
        return None, None

    drive_df = engineer_drive_features(drive_df)
    drive_df.to_csv(os.path.join(car_dir, "drive_summary.csv"), index=False)

    n_serviced = int(drive_df["service_applied"].sum())
    print(f"\n     Drive summary: {len(drive_df)} drives  ({n_serviced} with service reset)")
    print(f"     days_to_service: {drive_df['days_to_service'].min():.0f} – "
          f"{drive_df['days_to_service'].max():.0f}")
    print(f"     failure_binary: {drive_df['failure_binary'].sum()} urgent drives")

    base_fcols = meta["feature_cols"]
    extra_cols = [
        "km_driven", "stress", "aggressive_frac", "stopgo_frac",
        "brake_wear_pct", "tire_wear_pct", "oil_wear_pct", "composite_wear_idx",
        "service_applied",   # ← v3: model learns post-service behaviour
    ]
    all_fcols = list(dict.fromkeys(base_fcols + extra_cols))

    X_base = make_X(drive_df, base_fcols)
    X_all  = make_X(drive_df, all_fcols)
    y_cls  = drive_df["failure_binary"].astype(int)
    y_reg  = drive_df["days_to_service"]

    print(f"     Training on all {len(drive_df)} drives")

    # ── Classifier ────────────────────────────────────────────────────────
    X_ft_base, y_ft = X_base.copy(), y_cls.copy()

    if y_cls.nunique() < 2:
        print("     ⚠️  No urgent drives — injecting near-threshold samples")
        n_s       = max(3, len(X_base) // 5)
        worst_idx = drive_df["composite_wear_idx"].nlargest(
            min(5, len(drive_df))
        ).index
        Xs_base = X_base.loc[worst_idx].iloc[
            np.random.choice(len(worst_idx), n_s, replace=True)
        ].copy()
        for col in ["brake_wear_pct", "tire_wear_pct", "oil_wear_pct"]:
            if col in Xs_base.columns:
                Xs_base[col] = np.random.uniform(0.88, 1.0, len(Xs_base))
        X_ft_base = pd.concat([X_base, Xs_base], ignore_index=True)
        y_ft      = pd.concat([y_cls, pd.Series([1] * n_s)], ignore_index=True)

    elif y_cls.sum() >= 3:
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(3, int(y_cls.sum()) - 1))
            X_ft_base, y_ft = sm.fit_resample(X_base, y_cls)
        except Exception as e:
            print(f"     SMOTE skip: {e}")

    ft_clf = LGBMClassifier(
        n_estimators=100, learning_rate=0.02, num_leaves=15,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=max(1, len(X_ft_base) // 10),
        random_state=42, verbose=-1, class_weight="balanced"
    )
    ft_clf.fit(X_ft_base, y_ft, init_model=base_clf.booster_)

    X_ft_all     = make_X(drive_df, all_fcols)
    X_ft_all_aug = X_ft_all.copy()
    y_ft_all     = y_cls.copy()

    if y_cls.nunique() >= 2 and y_cls.sum() >= 3 and len(X_ft_all) >= 4:
        try:
            sm2 = SMOTE(random_state=43, k_neighbors=min(3, int(y_cls.sum()) - 1))
            X_ft_all_aug, y_ft_all = sm2.fit_resample(X_ft_all, y_cls)
        except Exception:
            pass

    ft_clf_full = LGBMClassifier(
        n_estimators=100, learning_rate=0.01, num_leaves=15,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=max(1, len(X_ft_all_aug) // 10),
        random_state=42, verbose=-1, class_weight="balanced"
    )
    ft_clf_full.fit(X_ft_all_aug, y_ft_all)
    ft_clf = ft_clf_full

    # ── Regressor ─────────────────────────────────────────────────────────
    n_drives  = len(drive_df)
    reg_type  = ""

    if n_drives < MIN_DRIVES_FOR_LGBM_REG:
        print(f"     Regressor: Ridge (n={n_drives} < {MIN_DRIVES_FOR_LGBM_REG})")
        ft_reg  = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        ft_reg.fit(X_all, y_reg)
        reg_type = "Ridge"
    else:
        print(f"     Regressor: LightGBM (n={n_drives})")
        ft_reg_base = LGBMRegressor(
            n_estimators=100, learning_rate=0.02, num_leaves=15,
            max_depth=4, subsample=0.8, random_state=42, verbose=-1
        )
        ft_reg_base.fit(X_base, y_reg, init_model=base_reg.booster_)
        ft_reg = LGBMRegressor(
            n_estimators=100, learning_rate=0.01, num_leaves=15,
            max_depth=4, subsample=0.8, random_state=42, verbose=-1
        )
        ft_reg.fit(X_all, y_reg)
        reg_type = "LightGBM"

    # ── Training metrics ──────────────────────────────────────────────────
    try:
        tr_f1  = f1_score(y_cls, ft_clf.predict(X_ft_all),
                          average="macro", zero_division=0)
        tr_auc = (roc_auc_score(y_cls, ft_clf.predict_proba(X_ft_all)[:, 1])
                  if y_cls.nunique() > 1 else 0.5)
    except Exception:
        tr_f1, tr_auc = 0.0, 0.5

    tr_mae = mean_absolute_error(y_reg, ft_reg.predict(X_all))
    tr_r2  = r2_score(y_reg, ft_reg.predict(X_all))

    print(f"\n     [Train]  F1: {tr_f1:.3f}  AUC: {tr_auc:.3f}  "
          f"MAE: {tr_mae:.1f} days  R²: {tr_r2:.4f}")
    print(f"     days range: {y_reg.min():.0f} – {y_reg.max():.0f} days")

    # ── Save ──────────────────────────────────────────────────────────────
    with open(os.path.join(car_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(ft_clf, f)
    with open(os.path.join(car_dir, "regressor.pkl"), "wb") as f:
        pickle.dump(ft_reg, f)

    meta_ft = dict(meta)
    meta_ft["feature_cols"] = all_fcols
    with open(os.path.join(car_dir, "columns.json"), "w") as f:
        json.dump(meta_ft, f, indent=2)

    report = (
        f"Car: {car_name}\n"
        f"Date: {datetime.now()}\n"
        f"Drives: {n_drives}  (with service resets: {n_serviced})\n"
        f"Regressor: {reg_type}\n"
        f"days_to_service range: {y_reg.min():.0f} – {y_reg.max():.0f} days\n\n"
        f"Training metrics (reference):\n"
        f"  Classifier  F1: {tr_f1:.3f}  AUC: {tr_auc:.3f}\n"
        f"  Regressor  MAE: {tr_mae:.1f} days  R²: {tr_r2:.4f}\n"
    )
    with open(os.path.join(car_dir, "report.txt"), "w") as f:
        f.write(report)

    print(f"\n     ✅ Saved to: {car_dir}")
    return tr_f1, tr_mae


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--car",  default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all",  action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  finetune_car.py v3 — Per-Car Fine-Tuning")
    print("  (service logs · cumulative wear · drive aggregation)")
    print("=" * 60)

    groups = group_by_car()

    if args.list:
        print(f"\n  Cars in {DATA_DIR}:\n")
        for car, files in sorted(groups.items()):
            safe = car.replace(" ", "_").replace("/", "_")
            svc  = os.path.exists(os.path.join(SERVICE_DIR, f"{safe}_service.csv")) or \
                   os.path.exists(os.path.join(SERVICE_DIR, f"{safe}_service.json"))
            tag  = " 📋 has service log" if svc else ""
            print(f"  🚗 {car}  ({len(files)} drive(s)){tag}")
        sys.exit(0)

    print("\n  Loading base model...")
    base_clf, base_reg, meta = load_base()

    if args.all:
        results = {}
        for car, files in sorted(groups.items()):
            f1, mae = finetune(car, files, base_clf, base_reg, meta)
            if f1 is not None:
                results[car] = {"f1": f1, "mae": mae}
        print(f"\n{'=' * 60}")
        print("  Summary:")
        for car, r in sorted(results.items()):
            print(f"  {car:<35} F1: {r['f1']:.3f}  MAE: {r['mae']:.1f} days")

    elif args.car:
        matched = {k: v for k, v in groups.items()
                   if args.car.lower() in k.lower()}
        if not matched:
            print(f"\n  ❌ No drives found for '{args.car}'")
            print(f"  Available: {list(groups.keys())}")
            sys.exit(1)
        for car, files in matched.items():
            finetune(car, files, base_clf, base_reg, meta)
    else:
        print("\n  python finetune_car_v3.py --list")
        print("  python finetune_car_v3.py --car 'Safari Storme EX'")
        print("  python finetune_car_v3.py --all")

    print("\n" + "=" * 60 + "\n  Done.\n" + "=" * 60)
