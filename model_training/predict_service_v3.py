"""
predict_service.py  (v3)
========================
Code 3 — Predicts maintenance needs per car using drive-level aggregation.

What's new in v3 vs v2:
  ─────────────────────────────────────────────────────────
  SERVICE LOG SUPPORT  (fixes the "stuck at 1 day" bug)
  ─────────────────────────────────────────────────────────
  The root cause: v2 wear accumulation is monotonically degrading. If a
  fault or worn component shows up in drive CSVs, the model pegs
  days_to_service = 1 forever — even after a service fixes the car.

  v3 loads a service log for each car from  data/service_logs/  and
  resets wear components at the exact chronological point a service
  occurred.  After an oil change, a brake replacement, or a full service,
  the subsequent drives correctly recover to healthy day ranges.

  ─────────────────────────────────────────────────────────
  NEW OUTPUT COLUMNS
  ─────────────────────────────────────────────────────────
  service_applied  — 1 if a service was applied before this drive
  services_done    — pipe-separated list of service types applied
  drive_date       — date extracted from filename (if available)

Usage:
    python predict_service_v3.py              # all cars
    python predict_service_v3.py --car "Safari Storme EX"
    python predict_service_v3.py --file data/drives/my_drive.csv
    python predict_service_v3.py --eval       # show metrics vs proxy labels
"""

import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, r2_score

# ── Import service-log helpers ────────────────────────────────────────────────
try:
    from service_log_utils import (load_service_log, get_drive_date,
                                   apply_pending_services)
except ImportError:
    print("  ⚠️  service_log_utils.py not found — service log support disabled.")
    def load_service_log(*a, **k): return []
    def get_drive_date(p): return None
    def apply_pending_services(wear, evs, *a, **k): return wear, []

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(HERE, "data", "drives")
SERVICE_DIR = os.path.join(HERE, "data", "service_logs")
FT_DIR      = os.path.join(HERE, "models", "finetuned")
SYN_DIR     = os.path.join(HERE, "models", "synthetic")
RES_DIR     = os.path.join(HERE, "results")
os.makedirs(RES_DIR,      exist_ok=True)
os.makedirs(SERVICE_DIR,  exist_ok=True)

STYLE_MAP   = {"Smooth":0, "Aggressive":1, "Stop-and-Go":2}
WEATHER_MAP = {"Clear":0, "Clouds":0, "Cloud":0, "Cloudy":0, "Haze":0,
               "Rain":1, "Drizzle":1, "Thunderstorm":1, "Snow":2, "Mist":3, "Fog":3}
ROAD_MAP    = {"Urban":0, "Highway":1, "Rural":2}
WEAR_PER_KM = {"brake_thickness":0.0012, "tire_tread":0.0006, "oil_degradation":0.00015}
THRESH      = {"brake_thickness":2.0, "tire_tread":1.6, "oil_degradation":0.85}


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
            with open(clf_p, "rb") as f: clf  = pickle.load(f)
            with open(reg_p, "rb") as f: reg  = pickle.load(f)
            with open(col_p)       as f: meta = json.load(f)
            return clf, reg, meta
    print("No base model."); sys.exit(1)


def load_finetuned(car_name):
    safe    = car_name.replace(" ", "_").replace("/", "_")
    car_dir = os.path.join(FT_DIR, safe)
    clf_p   = os.path.join(car_dir, "classifier.pkl")
    reg_p   = os.path.join(car_dir, "regressor.pkl")
    col_p   = os.path.join(car_dir, "columns.json")
    if os.path.exists(clf_p) and os.path.exists(reg_p):
        with open(clf_p, "rb") as f: clf = pickle.load(f)
        with open(reg_p, "rb") as f: reg = pickle.load(f)
        meta = None
        if os.path.exists(col_p):
            with open(col_p) as f: meta = json.load(f)
        return clf, reg, meta, True
    return None, None, None, False


def to_numeric(df):
    skip = {"driving_style", "weather_condition", "road_type", "vehicle_name",
            "timestamp", "city", "trouble_codes", "severity", "needs_maintenance",
            "weather_raw", "service_event", "date", "actual_daily_km", "source_file"}
    for col in df.columns:
        if col not in skip:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def compute_stress(row):
    sm  = 1.0 + row.get("aggressive_frac", 0)*0.5 + row.get("stopgo_frac", 0)*0.3
    rm  = 1.0 + (row.get("avg_road_roughness", 2.0) - 2.0)*0.06
    bm  = 1.0 + row.get("avg_hard_braking", 0.5)*0.15
    tm  = 1.0 + (row.get("avg_traffic_density", 50) - 50)*0.003
    return max(0.8, sm * rm * bm * tm)


def aggregate_drive(df, source_file=""):
    df  = df.copy()
    df  = to_numeric(df)
    mil = df["mileage"].clip(lower=0) if "mileage" in df.columns else pd.Series([0]*len(df))
    km  = float(mil.iloc[-1] - mil.iloc[0])
    if km <= 0:
        spd = df["speed"].clip(lower=0) if "speed" in df.columns else pd.Series([0]*len(df))
        dt  = df["elapsed_seconds"].diff().fillna(2.0) if "elapsed_seconds" in df.columns else pd.Series([2.0]*len(df))
        km  = float((spd * dt / 3600).sum())
    km = max(1.0, km)

    sc  = df["driving_style"].fillna("Smooth").value_counts(normalize=True) if "driving_style" in df.columns else pd.Series({"Smooth":1.0})
    af  = float(sc.get("Aggressive",  0))
    sf2 = float(sc.get("Stop-and-Go", 0))
    dom = sc.index[0] if len(sc) > 0 else "Smooth"
    wm  = df["weather_condition"].fillna("Clear").mode()[0] if "weather_condition" in df.columns else "Clear"
    rm  = df["road_type"].fillna("Urban").mode()[0]         if "road_type" in df.columns else "Urban"

    def sm(c, d=0.0):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            return float(v.mean()) if v.notna().any() else d
        return d

    def sl(c, d=0.0):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            return float(v.iloc[-1]) if len(v) > 0 else d
        return d

    def sfirst(c, d=0.0):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            return float(v.iloc[0]) if len(v) > 0 else d
        return d

    return {
        "source_file":           source_file,
        "km_driven":             round(km, 2),
        "engine_temp":           sm("engine_temp", 85),
        "fuel_level":            sm("fuel_level", 0.5),
        "battery_health":        sm("battery_health", 0.8),
        "mileage":               sl("mileage", 0),
        "vehicle_age":           sfirst("vehicle_age", 5),
        "brake_thickness":       sl("brake_thickness", 6.0),
        "tire_tread":            sl("tire_tread", 5.0),
        "oil_degradation":       sl("oil_degradation", 0.3),
        "hard_braking_freq":     sm("hard_braking_freq", 0.5),
        "accel_variance":        sm("accel_variance", 1.0),
        "idle_ratio":            sm("idle_ratio", 0.1),
        "aggressive_frac":       round(af, 3),
        "stopgo_frac":           round(sf2, 3),
        "smooth_frac":           round(1 - af - sf2, 3),
        "ambient_temp":          sm("ambient_temp", 28),
        "road_roughness":        sm("road_roughness", 2.5),
        "monthly_precipitation": sm("monthly_precipitation", 20),
        "traffic_density":       sm("traffic_density", 50),
        "wind_speed":            sm("wind_speed", 4),
        "humidity":              sm("humidity", 50),
        "pressure_hpa":          sm("pressure_hpa", 1010),
        "driving_style_enc":     STYLE_MAP.get(dom, 0),
        "weather_condition_enc": WEATHER_MAP.get(wm, 0),
        "road_type_enc":         ROAD_MAP.get(rm, 0),
        "avg_road_roughness":    sm("road_roughness", 2.5),
        "avg_hard_braking":      sm("hard_braking_freq", 0.5),
        "avg_traffic_density":   sm("traffic_density", 50),
    }


# ── v3: service-log-aware cumulative wear tracking ────────────────────────────

def build_summaries(car_name: str, csv_files: list, prior_wear: dict = None) -> tuple:
    """
    Build per-drive summaries with cumulative wear tracking.

    v3 change: loads the service log for *car_name* and resets wear
    components whenever a service event falls between consecutive drives.
    This is the core fix for the "stuck at 1 day" issue.

    Returns (DataFrame of summaries, final_wear dict)
    """
    sorted_files = sorted(csv_files)

    # ── Load service log ───────────────────────────────────────────────────
    service_events = load_service_log(car_name, SERVICE_DIR, verbose=True)

    # ── Initial wear state ─────────────────────────────────────────────────
    if prior_wear:
        wear = dict(prior_wear)
    else:
        df0 = pd.read_csv(sorted_files[0], low_memory=False)
        df0 = to_numeric(df0)
        def fv(c, d):
            if c in df0.columns:
                v = pd.to_numeric(df0[c], errors="coerce").dropna()
                return float(v.iloc[0]) if len(v) > 0 else d
            return d
        wear = {
            "brake_thickness": fv("brake_thickness", 8.0),
            "tire_tread":      fv("tire_tread",      6.0),
            "oil_degradation": fv("oil_degradation", 0.1),
            "battery_health":  fv("battery_health",  0.8),
        }

    # Warn if already past threshold on first read
    for comp, thresh in THRESH.items():
        val     = wear.get(comp, 0)
        is_worn = (val <= thresh) if comp != "oil_degradation" else (val >= thresh)
        if is_worn:
            print(f"  ⚠️  Initial {comp} ({val:.3f}) past threshold ({thresh}) — "
                  f"service log reset needed")

    rows          = []
    total_km      = 0.0
    total_skm     = 0.0
    prev_date     = None

    for i, fpath in enumerate(sorted_files):
        df = pd.read_csv(fpath, low_memory=False)
        if len(df) == 0:
            continue

        curr_date = get_drive_date(fpath)

        # ── Apply any service events since the last drive ──────────────────
        wear, applied_services = apply_pending_services(
            wear, service_events, prev_date, curr_date, drive_index=i
        )
        if applied_services:
            print(f"  🔧 Service before drive {i+1} "
                  f"({os.path.basename(fpath)[:40]}): "
                  f"{', '.join(applied_services)}")
            print(f"     → brake: {wear['brake_thickness']:.3f}mm  "
                  f"tread: {wear['tire_tread']:.3f}mm  "
                  f"oil: {wear['oil_degradation']:.3f}")

        # ── Aggregate and degrade ──────────────────────────────────────────
        row    = aggregate_drive(df, os.path.basename(fpath))
        stress = compute_stress(row)
        km     = row["km_driven"]

        total_km  += km
        total_skm += km * stress

        wear["brake_thickness"] = max(
            0.1, wear["brake_thickness"] - WEAR_PER_KM["brake_thickness"] * km * stress
        )
        wear["tire_tread"] = max(
            0.1, wear["tire_tread"] - WEAR_PER_KM["tire_tread"] * km * stress
        )
        wear["oil_degradation"] = min(
            1.0, wear["oil_degradation"] + WEAR_PER_KM["oil_degradation"] * km * stress
        )

        row["brake_thickness"] = round(wear["brake_thickness"], 4)
        row["tire_tread"]      = round(wear["tire_tread"],      4)
        row["oil_degradation"] = round(wear["oil_degradation"], 4)

        daily_km = max(5.0, total_km / max(1, i + 1))
        avg_st   = max(1.0, total_skm / max(total_km, 1.0))

        db  = max(0, (wear["brake_thickness"] - THRESH["brake_thickness"])
                  / (WEAR_PER_KM["brake_thickness"] * avg_st * daily_km)
                  ) if wear["brake_thickness"] > THRESH["brake_thickness"] else 0
        dt2 = max(0, (wear["tire_tread"] - THRESH["tire_tread"])
                  / (WEAR_PER_KM["tire_tread"] * avg_st * daily_km)
                  ) if wear["tire_tread"] > THRESH["tire_tread"] else 0
        do  = max(0, (THRESH["oil_degradation"] - wear["oil_degradation"])
                  / (WEAR_PER_KM["oil_degradation"] * avg_st * daily_km)
                  ) if wear["oil_degradation"] < THRESH["oil_degradation"] else 0

        dts = round(max(1, min(365, min(db, dt2, do))), 1)

        bt = wear["brake_thickness"]
        row["stress"]               = round(stress, 3)
        row["avg_stress"]           = round(avg_st, 3)
        row["daily_km_avg"]         = round(daily_km, 2)
        row["total_km"]             = round(total_km, 2)
        row["days_to_service_proxy"]= dts
        row["failure_proxy"]        = int(dts <= 30)
        row["engine_thermal_load"]  = row["engine_temp"]*0.5 + row["avg_traffic_density"]*0.3 + row["ambient_temp"]*0.2
        row["brake_stress_idx"]     = row["hard_braking_freq"]*0.4 + row["avg_road_roughness"]*0.4 + (1/max(0.1, bt))*0.2
        row["traffic_road_impact"]  = row["avg_traffic_density"] * row["avg_road_roughness"]
        row["engine_battery_ratio"] = row["engine_temp"] / max(0.1, row["battery_health"])
        row["brake_wear_pct"]       = float(np.clip(1 - (bt - THRESH["brake_thickness"])/(8.0 - THRESH["brake_thickness"]), 0, 1))
        row["tire_wear_pct"]        = float(np.clip(1 - (wear["tire_tread"] - THRESH["tire_tread"])/(8.0 - THRESH["tire_tread"]), 0, 1))
        row["oil_wear_pct"]         = float(np.clip(wear["oil_degradation"] / THRESH["oil_degradation"], 0, 1))
        row["composite_wear_idx"]   = row["brake_wear_pct"]*0.4 + row["tire_wear_pct"]*0.35 + row["oil_wear_pct"]*0.25
        # ── v3 service fields ──────────────────────────────────────────────
        row["service_applied"]      = int(bool(applied_services))
        row["services_done"]        = "|".join(applied_services)
        row["drive_date"]           = str(curr_date.date()) if curr_date else ""

        rows.append(row)
        prev_date = curr_date

    return pd.DataFrame(rows), wear


def make_X(df, fcols):
    X = pd.DataFrame(index=df.index)
    for c in fcols:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0.0
    return X.fillna(0)


def sev(p):
    return "CRITICAL" if p >= 0.75 else "WARNING" if p >= 0.5 else "WATCH" if p >= 0.3 else "OK"


def get_car_name(df, path):
    if "vehicle_name" in df.columns:
        n = str(df["vehicle_name"].iloc[0]).strip()
        if n and n != "nan":
            return n
    parts = os.path.basename(path).replace(".csv", "").split("_")
    return " ".join(parts[2:]) if len(parts) > 2 else "Unknown"


def predict_car(car_name, csv_files, base_clf, base_reg, base_meta,
                eval_mode=False, prior_wear=None):
    ft_clf, ft_reg, ft_meta, is_ft = load_finetuned(car_name)

    if is_ft:
        clf   = ft_clf
        reg   = ft_reg
        fcols = ft_meta["feature_cols"] if ft_meta else base_meta["feature_cols"]
        tag   = "fine-tuned ✨"
    else:
        clf   = base_clf
        reg   = base_reg
        fcols = base_meta["feature_cols"]
        tag   = "synthetic base"

    # build_summaries now handles service log loading internally
    summaries, final_wear = build_summaries(car_name, csv_files, prior_wear=prior_wear)

    if len(summaries) == 0:
        print(f"\n  ⚠️  {car_name} — no valid drives")
        return None

    X    = make_X(summaries, fcols)
    probs= clf.predict_proba(X)[:, 1]
    days = np.clip(reg.predict(X), 1, 365)
    summaries["maintenance_probability"] = probs.round(4)
    summaries["days_to_service_pred"]    = days.round(1)
    summaries["severity"]                = [sev(p) for p in probs]
    summaries["needs_maintenance_pred"]  = (probs >= 0.5).astype(int)

    print(f"\n  🚗 {car_name}  [{tag}]")
    print(f"  {'─'*82}")
    print(f"  {'Drive File':<45} {'km':>5}  {'Risk':>6}  {'Days':>5}  {'Severity':<10}  Service")
    print(f"  {'─'*82}")
    for _, row in summaries.iterrows():
        svc_tag = f"🔧{row.get('services_done','')}" if row.get("service_applied", 0) else ""
        print(f"  {str(row['source_file'])[:44]:<44} "
              f"{row['km_driven']:>5.1f}  "
              f"{row['maintenance_probability']:>5.1%}  "
              f"{row['days_to_service_pred']:>5.0f}  "
              f"{row['severity']:<10}  {svc_tag}")

    print(f"\n  Avg risk: {probs.mean():.1%}  |  Max risk: {probs.max():.1%}  |  Avg days: {days.mean():.0f}")
    print(f"  Final wear → brake: {final_wear['brake_thickness']:.3f}mm  "
          f"tread: {final_wear['tire_tread']:.3f}mm  "
          f"oil: {final_wear['oil_degradation']:.3f}")

    # Warn if car is still in a critical state with no service log
    worst_comp = None
    if final_wear["brake_thickness"] <= THRESH["brake_thickness"]:
        worst_comp = f"brake_thickness ({final_wear['brake_thickness']:.3f}mm)"
    elif final_wear["tire_tread"] <= THRESH["tire_tread"]:
        worst_comp = f"tire_tread ({final_wear['tire_tread']:.3f}mm)"
    elif final_wear["oil_degradation"] >= THRESH["oil_degradation"]:
        worst_comp = f"oil_degradation ({final_wear['oil_degradation']:.3f})"
    if worst_comp:
        print(f"\n  ⚠️  {worst_comp} is at/past threshold. If the car was serviced, "
              f"add an entry to data/service_logs/ to reset wear state.")

    if eval_mode:
        y_reg = summaries["days_to_service_proxy"]
        y_cls = summaries["failure_proxy"]
        mae   = mean_absolute_error(y_reg, days)
        r2    = r2_score(y_reg, days)
        f1    = f1_score(y_cls, (probs >= 0.5).astype(int),
                         average="macro", zero_division=0) if y_cls.nunique() > 1 else float("nan")
        auc   = roc_auc_score(y_cls, probs) if y_cls.nunique() > 1 else float("nan")
        print(f"\n  ── Eval vs proxy labels ──────────────────────────")
        print(f"  MAE: {mae:.1f} days  R²: {r2:.4f}  F1: {f1:.3f}  AUC: {auc:.3f}")

    out = os.path.join(RES_DIR, f"{car_name.replace(' ', '_')}_predicted.csv")
    summaries.to_csv(out, index=False)
    print(f"  💾 {out}")
    return summaries


def group_by_car(csv_files):
    groups = defaultdict(list)
    for f in csv_files:
        try:
            df_tmp = pd.read_csv(f, nrows=1)
            if len(df_tmp) == 0:
                continue
            groups[get_car_name(df_tmp, f)].append(f)
        except Exception:
            pass
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--car",  default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  predict_service.py v3 — Service-Log-Aware Predictions")
    print("=" * 60)

    ft_cars = (
        [d for d in os.listdir(FT_DIR) if os.path.isdir(os.path.join(FT_DIR, d))]
        if os.path.exists(FT_DIR) else []
    )
    if ft_cars:
        print(f"\n  Fine-tuned: {', '.join(c.replace('_', ' ') for c in ft_cars)}")

    print("\n  Loading base model...")
    base_clf, base_reg, base_meta = load_base()
    print(f"  ✓ Base model ({len(base_meta['feature_cols'])} features)")

    if args.file:
        df_tmp = pd.read_csv(args.file, nrows=1)
        car    = get_car_name(df_tmp, args.file)
        predict_car(car, [args.file], base_clf, base_reg, base_meta,
                    eval_mode=args.eval)

    elif args.car:
        all_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
        groups    = group_by_car(all_files)
        matched   = {k: v for k, v in groups.items() if args.car.lower() in k.lower()}
        if not matched:
            print(f"  ❌ No drives for '{args.car}'")
            sys.exit(1)
        for car, files in matched.items():
            predict_car(car, files, base_clf, base_reg, base_meta,
                        eval_mode=args.eval)

    else:
        all_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
        if not all_files:
            print(f"  No CSV files in {DATA_DIR}")
            sys.exit(1)
        groups = group_by_car(all_files)
        print(f"\n  {len(groups)} car(s) found:")
        for car, files in sorted(groups.items()):
            safe  = car.replace(" ", "_").replace("/", "_")
            has_log = (os.path.exists(os.path.join(SERVICE_DIR, f"{safe}_service.csv")) or
                       os.path.exists(os.path.join(SERVICE_DIR, f"{safe}_service.json")))
            tag   = " 📋" if has_log else ""
            print(f"    {car:<35} {len(files)} drive(s){tag}")
        for car, files in sorted(groups.items()):
            predict_car(car, files, base_clf, base_reg, base_meta,
                        eval_mode=args.eval)

    print("\n" + "="*60 + "\n  Done.\n" + "="*60)
