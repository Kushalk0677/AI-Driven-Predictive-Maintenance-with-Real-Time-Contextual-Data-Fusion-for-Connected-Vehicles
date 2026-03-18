"""
testing/test_3_predict.py
==========================
STEP 3 — Live maintenance predictions from your real car.

Reads OBD sensors every 5 seconds and shows maintenance
risk probability with colour-coded severity alerts.

Usage:
    python testing/test_3_predict.py

Fill in YOUR VEHICLE DETAILS below before running.
Works with any car. Run for as long as you like (Ctrl+C to stop).
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

# ── YOUR VEHICLE DETAILS — fill these in ─────────────────────────────────────
PORT             = None      # e.g. "COM6" or "/dev/rfcomm0". None = auto-detect
MILEAGE          = 95000     # km on your odometer
VEHICLE_AGE      = 10        # years since manufacture
BRAKE_THICKNESS  = 5.5       # mm — check last service report
TIRE_TREAD       = 4.0       # mm
OIL_DEGRADATION  = 0.3       # 0.0 = fresh oil, 1.0 = very degraded
ROAD_TYPE        = "Urban"   # "Urban", "Highway", or "Rural"
TRAFFIC_DENSITY  = 30.0      # rough vehicles per km
INTERVAL         = 5.0       # seconds between predictions
# ─────────────────────────────────────────────────────────────────────────────

from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset, VehicleContext
from vehiclepm.features import build_feature_matrix
from vehiclepm.obd.reader import OBDReader
from vehiclepm.obd.adapter import OBDFeatureAdapter
from vehiclepm.obd.live import _get_severity

def align_columns(X_live, training_columns):
    for col in training_columns:
        if col not in X_live.columns:
            X_live[col] = 0
    return X_live[training_columns]

print("=" * 60)
print("  Step 3 — Live Maintenance Prediction")
print("=" * 60)

# Train model
print("\n  Training model...")
df = generate_synthetic_dataset(n_samples=2000, random_state=42)
X_train = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
y = df["maintenance_needed"]
TRAINING_COLUMNS = list(X_train.columns)
clf = VehiclePMClassifier(model_type="lightgbm")
clf.fit(X_train, y)
print(f"  Model ready. ✓")

# Vehicle context
ctx = VehicleContext(
    mileage=MILEAGE,
    vehicle_age=VEHICLE_AGE,
    brake_thickness=BRAKE_THICKNESS,
    tire_tread=TIRE_TREAD,
    oil_degradation=OIL_DEGRADATION,
    road_type=ROAD_TYPE,
    traffic_density=TRAFFIC_DENSITY,
)

colours = {
    "OK":       "\033[92m",
    "WATCH":    "\033[93m",
    "WARNING":  "\033[33m",
    "CRITICAL": "\033[91m",
}
icons = {"OK": "✅", "WATCH": "👀", "WARNING": "⚠️ ", "CRITICAL": "🚨"}
reset = "\033[0m"

print(f"\n  Connecting to OBD dongle on {PORT or 'auto'}...")

adapter = OBDFeatureAdapter(context=ctx, window_size=60)
reading_count = 0

try:
    with OBDReader(port=PORT) as reader:
        print(f"  ✅ Connected! Predicting every {INTERVAL}s. Ctrl+C to stop.\n")
        print(f"  {'#':<5} {'Severity':<12} {'Risk':<8} {'Engine':<12} "
              f"{'Speed':<10} {'Battery':<10} {'DTCs':<6} {'Style'}")
        print("  " + "-" * 75)

        for obd_reading in reader.stream(interval=INTERVAL):
            reading_count += 1

            raw_df = adapter.to_features(obd_reading)
            X_live = build_feature_matrix(raw_df)
            X_live = align_columns(X_live, TRAINING_COLUMNS)

            prob     = float(clf.predict_proba(X_live)[0, 1])
            severity = _get_severity(prob)
            driver   = adapter._compute_driver_features()
            colour   = colours.get(severity, "")
            icon     = icons.get(severity, "")

            eng  = f"{obd_reading.engine_temp:.1f}°C" if obd_reading.engine_temp else "N/A"
            spd  = f"{obd_reading.speed:.0f} km/h"   if obd_reading.speed       else "N/A"
            bat  = f"{obd_reading.battery_health:.0%}" if obd_reading.battery_health else "N/A"
            dtcs = obd_reading.dtc_count

            print(
                f"  {reading_count:<5} "
                f"{colour}{icon} {severity:<10}{reset} "
                f"{f'{prob:.1%}':<8} "
                f"{eng:<12} "
                f"{spd:<10} "
                f"{bat:<10} "
                f"{dtcs:<6} "
                f"{driver['driving_style']}"
            )

except KeyboardInterrupt:
    print(f"\n\n  Stopped after {reading_count} readings.")
except Exception as e:
    print(f"\n  ❌ Error: {e}")
    print("  Make sure ignition is on and dongle is paired.")

print("\n" + "=" * 60)
print(f"  Done. {reading_count} predictions made.")
print("  Run test_4_log_drive.py to record a full drive to CSV.")
print("=" * 60)
