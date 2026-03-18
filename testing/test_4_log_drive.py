"""
testing/test_4_log_drive.py
=============================
STEP 4 — Log a full drive to CSV with weather + location.

Records all OBD sensors, computed driver behaviour, weather,
and maintenance predictions to a timestamped CSV file.

Usage:
    python testing/test_4_log_drive.py

Requirements:
    pip install requests

Optional — get a free OpenWeatherMap API key at:
    https://openweathermap.org/api
    (1000 free calls/day — enough for daily drives)

Fill in YOUR VEHICLE DETAILS below. Works with any car.
Ctrl+C to stop logging.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

# ── YOUR VEHICLE DETAILS — fill these in ─────────────────────────────────────
PORT                = None       # e.g. "COM6". None = auto-detect
VEHICLE_NAME        = "My Car"   # e.g. "Safari Storme", "Swift", "Creta"
MILEAGE             = 95000      # km on odometer
VEHICLE_AGE         = 10         # years
BRAKE_THICKNESS     = 5.5        # mm
TIRE_TREAD          = 4.0        # mm
OIL_DEGRADATION     = 0.3        # 0=new oil, 1=very degraded
ROAD_TYPE           = "Urban"    # Urban / Highway / Rural
TRAFFIC_DENSITY     = 30.0       # vehicles per km (rough estimate)
OPENWEATHER_API_KEY = None       # paste your key here, or leave None
INTERVAL            = 5.0        # seconds between readings
# ─────────────────────────────────────────────────────────────────────────────

from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset, VehicleContext
from vehiclepm.features import build_feature_matrix
from vehiclepm.obd.reader import OBDReader
from vehiclepm.obd.adapter import OBDFeatureAdapter
from vehiclepm.obd.logger import DriveLogger
from vehiclepm.obd.live import _get_severity

def align_columns(X_live, training_columns):
    for col in training_columns:
        if col not in X_live.columns:
            X_live[col] = 0
    return X_live[training_columns]

print("=" * 60)
print(f"  Step 4 — Drive Logger: {VEHICLE_NAME}")
print("=" * 60)

# Train model
print("\n  Training model...")
df = generate_synthetic_dataset(n_samples=2000, random_state=42)
X_train = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
y = df["maintenance_needed"]
TRAINING_COLUMNS = list(X_train.columns)
clf = VehiclePMClassifier(model_type="lightgbm")
clf.fit(X_train, y)
print("  Model ready. ✓")

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

# Output path includes vehicle name
os.makedirs("logs", exist_ok=True)
from datetime import datetime
ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
safe_name = VEHICLE_NAME.replace(" ", "_")
output_path = f"logs/{ts}_{safe_name}.csv"

drive_logger = DriveLogger(
    output_path=output_path,
    weather_api_key=OPENWEATHER_API_KEY,
    weather_interval=60.0,
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
        drive_logger.start()
        print(f"\n  Logging every {INTERVAL}s to: {output_path}")
        print(f"  Ctrl+C to stop.\n")
        print(f"  {'Time':<8} {'Severity':<12} {'Risk':<8} {'Engine':<12} "
              f"{'Speed':<10} {'Weather':<12} {'DTCs'}")
        print("  " + "-" * 75)

        for obd_reading in reader.stream(interval=INTERVAL):
            row = drive_logger.log_row(
                obd_reading, adapter, clf, TRAINING_COLUMNS, ctx
            )
            reading_count += 1

            colour = colours.get(row.severity, "")
            icon   = icons.get(row.severity, "")
            mins   = int(row.elapsed_seconds // 60)
            secs   = int(row.elapsed_seconds % 60)

            print(
                f"  {mins:02d}:{secs:02d}{'':>2} "
                f"{colour}{icon} {row.severity:<10}{reset} "
                f"{f'{row.maintenance_probability:.1%}':<8} "
                f"{row.engine_temp:.1f}°C{'':>4} "
                f"{row.speed:.0f} km/h{'':>3} "
                f"{row.weather_condition:<12} "
                f"{row.dtc_count}"
            )

except KeyboardInterrupt:
    print(f"\n\n  Drive stopped.")
except Exception as e:
    print(f"\n  ❌ Error: {e}")
finally:
    drive_logger.stop()

print(f"\n  📁 Log saved to: {output_path}")
print(f"  Run test_5_replay_drive.py to replay this drive.")
print("=" * 60)
