"""
drive.py — vehiclepm Live Drive Tool
=====================================
Log a real OBD-II drive to CSV, or replay a logged drive offline.

USAGE
-----

Log mode (plug in OBD dongle, start car, run):
    python drive.py --mode log --port COM6 --api-key YOUR_OPENWEATHER_KEY

Replay mode (no dongle needed):
    python drive.py --mode replay --input logs/2026-03-18_drive.csv

    # Replay at 5x speed
    python drive.py --mode replay --input logs/2026-03-18_drive.csv --speed 5

    # Replay instantly
    python drive.py --mode replay --input logs/2026-03-18_drive.csv --speed 0

GET A FREE OPENWEATHERMAP API KEY
----------------------------------
1. Go to https://openweathermap.org/api
2. Sign up (free)
3. Copy your API key
4. Pass it with --api-key YOUR_KEY

Without an API key, weather defaults to Clear / 25°C.

FIND YOUR COM PORT (Windows)
-----------------------------
1. Pair OBD dongle via Bluetooth
2. Device Manager → Ports (COM & LPT)
3. Note the COM number e.g. COM6
"""

import argparse
import os
import sys
import time

# ── Add vehiclepm to path ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vehiclepm"))

from vehiclepm import VehiclePMClassifier, generate_synthetic_dataset, VehicleContext
from vehiclepm.features import build_feature_matrix
from vehiclepm.obd.logger import DriveLogger, replay_drive
from vehiclepm.obd.live import _get_severity, MaintenanceAlert


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="vehiclepm — Live OBD Drive Logger & Replayer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["log", "replay"], required=True,
                        help="'log' to record a drive, 'replay' to replay a CSV")

    # Log mode args
    parser.add_argument("--port", default=None,
                        help="OBD COM port e.g. COM6 (Windows) or /dev/rfcomm0 (Linux)")
    parser.add_argument("--api-key", default=None,
                        help="OpenWeatherMap API key (free at openweathermap.org)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between readings (default: 5)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: logs/TIMESTAMP_drive.csv)")

    # Vehicle context (log mode)
    parser.add_argument("--mileage",         type=float, default=None)
    parser.add_argument("--vehicle-age",     type=float, default=None)
    parser.add_argument("--brake-thickness", type=float, default=None)
    parser.add_argument("--tire-tread",      type=float, default=None)
    parser.add_argument("--oil-degradation", type=float, default=None)
    parser.add_argument("--road-type",       default="Urban",
                        choices=["Urban", "Highway", "Rural"])
    parser.add_argument("--traffic-density", type=float, default=30.0)

    # Replay mode args
    parser.add_argument("--input",  default=None, help="CSV file to replay")
    parser.add_argument("--speed",  type=float,   default=1.0,
                        help="Replay speed multiplier (default: 1.0, use 0 for instant)")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle context setup (interactive if args not provided)
# ─────────────────────────────────────────────────────────────────────────────

def get_vehicle_context(args) -> VehicleContext:
    print("\n" + "=" * 50)
    print("  Vehicle Setup (press Enter to use default)")
    print("=" * 50)

    def ask(prompt, default, cast=float):
        val = input(f"  {prompt} [{default}]: ").strip()
        return cast(val) if val else default

    mileage         = args.mileage         or ask("Odometer reading (km)", 95000)
    vehicle_age     = args.vehicle_age     or ask("Vehicle age (years)", 11)
    brake_thickness = args.brake_thickness or ask("Brake thickness (mm)", 5.5)
    tire_tread      = args.tire_tread      or ask("Tire tread depth (mm)", 4.0)
    oil_degradation = args.oil_degradation or ask("Oil degradation 0.0-1.0", 0.3)

    road_type = args.road_type
    if not args.road_type:
        rt = input("  Road type (Urban/Highway/Rural) [Urban]: ").strip()
        road_type = rt if rt in ["Urban", "Highway", "Rural"] else "Urban"

    return VehicleContext(
        mileage=mileage,
        vehicle_age=vehicle_age,
        brake_thickness=brake_thickness,
        tire_tread=tire_tread,
        oil_degradation=oil_degradation,
        road_type=road_type,
        traffic_density=args.traffic_density,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Train model
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    print("\n  Training model on synthetic data...")
    df = generate_synthetic_dataset(n_samples=2000, random_state=42)
    X  = build_feature_matrix(df.drop(columns=["risk_score", "maintenance_needed"]))
    y  = df["maintenance_needed"]
    training_columns = list(X.columns)

    clf = VehiclePMClassifier(model_type="lightgbm")
    clf.fit(X, y)
    print(f"  Model ready ({len(training_columns)} features). ✓")
    return clf, training_columns


# ─────────────────────────────────────────────────────────────────────────────
# Log mode
# ─────────────────────────────────────────────────────────────────────────────

def run_log(args):
    from vehiclepm.obd.reader import OBDReader
    from vehiclepm.obd.adapter import OBDFeatureAdapter

    print("\n" + "=" * 55)
    print("  vehiclepm — Drive Logger")
    print("=" * 55)

    clf, training_columns = train_model()
    ctx = get_vehicle_context(args)

    drive_logger = DriveLogger(
        output_path=args.output,
        weather_api_key=args.api_key,
        weather_interval=60.0,
    )

    colours = {
        "OK": "\033[92m", "WATCH": "\033[93m",
        "WARNING": "\033[33m", "CRITICAL": "\033[91m",
    }
    icons = {"OK": "✅", "WATCH": "👀", "WARNING": "⚠️ ", "CRITICAL": "🚨"}
    reset = "\033[0m"

    print(f"\n  Connecting to OBD dongle on {args.port or 'auto'}...")

    adapter = OBDFeatureAdapter(context=ctx, window_size=60)

    try:
        with OBDReader(port=args.port) as reader:
            drive_logger.start()
            print(f"\n  Logging every {args.interval}s. Press Ctrl+C to stop.\n")
            print(f"  {'Time':<10} {'Sev':<10} {'Risk':<8} {'Engine':<10} "
                  f"{'Speed':<10} {'Battery':<10} {'DTCs'}")
            print("  " + "-" * 63)

            for obd_reading in reader.stream(interval=args.interval):
                row = drive_logger.log_row(
                    obd_reading, adapter, clf, training_columns, ctx
                )
                colour = colours.get(row.severity, "")
                icon   = icons.get(row.severity, "")
                mins   = int(row.elapsed_seconds // 60)
                secs   = int(row.elapsed_seconds % 60)

                print(
                    f"  {mins:02d}:{secs:02d}{'':>4} "
                    f"{colour}{icon} {row.severity:<8}{reset} "
                    f"{f'{row.maintenance_probability:.1%}':<8} "
                    f"{row.engine_temp:.1f}°C{'':>4} "
                    f"{row.speed:.0f} km/h{'':>3} "
                    f"{row.battery_health:.0%}{'':>6} "
                    f"{row.dtc_count}"
                )

    except KeyboardInterrupt:
        print("\n\n  Drive stopped by user.")
    finally:
        drive_logger.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Replay mode
# ─────────────────────────────────────────────────────────────────────────────

def run_replay(args):
    if not args.input:
        # List available logs
        if os.path.exists("logs"):
            logs = [f for f in os.listdir("logs") if f.endswith(".csv")]
            if logs:
                print("\nAvailable drive logs:")
                for i, f in enumerate(sorted(logs)):
                    print(f"  [{i+1}] {f}")
                choice = input("\nEnter number or full path: ").strip()
                try:
                    idx = int(choice) - 1
                    args.input = os.path.join("logs", sorted(logs)[idx])
                except (ValueError, IndexError):
                    args.input = choice
            else:
                print("No drive logs found in logs/ folder.")
                print("Run: python drive.py --mode log --port COM6")
                return
        else:
            print("No logs/ folder found. Run log mode first.")
            return

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}")
        return

    replay_drive(args.input, speed=args.speed)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "log":
        run_log(args)
    elif args.mode == "replay":
        run_replay(args)
