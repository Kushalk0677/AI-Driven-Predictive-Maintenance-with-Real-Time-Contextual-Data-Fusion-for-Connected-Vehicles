"""
testing/test_2_live_read.py
============================
STEP 2 — Read live sensor data from your car's OBD port.

Prints raw sensor readings every second for 30 seconds.
Use this to confirm your dongle is sending real data.

Usage:
    python testing/test_2_live_read.py

Change PORT below to your COM port from Step 1.
Works with any car + any ELM327 Bluetooth dongle.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

# ── CONFIG — change this to your COM port from Step 1 ────────────────────────
PORT = None   # None = auto-detect, or set e.g. "COM6" or "/dev/rfcomm0"
DURATION = 30  # seconds to read
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Step 2 — Live OBD Sensor Reading")
print("=" * 60)

try:
    import obd
except ImportError:
    print("  Run: pip install obd")
    sys.exit(1)

print(f"\n  Connecting on port: {PORT or 'auto'}...")
conn = obd.OBD(PORT) if PORT else obd.OBD()

if not conn.is_connected():
    print("  ❌ Could not connect. Run test_1_find_port.py first.")
    sys.exit(1)

print(f"  ✅ Connected: {conn.port_name()}")
print(f"\n  Reading sensors for {DURATION}s. Start your engine!\n")
print(f"  {'Time':<6} {'RPM':<8} {'Coolant':<10} {'Speed':<8} "
      f"{'Load':<8} {'Throttle':<10} {'Fuel':<8} {'Battery':<10} {'DTCs'}")
print("  " + "-" * 75)

start = time.time()

while (time.time() - start) < DURATION:
    def read(pid):
        try:
            r = conn.query(obd.commands[pid])
            return round(r.value.magnitude, 1) if not r.is_null() else None
        except:
            return None

    rpm      = read("RPM")
    coolant  = read("COOLANT_TEMP")
    speed    = read("SPEED")
    load     = read("ENGINE_LOAD")
    throttle = read("THROTTLE_POS")
    fuel     = read("FUEL_LEVEL")

    try:
        batt_r = conn.query(obd.commands["CONTROL_MODULE_VOLTAGE"])
        battery = round(batt_r.value.magnitude, 2) if not batt_r.is_null() else None
    except:
        battery = None

    try:
        dtc_r = conn.query(obd.commands["GET_DTC"])
        dtcs = len(dtc_r.value) if not dtc_r.is_null() else 0
    except:
        dtcs = 0

    elapsed = int(time.time() - start)

    def fmt(v, unit=""): return f"{v}{unit}" if v is not None else "N/A"

    print(
        f"  {elapsed:<6} "
        f"{fmt(rpm, ' rpm'):<8} "
        f"{fmt(coolant, '°C'):<10} "
        f"{fmt(speed, ' km/h'):<8} "
        f"{fmt(load, '%'):<8} "
        f"{fmt(throttle, '%'):<10} "
        f"{fmt(fuel, '%'):<8} "
        f"{fmt(battery, 'V'):<10} "
        f"{dtcs}"
    )

    time.sleep(1.0)

conn.close()

print("\n" + "=" * 60)
print("  ✅ Step 2 complete!")
print("  If you see real values above, your dongle is working.")
print("  If most values are N/A, try a different protocol.")
print("  Run test_3_predict.py next.")
print("=" * 60)
