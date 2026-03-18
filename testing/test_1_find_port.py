"""
testing/test_1_find_port.py
============================
STEP 1 — Find your OBD dongle's COM port and confirm it connects.

Run this BEFORE anything else. Plug in your OBD dongle, turn
ignition to ACC, pair via Bluetooth, then run this script.

Usage:
    python testing/test_1_find_port.py

Works with any car + any ELM327 Bluetooth dongle.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

print("=" * 60)
print("  Step 1 — Find OBD Port & Test Connection")
print("=" * 60)

# ── Check if obd library is installed ────────────────────────────────────────
try:
    import obd
    print("\n  ✅ python-obd installed")
except ImportError:
    print("\n  ❌ python-obd not installed.")
    print("     Run: pip install obd")
    sys.exit(1)

# ── Scan for available ports ──────────────────────────────────────────────────
print("\n  Scanning for OBD ports...")
ports = obd.scan_serial()

if not ports:
    print("\n  ❌ No OBD ports found.")
    print("\n  Troubleshooting:")
    print("  1. Make sure dongle is plugged into OBD port (under dashboard)")
    print("  2. Turn ignition to ACC (one click before starting)")
    print("  3. Pair dongle via Bluetooth: Settings → Bluetooth → Add device")
    print("  4. Check Device Manager → Ports (COM & LPT) for COM number")
    print("  5. Make sure no other app (Torque, etc.) is using the port")
    sys.exit(1)

print(f"\n  Found {len(ports)} port(s):")
for p in ports:
    print(f"    → {p}")

# ── Try connecting ────────────────────────────────────────────────────────────
print("\n  Attempting connection...")

connection = obd.OBD()  # auto-scan

if not connection.is_connected():
    print("\n  ❌ Could not connect.")
    print("\n  Try specifying port manually:")
    print("     connection = obd.OBD('COM6')  # replace with your port")
    sys.exit(1)

print(f"\n  ✅ Connected on port: {connection.port_name()}")
print(f"  Protocol: {connection.protocol_name()}")

# ── Check supported PIDs ──────────────────────────────────────────────────────
print("\n  Checking supported PIDs...")

pids_to_check = {
    "ENGINE_RPM":               "RPM",
    "COOLANT_TEMP":             "Engine Coolant Temperature",
    "SPEED":                    "Vehicle Speed",
    "THROTTLE_POS":             "Throttle Position",
    "ENGINE_LOAD":              "Engine Load",
    "FUEL_LEVEL":               "Fuel Level",
    "INTAKE_TEMP":              "Intake Air Temperature",
    "MAF":                      "Mass Air Flow",
    "CONTROL_MODULE_VOLTAGE":   "Battery Voltage",
}

supported = []
unsupported = []

for pid, name in pids_to_check.items():
    cmd = obd.commands.get(pid)
    if cmd and connection.supports(cmd):
        supported.append(name)
        print(f"    ✅ {name}")
    else:
        unsupported.append(name)
        print(f"    ⚠️  {name} (not supported by this vehicle — will use default)")

connection.close()

print(f"\n  Summary: {len(supported)}/{len(pids_to_check)} PIDs supported")
print(f"\n  Your COM port: {connection.port_name()}")
print(f"\n  Use this in all other scripts:")
print(f"    PORT = \"{connection.port_name()}\"")

print("\n" + "=" * 60)
print("  ✅ Step 1 complete! Note your COM port above.")
print("     Run test_2_live_read.py next.")
print("=" * 60)
