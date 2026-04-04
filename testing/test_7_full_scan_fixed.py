"""
testing/test_7_full_scan.py
============================
Scans ALL available OBD-II PIDs on your car and logs
everything to CSV — fuel consumption, torque, power,
oxygen sensors, timing, fuel trims, weather and more.

Bluetooth-safe version:
    - Warm-up delay after connect (ELM327 needs time to initialise)
    - Per-query delay prevents ELM327 buffer overflow
    - Auto-reconnect on BT drop (up to MAX_RECONNECT_ATTEMPTS)
    - Priority tiers: fast PIDs every cycle, slow PIDs every Nth cycle
    - Connection health check before every polling cycle

Usage:
    python testing/test_7_full_scan.py
"""

import sys
import os
import time
import csv
import logging
import requests
from datetime import datetime
from collections import OrderedDict

logging.getLogger("obd").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

# ── CONFIG ────────────────────────────────────────────────────────────────────
PORT                    = "COM4"
BAUDRATE                = 38400
INTERVAL                = 2.0       # seconds between each full poll cycle
QUERY_DELAY             = 0.07      # 70ms between each PID query — prevents BT overflow
CONNECT_WARMUP          = 3.0       # FIX 1: seconds to wait after connect before querying
                                    #        ELM327 needs this time to fully initialise
OPENWEATHER_API_KEY     = "bd5e378503939ddaee76f12ad7a97608"
WEATHER_INTERVAL        = 60.0

MAX_RECONNECT_ATTEMPTS  = 5
RECONNECT_DELAY         = 5.0

SLOW_PID_EVERY          = 5         # poll slow PIDs every 5th cycle (~10 seconds)

# ── VEHICLE SPECS ─────────────────────────────────────────────────────────────
VEHICLE_SPECS = {
    "name":             "Safari Storme EX",
    "engine":           "2.2 DiCOR",
    "fuel_type":        "diesel",
    "max_torque_nm":    320,
    "max_torque_rpm":   2000,
    "max_power_kw":     103,
    "max_power_rpm":    4000,
    "displacement_cc":  2179,
    "cylinders":        4,
    "fuel_tank_litres": 37,
}
# For other cars:
# Swift Dzire 1.2 petrol: max_torque_nm=113, max_pcower_kw=61, displacement_cc=1197
# Creta 1.5 petrol:       max_torque_nm=144, max_power_kw=85, displacement_cc=1497
# Innova 2.4 diesel:      max_torque_nm=343, max_power_kw=110, displacement_cc=2393
# Tata Nexon 1.2 petrol:  max_torque_nm=170, max_power_kw=88, displacement_cc=1199
# Wagon R 1.0 petrol:     max_torque_nm=89,  max_power_kw=50, displacement_cc=998
# ─────────────────────────────────────────────────────────────────────────────

try:
    import obd
except ImportError:
    print("Run: pip install obd")
    sys.exit(1)

# ── PIDs split into FAST and SLOW tiers ──────────────────────────────────────
# FAST: change every second — poll every cycle
FAST_PIDS = {
    "RPM":                      "Engine RPM (rpm)",
    "COOLANT_TEMP":             "Engine Coolant Temp (°C)",
    "ENGINE_LOAD":              "Engine Load (%)",
    "INTAKE_TEMP":              "Intake Air Temp (°C)",
    "MAF":                      "Mass Air Flow (g/s)",
    "THROTTLE_POS":             "Throttle Position (%)",
    "ACCELERATOR_POS_D":        "Accelerator Pedal D (%)",
    "ACCELERATOR_POS_E":        "Accelerator Pedal E (%)",
    "SPEED":                    "Vehicle Speed (km/h)",
    "FUEL_RATE":                "Fuel Consumption Rate (L/h)",
    "SHORT_FUEL_TRIM_1":        "Short Term Fuel Trim Bank 1 (%)",
    "ABSOLUTE_LOAD":            "Absolute Engine Load (%)",
    "COMMANDED_EQUIV_RATIO":    "Commanded Equivalence Ratio",
    "CONTROL_MODULE_VOLTAGE":   "Battery Voltage (V)",
    "OIL_TEMP":                 "Engine Oil Temp (°C)",
    "INTAKE_MANIFOLD_PRESSURE": "Intake Manifold Pressure (kPa)",
    "TIMING_ADVANCE":           "Timing Advance (°)",
    "FUEL_INJECT_TIMING":       "Fuel Injection Timing (°)",
}

# SLOW: change rarely — poll every SLOW_PID_EVERY cycles
SLOW_PIDS = {
    "THROTTLE_POS_B":             "Throttle Position B (%)",
    "THROTTLE_POS_C":             "Throttle Position C (%)",
    "ACCELERATOR_POS_F":          "Accelerator Pedal F (%)",
    "COMMANDED_THROTTLE_ACTUATOR":"Commanded Throttle (%)",
    "ENGINE_RUNTIME":             "Engine Runtime (s)",
    "FUEL_LEVEL":                 "Fuel Level (%)",
    "FUEL_STATUS":                "Fuel System Status",
    "FUEL_RAIL_PRESSURE_DIRECT":  "Fuel Rail Pressure Direct (kPa)",
    "FUEL_RAIL_PRESSURE_VAC":     "Fuel Rail Pressure Vac (kPa)",
    "FUEL_RAIL_PRESSURE_ABS":     "Fuel Rail Pressure Abs (kPa)",
    "LONG_FUEL_TRIM_1":           "Long Term Fuel Trim Bank 1 (%)",
    "SHORT_FUEL_TRIM_2":          "Short Term Fuel Trim Bank 2 (%)",
    "LONG_FUEL_TRIM_2":           "Long Term Fuel Trim Bank 2 (%)",
    "RELATIVE_THROTTLE_POS":      "Relative Throttle Position (%)",
    "O2_B1S1":                    "O2 Sensor B1S1 (V)",
    "O2_B1S2":                    "O2 Sensor B1S2 (V)",
    "O2_B2S1":                    "O2 Sensor B2S1 (V)",
    "O2_B2S2":                    "O2 Sensor B2S2 (V)",
    "AMBIANT_AIR_TEMP":           "Ambient Air Temp OBD (°C)",
    "CATALYST_TEMP_B1S1":         "Catalyst Temp B1S1 (°C)",
    "BAROMETRIC_PRESSURE":        "Barometric Pressure (kPa)",
    "EGR_ERROR":                  "EGR Error (%)",
    "COMMANDED_EGR":              "Commanded EGR (%)",
    "EVAPORATIVE_PURGE":          "Evaporative Purge (%)",
    "DISTANCE_SINCE_DTC_CLEAR":   "Distance Since DTC Clear (km)",
    "DISTANCE_W_MIL":             "Distance With MIL On (km)",
    "TIME_SINCE_DTC_CLEARED":     "Time Since DTC Cleared (min)",
    "WARMUPS_SINCE_DTC_CLEAR":    "Warmups Since DTC Clear",
    "GET_DTC":                    "Diagnostic Trouble Codes",
    "FREEZE_DTC":                 "Freeze Frame DTC",
    "STATUS":                     "MIL Status",
    "ETHANOL_PERCENT":            "Ethanol Fuel Percent (%)",
    "MAX_MAF":                    "Maximum MAF (g/s)",
}

ALL_PIDS = {**FAST_PIDS, **SLOW_PIDS}

# ── DERIVED CALCULATIONS ──────────────────────────────────────────────────────

def calculate_derived(vals, specs):
    derived = {}

    rpm        = vals.get("RPM")
    load       = vals.get("ENGINE_LOAD")
    abs_load   = vals.get("ABSOLUTE_LOAD")
    speed      = vals.get("SPEED")
    maf        = vals.get("MAF")
    fuel_rate  = vals.get("FUEL_RATE")
    intake_tmp = vals.get("INTAKE_TEMP")
    throttle   = vals.get("THROTTLE_POS")

    effective_load = abs_load if abs_load is not None else load

    # Torque (Nm)
    if effective_load is not None and rpm is not None:
        load_frac  = effective_load / 100.0
        max_tq_rpm = specs["max_torque_rpm"]
        if rpm < 800:
            rpm_factor = 0.3
        elif rpm < max_tq_rpm:
            rpm_factor = 0.6 + 0.4 * (rpm / max_tq_rpm)
        elif rpm < max_tq_rpm * 1.5:
            rpm_factor = 1.0
        else:
            rpm_factor = max(0.5, 1.0 - (rpm - max_tq_rpm * 1.5) / (specs["max_power_rpm"] * 2))
        derived["calc_torque_nm"] = round(load_frac * specs["max_torque_nm"] * rpm_factor, 1)
    else:
        derived["calc_torque_nm"] = None

    # Power (kW and hp)
    if derived["calc_torque_nm"] is not None and rpm is not None and rpm > 0:
        power_kw = round((derived["calc_torque_nm"] * rpm) / 9549.0, 2)
        derived["calc_power_kw"] = power_kw
        derived["calc_power_hp"] = round(power_kw * 1.341, 1)
    else:
        derived["calc_power_kw"] = None
        derived["calc_power_hp"] = None

    # Fuel consumption
    if fuel_rate is not None:
        derived["calc_fuel_rate_lph"] = round(fuel_rate, 2)
        derived["calc_fuel_l100km"] = (
            round((fuel_rate / speed) * 100, 2) if speed is not None and speed > 2 else None
        )
    elif maf is not None:
        if specs["fuel_type"] == "diesel":
            # Diesel engines run lean — effective AFR is load-dependent.
            # Stoichiometric AFR for diesel = 14.5, but actual lambda varies:
            #   Idle / light load (<25%):  lambda ~3.5  → eff. AFR ~50
            #   Cruise (25-60%):           lambda ~2.2  → eff. AFR ~32
            #   Moderate load (60-80%):    lambda ~1.6  → eff. AFR ~23
            #   Full load (>80%):          lambda ~1.25 → eff. AFR ~18
            # Using ENGINE_LOAD (or ABSOLUTE_LOAD) to pick the lambda.
            eff_load = effective_load if effective_load is not None else 40.0
            if eff_load < 25:
                eff_afr = 50.0
            elif eff_load < 60:
                eff_afr = 32.0
            elif eff_load < 80:
                eff_afr = 23.0
            else:
                eff_afr = 18.0
            density = 0.832  # diesel density kg/L
        else:
            eff_afr = 14.7   # petrol stoichiometric
            density = 0.745  # petrol density kg/L
        fuel_rate_est = round((maf * 3600) / (eff_afr * density * 1000), 2)
        derived["calc_fuel_rate_lph"] = fuel_rate_est
        derived["calc_fuel_l100km"] = (
            round((fuel_rate_est / speed) * 100, 2) if speed is not None and speed > 10 else None
        )
    else:
        derived["calc_fuel_rate_lph"] = None
        derived["calc_fuel_l100km"]   = None

    # Engine efficiency
    if derived["calc_power_kw"] is not None and rpm:
        theoretical_max = specs["max_power_kw"] * min(1.0, rpm / specs["max_power_rpm"])
        derived["calc_engine_efficiency_pct"] = round(
            min(100, (derived["calc_power_kw"] / max(theoretical_max, 0.1)) * 100), 1
        )
    else:
        derived["calc_engine_efficiency_pct"] = None

    # Volumetric efficiency
    if maf is not None and rpm is not None and rpm > 0:
        air_density = 1.293 * (273.15 / (273.15 + intake_tmp)) if intake_tmp is not None else 1.20
        theoretical_maf = (specs["displacement_cc"] / 1e6) / 2 * (rpm / 60) * air_density * 1000
        derived["calc_volumetric_efficiency_pct"] = round(
            min(120, (maf / max(theoretical_maf, 0.1)) * 100), 1
        )
    else:
        derived["calc_volumetric_efficiency_pct"] = None

    # Throttle response
    if throttle is not None and effective_load is not None and throttle > 5:
        derived["calc_throttle_response"] = round(effective_load / throttle, 3)
    else:
        derived["calc_throttle_response"] = None

    # CO2 g/km
    if derived["calc_fuel_l100km"] is not None:
        co2_factor = 2640 if specs["fuel_type"] == "diesel" else 2310
        derived["calc_co2_g_per_km"] = round(derived["calc_fuel_l100km"] * co2_factor / 100, 1)
    else:
        derived["calc_co2_g_per_km"] = None

    return derived


DERIVED_COLS = [
    "calc_torque_nm", "calc_power_kw", "calc_power_hp",
    "calc_fuel_rate_lph", "calc_fuel_l100km",
    "calc_engine_efficiency_pct", "calc_volumetric_efficiency_pct",
    "calc_throttle_response", "calc_co2_g_per_km",
]

# ── WEATHER ───────────────────────────────────────────────────────────────────

_weather_cache = {
    "ambient_temp": 25.0, "weather_condition": "Clear", "weather_raw": "Unknown",
    "wind_speed": 0.0, "humidity": 50.0, "pressure_hpa": 1013.0,
    "monthly_precipitation": 0.0, "latitude": 0.0, "longitude": 0.0, "city": "Unknown",
}
_last_weather_fetch = 0


def fetch_location():
    try:
        r = requests.get("http://ip-api.com/json/", timeout=5)
        d = r.json()
        if d.get("status") == "success":
            return d["lat"], d["lon"], d.get("city", "Unknown")
    except Exception:
        pass
    return 0.0, 0.0, "Unknown"


def fetch_weather(lat, lon, api_key):
    global _weather_cache
    if not api_key or api_key == "your_key_here":
        return
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={lat}&lon={lon}&appid={api_key}&units=metric")
        d = requests.get(url, timeout=5).json()
        main = d["weather"][0]["main"]
        cmap = {
            "Clear": "Clear", "Clouds": "Clear", "Rain": "Rain", "Drizzle": "Rain",
            "Thunderstorm": "Rain", "Snow": "Snow", "Mist": "Fog", "Fog": "Fog", "Haze": "Fog",
        }
        _weather_cache.update({
            "ambient_temp":          round(d["main"]["temp"], 1),
            "weather_condition":     cmap.get(main, "Clear"),
            "weather_raw":           main,
            "wind_speed":            round(d["wind"]["speed"], 1),
            "humidity":              round(d["main"]["humidity"], 1),
            "pressure_hpa":          round(d["main"]["pressure"], 1),
            "monthly_precipitation": round(d.get("rain", {}).get("1h", 0) * 24 * 30, 1),
        })
    except Exception:
        pass


def get_weather(lat, lon):
    global _last_weather_fetch
    if time.time() - _last_weather_fetch > WEATHER_INTERVAL:
        fetch_weather(lat, lon, OPENWEATHER_API_KEY)
        _last_weather_fetch = time.time()
    return _weather_cache.copy()


WEATHER_COLS = [
    "ambient_temp", "weather_condition", "weather_raw", "wind_speed",
    "humidity", "pressure_hpa", "monthly_precipitation", "latitude", "longitude", "city",
]


def fmt(v, d=1):
    return f"{round(v, d)}" if v is not None and v != "N/A" else "N/A"


# ── CONNECT WITH RETRY ────────────────────────────────────────────────────────

def connect_obd(attempt=1):
    """
    Try to open OBD connection. Returns obd.OBD or None.
    Includes CONNECT_WARMUP delay so ELM327 is ready before any queries.
    """
    print(f"  Connecting to {PORT} at {BAUDRATE} baud"
          f"{' (attempt ' + str(attempt) + ')' if attempt > 1 else ''}...")
    try:
        conn = obd.OBD(PORT, baudrate=BAUDRATE, fast=False, timeout=30)
        if conn.is_connected():
            # FIX 1: Wait for ELM327 to fully initialise before any queries.
            # Without this, the adapter returns null for everything even though
            # the serial handshake already succeeded.
            print(f"  ⏳ Waiting {CONNECT_WARMUP}s for ELM327 to initialise...")
            time.sleep(CONNECT_WARMUP)
            return conn
        print("  ⚠️  Port opened but ECU not responding.")
        conn.close()
    except Exception as e:
        print(f"  ❌ {e}")
    return None


# ── PID QUERY HELPERS ─────────────────────────────────────────────────────────

def get_cmd(pid_name):
    """
    FIX 2: Use obd.commands[pid_name] (correct python-obd API).
    obd.commands.get() is not a standard method and silently returns None,
    which made every single PID look unsupported.
    """
    try:
        return obd.commands[pid_name]
    except KeyError:
        return None


def safe_query(conn, pid_name):
    """
    Query one PID with QUERY_DELAY guard. Returns extracted value or None.
    """
    try:
        cmd = get_cmd(pid_name)
        if cmd is None:
            return None
        time.sleep(QUERY_DELAY)
        resp = conn.query(cmd)
        if resp.is_null():
            return None
        val = resp.value
        if hasattr(val, 'magnitude'): return round(float(val.magnitude), 3)
        if isinstance(val, list):     return str(val)
        return val
    except Exception:
        return None


# ── PID SCAN ──────────────────────────────────────────────────────────────────

def run_pid_scan(conn):
    """
    Query all PIDs once. Returns (supported OrderedDict, unsupported list).
    """
    supported   = OrderedDict()
    unsupported = []

    print(f"\n  Scanning {len(ALL_PIDS)} PIDs ({int(QUERY_DELAY*1000)}ms gap)...\n")

    for pid_name, description in ALL_PIDS.items():
        cmd = get_cmd(pid_name)
        if cmd is None:
            unsupported.append((pid_name, description, "not in obd library"))
            continue

        try:
            time.sleep(QUERY_DELAY)
            response = conn.query(cmd)

            if response.is_null():
                unsupported.append((pid_name, description, "null response"))
            else:
                val = response.value
                if hasattr(val, 'magnitude'): val = round(float(val.magnitude), 3)
                elif isinstance(val, list):   val = str(val)
                supported[pid_name] = {"description": description, "current_value": val}
                print(f"  ✅ {description:<45} = {val}")

        except Exception as e:
            unsupported.append((pid_name, description, str(e)))

    return supported, unsupported


# ── MAIN ──────────────────────────────────────────────────────────────────────

print("=" * 65)
print(f"  test_7_full_scan.py — {VEHICLE_SPECS['name']}")
print(f"  {VEHICLE_SPECS['engine']} | {VEHICLE_SPECS['max_power_kw']}kW | "
      f"{VEHICLE_SPECS['max_torque_nm']}Nm")
print("=" * 65)

print("\n  📍 Getting location...")
lat, lon, city = fetch_location()
_weather_cache.update({"latitude": lat, "longitude": lon, "city": city})
print(f"  Location: {city} ({lat:.2f}, {lon:.2f})")

if OPENWEATHER_API_KEY != "your_key_here":
    print("  🌤  Fetching weather...")
    fetch_weather(lat, lon, OPENWEATHER_API_KEY)
    _last_weather_fetch = time.time()
    print(f"  {_weather_cache['weather_raw']} | {_weather_cache['ambient_temp']}°C | "
          f"Humidity: {_weather_cache['humidity']}%")
else:
    print("  ⚠️  No API key — weather defaults used")

# ── Initial connection ────────────────────────────────────────────────────────
conn = None
for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
    conn = connect_obd(attempt)
    if conn:
        print(f"  ✅ Connected! Protocol: {conn.protocol_name()}")
        break
    if attempt < MAX_RECONNECT_ATTEMPTS:
        print(f"  Retrying in {RECONNECT_DELAY}s...")
        time.sleep(RECONNECT_DELAY)

if conn is None:
    print(f"\n  ❌ Could not connect after {MAX_RECONNECT_ATTEMPTS} attempts.")
    sys.exit(1)

# ── PID scan ──────────────────────────────────────────────────────────────────
supported_pids, unsupported_pids = run_pid_scan(conn)

print(f"\n  {'─'*50}")
print(f"  Supported PIDs:   {len(supported_pids)}")
print(f"  Unsupported PIDs: {len(unsupported_pids)}")

if not supported_pids:
    print("\n  ❌ No PIDs supported.")
    print("     Things to check:")
    print("     1. Engine running (not just ignition on)?")
    print("     2. Try BAUDRATE = 9600 if 38400 fails")
    print("     3. Some ELM327 clones need fast=True — try changing that")
    conn.close()
    sys.exit(1)

# Calculated values at scan time
test_vals = {
    p: info["current_value"] for p, info in supported_pids.items()
    if isinstance(info["current_value"], (int, float))
}
test_derived = calculate_derived(test_vals, VEHICLE_SPECS)
print(f"\n  Calculated values (at scan time):")
print(f"    Torque:       {test_derived['calc_torque_nm']} Nm")
print(f"    Power:        {test_derived['calc_power_kw']} kW  ({test_derived['calc_power_hp']} hp)")
print(f"    Fuel rate:    {test_derived['calc_fuel_rate_lph']} L/h")
print(f"    Fuel economy: {test_derived['calc_fuel_l100km']} L/100km")
print(f"    CO2:          {test_derived['calc_co2_g_per_km']} g/km")

# Save scan report
os.makedirs("logs", exist_ok=True)
ts        = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
safe_name = VEHICLE_SPECS["name"].replace(" ", "_")
scan_path = f"logs/pid_scan_{ts}.txt"
log_path  = f"logs/full_scan_{ts}_{safe_name}.csv"

with open(scan_path, "w") as sc:
    sc.write(f"Full PID Scan — {VEHICLE_SPECS['name']}\nDate: {datetime.now()}\n\n")
    sc.write("SUPPORTED:\n")
    for pid, info in supported_pids.items():
        sc.write(f"  {pid:<40} {info['description']} = {info['current_value']}\n")
    sc.write("\nUNSUPPORTED:\n")
    for pid, desc, reason in unsupported_pids:
        sc.write(f"  {pid:<40} {desc} ({reason})\n")
    sc.write("\nCALCULATED SPECS:\n")
    for k, v in test_derived.items():
        sc.write(f"  {k}: {v}\n")
print(f"  📝 Scan saved: {scan_path}")

# Split into confirmed fast/slow tiers
supported_fast = [p for p in FAST_PIDS if p in supported_pids]
supported_slow = [p for p in SLOW_PIDS if p in supported_pids]
print(f"\n  Fast PIDs (every cycle):       {len(supported_fast)}")
print(f"  Slow PIDs (every {SLOW_PID_EVERY} cycles):    {len(supported_slow)}")

# CSV columns
csv_cols = (
    ["timestamp", "elapsed_seconds", "reconnects"]
    + list(supported_pids.keys())
    + DERIVED_COLS
    + WEATHER_COLS
)

# Carry-forward buffer for slow PIDs between cycles
slow_vals_cache = {p: None for p in supported_slow}

print(f"\n  Logging every {INTERVAL}s to: {log_path}")
print(f"  Ctrl+C to stop.\n")
print(f"  {'Time':<10} {'RPM':<8} {'Temp':<7} {'Speed':<8} {'Load':<7} "
      f"{'Torque':<9} {'Power':<9} {'L/100km':<9} {'Weather':<10} {'°C'}")
print("  " + "─" * 88)

start_time    = time.time()
reading_count = 0
reconnects    = 0
cycle_count   = 0

try:
    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_cols)
        writer.writeheader()

        while True:
            cycle_start = time.time()

            # ── Health check + auto-reconnect ─────────────────────────────────
            if not conn.is_connected():
                print(f"\n  ⚠️  BT disconnected. Reconnecting...")
                try:
                    conn.close()
                except Exception:
                    pass
                reconnected = False
                for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
                    time.sleep(RECONNECT_DELAY)
                    conn = connect_obd(attempt)
                    if conn:
                        reconnects += 1
                        reconnected = True
                        print(f"  ✅ Reconnected! (total: {reconnects})")
                        break
                if not reconnected:
                    print(f"\n  ❌ Could not reconnect after "
                          f"{MAX_RECONNECT_ATTEMPTS} attempts. Stopping.")
                    break

            elapsed = time.time() - start_time
            weather = get_weather(lat, lon)

            row = {
                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(elapsed, 1),
                "reconnects":      reconnects,
            }

            raw_vals = {}

            # Fast PIDs — every cycle
            for pid_name in supported_fast:
                val = safe_query(conn, pid_name)
                row[pid_name]      = val
                raw_vals[pid_name] = val

            # Slow PIDs — every SLOW_PID_EVERY cycles; else carry forward
            if cycle_count % SLOW_PID_EVERY == 0:
                for pid_name in supported_slow:
                    val = safe_query(conn, pid_name)
                    slow_vals_cache[pid_name] = val
                    row[pid_name]             = val
                    raw_vals[pid_name]        = val
            else:
                for pid_name in supported_slow:
                    cached             = slow_vals_cache.get(pid_name)
                    row[pid_name]      = cached
                    raw_vals[pid_name] = cached

            # Derived columns
            derived = calculate_derived(raw_vals, VEHICLE_SPECS)
            for col in DERIVED_COLS:
                row[col] = derived.get(col)

            # Weather columns
            for wc in WEATHER_COLS:
                row[wc] = weather.get(wc, "")

            writer.writerow(row)
            csvfile.flush()
            reading_count += 1
            cycle_count   += 1

            # Terminal output
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            reconnect_tag = f"[R:{reconnects}]" if reconnects > 0 else ""
            print(
                f"  {mins:02d}:{secs:02d} {reconnect_tag:<6} "
                f"{fmt(raw_vals.get('RPM')):<8} "
                f"{fmt(raw_vals.get('COOLANT_TEMP'))}°C  "
                f"{fmt(raw_vals.get('SPEED'))}km/h  "
                f"{fmt(raw_vals.get('ENGINE_LOAD'))}%   "
                f"{fmt(derived.get('calc_torque_nm'))}Nm   "
                f"{fmt(derived.get('calc_power_kw'))}kW    "
                f"{fmt(derived.get('calc_fuel_l100km'))}L/100  "
                f"{weather['weather_condition']:<10} {weather['ambient_temp']}°C"
            )

            # Sleep only the remaining interval (queries already consumed some time)
            elapsed_this_cycle = time.time() - cycle_start
            sleep_remaining    = max(0.0, INTERVAL - elapsed_this_cycle)
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

except KeyboardInterrupt:
    print(f"\n\n  Stopped — {reading_count} readings, {reconnects} reconnect(s).")
except Exception as e:
    print(f"\n  ❌ Unexpected error: {e}")
finally:
    try:
        conn.close()
    except Exception:
        pass

print(f"\n  📁 Full data: {log_path}")
print(f"  📝 PID scan:  {scan_path}")
print("=" * 65)