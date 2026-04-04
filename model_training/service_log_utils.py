"""
service_log_utils.py
====================
Shared helpers for loading and applying service logs.

Drop a service CSV (or JSON) in  data/service_logs/  named after the car:
    data/service_logs/Safari_Storme_EX_service.csv
    data/service_logs/Safari_Storme_EX_service.json

Or use a single combined log with a  vehicle_name  column:
    data/service_logs/fleet_services.csv

────────────────────────────────────────────────────────────
Service CSV format
────────────────────────────────────────────────────────────
service_date,service_type,brake_thickness_after,tire_tread_after,oil_degradation_after,battery_health_after,notes
2026-01-15,oil_change,,,,,"Regular oil change"
2026-02-20,brake_replacement,9.0,,,,"New brake pads"
2026-03-10,full_service,9.0,7.5,0.05,,"Annual dealer service"
2026-03-10,battery_replacement,,,, 0.93,"Battery replaced"

service_type values:
  oil_change          — resets oil_degradation → 0.05
  brake_replacement   — resets brake_thickness → 9.0  mm
  tire_replacement    — resets tire_tread      → 7.5  mm
  tire_rotation       — partial reset tire_tread → 6.0 mm (extends life)
  battery_replacement — resets battery_health  → 0.92
  full_service        — resets brakes + tires + oil
  custom              — use the _after columns exclusively

The _after columns override the defaults when provided.

────────────────────────────────────────────────────────────
Service JSON format  (per-car file)
────────────────────────────────────────────────────────────
[
  {"date": "2026-01-15", "type": "oil_change"},
  {"date": "2026-02-20", "type": "brake_replacement", "brake_thickness": 9.0},
  {"date": "2026-03-10", "type": "full_service", "notes": "Dealer annual"}
]

────────────────────────────────────────────────────────────
Date matching
────────────────────────────────────────────────────────────
Drive CSV filenames must start with YYYY-MM-DD (the collection date):
    2026-03-21_session1_Safari_Storme_EX.csv
Service events are applied between consecutive drive dates.
If a filename has no parseable date prefix, drives are matched by index
(service events are applied before drive N where N = drive_index column).
"""

import os, json
import pandas as pd
from glob import glob

# ── Default wear values after each service type ───────────────────────────────
SERVICE_RESETS = {
    "oil_change":          {"oil_degradation": 0.05},
    "brake_replacement":   {"brake_thickness": 9.0},
    "tire_replacement":    {"tire_tread":      7.5},
    "tire_rotation":       {"tire_tread":      6.0},   # partial — extends life
    "battery_replacement": {"battery_health":  0.92},
    "full_service":        {"brake_thickness": 9.0, "tire_tread": 7.5,
                            "oil_degradation": 0.05},
    "custom":              {},                          # use _after columns only
}

# Wear state components that can be reset (maps component → (min, max))
WEAR_BOUNDS = {
    "brake_thickness": (0.1, 12.0),
    "tire_tread":      (0.1, 10.0),
    "oil_degradation": (0.0,  1.0),
    "battery_health":  (0.0,  1.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_service_log(car_name: str, service_dir: str, verbose: bool = True) -> list:
    """
    Load and return service events for *car_name* sorted by date ascending.

    Each event is a dict:
        {
            "date":         pd.Timestamp,
            "service_type": str,
            "resets":       {component: float},   # what to reset and to what value
            "notes":        str,
        }

    Returns an empty list if no log is found (car runs without service history —
    the wear model stays fully degradation-only, which is the v2 behaviour).
    """
    safe   = car_name.replace(" ", "_").replace("/", "_")
    events = []
    found  = None

    # 1. Per-car CSV
    csv_p = os.path.join(service_dir, f"{safe}_service.csv")
    if os.path.exists(csv_p):
        events = _parse_csv(pd.read_csv(csv_p))
        found  = csv_p

    # 2. Per-car JSON
    if not events:
        json_p = os.path.join(service_dir, f"{safe}_service.json")
        if os.path.exists(json_p):
            with open(json_p) as f:
                events = _parse_json(json.load(f))
            found = json_p

    # 3. Fleet-wide CSV with vehicle_name column
    if not events:
        for fpath in sorted(glob(os.path.join(service_dir, "*.csv"))):
            if fpath == csv_p:
                continue
            try:
                df = pd.read_csv(fpath)
                if "vehicle_name" not in df.columns:
                    continue
                sub = df[df["vehicle_name"].str.strip().str.lower() == car_name.lower()]
                if len(sub) == 0:
                    continue
                events = _parse_csv(sub)
                found  = fpath
                break
            except Exception:
                pass

    if not events:
        return []

    events.sort(key=lambda e: e["date"])

    if verbose:
        print(f"     📋 Service log: {os.path.basename(found)} ({len(events)} event(s))")
        for ev in events:
            reset_str = ", ".join(f"{k}→{v}" for k, v in ev["resets"].items()) or "(no wear resets)"
            print(f"        {ev['date'].date()}  {ev['service_type']:<22}  {reset_str}"
                  + (f"  — {ev['notes']}" if ev["notes"] else ""))

    return events


def get_drive_date(fpath: str):
    """
    Extract a pd.Timestamp from the YYYY-MM-DD prefix of the drive filename.
    Returns None if the filename doesn't start with a parseable date.
    """
    fname = os.path.basename(fpath)
    try:
        return pd.to_datetime(fname[:10])
    except Exception:
        return None


def apply_pending_services(wear: dict,
                           service_events: list,
                           prev_date,   # pd.Timestamp | None
                           curr_date,   # pd.Timestamp | None
                           drive_index: int = 0) -> tuple:
    """
    Apply service events that fall between *prev_date* (exclusive) and
    *curr_date* (inclusive) to *wear* in place.

    When dates are unavailable (None), falls back to drive_index matching:
    events whose "drive_index_before" field equals drive_index are applied.

    Returns:
        (updated_wear, applied_services_list)
        applied_services_list — list of service_type strings applied this step
    """
    applied = []
    for ev in service_events:
        trigger = False

        if curr_date is not None and ev["date"] is not None:
            if prev_date is None:
                # First drive: apply any service on or before this drive
                trigger = ev["date"] <= curr_date
            else:
                trigger = prev_date < ev["date"] <= curr_date
        else:
            # Date-less fallback — apply before first drive only
            trigger = drive_index == 0 and prev_date is None

        if trigger:
            for comp, val in ev["resets"].items():
                if comp in wear:
                    lo, hi  = WEAR_BOUNDS.get(comp, (None, None))
                    clamped = float(val)
                    if lo is not None:
                        clamped = max(lo, min(hi, clamped))
                    wear[comp] = round(clamped, 4)
            applied.append(ev["service_type"])

    return wear, applied


# ─────────────────────────────────────────────────────────────────────────────
# Internal parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_csv(df: pd.DataFrame) -> list:
    events = []
    date_col = next((c for c in ("service_date", "date") if c in df.columns), None)
    if date_col is None:
        return events

    for _, row in df.iterrows():
        date_str = str(row.get(date_col, "")).strip()
        if not date_str or date_str.lower() in ("nan", ""):
            continue
        try:
            dt = pd.to_datetime(date_str)
        except Exception:
            continue

        stype  = str(row.get("service_type", "custom")).strip().lower()
        resets = dict(SERVICE_RESETS.get(stype, {}))

        # Override defaults with explicit _after columns
        col_map = {
            "brake_thickness_after": "brake_thickness",
            "tire_tread_after":      "tire_tread",
            "oil_degradation_after": "oil_degradation",
            "battery_health_after":  "battery_health",
        }
        for col, comp in col_map.items():
            raw = row.get(col, None)
            if raw is not None and str(raw).strip() not in ("", "nan"):
                try:
                    resets[comp] = float(raw)
                except Exception:
                    pass

        events.append({
            "date":         dt,
            "service_type": stype,
            "resets":       resets,
            "notes":        str(row.get("notes", "")).strip().replace("nan", ""),
        })
    return events


def _parse_json(data) -> list:
    if isinstance(data, dict):
        data = [data]
    events = []
    for item in data:
        date_str = item.get("date", item.get("service_date", ""))
        if not date_str:
            continue
        try:
            dt = pd.to_datetime(date_str)
        except Exception:
            continue

        stype  = item.get("type", item.get("service_type", "custom")).lower()
        resets = dict(SERVICE_RESETS.get(stype, {}))

        # Direct component keys override defaults
        for comp in WEAR_BOUNDS:
            if comp in item:
                try:
                    resets[comp] = float(item[comp])
                except Exception:
                    pass

        events.append({
            "date":         dt,
            "service_type": stype,
            "resets":       resets,
            "notes":        str(item.get("notes", "")),
        })
    return events
