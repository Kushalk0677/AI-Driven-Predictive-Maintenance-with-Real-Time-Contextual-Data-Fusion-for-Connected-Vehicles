"""
testing/test_6_compare_cars.py
================================
STEP 6 — Compare multiple logged drives side by side.

After logging drives from different cars, use this to compare
their maintenance risk profiles, driving styles, and engine health.

Usage:
    python testing/test_6_compare_cars.py

Automatically reads all CSVs from the logs/ folder.
"""

import sys
import os
import pandas as pd
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

print("=" * 65)
print("  Step 6 — Multi-Car Drive Comparison")
print("=" * 65)

# ── Find all drive logs ───────────────────────────────────────────────────────
logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')

if not os.path.exists(logs_dir):
    print("\n  No logs/ folder found.")
    print("  Run test_4_log_drive.py for each car first.")
    sys.exit(1)

csv_files = sorted(glob.glob(os.path.join(logs_dir, "*.csv")))

if len(csv_files) == 0:
    print("\n  No drive logs found.")
    print("  Run test_4_log_drive.py for each car first.")
    sys.exit(1)

if len(csv_files) == 1:
    print("\n  Only 1 drive logged so far — log more cars to compare.")
    print(f"  Found: {os.path.basename(csv_files[0])}")

print(f"\n  Found {len(csv_files)} drive log(s):\n")

# ── Per-car summary ───────────────────────────────────────────────────────────
summaries = []

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    name = os.path.basename(csv_path).replace(".csv", "")

    summary = {
        "Drive":            name[:40],
        "Readings":         len(df),
        "Duration (min)":   round(df["elapsed_seconds"].max() / 60, 1),
        "Avg Risk":         f"{df['maintenance_probability'].mean():.1%}",
        "Max Risk":         f"{df['maintenance_probability'].max():.1%}",
        "Avg Engine °C":    round(df["engine_temp"].mean(), 1),
        "Max Engine °C":    round(df["engine_temp"].max(), 1),
        "Avg Speed km/h":   round(df["speed"].mean(), 1),
        "Max Speed km/h":   round(df["speed"].max(), 1),
        "Avg Battery":      f"{df['battery_health'].mean():.0%}",
        "Total DTCs":       int(df["dtc_count"].sum()),
        "Dominant Style":   df["driving_style"].mode()[0],
        "CRITICAL readings":int((df["severity"] == "CRITICAL").sum()),
        "WARNING readings": int((df["severity"] == "WARNING").sum()),
        "City":             df["city"].iloc[0] if "city" in df.columns else "N/A",
    }
    summaries.append(summary)

    # Per-car detail
    print(f"  {'─'*55}")
    print(f"  📁 {name}")
    print(f"  {'─'*55}")
    print(f"  Duration:       {summary['Duration (min)']} min  ({summary['Readings']} readings)")
    print(f"  Location:       {summary['City']}")
    print(f"  Avg risk:       {summary['Avg Risk']}  (max: {summary['Max Risk']})")
    print(f"  Engine temp:    avg {summary['Avg Engine °C']}°C  max {summary['Max Engine °C']}°C")
    print(f"  Speed:          avg {summary['Avg Speed km/h']} km/h  max {summary['Max Speed km/h']} km/h")
    print(f"  Battery health: {summary['Avg Battery']}")
    print(f"  Fault codes:    {summary['Total DTCs']} total DTC readings")
    print(f"  Driving style:  {summary['Dominant Style']}")

    # Severity breakdown
    sev_counts = df["severity"].value_counts()
    total = len(df)
    for sev in ["OK", "WATCH", "WARNING", "CRITICAL"]:
        count = sev_counts.get(sev, 0)
        pct   = count / total * 100
        bar   = "█" * int(pct / 5)
        print(f"  {sev:<10} {bar:<20} {count:>4} ({pct:.0f}%)")
    print()

# ── Side-by-side comparison table ────────────────────────────────────────────
if len(summaries) > 1:
    print(f"\n  {'─'*65}")
    print("  📊 Side-by-Side Comparison")
    print(f"  {'─'*65}")

    df_summary = pd.DataFrame(summaries)
    cols = ["Drive", "Avg Risk", "Max Risk", "Avg Engine °C",
            "Avg Battery", "Total DTCs", "Dominant Style"]
    print(df_summary[cols].to_string(index=False))

    # Highest risk car
    highest_idx = df_summary["Max Risk"].apply(lambda x: float(x.strip('%'))/100).idxmax()
    print(f"\n  ⚠️  Highest max risk: {df_summary.loc[highest_idx, 'Drive']}")

    # Most DTCs
    most_dtc_idx = df_summary["Total DTCs"].idxmax()
    if df_summary.loc[most_dtc_idx, "Total DTCs"] > 0:
        print(f"  🚨 Most fault codes: {df_summary.loc[most_dtc_idx, 'Drive']}")

# ── Save comparison ───────────────────────────────────────────────────────────
out = os.path.join(logs_dir, "comparison_summary.csv")
pd.DataFrame(summaries).to_csv(out, index=False)
print(f"\n  💾 Comparison saved to: {out}")
print("=" * 65)
