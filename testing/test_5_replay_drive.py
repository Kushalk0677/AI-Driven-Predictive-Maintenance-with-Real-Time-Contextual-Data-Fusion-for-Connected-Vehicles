"""
testing/test_5_replay_drive.py
================================
STEP 5 — Replay a logged drive CSV offline.

No OBD dongle needed. Replays a drive log from test_4_log_drive.py
with colour-coded predictions and a full summary at the end.

Usage:
    # Auto-select from logs/ folder
    python testing/test_5_replay_drive.py

    # Specify a file
    python testing/test_5_replay_drive.py --file logs/2026-03-18_Safari.csv

    # Replay at 5x speed
    python testing/test_5_replay_drive.py --speed 5

    # Instant replay (no delay)
    python testing/test_5_replay_drive.py --speed 0
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vehiclepm'))

from vehiclepm.obd.logger import replay_drive

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Replay a logged drive CSV")
parser.add_argument("--file",  default=None, help="Path to drive log CSV")
parser.add_argument("--speed", type=float,   default=1.0,
                    help="Playback speed (1=realtime, 5=5x faster, 0=instant)")
args = parser.parse_args()

# ── Find log file ─────────────────────────────────────────────────────────────
log_file = args.file

if not log_file:
    logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    if os.path.exists(logs_dir):
        logs = sorted([f for f in os.listdir(logs_dir) if f.endswith(".csv")])
        if logs:
            print("\nAvailable drive logs:")
            for i, f in enumerate(logs):
                size = os.path.getsize(os.path.join(logs_dir, f)) // 1024
                print(f"  [{i+1}] {f} ({size} KB)")
            choice = input("\nEnter number (or press Enter for latest): ").strip()
            if choice == "":
                idx = len(logs) - 1  # latest
            else:
                idx = int(choice) - 1
            log_file = os.path.join(logs_dir, logs[idx])
        else:
            print("No drive logs found in logs/ folder.")
            print("Run test_4_log_drive.py first to record a drive.")
            sys.exit(1)
    else:
        print("No logs/ folder found. Run test_4_log_drive.py first.")
        sys.exit(1)

if not os.path.exists(log_file):
    print(f"File not found: {log_file}")
    sys.exit(1)

# ── Replay ────────────────────────────────────────────────────────────────────
replay_drive(log_file, speed=args.speed, show_summary=True)
