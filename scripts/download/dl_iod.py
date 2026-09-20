#!/usr/bin/env python3
"""
Download: Indian Ocean Dipole (IOD) Index — NOAA PSL + JAMSTEC
==============================================================
Run this script directly:
    python3 scripts/download/dl_iod.py

What gets downloaded:
    • IOD / Dipole Mode Index (DMI) monthly time series
    • Source 1: NOAA PSL (Hadley Centre)
    • Source 2: JAMSTEC (Japan Agency for Marine-Earth Science)
    • Output → data/raw/iod/

No account needed — fully open access.

What is IOD?
    The Indian Ocean Dipole is a large-scale climate mode that flips between
    positive (warm west, cold east) and negative (cold west, warm east) phases.
    It directly affects Bay of Bengal heat content, monsoon strength, and
    cyclone intensity every season.
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["requests", "pandas"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦  Installing {pkg} …")
        pip_install(pkg)

import requests
import pandas as pd
from pathlib import Path
from io import StringIO

OUTPUT_DIR = "data/raw/iod"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — IOD / Dipole Mode Index Download")
print(f"   Sources : NOAA PSL (primary) + JAMSTEC (secondary)")
print(f"   Output  : {OUTPUT_DIR}/")
print("="*60)
print()

# ── Source 1: NOAA PSL — Hadley Centre DMI ───────────────────────────────────
NOAA_URL     = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
NOAA_OUT     = Path(OUTPUT_DIR) / "iod_dmi_noaa_monthly.csv"

print("  → Downloading IOD DMI from NOAA PSL …")
try:
    r = requests.get(NOAA_URL, timeout=30)
    r.raise_for_status()
    raw = r.text.strip().splitlines()

    # Parse: first line = year range, then year + 12 monthly values
    rows = []
    for line in raw[1:]:
        parts = line.split()
        if len(parts) == 13:
            year = int(parts[0])
            for m, val in enumerate(parts[1:], start=1):
                v = float(val)
                if v < -99:        # missing value flag
                    v = float("nan")
                rows.append({"year": year, "month": m, "dmi": v})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df[["date", "dmi"]].set_index("date")
    # Filter to our period + a few years context
    df = df["2015":"2024"]
    df.to_csv(NOAA_OUT)
    print(f"  ✅ NOAA DMI saved → {NOAA_OUT}")
    print(f"     Records: {len(df)} months")

except Exception as e:
    print(f"  ❌  NOAA download failed: {e}")

# ── Source 2: JAMSTEC DMI ─────────────────────────────────────────────────────
JAMSTEC_URL  = "https://www.jamstec.go.jp/aplec/cl/iod_val.dat"
JAMSTEC_OUT  = Path(OUTPUT_DIR) / "iod_dmi_jamstec_monthly.csv"

print()
print("  → Downloading IOD DMI from JAMSTEC …")
try:
    r = requests.get(JAMSTEC_URL, timeout=30)
    r.raise_for_status()
    raw = r.text.strip().splitlines()

    rows = []
    for line in raw:
        parts = line.split()
        if len(parts) >= 2:
            try:
                date = pd.to_datetime(parts[0])
                val  = float(parts[1])
                rows.append({"date": date, "dmi": val})
            except Exception:
                continue

    df = pd.DataFrame(rows).set_index("date").sort_index()
    df = df["2015":"2024"]
    df.to_csv(JAMSTEC_OUT)
    print(f"  ✅ JAMSTEC DMI saved → {JAMSTEC_OUT}")
    print(f"     Records: {len(df)} months")

except Exception as e:
    print(f"  ⚠️  JAMSTEC download failed (optional): {e}")
    print("      NOAA source is sufficient.")

print()
print("✅  IOD download complete!")
print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
print()
print("  Usage in preprocessing:")
print("    df = pd.read_csv('data/raw/iod/iod_dmi_noaa_monthly.csv', index_col=0, parse_dates=True)")
print("    iod_value = df.loc[target_date.strftime('%Y-%m'), 'dmi'].values[0]")
