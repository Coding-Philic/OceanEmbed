#!/usr/bin/env python3
"""
Download: Indian Ocean Dipole (IOD) Index — NOAA PSL
====================================================
Run this script directly:
    python3 scripts/download/dl_iod.py

What gets downloaded:
    * IOD / Dipole Mode Index (DMI) monthly time series from NOAA PSL (Hadley Centre)
    * Output -> data/raw/iod/iod_dmi_noaa_monthly.csv and data/raw/iod/dmi_noaa.txt
"""

import os
import sys
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("data/raw/iod")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOAA_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
RAW_OUT = OUTPUT_DIR / "dmi_noaa.txt"
CSV_OUT = OUTPUT_DIR / "iod_dmi_noaa_monthly.csv"


def download_iod():
    print("=" * 60)
    print("OceanEmbed - Indian Ocean Dipole (IOD) Index Download")
    print(f"Source : NOAA Physical Sciences Laboratory (Hadley Centre DMI)")
    print(f"Target : {OUTPUT_DIR}/")
    print("=" * 60)

    try:
        import requests
        import pandas as pd
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests", "pandas"])
        import requests
        import pandas as pd

    print("\n--- Downloading IOD DMI from NOAA PSL ---")
    r = requests.get(NOAA_URL, timeout=30)
    r.raise_for_status()

    # Save raw text file
    with open(RAW_OUT, "w") as f:
        f.write(r.text)
    print(f"[OK] Raw text saved -> {RAW_OUT}")

    # Parse into tabular format
    raw = r.text.strip().splitlines()
    rows = []
    for line in raw[1:]:
        parts = line.split()
        if len(parts) == 13:
            try:
                year = int(parts[0])
                for m, val in enumerate(parts[1:], start=1):
                    v = float(val)
                    if v < -99:
                        v = float("nan")
                    rows.append({"year": year, "month": m, "dmi": v})
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df[["date", "dmi"]].set_index("date")

    # Filter to period: 2015 to present
    df = df["2015":]
    df.to_csv(CSV_OUT)
    print(f"[OK] Processed monthly CSV saved -> {CSV_OUT} ({len(df)} records)")
    print("\nIOD Download Complete.")
    return True


if __name__ == "__main__":
    download_iod()
