#!/usr/bin/env python3
"""
Download: Argo Float Profiles (Validation Dataset)
===================================================
Run this script directly:
    python3 scripts/download/dl_argo.py

What gets downloaded:
    * Argo Float Profiles in Bay of Bengal (75E-100E, 5N-25N, 0-1500m)
    * Years: 2017 to 2023 (yearly NetCDF files)
    * Quality filtered: Delayed Mode (data_mode='D') and Quality Control (temp_qc <= 2)
    * Source: Ifremer ERDDAP (Global Argo Data Repository)
    * Output -> data/validation/argo_bob_dm_<year>.nc
"""

import os
import sys
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("data/validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_ERDDAP = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc"
VARS = "platform_number,time,latitude,longitude,pres,temp,psal,temp_qc,psal_qc,data_mode"


def download_argo_year(year: int) -> bool:
    fname = f"argo_bob_dm_{year}.nc"
    out_file = OUTPUT_DIR / fname

    if out_file.exists() and out_file.stat().st_size > 1_000_000:
        print(f"[OK] Year {year} already exists: {fname} ({out_file.stat().st_size / 1e6:.1f} MB)")
        return True

    url = (
        f"{BASE_ERDDAP}?{VARS}"
        f"&latitude%3E=5&latitude%3C=25"
        f"&longitude%3E=75&longitude%3C=100"
        f"&pres%3E=0&pres%3C=1500"
        f"&time%3E={year}-01-01T00:00:00Z"
        f"&time%3C={year}-12-31T23:59:59Z"
        f"&data_mode=%22D%22"
    )

    print(f"\n--- Downloading Argo Profiles for Year {year} ---")
    try:
        cmd = ["curl", "-L", "-o", str(out_file), url]
        subprocess.run(cmd, check=True)
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Saved {fname} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download year {year}: {e}")
        return False


def main():
    print("=" * 60)
    print("OceanEmbed - Argo Float Profiles Download (2017-2023)")
    print("Source : Ifremer ERDDAP (Delayed Mode D, QC <= 2)")
    print(f"Target : {OUTPUT_DIR}/")
    print("=" * 60)

    years = range(2017, 2024)
    success = 0
    for yr in years:
        if download_argo_year(yr):
            success += 1

    print("\n" + "=" * 60)
    print(f"Argo Download Complete: {success}/{len(years)} years ready.")
    print("=" * 60)
    if success < len(years):
        sys.exit(1)


if __name__ == "__main__":
    main()
