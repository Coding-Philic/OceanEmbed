#!/usr/bin/env python3
"""
Download: WOA23 Climatology (NOAA NCEI) — Temperature & Salinity
================================================================
Run this script directly:
    python3 scripts/download/dl_woa23.py --var temperature
    python3 scripts/download/dl_woa23.py --var salinity
    python3 scripts/download/dl_woa23.py --var all

What gets downloaded:
    * World Ocean Atlas 2023 (WOA23) Decadal Climatology (1955-2022)
    * 12 monthly NetCDF files per variable (1.00 degree, 102 depth levels)
    * Output -> data/raw/woa23/<variable>/
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

VAR_CONFIG = {
    "temperature": {
        "prefix": "t",
        "url": "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/1.00/",
        "dir": Path("data/raw/woa23/temperature"),
        "min_size": 50_000_000,
    },
    "salinity": {
        "prefix": "s",
        "url": "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/salinity/netcdf/decav/1.00/",
        "dir": Path("data/raw/woa23/salinity"),
        "min_size": 45_000_000,
    },
}


def download_variable(var_name: str) -> bool:
    cfg = VAR_CONFIG[var_name]
    out_dir = cfg["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"OceanEmbed - WOA23 {var_name.capitalize()} Climatology Download")
    print(f"Source : NOAA NCEI WOA23 (Decadal 1955-2022, 1.00 deg)")
    print(f"Target : {out_dir}/")
    print("=" * 60)

    success_count = 0
    total_months = 12

    for m in range(1, total_months + 1):
        fname = f"woa23_decav_{cfg['prefix']}{m:02d}_01.nc"
        out_file = out_dir / fname
        url = cfg["url"] + fname

        if out_file.exists() and out_file.stat().st_size > cfg["min_size"]:
            print(f"[OK] Month {m:02d} already exists: {fname} ({out_file.stat().st_size / 1e6:.1f} MB)")
            success_count += 1
            continue

        print(f"\n[{m:02d}/{total_months:02d}] Downloading {fname}...")
        try:
            cmd = ["curl", "-L", "-o", str(out_file), url]
            subprocess.run(cmd, check=True)
            print(f"[OK] Saved -> {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to download {fname}: {e}")

    print("\n" + "=" * 60)
    print(f"WOA23 {var_name.capitalize()} Complete: {success_count}/{total_months} files ready.")
    print("=" * 60)
    return success_count == total_months


def main():
    parser = argparse.ArgumentParser(description="Download WOA23 Climatology")
    parser.add_argument(
        "--var",
        choices=["temperature", "salinity", "all"],
        default="all",
        help="Variable to download (temperature, salinity, or all)",
    )
    args = parser.parse_args()

    targets = ["temperature", "salinity"] if args.var == "all" else [args.var]
    all_ok = True
    for t in targets:
        ok = download_variable(t)
        if not ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
