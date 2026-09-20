#!/usr/bin/env python3
"""
Download: Sea Level Pressure (SLP) — ERA5 via CDS API
======================================================
Run this script directly:
    python3 scripts/download/dl_slp.py

What gets downloaded:
    • mean_sea_level_pressure  (hPa, daily)
    • Output → data/raw/slp/

Setup (one-time):
    pip install cdsapi
    Create ~/.cdsapirc: https://cds.climate.copernicus.eu/api-how-to
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import cdsapi  # noqa: F401
except ImportError:
    print("📦  Installing cdsapi …")
    pip_install("cdsapi")

import cdsapi
from pathlib import Path

OUTPUT_DIR  = "data/raw/slp"
LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
START_YEAR  = 2017
END_YEAR    = 2023

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — Sea Level Pressure Download (ERA5)")
print(f"   Region : lon {LON_MIN}–{LON_MAX}  lat {LAT_MIN}–{LAT_MAX}")
print(f"   Period : {START_YEAR} → {END_YEAR}")
print(f"   Output : {OUTPUT_DIR}/")
print("="*60)

c = cdsapi.Client()

for year in range(START_YEAR, END_YEAR + 1):
    out_file = Path(OUTPUT_DIR) / f"slp_{year}.nc"
    if out_file.exists():
        print(f"  ⏭  Skipping {year} (already exists)")
        continue

    print(f"\n  → Downloading {year} …")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable":     ["mean_sea_level_pressure"],
            "year":         str(year),
            "month":        [f"{m:02d}" for m in range(1, 13)],
            "day":          [f"{d:02d}" for d in range(1, 32)],
            "time":         "12:00",
            "area":         [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
            "format":       "netcdf",
        },
        str(out_file),
    )
    print(f"  ✅  {year} done → {out_file}")

print("\n✅  SLP download complete!")
print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
