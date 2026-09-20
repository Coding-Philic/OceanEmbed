#!/usr/bin/env python3
"""
Download: River Discharge — GloFAS via CDS API
===============================================
Run this script directly:
    python3 scripts/download/dl_glofas.py

What gets downloaded:
    • GloFAS-ERA5 Historical River Discharge
    • Variable: river_discharge_in_the_last_24_hours
    • Output → data/raw/glofas/

Setup (one-time):
    1. Register at: https://cds.climate.copernicus.eu/
    2. Create ~/.cdsapirc file:
          url: https://cds.climate.copernicus.eu/api/v2
          key: YOUR-UID:YOUR-API-KEY
       (Find your key at: https://cds.climate.copernicus.eu/api-how-to)
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

OUTPUT_DIR  = "data/raw/glofas"
START_YEAR  = 2017
END_YEAR    = 2023

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — GloFAS River Discharge Download")
print(f"   Dataset : cems-glofas-historical (v4.0)")
print(f"   Period  : {START_YEAR} → {END_YEAR}")
print(f"   Output  : {OUTPUT_DIR}/")
print("="*60)
print()

c = cdsapi.Client()

for year in range(START_YEAR, END_YEAR + 1):
    out_file = Path(OUTPUT_DIR) / f"glofas_discharge_{year}.nc"

    if out_file.exists():
        print(f"  ⏭  Skipping {year} (already exists)")
        continue

    print(f"\n  → Requesting {year} from CDS …")
    c.retrieve(
        "cems-glofas-historical",
        {
            "system_version":     "version_4_0",
            "variable":           "river_discharge_in_the_last_24_hours",
            "format":             "netcdf",
            "hydrological_model": "lisflood",
            "product_type":       "consolidated",
            "year":               str(year),
            "month":              [f"{m:02d}" for m in range(1, 13)],
            "day":                [f"{d:02d}" for d in range(1, 32)],
        },
        str(out_file),
    )
    print(f"  ✅  {year} done → {out_file}")

print("\n✅  GloFAS download complete!")
print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
