#!/usr/bin/env python3
"""
Download: River Discharge — GloFAS via EWDS API
=================================================
Run this script directly:
    python3 scripts/download/dl_glofas.py

What gets downloaded:
    • GloFAS-ERA5 Historical River Discharge
    • Variable: average_river_discharge_in_the_last_24_hours
    • Output → data/raw/glofas/
"""

import os, sys
from pathlib import Path

try:
    import cdsapi
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cdsapi>=0.7.7"])
    import cdsapi

OUTPUT_DIR  = "data/raw/glofas"
START_YEAR  = 2017
END_YEAR    = 2023

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("OceanEmbed v2 — GloFAS River Discharge Download")
print(f"   Dataset : cems-glofas-historical (v4.0 on EWDS)")
print(f"   Period  : {START_YEAR} → {END_YEAR}")
print(f"   Output  : {OUTPUT_DIR}/")
print("="*60)

# Connect to EWDS (CEMS Early Warning Data Store)
client = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api")

for year in range(START_YEAR, END_YEAR + 1):
    out_file = Path(OUTPUT_DIR) / f"glofas_discharge_{year}.nc"

    if out_file.exists():
        print(f"  Skipping {year} (already exists)")
        continue

    print(f"\n  Requesting {year} from EWDS...")
    client.retrieve(
        "cems-glofas-historical",
        {
            "system_version":     ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type":       ["consolidated"],
            "variable":           ["average_river_discharge_in_the_last_24_hours"],
            "timespan":           ["time_mean"],
            "year":               [str(year)],
            "month":              [f"{m:02d}" for m in range(1, 13)],
            "day":                [f"{d:02d}" for d in range(1, 32)],
            "area":               [25, 75, 5, 100],  # Bay of Bengal river basins
            "data_format":        "netcdf",
            "download_format":    "unarchived",
        },
        str(out_file),
    )
    print(f"  {year} done → {out_file}")

print("\nGloFAS download complete!")
print(f"Files saved to: {Path(OUTPUT_DIR).resolve()}")
