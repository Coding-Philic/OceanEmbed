#!/usr/bin/env python3
"""
Download: Surface Currents U & V — CMEMS GlobCurrent L4
========================================================
Run this script directly:
    python3 scripts/download/dl_currents.py

What gets downloaded:
    * uo — Total Eastward surface current velocity (m/s)
    * vo — Total Northward surface current velocity (m/s)
    * Output -> data/raw/currents/
"""

import os
import sys
import subprocess
from pathlib import Path

DATASET_ID = "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m"
VARIABLES = ["uo", "vo"]
OUTPUT_DIR = Path("data/raw/currents")

LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0, 25.0
START_DATE = "2017-01-01"
END_DATE = "2023-12-31"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("OceanEmbed - Surface Currents Download (Dataset 6)")
print(f"Dataset  : {DATASET_ID}")
print(f"Variables: {', '.join(VARIABLES)}")
print(f"Region   : lon {LON_MIN} to {LON_MAX}, lat {LAT_MIN} to {LAT_MAX}")
print(f"Period   : {START_DATE} to {END_DATE}")
print(f"Output   : {OUTPUT_DIR}/")
print("=" * 60)

cmd = [
    "copernicusmarine", "subset",
    "--dataset-id", DATASET_ID,
    "--minimum-longitude", str(LON_MIN),
    "--maximum-longitude", str(LON_MAX),
    "--minimum-latitude", str(LAT_MIN),
    "--maximum-latitude", str(LAT_MAX),
    "--start-datetime", f"{START_DATE}T00:00:00",
    "--end-datetime", f"{END_DATE}T23:59:59",
    "--output-directory", str(OUTPUT_DIR),
    "--output-filename", "currents_2017_2023.nc",
    "--overwrite",
]
for v in VARIABLES:
    cmd += ["--variable", v]

result = subprocess.run(cmd)
if result.returncode == 0:
    print(f"\n[OK] Currents download complete -> {OUTPUT_DIR.resolve()}")
else:
    print("\n[ERROR] Download failed. Check your Copernicus Marine credentials.")
    sys.exit(1)
