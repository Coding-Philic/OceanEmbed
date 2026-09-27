#!/usr/bin/env python3
"""
Download: Chlorophyll-a — CMEMS GlobColour L4
==============================================
Run this script directly:
    python3 scripts/download/dl_chl.py

What gets downloaded:
    * CHL — Chlorophyll-a concentration (mg/m^3)
    * Output -> data/raw/chl/
"""

import os
import sys
import subprocess
from pathlib import Path

DATASET_ID = "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D"
VARIABLES = ["CHL"]
OUTPUT_DIR = Path("data/raw/chl")

LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0, 25.0
START_DATE = "2017-01-01"
END_DATE = "2023-12-31"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("OceanEmbed - Chlorophyll-a Download (Dataset 7)")
print(f"Dataset  : {DATASET_ID}")
print(f"Variable : CHL")
print(f"Region   : lon {LON_MIN} to {LON_MAX}, lat {LAT_MIN} to {LAT_MAX}")
print(f"Period   : {START_DATE} to {END_DATE}")
print(f"Output   : {OUTPUT_DIR}/")
print("=" * 60)

cmd = [
    "copernicusmarine", "subset",
    "--dataset-id", DATASET_ID,
    "--variable", "CHL",
    "--minimum-longitude", str(LON_MIN),
    "--maximum-longitude", str(LON_MAX),
    "--minimum-latitude", str(LAT_MIN),
    "--maximum-latitude", str(LAT_MAX),
    "--start-datetime", f"{START_DATE}T00:00:00",
    "--end-datetime", f"{END_DATE}T23:59:59",
    "--output-directory", str(OUTPUT_DIR),
    "--output-filename", "chl_2017_2023.nc",
    "--overwrite",
]

result = subprocess.run(cmd)
if result.returncode == 0:
    print(f"\n[OK] Chlorophyll-a download complete -> {OUTPUT_DIR.resolve()}")
else:
    print("\n[ERROR] Download failed. Check your Copernicus Marine credentials.")
    sys.exit(1)
