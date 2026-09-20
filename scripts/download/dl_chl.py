#!/usr/bin/env python3
"""
Download: Chlorophyll-a — CMEMS GlobColour L4
==============================================
Run this script directly:
    python3 scripts/download/dl_chl.py

What gets downloaded:
    • CHL  — Chlorophyll-a concentration (mg/m³)
    • Output → data/raw/chl/

Before first run:
    Run once in terminal: copernicusmarine login
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import copernicusmarine  # noqa: F401
except ImportError:
    print("📦  Installing copernicusmarine …")
    pip_install("copernicusmarine")

from pathlib import Path

DATASET_ID  = "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D"
VARIABLES   = ["CHL"]
OUTPUT_DIR  = "data/raw/chl"

LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
START_DATE       = "2017-01-01"
END_DATE         = "2023-12-31"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — Chlorophyll-a Download")
print(f"   Dataset  : {DATASET_ID}")
print(f"   Variables: {', '.join(VARIABLES)}")
print(f"   Region   : lon {LON_MIN}–{LON_MAX}  lat {LAT_MIN}–{LAT_MAX}")
print(f"   Period   : {START_DATE} → {END_DATE}")
print(f"   Output   : {OUTPUT_DIR}/")
print("="*60)

cmd = [
    "copernicusmarine", "subset",
    "--dataset-id",        DATASET_ID,
    "--minimum-longitude", str(LON_MIN),
    "--maximum-longitude", str(LON_MAX),
    "--minimum-latitude",  str(LAT_MIN),
    "--maximum-latitude",  str(LAT_MAX),
    "--start-datetime",    f"{START_DATE}T00:00:00",
    "--end-datetime",      f"{END_DATE}T23:59:59",
    "--output-directory",  OUTPUT_DIR,
    "--force-download",
    "--skip-existing",
]
for v in VARIABLES:
    cmd += ["--variable", v]

result = subprocess.run(cmd)
if result.returncode == 0:
    print("\n✅  Chlorophyll-a download complete!")
    print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
else:
    print("\n❌  Download failed. Check your copernicusmarine login.")
    sys.exit(1)
