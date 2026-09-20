#!/usr/bin/env python3
"""
Download: Precipitation — NASA GPM IMERG Daily L3
==================================================
Run this script directly:
    python3 scripts/download/dl_precip.py

What gets downloaded:
    • GPM IMERG Final Run Daily (GPM_3IMERGDF)
    • 0.1° × 0.1° resolution, mm/hr
    • Output → data/raw/precip/

Setup (one-time):
    1. Register at: https://urs.earthdata.nasa.gov/
    2. Create ~/.netrc with:
          machine urs.earthdata.nasa.gov
          login YOUR_USERNAME
          password YOUR_PASSWORD
    OR: just run this script and it will prompt for login interactively.
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import earthaccess  # noqa: F401
except ImportError:
    print("📦  Installing earthaccess …")
    pip_install("earthaccess")

import earthaccess
from pathlib import Path

OUTPUT_DIR       = "data/raw/precip"
LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
START_DATE       = "2017-01-01"
END_DATE         = "2023-12-31"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — GPM IMERG Precipitation Download")
print(f"   Dataset : GPM_3IMERGDF (IMERG Final Run Daily, 0.1°)")
print(f"   Region  : lon {LON_MIN}–{LON_MAX}  lat {LAT_MIN}–{LAT_MAX}")
print(f"   Period  : {START_DATE} → {END_DATE}")
print(f"   Output  : {OUTPUT_DIR}/")
print("="*60)
print()

print("  → Logging in to NASA Earthdata …")
earthaccess.login(strategy="netrc")

print("  → Searching for GPM IMERG granules …")
results = earthaccess.search_data(
    short_name   = "GPM_3IMERGDF",
    version      = "07",
    temporal     = (START_DATE, END_DATE),
    bounding_box = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
)
print(f"  Found {len(results)} granules.")

if not results:
    print("  ⚠️  No granules found. Check your date range or Earthdata account.")
    sys.exit(1)

print(f"  → Downloading {len(results)} files …")
earthaccess.download(results, OUTPUT_DIR)

print("\n✅  GPM IMERG Precipitation download complete!")
print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
