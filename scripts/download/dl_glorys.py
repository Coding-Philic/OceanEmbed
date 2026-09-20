#!/usr/bin/env python3
"""
Download: GLORYS12 — 3-D Temperature, Salinity, MLD, SSH
=========================================================
Run this script directly:
    python3 scripts/download/dl_glorys.py

What gets downloaded:
    • thetao  — Ocean potential temperature at depth (°C)
    • so      — Ocean salinity at depth (PSU)
    • mlotst  — Mixed Layer Depth (m)
    • zos     — Sea Surface Height (m)
    • Depths  : 0 m → 1500 m
    • Output  → data/raw/glorys/

⚠️  WARNING: This is a LARGE download (~100s of GB for 11 years).
             Consider running year by year and check disk space first.

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

DATASET_ID  = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
VARIABLES   = ["thetao", "so", "mlotst", "zos"]
OUTPUT_DIR  = "data/raw/glorys"

LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
DEPTH_MIN        = 0.49
DEPTH_MAX        = 1500.0
START_DATE       = "2017-01-01"
END_DATE         = "2023-12-31"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — GLORYS12 3-D Download")
print(f"   Dataset  : {DATASET_ID}")
print(f"   Variables: {', '.join(VARIABLES)}")
print(f"   Region   : lon {LON_MIN}–{LON_MAX}  lat {LAT_MIN}–{LAT_MAX}")
print(f"   Depths   : {DEPTH_MIN} m → {DEPTH_MAX} m")
print(f"   Period   : {START_DATE} → {END_DATE}")
print(f"   Output   : {OUTPUT_DIR}/")
print("="*60)
print()
print("⚠️  This is a large 3-D dataset. Download may take several hours.")
print("   Press Ctrl+C to cancel.")
print()

cmd = [
    "copernicusmarine", "subset",
    "--dataset-id",        DATASET_ID,
    "--minimum-longitude", str(LON_MIN),
    "--maximum-longitude", str(LON_MAX),
    "--minimum-latitude",  str(LAT_MIN),
    "--maximum-latitude",  str(LAT_MAX),
    "--minimum-depth",     str(DEPTH_MIN),
    "--maximum-depth",     str(DEPTH_MAX),
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
    print("\n✅  GLORYS12 download complete!")
    print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
else:
    print("\n❌  Download failed. Check your copernicusmarine login.")
    sys.exit(1)
