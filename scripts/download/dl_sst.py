#!/usr/bin/env python3
"""
Download: Sea Surface Temperature (SST) — CMEMS OSTIA L4
=========================================================
Run this script directly:
    python3 scripts/download/dl_sst.py

What gets downloaded:
    • analysed_sst  (°C, daily, 0.05°)
    • Output → data/raw/sst/

Before first run:
    Run once in terminal: copernicusmarine login
"""

import subprocess, sys

# ── Auto-install dependency ───────────────────────────────────────────────────
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import copernicusmarine  # noqa: F401
except ImportError:
    print("📦  Installing copernicusmarine …")
    pip_install("copernicusmarine")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_ID  = "cmems_obs-sst_glo_phy_my_l4_P1D-m"
VARIABLE    = "analysed_sst"
OUTPUT_DIR  = "data/raw/sst"

LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
START_DATE       = "2017-01-01"
END_DATE         = "2023-12-31"

# ── Download ──────────────────────────────────────────────────────────────────
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — SST Download")
print(f"   Dataset  : {DATASET_ID}")
print(f"   Variable : {VARIABLE}")
print(f"   Region   : lon {LON_MIN}–{LON_MAX}  lat {LAT_MIN}–{LAT_MAX}")
print(f"   Period   : {START_DATE} → {END_DATE}")
print(f"   Output   : {OUTPUT_DIR}/")
print("="*60)

cmd = [
    "copernicusmarine", "subset",
    "--dataset-id",        DATASET_ID,
    "--variable",          VARIABLE,
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

result = subprocess.run(cmd)
if result.returncode == 0:
    print("\n✅  SST download complete!")
    print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
else:
    print("\n❌  Download failed. Check your copernicusmarine login.")
    print("    Run: copernicusmarine login")
    sys.exit(1)
