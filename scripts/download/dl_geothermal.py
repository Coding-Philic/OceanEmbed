#!/usr/bin/env python3
"""
Download: Geothermal Heat Flux — IHFC Global Database (Static)
==============================================================
Run this script directly:
    python3 scripts/download/dl_geothermal.py

What gets downloaded:
    • Global Heat Flow Database from IHFC (International Heat Flow Commission)
    • 1° resolution global grid
    • Output → data/raw/geothermal/

⚠️  Note: IHFC may require free registration.
    If auto-download fails, get it manually from:
    https://ihfc-iugg.org/products/global-heat-flow-database/

No credentials needed for the main CSV download.
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["requests", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦  Installing {pkg} …")
        pip_install(pkg)

import requests
from tqdm import tqdm
from pathlib import Path

OUTPUT_DIR = "data/raw/geothermal"
# Primary: IHFC download page
PRIMARY_URL = "https://ihfc-iugg.org/products/global-heat-flow-database/download/"
# Fallback: Shapiro & Ritzwoller 2004 interpolated grid (widely used)
FALLBACK_URL = (
    "https://ds.iris.edu/files/products/emc/emc-files/"
    "Shapiro.Ritzwoller-2004-GlobalHeatFlow-0.5x0.5deg.nc"
)
OUT_FILE_PRIMARY  = Path(OUTPUT_DIR) / "global_heatflow_ihfc.csv"
OUT_FILE_FALLBACK = Path(OUTPUT_DIR) / "shapiro_ritzwoller_heatflow.nc"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — Geothermal Heat Flux Download")
print(f"   Primary source : IHFC Global Database")
print(f"   Fallback source: Shapiro & Ritzwoller 2004 (IRIS)")
print(f"   Output         : {OUTPUT_DIR}/")
print("="*60)
print()

def download_file(url, out_file, label):
    print(f"  → Downloading {label} …")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_file, "wb") as f, tqdm(
                desc=label,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ❌  Failed: {e}")
        return False

# Try primary
if OUT_FILE_PRIMARY.exists():
    print(f"  ✓ IHFC file already exists: {OUT_FILE_PRIMARY}")
elif download_file(PRIMARY_URL, OUT_FILE_PRIMARY, "IHFC Heat Flow"):
    print(f"\n✅  IHFC download complete → {OUT_FILE_PRIMARY.resolve()}")
    sys.exit(0)
else:
    print()
    print("  ⚠️  Primary (IHFC) failed. Trying fallback: Shapiro & Ritzwoller 2004 …")
    print()

# Try fallback
if OUT_FILE_FALLBACK.exists():
    print(f"  ✓ Fallback file already exists: {OUT_FILE_FALLBACK}")
elif download_file(FALLBACK_URL, OUT_FILE_FALLBACK, "Shapiro-Ritzwoller Heatflow"):
    print(f"\n✅  Fallback download complete → {OUT_FILE_FALLBACK.resolve()}")
else:
    print()
    print("❌  Both downloads failed.")
    print("   Manual download instructions:")
    print("   1. Go to: https://ihfc-iugg.org/products/global-heat-flow-database/")
    print("   2. Register (free) and download the dataset")
    print(f"   3. Save file to: {Path(OUTPUT_DIR).resolve()}/")
    sys.exit(1)

print(f"\n📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
