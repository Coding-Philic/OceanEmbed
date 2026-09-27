#!/usr/bin/env python3
"""
Download: Bathymetry — GEBCO 2023 (Static)
===========================================
Run this script directly:
    python3 scripts/download/dl_bathymetry.py

What gets downloaded:
    • GEBCO 2023 global bathymetry grid (NetCDF4)
    • 15 arc-second resolution (~7.46 GB)
    • Output → data/raw/bathymetry/gebco_2023.nc

No login required — GEBCO is fully open access.
"""

import subprocess, sys
from pathlib import Path

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["requests", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        pip_install(pkg)

import requests
from tqdm import tqdm

OUTPUT_DIR = "data/raw/bathymetry"
URL        = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2023/ice_surface_elevation/netcdf/GEBCO_2023_CF.nc"
OUT_FILE   = Path(OUTPUT_DIR) / "gebco_2023.nc"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("OceanEmbed v2 — GEBCO 2023 Bathymetry Download")
print(f"   Source  : {URL}")
print(f"   Output  : {OUT_FILE}")
print(f"   Size    : ~7.46 GB")
print("="*60)
print()

if OUT_FILE.exists():
    size_gb = OUT_FILE.stat().st_size / 1e9
    print(f"File already exists ({size_gb:.2f} GB): {OUT_FILE}")
    sys.exit(0)

print("Starting download...")

try:
    with requests.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(OUT_FILE, "wb") as f, tqdm(
            desc="GEBCO 2023",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1048576):  # 1MB chunks
                f.write(chunk)
                bar.update(len(chunk))
except requests.RequestException as e:
    print(f"\nDownload failed: {e}")
    sys.exit(1)

print(f"\nGEBCO 2023 download complete!")
print(f"Saved to: {OUT_FILE.resolve()}")
