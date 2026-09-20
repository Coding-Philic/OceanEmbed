#!/usr/bin/env python3
"""
Download: Bathymetry — GEBCO 2023 (Static)
===========================================
Run this script directly:
    python3 scripts/download/dl_bathymetry.py

What gets downloaded:
    • GEBCO 2023 global bathymetry grid
    • 15 arc-second resolution (~7 GB zip file)
    • Output → data/raw/bathymetry/gebco_2023.zip

After downloading, unzip with:
    unzip data/raw/bathymetry/gebco_2023.zip -d data/raw/bathymetry/

No login required — GEBCO is fully open access.
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

OUTPUT_DIR = "data/raw/bathymetry"
URL        = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023/zip/"
OUT_FILE   = Path(OUTPUT_DIR) / "gebco_2023.zip"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — GEBCO 2023 Bathymetry Download")
print(f"   Source  : {URL}")
print(f"   Output  : {OUT_FILE}")
print(f"   Size    : ~7 GB")
print("="*60)
print()

if OUT_FILE.exists():
    size_gb = OUT_FILE.stat().st_size / 1e9
    print(f"  ✓ File already exists ({size_gb:.1f} GB): {OUT_FILE}")
    print("  Delete it and re-run to re-download.")
    sys.exit(0)

print("  → Starting download …")
print("  ⚠️  This may take 30+ minutes on a slow connection.")
print()

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
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
except requests.RequestException as e:
    print(f"\n❌  Download failed: {e}")
    print("    Try downloading manually from: https://www.gebco.net/data_and_products/gridded_bathymetry_data/")
    sys.exit(1)

print(f"\n✅  GEBCO 2023 download complete!")
print(f"📁  Saved to: {OUT_FILE.resolve()}")
print()
print("  Next step — unzip the file:")
print(f"    unzip {OUT_FILE} -d {OUTPUT_DIR}/")
