#!/usr/bin/env python3
"""
Download: Ocean Heat Content 0-300m (OHC) — CMEMS GLORYS12
============================================================
Run this script directly:
    python3 scripts/download/dl_ohc.py

What gets downloaded:
    • Ocean Heat Content integrated 0–300 m  (J/m²)
    • Computed on-the-fly from GLORYS12 thetao (sea water temperature)
    • OR: downloaded directly from CMEMS if a dedicated OHC product exists
    • Output → data/raw/ohc/

Strategy:
    GLORYS12 thetao (already in data/raw/glorys/) contains temperature at
    every depth level. OHC is computed by integrating:
        OHC = ρ × Cp × ∫₀³⁰⁰ T(z) dz
    where ρ=1025 kg/m³, Cp=3985 J/(kg·K)

    This script:
    1. First tries to download the CMEMS dedicated OHC product directly
    2. Falls back to computing OHC from already-downloaded GLORYS thetao

Before first run:
    Run once in terminal: copernicusmarine login
    Also requires: pip install xarray scipy numpy
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import copernicusmarine  # noqa: F401
except ImportError:
    print("📦  Installing copernicusmarine …")
    pip_install("copernicusmarine")

for pkg in ["xarray", "numpy", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦  Installing {pkg} …")
        pip_install(pkg)

from pathlib import Path
import numpy as np

OUTPUT_DIR  = "data/raw/ohc"
GLORYS_DIR  = "data/raw/glorys"

LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
START_DATE       = "2017-01-01"
END_DATE         = "2023-12-31"

# Physical constants
RHO_SW = 1025.0     # kg/m³ — sea water density
CP_SW  = 3985.0     # J/(kg·K) — specific heat capacity of sea water
OHC_DEPTH_MAX = 300 # m — integrate to 300 m

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — Ocean Heat Content (OHC) Download")
print(f"   Depth    : 0 → {OHC_DEPTH_MAX} m (integrated)")
print(f"   Region   : lon {LON_MIN}–{LON_MAX}  lat {LAT_MIN}–{LAT_MAX}")
print(f"   Period   : {START_DATE} → {END_DATE}")
print(f"   Output   : {OUTPUT_DIR}/")
print("="*60)
print()

# ── Strategy 1: Try CMEMS direct OHC product ─────────────────────────────────
print("  → Attempting CMEMS direct OHC download …")
CMEMS_OHC_ID = "cmems_mod_glo_phy_myint_0.25deg_P1M-m"   # Monthly mean physics

try:
    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id",        CMEMS_OHC_ID,
        "--variable",          "ohcvthermcline300",   # OHC 0-300m variable
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print(f"  ✅ CMEMS OHC download complete → {OUTPUT_DIR}/")
        sys.exit(0)
    else:
        print(f"  ⚠️  CMEMS direct OHC failed (dataset may not exist). Computing from GLORYS …")
except Exception as e:
    print(f"  ⚠️  CMEMS attempt failed: {e}. Computing from GLORYS …")

# ── Strategy 2: Compute OHC from GLORYS thetao ───────────────────────────────
print()
print("  → Computing OHC from GLORYS12 thetao …")
print(f"    Looking for GLORYS files in: {GLORYS_DIR}/")

glorys_files = sorted(Path(GLORYS_DIR).glob("*.nc"))
if not glorys_files:
    print(f"  ❌  No GLORYS files found in {GLORYS_DIR}/")
    print(f"      Run first: python3 scripts/download/dl_glorys.py")
    sys.exit(1)

try:
    import xarray as xr

    print(f"  Found {len(glorys_files)} GLORYS file(s). Processing …")

    for f in glorys_files:
        out_file = Path(OUTPUT_DIR) / f"ohc_300m_{f.stem}.nc"
        if out_file.exists():
            print(f"  ⏭  Skipping {f.name} (already computed)")
            continue

        print(f"  → Computing OHC from {f.name} …")
        ds = xr.open_dataset(f)

        if "thetao" not in ds:
            print(f"    ⚠️  No thetao in {f.name}, skipping.")
            continue

        # Select depths 0 → 300 m
        ds_300 = ds["thetao"].sel(depth=slice(0, OHC_DEPTH_MAX))

        # OHC = ρ × Cp × ∫T dz  (trapezoid integration over depth)
        depths = ds_300.depth.values
        dz = np.gradient(depths)                           # layer thicknesses

        # Integrate: sum(T * dz) * RHO * CP
        ohc = (ds_300 * xr.DataArray(dz, dims="depth")).sum(dim="depth") * RHO_SW * CP_SW

        ohc = ohc.rename("ohc_300m")
        ohc.attrs["long_name"]  = f"Ocean Heat Content 0-{OHC_DEPTH_MAX}m"
        ohc.attrs["units"]      = "J/m2"
        ohc.attrs["formula"]    = f"rho={RHO_SW} kg/m3, Cp={CP_SW} J/(kg.K), integrated 0-{OHC_DEPTH_MAX}m"

        ohc.to_netcdf(out_file)
        print(f"  ✅  {f.name} → {out_file.name}")

    print()
    print("✅  OHC computation complete!")
    print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")

except ImportError:
    print("  ❌  xarray not installed. Run: pip install xarray")
    sys.exit(1)
except Exception as e:
    print(f"  ❌  OHC computation failed: {e}")
    sys.exit(1)
