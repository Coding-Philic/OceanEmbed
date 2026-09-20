#!/usr/bin/env python3
"""
Download: Heat Flux (SWR / LWR / LHF / SHF) — ERA5 via CDS API
================================================================
Run this script directly:
    python3 scripts/download/dl_heatflux.py

What gets downloaded:
    • surface_net_solar_radiation      (SWR, W/m²)
    • surface_net_thermal_radiation    (LWR, W/m²)
    • surface_latent_heat_flux         (LHF, W/m²)
    • surface_sensible_heat_flux       (SHF, W/m²)
    • Output → data/raw/heatflux/

Setup (one-time):
    pip install cdsapi
    Create ~/.cdsapirc with your key from:
    https://cds.climate.copernicus.eu/api-how-to
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import cdsapi  # noqa: F401
except ImportError:
    print("📦  Installing cdsapi …")
    pip_install("cdsapi")

import cdsapi
from pathlib import Path

OUTPUT_DIR  = "data/raw/heatflux"
LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
START_YEAR  = 2017
END_YEAR    = 2023

VARIABLES = [
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
]

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*60)
print("🌊  OceanEmbed v2 — Heat Flux Download (ERA5 / CDS API)")
print(f"   Variables: {', '.join(VARIABLES)}")
print(f"   Region   : lon {LON_MIN}–{LON_MAX}  lat {LAT_MIN}–{LAT_MAX}")
print(f"   Period   : {START_YEAR} → {END_YEAR}")
print(f"   Output   : {OUTPUT_DIR}/")
print("="*60)

c = cdsapi.Client()

# Download year by year to keep file sizes manageable
for year in range(START_YEAR, END_YEAR + 1):
    out_file = Path(OUTPUT_DIR) / f"heatflux_{year}.nc"
    if out_file.exists():
        print(f"  ⏭  Skipping {year} (already exists)")
        continue

    print(f"\n  → Downloading {year} …")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable":     VARIABLES,
            "year":         str(year),
            "month":        [f"{m:02d}" for m in range(1, 13)],
            "day":          [f"{d:02d}" for d in range(1, 32)],
            "time":         "12:00",          # daily noon snapshot
            "area":         [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # N/W/S/E
            "format":       "netcdf",
        },
        str(out_file),
    )
    print(f"  ✅  {year} done → {out_file}")

print("\n✅  Heat Flux download complete!")
print(f"📁  Files saved to: {Path(OUTPUT_DIR).resolve()}")
