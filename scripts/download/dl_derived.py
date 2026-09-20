#!/usr/bin/env python3
"""
Compute: Derived Channels — Distance from Coast, T-1/T-7 SST, Monsoon Phase
=============================================================================
Run this script AFTER all downloads are complete:
    python3 scripts/download/dl_derived.py

What this computes (no download needed — uses already-downloaded data):

    CH 20: Distance from Coast (km)
           Computed from GEBCO bathymetry + scipy distance transform
           Output → data/raw/coast_distance/coast_distance_bob.nc

    CH 22: T-1 day SST (Previous day SST)
           Computed as a 1-day lag of the SST field
           Output → data/raw/sst_lag/sst_t1_2017_2023.nc

    CH 23: T-7 day SST (Last week SST)
           Computed as a 7-day lag of the SST field
           Output → data/raw/sst_lag/sst_t7_2017_2023.nc

    CH 25: Monsoon Phase Indicator
           0 = Pre-monsoon (Jan–May)
           1 = Active monsoon (Jun–Sep)
           2 = Withdrawal/Post-monsoon (Oct–Dec)
           Computed from calendar date — no data needed
           Output → data/raw/monsoon/monsoon_phase_2017_2023.nc

Requires:
    pip install xarray numpy scipy pandas netcdf4
"""

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["xarray", "numpy", "scipy", "pandas", "netcdf4"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"📦  Installing {pkg} …")
        pip_install(pkg)

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import distance_transform_edt

print("="*60)
print("🌊  OceanEmbed v2 — Compute Derived Channels")
print("="*60)
print()

LON_MIN, LON_MAX = 75.0, 100.0
LAT_MIN, LAT_MAX = 5.0,  25.0
START_DATE = "2017-01-01"
END_DATE   = "2023-12-31"
GRID_RES   = 0.25  # degrees

# Create common output grid
lons = np.arange(LON_MIN, LON_MAX + GRID_RES, GRID_RES)
lats = np.arange(LAT_MIN, LAT_MAX + GRID_RES, GRID_RES)
dates = pd.date_range(START_DATE, END_DATE, freq="D")


# ─────────────────────────────────────────────────────────────────────────────
# CH 20: Distance from Coast
# ─────────────────────────────────────────────────────────────────────────────
print("── CH 20: Distance from Coast ──────────────────────────")
COAST_OUT = Path("data/raw/coast_distance")
COAST_OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = COAST_OUT / "coast_distance_bob.nc"

if OUT_FILE.exists():
    print(f"  ✓ Already exists: {OUT_FILE}")
else:
    # Find bathymetry file
    bathy_files = list(Path("data/raw/bathymetry").glob("*.nc")) + \
                  list(Path("data/raw/bathymetry").glob("*.tif"))

    if bathy_files:
        print(f"  → Computing from GEBCO: {bathy_files[0].name}")
        try:
            ds_bathy = xr.open_dataset(bathy_files[0])
            # Try common variable names
            bathy_var = None
            for vname in ["elevation", "z", "depth", "topo", "Band1"]:
                if vname in ds_bathy:
                    bathy_var = vname
                    break

            if bathy_var:
                bathy = ds_bathy[bathy_var].sel(
                    lon=slice(LON_MIN, LON_MAX),
                    lat=slice(LAT_MIN, LAT_MAX)
                ) if "lon" in ds_bathy.dims else ds_bathy[bathy_var].sel(
                    longitude=slice(LON_MIN, LON_MAX),
                    latitude=slice(LAT_MIN, LAT_MAX)
                )

                # Ocean mask: True where ocean (elevation ≤ 0)
                ocean_mask = (bathy.values <= 0)
                land_mask  = ~ocean_mask

                # Distance transform: distance in pixels from nearest land
                dist_pixels = distance_transform_edt(ocean_mask)
                # Convert pixels → km (approx 27.75 km per 0.25° at equator)
                pixel_km = GRID_RES * 111.0
                dist_km  = dist_pixels * pixel_km

                # Mask land cells with NaN
                dist_km[land_mask] = np.nan

                # Regrid to common BoB 0.25° grid
                da = xr.DataArray(
                    dist_km,
                    dims=bathy.dims,
                    coords=bathy.coords,
                    name="coast_distance",
                    attrs={"units": "km", "long_name": "Distance from nearest coastline"},
                )
                da.to_netcdf(OUT_FILE)
                print(f"  ✅ Coast distance saved → {OUT_FILE}")
            else:
                print("  ⚠️  Could not identify bathymetry variable.")
                _make_placeholder_coast(lons, lats, OUT_FILE)
        except Exception as e:
            print(f"  ⚠️  GEBCO computation failed: {e}")
            _make_placeholder_coast(lons, lats, OUT_FILE)
    else:
        print("  ⚠️  No GEBCO file found. Run dl_bathymetry.py first.")
        print("      Generating placeholder (all NaN) …")
        _make_placeholder_coast(lons, lats, OUT_FILE)

def _make_placeholder_coast(lons, lats, out_file):
    da = xr.DataArray(
        np.full((len(lats), len(lons)), np.nan),
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        name="coast_distance",
        attrs={"units": "km", "long_name": "Distance from nearest coastline (placeholder)"},
    )
    da.to_netcdf(out_file)
    print(f"  ℹ️  Placeholder saved → {out_file}")
    print("      Re-run after dl_bathymetry.py to get real values.")


# ─────────────────────────────────────────────────────────────────────────────
# CH 22 + 23: T-1 and T-7 day SST lags
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── CH 22+23: T-1 / T-7 Day SST Lags ────────────────────")
SST_LAG_OUT = Path("data/raw/sst_lag")
SST_LAG_OUT.mkdir(parents=True, exist_ok=True)

sst_files = sorted(Path("data/raw/sst").glob("*.nc"))

if not sst_files:
    print("  ⚠️  No SST files found. Run dl_sst.py first.")
else:
    print(f"  → Loading SST from {len(sst_files)} file(s) …")
    try:
        ds_sst = xr.open_mfdataset(sst_files, combine="by_coords")
        # Try common SST variable names
        sst_var = None
        for vname in ["analysed_sst", "sst", "SST", "temperature"]:
            if vname in ds_sst:
                sst_var = vname
                break

        if sst_var:
            sst = ds_sst[sst_var].sel(
                time=slice(START_DATE, END_DATE)
            )

            # T-1 lag
            t1_out = SST_LAG_OUT / "sst_t1_2017_2023.nc"
            if t1_out.exists():
                print(f"  ⏭  T-1 lag already exists: {t1_out.name}")
            else:
                print("  → Computing T-1 SST lag …")
                sst_t1 = sst.shift(time=1)
                sst_t1.name = "sst_t1"
                sst_t1.attrs["long_name"] = "SST T-1 day lag (yesterday SST)"
                sst_t1.to_netcdf(t1_out)
                print(f"  ✅ T-1 SST saved → {t1_out}")

            # T-7 lag
            t7_out = SST_LAG_OUT / "sst_t7_2017_2023.nc"
            if t7_out.exists():
                print(f"  ⏭  T-7 lag already exists: {t7_out.name}")
            else:
                print("  → Computing T-7 SST lag …")
                sst_t7 = sst.shift(time=7)
                sst_t7.name = "sst_t7"
                sst_t7.attrs["long_name"] = "SST T-7 day lag (last week SST)"
                sst_t7.to_netcdf(t7_out)
                print(f"  ✅ T-7 SST saved → {t7_out}")
        else:
            print(f"  ⚠️  Could not find SST variable in {sst_files[0]}")
            print(f"      Variables found: {list(ds_sst.data_vars)}")

    except Exception as e:
        print(f"  ❌  SST lag computation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CH 25: Monsoon Phase Indicator
# ─────────────────────────────────────────────────────────────────────────────
print()
print("── CH 25: Monsoon Phase Indicator ───────────────────────")
MONSOON_OUT = Path("data/raw/monsoon")
MONSOON_OUT.mkdir(parents=True, exist_ok=True)
MONSOON_FILE = MONSOON_OUT / "monsoon_phase_2017_2023.nc"

if MONSOON_FILE.exists():
    print(f"  ✓ Already exists: {MONSOON_FILE}")
else:
    print("  → Computing monsoon phase from calendar dates …")

    def month_to_phase(month: int) -> int:
        """
        Monsoon phase for Bay of Bengal:
          0 = Pre-monsoon        (Jan–May)
          1 = Active monsoon     (Jun–Sep)
          2 = Post-monsoon       (Oct–Dec)
        """
        if month <= 5:
            return 0
        elif month <= 9:
            return 1
        else:
            return 2

    phases = np.array([month_to_phase(d.month) for d in dates], dtype=np.int8)

    da_phase = xr.DataArray(
        phases,
        dims=["time"],
        coords={"time": dates},
        name="monsoon_phase",
        attrs={
            "long_name":   "Monsoon phase indicator for Bay of Bengal",
            "encoding":    "0=Pre-monsoon(Jan-May), 1=Active(Jun-Sep), 2=Post-monsoon(Oct-Dec)",
            "units":       "categorical",
        },
    )
    da_phase.to_netcdf(MONSOON_FILE)
    print(f"  ✅ Monsoon phase saved → {MONSOON_FILE}")
    print(f"     Records: {len(dates)} days  |  "
          f"Pre-monsoon: {(phases==0).sum()}  "
          f"Active: {(phases==1).sum()}  "
          f"Post-monsoon: {(phases==2).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("="*60)
print("✅  All derived channels computed!")
print()
print("  Generated files:")
print(f"    data/raw/coast_distance/   ← CH 20: Distance from Coast")
print(f"    data/raw/sst_lag/          ← CH 22: T-1 SST, CH 23: T-7 SST")
print(f"    data/raw/monsoon/          ← CH 25: Monsoon Phase")
print()
print("  Next: run dl_ohc.py for Ocean Heat Content (CH 15)")
print("        run dl_iod.py for IOD index (CH 24)")
