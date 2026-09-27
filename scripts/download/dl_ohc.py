#!/usr/bin/env python3
"""
Compute: Ocean Heat Content 0-300m (OHC) — CMEMS GLORYS12
===========================================================
Run this script directly:
    python3 scripts/download/dl_ohc.py

What gets computed:
    * Ocean Heat Content integrated 0–300 m (J/m^2)
    * Computed from GLORYS12 potential temperature (thetao)
    * Formula: OHC = rho * Cp * sum(thetao * dz) over 0-300m
      where rho = 1025.0 kg/m^3, Cp = 3990.0 J/(kg*K)
    * Also computes Tropical Cyclone Heat Potential (TCHP relative to 26 deg C)
    * Output -> data/raw/ohc/ohc_<year>.nc
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("data/raw/ohc")
GLORYS_DIR = Path("data/raw/glorys")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants
RHO_SW = 1025.0       # kg/m^3 — sea water density
CP_SW = 3990.0        # J/(kg*K) — specific heat capacity of seawater
OHC_DEPTH_MAX = 300.0 # meters — integration limit
T_REF = 26.0          # deg C — reference temperature for tropical cyclone heat potential


def compute_ohc():
    try:
        import xarray as xr
    except ImportError:
        print("[ERROR] xarray is required. Run: pip install xarray")
        sys.exit(1)

    print("=" * 60)
    print("OceanEmbed - Ocean Heat Content (0-300m) Computation")
    print(f"Source Directory : {GLORYS_DIR}/")
    print(f"Output Directory : {OUTPUT_DIR}/")
    print(f"Depth Limit      : 0 to {OHC_DEPTH_MAX} m")
    print("=" * 60)

    glorys_files = sorted(GLORYS_DIR.glob("*.nc"))
    if not glorys_files:
        print(f"[ERROR] No GLORYS NetCDF files found in {GLORYS_DIR}/")
        print("Please ensure GLORYS files are present before computing OHC.")
        sys.exit(1)

    print(f"\nFound {len(glorys_files)} GLORYS file(s) to process.\n")

    for f in glorys_files:
        print(f"--- Processing {f.name} ---")
        try:
            ds = xr.open_dataset(f, chunks={"time": 100})
        except Exception:
            ds = xr.open_dataset(f)

        if "thetao" not in ds:
            print(f"[WARN] 'thetao' variable not found in {f.name}, skipping.")
            ds.close()
            continue

        # Extract depths down to 300m
        t300 = ds["thetao"].sel(depth=slice(0, OHC_DEPTH_MAX))
        depths = t300.depth.values
        dz = np.gradient(depths)
        dz_da = xr.DataArray(dz, dims=["depth"], coords={"depth": t300.depth})

        # Total Ocean Heat Content (0 to 300m) in J/m^2
        ohc_total = (RHO_SW * CP_SW * (t300 * dz_da)).sum(dim="depth")
        ohc_total.name = "ohc_300m"
        ohc_total.attrs["long_name"] = "Total Ocean Heat Content (0-300m)"
        ohc_total.attrs["units"] = "J/m^2"

        # Tropical Cyclone Heat Potential (TCHP relative to 26 deg C isotherm)
        t_excess = (t300 - T_REF).clip(min=0)
        tchp = (RHO_SW * CP_SW * (t_excess * dz_da)).sum(dim="depth")
        tchp.name = "tchp_26c"
        tchp.attrs["long_name"] = "Tropical Cyclone Heat Potential (T > 26 deg C)"
        tchp.attrs["units"] = "kJ/cm^2"

        # Create output dataset
        ds_out = xr.Dataset(
            data_vars={
                "ohc_300m": ohc_total.astype(np.float32),
                "tchp_26c": (tchp / 1e7).astype(np.float32),  # converted to standard kJ/cm^2
            },
            coords=ohc_total.coords,
            attrs={
                "title": "OceanEmbed Computed Ocean Heat Content (0-300m)",
                "source": f"Computed from {f.name}",
                "density": f"{RHO_SW} kg/m^3",
                "specific_heat": f"{CP_SW} J/(kg*K)",
            },
        )

        # Output filename
        if "time" in ds and len(ds.time) > 0:
            try:
                first_year = str(ds.time.dt.year.values[0])
                last_year = str(ds.time.dt.year.values[-1])
                year_label = first_year if first_year == last_year else f"{first_year}_{last_year}"
            except Exception:
                year_label = f.stem
        else:
            year_label = f.stem

        out_file = OUTPUT_DIR / f"ohc_{year_label}.nc"
        ds_out.to_netcdf(out_file)
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Saved OHC dataset: {out_file} ({size_mb:.2f} MB)")
        ds.close()

    print("\n" + "=" * 60)
    print("All OHC computations finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    compute_ohc()
