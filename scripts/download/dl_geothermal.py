#!/usr/bin/env python3
"""
Download: Geothermal Heat Flux — IHFC Global Database (Static)
==============================================================
Run this script directly:
    python3 scripts/download/dl_geothermal.py

What gets downloaded:
    * Global Heat Flow Database from IHFC (International Heat Flow Commission)
      hosted by GFZ Data Services (Helmholtz Centre Potsdam)
    * Output -> data/raw/geothermal/IHFC_2023_GHFDB.CSV
    * Automatically generates a gridded NetCDF file for ocean modeling
      (data/raw/geothermal/geothermal_heatflow_grid.nc)
"""

import os
import sys
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("data/raw/geothermal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IHFC_URL = (
    "https://datapub.gfz.de/download/10.5880.FIDGEO.2023.008-VENOun/IHFC_2023_GHFDB.CSV"
)
CSV_FILE = OUTPUT_DIR / "IHFC_2023_GHFDB.CSV"
NC_FILE = OUTPUT_DIR / "geothermal_heatflow_grid.nc"


def download_csv():
    print("=" * 60)
    print("OceanEmbed - Geothermal Heat Flux Download")
    print(f"Source : IHFC Global Heat Flow Database (GFZ Potsdam)")
    print(f"Target : {CSV_FILE}")
    print("=" * 60)

    if CSV_FILE.exists() and CSV_FILE.stat().st_size > 10_000_000:
        print(f"[OK] IHFC CSV already exists: {CSV_FILE} ({CSV_FILE.stat().st_size / 1e6:.1f} MB)")
        return True

    print("Downloading IHFC CSV from GFZ Potsdam...")
    # Try requests first
    try:
        import requests
        with requests.get(IHFC_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(CSV_FILE, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        print(f"[OK] Download complete -> {CSV_FILE} ({CSV_FILE.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"[WARN] Requests download failed: {e}. Trying curl...")

    # Fallback to curl
    try:
        cmd = ["curl", "-L", "-o", str(CSV_FILE), IHFC_URL]
        subprocess.run(cmd, check=True)
        print(f"[OK] Curl download complete -> {CSV_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] Curl download failed: {e}")
        return False


def generate_gridded_netcdf():
    """Converts the irregular IHFC heat flow observations into a regular 0.25 deg NetCDF grid."""
    try:
        import pandas as pd
        import numpy as np
        import xarray as xr
    except ImportError:
        print("[INFO] pandas/numpy/xarray not installed. Skipping NetCDF grid generation.")
        print("      Raw CSV is available at: " + str(CSV_FILE))
        return

    print("Interpolating IHFC heat flow to Bay of Bengal & Indian Ocean grid (0.25 deg)...")
    try:
        df = pd.read_csv(CSV_FILE, sep=";", encoding="latin1", low_memory=False)
        # Columns: q (heat flow in mW/m2), lat, lng
        df["q"] = pd.to_numeric(df["q"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
        clean = df.dropna(subset=["q", "lat", "lng"])

        # Target grid: 5N to 25N, 75E to 100E
        lats = np.arange(5.0, 25.25, 0.25)
        lons = np.arange(75.0, 100.25, 0.25)
        grid_lon, grid_lat = np.meshgrid(lons, lats)

        from scipy.interpolate import griddata
        points = clean[["lng", "lat"]].values
        values = clean["q"].values

        # Grid using linear then nearest to fill NaN
        grid_q = griddata(points, values, (grid_lon, grid_lat), method="linear")
        nan_mask = np.isnan(grid_q)
        if np.any(nan_mask):
            grid_q_near = griddata(points, values, (grid_lon, grid_lat), method="nearest")
            grid_q[nan_mask] = grid_q_near[nan_mask]

        ds = xr.Dataset(
            data_vars={"heat_flow": (["latitude", "longitude"], grid_q.astype(np.float32))},
            coords={"latitude": lats, "longitude": lons},
            attrs={
                "title": "OceanEmbed Geothermal Heat Flow (IHFC 2023)",
                "source": "International Heat Flow Commission (GFZ Potsdam DOI 10.5880/fidgeo.2023.008)",
                "units": "mW/m^2",
            },
        )
        ds["heat_flow"].attrs["units"] = "mW/m^2"
        ds["heat_flow"].attrs["long_name"] = "Surface Heat Flow"
        ds.to_netcdf(NC_FILE)
        print(f"[OK] Gridded NetCDF created -> {NC_FILE}")
    except Exception as e:
        print(f"[WARN] NetCDF grid generation encountered an error: {e}")


if __name__ == "__main__":
    success = download_csv()
    if success:
        generate_gridded_netcdf()
        print("\nDataset 18 (Geothermal Heat Flow) ready.")
    else:
        sys.exit(1)
