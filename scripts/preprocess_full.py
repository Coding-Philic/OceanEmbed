#!/usr/bin/env python3
"""
OceanEmbed Full Preprocessing — Align ALL 25 Datasets to a Common Grid
=======================================================================

Reads raw NetCDF files from data/raw/<subfolder>/ and writes aligned
yearly files to data/aligned/inputs/<channel>_<year>.nc and
data/aligned/targets/glorys_temp_<year>.nc on a common 0.25-degree grid.

Handles:
  - Dynamic satellite channels (time-varying 2D)
  - ERA5 yearly files (multiple variables per file)
  - Static spatial fields (bathymetry, geothermal) replicated across time
  - Monthly climatology (WOA23) interpolated to daily
  - Scalar time series (IOD) broadcast to spatial grid
  - GPM IMERG multi-file daily precipitation

Usage:
    python scripts/preprocess_full.py --config configs/kaggle_25ch.yaml
    python scripts/preprocess_full.py --config configs/kaggle_25ch.yaml --raw-dir /path/to/raw
"""

import os
import glob
import argparse
import traceback

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf


# ============================================================================
#  Channel Registry — maps channel name to raw data source and variable
# ============================================================================

CHANNEL_REGISTRY = {
    # ---- Copernicus Marine satellite obs (single multi-year NetCDF) ----
    "sst": {
        "raw_subdir": "sst",
        "var_names": ["analysed_sst"],
        "type": "dynamic",
    },
    "sla": {
        "raw_subdir": "sla",
        "var_names": ["sla"],
        "type": "dynamic",
    },
    "wind_u": {
        "raw_subdir": "wind",
        "var_names": ["eastward_wind"],
        "type": "dynamic",
    },
    "wind_v": {
        "raw_subdir": "wind",
        "var_names": ["northward_wind"],
        "type": "dynamic",
    },
    "cur_u": {
        "raw_subdir": "currents",
        "var_names": ["uo"],
        "type": "dynamic",
    },
    "cur_v": {
        "raw_subdir": "currents",
        "var_names": ["vo"],
        "type": "dynamic",
    },
    "chl": {
        "raw_subdir": "chl",
        "var_names": ["CHL"],
        "type": "dynamic",
    },
    "kd490": {
        "raw_subdir": "kd490",
        "var_names": ["KD490"],
        "type": "dynamic",
    },
    "sss": {
        "raw_subdir": "sss",
        "var_names": ["sos"],
        "type": "dynamic",
    },
    # ---- ERA5 reanalysis (yearly NetCDF files, CDS short names) ----
    "solar_rad": {
        "raw_subdir": "heatflux",
        "file_pattern": "heatflux_*.nc",
        "var_names": ["ssr", "surface_net_solar_radiation"],
        "type": "era5_yearly",
    },
    "thermal_rad": {
        "raw_subdir": "heatflux",
        "file_pattern": "heatflux_*.nc",
        "var_names": ["str", "surface_net_thermal_radiation"],
        "type": "era5_yearly",
    },
    "latent_heat": {
        "raw_subdir": "heatflux",
        "file_pattern": "heatflux_*.nc",
        "var_names": ["slhf", "surface_latent_heat_flux"],
        "type": "era5_yearly",
    },
    "sensible_heat": {
        "raw_subdir": "heatflux",
        "file_pattern": "heatflux_*.nc",
        "var_names": ["sshf", "surface_sensible_heat_flux"],
        "type": "era5_yearly",
    },
    "slp": {
        "raw_subdir": "slp",
        "var_names": ["msl", "sp", "mean_sea_level_pressure"],
        "type": "era5_yearly",
    },
    # ---- Auxiliary dynamic ----
    "precip": {
        "raw_subdir": "precip",
        "var_names": ["precipitation", "precipitationCal", "HQprecipitation"],
        "type": "dynamic",
    },
    "river": {
        "raw_subdir": "glofas",
        "var_names": ["dis24", "dis", "river_discharge_in_the_last_24_hours"],
        "type": "era5_yearly",
    },
    # ---- Static spatial fields ----
    "bathymetry": {
        "raw_subdir": "bathymetry",
        "var_names": ["elevation", "z"],
        "type": "static",
    },
    "geothermal": {
        "raw_subdir": "geothermal",
        "file_pattern": "*grid*.nc",
        "var_names": ["heat_flow", "heatflow", "z"],
        "type": "static",
    },
    # ---- Climate indices ----
    "iod": {
        "raw_subdir": "iod",
        "type": "scalar_csv",
    },
    # ---- WOA23 monthly climatology ----
    "woa_temp": {
        "raw_subdir": "woa23/temperature",
        "var_names": ["t_an", "t_mn"],
        "type": "climatology",
    },
    "woa_sal": {
        "raw_subdir": "woa23/salinity",
        "var_names": ["s_an", "s_mn"],
        "type": "climatology",
    },
}


# ============================================================================
#  Helper Functions
# ============================================================================

def standardize_coords(ds):
    """Rename coordinate names to standard lon/lat/time."""
    rename_map = {}
    for old, new in [("longitude", "lon"), ("latitude", "lat")]:
        if old in ds.dims or old in ds.coords:
            rename_map[old] = new
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def find_variable(ds, var_names):
    """Find the first matching variable name in the dataset."""
    for vn in var_names:
        if vn in ds.data_vars:
            return vn
    # Fallback: return the first data variable
    available = list(ds.data_vars)
    if available:
        return available[0]
    return None


def clip_and_regrid(da, new_lon, new_lat, lon_range, lat_range):
    """Clip to spatial domain and regrid to target resolution."""
    # Clip with buffer
    da = da.sel(
        lon=slice(lon_range[0] - 1, lon_range[1] + 1),
        lat=slice(lat_range[0] - 1, lat_range[1] + 1),
    )
    # Check if lat is descending (common in ERA5)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    # Interpolate to target grid
    da = da.interp(lon=new_lon, lat=new_lat, method="linear")
    return da


def get_nc_files(raw_dir, subdir, pattern="*.nc"):
    """Get sorted list of NetCDF files in a subdirectory."""
    search_dir = raw_dir / subdir
    if not search_dir.exists():
        return []
    files = sorted(glob.glob(str(search_dir / pattern)))
    # Also try .nc4 extension (GPM files)
    if not files:
        files = sorted(glob.glob(str(search_dir / "*.nc4")))
    # Also try HDF5 extension
    if not files:
        files = sorted(glob.glob(str(search_dir / "*.HDF5")))
    return files


# ============================================================================
#  Channel Processing Functions
# ============================================================================

def process_dynamic_channel(ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                            lon_range, lat_range, years):
    """Process a dynamic (time-varying 2D) channel from NetCDF files."""
    pattern = ch_cfg.get("file_pattern", "*.nc")
    files = get_nc_files(raw_dir, ch_cfg["raw_subdir"], pattern)
    if not files:
        print(f"  [SKIP] No files found in {ch_cfg['raw_subdir']}/")
        return False

    try:
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 100})
        ds = standardize_coords(ds)
        var_name = find_variable(ds, ch_cfg["var_names"])
        if var_name is None:
            print(f"  [SKIP] No matching variable found. Available: {list(ds.data_vars)}")
            ds.close()
            return False

        da = ds[var_name]
        # If 3D (has depth), take surface level
        if "depth" in da.dims:
            da = da.isel(depth=0)

        saved = 0
        for year in years:
            out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
            try:
                da_year = da.sel(time=str(year))
                if len(da_year.time) == 0:
                    continue
                da_year = da_year.load()
                da_regrid = clip_and_regrid(da_year, new_lon, new_lat, lon_range, lat_range)
                da_regrid.to_dataset(name=ch_name).to_netcdf(out_path)
                saved += 1
            except (KeyError, ValueError):
                continue

        ds.close()
        print(f"  [OK] {ch_name}: saved {saved} years (var={var_name})")
        return saved > 0

    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        return False


def process_era5_yearly_channel(ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                                lon_range, lat_range, years):
    """Process ERA5-style yearly NetCDF files (one file per year, CDS short names)."""
    subdir = ch_cfg["raw_subdir"]
    files = get_nc_files(raw_dir, subdir)
    if not files:
        print(f"  [SKIP] No files found in {subdir}/")
        return False

    saved = 0
    for f in files:
        try:
            ds = xr.open_dataset(f)
            ds = standardize_coords(ds)
            var_name = find_variable(ds, ch_cfg["var_names"])
            if var_name is None:
                ds.close()
                continue

            da = ds[var_name]
            if "depth" in da.dims:
                da = da.isel(depth=0)

            for year in years:
                out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
                if out_path.exists():
                    saved += 1
                    continue
                try:
                    da_year = da.sel(time=str(year))
                    if len(da_year.time) == 0:
                        continue
                    da_year = da_year.load()
                    da_regrid = clip_and_regrid(da_year, new_lon, new_lat, lon_range, lat_range)
                    da_regrid.to_dataset(name=ch_name).to_netcdf(out_path)
                    saved += 1
                except (KeyError, ValueError):
                    continue

            ds.close()

        except Exception as e:
            print(f"  [WARNING] {ch_name} file {Path(f).name}: {e}")
            continue

    print(f"  [OK] {ch_name}: saved {saved} year-files (ERA5 yearly)")
    return saved > 0


def process_static_channel(ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                           lon_range, lat_range, years, ref_times):
    """Process static spatial fields (bathymetry, geothermal) by replicating across time."""
    pattern = ch_cfg.get("file_pattern", "*.nc")
    files = get_nc_files(raw_dir, ch_cfg["raw_subdir"], pattern)
    if not files:
        print(f"  [SKIP] No files found in {ch_cfg['raw_subdir']}/")
        return False

    try:
        ds = xr.open_dataset(files[0])
        ds = standardize_coords(ds)
        var_name = find_variable(ds, ch_cfg["var_names"])
        if var_name is None:
            print(f"  [SKIP] {ch_name}: no matching variable. Available: {list(ds.data_vars)}")
            ds.close()
            return False

        da = ds[var_name]
        # Remove time/depth dims if present
        if "time" in da.dims:
            da = da.isel(time=0)
        if "depth" in da.dims:
            da = da.isel(depth=0)

        da = da.load()
        da_regrid = clip_and_regrid(da, new_lon, new_lat, lon_range, lat_range)
        static_values = da_regrid.values.astype(np.float32)  # [H, W]
        ds.close()

        # Replicate static field across time for each year
        saved = 0
        for year in years:
            if year not in ref_times:
                continue
            times = ref_times[year]
            n_days = len(times)
            # Broadcast static field to [time, lat, lon]
            data_3d = np.broadcast_to(static_values[np.newaxis, :, :], (n_days,) + static_values.shape)
            ds_out = xr.Dataset({
                ch_name: (["time", "lat", "lon"], data_3d.astype(np.float32))
            }, coords={"time": times, "lat": new_lat, "lon": new_lon})
            out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
            ds_out.to_netcdf(out_path)
            saved += 1

        print(f"  [OK] {ch_name}: static field replicated for {saved} years")
        return saved > 0

    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        return False


def process_iod_channel(ch_name, raw_dir, out_dir, new_lon, new_lat, years, ref_times):
    """Process IOD index CSV: broadcast monthly scalar to spatial grid."""
    csv_files = sorted(glob.glob(str(raw_dir / "iod" / "*.csv")))
    if not csv_files:
        print(f"  [SKIP] No IOD CSV files found")
        return False

    try:
        df = pd.read_csv(csv_files[0])
        # Try to find date and value columns
        date_col = None
        val_col = None
        for c in df.columns:
            cl = c.lower()
            if "date" in cl or "time" in cl or "year" in cl:
                date_col = c
            if "dmi" in cl or "iod" in cl or "value" in cl or "index" in cl:
                val_col = c

        if date_col is None or val_col is None:
            # Try treating first two columns as date/value
            cols = list(df.columns)
            if len(cols) >= 2:
                date_col, val_col = cols[0], cols[1]
            else:
                print(f"  [SKIP] IOD CSV: cannot identify date/value columns")
                return False

        df["date"] = pd.to_datetime(df[date_col])
        df = df.set_index("date").sort_index()
        iod_series = df[val_col].astype(float)

        H, W = len(new_lat), len(new_lon)
        saved = 0

        for year in years:
            if year not in ref_times:
                continue
            times = ref_times[year]
            n_days = len(times)
            data_3d = np.zeros((n_days, H, W), dtype=np.float32)

            for i, t in enumerate(times):
                t_pd = pd.Timestamp(t.values) if hasattr(t, 'values') else pd.Timestamp(t)
                # Find nearest monthly IOD value
                month_start = t_pd.replace(day=1)
                nearest_idx = iod_series.index.get_indexer([month_start], method="nearest")[0]
                if 0 <= nearest_idx < len(iod_series):
                    data_3d[i, :, :] = iod_series.iloc[nearest_idx]

            ds_out = xr.Dataset({
                ch_name: (["time", "lat", "lon"], data_3d)
            }, coords={"time": times, "lat": new_lat, "lon": new_lon})
            out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
            ds_out.to_netcdf(out_path)
            saved += 1

        print(f"  [OK] {ch_name}: IOD scalar broadcast for {saved} years")
        return saved > 0

    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        return False


def process_climatology_channel(ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                                lon_range, lat_range, years, ref_times):
    """Process WOA23 monthly climatology: interpolate to daily and regrid."""
    files = get_nc_files(raw_dir, ch_cfg["raw_subdir"])
    if not files:
        print(f"  [SKIP] No files found in {ch_cfg['raw_subdir']}/")
        return False

    try:
        # WOA23 has 12 monthly files or a single file with month dim
        ds = xr.open_mfdataset(files, combine="by_coords")
        ds = standardize_coords(ds)
        var_name = find_variable(ds, ch_cfg["var_names"])
        if var_name is None:
            print(f"  [SKIP] {ch_name}: no variable found. Available: {list(ds.data_vars)}")
            ds.close()
            return False

        da = ds[var_name]
        # Take surface level (depth=0)
        if "depth" in da.dims:
            da = da.isel(depth=0)

        da = da.load()

        # Determine the time/month dimension
        if "time" in da.dims:
            monthly_values = da  # assume 12 time steps = 12 months
        elif "month" in da.dims:
            monthly_values = da
        else:
            # Single static field, no monthly variation
            da_regrid = clip_and_regrid(da, new_lon, new_lat, lon_range, lat_range)
            static_values = da_regrid.values.astype(np.float32)
            ds.close()
            # Treat as static
            saved = 0
            for year in years:
                if year not in ref_times:
                    continue
                times = ref_times[year]
                n_days = len(times)
                data_3d = np.broadcast_to(static_values[np.newaxis, :, :], (n_days,) + static_values.shape)
                ds_out = xr.Dataset({
                    ch_name: (["time", "lat", "lon"], data_3d.copy().astype(np.float32))
                }, coords={"time": times, "lat": new_lat, "lon": new_lon})
                out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
                ds_out.to_netcdf(out_path)
                saved += 1
            print(f"  [OK] {ch_name}: climatology (static) for {saved} years")
            return saved > 0

        # Regrid each month
        n_months = len(monthly_values.time) if "time" in monthly_values.dims else len(monthly_values.month)
        regridded_months = []
        for m in range(min(n_months, 12)):
            if "time" in monthly_values.dims:
                slice_m = monthly_values.isel(time=m)
            else:
                slice_m = monthly_values.isel(month=m)
            regridded_months.append(
                clip_and_regrid(slice_m, new_lon, new_lat, lon_range, lat_range).values.astype(np.float32)
            )

        ds.close()

        # Interpolate monthly to daily for each year
        H, W = len(new_lat), len(new_lon)
        saved = 0
        for year in years:
            if year not in ref_times:
                continue
            times = ref_times[year]
            n_days = len(times)
            data_3d = np.zeros((n_days, H, W), dtype=np.float32)

            for i, t in enumerate(times):
                t_pd = pd.Timestamp(t.values) if hasattr(t, 'values') else pd.Timestamp(t)
                month_idx = t_pd.month - 1  # 0-indexed
                data_3d[i] = regridded_months[month_idx % len(regridded_months)]

            ds_out = xr.Dataset({
                ch_name: (["time", "lat", "lon"], data_3d)
            }, coords={"time": times, "lat": new_lat, "lon": new_lon})
            out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
            ds_out.to_netcdf(out_path)
            saved += 1

        print(f"  [OK] {ch_name}: climatology interpolated for {saved} years")
        return saved > 0

    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        return False


def process_glorys_target(raw_dir, out_dir, new_lon, new_lat, lon_range, lat_range,
                          depth_levels, years):
    """Process GLORYS12V1 3D temperature target."""
    files = get_nc_files(raw_dir, "glorys")
    if not files:
        print("  [SKIP] No GLORYS files found!")
        return {}, False

    print("=== Aligning Target (GLORYS thetao) ===")
    ref_times = {}

    try:
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 50})
        ds = standardize_coords(ds)
        var_name = "thetao"
        if var_name not in ds:
            # Try first variable
            var_name = find_variable(ds, ["thetao", "votemper", "temperature"])
            if var_name is None:
                print("  [SKIP] GLORYS: no temperature variable found")
                ds.close()
                return {}, False

        da = ds[var_name]
        da = da.sel(
            lon=slice(lon_range[0] - 1, lon_range[1] + 1),
            lat=slice(lat_range[0] - 1, lat_range[1] + 1),
        )
        if "depth" in da.dims:
            da = da.sel(depth=depth_levels, method="nearest")

        saved = 0
        for year in years:
            out_path = out_dir / "targets" / f"glorys_temp_{year}.nc"
            try:
                da_year = da.sel(time=str(year))
                if len(da_year.time) == 0:
                    continue
                da_year = da_year.load()
                # Sort lat if descending
                if da_year.lat.values[0] > da_year.lat.values[-1]:
                    da_year = da_year.sortby("lat")
                da_regrid = da_year.interp(lon=new_lon, lat=new_lat, method="linear")
                da_regrid.to_dataset(name="thetao").to_netcdf(out_path)

                # Store reference time coordinates
                ref_times[year] = da_regrid.time.values
                saved += 1
                print(f"  -> Saved GLORYS target {year} ({len(da_regrid.time)} days)")
            except Exception as e:
                print(f"  -> Skipped GLORYS {year}: {e}")

        ds.close()
        print(f"  [OK] GLORYS target: {saved} years aligned")
        return ref_times, saved > 0

    except Exception as e:
        print(f"  [ERROR] GLORYS target: {e}")
        traceback.print_exc()
        return {}, False


def create_placeholder_channel(ch_name, out_dir, new_lon, new_lat, years, ref_times):
    """Create zero-filled placeholder files for missing channels."""
    H, W = len(new_lat), len(new_lon)
    saved = 0
    for year in years:
        if year not in ref_times:
            continue
        times = ref_times[year]
        n_days = len(times)
        data_3d = np.zeros((n_days, H, W), dtype=np.float32)
        ds_out = xr.Dataset({
            ch_name: (["time", "lat", "lon"], data_3d)
        }, coords={"time": times, "lat": new_lat, "lon": new_lon})
        out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
        ds_out.to_netcdf(out_path)
        saved += 1
    return saved


# ============================================================================
#  Main Preprocessing Pipeline
# ============================================================================

def preprocess_full(cfg, raw_dir_arg=None):
    raw_dir = Path(raw_dir_arg or cfg.data.get("raw_dir", "data/raw"))
    out_dir = Path(cfg.data.aligned_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(exist_ok=True)
    (out_dir / "targets").mkdir(exist_ok=True)

    lon_range = list(cfg.data.lon_range)
    lat_range = list(cfg.data.lat_range)
    depth_levels = list(cfg.data.depth_levels)
    years = sorted(set(cfg.data.train_years + cfg.data.val_years + cfg.data.test_years))
    input_channels = list(cfg.data.input_channels)

    # Common target grid (0.25 degree)
    new_lon = np.arange(lon_range[0], lon_range[1] + 0.25, 0.25).astype(np.float32)
    new_lat = np.arange(lat_range[0], lat_range[1] + 0.25, 0.25).astype(np.float32)

    print("=" * 60)
    print("OceanEmbed Full Preprocessing (25 Datasets)")
    print(f"  Raw data  : {raw_dir}")
    print(f"  Output    : {out_dir}")
    print(f"  Grid      : {len(new_lon)} x {len(new_lat)} (0.25 deg)")
    print(f"  Years     : {years}")
    print(f"  Channels  : {len(input_channels)}")
    print(f"  Depths    : {depth_levels}")
    print("=" * 60)

    # --- Step 1: Process GLORYS target first (we need its time axis) ---
    ref_times, glorys_ok = process_glorys_target(
        raw_dir, out_dir, new_lon, new_lat, lon_range, lat_range, depth_levels, years
    )
    if not glorys_ok:
        print("\nFATAL: GLORYS target processing failed. Cannot continue without target data.")
        return

    # --- Step 2: Process each input channel ---
    print(f"\n=== Aligning {len(input_channels)} Input Channels ===")
    success_count = 0
    failed_channels = []

    for ch_name in input_channels:
        print(f"\nProcessing channel: {ch_name}")

        if ch_name not in CHANNEL_REGISTRY:
            print(f"  [SKIP] Unknown channel '{ch_name}' (not in registry)")
            failed_channels.append(ch_name)
            continue

        ch_cfg = CHANNEL_REGISTRY[ch_name]
        ch_type = ch_cfg["type"]

        try:
            if ch_type == "dynamic":
                ok = process_dynamic_channel(
                    ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                    lon_range, lat_range, years
                )
            elif ch_type == "era5_yearly":
                ok = process_era5_yearly_channel(
                    ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                    lon_range, lat_range, years
                )
            elif ch_type == "static":
                ok = process_static_channel(
                    ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                    lon_range, lat_range, years, ref_times
                )
            elif ch_type == "scalar_csv":
                ok = process_iod_channel(
                    ch_name, raw_dir, out_dir, new_lon, new_lat, years, ref_times
                )
            elif ch_type == "climatology":
                ok = process_climatology_channel(
                    ch_name, ch_cfg, raw_dir, out_dir, new_lon, new_lat,
                    lon_range, lat_range, years, ref_times
                )
            else:
                print(f"  [SKIP] Unknown channel type: {ch_type}")
                ok = False

            if ok:
                success_count += 1
            else:
                failed_channels.append(ch_name)

        except Exception as e:
            print(f"  [ERROR] {ch_name}: {e}")
            traceback.print_exc()
            failed_channels.append(ch_name)

    # --- Step 3: Create placeholder files for failed channels ---
    if failed_channels:
        print(f"\n=== Creating Zero-Fill Placeholders for {len(failed_channels)} Missing Channels ===")
        for ch_name in failed_channels:
            n = create_placeholder_channel(ch_name, out_dir, new_lon, new_lat, years, ref_times)
            print(f"  [PLACEHOLDER] {ch_name}: {n} zero-filled year files created")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Preprocessing Complete!")
    print(f"  Channels aligned: {success_count}/{len(input_channels)}")
    if failed_channels:
        print(f"  Zero-filled:      {', '.join(failed_channels)}")
    print(f"  Target (GLORYS):  {len(ref_times)} years")
    print(f"  Output directory: {out_dir}")
    print("=" * 60)


# ============================================================================
#  CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OceanEmbed Full Preprocessing — Align ALL 25 Datasets"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--raw-dir", default=None, help="Override raw data directory")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    preprocess_full(cfg, raw_dir_arg=args.raw_dir)
