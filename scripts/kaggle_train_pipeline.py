#!/usr/bin/env python3
"""
OceanEmbed Kaggle-Optimized Preprocessing Pipeline
====================================================
Processes all 25 datasets ONE CHANNEL AT A TIME to stay within
Kaggle's 20 GB disk limit.

Flow for each channel:
  1. rclone sync raw data from Google Drive (~1-7 GB)
  2. Align to 0.25-degree grid (~80 MB per channel)
  3. DELETE raw data immediately
  4. Repeat for next channel

Peak disk usage: ~12 GB (largest single raw file + all aligned data)
Final aligned data: ~2.5 GB total (fits easily)

Usage:
    python scripts/kaggle_train_pipeline.py
"""

import os
import sys
import glob
import shutil
import subprocess
import traceback

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


# ============================================================================
#  Configuration
# ============================================================================

GDRIVE_BASE = "gdrive:OceanEmbed/data/raw"
TMP_RAW     = Path("/kaggle/working/tmp_raw")
ALIGNED_DIR = Path("/kaggle/working/aligned")
PROCESSED   = Path("/kaggle/working/processed")

LON_RANGE = [75.0, 100.0]
LAT_RANGE = [5.0, 25.0]
DEPTH_LEVELS = [0, 50, 100, 150, 200, 500]
YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]

NEW_LON = np.arange(75.0, 100.25, 0.25).astype(np.float32)
NEW_LAT = np.arange(5.0, 25.25, 0.25).astype(np.float32)


# ============================================================================
#  Helper Functions
# ============================================================================

def disk_free_gb():
    """Return free disk space in GB."""
    st = os.statvfs("/kaggle/working")
    return (st.f_bavail * st.f_frsize) / (1024**3)


def sync_from_gdrive(folder):
    """Sync a single folder from Google Drive to tmp_raw."""
    local = TMP_RAW / folder
    local.mkdir(parents=True, exist_ok=True)
    print(f"  Syncing {folder} from GDrive... (Free disk: {disk_free_gb():.1f} GB)")
    subprocess.run(
        ["rclone", "copy", f"{GDRIVE_BASE}/{folder}", str(local), "--progress"],
        check=False
    )
    size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / (1024**2)
    print(f"  Synced: {size:.1f} MB")
    return local


def cleanup_raw(folder):
    """Delete temporary raw data to free disk."""
    path = TMP_RAW / folder
    if path.exists():
        shutil.rmtree(path)
        print(f"  Cleaned up raw/{folder}/")


def standardize_coords(ds):
    rename = {}
    for old, new in [("longitude", "lon"), ("latitude", "lat")]:
        if old in ds.dims or old in ds.coords:
            rename[old] = new
    return ds.rename(rename) if rename else ds


def find_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    avail = list(ds.data_vars)
    return avail[0] if avail else None


def regrid(da, lon_range=LON_RANGE, lat_range=LAT_RANGE):
    da = da.sel(
        lon=slice(lon_range[0] - 1, lon_range[1] + 1),
        lat=slice(lat_range[0] - 1, lat_range[1] + 1),
    )
    if len(da.lat) > 1 and da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    return da.interp(lon=NEW_LON, lat=NEW_LAT, method="linear")


def save_channel_year(ch_name, da_year, year):
    out = ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc"
    da_year.to_dataset(name=ch_name).to_netcdf(out)


# ============================================================================
#  Channel Processors
# ============================================================================

def process_copernicus_channel(ch_name, gdrive_folder, var_candidates):
    """Copernicus Marine single-file channels (SST, SLA, Wind, Currents, CHL, etc.)"""
    local = sync_from_gdrive(gdrive_folder)
    files = sorted(glob.glob(str(local / "*.nc")))
    if not files:
        print(f"  [SKIP] No NC files in {gdrive_folder}/")
        cleanup_raw(gdrive_folder)
        return False

    try:
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 100})
        ds = standardize_coords(ds)
        var = find_var(ds, var_candidates)
        if not var:
            print(f"  [SKIP] No variable found. Available: {list(ds.data_vars)}")
            ds.close()
            cleanup_raw(gdrive_folder)
            return False

        da = ds[var]
        if "depth" in da.dims:
            da = da.isel(depth=0)

        saved = 0
        for year in YEARS:
            try:
                da_y = da.sel(time=str(year))
                if len(da_y.time) == 0:
                    continue
                da_y = regrid(da_y.load())
                save_channel_year(ch_name, da_y, year)
                saved += 1
            except (KeyError, ValueError):
                continue

        ds.close()
        print(f"  [OK] {ch_name}: {saved} years aligned")
    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        saved = 0

    cleanup_raw(gdrive_folder)
    return saved > 0


def process_era5_channel(ch_name, gdrive_folder, var_candidates):
    """ERA5 yearly NetCDF files (heatflux, SLP, GloFAS)."""
    local = sync_from_gdrive(gdrive_folder)
    files = sorted(glob.glob(str(local / "*.nc")))
    if not files:
        print(f"  [SKIP] No NC files in {gdrive_folder}/")
        cleanup_raw(gdrive_folder)
        return False

    saved = 0
    for f in files:
        try:
            ds = xr.open_dataset(f)
            ds = standardize_coords(ds)
            var = find_var(ds, var_candidates)
            if not var:
                ds.close()
                continue

            da = ds[var]
            if "depth" in da.dims:
                da = da.isel(depth=0)

            for year in YEARS:
                out = ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc"
                if out.exists():
                    saved += 1
                    continue
                try:
                    da_y = da.sel(time=str(year))
                    if len(da_y.time) == 0:
                        continue
                    da_y = regrid(da_y.load())
                    save_channel_year(ch_name, da_y, year)
                    saved += 1
                except (KeyError, ValueError):
                    continue
            ds.close()
        except Exception as e:
            print(f"  [WARNING] {ch_name} file {Path(f).name}: {e}")

    print(f"  [OK] {ch_name}: {saved} year-files")
    cleanup_raw(gdrive_folder)
    return saved > 0


def process_static_channel(ch_name, gdrive_folder, var_candidates, ref_times):
    """Static fields (bathymetry, geothermal) replicated across time."""
    local = sync_from_gdrive(gdrive_folder)
    files = sorted(glob.glob(str(local / "*.nc")))
    if not files:
        print(f"  [SKIP] No NC files in {gdrive_folder}/")
        cleanup_raw(gdrive_folder)
        return False

    try:
        ds = xr.open_dataset(files[0])
        ds = standardize_coords(ds)
        var = find_var(ds, var_candidates)
        if not var:
            print(f"  [SKIP] {ch_name}: no variable. Available: {list(ds.data_vars)}")
            ds.close()
            cleanup_raw(gdrive_folder)
            return False

        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=0)
        if "depth" in da.dims:
            da = da.isel(depth=0)

        static = regrid(da.load()).values.astype(np.float32)
        ds.close()

        H, W = len(NEW_LAT), len(NEW_LON)
        saved = 0
        for year in YEARS:
            if year not in ref_times:
                continue
            times = ref_times[year]
            n = len(times)
            data = np.broadcast_to(static[np.newaxis], (n, H, W)).copy()
            ds_out = xr.Dataset(
                {ch_name: (["time", "lat", "lon"], data)},
                coords={"time": times, "lat": NEW_LAT, "lon": NEW_LON},
            )
            out = ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc"
            ds_out.to_netcdf(out)
            saved += 1

        print(f"  [OK] {ch_name}: static replicated for {saved} years")
    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        saved = 0

    cleanup_raw(gdrive_folder)
    return saved > 0


def process_iod_channel(ref_times):
    """IOD index: CSV scalar broadcast to spatial grid."""
    local = sync_from_gdrive("iod")
    csvs = sorted(glob.glob(str(local / "*.csv")))
    if not csvs:
        print("  [SKIP] No IOD CSV files")
        cleanup_raw("iod")
        return False

    try:
        df = pd.read_csv(csvs[0])
        cols = list(df.columns)
        date_col = cols[0]
        val_col = cols[1] if len(cols) >= 2 else cols[0]
        df["date"] = pd.to_datetime(df[date_col])
        df = df.set_index("date").sort_index()
        series = df[val_col].astype(float)

        H, W = len(NEW_LAT), len(NEW_LON)
        saved = 0
        for year in YEARS:
            if year not in ref_times:
                continue
            times = ref_times[year]
            data = np.zeros((len(times), H, W), dtype=np.float32)
            for i, t in enumerate(times):
                tp = pd.Timestamp(t.values) if hasattr(t, "values") else pd.Timestamp(t)
                ms = tp.replace(day=1)
                idx = series.index.get_indexer([ms], method="nearest")[0]
                if 0 <= idx < len(series):
                    data[i, :, :] = series.iloc[idx]
            ds_out = xr.Dataset(
                {"iod": (["time", "lat", "lon"], data)},
                coords={"time": times, "lat": NEW_LAT, "lon": NEW_LON},
            )
            (ALIGNED_DIR / "inputs" / f"iod_{year}.nc").pipe(lambda p: ds_out.to_netcdf(p))
            saved += 1

        print(f"  [OK] iod: broadcast for {saved} years")
    except Exception as e:
        print(f"  [ERROR] iod: {e}")
        saved = 0

    cleanup_raw("iod")
    return saved > 0


def process_woa_channel(ch_name, subdir, var_candidates, ref_times):
    """WOA23 monthly climatology: select surface, repeat by month."""
    local = sync_from_gdrive(subdir)
    files = sorted(glob.glob(str(local / "*.nc")))
    if not files:
        print(f"  [SKIP] No files in {subdir}/")
        cleanup_raw(subdir)
        return False

    try:
        ds = xr.open_mfdataset(files, combine="by_coords")
        ds = standardize_coords(ds)
        var = find_var(ds, var_candidates)
        if not var:
            ds.close()
            cleanup_raw(subdir)
            return False

        da = ds[var]
        if "depth" in da.dims:
            da = da.isel(depth=0)
        da = da.load()

        # Get monthly regridded values
        monthly = []
        n_time = len(da.time) if "time" in da.dims else 1
        for m in range(min(n_time, 12)):
            if "time" in da.dims:
                sl = da.isel(time=m)
            else:
                sl = da
            monthly.append(regrid(sl).values.astype(np.float32))
        ds.close()

        H, W = len(NEW_LAT), len(NEW_LON)
        saved = 0
        for year in YEARS:
            if year not in ref_times:
                continue
            times = ref_times[year]
            data = np.zeros((len(times), H, W), dtype=np.float32)
            for i, t in enumerate(times):
                tp = pd.Timestamp(t.values) if hasattr(t, "values") else pd.Timestamp(t)
                mi = (tp.month - 1) % len(monthly)
                data[i] = monthly[mi]
            ds_out = xr.Dataset(
                {ch_name: (["time", "lat", "lon"], data)},
                coords={"time": times, "lat": NEW_LAT, "lon": NEW_LON},
            )
            (ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc").pipe(lambda p: ds_out.to_netcdf(p))
            saved += 1

        print(f"  [OK] {ch_name}: climatology for {saved} years")
    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        saved = 0

    cleanup_raw(subdir)
    return saved > 0


def process_glorys_target():
    """Process GLORYS target (thetao at depth levels). Returns ref_times dict."""
    local = sync_from_gdrive("glorys")
    files = sorted(glob.glob(str(local / "*.nc")))
    if not files:
        print("FATAL: No GLORYS files found!")
        cleanup_raw("glorys")
        return {}

    ref_times = {}
    try:
        ds = xr.open_mfdataset(files, combine="by_coords", chunks={"time": 30})
        ds = standardize_coords(ds)
        var = find_var(ds, ["thetao", "votemper", "temperature"])
        if not var:
            print("FATAL: No temperature variable in GLORYS")
            ds.close()
            cleanup_raw("glorys")
            return {}

        da = ds[var]
        da = da.sel(
            lon=slice(LON_RANGE[0] - 1, LON_RANGE[1] + 1),
            lat=slice(LAT_RANGE[0] - 1, LAT_RANGE[1] + 1),
        )
        if "depth" in da.dims:
            da = da.sel(depth=DEPTH_LEVELS, method="nearest")

        for year in YEARS:
            try:
                da_y = da.sel(time=str(year))
                if len(da_y.time) == 0:
                    continue
                da_y = da_y.load()
                if da_y.lat.values[0] > da_y.lat.values[-1]:
                    da_y = da_y.sortby("lat")
                da_y = da_y.interp(lon=NEW_LON, lat=NEW_LAT, method="linear")
                out = ALIGNED_DIR / "targets" / f"glorys_temp_{year}.nc"
                da_y.to_dataset(name="thetao").to_netcdf(out)
                ref_times[year] = da_y.time.values
                print(f"  -> GLORYS {year}: {len(da_y.time)} days")
            except Exception as e:
                print(f"  -> Skip GLORYS {year}: {e}")

        ds.close()
    except Exception as e:
        print(f"FATAL ERROR processing GLORYS: {e}")
        traceback.print_exc()

    cleanup_raw("glorys")
    return ref_times


def create_placeholder(ch_name, ref_times):
    """Zero-fill a missing channel."""
    H, W = len(NEW_LAT), len(NEW_LON)
    for year in YEARS:
        if year not in ref_times:
            continue
        times = ref_times[year]
        data = np.zeros((len(times), H, W), dtype=np.float32)
        ds = xr.Dataset(
            {ch_name: (["time", "lat", "lon"], data)},
            coords={"time": times, "lat": NEW_LAT, "lon": NEW_LON},
        )
        ds.to_netcdf(ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc")


# ============================================================================
#  Pipeline Execution Order
# ============================================================================

CHANNEL_PIPELINE = [
    # (channel_name, processor_func, args)
    # --- Copernicus Marine satellite obs ---
    ("sst",    "copernicus", "sst",      ["analysed_sst"]),
    ("sla",    "copernicus", "sla",      ["sla"]),
    ("wind_u", "copernicus", "wind",     ["eastward_wind"]),
    ("wind_v", "copernicus", "wind",     ["northward_wind"]),
    ("cur_u",  "copernicus", "currents", ["uo"]),
    ("cur_v",  "copernicus", "currents", ["vo"]),
    ("chl",    "copernicus", "chl",      ["CHL"]),
    ("kd490",  "copernicus", "kd490",    ["KD490"]),
    ("sss",    "copernicus", "sss",      ["sos"]),
    # --- ERA5 atmospheric forcing ---
    ("solar_rad",     "era5", "heatflux", ["ssr", "surface_net_solar_radiation"]),
    ("thermal_rad",   "era5", "heatflux", ["str", "surface_net_thermal_radiation"]),
    ("latent_heat",   "era5", "heatflux", ["slhf", "surface_latent_heat_flux"]),
    ("sensible_heat", "era5", "heatflux", ["sshf", "surface_sensible_heat_flux"]),
    ("slp",           "era5", "slp",      ["msl", "sp", "mean_sea_level_pressure"]),
    # --- Auxiliary dynamic ---
    ("precip", "copernicus", "precip", ["precipitation", "precipitationCal"]),
    ("river",  "era5",       "glofas", ["dis24", "dis"]),
    # --- Static ---
    ("bathymetry",  "static", "bathymetry",  ["elevation", "z"]),
    ("geothermal",  "static", "geothermal",  ["heat_flow", "heatflow", "z"]),
    # --- Climate / Climatology ---
    ("iod",      "iod",   None, None),
    ("woa_temp", "woa",   "woa23/temperature", ["t_an", "t_mn"]),
    ("woa_sal",  "woa",   "woa23/salinity",    ["s_an", "s_mn"]),
]


def run_pipeline():
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    (ALIGNED_DIR / "inputs").mkdir(exist_ok=True)
    (ALIGNED_DIR / "targets").mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TMP_RAW.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OceanEmbed Kaggle Pipeline (Disk-Optimized)")
    print(f"  Free disk: {disk_free_gb():.1f} GB")
    print(f"  Grid: {len(NEW_LON)} x {len(NEW_LAT)} (0.25 deg)")
    print(f"  Years: {YEARS}")
    print(f"  Channels: {len(CHANNEL_PIPELINE)}")
    print("=" * 60)

    # --- Step 1: GLORYS target (largest file, process first) ---
    print("\n[STEP 1/3] Processing GLORYS target...")
    ref_times = process_glorys_target()
    if not ref_times:
        print("FATAL: No GLORYS data could be processed. Aborting.")
        return False
    print(f"  Free disk after GLORYS: {disk_free_gb():.1f} GB")

    # --- Step 2: Process each input channel one at a time ---
    print(f"\n[STEP 2/3] Processing {len(CHANNEL_PIPELINE)} input channels...")
    success = 0
    failed = []

    for ch_name, proc_type, folder, var_list in CHANNEL_PIPELINE:
        print(f"\n--- Channel: {ch_name} (type={proc_type}) ---")
        print(f"    Disk free: {disk_free_gb():.1f} GB")

        try:
            if proc_type == "copernicus":
                ok = process_copernicus_channel(ch_name, folder, var_list)
            elif proc_type == "era5":
                ok = process_era5_channel(ch_name, folder, var_list)
            elif proc_type == "static":
                ok = process_static_channel(ch_name, folder, var_list, ref_times)
            elif proc_type == "iod":
                ok = process_iod_channel(ref_times)
            elif proc_type == "woa":
                ok = process_woa_channel(ch_name, folder, var_list, ref_times)
            else:
                ok = False

            if ok:
                success += 1
            else:
                failed.append(ch_name)
        except Exception as e:
            print(f"  [ERROR] {ch_name}: {e}")
            failed.append(ch_name)
            # Ensure cleanup even on error
            if folder:
                cleanup_raw(folder.split("/")[0] if "/" in folder else folder)

    # --- Step 3: Create placeholders for failed channels ---
    if failed:
        print(f"\n[STEP 3/3] Creating zero-fill placeholders for {len(failed)} missing channels...")
        for ch_name in failed:
            create_placeholder(ch_name, ref_times)
            print(f"  [PLACEHOLDER] {ch_name}")

    # --- Cleanup ---
    if TMP_RAW.exists():
        shutil.rmtree(TMP_RAW)

    # --- Summary ---
    aligned_size = sum(
        f.stat().st_size for f in ALIGNED_DIR.rglob("*.nc") if f.is_file()
    ) / (1024**3)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print(f"  Channels aligned: {success}/{len(CHANNEL_PIPELINE)}")
    if failed:
        print(f"  Zero-filled: {', '.join(failed)}")
    print(f"  Aligned data size: {aligned_size:.2f} GB")
    print(f"  Free disk: {disk_free_gb():.1f} GB")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_pipeline()
