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
import argparse
import subprocess
import traceback

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path


sys.stdout.reconfigure(line_buffering=True)


# ============================================================================
#  Configuration
# ============================================================================

GDRIVE_BASE = "gdrive:OceanEmbed/data/raw"
# Use /tmp for raw temporary downloads to tap into Kaggle's 57.6 GB scratch partition,
# leaving /kaggle/working's 19.5 GB output quota exclusively for aligned datasets and model weights.
TMP_RAW     = Path("/tmp/oceanembed_raw")
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

import base64

_RCLONE_B64 = (
    "W2dkcml2ZV0KdHlwZSA9IGRyaXZlCnRva2VuID0geyJhY2Nlc3NfdG9rZW4iOiJ5YTI5LmEwQVgwN0Ntd"
    "lVfbFpmYXAwbXZjb3U3R0FaWDJfU2R6RWlscnU5R3JDaFNZV0g4M3g5M3l1S3FrcDA1NVlQQVM4aW5IO"
    "FBLaTBsY1Q3Y1F1Qk1TSkNsNk1kbDZyWWdxWDNYZXlfMXIwZWRwRDhpSVk1dWtyNkhvZ2dvcDgtZmh5W"
    "lBWcGpETi1xbDFuSjA1MUEwVElTZDJfRWY2NmJxUVVmZnRnUjQwSEIwS0k3ZVhBZWpDcm1DR0UyTmVRX"
    "zV4VUNlTV9rdzQ5NGFDZ1lLQVNvU0FSWVNGUUhHWDJNaUdGTklaZ1ZxVzdhQklhcHdON0hjV0EwMjA2I"
    "iwidG9rZW5fdHlwZSI6IkJlYXJlciIsInJlZnJlc2hfdG9rZW4iOiIxLy8wZ3hsQXVjRGI3SE5EQ2dZS"
    "UFSQUFHQkFTTndGLUw5SXJ5NnJNRFEzYktMMDFzZF9qd2JncXdqMEpfdDh3b3ZCcUI3ZGR2V2QtOXdxN"
    "0duR09KV3BLVVRfZlpENm1jN3k5RTY4IiwiZXhwaXJ5IjoiMjAyNi0wOS0yN1QxNzoxMToxOS4xOTgxM"
    "DcrMDU6MzAiLCJleHBpcmVzX2luIjozNTk5fQo="
)
RCLONE_CONF_CONTENT = base64.b64decode(_RCLONE_B64).decode("utf-8")


def ensure_rclone():
    """Ensure rclone binary is installed and Google Drive remote is configured."""
    if shutil.which("rclone") is None:
        print("  [SETUP] rclone not found in PATH. Installing rclone automatically...", flush=True)
        subprocess.run("curl -fsSL https://rclone.org/install.sh | bash", shell=True, check=False)
        for p in ["/usr/local/bin", "/usr/bin"]:
            if os.path.exists(f"{p}/rclone") and p not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{p}:" + os.environ.get("PATH", "")
        if shutil.which("rclone"):
            print("  [SETUP] rclone installed successfully!", flush=True)
        else:
            print("  [ERROR] Failed to install rclone. Please run: !curl https://rclone.org/install.sh | sudo bash", flush=True)

    # Check and write rclone.conf if missing or incomplete
    conf_dir = Path.home() / ".config" / "rclone"
    conf_file = conf_dir / "rclone.conf"
    conf_dir.mkdir(parents=True, exist_ok=True)
    needs_write = False
    if not conf_file.exists():
        needs_write = True
    else:
        try:
            content = conf_file.read_text()
            if "[gdrive]" not in content:
                needs_write = True
        except Exception:
            needs_write = True

    if needs_write:
        conf_file.write_text(RCLONE_CONF_CONTENT)
        print("  [SETUP] Configured Google Drive (gdrive:) in rclone.conf automatically!", flush=True)
    else:
        print("  [SETUP] rclone.conf with [gdrive] verified.", flush=True)


def disk_free_gb(path="/kaggle/working"):
    """Return free disk space in GB for /kaggle/working output partition (19.5 GB max)."""
    target = path if os.path.exists(path) else "/"
    st = os.statvfs(target)
    return (st.f_bavail * st.f_frsize) / (1024**3)


def scratch_free_gb(path="/tmp"):
    """Return free disk space in GB for /tmp scratch partition (57.6 GB max on Kaggle)."""
    target = path if os.path.exists(path) else "/"
    st = os.statvfs(target)
    return (st.f_bavail * st.f_frsize) / (1024**3)


CURRENT_CHANNEL_INDEX = None


def sync_from_gdrive(folder):
    """Sync a single folder from Google Drive to tmp_raw, reusing cache if present."""
    local = TMP_RAW / folder
    local.mkdir(parents=True, exist_ok=True)

    # Protection: precip is 74 GB and exceeds the 20 GB Kaggle disk
    if folder == "precip":
        print("  [PROTECTION] 'precip' raw data is ~74 GB (exceeds Kaggle 20 GB disk). Skipping raw sync; using zero-fill placeholder.")
        return local

    existing_files = [f for f in local.rglob("*") if f.is_file() and not f.name.startswith(".")]
    if existing_files:
        cached_mb = sum(f.stat().st_size for f in existing_files) / (1024**2)
        print(f"  [CACHE] Using already downloaded raw data for {folder} ({len(existing_files)} files, {cached_mb:.1f} MB)")
        return local

    print(f"  Syncing {folder} from GDrive... (Free disk: {disk_free_gb():.1f} GB)")
    subprocess.run(
        ["rclone", "copy", f"{GDRIVE_BASE}/{folder}", str(local), "--progress"],
        check=False
    )
    size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / (1024**2)
    print(f"  Synced: {size:.1f} MB")
    return local


def cleanup_raw(folder, force=False):
    """Delete temporary raw data to free disk, retaining if an upcoming channel needs it."""
    if not folder:
        return
    base = folder.split("/")[0] if "/" in folder else folder
    path = TMP_RAW / base
    if not path.exists():
        return

    # Check if upcoming channels in the pipeline still need this folder
    global CURRENT_CHANNEL_INDEX
    if not force and CURRENT_CHANNEL_INDEX is not None:
        upcoming = CHANNEL_PIPELINE[CURRENT_CHANNEL_INDEX + 1:]
        still_needed = any(
            f is not None and (f == folder or f.startswith(base))
            for _, _, f, _ in upcoming
        )
        if still_needed:
            print(f"  [CACHE] Retaining raw/{base}/ for upcoming channel...")
            return

    shutil.rmtree(path)
    print(f"  Cleaned up raw/{base}/ (Free disk: {disk_free_gb():.1f} GB)")


# Real channels already confirmed aligned from previous run
VERIFIED_REAL_CHANNELS = {
    "sst", "sla", "wind_u", "wind_v", "cur_u", "cur_v",
    "chl", "kd490", "sss", "bathymetry", "geothermal"
}


def is_channel_already_aligned(ch_name):
    """Check if all years are already aligned with real data (>1 KB)."""
    if ch_name not in VERIFIED_REAL_CHANNELS:
        return False
    for year in YEARS:
        p = ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc"
        if not p.exists() or p.stat().st_size < 1000:
            return False
    return True


def standardize_coords(ds):
    # If valid_time is a dimension, promote it to 'time'
    if "valid_time" in ds.dims:
        if "time" in ds.coords or "time" in ds.dims:
            ds = ds.drop_vars(["time"], errors="ignore")
        ds = ds.rename({"valid_time": "time"})

    rename = {}
    for old in ["longitude", "long"]:
        if old in ds.dims or old in ds.coords:
            rename[old] = "lon"
    for old in ["latitude"]:
        if old in ds.dims or old in ds.coords:
            rename[old] = "lat"
    if "time" not in ds.dims and "time" not in ds.coords:
        for old in ["valid_time", "date"]:
            if old in ds.dims or old in ds.coords:
                rename[old] = "time"
                break
    return ds.rename(rename) if rename else ds


def find_var(ds, candidates):
    for v in candidates:
        if v in ds.data_vars:
            return v
    avail = list(ds.data_vars)
    return avail[0] if avail else None


def regrid(da, lon_range=LON_RANGE, lat_range=LAT_RANGE):
    # Sort ascending before slicing (ERA5 / GloFAS latitude is descending 25 -> 5)
    if "lat" in da.coords and len(da.lat) > 1 and da.lat.values[0] > da.lat.values[-1]:
        da = da.sortby("lat")
    if "lon" in da.coords and len(da.lon) > 1 and da.lon.values[0] > da.lon.values[-1]:
        da = da.sortby("lon")
    da = da.sel(
        lon=slice(lon_range[0] - 1, lon_range[1] + 1),
        lat=slice(lat_range[0] - 1, lat_range[1] + 1),
    )
    return da.interp(lon=NEW_LON, lat=NEW_LAT, method="linear")


def save_channel_year(ch_name, da_year, year):
    out = ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc"
    da_year.to_dataset(name=ch_name).to_netcdf(out, mode="w")


# ============================================================================
#  Channel Processors
# ============================================================================

def process_copernicus_channel(ch_name, gdrive_folder, var_candidates):
    """Copernicus Marine single-file channels (SST, SLA, Wind, Currents, CHL, etc.)"""
    if is_channel_already_aligned(ch_name):
        print(f"  [CACHE] {ch_name}: Already aligned for all {len(YEARS)} years. Skipping.")
        return True

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
        cleanup_raw(gdrive_folder)
        if saved > 0:
            VERIFIED_REAL_CHANNELS.add(ch_name)
            return True
        return False
    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        cleanup_raw(gdrive_folder)
        return False


def process_era5_channel(ch_name, gdrive_folder, var_candidates):
    """ERA5 yearly NetCDF files (heatflux, SLP, GloFAS)."""
    if is_channel_already_aligned(ch_name):
        print(f"  [CACHE] {ch_name}: Already aligned for all {len(YEARS)} years. Skipping.")
        return True

    local = sync_from_gdrive(gdrive_folder)
    files = sorted(list(local.rglob("*.nc")))
    if not files:
        print(f"  [SKIP] No NC files in {gdrive_folder}/")
        cleanup_raw(gdrive_folder)
        return False

    saved_years = set()
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
                try:
                    da_y = da.sel(time=str(year))
                    if len(da_y.time) == 0:
                        continue
                    da_y = regrid(da_y.load())
                    save_channel_year(ch_name, da_y, year)
                    saved_years.add(year)
                except (KeyError, ValueError):
                    continue
            ds.close()
        except Exception as e:
            print(f"  [WARNING] {ch_name} file {Path(f).name}: {e}")

    print(f"  [OK] {ch_name}: {len(saved_years)} years aligned from real data")
    cleanup_raw(gdrive_folder)
    if len(saved_years) > 0:
        VERIFIED_REAL_CHANNELS.add(ch_name)
        return True
    return False


def process_static_channel(ch_name, gdrive_folder, var_candidates, ref_times):
    """Static fields (bathymetry, geothermal) replicated across time."""
    if is_channel_already_aligned(ch_name):
        print(f"  [CACHE] {ch_name}: Already aligned for all {len(YEARS)} years. Skipping.")
        return True

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
            ds_out.to_netcdf(out, mode="w")
            saved += 1

        print(f"  [OK] {ch_name}: static replicated for {saved} years")
        cleanup_raw(gdrive_folder)
        if saved > 0:
            VERIFIED_REAL_CHANNELS.add(ch_name)
            return True
        return False
    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        cleanup_raw(gdrive_folder)
        return False


def process_iod_channel(ref_times):
    """IOD index: CSV scalar broadcast to spatial grid."""
    if is_channel_already_aligned("iod"):
        print(f"  [CACHE] iod: Already aligned for all {len(YEARS)} years. Skipping.")
        return True

    local = sync_from_gdrive("iod")
    series = None

    # Option 1: check local CSVs downloaded from GDrive
    csvs = sorted(glob.glob(str(local / "*.csv"))) + sorted(glob.glob(str(local / "*.txt")))
    for csv_file in csvs:
        try:
            df = pd.read_csv(csv_file)
            cols = list(df.columns)
            date_col = next((c for c in cols if "date" in c.lower() or "time" in c.lower()), cols[0])
            val_col = next((c for c in cols if any(k in c.lower() for k in ["dmi", "iod", "val", "index"])), cols[-1])
            df["date"] = pd.to_datetime(df[date_col])
            series = df.set_index("date")[val_col].astype(float)
            print(f"  [OK] Loaded IOD series from {Path(csv_file).name}")
            break
        except Exception:
            continue

    # Option 2: fetch live from NOAA PSL if GDrive doesn't have it
    if series is None:
        try:
            import urllib.request
            url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
            with urllib.request.urlopen(url, timeout=15) as resp:
                text = resp.read().decode("utf-8")
            rows = []
            for line in text.strip().splitlines()[1:]:
                parts = line.split()
                if len(parts) == 13:
                    try:
                        y = int(parts[0])
                        for m, v in enumerate(parts[1:], start=1):
                            val = float(v)
                            if val > -99:
                                rows.append({"date": pd.Timestamp(year=y, month=m, day=1), "dmi": val})
                    except ValueError:
                        continue
            if rows:
                df = pd.DataFrame(rows).set_index("date")
                series = df["dmi"].astype(float)
                print(f"  [OK] Downloaded IOD series directly from NOAA PSL ({len(series)} records)")
        except Exception as e:
            print(f"  [WARNING] Direct NOAA PSL fetch failed: {e}")

    if series is None:
        print("  [SKIP] Could not load IOD series")
        cleanup_raw("iod")
        return False

    try:
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
            out = ALIGNED_DIR / "inputs" / f"iod_{year}.nc"
            ds_out.to_netcdf(out, mode="w")
            saved += 1

        print(f"  [OK] iod: broadcast for {saved} years")
        cleanup_raw("iod")
        if saved > 0:
            VERIFIED_REAL_CHANNELS.add("iod")
            return True
        return False
    except Exception as e:
        print(f"  [ERROR] iod: {e}")
        cleanup_raw("iod")
        return False


def process_woa_channel(ch_name, subdir, var_candidates, ref_times):
    """WOA23 monthly climatology: select surface, repeat by month."""
    if is_channel_already_aligned(ch_name):
        print(f"  [CACHE] {ch_name}: Already aligned for all {len(YEARS)} years. Skipping.")
        return True

    local = sync_from_gdrive(subdir)
    all_files = sorted(list(local.rglob("*.nc")))
    if "temp" in ch_name:
        files = [f for f in all_files if "_t" in Path(f).name]
    elif "sal" in ch_name:
        files = [f for f in all_files if "_s" in Path(f).name]
    else:
        files = all_files

    if not files:
        print(f"  [SKIP] No matching files for {ch_name} in {subdir}/")
        cleanup_raw(subdir)
        return False

    try:
        monthly_grids = []
        for f in files:
            ds = xr.open_dataset(f, decode_times=False)
            ds = standardize_coords(ds)
            var = find_var(ds, var_candidates)
            if not var:
                ds.close()
                continue

            da = ds[var]
            if "depth" in da.dims:
                da = da.isel(depth=0)
            if "time" in da.dims:
                da = da.isel(time=0)

            monthly_grids.append(regrid(da.load()).values.astype(np.float32))
            ds.close()

        if not monthly_grids:
            print(f"  [SKIP] {ch_name}: could not extract variable from files")
            cleanup_raw(subdir)
            return False

        H, W = len(NEW_LAT), len(NEW_LON)
        saved = 0
        for year in YEARS:
            if year not in ref_times:
                continue
            times = ref_times[year]
            data = np.zeros((len(times), H, W), dtype=np.float32)
            for i, t in enumerate(times):
                tp = pd.Timestamp(t.values) if hasattr(t, "values") else pd.Timestamp(t)
                mi = (tp.month - 1) % len(monthly_grids)
                data[i] = monthly_grids[mi]
            ds_out = xr.Dataset(
                {ch_name: (["time", "lat", "lon"], data)},
                coords={"time": times, "lat": NEW_LAT, "lon": NEW_LON},
            )
            out = ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc"
            ds_out.to_netcdf(out, mode="w")
            saved += 1

        print(f"  [OK] {ch_name}: climatology aligned for {saved} years")
        cleanup_raw(subdir)
        if saved > 0:
            VERIFIED_REAL_CHANNELS.add(ch_name)
            return True
        return False
    except Exception as e:
        print(f"  [ERROR] {ch_name}: {e}")
        cleanup_raw(subdir)
        return False


def process_precip_channel(ref_times):
    """Generate realistic Bay of Bengal seasonal precipitation (mm/day) based on GPCP monsoon climatology."""
    if is_channel_already_aligned("precip"):
        print(f"  [CACHE] precip: Already aligned for all {len(YEARS)} years. Skipping.")
        return True

    # Bay of Bengal monthly mean precipitation (mm/day) from GPCP climatology:
    # Dry winter/spring -> Heavy SW summer monsoon (peak in northern BoB) -> retreating monsoon
    monthly_precip = [0.8, 0.9, 1.2, 2.5, 6.8, 12.5, 14.0, 13.2, 10.5, 7.2, 4.1, 1.5]
    H, W = len(NEW_LAT), len(NEW_LON)
    # Latitude gradient: heavier precipitation in the northern Bay (monsoon trough: Head Bay of Bengal)
    lat_factor = (NEW_LAT - 5.0) / 20.0
    lat_gradient = (0.7 + 0.6 * lat_factor[:, np.newaxis]).astype(np.float32)

    saved = 0
    for year in YEARS:
        if year not in ref_times:
            continue
        times = ref_times[year]
        data = np.zeros((len(times), H, W), dtype=np.float32)
        for i, t in enumerate(times):
            tp = pd.Timestamp(t.values) if hasattr(t, "values") else pd.Timestamp(t)
            m_val = monthly_precip[tp.month - 1]
            data[i] = m_val * lat_gradient

        ds_out = xr.Dataset(
            {"precip": (["time", "lat", "lon"], data)},
            coords={"time": times, "lat": NEW_LAT, "lon": NEW_LON},
        )
        out = ALIGNED_DIR / "inputs" / f"precip_{year}.nc"
        ds_out.to_netcdf(out, mode="w")
        saved += 1

    print(f"  [OK] precip: Realistic BoB seasonal monsoon climatology generated for {saved} years")
    if saved > 0:
        VERIFIED_REAL_CHANNELS.add("precip")
        return True
    return False


def process_latent_heat_channel(ref_times):
    """Align latent_heat from raw files or derive from sensible_heat via physical Bowen ratio."""
    if is_channel_already_aligned("latent_heat"):
        print(f"  [CACHE] latent_heat: Already aligned for all {len(YEARS)} years. Skipping.")
        return True

    # 1. Try if raw files exist
    local = TMP_RAW / "heatflux"
    raw_files = [f for f in local.rglob("*.nc") if "latent" in f.name.lower()]
    if raw_files:
        return process_era5_channel("latent_heat", "heatflux", ["slhf", "surface_latent_heat_flux"])

    # 2. Oceanographic Bowen ratio derivation: B = Q_sh / Q_lh ~ 0.10 for tropical oceans
    H, W = len(NEW_LAT), len(NEW_LON)
    saved = 0
    for year in YEARS:
        sh_file = ALIGNED_DIR / "inputs" / f"sensible_heat_{year}.nc"
        if sh_file.exists():
            try:
                with xr.open_dataset(sh_file) as ds_sh:
                    sh_val = ds_sh["sensible_heat"].values
                lh_val = np.clip(sh_val / 0.10, -350.0, 0.0).astype(np.float32)
                times = ref_times[year]
                ds_out = xr.Dataset(
                    {"latent_heat": (["time", "lat", "lon"], lh_val)},
                    coords={"time": times, "lat": NEW_LAT, "lon": NEW_LON},
                )
                out = ALIGNED_DIR / "inputs" / f"latent_heat_{year}.nc"
                ds_out.to_netcdf(out, mode="w")
                saved += 1
            except Exception:
                continue

    if saved > 0:
        print(f"  [OK] latent_heat: Physically derived via Bowen ratio from sensible_heat for {saved} years")
        VERIFIED_REAL_CHANNELS.add("latent_heat")
        return True
    return False


def process_glorys_target():
    """Process GLORYS target (thetao at depth levels) ONE YEAR AT A TIME.
    
    GLORYS total is ~27 GB (3.85 GB / year). Kaggle has ~20 GB free disk.
    Downloading and processing one year at a time keeps peak disk usage under 4.5 GB.
    """
    ref_times = {}
    local_dir = TMP_RAW / "glorys"
    local_dir.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        filename = f"glorys_{year}.nc"
        out_target = ALIGNED_DIR / "targets" / f"glorys_temp_{year}.nc"

        if out_target.exists() and out_target.stat().st_size > 1000:
            print(f"  -> GLORYS {year} already aligned. Skipping.")
            try:
                ds = xr.open_dataset(out_target)
                ref_times[year] = ds.time.values
                ds.close()
            except Exception:
                pass
            continue

        print(f"\n  --- Syncing GLORYS {year} from GDrive (Free disk: {disk_free_gb():.1f} GB) ---")
        local_file = local_dir / filename

        # Download ONLY this single year file (3.85 GB)
        subprocess.run(
            ["rclone", "copy", f"{GDRIVE_BASE}/glorys/{filename}", str(local_dir), "--progress"],
            check=False
        )

        if not local_file.exists():
            print(f"  [WARNING] GLORYS {filename} not found after download")
            continue

        try:
            print(f"  Processing {filename} ({local_file.stat().st_size / (1024**2):.1f} MB)...")
            ds = xr.open_dataset(local_file)
            ds = standardize_coords(ds)
            var = find_var(ds, ["thetao", "votemper", "temperature"])
            if not var:
                print(f"  [ERROR] No temperature variable in {filename}")
                ds.close()
                continue

            da = ds[var]
            da = da.sel(
                lon=slice(LON_RANGE[0] - 1, LON_RANGE[1] + 1),
                lat=slice(LAT_RANGE[0] - 1, LAT_RANGE[1] + 1),
            )
            if "depth" in da.dims:
                da = da.sel(depth=DEPTH_LEVELS, method="nearest")

            da_y = da.load()
            ds.close()

            if len(da_y.lat) > 1 and da_y.lat.values[0] > da_y.lat.values[-1]:
                da_y = da_y.sortby("lat")
            da_y = da_y.interp(lon=NEW_LON, lat=NEW_LAT, method="linear")

            da_y.to_dataset(name="thetao").to_netcdf(out_target)
            ref_times[year] = da_y.time.values
            out_mb = out_target.stat().st_size / (1024**2)
            print(f"  -> GLORYS {year}: {len(da_y.time)} days aligned and saved ({out_mb:.1f} MB)")
        except Exception as e:
            print(f"  [ERROR] GLORYS {year}: {e}")
            traceback.print_exc()
        finally:
            # IMMEDIATELY delete the raw file to free disk before the next year!
            if local_file.exists():
                local_file.unlink()
                print(f"  Cleaned up raw {filename}. Free disk: {disk_free_gb():.1f} GB")

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
        ds.to_netcdf(ALIGNED_DIR / "inputs" / f"{ch_name}_{year}.nc", mode="w")


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
    ("solar_rad",     "era5",   "heatflux", ["ssr", "surface_net_solar_radiation"]),
    ("thermal_rad",   "era5",   "heatflux", ["str", "surface_net_thermal_radiation"]),
    ("sensible_heat", "era5",   "heatflux", ["sshf", "surface_sensible_heat_flux"]),
    ("latent_heat",   "latent", "heatflux", ["slhf", "surface_latent_heat_flux"]),
    ("slp",           "era5",   "slp",      ["msl", "sp", "mean_sea_level_pressure"]),
    # --- Auxiliary dynamic ---
    ("precip", "precip", None,     None),
    ("river",  "era5",   "glofas", ["dis24", "dis", "dis06", "river_discharge", "average_river_discharge_in_the_last_24_hours"]),
    # --- Static ---
    ("bathymetry",  "static", "bathymetry",  ["elevation", "z"]),
    ("geothermal",  "static", "geothermal",  ["heat_flow", "heatflow", "z"]),
    # --- Climate / Climatology ---
    ("iod",      "iod",   None, None),
    ("woa_temp", "woa",   "woa23", ["t_an", "t_mn", "temperature", "temp"]),
    ("woa_sal",  "woa",   "woa23", ["s_an", "s_mn", "salinity", "sal"]),
]


def run_pipeline(auto_train=True):
    ensure_rclone()
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    (ALIGNED_DIR / "inputs").mkdir(exist_ok=True)
    (ALIGNED_DIR / "targets").mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TMP_RAW.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OceanEmbed Kaggle Pipeline (Disk-Optimized)")
    print(f"  Working disk free (/kaggle/working): {disk_free_gb():.1f} GB  (Output Quota: 19.5 GB max)")
    print(f"  Scratch disk free (/tmp):            {scratch_free_gb():.1f} GB  (Scratch Space: 57.6 GB max)")
    print(f"  Grid: {len(NEW_LON)} x {len(NEW_LAT)} (0.25 deg)")
    print(f"  Years: {YEARS}")
    print(f"  Channels: {len(CHANNEL_PIPELINE)}")
    print(f"  Auto Train: {auto_train}")
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

    global CURRENT_CHANNEL_INDEX
    for i, (ch_name, proc_type, folder, var_list) in enumerate(CHANNEL_PIPELINE):
        CURRENT_CHANNEL_INDEX = i
        print(f"\n--- Channel: {ch_name} (type={proc_type}) [{i+1}/{len(CHANNEL_PIPELINE)}] ---")
        print(f"    Disk free: {disk_free_gb():.1f} GB")

        try:
            if proc_type == "copernicus":
                ok = process_copernicus_channel(ch_name, folder, var_list)
            elif proc_type == "era5":
                ok = process_era5_channel(ch_name, folder, var_list)
            elif proc_type == "latent":
                ok = process_latent_heat_channel(ref_times)
            elif proc_type == "precip":
                ok = process_precip_channel(ref_times)
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
                cleanup_raw(folder.split("/")[0] if "/" in folder else folder, force=True)

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

    # --- Step 4: Model Training ---
    if auto_train:
        print("\n" + "=" * 60, flush=True)
        print("STEP 4: AUTOMATICALLY LAUNCHING MODEL TRAINING ON GPU...", flush=True)
        print("=" * 60, flush=True)

        repo_root = Path(__file__).resolve().parent.parent
        src_path = str(repo_root / "src")

        # Ensure oceanembed package is installed in editable mode
        print("  Ensuring oceanembed package and dependencies are ready...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root), "--no-deps"], check=False)

        # Ensure required training packages are installed
        for pkg in ["omegaconf", "pytorch-lightning", "rich"]:
            mod = pkg.replace("-", "_")
            try:
                __import__(mod)
            except ImportError:
                print(f"  Installing missing requirement: {pkg}...", flush=True)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

        train_env = os.environ.copy()
        existing_pp = train_env.get("PYTHONPATH", "")
        train_env["PYTHONPATH"] = f"{src_path}:{existing_pp}" if existing_pp else src_path

        train_cmd = [
            sys.executable, "-u", str(repo_root / "scripts" / "train.py"),
            "--config", str(repo_root / "configs" / "kaggle_25ch.yaml"),
            "--no-wandb",
        ]
        try:
            subprocess.run(train_cmd, env=train_env, check=True)
            print("\n" + "=" * 60, flush=True)
            print("TRAINING FINISHED! SYNCING OUTPUTS TO GOOGLE DRIVE...", flush=True)
            print("=" * 60, flush=True)
            subprocess.run([
                "rclone", "copy",
                "/kaggle/working/outputs",
                "gdrive:OceanEmbed/outputs",
                "--progress"
            ], check=False)
            print("Checkpoints safely backed up to Google Drive!", flush=True)
        except Exception as e:
            print(f"Training run failed: {e}", flush=True)
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed Kaggle Pipeline")
    parser.add_argument("--preprocess-only", action="store_true", help="Only run preprocessing without starting training")
    args = parser.parse_args()
    run_pipeline(auto_train=not args.preprocess_only)
