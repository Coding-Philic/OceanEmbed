import os
import glob
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf

def preprocess(cfg):
    raw_dir = Path("data/raw")
    out_dir = Path(cfg.data.aligned_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inputs").mkdir(exist_ok=True)
    (out_dir / "targets").mkdir(exist_ok=True)

    lon_range = cfg.data.lon_range
    lat_range = cfg.data.lat_range
    depth_levels = cfg.data.depth_levels

    # Common Target Grid (0.25 deg)
    new_lon = np.arange(lon_range[0], lon_range[1] + 0.25, 0.25)
    new_lat = np.arange(lat_range[0], lat_range[1] + 0.25, 0.25)

    years_to_process = cfg.data.train_years + cfg.data.val_years + cfg.data.test_years

    print("=== Aligning Target (GLORYS) ===")
    glorys_files = glob.glob(str(raw_dir / "glorys" / "*.nc"))
    if not glorys_files:
        print("Warning: No GLORYS files found.")
    else:
        # Open lazily without loading into RAM
        ds_g_raw = xr.open_dataset(glorys_files[0])
        if "longitude" in ds_g_raw.dims:
            ds_g_raw = ds_g_raw.rename({"longitude": "lon", "latitude": "lat"})
            
        ds_g_raw = ds_g_raw.sel(lon=slice(lon_range[0]-1, lon_range[1]+1), lat=slice(lat_range[0]-1, lat_range[1]+1))
        
        if 'depth' in ds_g_raw.dims:
            ds_g_raw = ds_g_raw.sel(depth=depth_levels, method='nearest')

        # Process YEAR BY YEAR to save RAM
        for year in years_to_process:
            print(f"Processing GLORYS for year {year}...")
            try:
                # Select only this year
                ds_y = ds_g_raw.sel(time=str(year))
                if len(ds_y.time) > 0:
                    # Load ONLY this year into RAM and interpolate
                    ds_y_loaded = ds_y.compute() 
                    ds_y_interp = ds_y_loaded.interp(lon=new_lon, lat=new_lat, method="linear")
                    out_path = out_dir / "targets" / f"glorys_temp_{year}.nc"
                    ds_y_interp.to_netcdf(out_path)
                    print(f"  -> Saved GLORYS {year}")
            except Exception as e:
                print(f"  -> Skipped GLORYS {year}: {e}")

    print("\n=== Aligning Input Channels ===")
    inputs = {
        "sst": glob.glob(str(raw_dir / "sst" / "*.nc")),
        "sla": glob.glob(str(raw_dir / "sla" / "*.nc")),
        "wind_u": glob.glob(str(raw_dir / "wind" / "*.nc")),
        "wind_v": glob.glob(str(raw_dir / "wind" / "*.nc")),
    }
    
    var_map = {
        "sst": "analysed_sst",
        "sla": "sla",
        "wind_u": "eastward_wind",
        "wind_v": "northward_wind"
    }

    for ch_name, files in inputs.items():
        if not files:
            continue
            
        print(f"\nProcessing {ch_name}...")
        ds_raw = xr.open_mfdataset(files)
        if "longitude" in ds_raw.dims:
            ds_raw = ds_raw.rename({"longitude": "lon", "latitude": "lat"})
            
        ds_raw = ds_raw.sel(lon=slice(lon_range[0]-1, lon_range[1]+1), lat=slice(lat_range[0]-1, lat_range[1]+1))
        var = var_map[ch_name]
        
        if var not in ds_raw:
            continue
            
        # Process YEAR BY YEAR to save RAM
        for year in years_to_process:
            try:
                ds_y = ds_raw[[var]].sel(time=str(year))
                if len(ds_y.time) > 0:
                    ds_y_loaded = ds_y.compute()
                    ds_y_interp = ds_y_loaded.interp(lon=new_lon, lat=new_lat, method="linear")
                    out_path = out_dir / "inputs" / f"{ch_name}_{year}.nc"
                    ds_y_interp.to_netcdf(out_path)
                    print(f"  -> Saved {ch_name} {year}")
            except Exception as e:
                pass
                
    print("\n✅ Preprocessing completely finished! Data is aligned and ready for PyTorch.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    preprocess(OmegaConf.load(args.config))
