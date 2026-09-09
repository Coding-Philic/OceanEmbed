"""
OceanEmbed — Live Prediction Pipeline
=======================================
Fetches TODAY's real satellite data from Copernicus Marine Service
and runs it through the trained PhysVSANet model.

Usage:
    python scripts/live_predict.py
    python scripts/live_predict.py --date 2024-08-15
    python scripts/live_predict.py --output outputs/live_today.nc

Requirements:
    - Copernicus Marine account credentials in .env file
    - Trained model checkpoint at outputs/poc-bob-v1/checkpoints/
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

# ── Load .env credentials ──────────────────────────────────────────────────
def load_env(env_path=".env"):
    """Simple .env loader — no external library needed."""
    env_file = Path(env_path)
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

load_env()

USERNAME = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME", "")
PASSWORD = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD", "")

# ── Bay of Bengal bounds ───────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 8.0,  22.0
LON_MIN, LON_MAX = 80.0, 95.0


def check_credentials():
    """Verify that credentials are set."""
    if not USERNAME or not PASSWORD or PASSWORD == "YOUR_PASSWORD_HERE":
        print("❌ Credentials not found!")
        print()
        print("   Open the file:  .env")
        print("   Find the line:  COPERNICUSMARINE_SERVICE_PASSWORD=YOUR_PASSWORD_HERE")
        print("   Replace with:   COPERNICUSMARINE_SERVICE_PASSWORD=<your actual password>")
        print()
        print("   You can get your password at: https://data.marine.copernicus.eu")
        sys.exit(1)
    print(f"✅ Credentials loaded for user: {USERNAME}")


def _fetch_with_fallback(label: str, emoji: str, dataset_id: str, variables: list,
                         date_str: str, extra_kwargs: dict | None = None) -> xr.Dataset:
    """
    Try to fetch `date_str` from Copernicus. If the date is beyond the
    dataset's available range (CoordinatesOutOfDatasetBounds), automatically
    fall back up to MAX_FALLBACK_DAYS earlier dates.
    """
    import copernicusmarine as cm
    from copernicusmarine.core_functions.exceptions import CoordinatesOutOfDatasetBounds

    MAX_FALLBACK_DAYS = 5
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    kwargs = extra_kwargs or {}

    for offset in range(MAX_FALLBACK_DAYS + 1):
        try_date = (base_date - timedelta(days=offset)).strftime("%Y-%m-%d")
        suffix = f" (fallback: -{offset}d)" if offset > 0 else ""
        print(f"   {emoji} Fetching {label} for {try_date}{suffix}...")
        try:
            ds = cm.open_dataset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
                minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
                start_datetime=try_date, end_datetime=try_date,
                username=USERNAME, password=PASSWORD,
                **kwargs,
            )
            if offset > 0:
                print(f"   ⚠️  {label} dataset lagged by {offset} day(s). Using {try_date}.")
            return ds
        except CoordinatesOutOfDatasetBounds as e:
            if offset < MAX_FALLBACK_DAYS:
                print(f"   ↩️  {try_date} not yet available for {label}, trying previous day...")
            else:
                raise RuntimeError(
                    f"Could not fetch {label}: no data available within {MAX_FALLBACK_DAYS} days of {date_str}"
                ) from e


def fetch_sst(date_str: str) -> xr.Dataset:
    """Fetch Sea Surface Temperature (OSTIA L4) from Copernicus."""
    return _fetch_with_fallback(
        "SST", "🌡",
        dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        variables=["analysed_sst"],
        date_str=date_str,
    )


def fetch_sla(date_str: str) -> xr.Dataset:
    """Fetch Sea Level Anomaly (CMEMS altimetry) from Copernicus."""
    return _fetch_with_fallback(
        "SLA", "🌊",
        dataset_id="cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        variables=["sla"],
        date_str=date_str,
    )


def fetch_wind(date_str: str) -> xr.Dataset:
    """Fetch Wind U/V from Copernicus. This dataset is hourly and often lags by 1+ days."""
    return _fetch_with_fallback(
        "Wind", "💨",
        dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        variables=["eastward_wind", "northward_wind"],
        date_str=date_str,
    )


def preprocess_inputs(sst_ds: xr.Dataset, sla_ds: xr.Dataset, wind_ds: xr.Dataset, date_obj) -> np.ndarray:
    """Regrid and stack inputs to the target shape."""
    import math
    from scipy.interpolate import RegularGridInterpolator
    print("\n⚙️  Preprocessing inputs...")
    target_lat, target_lon = 57, 61
    lats_out = np.linspace(LAT_MIN, LAT_MAX, target_lat)
    lons_out = np.linspace(LON_MIN, LON_MAX, target_lon)

    def regrid(da: xr.DataArray) -> np.ndarray:
        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"

        lats_in = da[lat_name].values
        lons_in = da[lon_name].values
        vals    = da.values
        if vals.ndim > 2:
            vals = vals.squeeze()

        # Handle lat direction
        if lats_in[0] > lats_in[-1]:
            lats_in = lats_in[::-1]
            vals    = vals[::-1]

        interp = RegularGridInterpolator(
            (lats_in, lons_in), vals,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        lo, la = np.meshgrid(lons_out, lats_out)
        return interp(np.stack([la.ravel(), lo.ravel()], axis=1)).reshape(target_lat, target_lon)

    # SST: convert Kelvin → Celsius
    sst_var = "analysed_sst"
    sst = regrid(sst_ds[sst_var]) - 273.15

    # SLA
    sla_var = "sla" if "sla" in sla_ds else list(sla_ds.data_vars)[0]
    sla = regrid(sla_ds[sla_var])

    # Wind
    u_var = "eastward_wind"  if "eastward_wind"  in wind_ds else "u10"
    v_var = "northward_wind" if "northward_wind" in wind_ds else "v10"
    wind_u = regrid(wind_ds[u_var])
    wind_v = regrid(wind_ds[v_var])

    # Coordinate channels (must match training!)
    H, W = target_lat, target_lon
    lon_grid = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    lat_grid = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)

    doy = date_obj.timetuple().tm_yday
    angle = 2.0 * math.pi * doy / 365.25
    sin_doy = np.full((H, W), math.sin(angle), dtype=np.float32)
    cos_doy = np.full((H, W), math.cos(angle), dtype=np.float32)

    # Stack: [SST, SLA, WindU, WindV, lon_grid, lat_grid, sin_doy, cos_doy]
    sat_inputs = np.stack([sst, sla, wind_u, wind_v], axis=0)
    coords = np.stack([lon_grid, lat_grid, sin_doy, cos_doy], axis=0)
    inputs = np.concatenate([sat_inputs, coords], axis=0)
    return inputs.astype(np.float32)  # [8, 57, 61]


def run_model(inputs: np.ndarray, checkpoint_path: Path) -> np.ndarray:
    """Run PhysVSANet inference on the preprocessed inputs."""
    import torch
    import sys
    root_dir = checkpoint_path.parent.parent.parent.parent
    sys.path.append(str(root_dir))
    from src.oceanembed.inference.predict import OceanEmbedPredictor

    print(f"   🤖 Running model inference...")
    config_path = root_dir / "configs" / "poc_bob.yaml"
    stats_path = root_dir / "data" / "processed" / "normalization_stats.json"
    predictor = OceanEmbedPredictor(str(checkpoint_path), str(config_path), str(stats_path))

    # Fill NaNs with 0.0
    inputs = np.where(np.isfinite(inputs), inputs, 0.0)

    # Normalize satellite channels
    input_channels = ["sst", "sla", "wind_u", "wind_v"]
    for i, ch in enumerate(input_channels):
        inputs[i] = predictor.stats.normalize(ch, inputs[i])

    tensor = torch.tensor(inputs[np.newaxis]).float()
    with torch.no_grad():
        output = predictor.predict_single(tensor)  # [1, 6, 57, 61]
    return output.squeeze(0).numpy()  # [6, 57, 61]


def save_output(predictions: np.ndarray, raw_inputs: np.ndarray, date_str: str, output_path: Path):
    """Save the 3D prediction and 2D inputs to a NetCDF file."""
    lats = np.linspace(LAT_MIN, LAT_MAX, predictions.shape[1])
    lons = np.linspace(LON_MIN, LON_MAX, predictions.shape[2])

    ds = xr.Dataset(
        {
            "temperature": (["depth", "lat", "lon"], predictions),
            "sst": (["lat", "lon"], raw_inputs[0]),
            "sla": (["lat", "lon"], raw_inputs[1]),
            "wind_u": (["lat", "lon"], raw_inputs[2]),
            "wind_v": (["lat", "lon"], raw_inputs[3]),
        },
        coords={
            "depth": [0, 50, 100, 150, 200, 500],
            "lat": lats,
            "lon": lons,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    print(f"\n✅ Live prediction saved → {output_path}")
    print(f"   Temperature range: {float(predictions.min()):.2f}°C → {float(predictions.max()):.2f}°C")


def find_checkpoint() -> Path:
    """Auto-find the best checkpoint."""
    ckpt_dir = Path("outputs/poc-bob-v1/checkpoints")
    if not ckpt_dir.exists():
        print("❌ No checkpoint directory found at outputs/poc-bob-v1/checkpoints/")
        print("   Train the model first using Google Colab!")
        sys.exit(1)
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        print("❌ No checkpoint files found!")
        sys.exit(1)
    return ckpts[0]


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed Live Prediction")
    parser.add_argument("--date",   type=str, default=None,
                        help="Date to fetch (YYYY-MM-DD). Defaults to yesterday.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output NetCDF file path.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path.")
    args = parser.parse_args()

    if args.date is None:
        from datetime import datetime, timedelta, timezone
        date_obj = datetime.now(timezone.utc) - timedelta(days=1)
        date = date_obj.strftime("%Y-%m-%d")
    else:
        date = args.date
        from datetime import datetime
        date_obj = datetime.strptime(date, "%Y-%m-%d")

    output_path = Path(args.output) if args.output else Path(f"outputs/live_{date}.nc")
    ckpt_path   = Path(args.checkpoint) if args.checkpoint else find_checkpoint()

    print("=" * 60)
    print("  OceanEmbed — Live Prediction Pipeline")
    print("=" * 60)
    print(f"  Date:       {date}")
    print(f"  Region:     Bay of Bengal ({LAT_MIN}°–{LAT_MAX}°N, {LON_MIN}°–{LON_MAX}°E)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output:     {output_path}")
    print("=" * 60)

    # 1. Verify credentials
    check_credentials()

    # 2. Fetch real-time satellite data
    print("\n📡 Fetching real-time data from Copernicus Marine Service...")
    sst_ds  = fetch_sst(date)
    sla_ds  = fetch_sla(date)
    wind_ds = fetch_wind(date)

    # 3. Preprocess
    print("\n⚙️  Preprocessing inputs...")
    inputs = preprocess_inputs(sst_ds, sla_ds, wind_ds, date_obj)
    print(f"   Input shape: {inputs.shape}")
    raw_inputs = inputs.copy()  # Save un-normalized inputs for UI

    # 4. Run model
    print("\n🧠 Running AI model...")
    predictions = run_model(inputs, ckpt_path)
    print(f"   Output shape: {predictions.shape}")

    # 5. Save output
    save_output(predictions, raw_inputs, date, output_path)

    print("\n🎉 Live prediction complete!")
    print(f"   Open the web app and load: {output_path}")


if __name__ == "__main__":
    main()
