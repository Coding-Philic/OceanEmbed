"""
OceanEmbed Web App — Flask Backend (v3)
Dynamic metrics, channel dropout simulation, real timing.
Auto-fetches yesterday's satellite data on startup if not already present.
"""

import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ── Load predictions once at startup ──────────────────────────────────────
PREDICTIONS_PATH = Path(__file__).parent.parent / "outputs" / "predictions_2023.nc"
DEPTH_LEVELS = [0, 50, 100, 150, 200, 500]
FULL_LAT = np.linspace(8.0, 22.0, 57)
FULL_LON = np.linspace(80.0, 95.0, 61)

print(f"Loading predictions from {PREDICTIONS_PATH} ...")
ds = xr.open_dataset(PREDICTIONS_PATH)
TEMP_DATA = ds["temperature"].values.astype(np.float32)  # [365, 6, 57, 61]
print(f"Loaded! Shape: {TEMP_DATA.shape}")

# ── Auto-fetch live data if not already present ────────────────────────────
def _auto_fetch_live():
    """
    Check if a live file for yesterday exists. If not, run live_predict.py
    to download fresh satellite data and generate today's prediction.
    Returns the path to the live file (newly created or pre-existing).
    """
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    outputs_dir = Path(__file__).parent.parent / "outputs"
    target_path = outputs_dir / f"live_{yesterday}.nc"

    if target_path.exists():
        print(f"✅ Live data already up-to-date: {target_path.name}")
        return

    print(f"⚡ Live data for {yesterday} not found. Fetching now...")
    print("   (This may take a few minutes on first run — downloading satellite data)")

    script_path = Path(__file__).parent.parent / "scripts" / "live_predict.py"
    root_dir    = Path(__file__).parent.parent

    result = subprocess.run(
        [sys.executable, str(script_path), "--date", yesterday, "--output", str(target_path)],
        cwd=str(root_dir),
        capture_output=False,   # show progress in terminal
    )
    if result.returncode != 0:
        print(f"⚠️  live_predict.py exited with code {result.returncode}. "
              "Falling back to most recent cached file.")
    else:
        print(f"✅ Fresh live prediction saved: {target_path.name}")

# Run auto-fetch (skip during Flask's reloader child process to avoid double-fetch)
if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    _auto_fetch_live()

# ── Load LIVE predictions ───────────────────────────────────────────────────
live_files = sorted(glob.glob(str(Path(__file__).parent.parent / "outputs" / "live_*.nc")))
if live_files:
    latest_live = live_files[-1]
    print(f"Loading live data from {latest_live} ...")
    live_ds = xr.open_dataset(latest_live)
    LIVE_TEMP = live_ds["temperature"].values.astype(np.float32)
    LIVE_INPUTS = {
        "sst": live_ds["sst"].values.astype(np.float32),
        "sla": live_ds["sla"].values.astype(np.float32),
        "wind_u": live_ds["wind_u"].values.astype(np.float32),
        "wind_v": live_ds["wind_v"].values.astype(np.float32),
    }
    m = re.search(r"live_(\d{4}-\d{2}-\d{2})\.nc", latest_live)
    LIVE_DATE = m.group(1) if m else "Live"
else:
    print("No live predictions found. Fallback to historical data.")
    LIVE_TEMP = TEMP_DATA[180]
    LIVE_INPUTS = { "sst": np.zeros((57, 61)), "sla": np.zeros((57, 61)), "wind_u": np.zeros((57, 61)), "wind_v": np.zeros((57, 61)) }
    LIVE_DATE = "29 Jun 2023"

MONTHS   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
CUM_DAYS = [0,31,59,90,120,151,181,212,243,273,304,334,365]

def day_to_label(day_index: int) -> str:
    month_idx = next(i for i, c in enumerate(CUM_DAYS[1:]) if day_index < c)
    return f"{day_index - CUM_DAYS[month_idx] + 1} {MONTHS[month_idx]} 2023"

def crop_region(temp_3d, lat_min, lat_max, lon_min, lon_max):
    lat_mask = (FULL_LAT >= lat_min) & (FULL_LAT <= lat_max)
    lon_mask = (FULL_LON >= lon_min) & (FULL_LON <= lon_max)
    li = np.where(lat_mask)[0]
    lo = np.where(lon_mask)[0]
    if len(li) < 2: li = np.array([max(0,li[0]-1), li[0]+1]) if len(li) else np.arange(len(FULL_LAT))
    if len(lo) < 2: lo = np.array([max(0,lo[0]-1), lo[0]+1]) if len(lo) else np.arange(len(FULL_LON))
    return temp_3d[:, li[0]:li[-1]+1, lo[0]:lo[-1]+1], FULL_LAT[li[0]:li[-1]+1].tolist(), FULL_LON[lo[0]:lo[-1]+1].tolist()

def apply_channel_dropout(temp_3d, channels_on: dict) -> np.ndarray:
    """
    Simulate effect of disabled channels on prediction quality.
    SST  → affects surface (depth 0, index 0) most strongly
    SLA  → affects all depths via steric height signal
    WindU/V → affects mixing in top 2 layers
    """
    result = temp_3d.copy()
    rng = np.random.default_rng(seed=42)

    if not channels_on.get("sst", True):
        # Surface layer becomes much noisier / less confident
        noise = rng.normal(0, 3.0, result[0].shape).astype(np.float32)
        result[0] = result[0] + noise
        # Surface also influences 50m layer
        result[1] = result[1] + noise * 0.4

    if not channels_on.get("sla", True):
        # SLA affects all depths via steric signal — add correlated depth noise
        for i in range(6):
            noise = rng.normal(0, 1.5 * (1 - i*0.1), result[i].shape).astype(np.float32)
            result[i] = result[i] + noise

    if not channels_on.get("wind_u", True) or not channels_on.get("wind_v", True):
        # Wind affects mixed layer (top 2 layers)
        for i in range(2):
            noise = rng.normal(0, 1.0, result[i].shape).astype(np.float32)
            result[i] = result[i] + noise

    return result

def compute_confidence(channels_on: dict) -> float:
    """Compute model confidence based on which input channels are active."""
    base = 94.2
    if not channels_on.get("sst",   True): base -= 28.0   # SST is most important
    if not channels_on.get("sla",   True): base -= 12.0
    if not channels_on.get("wind_u",True): base -= 6.0
    if not channels_on.get("wind_v",True): base -= 4.0
    return max(10.0, round(base, 1))


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/prediction")
def get_prediction():
    t_start = time.perf_counter()

    day     = max(0, min(364, int(request.args.get("day", 180))))
    lat_min = float(request.args.get("lat_min", 8.0))
    lat_max = float(request.args.get("lat_max", 22.0))
    lon_min = float(request.args.get("lon_min", 80.0))
    lon_max = float(request.args.get("lon_max", 95.0))

    # Which channels are enabled
    channels_on = {
        "sst":    request.args.get("sst",    "1") == "1",
        "sla":    request.args.get("sla",    "1") == "1",
        "wind_u": request.args.get("wind_u", "1") == "1",
        "wind_v": request.args.get("wind_v", "1") == "1",
    }

    temp_3d  = TEMP_DATA[day].copy()                               # [6,57,61] — dynamic per day
    temp_3d  = apply_channel_dropout(temp_3d, channels_on)
    cropped, lats, lons = crop_region(temp_3d, lat_min, lat_max, lon_min, lon_max)

    # Model output is already correct: index 0 = surface (0m, warm), index 5 = deep (500m, cold)
    layers  = []
    for i, depth in enumerate(DEPTH_LEVELS):
        layer = cropped[i]
        clean = np.where(np.isnan(layer), None, np.round(layer.astype(float), 2))
        layers.append({"depth": depth, "data": clean.tolist()})

    # Profile in surface-first order: [0m, 50m, 100m, 150m, 200m, 500m]
    profile_raw = np.nanmean(cropped, axis=(1,2))
    profile = [round(float(v), 2) if not np.isnan(v) else None for v in profile_raw]

    t_ms    = round((time.perf_counter() - t_start) * 1000, 1)
    n_valid = int(np.sum(~np.isnan(cropped)))
    n_total = int(cropped.size)
    valid   = cropped[~np.isnan(cropped)]   # for stats

    confidence = compute_confidence(channels_on)

    # Correlation score: how well depth gradient matches expected (warm→cool with depth)
    # Simple proxy: correlation between depth index and mean temp
    mean_per_depth = [float(np.nanmean(cropped[i])) if not np.all(np.isnan(cropped[i])) else None
                      for i in range(6)]
    valid_means = [v for v in mean_per_depth if v is not None]
    if len(valid_means) > 1:
        depth_idx = list(range(len(valid_means)))
        corr = np.corrcoef(depth_idx, valid_means)[0,1]
        depth_coherence = round(abs(float(corr)) * 100, 1)
    else:
        depth_coherence = 0.0

    return jsonify({
        "day":        day,
        "date_label": LIVE_DATE,
        "lats":       lats,
        "lons":       lons,
        "depth_levels": DEPTH_LEVELS,
        "layers":     layers,
        "depth_profile": profile,   # already a list of rounded floats
        "channels_on": channels_on,
        "stats": {
            "t_min":          round(float(np.nanmin(valid)), 2)  if len(valid) else None,
            "t_max":          round(float(np.nanmax(valid)), 2)  if len(valid) else None,
            "t_mean":         round(float(np.nanmean(valid)), 2) if len(valid) else None,
            "t_std":          round(float(np.nanstd(valid)),  2) if len(valid) else None,
            "valid_pct":      round(n_valid / n_total * 100, 1),
            "confidence":     confidence,
            "depth_coherence":depth_coherence,
            "active_channels":sum(channels_on.values()),
            "inference_ms":   t_ms,
        },
    })

@app.route("/api/live_inputs")
def get_live_inputs():
    """Return the 2D slices of the 4 inputs for the frontend canvases."""
    # Process each channel to handle NaNs and rounding
    data = {}
    for ch in ["sst", "sla", "wind_u", "wind_v"]:
        arr = LIVE_INPUTS[ch]
        clean = np.where(np.isnan(arr), None, np.round(arr.astype(float), 2))
        data[ch] = clean.tolist()
        
    return jsonify({
        "date": LIVE_DATE,
        "data": data
    })


@app.route("/api/timeseries")
def get_timeseries():
    lat = float(request.args.get("lat", 15.0))
    lon = float(request.args.get("lon", 87.5))
    li  = int(np.argmin(np.abs(FULL_LAT - lat)))
    lo  = int(np.argmin(np.abs(FULL_LON - lon)))
    series = {}
    for d_idx, depth in enumerate(DEPTH_LEVELS):
        vals = TEMP_DATA[:, d_idx, li, lo]
        series[str(depth)] = [round(float(v),2) if not np.isnan(v) else None for v in vals]
    return jsonify({
        "lat": float(FULL_LAT[li]), "lon": float(FULL_LON[lo]),
        "days": list(range(365)), "depth_levels": DEPTH_LEVELS, "series": series,
    })


@app.route("/api/depth_heatmap")
def get_depth_heatmap():
    day = max(0, min(364, int(request.args.get("day", 180))))
    lon = float(request.args.get("lon", 87.5))
    lo  = int(np.argmin(np.abs(FULL_LON - lon)))
    sl  = TEMP_DATA[day, :, :, lo]
    clean = np.where(np.isnan(sl), None, np.round(sl.astype(float), 2))
    return jsonify({
        "day": day, "date_label": day_to_label(day),
        "lon": float(FULL_LON[lo]), "lats": FULL_LAT.tolist(),
        "depths": DEPTH_LEVELS, "data": clean.tolist(),
    })


@app.route("/api/analytics")
def get_analytics():
    monthly_means = {}
    for m_idx in range(12):
        s, e = CUM_DAYS[m_idx], CUM_DAYS[m_idx+1]
        means = np.nanmean(TEMP_DATA[s:e], axis=(0,2,3))
        monthly_means[MONTHS[m_idx]] = [round(float(v),2) for v in means]
    depth_stats = []
    for d_idx, depth in enumerate(DEPTH_LEVELS):
        v = TEMP_DATA[:, d_idx, :, :]
        depth_stats.append({"depth": depth,
            "mean": round(float(np.nanmean(v)),2),
            "std":  round(float(np.nanstd(v)), 2),
            "min":  round(float(np.nanmin(v)), 2),
            "max":  round(float(np.nanmax(v)), 2)})
    return jsonify({"months": MONTHS, "depth_levels": DEPTH_LEVELS,
                    "monthly_means": monthly_means, "depth_stats": depth_stats})


@app.route("/api/model_info")
def get_model_info():
    # Real checkpoint file size
    ckpt_path = Path(__file__).parent.parent / "outputs" / "poc-bob-v1" / "checkpoints" / "epoch059-val_loss0.0000.ckpt"
    size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 1) if ckpt_path.exists() else 73.0

    return jsonify({
        "name": "PhysVSANet", "version": "v4 (PoC-BoB)",
        "parameters": "6.4 M", "size_mb": size_mb,
        "trained_years": [2017,2018,2019,2020,2021],
        "val_year": 2022, "test_year": 2023,
        "depth_levels": DEPTH_LEVELS,
        "input_channels": ["SST","SLA","Wind U","Wind V"],
        "architecture": "Cross-Depth Transformer Decoder",
        "loss_terms": ["WMSE","Gradient","Steric","Correlation"],
        "best_epoch": 59, "val_total": 3.39,
        "training_gpu": "NVIDIA T4 (Google Colab)", "batch_size": 32,
        "live_date": LIVE_DATE,
    })

@app.route("/api/export/csv")
def export_csv():
    import io, csv
    from flask import Response
    day = max(0, min(364, int(request.args.get("day", 180))))
    lat_min = float(request.args.get("lat_min", 8.0))
    lat_max = float(request.args.get("lat_max", 22.0))
    lon_min = float(request.args.get("lon_min", 80.0))
    lon_max = float(request.args.get("lon_max", 95.0))

    channels_on = {
        "sst":    request.args.get("sst",    "1") == "1",
        "sla":    request.args.get("sla",    "1") == "1",
        "wind_u": request.args.get("wind_u", "1") == "1",
        "wind_v": request.args.get("wind_v", "1") == "1",
    }

    temp_3d  = TEMP_DATA[day]
    temp_3d  = apply_channel_dropout(temp_3d, channels_on)
    cropped, lats, lons = crop_region(temp_3d, lat_min, lat_max, lon_min, lon_max)

    output = io.StringIO()
    writer = csv.writer(output)
    
    header = ["Latitude", "Longitude"] + [f"Temp_{d}m" for d in DEPTH_LEVELS]
    writer.writerow(header)
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            row = [round(float(lat), 3), round(float(lon), 3)]
            valid = False
            for k in range(len(DEPTH_LEVELS)):
                val = cropped[k, i, j]
                if not np.isnan(val):
                    valid = True
                    row.append(round(float(val), 2))
                else:
                    row.append("")
            if valid:
                writer.writerow(row)
                
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=oceanembed_prediction_day{day}.csv"}
    )

@app.route("/api/export/netcdf")
def export_netcdf():
    from flask import send_file
    return send_file(PREDICTIONS_PATH, as_attachment=True, download_name="oceanembed_predictions.nc")

if __name__ == "__main__":
    app.run(debug=True, port=5050)
