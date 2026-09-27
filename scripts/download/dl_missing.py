#!/usr/bin/env python3
"""
Download Missing OceanEmbed Datasets (SST, SLA, Wind, Argo)
Syncs directly to Google Drive via rclone with automated scratch disk cleanup.
"""

import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Credentials & Configurations
CMEMS_USER = "akhan12345"
CMEMS_PASS = "Ak@!786AkTsaza"

LON_MIN, LON_MAX = "75.0", "100.0"
LAT_MIN, LAT_MAX = "5.0", "25.0"
START_TIME = "2017-01-01T00:00:00"
END_TIME = "2023-12-31T23:59:59"

GDRIVE_BASE = "gdrive:OceanEmbed/data/raw"

# Determine working directory: /kaggle/working/scratch if in Kaggle, else ./data/raw
if os.path.exists("/kaggle/working"):
    SCRATCH_BASE = Path("/kaggle/working/scratch")
    IS_KAGGLE = True
else:
    SCRATCH_BASE = Path("data/raw")
    IS_KAGGLE = False

SCRATCH_BASE.mkdir(parents=True, exist_ok=True)


def check_rclone() -> bool:
    return shutil.which("rclone") is not None


def ensure_copernicusmarine():
    try:
        import copernicusmarine  # noqa: F401
    except ImportError:
        print("Installing copernicusmarine library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "copernicusmarine"], check=True)


def upload_and_cleanup(local_path: Path, gdrive_target: str):
    if not local_path.exists():
        print(f"Error: Local file not found: {local_path}")
        return

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded: {local_path.name} ({size_mb:.2f} MB)")

    if check_rclone():
        print(f"Uploading to {gdrive_target} via rclone...")
        res = subprocess.run(["rclone", "copy", str(local_path), gdrive_target, "--progress"])
        if res.returncode == 0:
            print("Upload complete!")
            if IS_KAGGLE:
                local_path.unlink()
                print("Scratch file cleaned up to save disk space.")
        else:
            print(f"Warning: rclone upload failed with returncode {res.returncode}. Keeping local file.")
    else:
        print("rclone not found. File kept locally at:", local_path.resolve())


def download_sst():
    print("\n" + "=" * 60)
    print("Dataset 1: Sea Surface Temperature (SST)")
    print("=" * 60)

    folder = SCRATCH_BASE / "sst"
    folder.mkdir(parents=True, exist_ok=True)
    out_file = folder / "sst_2017_2023.nc"

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "--variable", "analysed_sst",
        "--minimum-longitude", LON_MIN,
        "--maximum-longitude", LON_MAX,
        "--minimum-latitude", LAT_MIN,
        "--maximum-latitude", LAT_MAX,
        "--start-datetime", START_TIME,
        "--end-datetime", END_TIME,
        "--output-directory", str(folder),
        "--output-filename", "sst_2017_2023.nc",
        "--username", CMEMS_USER,
        "--password", CMEMS_PASS,
        "--overwrite"
    ]

    print("Running Copernicus Marine subset for SST...")
    subprocess.run(cmd, check=True)
    upload_and_cleanup(out_file, f"{GDRIVE_BASE}/sst")


def download_sla():
    print("\n" + "=" * 60)
    print("Dataset 2: Sea Level Anomaly (SLA)")
    print("=" * 60)

    folder = SCRATCH_BASE / "sla"
    folder.mkdir(parents=True, exist_ok=True)
    out_file = folder / "sla_2017_2023.nc"

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "--variable", "sla",
        "--variable", "adt",
        "--minimum-longitude", LON_MIN,
        "--maximum-longitude", LON_MAX,
        "--minimum-latitude", LAT_MIN,
        "--maximum-latitude", LAT_MAX,
        "--start-datetime", START_TIME,
        "--end-datetime", END_TIME,
        "--output-directory", str(folder),
        "--output-filename", "sla_2017_2023.nc",
        "--username", CMEMS_USER,
        "--password", CMEMS_PASS,
        "--overwrite"
    ]

    print("Running Copernicus Marine subset for SLA...")
    subprocess.run(cmd, check=True)
    upload_and_cleanup(out_file, f"{GDRIVE_BASE}/sla")


def download_wind():
    print("\n" + "=" * 60)
    print("Datasets 3 & 4: Wind U and V Components")
    print("=" * 60)

    folder = SCRATCH_BASE / "wind"
    folder.mkdir(parents=True, exist_ok=True)
    out_file = folder / "wind_2017_2023.nc"

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id", "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        "--variable", "eastward_wind",
        "--variable", "northward_wind",
        "--minimum-longitude", LON_MIN,
        "--maximum-longitude", LON_MAX,
        "--minimum-latitude", LAT_MIN,
        "--maximum-latitude", LAT_MAX,
        "--start-datetime", START_TIME,
        "--end-datetime", END_TIME,
        "--output-directory", str(folder),
        "--output-filename", "wind_2017_2023.nc",
        "--username", CMEMS_USER,
        "--password", CMEMS_PASS,
        "--overwrite"
    ]

    print("Running Copernicus Marine subset for Wind...")
    subprocess.run(cmd, check=True)
    upload_and_cleanup(out_file, f"{GDRIVE_BASE}/wind")


def download_argo():
    print("\n" + "=" * 60)
    print("Dataset 21: Argo Float Profiles (2017-2023)")
    print("=" * 60)

    folder = SCRATCH_BASE / "argo"
    folder.mkdir(parents=True, exist_ok=True)

    base_erddap = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc"
    variables = "platform_number,time,latitude,longitude,pres,temp,psal,temp_qc,psal_qc,data_mode"

    for year in range(2017, 2024):
        fname = f"argo_bob_dm_{year}.nc"
        out_file = folder / fname
        url = (
            f"{base_erddap}?{variables}"
            f"&latitude%3E=5&latitude%3C=25"
            f"&longitude%3E=75&longitude%3C=100"
            f"&pres%3E=0&pres%3C=1500"
            f"&time%3E={year}-01-01T00:00:00Z"
            f"&time%3C={year}-12-31T23:59:59Z"
            f"&data_mode=%22D%22"
        )

        print(f"\nDownloading Argo Delayed-Mode profiles for {year}...")
        try:
            cmd = ["curl", "-L", "-s", "-o", str(out_file), url]
            subprocess.run(cmd, check=True)
            upload_and_cleanup(out_file, f"{GDRIVE_BASE}/argo")
        except Exception as err:
            print(f"Failed to download Argo for {year}: {err}")


def main():
    print("=" * 60)
    print("OceanEmbed - Download All Missing Datasets")
    print(f"Target Google Drive: {GDRIVE_BASE}")
    print("=" * 60)

    ensure_copernicusmarine()

    # Run in sequential order
    download_sst()
    download_sla()
    download_wind()
    download_argo()

    print("\n" + "=" * 60)
    print("All missing datasets downloaded and synced to Google Drive!")
    print("=" * 60)


if __name__ == "__main__":
    main()
