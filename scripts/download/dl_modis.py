#!/usr/bin/env python3
"""
Download: MODIS Aqua Level 3 SST (NASA Earthdata)
=================================================
Run this script directly:
    python3 scripts/download/dl_modis.py --freq monthly
    python3 scripts/download/dl_modis.py --freq daily

What gets downloaded:
    * NASA MODIS Aqua Level 3 Mapped SST (4km thermal IR)
    * Output -> data/raw/modis_sst/
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("data/raw/modis_sst")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NASA Earthdata credentials (environment variables or defaults)
USERNAME = os.environ.get("EARTHDATA_USERNAME", "akhan73766")
PASSWORD = os.environ.get("EARTHDATA_PASSWORD", "Ak@!123AkTsaza")

CONCEPTS = {
    "monthly": "C2036882228-POCLOUD",  # MODIS Aqua L3 SST Monthly 4km Daytime (~2 GB total)
    "daily": "C2036880650-POCLOUD",    # MODIS Aqua L3 SST Daily 4km Daytime
}


def download_modis(freq: str = "monthly"):
    try:
        import earthaccess
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "earthaccess"])
        import earthaccess

    os.environ["EARTHDATA_USERNAME"] = USERNAME
    os.environ["EARTHDATA_PASSWORD"] = PASSWORD

    print("=" * 60)
    print("OceanEmbed - MODIS Aqua SST Download")
    print(f"Frequency : {freq.capitalize()}")
    print("Account   :", USERNAME)
    print(f"Target    : {OUTPUT_DIR}/")
    print("=" * 60)

    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        print("[ERROR] Failed to authenticate with NASA Earthdata.")
        return False

    concept_id = CONCEPTS.get(freq, CONCEPTS["monthly"])
    print(f"\n--- Searching NASA CMR for {freq} granules (2017-2023) ---")

    results = earthaccess.search_data(
        concept_id=concept_id,
        temporal=("2017-01-01", "2023-12-31"),
    )

    # Filter out NRT files if searching daily
    if freq == "daily":
        results = [r for r in results if ".NRT." not in r.data_links()[0]]

    print(f"Found {len(results)} valid {freq} granules.")
    if not results:
        print("No granules found.")
        return False

    print("\n--- Downloading files ---")
    files = earthaccess.download(results, str(OUTPUT_DIR))
    print(f"\n[OK] Download complete. {len(files)} files saved to {OUTPUT_DIR}/")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NASA MODIS SST")
    parser.add_argument(
        "--freq",
        choices=["monthly", "daily"],
        default="monthly",
        help="Temporal frequency (monthly: ~2 GB, daily: ~58 GB)",
    )
    args = parser.parse_args()
    success = download_modis(freq=args.freq)
    if not success:
        sys.exit(1)
