"""
OceanEmbed — Automated Data Download Script
Bay of Bengal PoC: 2017–2023

Downloads all required satellite + reanalysis datasets from:
  - Copernicus Marine Service (CMEMS)
  - NASA Earthdata (SMAP SSS via earthaccess)

Usage:
    python scripts/download_data.py --config configs/poc_bob.yaml
    python scripts/download_data.py --config configs/poc_bob.yaml --skip-existing

Requirements:
    copernicusmarine login    (run once before this script)
    earthaccess.login()       (handled automatically via ~/.netrc or prompt)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


# ── Dataset definitions ──────────────────────────────────────────────────────

CMEMS_DATASETS = [
    {
        "name": "SST (OSTIA L4)",
        "dataset_id": "cmems_obs-sst_glo_phy_my_l4_P1D-m",
        "variable": "analysed_sst",
        "output_dir": "data/raw/sst",
    },
    {
        "name": "SLA (DUACS L4)",
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "variable": "sla",
        "output_dir": "data/raw/sla",
    },
    {
        "name": "Wind U (CERSAT Blended)",
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_P1D",
        "variable": "eastward_wind",
        "output_dir": "data/raw/wind",
    },
    {
        "name": "Wind V (CERSAT Blended)",
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_P1D",
        "variable": "northward_wind",
        "output_dir": "data/raw/wind",
    },
    {
        "name": "Currents U (GlobCurrent L4)",
        "dataset_id": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D",
        "variable": "u",
        "output_dir": "data/raw/currents",
    },
    {
        "name": "Currents V (GlobCurrent L4)",
        "dataset_id": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D",
        "variable": "v",
        "output_dir": "data/raw/currents",
    },
    {
        "name": "GLORYS12V1 Subsurface Temperature",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variable": "thetao",
        "output_dir": "data/raw/glorys",
    },
]


def run_cmems_download(
    dataset_id: str,
    variable: str,
    output_dir: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    start_date: str,
    end_date: str,
    skip_existing: bool = False,
) -> bool:
    """Run a copernicusmarine subset command and return True on success."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "copernicusmarine", "subset",
        "--dataset-id",          dataset_id,
        "--variable",            variable,
        "--minimum-longitude",   str(lon_min),
        "--maximum-longitude",   str(lon_max),
        "--minimum-latitude",    str(lat_min),
        "--maximum-latitude",    str(lat_max),
        "--start-datetime",      f"{start_date}T00:00:00",
        "--end-datetime",        f"{end_date}T23:59:59",
        "--output-directory",    str(out_path),
        "--force-download",
    ]

    if skip_existing:
        cmd.append("--skip-existing")

    print(f"\n  → Downloading {variable} from {dataset_id}")
    print(f"    Period: {start_date} → {end_date}")
    print(f"    Region: lon [{lon_min}, {lon_max}]  lat [{lat_min}, {lat_max}]")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def download_smap_sss(
    output_dir: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start_year: int,
    end_year: int,
) -> None:
    """Download SMAP L3 SSS via NASA earthaccess."""
    try:
        import earthaccess
    except ImportError:
        print("  ⚠ earthaccess not installed — skipping SMAP SSS download.")
        print("    Install with: pip install earthaccess")
        return

    print("\n  → Logging into NASA Earthdata …")
    earthaccess.login(strategy="interactive")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = earthaccess.search_data(
        short_name  = "SPL3SMP_E",   # SMAP Enhanced L3 Radiometer SSS
        temporal    = (f"{start_year}-01-01", f"{end_year}-12-31"),
        bounding_box= (lon_min, lat_min, lon_max, lat_max),
    )
    print(f"  Found {len(results)} SMAP granules. Downloading …")
    earthaccess.download(results, str(out_path))


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)

    lon_min, lon_max = cfg.data.lon_range
    lat_min, lat_max = cfg.data.lat_range

    all_years  = sorted(
        set(cfg.data.train_years) | set(cfg.data.val_years) | set(cfg.data.test_years)
    )
    start_date = f"{min(all_years)}-01-01"
    end_date   = f"{max(all_years)}-12-31"

    print("=" * 60)
    print("OceanEmbed — Data Download")
    print(f"  Region:  lon {lon_min}–{lon_max}  lat {lat_min}–{lat_max}")
    print(f"  Period:  {start_date} → {end_date}")
    print(f"  Config:  {args.config}")
    print("=" * 60)

    # ── CMEMS datasets ────────────────────────────────────────────────
    failures = []
    for ds in CMEMS_DATASETS:
        if args.only and ds["name"].lower().split()[0] not in args.only:
            continue
        ok = run_cmems_download(
            dataset_id   = ds["dataset_id"],
            variable     = ds["variable"],
            output_dir   = ds["output_dir"],
            lon_min      = lon_min,
            lon_max      = lon_max,
            lat_min      = lat_min,
            lat_max      = lat_max,
            start_date   = start_date,
            end_date     = end_date,
            skip_existing= args.skip_existing,
        )
        if not ok:
            failures.append(ds["name"])
            print(f"  ❌ Failed: {ds['name']}")
        else:
            print(f"  ✅ Done: {ds['name']}")

    # ── SMAP SSS ─────────────────────────────────────────────────────
    if not args.only or "sss" in args.only:
        download_smap_sss(
            output_dir = "data/raw/sss",
            lat_min    = lat_min,
            lat_max    = lat_max,
            lon_min    = lon_min,
            lon_max    = lon_max,
            start_year = min(all_years),
            end_year   = max(all_years),
        )

    print("\n" + "=" * 60)
    if failures:
        print(f"⚠  Completed with {len(failures)} failure(s): {failures}")
        print("   Re-run with --skip-existing to retry only missing files.")
    else:
        print("✅  All datasets downloaded successfully!")
    print(f"📁  Data saved to:  {Path('data/raw').resolve()}")
    print("\nNext step:")
    print("  python scripts/preprocess.py --config configs/poc_bob.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OceanEmbed training data")
    parser.add_argument("--config",        required=True,      help="YAML config path")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already downloaded files")
    parser.add_argument("--only",          nargs="*",           help="Download only: sst sla wind currents glorys sss")
    main(parser.parse_args())
