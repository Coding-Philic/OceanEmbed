#!/usr/bin/env python3
"""
OceanEmbed v2 — Unified Data Download CLI
==========================================
Downloads ALL 22+ input channels described in docs/NEXT_VERSION_PLAN.md.

Supported sources
─────────────────
  CMEMS   – SST, SLA, Wind, Heat Flux, SLP, Currents, Salinity (GLORYS),
             MLD, OHC, Chlorophyll-a, Kd490
  NASA    – GPM IMERG precipitation  (via earthaccess)
  NASA    – SMAP L3 SSS              (via earthaccess)
  GEBCO   – Bathymetry 2023          (HTTP direct download)
  IHFC    – Geothermal heat flux     (HTTP direct download)
  GloFAS  – River discharge          (via cdsapi)

Prerequisites
─────────────
  pip install copernicusmarine earthaccess omegaconf requests tqdm cdsapi
  copernicusmarine login          ← run once; caches credentials
  # NASA Earthdata: set ~/.netrc  OR be prompted on first run
  # CDS API: create ~/.cdsapirc   — see https://cds.climate.copernicus.eu/api-how-to

Usage
─────
  # Download everything (2010-2021, Bay of Bengal default region)
  python scripts/download_v2.py

  # Download only specific sources
  python scripts/download_v2.py --only sst sla mld sss

  # Custom date range and region
  python scripts/download_v2.py --start 2015-01-01 --end 2023-12-31 \\
      --lon-min 75 --lon-max 100 --lat-min 5 --lat-max 25

  # Skip already downloaded files
  python scripts/download_v2.py --skip-existing

  # List all available source keys
  python scripts/download_v2.py --list

  # Dry run — show what would be downloaded without doing it
  python scripts/download_v2.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

# ── Terminal colours ─────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
DIM    = "\033[2m"


def _c(text: object, *codes: str) -> str:
    """Wrap text with ANSI codes (no-op if not a TTY)."""
    if not sys.stdout.isatty():
        return str(text)
    return "".join(codes) + str(text) + RESET


# ── Default region: Bay of Bengal ────────────────────────────────────────────

DEFAULT_LON_MIN   = 75.0
DEFAULT_LON_MAX   = 100.0
DEFAULT_LAT_MIN   = 5.0
DEFAULT_LAT_MAX   = 25.0
DEFAULT_START     = "2017-01-01"
DEFAULT_END       = "2023-12-31"
DEFAULT_DEPTH_MIN = 0.49
DEFAULT_DEPTH_MAX = 1500.0   # GLORYS goes this deep


# ── CMEMS dataset catalogue ───────────────────────────────────────────────────
#
# Each entry:
#   key         – short name used with --only
#   name        – human-readable label
#   dataset_id  – Copernicus Marine dataset ID
#   variables   – list of netCDF variable names to pull
#   output_dir  – relative output path
#   depth       – whether to request depth range (for 3-D products)
#   tier        – from the v2 plan (for display)

CMEMS_DATASETS: list[dict] = [
    # ── Tier 1: Core surface ─────────────────────────────────────────
    {
        "key": "sst",
        "name": "SST — OSTIA L4",
        "dataset_id": "cmems_obs-sst_glo_phy_my_l4_P1D-m",
        "variables": ["analysed_sst"],
        "output_dir": "data/raw/sst",
        "depth": False,
        "tier": 1,
    },
    {
        "key": "sla",
        "name": "SLA — DUACS L4 (+ ADT)",
        "dataset_id": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "variables": ["sla", "adt"],
        "output_dir": "data/raw/sla",
        "depth": False,
        "tier": 1,
    },
    {
        "key": "wind",
        "name": "Wind U/V — CERSAT Blended L4",
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_P1D",
        "variables": ["eastward_wind", "northward_wind"],
        "output_dir": "data/raw/wind",
        "depth": False,
        "tier": 1,
    },
    # ── Tier 2: Atmosphere-Ocean Coupling ────────────────────────────
    {
        "key": "heatflux",
        "name": "Heat Flux (SWR/LWR/LHF/SHF) — ERA5 via CMEMS",
        "dataset_id": "reanalysis-era5-single-levels",
        "variables": [
            "surface_net_solar_radiation",
            "surface_net_thermal_radiation",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
        ],
        "output_dir": "data/raw/heatflux",
        "depth": False,
        "tier": 2,
        "note": "ERA5 via CDS API — if this CMEMS dataset_id fails, use cdsapi directly",
    },
    {
        "key": "slp",
        "name": "Sea Level Pressure — ERA5 via CMEMS",
        "dataset_id": "reanalysis-era5-single-levels",
        "variables": ["mean_sea_level_pressure"],
        "output_dir": "data/raw/slp",
        "depth": False,
        "tier": 2,
        "note": "ERA5 via CDS API",
    },
    # ── Tier 3: Ocean State ──────────────────────────────────────────
    {
        "key": "currents",
        "name": "Surface Currents U/V — GlobCurrent L4",
        "dataset_id": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D",
        "variables": ["u", "v"],
        "output_dir": "data/raw/currents",
        "depth": False,
        "tier": 3,
    },
    {
        "key": "glorys",
        "name": "GLORYS12 — Temperature / Salinity / MLD / SSH (3-D)",
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao", "so", "mlotst", "zos"],
        "output_dir": "data/raw/glorys",
        "depth": True,
        "tier": 3,
    },
    # ── Tier 4: Biogeochemical ───────────────────────────────────────
    {
        "key": "chl",
        "name": "Chlorophyll-a — GlobColour L4 Multi-sensor",
        "dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1D",
        "variables": ["CHL"],
        "output_dir": "data/raw/chl",
        "depth": False,
        "tier": 4,
    },
    {
        "key": "kd490",
        "name": "Diffuse Attenuation Kd490 — GlobColour L4",
        "dataset_id": "cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1D",
        "variables": ["KD490"],
        "output_dir": "data/raw/kd490",
        "depth": False,
        "tier": 4,
    },
]

# Keys handled separately (non-CMEMS)
SPECIAL_KEYS = ["sss", "precip", "bathymetry", "geothermal", "glofas"]

ALL_KEYS = [ds["key"] for ds in CMEMS_DATASETS] + SPECIAL_KEYS


# ── Helper: CMEMS subset ─────────────────────────────────────────────────────

def _cmems_subset(
    *,
    dataset_id: str,
    variables: list[str],
    output_dir: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    start: str,
    end: str,
    depth: bool = False,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> bool:
    """Build & run a `copernicusmarine subset` command. Returns True on success."""
    out_path = Path(output_dir)
    if not dry_run:
        out_path.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        "copernicusmarine", "subset",
        "--dataset-id",        dataset_id,
        "--minimum-longitude", str(lon_min),
        "--maximum-longitude", str(lon_max),
        "--minimum-latitude",  str(lat_min),
        "--maximum-latitude",  str(lat_max),
        "--start-datetime",    f"{start}T00:00:00",
        "--end-datetime",      f"{end}T23:59:59",
        "--output-directory",  str(out_path),
        "--force-download",
    ]

    for var in variables:
        cmd += ["--variable", var]

    if depth:
        cmd += [
            "--minimum-depth", str(DEFAULT_DEPTH_MIN),
            "--maximum-depth", str(DEFAULT_DEPTH_MAX),
        ]

    if skip_existing:
        cmd.append("--skip-existing")

    if dry_run:
        print(_c("    [DRY RUN] " + " ".join(cmd), DIM))
        return True

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ── Special downloaders ───────────────────────────────────────────────────────

def _download_smap_sss(
    *,
    output_dir: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start: str,
    end: str,
    dry_run: bool = False,
) -> bool:
    """SMAP Enhanced L3 SSS via NASA earthaccess (SPL3SMP_E, 9 km, 8-day)."""
    try:
        import earthaccess
    except ImportError:
        print(_c("  ⚠  earthaccess not installed. Run: pip install earthaccess", YELLOW))
        return False

    if dry_run:
        print(_c(f"    [DRY RUN] earthaccess.search_data(SPL3SMP_E, {start}→{end})", DIM))
        return True

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(_c("  → Authenticating with NASA Earthdata …", CYAN))
    earthaccess.login(strategy="netrc")   # falls back to interactive prompt

    results = earthaccess.search_data(
        short_name   = "SPL3SMP_E",
        temporal     = (start, end),
        bounding_box = (lon_min, lat_min, lon_max, lat_max),
    )
    print(f"  Found {_c(len(results), BOLD)} SMAP SSS granules. Downloading …")
    earthaccess.download(results, str(out_path))
    return True


def _download_gpm_imerg(
    *,
    output_dir: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    start: str,
    end: str,
    dry_run: bool = False,
) -> bool:
    """GPM IMERG Final Run Daily L3 0.1° via NASA earthaccess (GPM_3IMERGDF)."""
    try:
        import earthaccess
    except ImportError:
        print(_c("  ⚠  earthaccess not installed. Run: pip install earthaccess", YELLOW))
        return False

    if dry_run:
        print(_c(f"    [DRY RUN] earthaccess.search_data(GPM_3IMERGDF, {start}→{end})", DIM))
        return True

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(_c("  → Authenticating with NASA Earthdata …", CYAN))
    earthaccess.login(strategy="netrc")

    results = earthaccess.search_data(
        short_name   = "GPM_3IMERGDF",
        temporal     = (start, end),
        bounding_box = (lon_min, lat_min, lon_max, lat_max),
    )
    print(f"  Found {_c(len(results), BOLD)} GPM IMERG granules. Downloading …")
    earthaccess.download(results, str(out_path))
    return True


def _download_gebco(*, output_dir: str, dry_run: bool = False) -> bool:
    """
    GEBCO 2023 global bathymetry — downloads the full netCDF zip from BODC (~7 GB).
    Trim to BoB during preprocessing with xarray/nco.
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print(_c("  ⚠  requests/tqdm not installed. Run: pip install requests tqdm", YELLOW))
        return False

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    url      = "https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023/zip/"
    out_file = out_path / "gebco_2023.zip"

    if out_file.exists():
        print(_c(f"  ✓ GEBCO already present: {out_file}", DIM))
        return True

    if dry_run:
        print(_c(f"    [DRY RUN] wget {url} → {out_file}", DIM))
        return True

    print(f"  → Downloading GEBCO 2023 from BODC ({_c('~7 GB', YELLOW)}) …")
    print(_c("    This will take a while on a slow connection.", DIM))

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_file, "wb") as f, tqdm(
            desc="GEBCO",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    print(_c(f"  ✅ GEBCO saved → {out_file}", GREEN))
    print(_c("     Unzip with: unzip data/raw/bathymetry/gebco_2023.zip -d data/raw/bathymetry/", DIM))
    return True


def _download_geothermal(*, output_dir: str, dry_run: bool = False) -> bool:
    """
    Global Heat Flow Database (IHFC) — interpolated 1° grid.
    Note: IHFC may require registration. Falls back with instructions.
    """
    try:
        import requests
    except ImportError:
        print(_c("  ⚠  requests not installed. Run: pip install requests", YELLOW))
        return False

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    url      = "https://ihfc-iugg.org/products/global-heat-flow-database/download/"
    out_file = out_path / "global_heatflow_2024.csv"

    if out_file.exists():
        print(_c(f"  ✓ Geothermal data already present: {out_file}", DIM))
        return True

    if dry_run:
        print(_c(f"    [DRY RUN] Download geothermal CSV from IHFC → {out_file}", DIM))
        return True

    print("  → Downloading IHFC Global Heat Flow Database …")
    print(_c("  ⚠  If auto-download fails, get it manually from:", YELLOW))
    print(_c("     https://ihfc-iugg.org/products/global-heat-flow-database/", YELLOW))

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(_c(f"  ✅ Geothermal data saved → {out_file}", GREEN))
        return True
    except Exception as e:
        print(_c(f"  ❌ Auto-download failed: {e}", RED))
        print(_c("     Download manually: https://ihfc-iugg.org", YELLOW))
        return False


def _download_glofas(
    *,
    output_dir: str,
    start: str,
    end: str,
    dry_run: bool = False,
) -> bool:
    """
    GloFAS river discharge via Copernicus Emergency Management CDS API.
    Requires: pip install cdsapi  and ~/.cdsapirc key file.
    """
    try:
        import cdsapi
    except ImportError:
        print(_c("  ⚠  cdsapi not installed. Run: pip install cdsapi", YELLOW))
        print(_c("     Also create ~/.cdsapirc — see https://cds.climate.copernicus.eu/api-how-to", YELLOW))
        return False

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_year = int(start[:4])
    end_year   = int(end[:4])
    years      = [str(y) for y in range(start_year, end_year + 1)]
    out_file   = out_path / f"glofas_discharge_{start_year}_{end_year}.nc"

    if out_file.exists():
        print(_c(f"  ✓ GloFAS data already present: {out_file}", DIM))
        return True

    if dry_run:
        print(_c(f"    [DRY RUN] cdsapi GloFAS discharge {years[0]}–{years[-1]} → {out_file}", DIM))
        return True

    print(f"  → Requesting GloFAS discharge for {years[0]}–{years[-1]} …")

    c = cdsapi.Client()
    c.retrieve(
        "cems-glofas-historical",
        {
            "system_version":    "version_4_0",
            "variable":          "river_discharge_in_the_last_24_hours",
            "format":            "netcdf",
            "hydrological_model": "lisflood",
            "product_type":      "consolidated",
            "year":              years,
            "month":             [f"{m:02d}" for m in range(1, 13)],
            "day":               [f"{d:02d}" for d in range(1, 32)],
        },
        str(out_file),
    )
    print(_c(f"  ✅ GloFAS saved → {out_file}", GREEN))
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

def _print_banner(args: argparse.Namespace) -> None:
    print()
    print(_c("╔══════════════════════════════════════════════════════════════╗", BLUE))
    print(_c("║         OceanEmbed v2  —  Data Download CLI                  ║", BLUE + BOLD))
    print(_c("╚══════════════════════════════════════════════════════════════╝", BLUE))
    print()
    print(f"  {_c('Region :', BOLD)} lon [{args.lon_min}, {args.lon_max}]  lat [{args.lat_min}, {args.lat_max}]")
    print(f"  {_c('Period :', BOLD)} {args.start} → {args.end}")
    if args.only:
        print(f"  {_c('Filter :', BOLD)} {', '.join(args.only)}")
    if args.dry_run:
        print(f"  {_c('Mode   :', BOLD)} {_c('DRY RUN — nothing will be downloaded', YELLOW)}")
    if args.skip_existing:
        print(f"  {_c('Option :', BOLD)} skip existing files")
    print()


def _print_source_list() -> None:
    print()
    print(_c("  Available --only keys:", BOLD))
    print()
    for ds in CMEMS_DATASETS:
        tier_label = f"Tier {ds['tier']}"
        print(f"  {_c(ds['key'].ljust(12), CYAN)}  {tier_label}   {ds['name']}")
    labels = {
        "sss":        "NASA SMAP  — Sea Surface Salinity (8-day L3)",
        "precip":     "NASA GPM   — IMERG Daily Precipitation (0.1°)",
        "bathymetry": "GEBCO 2023 — Static Bathymetry (~7 GB zip)",
        "geothermal": "IHFC       — Global Geothermal Heat Flow",
        "glofas":     "Copernicus — GloFAS River Discharge (via cdsapi)",
    }
    for key in SPECIAL_KEYS:
        print(f"  {_c(key.ljust(12), CYAN)}  Static    {labels[key]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_v2.py",
        description=textwrap.dedent("""\
            OceanEmbed v2 — Download all 22+ input channels.
            Run --list to see all available source keys.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Region
    parser.add_argument("--lon-min",  type=float, default=DEFAULT_LON_MIN,
                        help=f"Minimum longitude  (default: {DEFAULT_LON_MIN})")
    parser.add_argument("--lon-max",  type=float, default=DEFAULT_LON_MAX,
                        help=f"Maximum longitude  (default: {DEFAULT_LON_MAX})")
    parser.add_argument("--lat-min",  type=float, default=DEFAULT_LAT_MIN,
                        help=f"Minimum latitude   (default: {DEFAULT_LAT_MIN})")
    parser.add_argument("--lat-max",  type=float, default=DEFAULT_LAT_MAX,
                        help=f"Maximum latitude   (default: {DEFAULT_LAT_MAX})")

    # Time range
    parser.add_argument("--start",    type=str,   default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end",      type=str,   default=DEFAULT_END,
                        help=f"End date   YYYY-MM-DD (default: {DEFAULT_END})")

    # Source filtering
    parser.add_argument("--only",     nargs="+",  metavar="KEY",
                        help="Download only these source keys (see --list)")
    parser.add_argument("--list",     action="store_true",
                        help="List all available source keys and exit")

    # Behaviour flags
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already-downloaded files")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print commands without executing them")

    args = parser.parse_args()

    if args.list:
        _print_source_list()
        sys.exit(0)

    _print_banner(args)

    # Normalise --only to lowercase set
    wanted: Optional[set[str]] = {k.lower() for k in args.only} if args.only else None

    def _wanted(key: str) -> bool:
        return wanted is None or key in wanted

    failures:  list[str] = []
    successes: list[str] = []

    # ── CMEMS datasets ────────────────────────────────────────────────────────
    for ds in CMEMS_DATASETS:
        if not _wanted(ds["key"]):
            continue

        note = ds.get("note", "")
        print(_c(f"\n[Tier {ds['tier']}] {ds['name']}", BOLD + CYAN))
        if note:
            print(_c(f"  Note: {note}", DIM))
        print(f"  Dataset : {ds['dataset_id']}")
        print(f"  Vars    : {', '.join(ds['variables'])}")
        print(f"  Output  : {ds['output_dir']}")

        ok = _cmems_subset(
            dataset_id    = ds["dataset_id"],
            variables     = ds["variables"],
            output_dir    = ds["output_dir"],
            lon_min       = args.lon_min,
            lon_max       = args.lon_max,
            lat_min       = args.lat_min,
            lat_max       = args.lat_max,
            start         = args.start,
            end           = args.end,
            depth         = ds.get("depth", False),
            skip_existing = args.skip_existing,
            dry_run       = args.dry_run,
        )

        if ok:
            print(_c(f"  ✅ {ds['name']}", GREEN))
            successes.append(ds["name"])
        else:
            print(_c(f"  ❌ FAILED: {ds['name']}", RED))
            failures.append(ds["name"])

    # ── NASA SMAP SSS ─────────────────────────────────────────────────────────
    if _wanted("sss"):
        print(_c("\n[Tier 3] SMAP L3 Sea Surface Salinity — NASA Earthdata", BOLD + CYAN))
        ok = _download_smap_sss(
            output_dir = "data/raw/sss",
            lat_min    = args.lat_min,
            lat_max    = args.lat_max,
            lon_min    = args.lon_min,
            lon_max    = args.lon_max,
            start      = args.start,
            end        = args.end,
            dry_run    = args.dry_run,
        )
        (successes if ok else failures).append("SMAP SSS")

    # ── NASA GPM IMERG Precipitation ─────────────────────────────────────────
    if _wanted("precip"):
        print(_c("\n[Tier 2] GPM IMERG Daily Precipitation — NASA Earthdata", BOLD + CYAN))
        ok = _download_gpm_imerg(
            output_dir = "data/raw/precip",
            lat_min    = args.lat_min,
            lat_max    = args.lat_max,
            lon_min    = args.lon_min,
            lon_max    = args.lon_max,
            start      = args.start,
            end        = args.end,
            dry_run    = args.dry_run,
        )
        (successes if ok else failures).append("GPM IMERG Precipitation")

    # ── GEBCO 2023 Bathymetry ─────────────────────────────────────────────────
    if _wanted("bathymetry"):
        print(_c("\n[Tier 5] GEBCO 2023 Bathymetry — static", BOLD + CYAN))
        ok = _download_gebco(
            output_dir = "data/raw/bathymetry",
            dry_run    = args.dry_run,
        )
        (successes if ok else failures).append("GEBCO Bathymetry")

    # ── IHFC Geothermal ───────────────────────────────────────────────────────
    if _wanted("geothermal"):
        print(_c("\n[Tier 5] IHFC Geothermal Heat Flux — static", BOLD + CYAN))
        ok = _download_geothermal(
            output_dir = "data/raw/geothermal",
            dry_run    = args.dry_run,
        )
        (successes if ok else failures).append("IHFC Geothermal")

    # ── GloFAS River Discharge ────────────────────────────────────────────────
    if _wanted("glofas"):
        print(_c("\n[Tier 5] GloFAS River Discharge — via CDS API", BOLD + CYAN))
        ok = _download_glofas(
            output_dir = "data/raw/glofas",
            start      = args.start,
            end        = args.end,
            dry_run    = args.dry_run,
        )
        (successes if ok else failures).append("GloFAS River Discharge")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(_c("═" * 64, BLUE))
    print(_c("  Download Summary", BOLD))
    print(_c("═" * 64, BLUE))

    if successes:
        print(_c(f"\n  ✅  Succeeded ({len(successes)}):", GREEN))
        for name in successes:
            print(f"       • {name}")

    if failures:
        print(_c(f"\n  ❌  Failed ({len(failures)}):", RED))
        for name in failures:
            print(f"       • {name}")
        print(_c("\n  Tip: re-run with --skip-existing to retry only missing files.", YELLOW))
        sys.exit(1)

    if not failures:
        print()
        print(_c("  🎉  All requested datasets downloaded successfully!", GREEN + BOLD))
        print()
        print(_c("  Next steps:", BOLD))
        print("    1. Regrid everything to a common 0.25° grid:")
        print("       python scripts/preprocess.py --config configs/v2_bob_full.yaml")
        print()
        print("    2. Compute normalization stats for all 22+ channels:")
        print("       python scripts/compute_stats.py --config configs/v2_bob_full.yaml")
        print()
        print(f"    📁  Data directory: {_c(Path('data/raw').resolve(), CYAN)}")


if __name__ == "__main__":
    main()
