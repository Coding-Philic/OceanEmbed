"""
Argo Float Colocation Engine.

Matches model predictions spatially and temporally to delayed-mode quality
controlled (DM-QC) Argo profiles.  Uses an efficient KD-tree on (lat, lon)
then validates timestamps within a configurable tolerance window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ArgoMatchup:
    """Collocated Argo observation–prediction pairs.

    Attributes:
        pred_profiles:  [N, K] — model predictions at matched locations.
        argo_profiles:  [N, K] — observed Argo temperature profiles (°C).
        depths:         [K]    — depth levels in metres.
        lats:           [N]    — matched latitude.
        lons:           [N]    — matched longitude.
        dates:          [N]    — matched date strings (YYYY-MM-DD).
        n_matches:      Number of matched pairs.
    """
    pred_profiles: np.ndarray
    argo_profiles: np.ndarray
    depths:        np.ndarray
    lats:          np.ndarray
    lons:          np.ndarray
    dates:         list[str]

    @property
    def n_matches(self) -> int:
        return len(self.lats)


def colocate_argo(
    pred_cube: np.ndarray,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    pred_dates: list[str],
    argo_dir: Path | str,
    depth_levels: list[float],
    spatial_tolerance_deg: float = 0.125,
    temporal_tolerance_days: int = 3,
) -> ArgoMatchup:
    """Match model predictions to Argo DM-QC profiles.

    Args:
        pred_cube:              [T, K, H, W] — time series of model predictions.
        grid_lats:              [H]          — latitude axis.
        grid_lons:              [W]          — longitude axis.
        pred_dates:             Length-T list of date strings (YYYY-MM-DD).
        argo_dir:               Directory containing Argo profile NetCDF files.
        depth_levels:           Target depth levels in metres.
        spatial_tolerance_deg:  Maximum spatial distance (°) for a valid match.
        temporal_tolerance_days: Maximum time offset (days) for a valid match.
    Returns:
        ArgoMatchup with collocated arrays.
    """
    import xarray as xr
    from datetime import datetime, timedelta

    argo_dir = Path(argo_dir)
    K = len(depth_levels)
    depths = np.array(depth_levels)

    pred_date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in pred_dates]

    # Build KD-tree over the model grid: (lat, lon) in degrees
    lat_grid, lon_grid = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    tree = cKDTree(grid_points)

    pred_profiles_list: list[np.ndarray] = []
    argo_profiles_list: list[np.ndarray] = []
    lats_list:  list[float] = []
    lons_list:  list[float] = []
    dates_list: list[str]   = []

    for argo_file in sorted(argo_dir.glob("*.nc")):
        try:
            ds = xr.open_dataset(argo_file)
        except Exception:
            continue

        for p_idx in range(ds.dims.get("N_PROF", 0)):
            try:
                argo_lat  = float(ds["LATITUDE"].isel(N_PROF=p_idx).values)
                argo_lon  = float(ds["LONGITUDE"].isel(N_PROF=p_idx).values)
                argo_time = str(ds["JULD"].isel(N_PROF=p_idx).values)[:10]
                argo_dt   = datetime.strptime(argo_time, "%Y-%m-%d")

                # ── Temporal matching ─────────────────────────────────
                time_diffs = [abs((argo_dt - pd).days) for pd in pred_date_objs]
                best_t_idx = int(np.argmin(time_diffs))
                if time_diffs[best_t_idx] > temporal_tolerance_days:
                    continue

                # ── Spatial matching ──────────────────────────────────
                dist, flat_idx = tree.query([argo_lat, argo_lon])
                if dist > spatial_tolerance_deg:
                    continue
                h_idx, w_idx = divmod(int(flat_idx), len(grid_lons))

                # ── Extract model profile ─────────────────────────────
                pred_prof = pred_cube[best_t_idx, :, h_idx, w_idx].copy()

                # ── Extract Argo temperature at target depths ─────────
                argo_pres = ds["PRES"].isel(N_PROF=p_idx).values
                argo_temp = ds["TEMP"].isel(N_PROF=p_idx).values
                valid = np.isfinite(argo_pres) & np.isfinite(argo_temp)
                if valid.sum() < 2:
                    continue
                argo_prof = np.interp(
                    depths, argo_pres[valid], argo_temp[valid],
                    left=np.nan, right=np.nan,
                )
                if not np.isfinite(argo_prof).all():
                    continue

                pred_profiles_list.append(pred_prof)
                argo_profiles_list.append(argo_prof)
                lats_list.append(argo_lat)
                lons_list.append(argo_lon)
                dates_list.append(argo_time)

            except Exception:
                continue

        ds.close()

    if not pred_profiles_list:
        # Return empty matchup
        return ArgoMatchup(
            pred_profiles = np.empty((0, K)),
            argo_profiles = np.empty((0, K)),
            depths        = depths,
            lats          = np.array([]),
            lons          = np.array([]),
            dates         = [],
        )

    return ArgoMatchup(
        pred_profiles = np.array(pred_profiles_list),
        argo_profiles = np.array(argo_profiles_list),
        depths        = depths,
        lats          = np.array(lats_list),
        lons          = np.array(lons_list),
        dates         = dates_list,
    )
