"""
Benchmark baselines: WOA23 climatology and GEM (Gravest Empirical Mode).

Used to compute the skill score denominator (reference MSE) and to produce
fair comparison tables for the evaluation report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from oceanembed.evaluation.metrics import SkillScore, compute_metrics


def load_woa23_climatology(
    woa23_path: Path | str,
    depth_levels: list[float],
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    month: int,
    temp_varname: str = "t_an",
) -> np.ndarray:
    """Load the WOA23 monthly climatology and interpolate to target depths.

    Args:
        woa23_path:   Path to the WOA23 temperature NetCDF file.
        depth_levels: Target depth levels in metres.
        lat_range:    (lat_min, lat_max) bounding box.
        lon_range:    (lon_min, lon_max) bounding box.
        month:        1-indexed month.
        temp_varname: Variable name in the WOA23 file.
    Returns:
        clim: [K, H, W] monthly mean temperature field at target depths.
    """
    ds = xr.open_dataset(woa23_path)
    ds = ds.sel(
        lat  = slice(lat_range[0],  lat_range[1]),
        lon  = slice(lon_range[0],  lon_range[1]),
        time = month - 1,          # 0-indexed
    )

    depths_nc = ds["depth"].values
    clim_all  = ds[temp_varname].values  # [n_depths, H, W]

    # Interpolate to requested depth levels
    K = len(depth_levels)
    H, W = clim_all.shape[1], clim_all.shape[2]
    clim_interp = np.zeros((K, H, W), dtype=np.float32)
    for h in range(H):
        for w in range(W):
            col = clim_all[:, h, w]
            valid = np.isfinite(col)
            if valid.sum() < 2:
                clim_interp[:, h, w] = np.nan
            else:
                clim_interp[:, h, w] = np.interp(
                    depth_levels, depths_nc[valid], col[valid],
                    left=np.nan, right=np.nan,
                )
    ds.close()
    return clim_interp


def evaluate_against_baseline(
    pred: np.ndarray,
    target: np.ndarray,
    climatology: np.ndarray,
    mask: np.ndarray | None = None,
    depth_levels: list[float] | None = None,
    eps: float = 1e-8,
) -> dict[str, SkillScore]:
    """Return skill scores for the model and the climatology baseline.

    Args:
        pred:        [K, H, W] or [T, K, H, W] — model predictions.
        target:      [K, H, W] or [T, K, H, W] — ground truth.
        climatology: [K, H, W] — reference climatology broadcast as needed.
        mask:        [H, W]    — ocean mask.
        depth_levels: Depth levels in metres.
        eps:         Numerical stability epsilon.
    Returns:
        dict with keys ``"model"`` and ``"climatology"`` mapping to SkillScore.
    """
    # If temporal dimension exists, flatten to spatial
    if pred.ndim == 4:
        T, K, H, W = pred.shape
        pred        = pred.mean(0)   # [K, H, W]
        target      = target.mean(0)

    # Climatology trivially scores skill=0 against itself
    clim_broadcast = np.broadcast_to(climatology, pred.shape).copy()

    model_score = compute_metrics(
        pred, target, clim_broadcast,
        mask=mask, depth_levels=depth_levels, eps=eps,
    )
    clim_score  = compute_metrics(
        clim_broadcast, target, clim_broadcast,
        mask=mask, depth_levels=depth_levels, eps=eps,
    )

    return {"model": model_score, "climatology": clim_score}
