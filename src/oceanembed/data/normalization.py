"""
Normalization utilities.

Pre-computes per-channel Z-score statistics from the processed input NetCDF
files and saves/loads them as a JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import xarray as xr


@dataclass
class NormalizationStats:
    """Per-channel mean and standard deviation."""
    mean: dict[str, float]
    std:  dict[str, float]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "NormalizationStats":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def normalize(self, channel: str, arr: np.ndarray) -> np.ndarray:
        """Z-score normalise a single channel array."""
        mu  = self.mean[channel]
        sig = self.std[channel]
        return (arr - mu) / max(sig, 1e-6)

    def denormalize(self, channel: str, arr: np.ndarray) -> np.ndarray:
        """Invert Z-score normalisation."""
        return arr * self.std[channel] + self.mean[channel]


def compute_normalization_stats(
    inputs_dir: Path | str,
    channel_names: list[str],
    years: list[int],
    sample_fraction: float = 0.1,
    seed: int = 42,
) -> NormalizationStats:
    """Compute per-channel mean and std from a random subset of the training data.

    Args:
        inputs_dir:       Directory containing per-variable per-year NetCDF files
                          with naming pattern ``{channel}_{year}.nc``.
        channel_names:    List of satellite channel variable names.
        years:            List of training years to sample from.
        sample_fraction:  Fraction of time-steps to include in the estimate.
        seed:             Random seed for reproducibility.
    Returns:
        NormalizationStats with ``mean`` and ``std`` dicts keyed by channel name.
    """
    rng = np.random.default_rng(seed)
    inputs_dir = Path(inputs_dir)

    mean_dict: dict[str, float] = {}
    std_dict:  dict[str, float] = {}

    for ch in channel_names:
        values_list: list[np.ndarray] = []
        for year in years:
            fpath = inputs_dir / f"{ch}_{year}.nc"
            if not fpath.exists():
                continue
            ds = xr.open_dataset(fpath)
            # Assume first non-dim variable holds the data
            varname = [v for v in ds.data_vars][0]
            arr = ds[varname].values.astype(np.float32)  # [time, lat, lon]
            n_t = arr.shape[0]
            n_sample = max(1, int(n_t * sample_fraction))
            idx = rng.choice(n_t, size=n_sample, replace=False)
            subset = arr[idx].ravel()
            subset = subset[np.isfinite(subset)]
            values_list.append(subset)
            ds.close()

        if values_list:
            all_vals = np.concatenate(values_list)
            mean_dict[ch] = float(np.mean(all_vals))
            std_dict[ch]  = float(np.std(all_vals))
        else:
            mean_dict[ch] = 0.0
            std_dict[ch]  = 1.0

    return NormalizationStats(mean=mean_dict, std=std_dict)
