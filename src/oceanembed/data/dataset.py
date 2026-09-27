"""
OceanEmbed Dataset & DataModule.

Each sample = one calendar day of aligned satellite observations + GLORYS12V1
subsurface temperature target on a common 0.25° grid.

File layout expected in ``data_dir``:
    data_dir/
    ├── inputs/
    │   ├── sst_2017.nc          # [time, lat, lon]
    │   ├── sss_2017.nc
    │   ├── sla_2017.nc
    │   ├── wind_u_2017.nc
    │   ├── wind_v_2017.nc
    │   ├── cur_u_2017.nc
    │   └── cur_v_2017.nc
    └── targets/
        └── glorys_temp_2017.nc  # [time, depth, lat, lon]
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from oceanembed.data.normalization import NormalizationStats


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OceanEmbedDataset(Dataset):
    """Multi-channel satellite + GLORYS dataset for subsurface temperature.

    Args:
        data_dir:           Root directory with ``inputs/`` and ``targets/``
                            sub-directories.
        years:              List of years to include.
        depth_levels:       List of depth levels in metres (for label selection).
        input_channels:     Ordered list of satellite variable names.
        stats:              Pre-computed ``NormalizationStats`` for the input
                            channels.  If ``None`` no normalisation is applied.
        sla_channel_name:   Name of the SLA variable in ``input_channels``.
                            Used to extract the SLA slice for the steric loss.
        nan_fill_value:     Scalar fill value for NaN pixels (land / gaps).
                            The valid-pixel mask is computed before filling.
        days_per_year:      Days per year for seasonal encoding (365.25 = mean
                            Gregorian year, accounting for leap years).
        channel_dropout_p:  Per-satellite-channel zero-out probability applied
                            during training to simulate missing observations.
        is_train:           When ``True`` channel dropout is enabled.
    """

    def __init__(
        self,
        data_dir: Path | str,
        years: list[int],
        depth_levels: list[float],
        input_channels: list[str],
        stats: NormalizationStats | None = None,
        sla_channel_name: str = "sla",
        nan_fill_value: float = 0.0,
        days_per_year: float = 365.25,
        channel_dropout_p: float = 0.0,
        is_train: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir            = Path(data_dir)
        self.years               = years
        self.depth_levels        = depth_levels
        self.input_channels      = input_channels
        self.num_sat_channels    = len(input_channels)
        self.stats               = stats
        self.sla_channel_name    = sla_channel_name
        self.nan_fill_value      = nan_fill_value
        self.days_per_year       = days_per_year
        self.channel_dropout_p   = channel_dropout_p
        self.is_train            = is_train

        # Validate SLA channel is present
        if sla_channel_name not in input_channels:
            raise ValueError(
                f"sla_channel_name='{sla_channel_name}' not found in "
                f"input_channels={input_channels}"
            )
        self.sla_idx = input_channels.index(sla_channel_name)

        # Build ordered list of (year, day_within_year) pairs
        self.samples: list[tuple[int, int]] = self._build_index()

        # Cache grid shape from first available file
        self._grid_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> list[tuple[int, int]]:
        samples: list[tuple[int, int]] = []
        first_ch = self.input_channels[0]
        for year in self.years:
            fpath = self.data_dir / "inputs" / f"{first_ch}_{year}.nc"
            if not fpath.exists():
                continue
            with xr.open_dataset(fpath) as ds:
                n_days = ds.dims["time"]
            for d in range(n_days):
                samples.append((year, d))
        return samples

    # ------------------------------------------------------------------
    # Grid helper
    # ------------------------------------------------------------------

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(H, W) of the common spatial grid."""
        if self._grid_shape is None:
            first_ch = self.input_channels[0]
            year     = self.years[0]
            fpath    = self.data_dir / "inputs" / f"{first_ch}_{year}.nc"
            with xr.open_dataset(fpath) as ds:
                varname = next(iter(ds.data_vars))
                H = ds.dims["lat"]
                W = ds.dims["lon"]
            self._grid_shape = (H, W)
        return self._grid_shape

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_input_channel(
        self, channel: str, year: int, day_idx: int
    ) -> np.ndarray:
        """Return a single [H, W] float32 array (NaNs preserved)."""
        fpath = self.data_dir / "inputs" / f"{channel}_{year}.nc"
        with xr.open_dataset(fpath) as ds:
            varname = next(iter(ds.data_vars))
            arr = ds[varname].isel(time=day_idx).values.astype(np.float32)
        return arr  # [H, W]

    def _load_target(self, year: int, day_idx: int) -> np.ndarray:
        """Return the [K, H, W] GLORYS temperature slice."""
        fpath = self.data_dir / "targets" / f"glorys_temp_{year}.nc"
        with xr.open_dataset(fpath) as ds:
            varname = next(iter(ds.data_vars))
            arr = ds[varname].isel(time=day_idx).values.astype(np.float32)
        # arr shape: [all_depths, H, W] — select configured depth levels
        # Assumes depth coordinate is accessible; here we select by index
        return arr  # [K, H, W]  (depth selection done during preprocessing)

    # ------------------------------------------------------------------
    # Coordinate encoding
    # ------------------------------------------------------------------

    def _compute_coords(self, year: int, day_idx: int) -> np.ndarray:
        """Return [4, H, W] coordinate channels: lon, lat, sin_doy, cos_doy."""
        H, W = self.grid_shape

        # Normalised longitude in [0, 1]
        lon_grid = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
        # Normalised latitude in [0, 1]
        lat_grid = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)

        # Day-of-year (1-indexed) within that year
        doy = day_idx + 1
        angle = 2.0 * math.pi * doy / self.days_per_year
        sin_doy = np.full((H, W), math.sin(angle), dtype=np.float32)
        cos_doy = np.full((H, W), math.cos(angle), dtype=np.float32)

        return np.stack([lon_grid, lat_grid, sin_doy, cos_doy], axis=0)  # [4, H, W]

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def _normalize(self, inputs: np.ndarray) -> np.ndarray:
        """Normalise the satellite input channels in-place.

        Coordinate channels (appended after satellite channels) are left as-is
        because lon/lat are already in [0,1] and sin/cos in [-1, 1].

        Args:
            inputs: [num_sat_channels + 4, H, W]
        Returns:
            Normalised array of the same shape.
        """
        if self.stats is None:
            return inputs
        out = inputs.copy()
        for i, ch in enumerate(self.input_channels):
            out[i] = self.stats.normalize(ch, inputs[i])
        return out

    # ------------------------------------------------------------------
    # Channel dropout
    # ------------------------------------------------------------------

    def _apply_channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly zero entire satellite channels to simulate missing data.

        Only satellite channels (indices 0 … num_sat_channels-1) are dropped.
        Coordinate channels are never touched.
        """
        for c in range(self.num_sat_channels):
            if torch.rand(1).item() < self.channel_dropout_p:
                x[c] = 0.0
        return x

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        year, day_idx = self.samples[idx]

        # ── Load satellite channels ──────────────────────────────────
        channel_arrays = [
            self._load_input_channel(ch, year, day_idx)
            for ch in self.input_channels
        ]  # list of [H, W], each float32

        inputs = np.stack(channel_arrays, axis=0)  # [num_sat_channels, H, W]

        # ── Coordinate channels ──────────────────────────────────────
        coords = self._compute_coords(year, day_idx)  # [4, H, W]

        # ── Concatenate → full input tensor ──────────────────────────
        x = np.concatenate([inputs, coords], axis=0)  # [num_sat + 4, H, W]

        # ── Target subsurface temperature ────────────────────────────
        target = self._load_target(year, day_idx)  # [K, H, W]

        # ── Valid-pixel mask (must be computed before NaN fill) ───────
        # Mask requires both inputs and target to be valid (no NaNs)
        mask_inputs = np.all(np.isfinite(inputs), axis=0)
        mask_target = np.all(np.isfinite(target), axis=0)
        mask = (mask_inputs & mask_target).astype(np.float32)  # [H, W]

        # ── Fill NaNs ────────────────────────────────────────────────
        x = np.where(np.isfinite(x), x, self.nan_fill_value)
        target = np.where(np.isfinite(target), target, self.nan_fill_value)

        # ── Normalise satellite channels ─────────────────────────────
        x = self._normalize(x)

        # ── SLA slice for steric loss ─────────────────────────────────
        # Extract from the raw (pre-normalised, NaN-filled) inputs array
        sla_obs = inputs[self.sla_idx : self.sla_idx + 1].copy()  # [1, H, W]
        sla_obs = np.where(np.isfinite(sla_obs), sla_obs, self.nan_fill_value)

        # ── Convert to tensors ────────────────────────────────────────
        x_t      = torch.from_numpy(x).float()
        target_t = torch.from_numpy(target).float()
        mask_t   = torch.from_numpy(mask).float()
        sla_t    = torch.from_numpy(sla_obs).float()

        # ── Channel dropout (training augmentation) ────────────────────
        if self.is_train and self.channel_dropout_p > 0.0:
            x_t = self._apply_channel_dropout(x_t)

        return {
            "input":   x_t,      # [num_sat + 4, H, W]
            "target":  target_t, # [K, H, W]
            "mask":    mask_t,   # [H, W]
            "sla_obs": sla_t,    # [1, H, W]
        }


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------

class OceanEmbedDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapping OceanEmbedDataset.

    All split definitions, paths, and loader parameters come from the config
    object so that nothing is hardcoded here.

    Args:
        config: OmegaConf / dict-like config with ``data``, ``training``,
                and ``hardware`` sections.
        stats:  Pre-computed normalisation statistics.
    """

    def __init__(self, config, stats: NormalizationStats | None = None) -> None:
        super().__init__()
        self.config = config
        self.stats  = stats

    def _make_dataset(
        self, years: list[int], is_train: bool = False
    ) -> OceanEmbedDataset:
        cfg = self.config.data
        return OceanEmbedDataset(
            data_dir          = cfg.aligned_dir,
            years             = years,
            depth_levels      = cfg.depth_levels,
            input_channels    = cfg.input_channels,
            stats             = self.stats,
            sla_channel_name  = cfg.get("sla_channel_name", "sla"),
            nan_fill_value    = cfg.get("nan_fill_value", 0.0),
            days_per_year     = cfg.get("days_per_year", 365.25),
            channel_dropout_p = cfg.get("channel_dropout_p",
                                        self.config.training.channel_dropout_p)
                                if is_train else 0.0,
            is_train          = is_train,
        )

    def setup(self, stage: str | None = None) -> None:
        cfg = self.config.data
        self.train_ds = self._make_dataset(cfg.train_years, is_train=True)
        self.val_ds   = self._make_dataset(cfg.val_years,   is_train=False)
        self.test_ds  = self._make_dataset(cfg.test_years,  is_train=False)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        cfg = self.config
        return DataLoader(
            ds,
            batch_size  = cfg.training.batch_size,
            shuffle     = shuffle,
            num_workers = cfg.hardware.num_workers,
            pin_memory  = True,
            drop_last   = shuffle,
            persistent_workers = cfg.hardware.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
