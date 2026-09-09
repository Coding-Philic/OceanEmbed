"""
Single-pass inference pipeline.

Loads a checkpoint, processes a directory of aligned satellite NetCDF files,
and writes the predicted 3D temperature cubes to Zarr / NetCDF output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

from oceanembed.models.phys_vsa_net  import PhysVSANet
from oceanembed.data.normalization   import NormalizationStats
from oceanembed.data.dataset         import OceanEmbedDataset


class OceanEmbedPredictor:
    """Load a trained OceanEmbed model and run inference on new satellite data.

    Args:
        checkpoint_path: Path to a Lightning ``.ckpt`` checkpoint file.
        config_path:     Path to the YAML config used for training.
        stats_path:      Path to the normalization stats JSON.
        device:          Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        config_path: Path | str,
        stats_path: Path | str,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # Load config
        self.config = OmegaConf.load(config_path)
        cfg = self.config

        # Load normalization stats
        self.stats = NormalizationStats.load(stats_path)

        # Rebuild model from config
        self.model = PhysVSANet(
            in_channels        = cfg.model.in_channels,
            num_depths         = cfg.model.num_depths,
            embed_dim          = cfg.model.embed_dim,
            num_heads          = cfg.model.num_heads,
            num_decoder_layers = cfg.model.num_decoder_layers,
            dropout            = cfg.model.dropout,
            stage_channels     = cfg.model.get("stage_channels", None),
            head_channels      = cfg.model.get("head_channels", None),
            stem_kernel_size   = cfg.model.get("stem_kernel_size", 7),
            res_kernel_size    = cfg.model.get("res_kernel_size", 3),
            head_kernel_size   = cfg.model.get("head_kernel_size", 3),
            se_reduction       = cfg.model.get("se_reduction", 16),
            mlp_ratio          = cfg.model.get("mlp_ratio", 2.0),
            init_std           = cfg.model.get("init_std", 0.02),
        )

        # Load weights from checkpoint (Lightning format)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_dataset(
        self,
        data_dir: Path | str,
        years: list[int],
        batch_size: int = 8,
        output_path: Path | str | None = None,
    ) -> np.ndarray:
        """Run inference over a dataset split.

        Args:
            data_dir:    Aligned data directory.
            years:       Years to predict.
            batch_size:  Samples per inference batch.
            output_path: If provided, saves results as a NetCDF file.
        Returns:
            predictions: [T, K, H, W] float32 array.
        """
        cfg = self.config.data
        dataset = OceanEmbedDataset(
            data_dir          = data_dir,
            years             = years,
            depth_levels      = cfg.depth_levels,
            input_channels    = cfg.input_channels,
            stats             = self.stats,
            sla_channel_name  = cfg.get("sla_channel_name", "sla"),
            nan_fill_value    = cfg.get("nan_fill_value", 0.0),
            days_per_year     = cfg.get("days_per_year", 365.25),
            channel_dropout_p = 0.0,
            is_train          = False,
        )

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0)

        all_preds: list[np.ndarray] = []
        for batch in loader:
            x = batch["input"].to(self.device)
            pred = self.model(x)           # [B, K, H, W]
            all_preds.append(pred.cpu().numpy())

        predictions = np.concatenate(all_preds, axis=0)  # [T, K, H, W]

        if output_path is not None:
            self._save_netcdf(predictions, dataset, output_path)

        return predictions

    @torch.no_grad()
    def predict_single(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a single pre-assembled input tensor.

        Args:
            x: [1, C, H, W] normalised input.
        Returns:
            T_pred: [1, K, H, W]
        """
        return self.model(x.to(self.device))

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_netcdf(
        self,
        predictions: np.ndarray,
        dataset: OceanEmbedDataset,
        output_path: Path | str,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        T, K, H, W = predictions.shape

        ds = xr.Dataset(
            {"temperature": (["time", "depth", "lat", "lon"], predictions)},
            attrs={
                "description": "OceanEmbed 3D subsurface temperature prediction",
                "units": "degrees_Celsius",
                "depth_levels": str(dataset.depth_levels),
            },
        )
        ds.to_netcdf(output_path)
        print(f"Saved predictions → {output_path}")
