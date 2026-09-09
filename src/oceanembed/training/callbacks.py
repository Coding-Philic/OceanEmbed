"""
Custom PyTorch Lightning callbacks for OceanEmbed.

  - DepthProfileCallback  — logs vertical profile comparison plots to W&B.
  - RichProgressCallback  — rich-formatted progress bar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch


class DepthProfileCallback(pl.Callback):
    """Logs a predicted vs. actual depth-profile comparison at validation end.

    Picks ``num_profiles`` random ocean pixels from the first validation batch,
    plots T(z) side-by-side, and saves to ``output_dir``. Optionally logs to
    W&B if the W&B logger is attached.

    Args:
        depth_levels:  List of depth levels in metres.
        output_dir:    Directory to write PNG files.
        num_profiles:  Number of random profiles to plot per validation epoch.
        log_every_n:   Log every N validation epochs (reduces overhead).
    """

    def __init__(
        self,
        depth_levels: list[float],
        output_dir: Path | str,
        num_profiles: int = 4,
        log_every_n: int = 5,
    ) -> None:
        self.depth_levels  = depth_levels
        self.output_dir    = Path(output_dir)
        self.num_profiles  = num_profiles
        self.log_every_n   = log_every_n
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self.log_every_n != 0:
            return
        if not hasattr(trainer, "_val_profile_batch"):
            return

        batch = trainer._val_profile_batch
        with torch.no_grad():
            pred = pl_module(batch["input"].to(pl_module.device))

        pred   = pred.cpu().numpy()    # [B, K, H, W]
        target = batch["target"].numpy()
        mask   = batch["mask"].numpy() # [B, H, W]

        B, K, H, W = pred.shape
        depths = np.array(self.depth_levels)

        fig, axes = plt.subplots(
            1, min(self.num_profiles, B),
            figsize=(min(self.num_profiles, B) * 3, 5),
            sharey=True,
        )
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, ax in enumerate(axes):
            # Find a valid (ocean) pixel
            valid = np.argwhere(mask[i] > 0)
            if len(valid) == 0:
                continue
            ij = valid[np.random.randint(len(valid))]
            h, w = int(ij[0]), int(ij[1])

            ax.plot(target[i, :, h, w], depths, "b-o", label="GLORYS", ms=3)
            ax.plot(pred[i,   :, h, w], depths, "r--s", label="Pred",   ms=3)
            ax.invert_yaxis()
            ax.set_xlabel("Temperature (°C)")
            ax.set_title(f"Sample {i} ({h},{w})")
            if i == 0:
                ax.set_ylabel("Depth (m)")
            ax.legend(fontsize=7)

        fig.suptitle(f"Epoch {trainer.current_epoch} — Depth Profiles")
        fig.tight_layout()

        out_path = self.output_dir / f"profiles_epoch{trainer.current_epoch:04d}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        # Log to W&B if available
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"val/depth_profiles": wandb.Image(str(out_path))},
                          step=trainer.global_step)
        except ImportError:
            pass


class StoreFirstValBatchCallback(pl.Callback):
    """Stores the first validation batch for downstream callbacks."""

    def on_validation_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: dict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and not hasattr(trainer, "_val_profile_batch"):
            trainer._val_profile_batch = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
