"""
PyTorch Lightning Training Module.

Wraps PhysVSANet + OceanEmbedLoss with training / validation steps,
optimizer configuration, LR scheduling, and per-depth RMSE logging.

All hyper-parameters are read from the config object — nothing is
hardcoded in this file.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from oceanembed.models.phys_vsa_net import PhysVSANet
from oceanembed.models.losses       import OceanEmbedLoss


class OceanEmbedLitModule(pl.LightningModule):
    """Lightning module for training Phys-VSA-Net.

    Args:
        config: OmegaConf / dict-like config with ``model``, ``data``,
                ``loss``, and ``training`` sections.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = config
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(config):
                self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
            else:
                self.save_hyperparameters()
        except Exception:
            self.save_hyperparameters()
        cfg = config

        # ── Model ────────────────────────────────────────────────────
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

        # ── Loss ─────────────────────────────────────────────────────
        self.loss_fn = OceanEmbedLoss(
            depth_levels       = cfg.data.depth_levels,
            lambda_grad        = cfg.loss.lambda_grad,
            lambda_steric      = cfg.loss.lambda_steric,
            lambda_corr        = cfg.loss.lambda_corr,
            thermocline_center = cfg.loss.thermocline_center,
            thermocline_sigma  = cfg.loss.thermocline_sigma,
            depth_weight_base  = cfg.loss.depth_weight_base,
            depth_weight_peak  = cfg.loss.depth_weight_peak,
            alpha_thermal      = cfg.loss.get("alpha_thermal", 2.1e-4),
            eps                = cfg.loss.get("eps", 1e-8),
        )

        self.depth_levels = cfg.data.depth_levels

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str
    ) -> torch.Tensor:
        pred = self.model(batch["input"])
        loss, loss_dict = self.loss_fn(
            pred           = pred,
            target         = batch["target"],
            sla_obs        = batch.get("sla_obs"),
            t_climatology  = batch.get("t_climatology"),
            mask           = batch.get("mask"),
        )

        # Log all loss components
        self.log_dict(
            {f"{stage}/{k.split('/')[1]}": v for k, v in loss_dict.items()},
            prog_bar = (stage == "val"),
            on_step  = (stage == "train"),
            on_epoch = True,
            sync_dist= True,
        )
        return loss, pred

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, _ = self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        _, pred = self._shared_step(batch, "val")

        # Per-depth RMSE
        target = batch["target"]
        for k, depth in enumerate(self.depth_levels):
            rmse = ((pred[:, k] - target[:, k]) ** 2).mean().sqrt()
            self.log(f"val/rmse_{int(depth)}m", rmse,
                     on_epoch=True, sync_dist=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        _, pred = self._shared_step(batch, "test")

        target = batch["target"]
        for k, depth in enumerate(self.depth_levels):
            rmse = ((pred[:, k] - target[:, k]) ** 2).mean().sqrt()
            self.log(f"test/rmse_{int(depth)}m", rmse,
                     on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Resolve training configuration flexibly across all checkpoint versions and formats
        cfg_train = None
        if hasattr(self, "cfg") and self.cfg is not None:
            if hasattr(self.cfg, "training"):
                cfg_train = self.cfg.training
            elif isinstance(self.cfg, dict) and "training" in self.cfg:
                cfg_train = self.cfg["training"]

        if cfg_train is None and hasattr(self, "hparams"):
            if hasattr(self.hparams, "config"):
                conf = self.hparams.config
                cfg_train = getattr(conf, "training", None) if not isinstance(conf, dict) else conf.get("training")
            elif hasattr(self.hparams, "training"):
                cfg_train = self.hparams.training
            elif isinstance(self.hparams, dict) and "training" in self.hparams:
                cfg_train = self.hparams["training"]

        if cfg_train is None:
            raise ValueError(
                f"Could not resolve 'training' configuration. Available hparams keys: "
                f"{list(getattr(self, 'hparams', {}).keys())}"
            )

        def _get(obj, key, default=None):
            if hasattr(obj, "get"):
                val = obj.get(key, default)
                return default if val is None else val
            return getattr(obj, key, default)

        lr           = float(_get(cfg_train, "lr", 1e-3))
        weight_decay = float(_get(cfg_train, "weight_decay", 1e-4))
        t_0          = int(_get(cfg_train, "scheduler_t0", 20))
        t_mult       = int(_get(cfg_train, "scheduler_t_mult", 2))
        min_lr       = float(_get(cfg_train, "min_lr", 1e-6))

        optimizer = AdamW(
            self.parameters(),
            lr           = lr,
            weight_decay = weight_decay,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0      = t_0,
            T_mult   = t_mult,
            eta_min  = min_lr,
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
                "frequency": 1,
            },
        }
