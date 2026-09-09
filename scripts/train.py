"""
OceanEmbed training entrypoint.

Usage:
    python scripts/train.py --config configs/poc_bob.yaml
    python scripts/train.py --config configs/full_nio.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from oceanembed.data.dataset        import OceanEmbedDataModule
from oceanembed.data.normalization  import NormalizationStats, compute_normalization_stats
from oceanembed.training.trainer    import OceanEmbedLitModule
from oceanembed.training.callbacks  import (
    DepthProfileCallback,
    StoreFirstValBatchCallback,
)


def build_callbacks(cfg, output_dir: Path) -> list[pl.Callback]:
    """Build the Lightning callback list from config."""
    callbacks = [
        ModelCheckpoint(
            dirpath   = str(output_dir / "checkpoints"),
            filename  = "epoch{epoch:03d}-val_loss{val/loss:.4f}",
            monitor   = cfg.training.monitor,
            mode      = cfg.training.mode,
            save_top_k= cfg.training.save_top_k,
            auto_insert_metric_name = False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        StoreFirstValBatchCallback(),
        DepthProfileCallback(
            depth_levels = cfg.data.depth_levels,
            output_dir   = output_dir / "plots",
            num_profiles = cfg.get("viz", {}).get("num_profiles", 4),
            log_every_n  = cfg.get("viz", {}).get("log_every_n", 5),
        ),
    ]

    # Optional early stopping
    patience = cfg.training.get("early_stopping_patience", 0)
    if patience > 0:
        callbacks.append(EarlyStopping(
            monitor  = cfg.training.monitor,
            patience = patience,
            mode     = cfg.training.mode,
        ))

    return callbacks


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)

    # Allow CLI overrides: --override key=value key2=value2 ...
    if args.override:
        cli_cfg = OmegaConf.from_dotlist(args.override)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    output_dir = Path(cfg.get("output_dir", "outputs")) / cfg.logging.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Normalization stats ───────────────────────────────────────────
    stats_path = Path(cfg.data.stats_file)
    if stats_path.exists():
        stats = NormalizationStats.load(stats_path)
        print(f"Loaded normalisation stats from {stats_path}")
    else:
        print("Computing normalisation stats …")
        stats = compute_normalization_stats(
            inputs_dir     = Path(cfg.data.aligned_dir) / "inputs",
            channel_names  = cfg.data.input_channels,
            years          = cfg.data.train_years,
            sample_fraction= cfg.get("norm_sample_fraction", 0.1),
        )
        stats.save(stats_path)
        print(f"Saved normalisation stats → {stats_path}")

    # ── DataModule ────────────────────────────────────────────────────
    datamodule = OceanEmbedDataModule(cfg, stats=stats)

    # ── Model ─────────────────────────────────────────────────────────
    lit_module = OceanEmbedLitModule(cfg)

    # ── Logger ────────────────────────────────────────────────────────
    logger = WandbLogger(
        project = cfg.logging.project,
        name    = cfg.logging.name,
        save_dir= str(output_dir),
    ) if not args.no_wandb else None

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs          = cfg.training.max_epochs,
        accelerator         = cfg.hardware.accelerator,
        devices             = cfg.hardware.gpus,
        strategy            = cfg.hardware.strategy,
        precision           = cfg.training.precision,
        gradient_clip_val   = cfg.training.gradient_clip_val,
        log_every_n_steps   = cfg.logging.log_every_n_steps,
        logger              = logger,
        callbacks           = build_callbacks(cfg, output_dir),
        enable_progress_bar = True,
    )

    trainer.fit(lit_module, datamodule=datamodule)
    print("Training complete.")
    print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OceanEmbed (Phys-VSA-Net)")
    parser.add_argument("--config",   required=True,      help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="OmegaConf dot-path overrides, e.g. training.lr=5e-5")
    main(parser.parse_args())
