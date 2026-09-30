"""
OceanEmbed training entrypoint.

Usage:
    python scripts/train.py --config configs/poc_bob.yaml
    python scripts/train.py --config configs/full_nio.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path even when not installed via pip
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.serialization
from omegaconf import OmegaConf, DictConfig, ListConfig

# Register OmegaConf safe globals for PyTorch 2.6 weights_only unpickler
try:
    safe_types = [DictConfig, ListConfig]
    for mod_name in ["omegaconf.dictconfig", "omegaconf.listconfig", "omegaconf.basecontainer", "omegaconf.nodes"]:
        try:
            mod = __import__(mod_name, fromlist=["*"])
            for attr in dir(mod):
                val = getattr(mod, attr)
                if isinstance(val, type):
                    safe_types.append(val)
        except Exception:
            pass
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(list(set(safe_types)))
except Exception:
    pass

# In PyTorch 2.6+, torch.load defaults to weights_only=True.
# PyTorch Lightning checkpoints contain full trainer state, optimizers, and OmegaConf objects.
# We set weights_only=False proactively so that checkpoint unpickling never fails with
# 'Unsupported global' or leaves stream pointers dirty.
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

try:
    import lightning_fabric.utilities.cloud_io as _cloud_io
    _orig_pl_load = _cloud_io.pl_load
    def _safe_pl_load(path_or_url, map_location=None, weights_only=None):
        return _orig_pl_load(path_or_url, map_location=map_location, weights_only=False)
    _cloud_io.pl_load = _safe_pl_load
except Exception:
    pass

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from oceanembed.data.dataset        import OceanEmbedDataModule
from oceanembed.data.normalization  import NormalizationStats, compute_normalization_stats
from oceanembed.training.trainer    import OceanEmbedLitModule
from oceanembed.training.callbacks  import (
    DepthProfileCallback,
    StoreFirstValBatchCallback,
)


import shutil
import subprocess

class CloudCheckpointSyncCallback(pl.Callback):
    """Automatically mirror checkpoints to Google Drive in background after each epoch."""
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "global_rank", 0) != 0:
            return
        epoch = trainer.current_epoch
        print(f"\n>>> [COMPLETED EPOCH {epoch}] Model checkpoint updating... (Syncing to Google Drive)", flush=True)
        ckpt_dir = self.output_dir / "checkpoints"
        csv_dir = self.output_dir / "csv_logs"
        if shutil.which("rclone"):
            if ckpt_dir.exists():
                subprocess.Popen([
                    "rclone", "copy",
                    str(ckpt_dir),
                    "gdrive:OceanEmbed/outputs/kaggle-25ch-v1/checkpoints",
                    "--retries", "5",
                    "-q"
                ], stderr=subprocess.DEVNULL)
            if csv_dir.exists():
                subprocess.Popen([
                    "rclone", "copy",
                    str(csv_dir),
                    "gdrive:OceanEmbed/outputs/kaggle-25ch-v1/logs",
                    "-q"
                ], stderr=subprocess.DEVNULL)


def build_callbacks(cfg, output_dir: Path) -> list[pl.Callback]:
    """Build the Lightning callback list from config."""
    callbacks = [
        ModelCheckpoint(
            dirpath   = str(output_dir / "checkpoints"),
            filename  = "epoch{epoch:03d}-val_loss{val/loss:.4f}",
            monitor   = cfg.training.monitor,
            mode      = cfg.training.mode,
            save_top_k= cfg.training.save_top_k,
            save_last = True,
            auto_insert_metric_name = False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        StoreFirstValBatchCallback(),
        CloudCheckpointSyncCallback(output_dir),
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

    print("\n" + "=" * 70, flush=True)
    print("  OCEANEMBED 21-DATASET TRAINING SPECIFICATION", flush=True)
    print("=" * 70, flush=True)
    print(f"  • Raw Archive Origin : ~160 GB multi-source observations", flush=True)
    print(f"  • Aligned Dataset Path: {cfg.data.aligned_dir} (10.68 GB dense tensors)", flush=True)
    print(f"  • Physical Channels ({len(cfg.data.input_channels)}): {', '.join(cfg.data.input_channels)}", flush=True)
    print(f"  • Coordinate Channels (4): lon_norm, lat_norm, sin_doy, cos_doy", flush=True)
    print(f"  • Total Model In-Channels: {cfg.model.in_channels} (Encoder Stem accepts {lit_module.model.encoder.encoder[0].in_channels} channels)", flush=True)
    print(f"  • Total Model Parameters : {sum(p.numel() for p in lit_module.model.parameters()) / 1e6:.2f} Million", flush=True)
    print(f"  • Target Depths ({cfg.model.num_depths}): {cfg.data.depth_levels} m", flush=True)
    print(f"  • Training Splits   : Train={cfg.data.train_years}, Val={cfg.data.val_years}, Test={cfg.data.test_years}", flush=True)
    print(f"  • Model State       : Fresh initialisation from scratch (NO old weights)", flush=True)
    print("=" * 70 + "\n", flush=True)

    # ── Logger ────────────────────────────────────────────────────────
    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv_logs")
    loggers = [csv_logger]
    if not args.no_wandb:
        loggers.append(WandbLogger(
            project = cfg.logging.project,
            name    = cfg.logging.name,
            save_dir= str(output_dir),
        ))

    # ── Hardware / Accelerator Auto-Detection ─────────────────────────
    import torch
    accelerator = cfg.hardware.accelerator
    devices = cfg.hardware.gpus
    strategy = cfg.hardware.strategy if accelerator == "gpu" else "auto"
    precision = cfg.training.precision

    if accelerator == "gpu" and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"  [HARDWARE] Detected {num_gpus} GPUs (Dual T4)! Enabling DDP multi-GPU training for 2x speed.", flush=True)
            devices = num_gpus
            strategy = "ddp"
        else:
            devices = 1
            strategy = "auto"
    elif accelerator == "gpu" and not torch.cuda.is_available():
        print("\n  [WARNING] No GPU detected! Please select 'GPU T4 x2' in Kaggle Notebook Settings.", flush=True)
        print("  Temporarily running on CPU (precision=32)...\n", flush=True)
        accelerator = "cpu"
        devices = "auto"
        strategy = "auto"
        precision = 32

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs          = cfg.training.max_epochs,
        accelerator         = accelerator,
        devices             = devices,
        strategy            = strategy,
        precision           = precision,
        gradient_clip_val   = cfg.training.gradient_clip_val,
        log_every_n_steps   = cfg.logging.log_every_n_steps,
        logger              = loggers,
        callbacks           = build_callbacks(cfg, output_dir),
        enable_progress_bar = True,
    )

    # ── Checkpoint Auto-Resume & Verification ────────────────────────
    ckpt_path = None
    if getattr(args, "ckpt", None):
        ckpt_path = args.ckpt
        print(f"  [RESUME] Explicit checkpoint passed: {ckpt_path}", flush=True)
    else:
        ckpt_dir = output_dir / "checkpoints"
        if ckpt_dir.exists():
            candidates = []
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists() and last_ckpt.stat().st_size > 1024 * 1024:
                candidates.append(last_ckpt)

            epoch_ckpts = sorted(
                [f for f in ckpt_dir.glob("*.ckpt") if f.name != "last.ckpt" and f.stat().st_size > 1024 * 1024],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            candidates.extend(epoch_ckpts)

            for cand in candidates:
                try:
                    print(f"  [RESUME] Verifying checkpoint integrity: {cand.name} ({cand.stat().st_size / (1024*1024):.1f} MB)...", flush=True)
                    test_load = torch.load(str(cand), map_location="cpu", weights_only=False)
                    if isinstance(test_load, dict) and ("state_dict" in test_load or "epoch" in test_load):
                        ckpt_path = str(cand)
                        print(f"  [RESUME] Successfully verified {cand.name}. Resuming training seamlessly...", flush=True)
                        break
                except Exception as err:
                    print(f"  [RESUME] Checkpoint {cand.name} is invalid or incomplete ({err}). Skipping candidate.", flush=True)

            if not ckpt_path:
                print("  [RESUME] No valid intact checkpoint found. Starting fresh initialisation from scratch.", flush=True)

    import inspect
    fit_kwargs = {"datamodule": datamodule, "ckpt_path": ckpt_path}
    if "weights_only" in inspect.signature(trainer.fit).parameters:
        fit_kwargs["weights_only"] = False
    trainer.fit(lit_module, **fit_kwargs)
    print("Training complete.", flush=True)
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OceanEmbed (Phys-VSA-Net)")
    parser.add_argument("--config",   required=True,      help="Path to YAML config")
    parser.add_argument("--ckpt",     default=None,       help="Path to checkpoint to resume training from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                        help="OmegaConf dot-path overrides, e.g. training.lr=5e-5")
    main(parser.parse_args())
