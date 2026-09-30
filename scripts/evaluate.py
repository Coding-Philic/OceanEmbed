#!/usr/bin/env python3
"""
OceanEmbed Model Evaluation & Benchmark Script.

1. Inspects metrics.csv to rank all epochs by validation performance.
2. Evaluates the best checkpoint on the holdout test set (Year 2023).
3. Outputs per-depth RMSE (0.5m, 10m, 50m, 100m, 200m, 500m) and correlation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from oceanembed.data.datamodule import OceanDataModule
from oceanembed.data.normalization import NormalizationStats
from oceanembed.training.trainer import OceanEmbedLightningModule


def analyze_metrics_csv(metrics_path: Path) -> dict:
    """Analyze metrics.csv and print rankings."""
    print("=" * 70)
    print(f"📊 ANALYZING TRAINING METRICS: {metrics_path.name}")
    print("=" * 70)

    df = pd.read_csv(metrics_path)
    
    # Filter epoch rows that contain validation metrics
    val_cols = [c for c in df.columns if "val/" in c or "val_loss" in c]
    if not val_cols:
        print("No validation columns found in metrics.csv.")
        return {}

    # Extract epoch summary
    epoch_df = df[df["epoch"].notna()].groupby("epoch").last().reset_index()

    sort_col = "val/total" if "val/total" in epoch_df.columns else ("val/wmse" if "val/wmse" in epoch_df.columns else val_cols[0])
    epoch_df_sorted = epoch_df.sort_values(by=sort_col, ascending=True)

    best_row = epoch_df_sorted.iloc[0]
    best_epoch = int(best_row["epoch"])

    print(f"\n🏆 BEST EPOCH DETECTED: Epoch {best_epoch:02d}")
    print(f"   • {sort_col}: {best_row[sort_col]:.4f}")
    if "val/wmse" in best_row:
        print(f"   • Val WMSE  : {best_row['val/wmse']:.4f} (Approx Subsurface Error: {best_row['val/wmse']**0.5:.2f}°C)")
    if "val/gradient" in best_row:
        print(f"   • Val Grad  : {best_row['val/gradient']:.4f}")

    print("\n📈 TOP 5 EPOCHS RANKING:")
    cols_to_show = ["epoch", sort_col]
    for c in ["val/wmse", "train/total", "train/wmse"]:
        if c in epoch_df.columns and c not in cols_to_show:
            cols_to_show.append(c)

    top5 = epoch_df_sorted[cols_to_show].head(5)
    print(top5.to_string(index=False))
    print("=" * 70 + "\n")

    return {"best_epoch": best_epoch, "best_score": best_row[sort_col]}


def evaluate_test_set(config_path: Path, ckpt_path: Path, output_dir: Path | None = None) -> None:
    """Run full evaluation on Year 2023 holdout test set."""
    print("=" * 70)
    print(f"🧪 EVALUATING CHECKPOINT ON 2023 TEST SET: {ckpt_path.name}")
    print("=" * 70)

    cfg = OmegaConf.load(config_path)
    
    # Load normalization stats
    stats_path = Path(cfg.data.stats_file)
    if not stats_path.exists():
        alt_stats = Path("/kaggle/working/processed/normalization_stats.json")
        if alt_stats.exists():
            stats_path = alt_stats

    stats = NormalizationStats.load(stats_path)
    print(f"  Loaded normalization stats from: {stats_path}")

    # Build DataModule
    datamodule = OceanDataModule(cfg, stats=stats)
    datamodule.setup(stage="test")

    # Load LightningModule with PyTorch 2.6 safe loading
    torch_load_kwargs = {"weights_only": False} if "weights_only" in torch.load.__code__.co_varnames else {}
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", **torch_load_kwargs)

    lit_module = OceanEmbedLightningModule(cfg)
    if "state_dict" in checkpoint:
        lit_module.load_state_dict(checkpoint["state_dict"])
    else:
        lit_module.load_state_dict(checkpoint)

    lit_module.eval()

    # Run test
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
    )

    results = trainer.test(lit_module, datamodule=datamodule)
    print("\n" + "=" * 70)
    print("✅ TEST BENCHMARK RESULTS (Year 2023 Unseen Data):")
    print("=" * 70)
    for r in results:
        for k, v in sorted(r.items()):
            if "rmse" in k:
                print(f"  • {k:<25}: {v:.4f} °C")
            else:
                print(f"  • {k:<25}: {v:.4f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OceanEmbed Checkpoints")
    parser.add_argument("--config", default="configs/kaggle_25ch.yaml", help="Path to config YAML")
    parser.add_argument("--metrics", default=None, help="Path to metrics.csv to analyze rankings")
    parser.add_argument("--ckpt", default=None, help="Path to .ckpt file to test on 2023 holdout")
    args = parser.parse_args()

    # 1. Check for metrics.csv if provided or search default locations
    metrics_file = None
    if args.metrics and Path(args.metrics).exists():
        metrics_file = Path(args.metrics)
    else:
        for candidate in [
            Path("/kaggle/working/outputs/kaggle-25ch-v1/csv_logs/version_0/metrics.csv"),
            Path("outputs/kaggle-25ch-v1/csv_logs/version_0/metrics.csv"),
        ]:
            if candidate.exists():
                metrics_file = candidate
                break

    if metrics_file:
        analyze_metrics_csv(metrics_file)

    # 2. Test evaluation on 2023 holdout
    if args.ckpt:
        evaluate_test_set(Path(args.config), Path(args.ckpt))


if __name__ == "__main__":
    main()
