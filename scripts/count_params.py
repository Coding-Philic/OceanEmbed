"""
Model Parameter Counter — PhysVSANet (PoC-BoB)
Run this to prove exact parameter count live.
Usage: python scripts/count_params.py
"""

import torch
from pathlib import Path

CKPT = Path(__file__).parent.parent / "outputs" / "poc-bob-v1" / "checkpoints" / "epoch059-val_loss0.0000.ckpt"

print("=" * 50)
print("  PhysVSANet — Parameter Count Verification")
print("=" * 50)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]

total      = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
n_layers   = len([k for k, v in sd.items() if hasattr(v, "numel")])
size_mb    = CKPT.stat().st_size / (1024 * 1024)

print(f"  Checkpoint   : {CKPT.name}")
print(f"  File Size    : {size_mb:.1f} MB")
print(f"  Total Layers : {n_layers}")
print(f"  Parameters   : {total:,}")
print(f"  In Millions  : {total / 1e6:.3f} M")
print("=" * 50)

print("\nTop 5 Largest Layers:")
sizes = [(k, v.numel()) for k, v in sd.items() if hasattr(v, "numel")]
for k, n in sorted(sizes, key=lambda x: -x[1])[:5]:
    print(f"  {n:>10,}  {k}")

print("\n✅ Verified from checkpoint weights directly.")
