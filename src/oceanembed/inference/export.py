"""
Model export utilities.

Exports the trained Phys-VSA-Net to:
  - TorchScript (``torch.jit.script``) for CPU/GPU serving.
  - ONNX (opset 17) for Triton Inference Server and cross-framework deployment.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from oceanembed.models.phys_vsa_net import PhysVSANet


def _load_model_from_checkpoint(
    checkpoint_path: Path | str,
    config_path: Path | str,
) -> tuple[PhysVSANet, object]:
    """Helper: rebuild model from config and load checkpoint weights."""
    cfg = OmegaConf.load(config_path)

    model = PhysVSANet(
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

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, cfg


def export_torchscript(
    checkpoint_path: Path | str,
    config_path: Path | str,
    output_path: Path | str,
    example_input_shape: tuple[int, ...] | None = None,
) -> Path:
    """Export model to TorchScript via ``torch.jit.trace``.

    Args:
        checkpoint_path:     Lightning checkpoint.
        config_path:         YAML config file.
        output_path:         Destination ``.pt`` file.
        example_input_shape: (B, C, H, W) for tracing.
                             Defaults to (1, in_channels, 56, 60) — PoC BoB.
    Returns:
        Path to the written file.
    """
    model, cfg = _load_model_from_checkpoint(checkpoint_path, config_path)

    if example_input_shape is None:
        C = cfg.model.in_channels
        example_input_shape = (1, C, 56, 60)

    example = torch.zeros(*example_input_shape)
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))
    print(f"TorchScript saved → {output_path}")
    return output_path


def export_onnx(
    checkpoint_path: Path | str,
    config_path: Path | str,
    output_path: Path | str,
    example_input_shape: tuple[int, ...] | None = None,
    opset_version: int = 17,
) -> Path:
    """Export model to ONNX with dynamic batch size.

    Args:
        checkpoint_path:     Lightning checkpoint.
        config_path:         YAML config file.
        output_path:         Destination ``.onnx`` file.
        example_input_shape: (B, C, H, W) for tracing.
        opset_version:       ONNX opset to use.
    Returns:
        Path to the written file.
    """
    model, cfg = _load_model_from_checkpoint(checkpoint_path, config_path)

    if example_input_shape is None:
        C = cfg.model.in_channels
        example_input_shape = (1, C, 56, 60)

    example = torch.zeros(*example_input_shape)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        example,
        str(output_path),
        opset_version    = opset_version,
        input_names      = ["surface_obs"],
        output_names     = ["temperature_3d"],
        dynamic_axes     = {
            "surface_obs":   {0: "batch"},
            "temperature_3d":{0: "batch"},
        },
        do_constant_folding = True,
    )
    print(f"ONNX saved → {output_path}  (opset {opset_version})")
    return output_path
