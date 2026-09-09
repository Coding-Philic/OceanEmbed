"""
Phys-VSA-Net — Full model assembly.

Encoder → Cross-Depth Transformer Decoder → Volumetric Reconstruction Head

All hyper-parameters are injected; no hardcoded values in the forward path.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from oceanembed.models.encoder import SpatialEncoder
from oceanembed.models.decoder import CrossDepthTransformerDecoder
from oceanembed.models.head    import VolumetricReconstructionHead


class PhysVSANet(nn.Module):
    """Phys-VSA-Net: Physics-informed Volumetric Spatial-Attention Network.

    Args:
        in_channels:       Number of input channels (satellite + coord channels).
        num_depths:        Number of target depth levels (K).
        embed_dim:         Shared embedding dimension throughout the network.
        num_heads:         Attention heads in each transformer decoder block.
        num_decoder_layers: Number of stacked decoder blocks.
        dropout:           Dropout probability in the decoder.
        stage_channels:    Per-stage channel widths for the encoder
                           (see ``SpatialEncoder``).  Defaults to
                           ``[embed_dim//4, embed_dim//2, embed_dim]``.
        head_channels:     Intermediate channel widths for the reconstruction
                           head (see ``VolumetricReconstructionHead``).
                           Defaults to ``[embed_dim//2, embed_dim//4]``.
        stem_kernel_size:  Encoder stem convolution kernel size.
        res_kernel_size:   Encoder residual block kernel size.
        head_kernel_size:  Reconstruction head convolution kernel size.
        se_reduction:      SE block reduction ratio in the encoder.
        mlp_ratio:         FFN hidden-dim multiplier in the decoder.
        init_std:          Depth query parameter initialisation std-dev.
    """

    def __init__(
        self,
        in_channels: int,
        num_depths: int,
        embed_dim: int,
        num_heads: int,
        num_decoder_layers: int,
        dropout: float,
        stage_channels: list[int] | None = None,
        head_channels: list[int] | None = None,
        stem_kernel_size: int = 7,
        res_kernel_size: int = 3,
        head_kernel_size: int = 3,
        se_reduction: int = 16,
        mlp_ratio: float = 2.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.encoder = SpatialEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            stage_channels=stage_channels,
            stem_kernel_size=stem_kernel_size,
            res_kernel_size=res_kernel_size,
            se_reduction=se_reduction,
        )

        self.decoder = CrossDepthTransformerDecoder(
            num_depths=num_depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            init_std=init_std,
        )

        self.head = VolumetricReconstructionHead(
            embed_dim=embed_dim,
            num_depths=num_depths,
            head_channels=head_channels,
            head_kernel_size=head_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W] — normalised satellite surface observations.
        Returns:
            T_pred: [B, K, H, W] — 3D temperature reconstruction (°C).
        """
        spatial_features = self.encoder(x)              # [B, C, H, W]
        depth_features   = self.decoder(spatial_features)  # [B, K, C]
        T_pred           = self.head(depth_features, spatial_features)  # [B, K, H, W]
        return T_pred

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts per sub-module."""
        def _count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "encoder": _count(self.encoder),
            "decoder": _count(self.decoder),
            "head":    _count(self.head),
            "total":   _count(self),
        }
