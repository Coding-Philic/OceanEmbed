"""
Volumetric Reconstruction Head.

Projects depth-level feature vectors back into spatially resolved 2D temperature
fields and stacks them into a 3D [B, K, H, W] output volume.

Fusion strategy: Hadamard (element-wise) product of depth embedding (broadcast)
with the spatial feature map, then a shared conv-MLP projects to temperature.
This allows the depth embedding to act as a learned, depth-specific spatial gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VolumetricReconstructionHead(nn.Module):
    """Per-depth spatial projection: (depth embedding × spatial features) → T(x,y,z).

    Args:
        embed_dim:       Input channel dimension (from encoder / decoder).
        num_depths:      Number of target depth levels (K).
        head_channels:   Ordered list of intermediate channel widths for the
                         shared projection conv-MLP.  Defaults to
                         ``[embed_dim // 2, embed_dim // 4]``.
        head_kernel_size: Conv kernel size for projection layers (must be odd).
    """

    def __init__(
        self,
        embed_dim: int,
        num_depths: int,
        head_channels: list[int] | None = None,
        head_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.num_depths = num_depths

        if head_channels is None:
            head_channels = [max(1, embed_dim // 2), max(1, embed_dim // 4)]

        pad = head_kernel_size // 2
        proj_layers: list[nn.Module] = []
        prev_ch = embed_dim
        for ch in head_channels:
            proj_layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=head_kernel_size, padding=pad),
                nn.GELU(),
            ])
            prev_ch = ch
        # Final 1×1 conv → single temperature value per pixel
        proj_layers.append(nn.Conv2d(prev_ch, 1, kernel_size=1))

        self.spatial_proj = nn.Sequential(*proj_layers)

    def forward(
        self,
        depth_features: torch.Tensor,
        spatial_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depth_features:   [B, K, C] — per-depth embeddings from the decoder.
            spatial_features: [B, C, H, W] — spatial feature map from the encoder.
        Returns:
            T_pred: [B, K, H, W] — reconstructed 3D temperature field (°C).
        """
        B, K, C = depth_features.shape
        _, _, H, W = spatial_features.shape

        depth_slices: list[torch.Tensor] = []
        for k in range(K):
            # Gate the spatial map with the k-th depth embedding: [B, C, 1, 1]
            gate = depth_features[:, k, :, None, None]
            fused = spatial_features * gate   # [B, C, H, W]
            T_k = self.spatial_proj(fused)    # [B, 1, H, W]
            depth_slices.append(T_k)

        return torch.cat(depth_slices, dim=1)  # [B, K, H, W]
