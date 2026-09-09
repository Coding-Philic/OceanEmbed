"""
Spatial Encoder — Multi-Scale ResNet Trunk with SE Attention.

Extracts multi-scale spatial features from the 2D surface observation tensor.
No spatial downsampling is applied so mesoscale eddy information is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation Block
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Channel-wise attention via global average pooling + MLP gating.

    Args:
        channels:  Number of input / output channels.
        reduction: Bottleneck reduction ratio (hidden = channels // reduction).
    """

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Channel-reweighted tensor of shape [B, C, H, W].
        """
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)       # [B, C]
        w = self.fc(w).view(b, c, 1, 1)   # [B, C, 1, 1]
        return x * w


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """2-conv residual block with optional SE attention.

    Args:
        in_ch:        Input channels.
        out_ch:       Output channels.
        kernel_size:  Convolution kernel size (must be odd).
        use_se:       Whether to append an SE attention layer.
        se_reduction: Channel reduction ratio for SE block.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.GELU()
        self.se    = SEBlock(out_ch, reduction=se_reduction) if use_se else nn.Identity()
        # 1×1 projection to match channel dims if needed
        self.skip  = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + identity)


# ---------------------------------------------------------------------------
# Full Spatial Encoder
# ---------------------------------------------------------------------------

class SpatialEncoder(nn.Module):
    """Multi-scale ResNet encoder — no stride, preserves spatial resolution.

    Architecture
    ────────────
    Stem Conv → [Stage 1] → [Stage 2 + SE] → ... → [Stage N + SE]

    Args:
        in_channels:    Number of input channels (satellite + coordinate channels).
        embed_dim:      Output embedding dimension (final number of channels).
        stage_channels: Channel widths for each stage. The first entry is also
                        the stem output width.  Defaults to
                        ``[embed_dim//4, embed_dim//2, embed_dim]``.
        stem_kernel_size: Kernel size of the stem convolution (must be odd).
        res_kernel_size:  Kernel size for residual block convolutions.
        se_reduction:   SE block reduction ratio.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        stage_channels: list[int] | None = None,
        stem_kernel_size: int = 7,
        res_kernel_size: int = 3,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()

        if stage_channels is None:
            stage_channels = [embed_dim // 4, embed_dim // 2, embed_dim]

        assert len(stage_channels) >= 1, "Need at least one stage in stage_channels."

        stem_ch  = stage_channels[0]
        stem_pad = stem_kernel_size // 2

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, stem_ch, kernel_size=stem_kernel_size,
                      padding=stem_pad, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.GELU(),
        ]

        prev_ch = stem_ch
        for stage_idx, ch in enumerate(stage_channels):
            # Apply SE from stage index 1 onward
            use_se = stage_idx >= 1
            layers.append(ResidualBlock(prev_ch, ch,
                                        kernel_size=res_kernel_size,
                                        use_se=use_se,
                                        se_reduction=se_reduction))
            layers.append(ResidualBlock(ch, ch,
                                        kernel_size=res_kernel_size,
                                        use_se=use_se,
                                        se_reduction=se_reduction))
            prev_ch = ch

        self.encoder = nn.Sequential(*layers)
        self.out_channels = prev_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            Z: [B, embed_dim, H, W]  (full spatial resolution maintained)
        """
        return self.encoder(x)
