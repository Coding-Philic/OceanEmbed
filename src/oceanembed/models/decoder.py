"""
Cross-Depth Transformer Decoder.

Depth levels are modelled as a learnable query sequence.
Each depth query attends to all spatial surface tokens (cross-attention),
then depth queries attend to each other (self-attention) to enforce
vertical baroclinic coupling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Single Transformer Decoder Block
# ---------------------------------------------------------------------------

class DepthCrossAttentionBlock(nn.Module):
    """One transformer decoder layer: cross-attention → self-attention → FFN.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Number of attention heads (embed_dim must be divisible).
        mlp_ratio: FFN hidden dimension = embed_dim * mlp_ratio.
        dropout:   Dropout probability applied inside attention and FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        mlp_dim = int(embed_dim * mlp_ratio)

        # 1. Cross-attention: depth queries ← spatial surface tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # 2. Self-attention: depth queries interact (vertical coupling)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # 3. Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        depth_queries: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            depth_queries:  [B, K, D]  — K depth-level query vectors.
            spatial_tokens: [B, N, D]  — N = H × W flattened spatial tokens.
        Returns:
            Updated depth features [B, K, D].
        """
        # --- Cross-attention ---
        ca_out, _ = self.cross_attn(
            query=depth_queries,
            key=spatial_tokens,
            value=spatial_tokens,
        )
        x = self.norm1(depth_queries + ca_out)

        # --- Self-attention (vertical coupling) ---
        sa_out, _ = self.self_attn(query=x, key=x, value=x)
        x = self.norm2(x + sa_out)

        # --- FFN ---
        x = self.norm3(x + self.ffn(x))

        return x


# ---------------------------------------------------------------------------
# Stacked Decoder
# ---------------------------------------------------------------------------

class CrossDepthTransformerDecoder(nn.Module):
    """Stack of ``num_layers`` DepthCrossAttentionBlock modules.

    Learnable depth query embeddings replace the positional encodings typical in
    DETR-style decoders; each query specialises for its target depth level.

    Args:
        num_depths: Number of target depth levels (K).
        embed_dim:  Token embedding dimension.
        num_heads:  Attention heads per block.
        num_layers: Number of stacked decoder blocks.
        dropout:    Dropout probability.
        mlp_ratio:  FFN hidden-dim multiplier.
        init_std:   Std-dev for depth query parameter initialisation.
    """

    def __init__(
        self,
        num_depths: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        mlp_ratio: float = 2.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        # One learnable embedding per depth level
        self.depth_queries = nn.Parameter(
            torch.randn(1, num_depths, embed_dim) * init_std
        )

        self.layers = nn.ModuleList([
            DepthCrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: [B, C, H, W] — output of the spatial encoder.
        Returns:
            depth_features: [B, K, C] — one feature vector per depth level.
        """
        B, C, H, W = spatial_features.shape

        # Flatten spatial dims → sequence for attention: [B, H*W, C]
        spatial_tokens = spatial_features.flatten(2).transpose(1, 2)

        # Broadcast depth queries to batch: [B, K, C]
        depth_q = self.depth_queries.expand(B, -1, -1)

        for layer in self.layers:
            depth_q = layer(depth_q, spatial_tokens)

        return depth_q  # [B, K, C]
