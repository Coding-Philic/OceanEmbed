"""
Unit tests for model components.

Tests cover:
  - Output shapes and dtypes for encoder, decoder, head, full model.
  - No hardcoded shapes — all derived from parametric configs.
  - Loss forward passes with and without optional terms.
"""

from __future__ import annotations

import pytest
import torch

from oceanembed.models.encoder       import SEBlock, ResidualBlock, SpatialEncoder
from oceanembed.models.decoder       import DepthCrossAttentionBlock, CrossDepthTransformerDecoder
from oceanembed.models.head          import VolumetricReconstructionHead
from oceanembed.models.losses        import OceanEmbedLoss
from oceanembed.models.phys_vsa_net  import PhysVSANet


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def poc_config():
    """Minimal PoC config values (no external files needed)."""
    from types import SimpleNamespace
    model = SimpleNamespace(
        in_channels=11, num_depths=6, embed_dim=64,
        num_heads=4, num_decoder_layers=2, dropout=0.0,
    )
    return SimpleNamespace(model=model)


@pytest.fixture
def depth_levels_poc():
    return [0.0, 50.0, 100.0, 150.0, 200.0, 500.0]


@pytest.fixture
def B():
    return 2   # batch size


@pytest.fixture
def HW():
    return (14, 15)  # small spatial grid for speed


# ── SEBlock ────────────────────────────────────────────────────────────────────

class TestSEBlock:
    def test_output_shape(self, B, HW):
        H, W = HW
        C = 32
        block = SEBlock(channels=C, reduction=8)
        x = torch.randn(B, C, H, W)
        out = block(x)
        assert out.shape == (B, C, H, W)

    def test_output_range(self, B, HW):
        """SE weights in Sigmoid → output magnitude ≤ input magnitude."""
        H, W = HW
        C = 16
        block = SEBlock(channels=C, reduction=4)
        x = torch.ones(B, C, H, W)
        out = block(x)
        assert out.abs().max().item() <= 1.0 + 1e-5


# ── ResidualBlock ─────────────────────────────────────────────────────────────

class TestResidualBlock:
    @pytest.mark.parametrize("in_ch,out_ch,use_se", [
        (16, 16, False), (16, 32, True), (32, 64, False),
    ])
    def test_output_shape(self, B, HW, in_ch, out_ch, use_se):
        H, W = HW
        block = ResidualBlock(in_ch=in_ch, out_ch=out_ch, use_se=use_se, se_reduction=4)
        x = torch.randn(B, in_ch, H, W)
        out = block(x)
        assert out.shape == (B, out_ch, H, W)


# ── SpatialEncoder ─────────────────────────────────────────────────────────────

class TestSpatialEncoder:
    @pytest.mark.parametrize("in_ch,embed_dim", [(11, 64), (7, 32)])
    def test_output_shape(self, B, HW, in_ch, embed_dim):
        H, W = HW
        enc = SpatialEncoder(in_channels=in_ch, embed_dim=embed_dim)
        x   = torch.randn(B, in_ch, H, W)
        out = enc(x)
        assert out.shape == (B, embed_dim, H, W), \
            f"Expected [B,{embed_dim},{H},{W}], got {out.shape}"

    def test_custom_stage_channels(self, B, HW):
        H, W = HW
        enc = SpatialEncoder(
            in_channels=11, embed_dim=48,
            stage_channels=[16, 32, 48],
        )
        x   = torch.randn(B, 11, H, W)
        out = enc(x)
        assert out.shape == (B, 48, H, W)

    def test_no_spatial_downsampling(self, B):
        """Encoder must preserve H and W exactly (stride=1 everywhere)."""
        H, W = 20, 25
        enc = SpatialEncoder(in_channels=11, embed_dim=32)
        out = enc(torch.randn(B, 11, H, W))
        assert out.shape[2] == H and out.shape[3] == W


# ── CrossDepthTransformerDecoder ──────────────────────────────────────────────

class TestDecoder:
    def test_output_shape(self, B, HW):
        H, W   = HW
        K      = 6
        D      = 32
        dec    = CrossDepthTransformerDecoder(
            num_depths=K, embed_dim=D, num_heads=4,
            num_layers=2, dropout=0.0, mlp_ratio=2.0,
        )
        spatial = torch.randn(B, D, H, W)
        out     = dec(spatial)
        assert out.shape == (B, K, D)

    def test_gradient_flows(self, B, HW):
        H, W   = HW
        K, D   = 4, 16
        dec    = CrossDepthTransformerDecoder(
            num_depths=K, embed_dim=D, num_heads=2,
            num_layers=1, dropout=0.0,
        )
        spatial = torch.randn(B, D, H, W, requires_grad=True)
        out     = dec(spatial).sum()
        out.backward()
        assert spatial.grad is not None


# ── VolumetricReconstructionHead ─────────────────────────────────────────────

class TestHead:
    def test_output_shape(self, B, HW):
        H, W = HW
        K, D = 6, 32
        head = VolumetricReconstructionHead(embed_dim=D, num_depths=K)
        depth_feats   = torch.randn(B, K, D)
        spatial_feats = torch.randn(B, D, H, W)
        out = head(depth_feats, spatial_feats)
        assert out.shape == (B, K, H, W)


# ── OceanEmbedLoss ────────────────────────────────────────────────────────────

class TestLoss:
    @pytest.fixture
    def loss_fn(self, depth_levels_poc):
        return OceanEmbedLoss(
            depth_levels = depth_levels_poc,
            lambda_grad  = 0.1,
            lambda_steric= 0.05,
            lambda_corr  = 0.1,
        )

    def test_loss_scalar(self, loss_fn, B, HW):
        H, W = HW
        K = 6
        pred   = torch.randn(B, K, H, W)
        target = torch.randn(B, K, H, W)
        total, ldict = loss_fn(pred, target)
        assert total.ndim == 0
        assert total.item() >= 0.0
        assert "loss/total" in ldict

    def test_with_mask(self, loss_fn, B, HW):
        H, W = HW
        K = 6
        pred   = torch.randn(B, K, H, W)
        target = torch.randn(B, K, H, W)
        mask   = torch.ones(B, H, W)
        mask[:, :H//2, :] = 0.0   # blank out half the pixels
        total, _ = loss_fn(pred, target, mask=mask)
        assert total.item() >= 0.0

    def test_with_steric(self, loss_fn, B, HW):
        H, W = HW
        K = 6
        pred    = torch.randn(B, K, H, W)
        target  = torch.randn(B, K, H, W)
        sla     = torch.randn(B, 1, H, W) * 0.05
        t_clim  = torch.randn(B, K, H, W)
        total, ldict = loss_fn(pred, target, sla_obs=sla, t_climatology=t_clim)
        assert "loss/steric" in ldict

    def test_zero_loss_on_perfect_pred(self, depth_levels_poc, B, HW):
        """L_WMSE and L_grad should be zero when pred == target."""
        loss_fn = OceanEmbedLoss(
            depth_levels=depth_levels_poc,
            lambda_corr=0.0,   # corr = 1 → 1-corr = 0 only if pred = target
        )
        H, W = HW
        K = 6
        t = torch.randn(B, K, H, W)
        total, ldict = loss_fn(t, t)
        assert ldict["loss/wmse"]     < 1e-6
        assert ldict["loss/gradient"] < 1e-6


# ── Full PhysVSANet ───────────────────────────────────────────────────────────

class TestPhysVSANet:
    @pytest.fixture
    def model(self):
        return PhysVSANet(
            in_channels=11, num_depths=6, embed_dim=32,
            num_heads=4, num_decoder_layers=2, dropout=0.0,
        )

    def test_output_shape(self, model, B, HW):
        H, W = HW
        x   = torch.randn(B, 11, H, W)
        out = model(x)
        assert out.shape == (B, 6, H, W)

    def test_count_parameters(self, model):
        params = model.count_parameters()
        assert params["total"] > 0
        assert params["total"] == params["encoder"] + params["decoder"] + params["head"]

    def test_gradient_e2e(self, model, depth_levels_poc, B, HW):
        """End-to-end gradient check through the full model + loss."""
        H, W   = HW
        x      = torch.randn(B, 11, H, W, requires_grad=True)
        target = torch.randn(B, 6, H, W)
        loss_fn = OceanEmbedLoss(depth_levels=depth_levels_poc)
        pred    = model(x)
        loss, _ = loss_fn(pred, target)
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum().item() > 0.0

    def test_inference_mode_no_grad(self, model, B, HW):
        H, W = HW
        with torch.no_grad():
            x   = torch.randn(B, 11, H, W)
            out = model(x)
        assert out.shape == (B, 6, H, W)
