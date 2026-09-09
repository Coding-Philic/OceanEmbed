"""
Physics-Informed Multi-Objective Loss Engine.

Four complementary loss terms:
  L_total = L_WMSE + λ_grad · L_grad + λ_steric · L_steric + λ_corr · L_corr

  1. L_WMSE   — Depth-weighted MSE (Gaussian peak at the thermocline).
  2. L_grad   — Vertical gradient loss (preserves thermocline sharpness).
  3. L_steric — Steric height consistency (integrates α · ΔT against satellite SLA).
  4. L_corr   — Profile shape correlation (maximises Pearson r along depth axis).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OceanEmbedLoss(nn.Module):
    """Physics-informed composite loss for subsurface temperature reconstruction.

    Args:
        depth_levels:       List of target depth levels in metres (length K).
        lambda_grad:        Weight for thermocline gradient loss (λ₁).
        lambda_steric:      Weight for steric height consistency loss (λ₂).
        lambda_corr:        Weight for profile correlation loss (λ₃).
        thermocline_center: Centre depth (m) of the Gaussian weighting kernel.
        thermocline_sigma:  Width (m) of the Gaussian weighting kernel.
        depth_weight_base:  Constant additive base applied to all depths.
        depth_weight_peak:  Peak multiplicative bonus at ``thermocline_center``.
        alpha_thermal:      Thermal expansion coefficient (1/°C).
                            Use ``gsw.alpha()`` for exact TEOS-10 values in
                            production; the default is a tropical-ocean mean.
        eps:                Numerical stability epsilon for correlation loss.
    """

    def __init__(
        self,
        depth_levels: list[float],
        lambda_grad: float = 0.1,
        lambda_steric: float = 0.05,
        lambda_corr: float = 0.1,
        thermocline_center: float = 150.0,
        thermocline_sigma: float = 75.0,
        depth_weight_base: float = 1.0,
        depth_weight_peak: float = 4.0,
        alpha_thermal: float = 2.1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.lambda_grad    = lambda_grad
        self.lambda_steric  = lambda_steric
        self.lambda_corr    = lambda_corr
        self.alpha_thermal  = alpha_thermal
        self.eps            = eps

        depths = torch.tensor(depth_levels, dtype=torch.float32)
        self.register_buffer("depth_levels", depths)

        # Gaussian depth weighting: peaks at the thermocline
        weights = depth_weight_base + depth_weight_peak * torch.exp(
            -((depths - thermocline_center) ** 2) / (2.0 * thermocline_sigma ** 2)
        )
        self.register_buffer("depth_weights", weights)

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------

    def depth_weighted_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted MSE with Gaussian emphasis on the thermocline.

        Args:
            pred, target: [B, K, H, W]
            mask:         [B, H, W] or [B, 1, H, W]  ocean mask (1=ocean, 0=land).
        Returns:
            Scalar loss.
        """
        diff_sq = (pred - target) ** 2           # [B, K, H, W]
        w = self.depth_weights.view(1, -1, 1, 1) # [1, K, 1, 1]
        weighted = w * diff_sq

        if mask is not None:
            m = mask.unsqueeze(1) if mask.dim() == 3 else mask  # [B, 1, H, W]
            weighted = weighted * m
            n_valid  = m.sum() * pred.shape[1]
            return weighted.sum() / n_valid.clamp(min=1)

        return weighted.mean()

    def thermocline_gradient_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Absolute error on ∂T/∂z — preserves thermocline sharpness.

        Args:
            pred, target: [B, K, H, W]
            mask:         [B, H, W] ocean mask.
        Returns:
            Scalar loss.
        """
        depths = self.depth_levels.to(pred.device)
        dz = (depths[1:] - depths[:-1]).view(1, -1, 1, 1)  # [1, K-1, 1, 1]

        grad_pred   = (pred[:, 1:]   - pred[:, :-1])   / dz
        grad_target = (target[:, 1:] - target[:, :-1]) / dz
        loss = torch.abs(grad_pred - grad_target)

        if mask is not None:
            m = mask.unsqueeze(1) if mask.dim() == 3 else mask
            loss = loss * m
            return loss.sum() / (m.sum() * loss.shape[1]).clamp(min=1)

        return loss.mean()

    def steric_height_loss(
        self,
        pred: torch.Tensor,
        target_sla: torch.Tensor,
        t_climatology: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Steric height consistency: ∫ α · ΔT dz ≈ SLA_satellite.

        Uses trapezoidal integration over the configured depth levels.

        Args:
            pred:          [B, K, H, W] — predicted temperature.
            target_sla:    [B, 1, H, W] — satellite SLA (m).
            t_climatology: [B, K, H, W] — WOA23 climatological temperature.
            mask:          [B, H, W] ocean mask.
        Returns:
            Scalar loss.
        """
        depths = self.depth_levels.to(pred.device)

        # Trapezoidal layer thicknesses
        dz = torch.zeros_like(depths)
        dz[0]  = (depths[1]  - depths[0])  / 2.0
        dz[-1] = (depths[-1] - depths[-2]) / 2.0
        dz[1:-1] = (depths[2:] - depths[:-2]) / 2.0
        dz = dz.view(1, -1, 1, 1)

        delta_T = pred - t_climatology                              # [B, K, H, W]
        steric  = (self.alpha_thermal * delta_T * dz).sum(1, True) # [B, 1, H, W]

        loss = (steric - target_sla) ** 2
        if mask is not None:
            m = mask.unsqueeze(1) if mask.dim() == 3 else mask
            loss = loss * m
            return loss.sum() / m.sum().clamp(min=1)

        return loss.mean()

    def profile_correlation_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pearson correlation loss along the depth axis.

        Penalises shape mismatch even when absolute values are close.

        Args:
            pred, target: [B, K, H, W]
            mask:         [B, H, W] ocean mask.
        Returns:
            Scalar loss in [0, 2].
        """
        p = pred   - pred.mean(1, keepdim=True)
        t = target - target.mean(1, keepdim=True)

        num = (p * t).sum(1, keepdim=True)
        den = (
            (p ** 2).sum(1, keepdim=True).clamp(min=self.eps).sqrt()
            * (t ** 2).sum(1, keepdim=True).clamp(min=self.eps).sqrt()
        )
        corr = num / den  # [B, 1, H, W]

        if mask is not None:
            m = mask.unsqueeze(1) if mask.dim() == 3 else mask
            return ((1.0 - corr) * m).sum() / m.sum().clamp(min=1)

        return (1.0 - corr).mean()

    # ------------------------------------------------------------------
    # Combined forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sla_obs: torch.Tensor | None = None,
        t_climatology: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total composite loss.

        Args:
            pred:          [B, K, H, W] — model output.
            target:        [B, K, H, W] — GLORYS12V1 ground truth.
            sla_obs:       [B, 1, H, W] — satellite SLA (optional; enables L_steric).
            t_climatology: [B, K, H, W] — WOA23 climatology (optional; enables L_steric).
            mask:          [B, H, W] or [B, 1, H, W] — ocean mask.
        Returns:
            (total_loss, loss_dict) where loss_dict contains individual term values.
        """
        l_wmse = self.depth_weighted_mse(pred, target, mask)
        l_grad = self.thermocline_gradient_loss(pred, target, mask)
        l_corr = self.profile_correlation_loss(pred, target, mask)

        total = l_wmse + self.lambda_grad * l_grad + self.lambda_corr * l_corr

        loss_dict: dict[str, float] = {
            "loss/wmse":        l_wmse.item(),
            "loss/gradient":    l_grad.item(),
            "loss/correlation": l_corr.item(),
        }

        if sla_obs is not None and t_climatology is not None:
            l_steric = self.steric_height_loss(pred, sla_obs, t_climatology, mask)
            total = total + self.lambda_steric * l_steric
            loss_dict["loss/steric"] = l_steric.item()

        loss_dict["loss/total"] = total.item()
        return total, loss_dict
