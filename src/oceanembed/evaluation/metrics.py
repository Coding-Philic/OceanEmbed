"""
Evaluation metrics for subsurface temperature reconstruction.

Provides RMSE, correlation, bias, and the standard oceanographic skill score
relative to a climatological baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SkillScore:
    """Comprehensive per-depth evaluation results.

    Attributes:
        rmse:      [K] RMSE per depth level (°C).
        bias:      [K] Mean bias per depth level (°C).
        corr:      [K] Pearson correlation coefficient per depth level.
        skill:     [K] Skill score vs. climatology:  1 - MSE_model / MSE_clim.
        depth_levels: Depth levels in metres corresponding to axis 0.
    """
    rmse:         np.ndarray
    bias:         np.ndarray
    corr:         np.ndarray
    skill:        np.ndarray
    depth_levels: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'Depth (m)':<12} {'RMSE (°C)':<12} {'Bias (°C)':<12} "
            f"{'Corr':<8} {'Skill':<8}",
            "-" * 56,
        ]
        for i, d in enumerate(self.depth_levels):
            lines.append(
                f"{int(d):<12} {self.rmse[i]:<12.4f} {self.bias[i]:<12.4f} "
                f"{self.corr[i]:<8.4f} {self.skill[i]:<8.4f}"
            )
        return "\n".join(lines)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    climatology: np.ndarray,
    mask: np.ndarray | None = None,
    depth_levels: list[float] | None = None,
    eps: float = 1e-8,
) -> SkillScore:
    """Compute per-depth RMSE, bias, correlation, and skill score.

    Args:
        pred:         [K, H, W] — model predictions.
        target:       [K, H, W] — ground-truth (e.g. GLORYS or Argo colocation).
        climatology:  [K, H, W] — reference climatology (e.g. WOA23 monthly mean).
        mask:         [H, W]    — ocean mask (1 = ocean, 0 = land / ice).
                                  If ``None``, all pixels are used.
        depth_levels: Length-K list of depth values in metres.
        eps:          Numerical stability epsilon for correlation denominator.
    Returns:
        SkillScore dataclass.
    """
    K = pred.shape[0]
    if mask is None:
        mask = np.ones(pred.shape[1:], dtype=np.float32)

    if depth_levels is None:
        depth_levels = list(range(K))

    rmse_vals  = np.zeros(K, dtype=np.float64)
    bias_vals  = np.zeros(K, dtype=np.float64)
    corr_vals  = np.zeros(K, dtype=np.float64)
    skill_vals = np.zeros(K, dtype=np.float64)

    for k in range(K):
        p = pred[k][mask > 0].ravel()
        t = target[k][mask > 0].ravel()
        c = climatology[k][mask > 0].ravel()

        diff        = p - t
        clim_diff   = c - t

        rmse_vals[k]  = float(np.sqrt(np.mean(diff ** 2)))
        bias_vals[k]  = float(np.mean(diff))

        # Pearson correlation
        p_c = p - p.mean()
        t_c = t - t.mean()
        num = float(np.sum(p_c * t_c))
        den = float(np.sqrt(np.sum(p_c ** 2) * np.sum(t_c ** 2)) + eps)
        corr_vals[k] = num / den

        # Skill score: 1 - MSE_model / MSE_climatology
        mse_model = float(np.mean(diff ** 2))
        mse_clim  = float(np.mean(clim_diff ** 2))
        skill_vals[k] = 1.0 - mse_model / (mse_clim + eps)

    return SkillScore(
        rmse         = rmse_vals,
        bias         = bias_vals,
        corr         = corr_vals,
        skill        = skill_vals,
        depth_levels = depth_levels,
    )
