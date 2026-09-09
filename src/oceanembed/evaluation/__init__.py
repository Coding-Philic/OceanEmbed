"""Evaluation: metrics, Argo matchup, and benchmarks."""

from oceanembed.evaluation.metrics import compute_metrics, SkillScore
from oceanembed.evaluation.argo_matchup import ArgoMatchup

__all__ = ["compute_metrics", "SkillScore", "ArgoMatchup"]
