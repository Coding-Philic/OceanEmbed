"""OceanEmbed — Satellite embedding-based 3D subsurface ocean temperature reconstruction."""

__version__ = "0.1.0"
__author__ = "OceanEmbed Team"

from oceanembed.models.phys_vsa_net import PhysVSANet
from oceanembed.models.losses import OceanEmbedLoss
from oceanembed.data.dataset import OceanEmbedDataset, OceanEmbedDataModule

__all__ = [
    "PhysVSANet",
    "OceanEmbedLoss",
    "OceanEmbedDataset",
    "OceanEmbedDataModule",
]
