"""Feature extractor for DinoV2 model."""

# Copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class DinoV2FeatureExtractor(nn.Module):
    """Extract features from DinoV2.

    Args:
        backbone (str): Backbone name.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.

    Example:
        >>> import torch
        >>> from anomalib.models.components import DinoV2FeatureExtractor

        >>> inputs = torch.rand((32, 3, 224, 224))
        >>> model = DinoV2FeatureExtractor(model="dinov2_vits14")
        >>> features = model(inputs)

        >>> features.shape
            torch.Size([32, 256, 384])
    """

    def __init__(self, backbone: str, requires_grad: bool = False):
        super().__init__()
        self.backbone = backbone
        self.requires_grad = requires_grad
        self.feature_extractor = self.create_dinov2_extractor(backbone)

    def create_dinov2_extractor(self, backbone):
        """Initialize and return a DinoV2 model.

        Args:
            backbone (str): Backbone name.
        """
        model = torch.hub.load("facebookresearch/dinov2", backbone)
        return model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward-pass input tensor into DinoV2

        Args:
            inputs (Tensor): Input tensor

        Returns:
            features (Tensor): Feature maps extracted from DinoV2.
        """
        if self.requires_grad:
            features = self.feature_extractor.forward_features(inputs)["x_norm_patchtokens"]
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor.forward_features(inputs)["x_norm_patchtokens"]
        return features
