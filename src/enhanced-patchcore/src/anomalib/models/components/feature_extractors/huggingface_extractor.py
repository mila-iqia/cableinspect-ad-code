"""Feature extractor for ViT models."""

# Copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn
from transformers import ViTModel

logger = logging.getLogger(__name__)


class HuggingFaceFeatureExtractor(nn.Module):
    """Extract features from a ViT encoder.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import HuggingFaceFeatureExtractor
        >>> from transformers import AutoImageProcessor
        >>> input = torch.rand((32, 3, 256, 256))
        >>> model = HuggingFaceFeatureExtractor(model="facebook/vit-mae-base")
        >>> feature_extractor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> image_pt = feature_extractor(input, return_tensors="pt")
        >>> features = model(image_pt["pixel_values"])
        >>> features.shape
            torch.Size([32, 196, 768])
    """

    def __init__(self, backbone: str, requires_grad: bool = False):
        super().__init__()
        self.backbone = backbone
        self.requires_grad = requires_grad
        self.feature_extractor = self.create_vit_extractor(backbone)

    def create_vit_extractor(self, backbone):
        """Initialize and return a ViT model.

        Args:
            backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        """
        model = ViTModel.from_pretrained(backbone, output_hidden_states=True, return_dict=True)
        return model

    def forward(self, inputs: Tensor) -> dict[str, Tensor]:
        """Forward-pass input tensor into the ViT encoder.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Tensor: Feature map extracted from the ViT.
        """
        inputs_dict = {"pixel_values": inputs}
        if self.requires_grad:
            features = self.feature_extractor(**inputs_dict)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(**inputs_dict)
        return features.last_hidden_state[:, 1:, :]
