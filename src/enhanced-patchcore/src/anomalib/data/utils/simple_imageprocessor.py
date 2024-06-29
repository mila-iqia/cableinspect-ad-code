"""Feature extractor using raw patches or PCA."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from torch import Tensor
from torchvision.transforms.functional import rgb_to_grayscale

from anomalib.models.components import PCA
from anomalib.pre_processing import Tiler

logger = logging.getLogger(__name__)


class SimpleImageProcessor:
    """Extract raw patch features or use PCA.

    Args:
        backbone (str): Name of the backbone.
        backbone_config (dict): Dict with patch_size and num_components.

    Example:
        >>> from anomalib.data.utils import SimpleImageProcessor
        >>> input = torch.rand((32, 3, 224, 224))
        >>> patch_size = 8
        >>> model = SimpleImageProcessor('raw', {'patch_size': patch_size})
        >>> features = model(input)
        >>> features.shape
            torch.Size([32, 28*28, 8*8])
    """

    def __init__(self, backbone: str, backbone_config: dict):
        self.patch_size = backbone_config["patch_size"]
        self.backbone = backbone

        if self.backbone == "pca":
            self.num_components = backbone_config["num_components"]
            self.is_pca_fit = False
            self.pca = PCA(self.num_components)

        self.tiler = Tiler(tile_size=self.patch_size, stride=self.patch_size)

    def fit(self, inputs: Tensor):
        inputs = inputs.reshape(-1, self.patch_size * self.patch_size)
        self.pca.fit(inputs)
        self.is_pca_fit = True

    def process_input(self, inputs: Tensor) -> Tensor:
        """Process input tensor to get features.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Tensor: Feature map extracted from raw image or PCA.
        """
        inputs = rgb_to_grayscale(inputs, num_output_channels=1)
        tiles = self.tiler.tile(inputs)
        tiles = tiles.squeeze()
        features = tiles.reshape(-1, self.patch_size * self.patch_size)

        if self.backbone == "pca" and self.is_pca_fit:
            features = self.pca.transform(features)

        return features
