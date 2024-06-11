"""Feature extractors."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .dinov2 import DinoV2FeatureExtractor
from .huggingface_extractor import HuggingFaceFeatureExtractor
from .timm import FeatureExtractor, TimmFeatureExtractor
from .torchfx import BackboneParams, TorchFXFeatureExtractor
from .utils import dryrun_find_featuremap_dims

__all__ = [
    "BackboneParams",
    "DinoV2FeatureExtractor",
    "dryrun_find_featuremap_dims",
    "FeatureExtractor",
    "HuggingFaceFeatureExtractor",
    "TimmFeatureExtractor",
    "TorchFXFeatureExtractor",
]
