"""Towards Total Recall in Industrial Anomaly Detection.

Paper https://arxiv.org/abs/2106.08265.
"""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.data.simple_backbones import SIMPLE_BACKBONES
from anomalib.models.components import AnomalyModule
from anomalib.models.patchcore.torch_model import PatchcoreModel

logger = logging.getLogger(__name__)


class Patchcore(AnomalyModule):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        input_size (tuple[int, int]): Size of the model input.
        backbone (str): Backbone CNN network
        layers (list[str]): Layers to extract features from the backbone CNN
        backbone_config (dict | None, optional): Configurations such as patch_size and num_components for number of
            principle components to extract. Used only when backbone is 'raw' or 'pca'. Defaults to None.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Default to True.
        feature_pooler (dict | None, optional): How to pool the extracted features from the backbone CNN.
            'name' can be in ['avg', 'max']. When name is 'avg', 'count_include_pad' should also be defined.
            Default to None.
        coreset_sampling_ratio (float, optional): Coreset sampling ratio to subsample embedding.
            Defaults to 0.1.
        num_neighbors (int, optional): Number of nearest neighbors. Defaults to 9.
        fully_unsupervised (bool, optional): Whether to run a fully unsupervised experiment. Defaults to False.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        layers: list[str],
        backbone_config: dict | None = None,
        pre_trained: bool = True,
        feature_pooler: dict | None = None,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        fully_unsupervised: bool = False,
    ) -> None:
        super().__init__()

        if feature_pooler is None:
            feature_pooler = {"name": "avg", "count_include_pad": True}
        self.model: PatchcoreModel = PatchcoreModel(
            input_size=input_size,
            backbone=backbone,
            backbone_config=backbone_config,
            pre_trained=pre_trained,
            feature_pooler=feature_pooler,
            layers=layers,
            num_neighbors=num_neighbors,
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[Tensor] = []
        self.input_paths: list[Any] = []
        self.fully_unsupervised = fully_unsupervised
        self.backbone = backbone
        self.init_memory = None
        self.end_memory = None

        # Get device and properties
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_properties = torch.cuda.get_device_properties(device)

    def configure_optimizers(self) -> None:
        """Configure optimizers.

        Returns:
            None: Do not set optimizers by returning None.
        """
        return None

    def on_train_start(self) -> None:
        """Save the memory usage at the beginning of training."""
        self.init_memory = self.device_properties.total_memory - torch.cuda.max_memory_allocated()

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Generate feature embedding of the batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            dict[str, np.ndarray]: Embedding Vector
        """
        del args, kwargs  # These variables are not used.

        if self.backbone not in SIMPLE_BACKBONES:
            # No pretrained backbone is used in this case.
            self.model.feature_extractor.eval()

        embedding = self.model(batch["image"])

        # NOTE: `self.embedding` appends each batch embedding to
        #   store the training set embedding. We manually append these
        #   values mainly due to the new order of hooks introduced after PL v1.4.0
        #   https://github.com/PyTorchLightning/pytorch-lightning/pull/7357
        self.embeddings.append(embedding)

        # Save input paths. This will be used as an id to track the patches in memory bank.
        self.input_paths.append(batch["image_path"])

    def on_validation_start(self) -> None:
        """Apply subsampling to the embedding collected from the training set."""
        # NOTE: Previous anomalib versions fit subsampling at the end of the epoch.
        #   This is not possible anymore with PyTorch Lightning v1.4.0 since validation
        #   is run within train epoch.
        logger.info("Aggregating the embedding extracted from the training set.")
        embeddings = torch.vstack(self.embeddings)

        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding(embeddings, self.coreset_sampling_ratio)

        # Get index range for each input in the list of embeddings.
        batch_input_paths = [j for i in self.input_paths for j in i]
        self.model.generate_embedding_index(batch_input_paths)

        end_memory = self.device_properties.total_memory - torch.cuda.max_memory_allocated()
        logger.info(f"Total memory used by the memory bank is {(self.init_memory - end_memory) / 1024**3:.2f} GB.")

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Get batch of anomaly maps from input image batch.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename,
                image, label and mask

        Returns:
            dict[str, Any]: Image filenames, test images, GT and predicted label/masks
        """
        del args  # These variables are not used.

        mask_neighbors = True if self.fully_unsupervised else False
        if mask_neighbors and "from_predict_step" in kwargs and kwargs["from_predict_step"]:
            mask_neighbors = False

        anomaly_maps, anomaly_score = self.model(batch["image"], batch["image_path"], mask_neighbors=mask_neighbors)
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = anomaly_score

        return batch


class PatchcoreLightning(Patchcore):
    """PatchcoreLightning Module to train PatchCore algorithm.

    Args:
        hparams (DictConfig | ListConfig): Model params
    """

    def __init__(self, hparams) -> None:
        super().__init__(
            input_size=hparams.model.input_size,
            backbone=hparams.model.backbone,
            layers=hparams.model.layers,
            pre_trained=hparams.model.pre_trained,
            coreset_sampling_ratio=hparams.model.coreset_sampling_ratio,
            feature_pooler=hparams.model.feature_pooler,
            num_neighbors=hparams.model.num_neighbors,
            fully_unsupervised=hparams.metrics.fully_unsupervised,
            backbone_config=hparams.model.backbone_config,
        )
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)
