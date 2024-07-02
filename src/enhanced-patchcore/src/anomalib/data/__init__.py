"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import Enum

from omegaconf import DictConfig, ListConfig

from .base import AnomalibDataModule, AnomalibDataset
from .hq import HQ
from .inference import InferenceDataset
from .task_type import TaskType

logger = logging.getLogger(__name__)


class DataFormat(str, Enum):
    """Supported Dataset Types"""
    HQ = "hq"


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    datamodule: AnomalibDataModule

    # convert center crop to tuple
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = (center_crop[0], center_crop[1])

    if config.dataset.format.lower() == DataFormat.HQ:
        datamodule = HQ(
            root=config.dataset.path,
            toy=config.dataset.toy,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            split_mode=config.dataset.split_mode,
            fully_unsupervised=config.metrics.fully_unsupervised,
            backbone=config.model.backbone,
            backbone_config=config.model.backbone_config,
        )
    else:
        raise ValueError(
            "Unknown dataset! \n"
            "If you use a custom dataset make sure you initialize it in"
            "`get_datamodule` in `anomalib.data.__init__.py"
        )

    return datamodule


__all__ = [
    "AnomalibDataset",
    "AnomalibDataModule",
    "get_datamodule",
    "InferenceDataset",
    "TaskType",
    "HQ",
]
