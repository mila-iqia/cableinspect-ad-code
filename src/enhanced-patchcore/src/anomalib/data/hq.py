"""HQ dataset."""

# Copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoImageProcessor

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.hq_utils import generate_kfold_labels
from anomalib.data.simple_backbones import SIMPLE_BACKBONES
from anomalib.data.task_type import TaskType
from anomalib.data.utils import (
    InputNormalizationMethod,
    SimpleImageProcessor,
    Split,
    get_transforms,
)
from anomalib.data.vit_backbones import VIT_BACKBONES

logger = logging.getLogger(__name__)


def generate_or_load_split_labels(root: str | Path, split_mode: DictConfig) -> pd.DataFrame:
    """Generate or load HQ dataset labels based on split mode.

    Args:
        root (str | Path): Path to dataset
        split_mode (DictConfig): Split mode config

    Returns:
        samples (pd.DataFrame): Labels dataframe
    """
    if split_mode.name == "custom":
        samples = pd.read_csv(root / Path(split_mode.labels_fname))
    elif split_mode.name == "kfold":
        labels = pd.read_csv(root / Path("labels.csv"))
        samples = generate_kfold_labels(
            labels,
            cbl=split_mode.cable,
            num_train=split_mode.num_train,
            num_val=split_mode.num_val,
            anomaly_group_id=split_mode.anomaly_group_id,
            buffer=split_mode.buffer,
            num_k_shot=split_mode.num_k_shot,
        )
    else:
        raise ValueError("split_mode.name should be in [kfold, custom]")

    # Modify image_path column by converting to absolute path
    samples["image_path"] = samples["image_path"].apply(lambda x: str(root / x))
    # Modify mask_path column by converting to absolute path
    samples["mask_path"] = samples["mask_path"].apply(lambda x: str(root / x) if not pd.isnull(x) else "")

    return samples


def make_hq_dataset(
    root: str | Path, split_mode: dict, toy: bool = False, split: str | Split | None = None
) -> pd.DataFrame:
    """Create HQ samples from HQ labels.

    Args:
        root (str | Path): Path to dataset
        split_mode (dict): Split mode config
        toy (bool, optional): Whether or not to use the toy version of the dataset. Defaults to False.
        split (str | Split | None, optional): Dataset split (ie., either train, val or test). Defaults to None.

    Returns:
        samples (pd.DataFrame): An output dataframe containing samples for the requested split
            (ie., train, val or test).
    """
    samples = generate_or_load_split_labels(root=root, split_mode=split_mode)

    # Get the data frame for the split.
    if split is not None and split in ["train", "val", "test"]:
        samples = samples[samples.split == split]

    # Get toy dataset samples.
    # The toy dataset corresponds to samples from the side A first pass on all available cables.
    if toy:
        samples = samples[(samples["side_id"] == "A") & (samples["pass_id"] == 1)]

    samples.reset_index(drop=True, inplace=True)

    return samples


class HQDataset(AnomalibDataset):
    """HQ dataset class.

    Args:
        task (TaskType): Task type, ``classification`` or ``segmentation``
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        root (Path | str): Path to the root of the dataset
        split_mode (dict): Split mode config
        toy (bool, optional): Whether or not to use the toy version of the dataset. Defaults to False.
        split (str | Split | None, optional): Split of the dataset, usually Split.TRAIN or Split.TEST. Defaults to None.
        backbone (str | None, optional): Backbone for feature extraction. Defaults to None.
        image_processor (AutoImageProcessor | None, optional): Image processor for ViT backbones. Defaults to None.
    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        root: Path | str,
        split_mode: dict,
        toy: bool = False,
        split: str | Split | None = None,
        backbone: str | None = None,
        image_processor: AutoImageProcessor | None = None,
    ) -> None:
        super().__init__(task=task, transform=transform, backbone=backbone, image_processor=image_processor)

        self.root = Path(root)
        self.split_mode = split_mode
        self.toy = toy
        self.split = split

    def _setup(self) -> None:
        self.samples = make_hq_dataset(self.root, split_mode=self.split_mode, split=self.split, toy=self.toy)


class HQ(AnomalibDataModule):
    """HQ Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset
        split_mode (dict): Split mode config
        toy (bool, optional): Whether or not to use the toy version of the dataset.
            Defaults to False
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to None.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions. Defaults to None.
        normalization (str | InputNormalizationMethod, optional): Normalization method, none, 'imagenet'.
            Defaults to InputNormalizationMethod.IMAGENET.
        train_batch_size (int, optional): Training batch size. Defaults to 32.
        eval_batch_size (int, optional): Test batch size. Defaults to 32.
        num_workers (int, optional): Number of workers. Defaults to 8.
        task (TaskType, optional): Task type, 'classification' or 'segmentation'
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training. Defaults to None.
        transform_config_eval (str | A.Compose | None, optional): Config for pre-processing
            during validation. Defaults to None.
        fully_unsupervised (bool): Whether to run in fully unsupervised mode by running validation on train set.
        backbone (str | None, optional): Backbone used to generate embeddings. Defaults to None.
        backbone_config (dict | None, optional): Backbone configuration for pca or raw backbones. Defaults to None.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to None.
    """

    def __init__(
        self,
        root: Path | str,
        split_mode: dict,
        toy: bool = False,
        image_size: int | tuple[int, int] | None = None,
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        fully_unsupervised: bool = False,
        backbone: str | None = None,
        backbone_config: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.root = Path(root)
        self.split_mode = split_mode
        self.toy = toy
        self.task = task
        self.backbone = backbone
        self.backbone_config = backbone_config

        if self.backbone in VIT_BACKBONES:
            self.image_processor = AutoImageProcessor.from_pretrained(backbone)

            # When a ViT backbone is used the image transforms hp are ignored and
            # it's done using same pre-processing as the one used for the ViT pre-training.
            self.transform_train = self.transform_eval = None
        else:
            self.image_processor = None
            self.transform_train = get_transforms(
                config=transform_config_train,
                image_size=image_size,
                center_crop=center_crop,
                normalization=InputNormalizationMethod(normalization),
            )
            self.transform_eval = get_transforms(
                config=transform_config_eval,
                image_size=image_size,
                center_crop=center_crop,
                normalization=InputNormalizationMethod(normalization),
            )

            if self.backbone in SIMPLE_BACKBONES:
                self.image_processor = SimpleImageProcessor(self.backbone, self.backbone_config)  # type: ignore

        self.fully_unsupervised = fully_unsupervised

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting."""

        self.train_data = HQDataset(
            task=self.task,
            transform=self.transform_train,
            split=Split.TRAIN,
            root=self.root,
            split_mode=self.split_mode,
            toy=self.toy,
            backbone=self.backbone,
            image_processor=self.image_processor,
        )
        self.train_data.setup()

        if self.backbone == "pca":
            images = [self.train_data[i]["image"] for i in range(self.train_data.__len__())]
            images = torch.stack(images)
            self.image_processor.fit(images)

        val_split = Split.TRAIN if self.fully_unsupervised else Split.VAL
        self.val_data = HQDataset(
            task=self.task,
            transform=self.transform_eval,
            split=val_split,
            root=self.root,
            split_mode=self.split_mode,
            toy=self.toy,
            backbone=self.backbone,
            image_processor=self.image_processor,
        )
        self.val_data.setup()

        self.test_data = HQDataset(
            task=self.task,
            transform=self.transform_eval,
            split=Split.TEST,
            root=self.root,
            split_mode=self.split_mode,
            toy=self.toy,
            backbone=self.backbone,
            image_processor=self.image_processor,
        )
        self.test_data.setup()
