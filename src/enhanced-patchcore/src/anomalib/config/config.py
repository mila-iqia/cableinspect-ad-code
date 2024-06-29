"""Get configurable parameters."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

# TODO: This would require a new design.
# TODO: https://jira.devtools.intel.com/browse/IAAALD-149

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from warnings import warn

from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoImageProcessor

from anomalib.data.task_type import TaskType
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.data.vit_backbones import VIT_BACKBONES


def _get_now_str(timestamp: float) -> str:
    """Standard format for datetimes is defined here."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")


def update_input_size_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Update config with image size as tuple, effective input size and tiling stride.

    Convert integer image size parameters into tuples, calculate the effective input size based on image size
    and crop size, and set tiling stride if undefined.

    Args:
        config (DictConfig | ListConfig): Configurable parameters object

    Returns:
        DictConfig | ListConfig: Configurable parameters with updated values
    """
    # Image size: Ensure value is in the form [height, width]

    # Do not use image_size from config when using a ViT backbone

    if config.model.backbone in VIT_BACKBONES and "image_size" in config.dataset:
        warn("image_size from the config will be ignored. image_size will be set based on the ViTConfig.")
        image_processor = AutoImageProcessor.from_pretrained(config.model.backbone)
        config.dataset.image_size = (image_processor.size["width"], image_processor.size["height"])
        config.model.input_size = config.dataset.image_size
        return config

    image_size = config.dataset.get("image_size")
    if isinstance(image_size, int):
        config.dataset.image_size = (image_size,) * 2
    elif isinstance(image_size, ListConfig):
        assert len(image_size) == 2, "image_size must be a single integer or tuple of length 2 for width and height."
    else:
        raise ValueError(f"image_size must be either int or ListConfig, got {type(image_size)}")

    # Center crop: Ensure value is in the form [height, width], and update input_size
    center_crop = config.dataset.get("center_crop")
    if center_crop is None:
        config.model.input_size = config.dataset.image_size
    elif isinstance(center_crop, int):
        config.dataset.center_crop = (center_crop,) * 2
        config.model.input_size = config.dataset.center_crop
    elif isinstance(center_crop, ListConfig):
        assert len(center_crop) == 2, "center_crop must be a single integer or tuple of length 2 for width and height."
        config.model.input_size = center_crop
    else:
        raise ValueError(f"center_crop must be either int or ListConfig, got {type(center_crop)}")

    if "tiling" in config.dataset.keys() and config.dataset.tiling.apply:
        if isinstance(config.dataset.tiling.tile_size, int):
            config.dataset.tiling.tile_size = (config.dataset.tiling.tile_size,) * 2
        if config.dataset.tiling.stride is None:
            config.dataset.tiling.stride = config.dataset.tiling.tile_size

    return config


def update_nncf_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Set the NNCF input size based on the value of the crop_size parameter in the configurable parameters object.

    Args
        config (DictConfig | ListConfig): Configurable parameters of the current run.

    Returns:
        DictConfig | ListConfig: Updated configurable parameters in DictConfig object.
    """
    crop_size = config.dataset.image_size
    sample_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
    if "optimization" in config.keys():
        if "nncf" in config.optimization.keys():
            if "input_info" not in config.optimization.nncf.keys():
                config.optimization.nncf["input_info"] = {"sample_size": None}
            config.optimization.nncf.input_info.sample_size = [1, 3, *sample_size]
            if config.optimization.nncf.apply:
                if "update_config" in config.optimization.nncf:
                    return OmegaConf.merge(config, config.optimization.nncf.update_config)
    return config


def update_multi_gpu_training_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Updates the config to change learning rate based on number of gpus assigned.

    Current behaviour is to ensure only ddp accelerator is used.

    Args:
        config (DictConfig | ListConfig): Configurable parameters for the current run

    Raises:
        ValueError: If unsupported accelerator is passed

    Returns:
        DictConfig | ListConfig: Updated config
    """
    # validate accelerator
    if config.trainer.accelerator is not None:
        if config.trainer.accelerator.lower() != "ddp":
            if config.trainer.accelerator.lower() in ("dp", "ddp_spawn", "ddp2"):
                warn(
                    f"Using accelerator {config.trainer.accelerator.lower()} is discouraged. "
                    f"Please use one of [null, ddp]. Setting accelerator to ddp"
                )
                config.trainer.accelerator = "ddp"
            else:
                raise ValueError(
                    f"Unsupported accelerator found: {config.trainer.accelerator}. Should be one of [null, ddp]"
                )
    # Increase learning rate
    # since pytorch averages the gradient over devices, the idea is to
    # increase the learning rate by the number of devices
    if "lr" in config.model:
        # Number of GPUs can either be passed as gpus: 2 or gpus: [0,1]
        n_gpus: int | list = 1
        if "trainer" in config and "gpus" in config.trainer:
            n_gpus = config.trainer.gpus
        lr_scaler = n_gpus if isinstance(n_gpus, int) else len(n_gpus)
        config.model.lr = config.model.lr * lr_scaler
    return config


def update_datasets_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Updates the dataset section of the config.

    Args:
        config (DictConfig | ListConfig): Configurable parameters for the current run.

    Returns:
        DictConfig | ListConfig: Updated config
    """
    if "format" not in config.dataset.keys():
        config.dataset.format = "mvtec"

    if "create_validation_set" in config.dataset.keys():
        warn(
            DeprecationWarning(
                "The 'create_validation_set' parameter is deprecated and will be removed in a future release. Please "
                "use 'validation_split_mode' instead."
            )
        )
        config.dataset.val_split_mode = "from_test" if config.dataset.create_validation_set else "same_as_test"

    if "test_batch_size" in config.dataset.keys():
        warn(
            DeprecationWarning(
                "The 'test_batch_size' parameter is deprecated and will be removed in a future release. Please use "
                "'eval_batch_size' instead."
            )
        )
        config.dataset.eval_batch_size = config.dataset.test_batch_size

    if "transform_config" in config.dataset.keys() and "val" in config.dataset.transform_config.keys():
        warn(
            DeprecationWarning(
                "The 'transform_config.val' parameter is deprecated and will be removed in a future release. Please "
                "use 'transform_config.eval' instead."
            )
        )
        config.dataset.transform_config.eval = config.dataset.transform_config.val

    config = update_input_size_config(config)

    if "clip_length_in_frames" in config.dataset.keys() and config.dataset.clip_length_in_frames > 1:
        warn(
            "Anomalib's models and visualizer are currently not compatible with video datasets with a clip length > 1. "
            "Custom changes to these modules will be needed to prevent errors and/or unpredictable behaviour."
        )

    if config.dataset.format == "folder" and "split_ratio" in config.dataset.keys():
        warn(
            DeprecationWarning(
                "The 'split_ratio' parameter is deprecated and will be removed in a future release. Please use "
                "'test_split_ratio' instead."
            )
        )
        config.dataset.test_split_ratio = config.dataset.split_ratio

    if config.dataset.get("test_split_mode") == TestSplitMode.NONE and config.dataset.get("val_split_mode") in (
        ValSplitMode.SAME_AS_TEST,
        ValSplitMode.FROM_TEST,
    ):
        warn(
            f"val_split_mode {config.dataset.val_split_mode} not allowed for test_split_mode = 'none'. "
            "Setting val_split_mode to 'none'."
        )
        config.dataset.val_split_mode = ValSplitMode.NONE

    if config.dataset.get("val_split_mode") == ValSplitMode.NONE and config.trainer.limit_val_batches != 0.0:
        warn("Running without validation set. Setting trainer.limit_val_batches to 0.")
        config.trainer.limit_val_batches = 0.0

    if "num_k_shot" in config.dataset.split_mode:
        if config.dataset.split_mode.get("num_k_shot") == 1:
            raise ValueError(f"num_k_shot = 1 is not feasible. Cannot have only one image in train.")
    return config


def get_configurable_parameters(
    model_name: str | None = None,
    config_path: Path | str | None = None,
    dataset_path: str | None = None,
    results_folder: str | None = None,
    weight_file: str | None = None,
    config_filename: str | None = "config",
    config_file_extension: str | None = "yaml",
) -> DictConfig | ListConfig:
    """Get configurable parameters.

    Args:
        model_name: str | None:  (Default value = None)
        config_path: Path | str | None:  (Default value = None)
        dataset_path: str | None: (Default value = None)
        results_folder: str | None: (Default value = None)
        weight_file: Path to the weight file
        config_filename: str | None:  (Default value = "config")
        config_file_extension: str | None:  (Default value = "yaml")

    Returns:
        DictConfig | ListConfig: Configurable parameters in DictConfig object.
    """
    if model_name is None is config_path:
        raise ValueError(
            "Both model_name and model config path cannot be None! "
            "Please provide a model name or path to a config file!"
        )

    if config_path is None:
        config_path = Path(f"src/anomalib/models/{model_name}/{config_filename}.{config_file_extension}")

    config = OmegaConf.load(config_path)

    # keep track of the original config file because it will be modified
    config_original: DictConfig = config.copy()

    # if the seed value is 0, notify a user that the behavior of the seed value zero has been changed.
    if config.project.get("seed") == 0:
        warn(
            "The seed value is now fixed to 0. "
            "Up to v0.3.7, the seed was not fixed when the seed value was set to 0. "
            "If you want to use the random seed, please select `None` for the seed value "
            "(`null` in the YAML file) or remove the `seed` key from the YAML file."
        )

    if dataset_path is not None:
        if "path" in config.dataset:
            raise ValueError("config.dataset.path already specify in the config.")
        config.dataset.path = dataset_path

    if results_folder is not None:
        if "path" in config.project:
            raise ValueError("config.project.path already specify in the config.")
        config.project.path = results_folder

    config = update_datasets_config(config)
    config = update_input_size_config(config)

    # Project Configs
    project_path = Path(config.project.path) / config.model.name / config.dataset.name

    if config.dataset.format == "folder":
        if "mask" in config.dataset:
            warn(
                DeprecationWarning(
                    "mask will be deprecated in favor of mask_dir in config.dataset in a future release."
                )
            )
            config.dataset.mask_dir = config.dataset.mask
        if "path" in config.dataset:
            warn(DeprecationWarning("path will be deprecated in favor of root in config.dataset in a future release."))
            config.dataset.root = config.dataset.path

    # add category subfolder if needed
    if config.dataset.format.lower() in ("btech", "mvtec", "visa"):
        project_path = project_path / config.dataset.category

    # set to False by default for backward compatibility
    config.project.setdefault("unique_dir", False)

    if "experiment_name" in config.project:
        project_path = project_path / config.project.experiment_name

    if config.project.unique_dir:
        project_path = project_path / f"run.{_get_now_str(time.time())}"

    else:
        project_path = project_path / "run"
        warn(
            "config.project.unique_dir is set to False. "
            "This does not ensure that your results will be written in an empty directory and you may overwrite files."
        )

    (project_path / "weights").mkdir(parents=True, exist_ok=True)
    (project_path / "images").mkdir(parents=True, exist_ok=True)
    # write the original config for eventual debug (modified config at the end of the function)
    (project_path / "config_original.yaml").write_text(OmegaConf.to_yaml(config_original))

    config.project.path = str(project_path)

    # loggers should write to results/model/dataset/category/ folder
    config.trainer.default_root_dir = str(project_path)

    if weight_file:
        config.trainer.resume_from_checkpoint = weight_file

    config = update_nncf_config(config)

    # thresholding
    if "metrics" in config.keys():
        # NOTE: Deprecate this once the new CLI is implemented.
        if "adaptive" in config.metrics.threshold.keys():
            warn(
                DeprecationWarning(
                    "adaptive will be deprecated in favor of method in config.metrics.threshold in a future release"
                )
            )
            config.metrics.threshold.method = "adaptive" if config.metrics.threshold.adaptive else "manual"
        if "image_default" in config.metrics.threshold.keys():
            warn(
                DeprecationWarning(
                    "image_default will be deprecated in favor of manual_image in config.metrics.threshold in a future "
                    "release."
                )
            )
            config.metrics.threshold.manual_image = (
                None if config.metrics.threshold.adaptive else config.metrics.threshold.image_default
            )
        if "pixel_default" in config.metrics.threshold.keys():
            warn(
                DeprecationWarning(
                    "pixel_default will be deprecated in favor of manual_pixel in config.metrics.threshold in a future "
                    "release."
                )
            )
            config.metrics.threshold.manual_pixel = (
                None if config.metrics.threshold.adaptive else config.metrics.threshold.pixel_default
            )

    # Model config
    model_config = config.model
    if "early_stopping" in model_config:
        if "report_objective" in model_config:
            warn("config.model.report_objective will be reset to config.model.early_stopping.metric")
        model_config.report_objective = model_config.early_stopping.metric
    elif "report_objective" not in model_config:
        warn("config.model.report_objective not set. 1. will be reported as objective for all runs.")
        model_config.report_objective = None

    # PatchCore config
    if config.model.name == "patchcore":
        if "feature_pooler" not in model_config:
            model_config.feature_pooler = {"name": "avg", "count_include_pad": True}
            warn(f"config.model.feature_pooler set to default: {model_config.feature_pooler}")

        if model_config.feature_pooler.name not in ["avg", "max"]:
            raise NotImplementedError("feature_pooler.name must be in ['avg', 'max'].")

        if model_config.feature_pooler.name == "avg" and "count_include_pad" not in model_config.feature_pooler:
            model_config.feature_pooler.count_include_pad = True
            warn("config.model.feature_pooler.count_include_pad set to default: True")

        if model_config.backbone in ["pca", "raw"]:
            if "backbone_config" not in model_config:
                raise ValueError("backbone_config not defined")
            if "patch_size" not in model_config.backbone_config:
                raise ValueError("patch_size not specified in backbone_config")
            if model_config.backbone == "raw" and "num_components" in model_config.backbone_config:
                warn("model_config.backbone is set to raw. Using raw patches, num_components will not be used.")
        else:
            model_config.backbone_config = None

    # Visualization config
    if "include" not in config.visualization:
        config.visualization.include = ["metrics", "images"]

    if "images" in config.visualization.include and config.dataset.task == TaskType.CLASSIFICATION:
        config.visualization.include.remove("images")
        warn(
            "Removing images from config.visualization.include,\
                    because it is not complatible with the classification task"
        )
    (project_path / "config.yaml").write_text(OmegaConf.to_yaml(config))

    return config
