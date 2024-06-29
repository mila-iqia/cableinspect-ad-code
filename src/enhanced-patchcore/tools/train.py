"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from argparse import ArgumentParser, Namespace

from orion.client.cli import report_objective
from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import (
    LoadModelCallback,
    SaveNormalizedPredictionsCallback,
    SavePredictionsCallback,
    get_callbacks,
)
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--dataset-path", type=str, required=False, help="Path to dataset.")
    parser.add_argument("--results-folder", type=str, required=False, help="Path to the results folder.")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def train(args: Namespace):
    """Train an anomaly model.

    Args:
        args (Namespace): The arguments from the command line.
    """

    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(
        model_name=args.model,
        config_path=args.config,
        results_folder=args.results_folder,
        dataset_path=args.dataset_path,
    )
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)

    save_predictions_config = None
    trainable_models = [
        "autoencoder",
        "cfa",
        "cflow",
        "csflow",
        "draem",
        "fastflow",
        "ganomaly",
        "reverse_distillation",
        "stfpm",
    ]
    if config.model.name in trainable_models:
        # Extract save predictions config for model that need training.
        if "save_predictions" in config.project:
            save_predictions_config = config.project.pop("save_predictions")

    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    report_objective_metric = config.model.report_objective
    if report_objective_metric is not None:
        logged_metrics = trainer.logged_metrics

    if save_predictions_config is not None:
        callback_names = [callback.__class__.__name__ for callback in trainer.callbacks]  # pylint: disable=no-member
        save_heatmaps = True if "images" in config.visualization.include else False
        # Add save prediction callback after metrics definition.
        idx = callback_names.index("MetricsConfigurationCallback")
        save_prediction_callback = SavePredictionsCallback(
            config.project.path, save_predictions_config.image_scores, save_predictions_config.pixel_scores
        )
        callback_names.insert(idx + 1, "SavePredictionsCallback")
        trainer.callbacks.insert(idx + 1, save_prediction_callback)  # pylint: disable=no-member

        normalization_callback_name = list(
            filter(lambda callback_name: "NormalizationCallback" in callback_name, callback_names)
        )
        if len(normalization_callback_name) > 1:
            raise ValueError(f"Multiple normalization callbacks have been found: {normalization_callback_name}.")
        if normalization_callback_name:
            # Add save prediction callback after normalization.
            idx = callback_names.index(normalization_callback_name[0])
            save_normalized_prediction_callback = SaveNormalizedPredictionsCallback(
                config.project.path,
                save_predictions_config.image_scores,
                save_predictions_config.pixel_scores,
                config.model.normalization_method,
                save_heatmaps,
            )
            trainer.callbacks.insert(idx + 1, save_normalized_prediction_callback)  # pylint: disable=no-member

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

    if config.model.name in trainable_models:
        # Compute scores and save visualization for best model on validation set.
        logger.info("Testing best model on validation set.")
        trainer.validate(model=model, datamodule=datamodule)
        if report_objective_metric is not None:
            # Update logged metrics
            logged_metrics = trainer.logged_metrics

    logger.info("Testing best model on test set.")
    trainer.test(model=model, datamodule=datamodule)

    if report_objective_metric is not None:
        score = float(logged_metrics[report_objective_metric].cpu().numpy())
    else:
        # No metrics has been specified, report "fake" objective
        score = 1.0
    return report_objective(score)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
