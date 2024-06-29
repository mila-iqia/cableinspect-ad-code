"""SaveNormalizedPredictionsCallback saves prediction scores after normalization.

Similar to SavePredictionsCallback. This was added for two reasons:
    1. With this new callback, we can arrange it after normalization callback such that the
    normalize method is called only once on the test set,
    2. For min max normalization, min and max are saved after normalization
"""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.data import TaskType
from anomalib.models.components import AnomalyModule
from anomalib.post_processing import VisualizationMode, Visualizer

from .cdf_normalization import CdfNormalizationCallback
from .min_max_normalization import MinMaxNormalizationCallback


class SaveNormalizedPredictionsCallback(Callback):
    """Callback that saves the image-level and pixel-level anomaly scores."""

    columns = ["image_path", "anomaly_score", "normalize_anomaly_score", "target"]

    def __init__(
        self,
        image_save_path: str,
        save_image_scores: bool,
        save_pixel_scores: bool,
        normalization: str,
        save_heatmaps: bool,
    ):
        """Define all the paths to save scores.

        Args:
            image_save_path (str): Base path to save the images
            save_image_scores (bool): Flag whether to save image level scores
            save_pixel_scores (bool): Flag whether to save pixel level scores
            normalization (str): Indicate which normalization method to use
            save_heatmaps (bool): Flag whether to save heatmaps in validation step.
        """
        self.image_save_path = Path(image_save_path)
        self.val_file_path = self.image_save_path / "validation_image_predictions.csv"
        self.test_file_path = self.image_save_path / "test_image_predictions.csv"
        self.normalization = normalization
        self.save_image_scores = save_image_scores
        self.save_pixel_scores = save_pixel_scores
        self.val_outputs = []  # type: list[dict]
        self.save_heatmaps = save_heatmaps
        self.visualizer = Visualizer(VisualizationMode.SIMPLE, TaskType.SEGMENTATION)

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the validation starts."""
        self.val_outputs = []

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, saves the predicted scores and anomaly maps."""
        keys = ["image_path", "image", "anomaly_maps", "pred_scores", "mask"]
        val_output = {k: v for k, v in outputs.items() if k in keys}
        self.val_outputs.append(val_output)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the validation ends, saves the predicted scores and anomaly maps."""
        stats = pl_module.normalization_metrics.cpu()
        stats_df = pd.DataFrame(
            [
                [
                    pl_module.image_threshold.value.cpu().item(),
                    pl_module.pixel_threshold.value.cpu().item(),
                    stats.image_min.item(),
                    stats.image_max.item(),
                    stats.pixel_min.item(),
                    stats.pixel_max.item(),
                ]
            ],
            columns=["image_threshold", "pixel_threshold", "image_min", "image_max", "pixel_min", "pixel_max"],
        )
        stats_df.to_csv(self.image_save_path / "normalization_stats.csv", index=False)
        for output in self.val_outputs:
            self.save_predictions(
                output,
                self.val_file_path,
                self.image_save_path,
                pl_module,
                "validation",
                self.save_image_scores,
                self.save_pixel_scores,
                self.normalization,
            )

            if self.save_heatmaps:
                output["pred_masks"] = output["anomaly_maps"] >= pl_module.pixel_threshold.value.cpu().item()
                output["pred_labels"] = output["pred_scores"] >= pl_module.image_threshold.value.cpu().item()
                for i, image in enumerate(self.visualizer.visualize_batch(output)):
                    filename = Path(output["image_path"][i])
                    file_path = self.image_save_path / "images" / "validation" / filename.parent.name / filename.name
                    self.visualizer.save(file_path, image)

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, saves the predicted scores and anomaly maps."""
        self.save_predictions(
            outputs,
            self.test_file_path,
            self.image_save_path,
            pl_module,
            "test",
            self.save_image_scores,
            self.save_pixel_scores,
            self.normalization,
        )

    @staticmethod
    def save_predictions(
        outputs: STEP_OUTPUT,
        file_path: Path,
        image_save_path: Path,
        pl_module: AnomalyModule,
        step: str,
        save_image_scores: bool,
        save_pixel_scores: bool,
        normalization: str,
    ):
        """Save normalized batch of predictions for test and valid.

        Args:
            outputs (STEP_OUTPUT): Outputs containing predictions
            file_path (Path): Path containing csv files that has image scores
            image_save_path (Path): Path to save pixel level scores
            pl_module (pl.LightningModule): Anomalib Model that inherits pl LightningModule
            step (str): Indicates whether it is training or validation
            save_image_scores (bool): flag whether to save image level scores
            save_pixel_scores (bool): flag whether to save pixel level scores
            normalization (str): indicate which normalization method to use (min-max or cdf)
        """
        if step == "validation":
            if normalization == "min_max":
                MinMaxNormalizationCallback._normalize_batch(outputs, pl_module)  # pylint: disable=protected-access
            elif normalization == "cdf":
                CdfNormalizationCallback._normalize_batch(outputs, pl_module)  # pylint: disable=protected-access

        if save_image_scores:
            pred_df = pd.read_csv(file_path, header=0)
            for image_path, pred_score in zip(outputs["image_path"], outputs["pred_scores"]):
                pred_df.loc[pred_df["image_path"] == image_path, "normalize_anomaly_score"] = pred_score.item()
            pred_df.to_csv(file_path, index=False)

        if save_pixel_scores:
            for image_path, anomaly_maps in zip(outputs["image_path"], outputs["anomaly_maps"]):
                filename = Path(image_path)
                dir_path = image_save_path / "pixel_predictions" / step / filename.parent.name
                pixel_file_path = (dir_path / filename.name).with_suffix(".pt")
                pixel_scores = torch.load(pixel_file_path)
                pixel_scores["normalized_anomaly_maps"] = anomaly_maps
                torch.save(pixel_scores, pixel_file_path)
