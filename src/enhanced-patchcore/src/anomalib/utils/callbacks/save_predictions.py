"""SavePredictionsCallback saves the predictions before normalization."""

# Copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule


class SavePredictionsCallback(Callback):
    """Callback that saves the image-level and pixel-level anomaly scores."""

    columns = ["image_path", "anomaly_score", "normalize_anomaly_score", "target"]

    def __init__(self, image_save_path: str, save_image_scores: bool, save_pixel_scores: bool):
        """Define all the paths to save files.

        Args:
            image_save_path (str): Base path to save the images
            save_image_scores (bool): Flag whether to save image level scores
            save_pixel_scores (bool): Flag whether to save pixel level scores
        """
        self.image_save_path = Path(image_save_path)
        self.val_file_path = self.image_save_path / "validation_image_predictions.csv"
        self.test_file_path = self.image_save_path / "test_image_predictions.csv"
        self.save_image_scores = save_image_scores
        self.save_pixel_scores = save_pixel_scores

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the validation starts."""
        # Create the csv with columns and save it
        if self.save_image_scores:
            val_df = pd.DataFrame(columns=self.columns)
            val_df.to_csv(self.val_file_path, columns=self.columns, index=False)

        if self.save_pixel_scores:
            os.makedirs(self.image_save_path / "pixel_predictions" / "validation", exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the validation epoch ends"""
        pred_df = pd.read_csv(self.val_file_path)
        groups = pred_df.groupby(["target"])["anomaly_score"].apply(list).tolist()
        label_groups = ["nominal"]
        if len(groups) == 2:
            label_groups.append("anomalous")
        min_score = pred_df["anomaly_score"].min()
        max_score = pred_df["anomaly_score"].max()
        bin_range = int(max_score + 0.5) - int(min_score - 0.5) + 1
        bin_width = 1
        if bin_range > 15:
            bin_width = bin_range // 15
        bins = [i for i in range(int(min_score - 0.5), int(max_score + 0.5) + bin_width + 1, bin_width)]
        image_threshold = pl_module.image_threshold.value.cpu().item()
        image_save_file = self.image_save_path / "validation_anomaly_score_histogram.png"
        self.plot_histogram(bins, groups, label_groups, image_threshold, "", 22, image_save_file)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test starts."""
        if self.save_pixel_scores:
            os.makedirs(self.image_save_path / "pixel_predictions" / "test", exist_ok=True)

        if self.save_image_scores:
            test_df = pd.DataFrame(columns=self.columns)
            test_df.to_csv(self.test_file_path, columns=self.columns, index=False)

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, saves the  predicted scores and anomaly maps."""
        self.save_predictions(
            outputs,
            self.val_file_path,
            self.image_save_path,
            "validation",
            self.save_image_scores,
            self.save_pixel_scores,
        )

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
            outputs, self.test_file_path, self.image_save_path, "test", self.save_image_scores, self.save_pixel_scores
        )

    @staticmethod
    def plot_histogram(
        bins: list,
        x: list,
        x_labels: list,
        x_threshold: float,
        legend_title: str,
        fontsize: int,
        file_path: Path,
    ) -> None:
        """Plot histogram of the scores.

        Args:
            bins (list): List of bins to use to build the histogram.
            x (list): Input values, sequence of list or arrays of scores which are not required to be of same length.
            x_labels (list): Labels, corresponding sequence of score labels.
            x_threshold (float): Threshold for the scores.
            legend_title (str): Legend title.
            fontsize (int): Font size for all text.
            file_path (Path): Path to save the image.
        """
        plt.figure(figsize=(20, 7), dpi=80)

        # Add color for anomalous image score if available.
        color = ["tab:blue"]
        if "anomalous" in x_labels:
            color.append("tab:orange")

        # Plot hist
        _, bins, _ = plt.hist(x, bins=bins, stacked=True, label=x_labels, color=color)
        plt.xticks(bins, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Plot threshold
        plt.axvline(
            x=x_threshold, color="r", linestyle="dashed", linewidth=4, label=f"threshold: {round(x_threshold, 2)}"
        )

        # Add title to legend
        leg = plt.legend(loc="upper right", title=legend_title, title_fontsize=fontsize, fontsize=fontsize)
        leg._legend_box.align = "left"

        # Add grid to figure
        plt.minorticks_on()
        plt.grid(which="major", linestyle="-", linewidth="0.5", color="black")
        plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")

        # Add labels and legend
        plt.ylabel("Count", fontsize=fontsize)
        plt.xlabel("Anomaly Scores", fontsize=fontsize)
        plt.savefig(file_path, bbox_inches="tight")

    @staticmethod
    def save_predictions(
        outputs: STEP_OUTPUT,
        file_path: Path,
        image_save_path: Path,
        step: str,
        save_image_scores: bool,
        save_pixel_scores: bool,
    ):
        """Saves a batch of predictions.

        Args:
            outputs (STEP_OUTPUT): Outputs containing predictions
            file_path (Path): Path containing csv files that has image scores
            image_save_path (Path): Path to save pixel level scores
            step (str): Indicates whether it is training or validation
            save_image_scores (bool): Flag whether to save image level scores
            save_pixel_scores (bool): Flag whether to save pixel level scores
        """
        if save_image_scores:
            pred_df = pd.read_csv(file_path)
            new_pred_df = pd.DataFrame(
                {
                    "image_path": outputs["image_path"],
                    "anomaly_score": outputs["pred_scores"],
                    "target": outputs["label"],
                }
            )
            pred_df = pd.concat([pred_df, new_pred_df], ignore_index=True)
            pred_df.to_csv(file_path, index=False)

        if save_pixel_scores:
            all_values = zip(outputs["image_path"], outputs["anomaly_maps"])
            if "masks" in outputs:
                all_values = zip(all_values, outputs["mask"])
            for values in all_values:
                image_path, anomaly_maps = values[0], values[1]
                masks = values[2] if len(values) == 3 else None
                filename = Path(image_path)
                dir_path = image_save_path / "pixel_predictions" / step / filename.parent.name
                os.makedirs(dir_path, exist_ok=True)
                pixel_file_path = (dir_path / filename.name).with_suffix(".pt")
                if masks:
                    torch.save({"anomaly_maps": anomaly_maps}, pixel_file_path)
                else:
                    torch.save({"anomaly_maps": anomaly_maps, "masks": masks}, pixel_file_path)
