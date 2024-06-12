"""Anomaly Score Normalization Callback that uses min-max normalization."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule
from anomalib.post_processing.normalization.min_max import normalize
from anomalib.utils.metrics import MinMax


class MinMaxNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization."""

    def setup(self, trainer: pl.Trainer, pl_module: AnomalyModule, stage: str | None = None) -> None:
        """Adds min_max metrics to normalization metrics."""
        del trainer, stage  # These variables are not used.

        if not hasattr(pl_module, "normalization_metrics"):
            pl_module.normalization_metrics = MinMax().cpu()
        elif not isinstance(pl_module.normalization_metrics, MinMax):
            raise AttributeError(
                f"Expected normalization_metrics to be of type MinMax, got {type(pl_module.normalization_metrics)}"
            )

    def on_test_start(self, trainer: pl.Trainer, pl_module: AnomalyModule) -> None:
        """Called when the test begins."""
        del trainer  # `trainer` variable is not used.

        for metric in (pl_module.image_metrics, pl_module.pixel_metrics):
            if metric is not None:
                metric.set_threshold(0.5)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, update the min and max observed values."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        if "anomaly_maps" in outputs:
            pl_module.normalization_metrics(anomaly_scores=None, anomaly_maps=outputs["anomaly_maps"])

        if "box_scores" in outputs:
            pl_module.normalization_metrics(anomaly_scores=None, anomaly_maps=torch.cat(outputs["box_scores"]))

        if "pred_scores" in outputs:
            pl_module.normalization_metrics(anomaly_scores=outputs["pred_scores"], anomaly_maps=None)

        if ("anomaly_maps" not in outputs) and ("box_scores" not in outputs) and ("pred_scores" not in outputs):
            raise ValueError("No values found for normalization, provide anomaly maps, bbox scores, or image scores")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: AnomalyModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        del trainer, batch, batch_idx, dataloader_idx  # These variables are not used.

        self._normalize_batch(outputs, pl_module)

    @staticmethod
    def _normalize_batch(outputs, pl_module) -> None:
        """Normalize a batch of predictions."""
        image_threshold = pl_module.image_threshold.value.cpu()
        pixel_threshold = pl_module.pixel_threshold.value.cpu()
        stats = pl_module.normalization_metrics.cpu()
        outputs["pred_scores"] = normalize(outputs["pred_scores"], image_threshold, stats.image_min, stats.image_max)
        if "anomaly_maps" in outputs:
            outputs["anomaly_maps"] = normalize(
                outputs["anomaly_maps"], pixel_threshold, stats.pixel_min, stats.pixel_max
            )
        if "box_scores" in outputs:
            outputs["box_scores"] = [
                normalize(scores, pixel_threshold, stats.pixel_min, stats.pixel_max) for scores in outputs["box_scores"]
            ]
