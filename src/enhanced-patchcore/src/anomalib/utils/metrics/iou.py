"""Implementation of IoU metric based on TorchMetrics."""

# Copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import JaccardIndex, Metric


class IoU(Metric):
    """Intersection over Union (IoU) Score.

    For normal images, a score of 1.0 is given when a normal image is well predicted and
    a score of 0.0 is given when it is not well predicted.

    For abnormal images, IoU between ground truth bbox and predicted mask is computed.

    Args:
        num_classes (int, optional): Number of classes in the dataset. Defaults 2.
        threshold (float, optional): Threshold value for binary classification. Defaults 0.5.
    """

    full_state_update: bool = True
    target: list[Tensor]
    preds: list[Tensor]

    def __init__(self, num_classes: int = 2, threshold: float = 0.5) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.iou_fct = JaccardIndex(num_classes=self.num_classes, absent_score=1.0, average="none")

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        """Compute the IoU score for the current batch."""
        self.target.append(targets)
        self.preds.append(predictions)

    def compute(self) -> Tensor:
        """Compute the macro average of the IoU score across all batches."""
        iou = []
        for idx, pred in enumerate(self.preds):
            pred = (pred >= self.threshold).type(torch.float)
            # We need to loop over the images in the batches because contrarily to
            # object detection, here we have some ground truth images without objects
            # to detect (i.e. without anomaly).
            for i in range(pred.shape[0]):
                iou.append(self.iou_fct(pred[i], self.target[idx][i])[1])
        return torch.mean(torch.stack(iou))
