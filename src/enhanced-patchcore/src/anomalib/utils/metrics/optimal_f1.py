"""Implementation of Optimal F1 score based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import PrecisionRecallCurve


class OptimalF1(PrecisionRecallCurve):
    """Optimal F1 Metric."""

    def compute(self) -> Tensor:
        """Compute the value of the optimal F1 score.

        Compute the F1 scores while varying the threshold.
        Return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        prec: Tensor
        rec: Tensor

        prec, rec = self._compute()
        f1_score = (2 * prec * rec) / (prec + rec + 1e-10)
        optimal_f1_score = torch.max(f1_score)
        return optimal_f1_score

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Need to flatten new values as PrecicionRecallCurve expects them in this format for binary classification.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[Tensor, Tensor]:
        """Compute prec/rec value pairs.

        Returns:
            Tuple containing Tensors for rec and prec
        """
        prec: Tensor
        rec: Tensor
        prec, rec, _ = super().compute()
        return (prec, rec)
