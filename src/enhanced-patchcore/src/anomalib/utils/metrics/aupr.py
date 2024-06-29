"""Implementation of AUROC metric based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import PrecisionRecallCurve
from torchmetrics.functional import auc, precision, recall
from torchmetrics.utilities.data import dim_zero_cat

from .plotting_utils import normalize, plot_figure


class AUPR(PrecisionRecallCurve):
    """Area under the PR curve."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def compute(self) -> Tensor:
        """First compute PR curve, then compute area under the curve.

        Returns:
            Value of the AUPR metric
        """
        prec: Tensor
        rec: Tensor

        prec, rec = self._compute()
        return auc(rec, prec, reorder=True)

    def compute_f1_score(self, preds: Tensor, target: Tensor, threshold: float) -> tuple[Tensor, Tensor, Tensor]:
        """Compute F1 score at threshold.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
            threshold (float): Threshold for prediction.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing recall, precision and F1 scores at threshold
        """
        rec_threshold = recall(preds, target, threshold=threshold)
        prec_threshold = precision(preds, target, threshold=threshold)
        f1_score = (2 * prec_threshold * rec_threshold) / (prec_threshold + rec_threshold + 1e-10)
        return (rec_threshold, prec_threshold, f1_score)

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

    def generate_figure(
        self, prefix_title: str, extra_threshold_points: bool = False, normalization: dict | None = None
    ) -> tuple[Figure, str]:
        """Generate a figure containing the PR curve as well as the random baseline and the AUC.

        Args:
            prefix_title (str): Prefix to add to the title
            extra_threshold_points (bool): flag whether to plot extra points in the curve
            normalization (dict | None): dict containing the prediction normalization method if any and it's arguments

        Returns:
            tuple[Figure, str]: Tuple containing both the PR curve and the figure title to be used for logging.
        """
        prec, rec = self._compute()
        aupr = self.compute()

        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
        xlabel = "Recall"
        ylabel = "Precision"
        loc = "best"
        title = f"{prefix_title} AUPR"

        fig, axis = plot_figure(rec, prec, aupr, xlim, ylim, xlabel, ylabel, loc, title)

        target = dim_zero_cat(self.target).detach().cpu()
        preds = dim_zero_cat(self.preds).detach().cpu()

        if self.threshold != 0.5 and normalization is not None:
            preds = normalize(preds, self.threshold, normalization)

        # F1 score at threshold(s)
        thresholds = [0.5]
        if extra_threshold_points:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for threshold in thresholds:
            rec_threshold, prec_threshold, f1_score = self.compute_f1_score(preds, target, threshold)
            marker = "ro" if threshold == 0.5 else "v"
            label = f"F1 Score: {f1_score:0.2f}, th: {threshold:0.2f}"
            axis.plot(rec_threshold, prec_threshold, marker, label=label)
            axis.annotate(
                f"({rec_threshold:0.2f}, {prec_threshold:0.2f})",
                xy=(rec_threshold, prec_threshold),
                textcoords="data",
            )

        # Baseline in PR-curve is the prevalence of the positive class
        rate = (target == 1).sum() / target.size(0)
        axis.plot((0, 1), (rate, rate), color="navy", lw=2, linestyle="--", figure=fig)

        axis.legend(loc=loc)

        return fig, title
