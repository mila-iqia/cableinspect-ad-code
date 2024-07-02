"""Implementation of AUROC metric based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import ROC
from torchmetrics.functional import auc, recall, specificity
from torchmetrics.utilities.data import dim_zero_cat

from .plotting_utils import normalize, plot_figure


class AUROC(ROC):
    """Area under the ROC curve."""

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def compute(self) -> Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            Tensor: Value of the AUROC metric
        """
        tpr: Tensor
        fpr: Tensor

        fpr, tpr = self._compute()
        return auc(fpr, tpr, reorder=True)

    def compute_j_stat(self, preds: Tensor, target: Tensor, threshold: float) -> tuple[Tensor, Tensor, Tensor]:
        """Compute J-stat at threshold.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
            threshold (float): Threshold for prediction.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing sensitivity, specificity and J-statistic at threshold
        """
        sens_threshold = recall(preds, target, threshold=threshold)
        spec_threshold = specificity(preds, target, threshold=threshold)
        jstat_threshold = sens_threshold + spec_threshold - 1
        return (sens_threshold, spec_threshold, jstat_threshold)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Need to flatten new values as ROC expects them in this format for binary classification.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[Tensor, Tensor]:
        """Compute fpr/tpr value pairs.

        Returns:
            Tuple containing Tensors for fpr and tpr
        """
        tpr: Tensor
        fpr: Tensor
        fpr, tpr, _ = super().compute()
        return (fpr, tpr)

    def generate_figure(
        self, prefix_title: str, extra_threshold_points: bool = False, normalization: dict | None = None
    ) -> tuple[Figure, str]:
        """Generate a figure containing the ROC curve, the baseline and the AUROC.

        Args:
            prefix_title (str): Prefix to add to the title.
            extra_threshold_points (bool): flag whether to plot extra points in the curve
            normalization (dict | None): dict containing the prediction normalization method if any and it's arguments

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
        """
        fpr, tpr = self._compute()
        auroc = self.compute()

        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        loc = "lower right"
        title = f"{prefix_title} ROC"

        fig, axis = plot_figure(fpr, tpr, auroc, xlim, ylim, xlabel, ylabel, loc, title)

        target = dim_zero_cat(self.target).detach().cpu()
        preds = dim_zero_cat(self.preds).detach().cpu()

        if self.threshold != 0.5 and normalization is not None:
            preds = normalize(preds, self.threshold, normalization)

        # J-stat at threshold(s)
        thresholds = [0.5]
        if extra_threshold_points:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for threshold in thresholds:
            sens_threshold, spec_threshold, jstat_threshold = self.compute_j_stat(preds, target, threshold)
            marker = "ro" if threshold == 0.5 else "v"
            label = f"J-stat: {jstat_threshold:0.2f} th: {threshold}"
            axis.plot(1 - spec_threshold, sens_threshold, marker, label=label)
            axis.annotate(
                f"({1 - spec_threshold:0.2f}, {sens_threshold:0.2f})",
                xy=(1 - spec_threshold, sens_threshold),
                textcoords="data",
            )

        axis.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", figure=fig)

        axis.legend(loc=loc)

        return fig, title
