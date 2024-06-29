"""Module that tracks the min and max values of the observations in each batch."""

# Copyright (C) 2022 Intel Corporation
# Modifications copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Metric


class MinMax(Metric):
    """Track the min and max values of the observations in each batch."""

    full_state_update: bool = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("image_min", torch.tensor(float("inf")), persistent=True)  # pylint: disable=not-callable
        self.add_state("image_max", torch.tensor(float("-inf")), persistent=True)  # pylint: disable=not-callable
        self.add_state("pixel_min", torch.tensor(float("inf")), persistent=True)  # pylint: disable=not-callable
        self.add_state("pixel_max", torch.tensor(float("-inf")), persistent=True)  # pylint: disable=not-callable

        self.image_min = torch.tensor(float("inf"))  # pylint: disable=not-callable
        self.image_max = torch.tensor(float("-inf"))  # pylint: disable=not-callable
        self.pixel_min = torch.tensor(float("inf"))  # pylint: disable=not-callable
        self.pixel_max = torch.tensor(float("-inf"))  # pylint: disable=not-callable

    def update(self, anomaly_scores: Tensor | None, anomaly_maps: Tensor | None, *args, **kwargs) -> None:
        """Update the min and max values."""
        del args, kwargs  # These variables are not used.

        if anomaly_scores is not None:
            self.image_min = torch.min(self.image_min, torch.min(anomaly_scores))
            self.image_max = torch.max(self.image_max, torch.max(anomaly_scores))

        if anomaly_maps is not None:
            self.pixel_min = torch.min(self.pixel_min, torch.min(anomaly_maps))
            self.pixel_max = torch.max(self.pixel_max, torch.max(anomaly_maps))

    def compute(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return min and max values."""
        return self.image_min, self.image_max, self.pixel_min, self.pixel_max
