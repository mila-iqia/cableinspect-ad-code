from enum import Enum

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class LabelName(int, Enum):
    """Name of label."""

    NORMAL = 0
    ABNORMAL = 1
