#!/bin/bash

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

# Make sure no previous conda env is activated
if command -v conda; then
  conda deactivate;
fi

# Load conda env
conda activate cableinspect_ad

# CableInspect-AD data directory
DATA_DIR=$HOME/CableInspect-AD

cd $HOME/cableinspect-ad-code/dataset
# Generate labels
time python -u generate_labels.py --data-folder $DATA_DIR --annotations-json-files $DATA_DIR/cable_1.json $DATA_DIR/cable_2.json $DATA_DIR/cable_3.json

# Generate k-fold labels
time python -u generate_k-shot_labels.py --data-folder $DATA_DIR

# Generate masks from segmentation labels (only for the first video pass)
time python -u generate_masks.py --data-folder $DATA_DIR