#!/bin/bash

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

# Make sure no previous conda env is activated
if command -v conda; then
  conda deactivate;
fi

# Load conda env
conda activate cableinspect_ad

# To be updated: CableInspect-AD data directory
DATA_DIR=$HOME/CableInspect-AD

# Copy CableInspect-AD into CableInspect-AD_cropped data directory
CROP_DATA_DIR=$DATA_DIR-cropped
cp -r $DATA_DIR $CROP_DATA_DIR

# Remove .json labels and k-fold labels
rm $CROP_DATA_DIR/*.json
rm -r $CROP_DATA_DIR/k_fold_labels

cd $HOME/cableinspect-ad-code/dataset
# Crop images, masks and correct labels
time python -u raw_to_cropped_dataset.py --data-folder $CROP_DATA_DIR

# Generate k-fold labels
time python -u generate_k-shot_labels.py --data-folder $CROP_DATA_DIR