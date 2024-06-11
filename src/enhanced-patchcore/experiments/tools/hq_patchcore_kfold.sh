# Copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: Apache-2.0

# Make sure no previous conda env is activated
if command -v conda; then
  conda deactivate;
fi

# Load conda env
conda activate anomalib_env

# TODO: ADAPT THE DATASET IF NECESSARY
DATASET=$HOME/CableInspect-AD

# Run model
cd $HOME/cableinspect-ad-code/src/enhanced-patchcore
CONFIG=experiments/configs/hq_patchcore_kfold.yaml
RESULTS=$HOME/results

MPLBACKEND=agg  python tools/train.py --config $CONFIG --dataset-path $DATASET --results-folder $RESULTS
