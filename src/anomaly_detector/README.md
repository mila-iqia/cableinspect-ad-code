<!---
Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
SPDX-License-Identifier: CC-BY-4.0
-->

# Code to reproduce the results of WinCLIP

We install the [anomalib library](https://github.com/openvinotoolkit/anomalib/) and run WinCLIP:
```bash
pip install -r requirements.txt
```

Generate anomaly scores from WinCLIP using the script.
```bash
export DATASET_PATH=$HOME/CableInspect-AD
export RESULTS=$HOME/results
python generate_winclip_score.py --dataset-path $DATASET_PATH --output-path $RESULTS
```
# Evaluation
The metrics can be generated using the `evaluate_winclipo.ipynb` notebook for all the VLMs and WinCLIP.

To generate the AUPRO metric, we follow the method [here](https://github.com/caoyunkang/WinClip/blob/master/README.md)
