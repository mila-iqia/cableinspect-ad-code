#!/usr/bin/env python3

"""Generate labels for the few, many and full shot experiments for the CableInspect-AD dataset."""

# Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
# SPDX-License-Identifier: CC-BY-4.0

import os
import time
from argparse import ArgumentParser, Namespace

import pandas as pd
from tqdm import tqdm

from utils import generate_kfold_labels


def parse_args() -> Namespace:
    """Parser for the command line arguments.

    Returns:
        arguments (Namespace): The arguments.
    """
    parser = ArgumentParser(description="Generate k-shot labels in data folder.")

    parser.add_argument("--data-folder", type=str, required=True, help="Data folder.")

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = parse_args()
    data_folder = args.data_folder
    labels = pd.read_csv(os.path.join(data_folder, "labels.csv"))
    preprocess_name = os.path.basename(data_folder)
    if "cropped" in data_folder:
        cables_max_anomaly_ids = {"C01": 49, "C02": 54, "C03": 35}
    else:
        cables_max_anomaly_ids = {"C01": 40, "C02": 46, "C03": 33}        
    output_folder = "k_fold_labels"

    num_train = num_k_shot = 100
    num_val = 0
    buffer = 5

    start_time = time.time()
    for cable in cables_max_anomaly_ids.keys():
        print(f"Generating kfold labels in {preprocess_name} for cable {cable}")
        folder_path = os.path.join(data_folder, output_folder, f"{cable}")
        os.makedirs(folder_path, exist_ok=True)
        for anomaly_group_id in tqdm(range(0, cables_max_anomaly_ids[cable])):
            if "cropped" in data_folder:
                if cable == "C03" and anomaly_group_id == 31:
                    # This anomaly ID cannot create sufficient number of images
                    continue
            else:
                if cable == "C03" and anomaly_group_id in [22, 26, 30]:
                    continue
            kfold_exp_labels = generate_kfold_labels(
                labels,
                cbl=cable,
                num_train=num_train,
                num_val=num_val,
                anomaly_group_id=anomaly_group_id,
                buffer=buffer,
                num_k_shot=num_k_shot,
            )

            labels_fname = f"label_cable-{cable}_anomaly_id-{anomaly_group_id}.csv"
            kfold_exp_labels.to_csv(os.path.join(folder_path, labels_fname), index=False)
    time_elapsed = time.time() - start_time
    print(f"Task completed in {time_elapsed}s")
