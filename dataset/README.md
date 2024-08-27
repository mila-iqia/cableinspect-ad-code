<!---
Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
SPDX-License-Identifier: Apache-2.0
-->

# CableInspect-AD Dataset

Download the dataset and extract files:

```bash
wget https://hydroquebec.com/data/documents-donnees/donnees-ouvertes/zip/CableInspect-AD.zip
unzip CableInspect-AD.zip
cd CableInspect-AD
tar -xzvf cable_1.tar.gz && rm cable_1.tar.gz
tar -xzvf cable_2.tar.gz && rm cable_2.tar.gz
tar -xzvf cable_3.tar.gz && rm cable_3.tar.gz
```

The `CableInspect-AD` dataset folder is structured as follows:

```text
CableInspect-AD
    ├── cable_1.json
    ├── cable_1_seg.json
    ├── cable_2.json
    ├── cable_2_seg.json
    ├── cable_3.json
    ├── cable_3_seg.json
    ├── Cable_1
    ├── Cable_2
    ├── Cable_3
    ├── licence.txt
    └── readme.txt
```

## Dataset Pre-processing

To pre-process the dataset, create the following conda environment:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
conda update conda
conda create -n cableinspect_ad python=3.10
conda activate cableinspect_ad
pip install -r dataset/requirements.txt
```

In the following instructions, we assume the dataset is stored in the `$HOME` directory.

To prepare the dataset, run the following command from the `cableinspect-ad-code` root directory:

```bash
bash dataset/tools/dataset.sh
```

This will create in the `CableInspect-AD` directory with the following structure:

```text
CableInspect-AD
    └── Cable_1
        └──images
            └── 01
                └──*.png
            └── 02
                └──*.png
            └── 03
                └──*.png
        └──masks
            └── 01
                └──*.png
        └── Cable_2
            └── ...
        └── Cable_3
            └── ...
        ├── labels.csv
        ├── k_fold_labels
        ├── licence.txt
        └── readme.txt
```

Below is an example row of the `labels.csv` file:

| image_path   | cable_id | side_id | pass_id | frame_id | bbox_area | bbox_x | bbox_y | bbox_width | bbox_height | bbox_rotation | anomaly_type_id | anomaly_type |  anomaly_grade | anomaly_id | label_index | mask_path   |
| :----------- | :------- | :------ | ------: | -------: | ---------:| ------:| ------:| ---------: | ----------: | ------------: | --------------: | :----------- | -------------: | ---------: | :---------- |:----------- |
| <image_path> | C01      | A       |       1 |        0 |  661.5496 | 240.32 | 691.39 |     26.98  |       24.52 |           0.0 |               7 | Deposit      | light          | 003_01     |           1 | <mask_path> |


The `k_fold_labels` folder has the following structure:

```text
k_fold_labels
└── C01
    └── 2
        ├── label_cable-C01_num_k_shot-2_anomaly_id-0.csv
        ├── label_cable-C01_num_k_shot-2_anomaly_id-1.csv
        └── ...
    └── 3
        ├── label_cable-C01_num_k_shot-3_anomaly_id-0.csv
        ├── label_cable-C01_num_k_shot-3_anomaly_id-1.csv
        └── ...
    └── ...

└── C02
    └── 2
        ├── label_cable-C02_num_k_shot-2_anomaly_id-0.csv
        ├── label_cable-C02_num_k_shot-2_anomaly_id-1.csv
        └── ...
    └── 3
        ├── label_cable-C02_num_k_shot-3_anomaly_id-0.csv
        ├── label_cable-C02_num_k_shot-3_anomaly_id-1.csv
        └── ...
    └── ...

└── C03
    └── 2
        ├── label_cable-C03_num_k_shot-2_anomaly_id-0.csv
        ├── label_cable-C03_num_k_shot-2_anomaly_id-1.csv
        └── ...
    └── 3
        ├── label_cable-C03_num_k_shot-3_anomaly_id-0.csv
        ├── label_cable-C03_num_k_shot-3_anomaly_id-1.csv
        └── ...
    └── ...
```

Below is an example row of the `label_cable-{cable_id}_num_k_shot-{num_k_shot}_anomaly_id-{anomaly_id}.csv` files:

| cable_id | side_id | pass_id | image_path   | mask_path   | label_index | split |
| :------- | :------ | ------: | :----------- | :---------- | :---------- | -----:|
| C01      | A       |       1 | <image_path> | <mask_path> |           1 |  test |

## Cropped Version of the Dataset

To prepare the cropped version of the dataset, run the following command from the `cableinspect-ad-code` root directory:

```bash
bash dataset/tools/cropped_dataset.sh
```

This will create a `CableInspect-AD-cropped` folder with the same structure as the `CableInspect-AD` folder after preprocessing.
