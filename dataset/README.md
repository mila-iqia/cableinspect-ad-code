<!---
Copyright (C) 2024 Mila - Institut québécois d'intelligence artificielle
SPDX-License-Identifier: CC-BY-4.0
-->

# CableInspect-AD Dataset

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

In the following instructions, we assume the dataset is stored in the `$HOME` directory.

To prepare the dataset, run the following command from the `cableinspect-ad-code` root directory:

```bash
bash dataset/tools/local/dataset.sh
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

| image_path   | cable_id | side_id | pass_id | frame_id | anomaly_type_id | anomaly_type |  anomaly_grade | anomaly_id | label_index | mask_path   |
| :----------- | :------- | :------ | ------: | -------: | --------------: | :----------- | -------------: | ---------: | :---------- |:----------- | 
| <image_path> | C01      | A       |       1 |        0 |               7 | Deposit      | light          | 003_01     |           1 | <mask_path> |


The `k_fold_labels` folder has the following structure:

```text
<output-folder>
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

## Cropped Version of the Dataset

To prepare the cropped version of the dataset, run the following command from the `cableinspect-ad-code` root directory:

```bash
bash dataset/tools/local/cropped_dataset.sh
```

This will create a `CableInspect-AD-cropped` folder with the same structure as the `CableInspect-AD` folder after preprocessing.
