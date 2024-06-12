<!---
Copyright (C) 2022 Mila - Institut québécois d'intelligence artificielle
SPDX-License-Identifier: Apache-2.0
-->

# Code to reproduce the results of Enhanced-PatchCore

## Prerequisite environment

Assuming the repository is cloned in your home directory.
To create and pack anomalib conda environment run the following commands from your home directory:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
conda update conda
conda create -n anomalib_env python=3.10
conda activate anomalib_env
cd cableinspect-ad-code/src/enhanced-patchcore
pip install -e .
pip install -r requirements/openvino.txt
pip install -r requirements/notebooks.txt
```

## CableInspect-Ad Dataset

More details about the dataset can be found [here](https://mila-iqia.github.io/cableinspect-ad/).

Some notebooks have been developed to visualize locally this dataset. They can be found in the `notebooks` folder:

- `anomaly_viewer.ipynb`

This notebook permits to visualize the original images and some relevant associated meta-data.

- `visualize_labels.ipynb`

This notebook permits to visualize the labels and their statistics for the tight crop and tight crop imagette dataset.

## Dataloader

Two files have been added in the `src/anomalib/data` folder for the PyTorch HQ cable dataset and dataloader:

- `hq.py`

In this file, the HQ cable dataset and dataloader classes are defined.

- `hq_utils.py`

In this file, the functions to generate the different dataset splits are available (i.e.: IID, calibration, OOD, and, K-fold).

**IID:** All cables can be used, and included cables are split into three non-overlapping regions - train, valid and test.

**Calibration:** Labels are assigned such that there are regions for train and validation on historical cables and, train, validation and test on the calibration cable. The use of historical cables is optional.

**OOD:** Labels are assigned such that there are separate cables for train, validation and test sets.

**K-Fold:** Labels are assigned based on an anomaly group position. First, we identify an anomaly group to indicate the beginning of the train set. First set of 'num_train' nominal frames immediately following the selected anomaly group position are assigned to the training set. Then, the next set of nominal frames are assigned to the validation set if requested (it is optional). Finally, all the remaining frames are assigned to the test set.

We only use the K-Fold splits in the weakly supevised setting described below for the analysis.

## Experiments

The configuration files to reproduce the experiments are available in the folder:

- `experiments/configs/`

The experiments can be run locally or on the cluster. The bash scripts are available in the folder:

- `experiments/tools/`

The results are saved in the `$HOME/results` folder. Prior running an experiment update it in the experiment bash file if necessary.

To run the following experiments, update the data location in the bash script if necessary. The following scripts assume that the dataset is preprocessed and the `anomalib_env` is set up. The scripts are executed from the root directory.

## Enhanced-PatchCore

The models are trained on the train set and the threshold is also set using the train set. The reporsitory contains code that is modified from the [anomalib repo](https://github.com/openvinotoolkit/anomalib)

The bash scripts support the following experiments:

**K-Fold:**

```bash
bash experiments/tools/hq_patchcore_kfold_kshot.sh
```
This runs experiments with the configurations specified in the `experiments/configs/hq_patchcore_kfold_kshot.yaml`. We use orion to run experiments on all k-shots as shown in an example config. The cables and other configuration are to be updated in the configuration files.

## Results

The predictions are stored in the `$RESULTS` folder defined in the bash script. The predictions of a particular run of an experiment `<config-experiment-name>` (defined in the config file of the experiment) can be found here: `${RESULTS}/patchcore/hq/<config-experiment-name>/run.<time-stamp>`. This folder contains:

- Image predictions: Files `validation_image_predictions.csv` and `test_image_predictions.csv` store the anomaly scores of validation and test image respectively.
- Pixel predictions: Folder `pixel_predictions` contain subfolders with per pixel anomaly score tensors for the validation and test images.
- Images: Folder `images` contain the metric curves. To save the predicted heatmaps of the test images, add `images` to the `visualization: include` field in the experiment config file.
- Weights: Contains the best model weights `weights/lightning/model.ckpt`.
- Scores: The metric scores on the test set are saved in the file `logs/lightning_logs/version_0/metrics.csv`.

In case of unsupervised learning, there is no validation set and the validation is run on the train set. Therefore, the validation prediction files in this setup corresponds to the anomaly scores on the train set.

To visualize the image predictions for a specific run in more detail, use the notebook `notebooks/image_prediction_stats.ipynb`.

To visualize the aggregated results of the unsupervised k-fold experiments over all the anomaly groups, first generate the `aggregated_results.csv` file in the experiments folder by running from the folder `post_processing`:

```bash
DATA_DIR=$HOME/CableInspect-AD
RESULTS_DIR=$HOME/results
EXP_DIR=$RESULTS_DIR/patchcore/hq/hq_kfold_unsupervised_cbl-C01_anomaly_group_id-0_k-10_shot
python post_processing_unsupervised_results.py --data-directory $DATA_DIR --experiment-directory $EXP_DIR
```

To generate aggregated results over all the cables, replace `hq_kfold_unsupervised_C01` with the name of the experiment defined in the experiment config file for other cables `hq_kfold_unsupervised_{cable_id}` and run the post processing script for each cable experiment.

Once the `aggregated_results.csv` file is generated, the results can be visualized using the notebook in the sub-folder `notebooks`:

- `fully_unsupervised_kfold_visualization.ipynb`.

This notebook permits to visualize the aggregated predictions and statistics for unsupervised k-fold experiments.

- `kfold_per_anomaly_type_metrics.ipynb`

This notebook permits to visualize the metrics per anomaly types.

- `kfold_per_unique_anomalies_metrics.ipynb`

This notebook permits to visualize the metrics per unique anomalies.

- `kfold_anomaly_ids_level_predictions_visualization.ipynb`

This notebook permits to visualize the ID level predictions of all runs in a kfold experiment.

- `kfold_custom_aupr_metrics.ipynb`

This notebook permits to visualize the custom metrics AUPR and F1 where precision is at the image level and recall at the ID level.
