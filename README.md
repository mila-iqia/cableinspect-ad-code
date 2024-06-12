# CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset

[Mila - Quebec AI Institute](https://mila.quebec/en/industry-services) & [Institut de recherche d'Hydro-Québec]()

Akshatha Arodi\*, Margaux Luck\*, Jean-Luc Bedwani, Aldo Zaimi, Ge Li, Nicolas Pouliot, Julien Beaudry, Gaétan Marceau Caron

\*Denotes equal contribution

[[Paper]()] [[Project](https://mila-iqia.github.io/cableinspect-ad/#)] [[Dataset]()] [[Bibtex]()]

The repository has the following structure.

```plaintext
cableinspect-ad-code/
├── dataset/                                    # Code to preprocess dataset
│   ├── README.md                               # Documentation of the data and preprocessing
│   ├── generate_labels.py                      # Generate labels
│   ├── generate_masks.py                       # Generate masks
│   ├── tools/local
│   │   ├── dataset.sh                          # Script to generate labels and masks
│   │   └── ...
│   └── ...
├── scripts/                                    # Scripts to run all the VLMs
│   ├── get_kfold_metrics.py                    # Generate threshold-dependant metrics
│   ├── prompts.yaml                            # Prompt for the VLMs
│   ├── cogvlm_ad.py                            # Inference script for CogVLM
│   ├── llava13b_ad.py                          # Inference script for LLaVA-13B
│   └── ...
├── src/                                        
│   ├── anomaly_detector/                       # Code for VLMs and WinCLIP
│   │   ├── README.md                           # Documentation of the models
│   │   ├── generate_winclip_score.py           # Script to run WinCLIP
│   │   ├── cogvlm_ad_inference.py              # Script to run CogVLM
│   │   ├── llava_ad_inference.py               # Script to run LLaVA
│   │   ├── evaluate.ipynb                      # Notebook to generate threshold-independant metrics
│   │   └── ...
│   ├── enhanced-patchcore/                     # Code for Enhanced-PatchCore
│   │   ├── README.md                           # Documentation of Enhanced-PatchCore
│   │   ├── notebooks/                          # Notebooks for data visualization and results
│   │   ├── postprocessing/                     # Script for postprocessing
│   │   ├── tools/
│   │   │   ├── hq_patchcore_kfold_kshot.sh     # Bash script for running the model
│   │   └── ...
└── README.md                                   # This README file
└── ...
```

## Table of Contents
- [Dataset](#dataset)
- [Enhanced-PatchCore](src/enhanced-patchcore/README.md)
- [Vision-Language Models](#vision-language-models)
  - [Installation](#installation)
  - [Usage](#usage)
- [Results](#results)

## Dataset
We provide code for the generation of labels and masks. After downloading the images and annotation files from the [project website](https://mila-iqia.github.io/cableinspect-ad/), follow the instructions [here](dataset/README.md).

## Enhanced-PatchCore

The instructions on installation and usage are provided [here](src/enhanced-patchcore/README.md). We also provide notebooks for results and dataset visualization.

## Vision-Language Models

We provide inference scripts to evaluate all the Vision-Language Models (VLMs) that we report in the paper. We also include WinCLIP in our evaluation and provide inference scripts.

### Installation

To setup the environment:
```bash
conda create -n ad_env python=3.10
conda activate ad_env
```
We need to set these environment variables before installing torch.
```bash
envPath=$(conda info --envs | grep ad_env | awk '{print $NF}')
export CUDA_HOME=$envPath
```
Install cudatoolkit
```bash
conda install nvidia/label/cuda-12.0.0::cuda-toolkit
```
To verify the installation, run the following:
```bash
nvcc --version
```
Install the dependancies:
```bash
pip install -r requirements.txt
```

Then install the package:
```bash
pip install -e .
```

### Pytest
```bash
pip install pytest

pytest tests/
```

### Usage

#### To perform inference with a VLM

```bash 
python scripts/cogvlm_ad.py --data-path DATA_PATH --test-csv labels.csv --batch-size 4 --out-csv cables_cogvlm_zero_shot_inference.csv
```

#### To compute the kfold threshold-dependent metrics (F1, FPR) of a VLM from its raw inference csv output
```bash 
python scripts/get_kfold_metrics.py --vlm-csv PATH_TO_VLM_INFERENCE_OUTPUT --kfold-dir DATA_PATH/k_fold_labels --output-csv-filename cables_vlm_kfold_metrics.csv
```

#### To calculate Anomaly Score (VQAScore)

```bash 
python scripts/cogvlm_ad.py --data-path DATA_PATH --test-csv labels.csv --batch-size 4 --out-csv cables_cogvlm_zero_shot_vqascore.csv --generate-scores True
```

## Results

### Performance Metrics at Image-Level

Mean and standard deviation are calculated across all cables after averaging over all folds. VLMs and WinCLIP are evaluated in a zero-shot setting, while *Enhanced-PatchCore* is evaluated in a 100-shot setting using the *beta-prime-95* thresholding strategy. Thresholded-metrics are not reported for WinCLIP since it necessitates a validation set.

| **Model**                | **F1 Score** ↑        | **FPR** ↓                | **AUPR** ↑               | **AUROC** ↑             |
|--------------------------|-----------------------|--------------------------|--------------------------|-------------------------|
| LLaVA 1.5-7B             | 0.59 ± 0.07           | 0.32 ± 0.19              | 0.75 ± 0.05              | 0.68 ± 0.04             |
| LLaVA 1.5-13B            | 0.69 ± 0.02           | 0.66 ± 0.21              | 0.74 ± 0.04              | 0.66 ± 0.03             |
| BakLLaVA-7B              | 0.69 ± 0.02           | 0.53 ± 0.19              | 0.77 ± 0.04              | 0.71 ± 0.03             |
| CogVLM-17B               | **0.77 ± 0.02**       | 0.34 ± 0.21              | 0.83 ± 0.03              | 0.79 ± 0.04             |
| CogVLM2-19B              | 0.66 ± 0.04           | **0.04 ± 0.01**          | **0.91 ± 0.02**          | **0.86 ± 0.03**         |
| WinCLIP                  | -                     | -                        | 0.76 ± 0.06              | 0.70 ± 0.04             |
| *Enhanced-PatchCore*     | 0.75 ± 0.03           | 0.55 ± 0.19              | 0.84 ± 0.06              | 0.78 ± 0.05             |

