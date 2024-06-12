# CableInspect-AD: An Expert-Annotated Anomaly Detection Dataset

[Mila - Quebec AI Institute](https://mila.quebec/en/industry-services) & [Institut de recherche d'Hydro-Québec]()

Akshatha Arodi\*, Margaux Luck\*, Jean-Luc Bedwani, Aldo Zaimi, Ge Li, Nicolas Pouliot, Julien Beaudry, Gaétan Marceau Caron

\*Denotes equal contribution

[[Paper]()] [[Project](https://mila-iqia.github.io/cableinspect-ad/#)] [[Dataset]()] [[Bibtex]()]

## Installation

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

## Usage

### To perform inference with a VLM

```bash 
python scripts/cogvlm_ad.py --data-path DATA_PATH --test-csv labels.csv --batch-size 4 --out-csv cables_cogvlm_zero_shot_inference.csv
```

### To compute the kfold threshold-dependent metrics (F1, FPR) of a VLM from its raw inference csv output
```bash 
python scripts/get_kfold_metrics.py --vlm-csv PATH_TO_VLM_INFERENCE_OUTPUT --kfold-dir DATA_PATH/k_fold_labels --output-csv-filename cables_vlm_kfold_metrics.csv
```

### To calculate Anomaly Score (VQAScore)

```bash 
python scripts/cogvlm_ad.py --data-path DATA_PATH --test-csv labels.csv --batch-size 4 --out-csv cables_cogvlm_zero_shot_vqascore.csv --generate-scores True
```
