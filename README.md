# cableinspect-ad-code

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

### Pytest
```bash
pip install pytest

pytest tests/
```

## Usage
### Calculate Anomaly Score (VQAScore)

```bash 
python scripts/cogvlm_vqascore.py --data-path DATA_PATH --test-csv labels.csv --batch-size 4 --out-csv cables_cogvlm_zero_shot_vqascore.csv
```