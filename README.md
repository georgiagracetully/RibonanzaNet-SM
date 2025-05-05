# RibonanzaNet_SM

Fine-tuning RibonanzaNet on small RNA datasets for chemical reactivity profile prediction.

## Overview

This repository contains code, data, and notebooks for fine-tuning [RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet) — a large RNA foundation model — to predict structural reactivity changes in RNA sequences upon binding with small molecules. This model can be applied to new libraries and used to generate informative reactivity features for downstream RNA analysis.

## Project Structure

```

.
├── data/             # Raw and processed data files
│   ├── PK50.fasta
│   ├── clustered\_sequences.fasta
│   ├── train\_set.pkl
│   ├── val\_set.pkl
│   └── test\_set.pkl
│
├── train/            # PyTorch fine-tuning scripts
│   └── train\_finetune.py
│
├── DEMO/             # SLURM job scripts for HPC training
│   └── train\_job.sbatch
│
├── ANALYSIS/         # Jupyter notebooks for result visualization
│   └── heatmap\_visualization.ipynb
│
├── configs/          # Model configuration files
│   └── pk50\_config.yaml
│
├── requirements.txt  # Required packages
└── README.md

````

## Data Preparation

The RNA library `PK50` was clustered using `cd-hit` with:
- Sequence identity threshold: 75%
- Word size: 5

Train/test/validation splits are precomputed and stored in the `data/` directory. See the provided notebook for reproducible processing.

## Usage

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
````

2. **Train model**:

   ```bash
   python train/train_finetune.py --config configs/pk50_config.yaml
   ```

3. **Run evaluation / analysis**:
   Open the notebooks in the `ANALYSIS/` directory using JupyterLab:

   ```bash
   jupyter lab
   ```

## Citation

If you use this repository or model in your work, please cite the original [RibonanzaNet paper](https://www.biorxiv.org/content/10.1101/2024.02.24.581671v1).

---

For questions or contributions, feel free to open an issue or submit a pull request.





