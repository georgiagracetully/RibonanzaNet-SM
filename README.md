# RibonanzaNet-SM

Fine-tuning RibonanzaNet on small RNA datasets for chemical reactivity profile prediction. Details on theory behind the project can be found [here](https://georgiagracetully.github.io/portfolio/project-5/).

## Overview

This repository contains code, data, and notebooks for fine-tuning [RibonanzaNet](https://github.com/Shujun-He/RibonanzaNet) — a large RNA foundation model — to predict structural reactivity changes in RNA sequences upon binding with small molecules. This model can be applied to new libraries and used to generate informative reactivity features for downstream RNA analysis.

## Project Structure

```


├── data/             # Raw and processed data files
│   ├── fasta_files/
│   ├── _train/
│   ├── _val/
│   └── analysis_ribonanzanet_sm_data_split.py
│
├── train/            # PyTorch fine-tuning scripts
│   └── SM_train_model_sim.py
│   └── SM_train_model_on_rdiff_and_nodr_abs.py
│
├── DEMO/             # SLURM job scripts for HPC training
│   └── sm_train_all_reac_diff_with_NoDr_abs.sbatch.sh
│   └── sm_train_all_reac_diff_simultaneously.sbatch.sh
│   └── sm_train_all_models.sbatch.sh
│
├── ANALYSIS/         # Jupyter notebooks for result visualization
│   ├── heatmaps
│
├── configs/          # Model configuration files
│   └── pairwise.yaml
│   ├── rnet_sm_configs
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
   ```

2. **Train model**:
   Arguments vary depending on training script, but here is one example: 
   ```bash
   python SM_train_model_on_rdiff_and_nodr_abs.py --data '/data/rdiff_with_NoDr_abs_train.json' --val_data '/data/rdiff_with_NoDr_abs_val.json' --test_data '/data/rdiff_with_NoDr_abs_test.json' --criterion 'mae' --epochs 40 --save_path 'RibonanzaNet-SM_005.pt' --train_snr_cutoff 1 --val_snr_cutoff 1

   ```

3. **Run evaluation / analysis**:
   Open the notebooks in the `ANALYSIS/` directory using JupyterLab:

   ```bash
   python -m notebook
   ```

## Citation

If you use this repository or model in your work, please cite the original [RibonanzaNet paper](https://www.biorxiv.org/content/10.1101/2024.02.24.581671v1).

---

For questions or contributions, feel free to open an issue or submit a pull request.





