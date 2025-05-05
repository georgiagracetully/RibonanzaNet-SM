#!/bin/bash
#SBATCH --job-name=sm_sim_train_all_sim_abs_reactivity
#SBATCH --output=sm_sim_train.%j.out
#SBATCH --error=sm_sim_train.%j.out
#SBATCH --error=sm_sim_train.%j.err
#Time: dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=24:00:00
#SBATCH -p biochem,owners,normal,rhiju
#SBATCH --mem=16G
#SBATCH --mail-user=gtully@stanford.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --gpus 1
source ~/.bashrc
conda activate rnanza


echo "START"
python SM_train_reac_diff_only.py --data '/home/groups/rhiju/gtully/data_sets/rdiff_train.json' --val_data '/home/groups/rhiju/gtully/data_sets/rdiff_val.json' --test_data '/home/groups/rhiju/gtully/data_sets/rdiff_test.json' --criterion 'mae' --epochs 20 --save_path 'RibonanzaNet-SM_0042.pt' --train_snr_cutoff 1 --val_snr_cutoff 1 
  
echo "DONE"
