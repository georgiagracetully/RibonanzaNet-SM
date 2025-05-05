#!/bin/bash
#SBATCH --job-name=sm_sim_rdiff_a_absndr
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
python SM_train_model_on_rdiff_and_nodr_abs.py --data '/home/groups/rhiju/gtully/data_sets/rdiff_with_NoDr_abs_train.json' --val_data '/home/groups/rhiju/gtully/data_sets/rdiff_with_NoDr_abs_val.json' --test_data '/home/groups/rhiju/gtully/data_sets/rdiff_with_NoDr_abs_test.json' --criterion 'mae' --epochs 40 --save_path 'RibonanzaNet-SM_0053.pt' --train_snr_cutoff 1 --val_snr_cutoff 1 
  
echo "DONE"
