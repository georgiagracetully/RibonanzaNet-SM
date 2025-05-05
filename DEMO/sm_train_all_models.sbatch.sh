#!/bin/bash
#SBATCH --job-name=rnet_sm_iter_00_test
#SBATCH --output=rnet_sm_iter_00.%j.out
#SBATCH --error=rnet_sm_iter_00.%j.out
#SBATCH --error=rnet_sm_iter_00.%j.err
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

#!/bin/bash
echo "START"

# Define arrays of hyperparameters and data paths
learning_rates=(0.001)
epochs=(20)
criteria=('MAE')  # Loss criteria
optimizers=('Ranger')  # Optimizer options
train_snr_cutoffs=(1.0)  # SNR cutoff for training data
val_snr_cutoffs=(1.0)  # SNR cutoff for validation/test data
configs=('/scratch/groups/rhiju/gtully/rnet2d/sherlock_file_organization/ribonanzanet2d-final/configs/pairwise.yaml')
weights=('/scratch/groups/rhiju/gtully/rnet2d/sherlock_file_organization/ribonanzanet-weights/RibonanzaNet.pt')

data=("/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Eryt_train.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Kana_train.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Mito_train.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Paro_train.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Spec_train.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Tetr_train.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_NoDr_train.json")


val_data=("/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Eryt_val.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Kana_val.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Mito_val.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Paro_val.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Spec_val.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_Tetr_val.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/SM_NoDr_val.json")

test_data=("/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/Eryt_test_no_clipped.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/Kana_test_no_clipped.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/Mito_test_no_clipped.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/Paro_test_no_clipped.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/Spec_test_no_clipped.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/Tetr_test_no_clipped.json" "/home/groups/rhiju/gtully/data_sets/SM_drug_new_iter/NoDr_test_no_clipped.json")
save_path=("RibonanzaNet-SM-Eryt-00.pt" "RibonanzaNet-SM-Kana-00.pt" "RibonanzaNet-SM-Mito-00.pt" "RibonanzaNet-SM-Paro-00.pt" "RibonanzaNet-SM-Spec-00.pt" "RibonanzaNet-SM-Tetr-00.pt" "RibonanzaNet-SM-NoDr-00.pt")

# Log file to save timestamps and job details
log_file="RibonanzaNet_001_iterative_training_log.txt"
echo "Training Log - $(date)" > "$log_file"
echo "Job ID | Start Time | End Time | Learning Rate | Epochs | Criterion | Optimizer | Train SNR Cutoff | Val SNR Cutoff | Config | Weights | Train Data | Test Data | Save Path" >> "$log_file"
echo "-----------------------------------------------------------------------------------------------------------------------------------------------------------------" >> "$log_file"

# Initialize job ID
job_id=1

# Loop over hyperparameters and data combinations
for lr in "${learning_rates[@]}"; do
  for epoch in "${epochs[@]}"; do
    for criterion in "${criteria[@]}"; do
      for optimizer in "${optimizers[@]}"; do
        for train_snr in "${train_snr_cutoffs[@]}"; do
          for val_snr in "${val_snr_cutoffs[@]}"; do
            for config in "${configs[@]}"; do
              for weight in "${weights[@]}"; do
                for i in "${!data[@]}"; do
                  # Record the start time
                  start_time=$(date "+%Y-%m-%d %H:%M:%S")

                  echo "Running job $job_id with lr=$lr, epochs=$epoch, criterion=$criterion, optimizer=$optimizer, train_snr=$train_snr, val_snr=$val_snr, config=$config, weights=$weight, train=${data[i]}, val=${val_data[i]}, test=${test_data[i]}, save_path=${save_path[i]}"

                  # Run the Python script
                  python SM_train_model_3.py \
                    --lr "$lr" \
                    --epochs "$epoch" \
                    --criterion "$criterion" \
                    --optimizer "$optimizer" \
                    --train_snr_cutoff "$train_snr" \
                    --val_snr_cutoff "$val_snr" \
                    --config "$config" \
                    --weights "$weight" \
                    --data "${data[i]}" \
		    --val_data "${val_data[i]}"\
                    --test_data "${test_data[i]}" \
                    --save_path "${save_path[i]}"

                  # Record the end time
                  end_time=$(date "+%Y-%m-%d %H:%M:%S")

                  # Log details to the file
                  echo "$job_id | $start_time | $end_time | $lr | $epoch | $criterion | $optimizer | $train_snr | $val_snr | $config | $weight | ${data[i]} | ${test_data[i]} | ${save_path[i]}" >> "$log_file"

                  # Increment job ID
                  job_id=$((job_id + 1))
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "All jobs completed. Log saved to $log_file"
"
