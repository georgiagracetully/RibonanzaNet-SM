import torch
import json
import pandas as pd
import numpy as np
import random
import ast
import argparse
import os

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# Define argument parsing for hyperparameters and options
parser = argparse.ArgumentParser(description="Train RibonanzaNet_SM with specified hyperparameters.")
parser.add_argument('--data', type=str, help='Path to the training data')
parser.add_argument('--val_data', type=str, help= 'Path to the validation data')
parser.add_argument('--test_data', type=str, help='Path to the test data')  
parser.add_argument('--config', type=str, default= '/scratch/groups/rhiju/gtully/rnet2d/sherlock_file_organization/ribonanzanet2d-final/configs/pairwise.yaml', help='Path to config')
parser.add_argument('--weights', type=str, default= '/scratch/groups/rhiju/gtully/rnet2d/sherlock_file_organization/ribonanzanet-weights/RibonanzaNet.pt', help='Path to pretrained weights')
parser.add_argument('--criterion', type=str, default='mcrse', choices=['mcrse', 'mae', 'mse'],
                    help="Choose the loss function to use: mcrse, mae, or mse.")
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--optimizer', type=str, default= 'Ranger', choices= ['Ranger', 'Adam'], help='optimizer')
parser.add_argument('--save_path', type=str, default='model.pt', help='Path to save the model')
parser.add_argument('--train_snr_cutoff', type=float, default=0, help= 'snr cutoff for training data')
parser.add_argument('--val_snr_cutoff', type=float, default=0, help= 'snr cutoff for test data used in validation loop') 
parser.add_argument('--cos_epochs', type=int, default=15, help= 'Number of cos epochs')


args = parser.parse_args()
##Load Reactivity Data

# Load the JSON file
train_data = pd.read_json(args.data)
val_data = pd.read_json(args.val_data)
test_data = pd.read_json(args.test_data)

from torch.utils.data import Dataset, DataLoader


class RNA_Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokens = {nt: i for i, nt in enumerate('ACGU')}
        self.label_names = ['Argi', 'Eryt', 'Kana', 'Mito', 'NoDr', 'Paro',
        'Spec', 'Tetr']
        self.SN_names = ['Argi_SN', 'Eryt_SN', 'Kana_SN', 'Mito_SN', 'NoDr_SN',
       'Paro_SN', 'Spec_SN', 'Tetr_SN']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = [self.tokens[nt] for nt in self.data.loc[idx, 'sequence']]
        sequence = torch.tensor(sequence, dtype=torch.long)

        labels = torch.tensor(np.stack([self.data.loc[idx, l] for l in self.label_names], -1), dtype=torch.float32)

        signal_to_noise = torch.tensor(np.stack([self.data.loc[idx, sn] for sn in self.SN_names], -1), dtype=torch.float32)

        return {'sequence': sequence, 'labels': labels, 'signal_to_noise': signal_to_noise}

#load the training data
train_dataset = RNA_Dataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

#load the validation data
val_dataset = RNA_Dataset(val_data)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

#load the test data 

test_dataset = RNA_Dataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  #Important that shuffle ==False!

import sys

sys.path.append("/scratch/groups/rhiju/gtully/rnet2d/sherlock_file_organization/ribonanzanet2d-final")

from Network import *
import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config, pretrained=False):
        super(finetuned_RibonanzaNet, self).__init__(config)
        if pretrained:
            self.load_state_dict(torch.load(args.weights,map_location='cpu'))
        self.decoder=nn.Linear(256,8)

    def forward(self,src):

        sequence_features, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))
        output=self.decoder(sequence_features)

        return output.squeeze(-1)

# Loss functions

def MCRMSE(y_pred,y_true):
    colwise_mse = torch.mean(torch.square(y_true - y_pred), axis=1)
    MCRMSE = torch.mean(torch.sqrt(colwise_mse), axis=1)
    return MCRMSE

#Define model and optimizer 
from ranger import Ranger
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr

config=load_config_from_yaml(args.config)
model=finetuned_RibonanzaNet(config=config,pretrained=True).cuda()

lr = args.lr

#criterion 
if args.criterion == 'mcrse':
    criterion = MCRMSE
elif args.criterion == 'mae':
    criterion = torch.nn.L1Loss()
else:
    criterion = torch.nn.MSELoss()

#optimizer 
if args.optimizer == 'Ranger':
    optimizer = Ranger(model.parameters(), weight_decay=0.001, lr= lr)

if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#Training Loop 

cos_epoch = args.cos_epochs
epochs = args.epochs
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - cos_epoch) * len(train_dataloader))

# Initialize lists to store losses for plotting
train_losses = []
val_losses = []
best_val_loss = np.inf

for epoch in range(epochs):
    model.train()
    tbar = tqdm(train_dataloader)
    total_loss = 0

    for idx, batch in enumerate(tbar):
        sequence = batch['sequence'].cuda()
        labels = batch['labels'].cuda()
        labels = labels.float()
        signal_to_noise = batch['signal_to_noise'].cuda()  # Shape: (batch_size, 8)

        for i in range(8):  # Iterate over each condition
            if signal_to_noise[:, i].item() > args.val_snr_cutoff:
                output = model(sequence).float()
                output = output[:, 26:76, :]  # Shape: (batch_size, 50, 8)

                loss = criterion(output, labels)
                loss = loss.mean()

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                optimizer.zero_grad()

                # Learning rate scheduling
                if (epoch + 1) > cos_epoch:
                    schedule.step()
                total_loss += loss.item()

        # Update progress bar
        tbar.set_description(f"Epoch {epoch + 1} Loss: {total_loss / (idx + 1):.4f}")

    # Validation loop
    model.eval()
    val_preds = []
    val_loss = 0

    tbar = tqdm(val_dataloader)

    for idx, batch in enumerate(tbar):
        sequence = batch['sequence'].cuda()
        labels = batch['labels'].cuda()
        signal_to_noise = batch['signal_to_noise'].cuda()  # Shape: (batch_size, 8)

        with torch.no_grad():
            output = model(sequence).float()  # Shape: (batch_size, 50, 8)
            output = output[:, 26:76, :]

            # Mask the output and labels where SN > 1 for each condition
            loss_per_condition = []
            for i in range(8):  # Iterate over each condition
                if signal_to_noise[:, i].item() > args.val_snr_cutoff:
                    # Calculate MSE loss only for this condition
                    loss = criterion(output, labels)
                    val_loss += loss.item()
            # Store predictions and true values for Pearson correlation calculation
            val_preds.append([labels.cpu().numpy(), output.cpu().numpy()])

    # Calculate final average validation loss
    avg_val_loss = val_loss / len(tbar)
    print(f"Validation loss (SN > {args.val_snr_cutoff}) : {avg_val_loss:.4f}")

    # Additional metrics for conditions with SN > 1
    true_flat = np.concatenate([x[0] for x in val_preds], axis=0).reshape(-1, 8)
    pred_flat = np.concatenate([x[1] for x in val_preds], axis=0).reshape(-1, 8)

    # Calculate MAE for SN > 1 filtered predictions
    mae_values = np.mean(np.abs(true_flat - pred_flat), axis=0)
    mae_avg = np.mean(mae_values)
    print("Mean Absolute Error (MAE) for each condition:", mae_values)
    print(f"Average Mean Absolute Error (MAE): {mae_avg:.4f}")

    # Pearson Correlation for each condition
    pcc_values = [float(pearsonr(true_flat[:, i], pred_flat[:, i])[0]) for i in range(true_flat.shape[1])]
    pcc_avg = np.mean(pcc_values)
    print("Pearson Correlation (PCC) for each condition:", pcc_values)
    print(f"Average Pearson Correlation (PCC): {pcc_avg:.4f}")
    
    # Save model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path} with improved val loss: {best_val_loss:.4f}")

print('Inference Loop Begins')
#Inference Loop  
model.load_state_dict(torch.load(args.save_path))
model.eval()
test_preds = []

tbar = tqdm(test_dataloader)

for idx, batch in enumerate(tbar):
  sequence = batch['sequence'].cuda()
  labels = batch['labels'].cuda()
  signal_to_noise = batch['signal_to_noise'].cuda()  # Shape: (batch_size, 8)

  with torch.no_grad():
    output = model(sequence).float()  # Shape: (batch_size, 50, 8)
    output = output[:, 26:76, :]
    output = output.cpu().numpy()
    output = np.squeeze(output)
    test_preds.append(output)

Argi_pred, Eryt_pred, Kana_pred, Mito_pred, Paro_pred, Spec_pred, Tetr_pred, NoDr_pred = [], [], [], [], [], [], [], []

# Iterate through each sequence's predicted reactivity matrix in val_preds
for pred in test_preds:
    # pred is of shape (50, 8), corresponding to 50 nucleotides and 8 conditions
    # We need to extract 50 values for each condition (i.e., each column)

    # Extract the 50 reactivity values for each condition
    Argi_pred.append(pred[:, 0])  # 50 reactivities for condition 1
    Eryt_pred.append(pred[:, 1])  # 50 reactivities for condition 2
    Kana_pred.append(pred[:, 2])  # 50 reactivities for condition 3
    Mito_pred.append(pred[:, 3])  # 50 reactivities for condition 4
    Paro_pred.append(pred[:, 4])  # 50 reactivities for condition 5
    Spec_pred.append(pred[:, 5])  # 50 reactivities for condition 6
    Tetr_pred.append(pred[:, 6])  # 50 reactivities for condition 7
    NoDr_pred.append(pred[:, 7])  # 50 reactivities for condition 8

# Now, each condition list (e.g., con1_pred) contains a list of 50 reactivity values per sequence
# Add these lists as new columns to the DataFrame
test_data['Argi_pred'] = Argi_pred
test_data['Eryt_pred'] = Eryt_pred
test_data['Kana_pred'] = Kana_pred
test_data['Mito_pred'] = Mito_pred
test_data['Paro_pred'] = Paro_pred
test_data['Spec_pred'] = Spec_pred
test_data['Tetr_pred'] = Tetr_pred
test_data['NoDr_pred'] = NoDr_pred

# Get the base name of the save path (without the extension)
base_name = os.path.splitext(os.path.basename(args.save_path))[0]

# Create the new file name with the base name
output_file = f'{base_name}_{criterion}_test_data_with_preds.json'

# Save the data to the new JSON file
test_data.to_json(output_file, index=False)

print(f'Inference loop of model with test set saved to {base_name}_{criterion}_test_data_with_preds.json')


