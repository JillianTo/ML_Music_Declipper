import os
from tqdm import tqdm
from audiodataset import AudioDataset
from functional import Functional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
print("Torch version: " + torch.__version__)

# Parameters
comp_path = "/mnt/MP600/data/comp/small/test/"
uncomp_path = "/mnt/MP600/data/uncomp/"
sample_rate = 44100
num_workers = 0
acc_bound = 2

# Get CPU, GPU, or MPS device for training
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
device = "cpu"
print(f"Using {device} device")

# Load wavs into training data
print("\nLoadling data...")

# Add inputs and labels to dataset
funct = Functional(sample_rate, None, device)
test_data = AudioDataset(funct, comp_path, uncomp_path, False)
print(f"Added {len(test_data)} file pairs to test data")

# Create data loaders
if __name__ == '__main__':
    testloader = DataLoader(test_data, batch_size=1, num_workers=num_workers, shuffle=False)

# Initialize model, loss function, and optimizer
print("\nInitializing loss function...")
criterion = nn.MSELoss()

# Training Loop
print("Starting test loop...")
avg_loss = 0
avg_acc = 0
pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Test")
for i, (comp_wav, uncomp_wav) in pbar:
    # Send tensors to device
    #comp_wav = comp_wav.to(device)
    #uncomp_wav = uncomp_wav.to(device)

    #comp_wav = funct.wav_to_db(comp_wav)
    #uncomp_wav = funct.wav_to_db(uncomp_wav)
    upsample = nn.Upsample(uncomp_wav.shape[2])
    comp_wav = upsample(comp_wav)

    #comp_wav = funct.wav_to_mel_db(comp_wav)
    #uncomp_wav = funct.wav_to_mel_db(uncomp_wav)
    comp_wav = funct.wav_to_pow_db(comp_wav)
    uncomp_wav = funct.wav_to_pow_db(uncomp_wav)

    # Compute the loss value: this measures how well the predicted outputs match the true labels
    #real_loss = criterion(comp_wav.real, uncomp_wav.real)
    #imag_loss = criterion(comp_wav.imag, uncomp_wav.imag)
    #loss = (real_loss+imag_loss)/2
    loss = criterion(comp_wav, uncomp_wav)

    # Compute accuracy
    acc = torch.sum(torch.logical_and((comp_wav > uncomp_wav-acc_bound), (comp_wav < uncomp_wav+acc_bound)))/torch.numel(uncomp_wav)

    # Add loss to total loss
    avg_loss += loss.item()
    avg_acc += acc

    # Update tqdm progress bar with fixed number of decimals for loss
    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

# Calculate average loss 
avg_loss /= len(testloader)

# Print average loss
print(f"Average MSE of test data: {avg_loss:>8f}\n")
print(f"Average accuracy of test data: {avg_acc:>8f}\n")
