import os
from tqdm import tqdm
from audiodataset import AudioDataset
from autoencoder import AutoEncoder, upsample 
from functional import Functional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
print("Torch version: " + torch.__version__)

# Save run parameters to variables for easier access
comp_path = "/home/jto/Documents/AIDeclip/data/comp/test/"
uncomp_path = "/home/jto/Documents/AIDeclip/data/uncomp/"
sample_rate = 44100

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
print("\nLoading data...")

# Add inputs and labels to dataset
funct = Functional(sample_rate, None)
test_data = AudioDataset(funct, comp_path, uncomp_path, False)
print(f"Added {len(test_data)} file pairs to test data")

# Create data loaders
testloader = DataLoader(test_data, batch_size=1, shuffle=False)

# Initialize model, loss function, and optimizer
print("\nInitializing loss function...")
criterion = nn.MSELoss()

# Training Loop
print("Starting test loop...")
avg_loss = 0
acc = 0
pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Test")
for i, (comp_wav, uncomp_wav) in pbar:
    # Send tensors to device
    comp_wav = comp_wav.to(device)
    uncomp_wav = uncomp_wav.to(device)

    # Upsample output to match uncompressed wav
    comp_wav = upsample(comp_wav, [uncomp_wav.shape[2], uncomp_wav.shape[3]])

    # Compute the loss value: this measures how well the predicted outputs match the true labels
    real_loss = criterion(comp_wav.real, uncomp_wav.real)
    imag_loss = criterion(comp_wav.imag, uncomp_wav.imag)
    loss = (real_loss+imag_loss)/2

    # Add loss to total loss
    avg_loss += loss.item()

    # Update tqdm progress bar with fixed number of decimals for loss
    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

# Calculate average loss 
avg_loss /= len(testloader)

# Print average loss
print(f"Average test loss: {avg_loss:>8f}\n")
