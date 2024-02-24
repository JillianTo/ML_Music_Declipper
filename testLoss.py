import os
import sys
import auraloss
import numpy as np
from tqdm import tqdm

from audiodataset import AudioDataset
from autoencoder import SpecAutoEncoder, WavAutoEncoder
from functional import Functional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
print("Torch version: " + torch.__version__)

# Parameters
mean = -7.6415
std = 14.3662
sample_rate = 44100
test_batch_size = 2
input_path = "/mnt/MP600/data/comp/small/test/"
label_path = "/mnt/MP600/data/uncomp/"
#test_max_time = 3500000
#test_max_time = 550000
test_max_time = 750000
num_workers = 0
prefetch_factor = None
pin_memory = False
acc_bound = 0.01
filelist = None

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

# Add inputs and labels to test dataset
test_funct = Functional(sample_rate, test_max_time, device)
if(test_batch_size > 1):
    test_data = AudioDataset(test_funct, 
                             input_path,
                             filelist,
                             label_path=label_path, 
                             same_time=test_batch_size!=1, 
                             pad_thres=2)
else:
    test_data = AudioDataset(test_funct, 
                             input_path,
                             filelist,
                             same_time=test_batch_size!=1, 
                             pad_thres=2)
print(f"Added {len(test_data)} file pairs to test data")

# Create data loaders
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    testloader = DataLoader(test_data, batch_size=test_batch_size, 
                            shuffle=False, num_workers=num_workers, 
                            pin_memory=pin_memory, 
                            prefetch_factor=prefetch_factor)

# Calculate mean and std for training set if not given
if(mean == None or std == None):
    print("Calculating mean and std...")
    mean = 0
    std = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), 
                desc=f"Train Mean and Standard Deviation")

    for i, (comp_wav, _) in pbar:
        curr_std, curr_mean = funct.compute_std_mean(comp_wav)
        mean = mean + curr_mean
        std = std + curr_std

        # Update tqdm progress bar with fixed number of decimals for loss
        pbar.set_postfix({"Mean": f"{curr_mean:.4f}", 
                          "Standard Deviation": f"{curr_std:.4f}"})

    mean = mean / (i+1)
    std = std / (i+1)
    print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")

# Initialize loss function
fft_sizes = [4096, 8192, 16384]
#criterion = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, 
#                                                  hop_sizes=[fft_size//8 for 
#                                                             fft_size in 
#                                                             fft_sizes], 
#                                                  win_lengths=fft_sizes,
                                                  #w_sc = 1.0,
                                                  #w_log_mag = 1.0,
                                                  #w_lin_mag = 1.0,
#                                                  sample_rate=sample_rate, 
#                                                  perceptual_weighting=False, 
#                                                  scale='mel', n_bins=64, 
#                                                  scale_invariance=True)
criterion = auraloss.time.SISDRLoss()
criterion.to(device)

print("\nCalculating loss...")
tot_loss = 0
tot_acc = 0
avg_loss = 0
avg_acc = 0
pbar = tqdm(enumerate(testloader), total=len(testloader), 
            desc=f"Test")
for i, (comp_wav, fileinfo) in pbar:

    # Get uncompressed waveform
    if(test_batch_size > 1):
        uncomp_wav = fileinfo
    else:
        fileinfo = (fileinfo[0][0], fileinfo[1].item())
        uncomp_wav = test_funct.process_wav(label_path, fileinfo, 
                                            test_batch_size != 1, 
                                            is_input=False)  
        uncomp_wav = torch.unsqueeze(uncomp_wav, dim=0)
        # Make sure input waveform is same length as label
        if(uncomp_wav.shape[2] != comp_wav.shape[2]):
            print(f"{fileinfo[0][0]} mismatched size, input is " 
                  f"{comp_wav.shape[2]} and label is "
                  f"{uncomp_wav.shape[2]}")
            comp_wav = funct.upsample(comp_wav, uncomp_wav.shape[2])

    # Calculate accuracy 
    acc = (torch.sum(torch.logical_and((comp_wav > 
                                        uncomp_wav-acc_bound), 
                                       (comp_wav < uncomp_wav
                                        +acc_bound)))
           / torch.numel(uncomp_wav))

    # Calculate individual channel loss
    #loss = criterion(comp_wav, uncomp_wav)

    # Sum waveforms to one channel
    comp_wav = comp_wav[:,0,:] + comp_wav[:,1,:]  
    comp_wav = comp_wav.unsqueeze(1)
    uncomp_wav = uncomp_wav[:,0,:] + uncomp_wav[:,1,:]  
    uncomp_wav = uncomp_wav.unsqueeze(1)

    # Calculate sum loss
    sum_loss = criterion(comp_wav, uncomp_wav)

    # Average sum and individual losses
    #loss = (loss+sum_loss) / 2
    loss = sum_loss

    # Explicitly delete tensors so they don't stay in memory
    del comp_wav
    del uncomp_wav

    # Add loss and accuracy to totals
    tot_loss += (abs(loss.item()))
    tot_acc += acc

    # Calculate average loss and accuracy so far
    avg_loss = tot_loss / (i+1)
    avg_acc = tot_acc / (i+1)

    # Update tqdm progress bar with fixed number of decimals for loss
    pbar.set_postfix({"Loss": f"{loss.item():.4f}", 
                      "Avg Loss": f"{avg_loss:.4f}", 
                      "Acc": f"{acc:.4f}", 
                      "Avg Acc": f"{avg_acc:.4f}"})

