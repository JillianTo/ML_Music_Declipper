import os
from audiodataset import AudioDataset
from autoencoder import SpecAutoEncoder, WavAutoEncoder
from functional import Functional

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# Parameters
path = "/mnt/MP600/data/comp/declip/"
weights_path = "/mnt/MP600/data/results/12-08/model01.pth"
output_path = "/mnt/MP600/AIDeclip/AIDeclipper/"
sample_rate = 44100
mean = -7.6415
std = 14.3662
spectrogram_autoencoder = True

# Get CPU, GPU, or MPS device for inference
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
device = "cpu"
print(f"Using {device} device")

# Get files to declip
funct = Functional(sample_rate, 7250000, device, n_fft=4096)
dataset = AudioDataset(funct, path, None, same_time=False, pad_thres=999)

# Initialize model with pre-trained weights
if spectrogram_autoencoder:
    model = SpecAutoEncoder(device, mean, std)
    print("Using spectrogram autoencoder")
else:
    model = WavAutoEncoder(device)
    print("Using waveform autoencoder")
model.to(device)
model.load_state_dict(torch.load(weights_path, 
                                 map_location=torch.device(device)))
model.eval()

with torch.no_grad():
    i = 0
    for tensor, fileinfo in dataset:
        tensor = tensor.to(device)

        tensor = torch.unsqueeze(tensor, 0)
        tensor = model(tensor)
        tensor = torch.squeeze(tensor)

        funct.save_wav(tensor, output_path+f"{i}"+fileinfo[0])
        print(f"Saved \"{output_path}{i}{fileinfo[0]}\"")
        i += 1

        # Explicitly delete tensor so it doesn't stay in memory
        del tensor

