import os
from audiodataset import AudioDataset
from autoencoder import SpecAutoEncoder, WavAutoEncoder
from functional import Functional

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

# Parameters
path = "/mnt/MP600/data/comp/declipTest/"
weights_path = "/home/jto/Documents/AIDeclip/results/model02.pth"
output_path = "/home/jto/Documents/AIDeclip/AIDeclipper/"
sample_rate = 44100
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
funct = Functional(sample_rate, 3500000, device)
dataset = AudioDataset(funct, path, None, True)

# Initialize model with pre-trained weights
if spectrogram_autoencoder:
    model = SpecAutoEncoder(device)
    print("Using spectrogram autoencoder")
else:
    model = WavAutoEncoder()
    print("Using waveform autoencoder")
model.to(device)
model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
model.eval()

with torch.no_grad():
    i = 0
    for input, filename in dataset:
        input = input.to(device)
        input = torch.unsqueeze(input, 0)
        output = model(input)
        output = torch.squeeze(output)
        funct.save_wav(output, output_path+f"{i}"+filename)
        print(f"Saved \"{output_path}{i}{filename}\"")
        i = i+1

