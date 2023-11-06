import os
from autoencoder import AutoEncoder, AutoEncoder2L
from functional import Functional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.io import StreamReader
import torchaudio.transforms as T

# Hyperparameters
path = "/mnt/Elements08/Music/ml/test/"
weights_path = "/home/jto/Documents/AIDeclip/results/model00.pth"
output_path = "/home/jto/Documents/AIDeclip/AIDeclipper/"
sample_rate = 44100

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
inputs = []
for filename in os.listdir(path):
    wav, _ = torchaudio.load(path+filename, normalize=True)

    # TODO: Resample if not expected sample rate

    # Transform wav to spectrogram
    funct = Functional(sample_rate, None)
    wav = funct.transform(wav)

    # Add wav to list
    inputs.append([wav, filename])

# Initialize model with pre-trained weights
model = AutoEncoder2L()
#model.to(device)
model.load_state_dict(torch.load(weights_path))
model.eval()

with torch.no_grad():
    for input, filename in inputs:
        #input = input.to(device)
        input = torch.unsqueeze(input, 0)
        output = model(input)
        input = torch.squeeze(input)
        #output = input
        funct.spec_to_wav(output, output_path+filename)

