import os
from autoencoder import AutoEncoder, transform_tensor, spec_to_wav, pad_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.io import StreamReader
import torchaudio.transforms as T

# Hyperparameters
path = "/mnt/Elements08/Music/ml/test/"
weights_path = "./model.pth"
sample_rate = 44100

# Get CPU, GPU, or MPS device for inference
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
print(f"Using {device} device")

# Get files to declip
inputs = []
for filename in os.listdir(path):
    wav, wav_sample_rate = torchaudio.load(path+filename, normalize=True)

    # TODO: Resample if not expected sample rate

    # Transform wav to spectrogram
    #wav = pad_tensor(wav, 14000000)
    wav = transform_tensor(wav, sample_rate)

    # Add wav to list
    inputs.append(wav)

# Initialize model with pre-trained weights
model = AutoEncoder()
model.load_state_dict(torch.load(weights_path))
model.eval()


with torch.no_grad():
    for input in inputs:
        output = model(input)
        #output = input
        spec_to_wav(output, sample_rate, device)

