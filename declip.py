import os
from autoencoder import AutoEncoder, upsample
from functional import Functional
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.io import StreamReader
import torchaudio.transforms as T

# Hyperparameters
path = "/mnt/Elements08/Music/ml/test/"
weights_path = "/home/jto/Documents/AIDeclip/results/model01.pth"
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
    wav, _ = torchaudio.load(path+filename)

    # TODO: Resample if not expected sample rate

    # Transform wav to spectrogram
    funct = Functional(sample_rate, None)
    #wav = funct.transform(wav)

    # Add wav to list
    inputs.append([wav, filename, wav.shape[1]])

# Initialize model with pre-trained weights
model = AutoEncoder()
model.to(device)
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    for input, filename, time in inputs:
        #print(input.shape)
        input = torch.unsqueeze(input, 0)
        input = input.to(device)
        output = model(input)
        #output = upsample(output, [2049, time])
        output = torch.squeeze(output)
        output = output.to("cpu")
        torchaudio.save(output_path+filename, output, sample_rate)

