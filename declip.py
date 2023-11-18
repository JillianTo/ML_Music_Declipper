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
path = "/mnt/MP600/data/comp/small/test/"
weights_path = "/home/jto/Documents/AIDeclip/results/model01.pth"
output_path = "/home/jto/Documents/AIDeclip/AIDeclipper/"
sample_rate = 44100
mean = -7.5930
std = 16.4029
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
    model = SpecAutoEncoder(device, mean, std)
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

        #input = funct.wav_to_complex(input)
        # Calculate phase of complex spectrogram
        #phase = torch.atan(input.imag/(input.real+1e-7))
        #phase[input.real < 0] += 3.14159265358979323846264338
        # Calculate magnitude of complex spectrogram
        #input = torch.sqrt(torch.pow(input.real, 2)+torch.pow(input.imag,2))
        #print(torch.min(input))
        #print(torch.max(input))
        #amp_to_db = T.AmplitudeToDB(stype='magnitude', top_db=8,0)
        #amp_to_db = amp_to_db.to(device)
        #input = amp_to_db(input)
        #print(torch.min(input))
        #print(torch.max(input))
        #input = F.DB_to_amplitude(input, 1, 0.5)
        #input = torch.polar(input, phase)
        #inv_spec = T.InverseSpectrogram(n_fft=4096)
        #inv_spec = inv_spec.to(device)
        #output = inv_spec(input)

        input = torch.unsqueeze(input, 0)
        output = model(input)
        output = torch.squeeze(output)

        funct.save_wav(output, output_path+f"{i}"+filename)
        print(f"Saved \"{output_path}{i}{filename}\"")
        i = i+1

