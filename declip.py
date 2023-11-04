import os
from autoencoder import AutoEncoder, process_wavs, transform_tensor, upsample_tensor
import numpy as np
import librosa
from scipy.io import wavfile

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

    # Transform wav to Mel spectrogram
    wav = transform_tensor(wav, sample_rate)

    # Add wav to list
    inputs.append(wav)

# Initialize model with pre-trained weights
model = AutoEncoder()
model.load_state_dict(torch.load(weights_path))
model.eval()

n_fft = 400

with torch.no_grad():
    for input in inputs:
        output = model(wav)
        output = output.unsqueeze(-3)
        output = upsample_tensor(output, [wav.shape[1], wav.shape[2]])
        output = output.squeeze(-3)
        output_array = output.numpy()
        output_array = librosa.feature.inverse.mel_to_audio(output_array, sr=sample_rate, n_fft=n_fft, hop_length=n_fft // 2)
        # Set the desired output path and filename
        output_path = "./"
        filename = "output_audio.wav"

        # Save the output as a WAV file
        wavfile.write(output_path + filename, sample_rate, output_array.T)

