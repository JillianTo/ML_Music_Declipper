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
#path = "/mnt/MP600/data/comp/small/declipTest/"
path = "/mnt/MP600/data/comp/declipTest/"
#weights_path = "/mnt/MP600/data/results/test/model18.pth"
weights_path = "/mnt/MP600/data/results/model01.pth"
output_path = "/home/jto/Documents/AIDeclip/AIDeclipper/"
sample_rate = 44100
#mean = -7.5930
#std = 16.4029
mean = -9.0133
std = 14.3514 
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
#funct = Functional(sample_rate, 9200000, device)
funct = Functional(sample_rate, 20000000, device, n_fft=2048)
dataset = AudioDataset(funct, path, None, same_time=False, pad_thres=999)

# Initialize model with pre-trained weights
if spectrogram_autoencoder:
    model = SpecAutoEncoder(device, mean, std)
    print("Using spectrogram autoencoder")
else:
    model = WavAutoEncoder(device)
    print("Using waveform autoencoder")
model.to(device)
model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
model.eval()

with torch.no_grad():
    i = 0
    for tensor, fileinfo in dataset:
        tensor = tensor.to(device)

        #tensor = funct.wav_to_complex(tensor)
        # Calculate phase of complex spectrogram
        #phase = torch.atan(tensor.imag/(tensor.real+1e-7))
        #phase[tensor.real < 0] += 3.14159265358979323846264338
        # Calculate magnitude of complex spectrogram
        #tensor = torch.sqrt(torch.pow(tensor.real, 2)+torch.pow(tensor.imag,2))
        #amp_to_db = T.AmplitudeToDB(stype='magnitude', top_db=80)
        #amp_to_db = amp_to_db.to(device)
        #tensor = amp_to_db(tensor)
        #tensor = F.DB_to_amplitude(tensor, 1, 0.5)
        #tensor = torch.polar(tensor, phase)
        #inv_spec = T.InverseSpectrogram(n_fft=2048)
        #inv_spec = inv_spec.to(device)
        #tensor = inv_spec(tensor)

        tensor = torch.unsqueeze(tensor, 0)
        tensor = model(tensor)
        tensor = torch.squeeze(tensor)

        funct.save_wav(tensor, output_path+f"{i}"+fileinfo[0])
        print(f"Saved \"{output_path}{i}{fileinfo[0]}\"")
        i = i+1

        # Explicitly delete tensor so it doesn't stay in memory
        del tensor

