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
path = "/mnt/PC801/declip/"
#path = "/mnt/MP600/data/comp/testDeclip/"
weights_path = "/mnt/MP600/data/results/02-16/model04.pth"
#weights_path = "/mnt/MP600/data/results/model01.pth"
output_path = "/mnt/PC801/declip/new/"
sample_rate = 44100
mean = -1.1004
std = 13.9308
spectrogram_autoencoder = True
norm = True
part_time = 2250000
#part_time = 2750000
overlap_factor = 1000
extra_factor = 0.1
fade_shape = 'logarithmic'
save_part_wav = False
test_fade = False

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
funct = Functional(sample_rate, part_time, device)
dataset = AudioDataset(funct, path, None, same_time=False, pad_thres=999, 
                       overlap_factor=overlap_factor)

# Initialize model with pre-trained weights
if spectrogram_autoencoder:
    model = SpecAutoEncoder(mean, std)
    print("Using spectrogram autoencoder")
else:
    model = WavAutoEncoder(device)
    print("Using waveform autoencoder")
model.to(device)
model.load_state_dict(torch.load(weights_path, 
                                 map_location=torch.device(device)))
model.eval()

def crossfade(filename):
    # Calculate time of overlap between parts
    fade_time = int(part_time/overlap_factor)
    # Calculate time from beginning of part to not include in overlap
    extra_time = int(fade_time*extra_factor)
    # Remove extra time from fade time
    fade_time = fade_time-extra_time
    # Initialize tensor for storing whole waveform
    whole_wav = torch.load(output_path+f"{1}"+filename+".pt")
    # Initialize fade transforms
    fade_in = T.Fade(fade_in_len=fade_time, fade_shape=fade_shape)
    fade_out = T.Fade(fade_out_len=fade_time, fade_shape=fade_shape)
    for i in range(2, part_num+1):
        part = torch.load(output_path+f"{i}"+filename+".pt")
        # Remove extra time from beginning
        part = part[:, extra_time:]
        if(i == part_num):
            time = torchaudio.info(path+filename).num_frames
            overlap_part = fade_out(whole_wav[:,:time-part_time+extra_time+fade_time])[:, time-part_time+extra_time:time-part_time+extra_time+fade_time] + fade_in(part)[:, :fade_time]
            whole_wav = whole_wav[:,:time-part_time+extra_time]
        else:
            overlap_part = fade_out(whole_wav)[:, whole_wav.shape[1]-fade_time:] + fade_in(part)[:, :fade_time]
            whole_wav = whole_wav[:,:whole_wav.shape[1]-fade_time]
        whole_wav = torch.cat((whole_wav,overlap_part,part[:,fade_time:]), dim=1)

    # Normalize peak to 0dB
    if(norm):
        peak = max(torch.max(whole_wav).item(), abs(torch.min(whole_wav).item()))
        whole_wav = whole_wav * (1.0/peak)

    # Save complete waveform
    funct.save_wav(whole_wav, output_path+filename)
    print(f"Saved \"{output_path}{filename}\"")

    # Explicitly delete tensors so they don't stay in memory
    #del part
    #del overlap_part
    #del whole_wav


with torch.no_grad():
    curr_filename = None
    part_num = 1
    for tensor, fileinfo in dataset:
        if(curr_filename != fileinfo[0]):
            if(curr_filename != None):
                crossfade(curr_filename)
            # Reset part number and set current filename to new file
            part_num = 1
            curr_filename = fileinfo[0]
        else:
            part_num += 1

        if(not test_fade):
            tensor = tensor.to(device)

            tensor = torch.unsqueeze(tensor, 0)
            tensor = model(tensor)
            tensor = torch.squeeze(tensor)

            # Save output to files
            filename = output_path + f"{part_num}" + fileinfo[0]
            torch.save(tensor, filename+".pt")
            print(f"Saved \"{filename}.pt\"")
            if(save_part_wav):
                funct.save_wav(tensor, filename)
                print(f"Saved \"{filename}\"")

            # Explicitly delete tensor so it doesn't stay in memory
            del tensor

    # Crossfade last file
    crossfade(curr_filename)

