import os
import math
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
#weights_path = "/mnt/MP600/data/results/02-16/model04.pth"
weights_path = "/mnt/MP600/data/results/model06.pth"
output_path = "/mnt/PC801/declip/new/"
sample_rate = 44100
mean = -0.4622
std = 13.8257
spectrogram_autoencoder = True
part_time = 2250000
#part_time = 2750000
overlap_factor = 20
extra_factor = 0.999
fade_shape = 'logarithmic'
save_part_wav = False
test_fade = True
norm = True
norm_part = True
eq = True
save_nonorm_wav = False
save_noeq_wav = True

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

# Combine file parts
def crossfade(filename, path, output_path, part_time, overlap_factor, extra_factor, fade_shape='logarithmic', norm=True, norm_part=False, save_nonorm_wav=False):
    # Calculate time of overlap between parts
    fade_time = int(part_time/overlap_factor)
    # Calculate time from beginning of part to not include in overlap
    extra_time = int(fade_time*extra_factor)
    # Remove extra time from fade time
    fade_time = fade_time-extra_time
    # Initialize tensor for storing whole waveform
    whole_wav = torch.load(output_path+"1_"+filename+".pt")
    # Initialize fade transforms
    fade_in = T.Fade(fade_in_len=fade_time, fade_shape=fade_shape)
    fade_out = T.Fade(fade_out_len=fade_time, fade_shape=fade_shape)
    for i in range(2, part_num+1):
        part = torch.load(output_path+f"{i}_"+filename+".pt")
        if(norm_part):
            # Equalize volume between two parts
            left_mean = funct.mean(abs(whole_wav[:, whole_wav.shape[1]-fade_time-extra_time:]))
            right_mean = funct.mean(abs(part[:, :fade_time+extra_time]))
            thres = 0.02
            mean_diff = left_mean-right_mean
            # If combined waveform so far is louder than part by an amount greater than threshold
            if(mean_diff > thres):
                whole_wav = whole_wav*(right_mean/left_mean)
            elif (mean_diff < -thres):
                part = part*(left_mean/right_mean)
            else:
                #print(f"Not normalizing part {i} because difference in means is only {mean_diff}")

        # Remove extra time from beginning
        part = part[:, extra_time:]

        # If last part
        if(i == part_num):
            time = torchaudio.info(path+filename).num_frames
            overlap_part = fade_out(whole_wav[:,:time-part_time+extra_time+fade_time])[:, time-part_time+extra_time:time-part_time+extra_time+fade_time] + fade_in(part)[:, :fade_time]
            whole_wav = whole_wav[:,:time-part_time+extra_time]
        # If not first or last part
        else:
            overlap_part = fade_out(whole_wav)[:, whole_wav.shape[1]-fade_time:] + fade_in(part)[:, :fade_time]
            whole_wav = whole_wav[:,:whole_wav.shape[1]-fade_time]

        # Combine parts
        whole_wav = torch.cat((whole_wav,overlap_part,part[:,fade_time:]), dim=1)

    # Normalize peak to 0dB
    if(norm):
        if(save_nonorm_wav):
            funct.save_wav(whole_wav, output_path+f"nonorm_"+filename)
        peak = max(torch.max(whole_wav).item(), abs(torch.min(whole_wav).item()))
        whole_wav = whole_wav * (1.0/peak)

    return whole_wav

# Equalize input tensor to same frequency response as file in filename
def equalize(whole_wav, filename, save_noeq_wav=False): 
    if(save_noeq_wav):
        funct.save_wav(whole_wav, output_path+f"noeq_"+filename)
    orig_wav, _ = torchaudio.load(path+filename)
    edit_wav = whole_wav
    orig_wav = funct.wav_to_mel_db(orig_wav)
    edit_wav = funct.wav_to_mel_db(edit_wav)
    _, orig_mean = torch.std_mean(orig_wav)
    _, edit_mean = torch.std_mean(edit_wav)
    orig_wav = orig_wav * (abs(edit_mean/orig_mean))

    # Consider third fourth of mel filterbanks
    lower_mel = int(orig_wav.shape[1]*0.5)
    upper_mel = int(orig_wav.shape[1]*0.75)
    orig_wav_part = orig_wav[:,lower_mel:upper_mel,:]
    edit_wav_part = edit_wav[:,lower_mel:upper_mel,:]
    orig_db = funct.mean(orig_wav_part)
    edit_db = funct.mean(edit_wav_part)
    del orig_wav_part
    del edit_wav_part

    diff_db = orig_db-edit_db
    eq_db = diff_db

    if(eq_db > 1):
        #q=0.707*math.log2(eq_db)
        q=0.707
        eq_freqs = [4,8,12,16,20]
        print(f"EQing {eq_freqs[0]}kHz to {eq_freqs[len(eq_freqs)-1]}kHz by {eq_db}dB")
        eq_freqs = [f * 1000 for f in eq_freqs]
        for f in eq_freqs:
            whole_wav = F.equalizer_biquad(whole_wav, sample_rate, f, eq_db, Q=q)
    else:
        print(f"Difference only {diff_db}dB, skipping EQ")

    return whole_wav

with torch.no_grad():
    curr_filename = None
    part_num = 1
    for tensor, fileinfo in dataset:
        if(curr_filename != fileinfo[0]):
            if(curr_filename != None):
                curr_tensor = crossfade(filename=curr_filename, path=path, output_path=output_path, part_time=part_time, overlap_factor=overlap_factor, extra_factor=extra_factor, fade_shape=fade_shape, norm=norm, norm_part=norm_part, save_nonorm_wav=save_nonorm_wav)
                # Equalize file to same frequency response as original
                if(eq):
                    curr_tensor = equalize(curr_tensor, filename=curr_filename, save_noeq_wav=save_noeq_wav)

                # Save complete waveform
                funct.save_wav(curr_tensor, output_path+curr_filename)

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
            filename = output_path + f"{part_num}_" + fileinfo[0]
            torch.save(tensor, filename+".pt")
            print(f"Saved \"{filename}.pt\"")
            if(save_part_wav):
                funct.save_wav(tensor, filename)

            # Explicitly delete tensor so it doesn't stay in memory
            del tensor

    # Crossfade last file
    tensor = crossfade(filename=curr_filename, path=path, output_path=output_path, part_time=part_time, overlap_factor=overlap_factor, extra_factor=extra_factor, fade_shape=fade_shape, norm=norm, norm_part=norm_part, save_nonorm_wav=save_nonorm_wav)

    # Equalize file to same frequency response as original
    if(eq):
        tensor = equalize(tensor, filename=curr_filename, save_noeq_wav=save_noeq_wav)

    # Save complete waveform
    funct.save_wav(tensor, output_path+curr_filename)
