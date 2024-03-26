import os
import sys
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
#weights_path = "/mnt/PC801/declip/results/03-10/model06.pth"
weights_path = "/mnt/PC801/declip/results/model01.pth"
output_path = "/mnt/PC801/declip/new/"
sample_rate = 44100
#mean = -0.4622
#mean = -3.9821
#std = 13.8257
#std = 13.9869
spectrogram_autoencoder = True
part_time = 2250000
#part_time = 2750000
overlap_factor = 20
extra_factor = 0.999
fade_shape = 'logarithmic'
save_part_wav = False
test_fade = False
norm_thres = 0.01
eq = True
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

# Get mean and std from file
stats_path = "db_stats.txt"
if(os.path.isfile(stats_path)):
    with open(stats_path, 'rb') as f:
        db_stats = pickle.load(f)
        mean = db_stats[0]
        std = db_stats[1]
else:
    sys.exit('No mean or std provided, force quitting.')

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
def crossfade(filename, path, output_path, part_time, overlap_factor, extra_factor, fade_shape='logarithmic', norm_thres=0.01):
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
        # Load part
        part = torch.load(output_path+f"{i}_"+filename+".pt")

        if(i == part_num):
            time = torchaudio.info(path+filename).num_frames
            left_mean = funct.mean(abs(whole_wav[:, time-part_time:]))
            right_mean = funct.mean(abs(part[:, :whole_wav.shape[1]-(time-part_time)]))
        else: 
            left_mean = funct.mean(abs(whole_wav[:, whole_wav.shape[1]-fade_time-extra_time:]))
            right_mean = funct.mean(abs(part[:, :fade_time+extra_time]))

        mean_diff = left_mean-right_mean
        #print(mean_diff)
        # Equalize volume between two parts
        # If combined waveform so far is louder than part by an amount greater than threshold
        if(mean_diff > norm_thres):
            whole_wav = whole_wav*(right_mean/left_mean)
        # If part is louder than combined waveform
        elif (mean_diff < -norm_thres):
            part = part*(left_mean/right_mean)
        #else:
        #    print(f"Not normalizing part {i} because difference in means is only {mean_diff}")

        # Remove extra time from beginning
        part = part[:, extra_time:]

        # If last part
        if(i == part_num):
            time = time-part_time+extra_time
            overlap_part = fade_out(whole_wav[:,:time+fade_time])[:, time:time+fade_time]
        # If not first or last part
        else:
            time = whole_wav.shape[1]-fade_time
            overlap_part = fade_out(whole_wav)[:, time:]

        overlap_part = overlap_part + fade_in(part)[:, :fade_time]
        whole_wav = whole_wav[:,:time]

        # Combine parts
        whole_wav = torch.cat((whole_wav,overlap_part,part[:,fade_time:]), dim=1)

    # Normalize peak to 0dB
    peak = max(torch.max(whole_wav).item(), abs(torch.min(whole_wav).item()))
    whole_wav = whole_wav * (1.0/peak)

    return whole_wav

# Equalize input tensor to same frequency response as file in filename
def equalize(tensor, filename, save_noeq_wav=False, thres=4): 
    # Load original waveform
    orig_wav, _ = torchaudio.load(path+filename)
    orig_wav = orig_wav.to(device)

    # Save copy of input tensor
    edit_wav = tensor

    # Merge both waveforms so dB conversion has consistent scale
    edit_wav = torch.cat((orig_wav,edit_wav),dim=1)

    # Convert combined waveform into dB scale Mel spectrogram
    edit_wav = funct.wav_to_mel_db(edit_wav)

    # Separate combined waveform
    orig_wav = edit_wav[:,:,:int(edit_wav.shape[2]/2)]
    edit_wav = edit_wav[:,:,int(edit_wav.shape[2]/2):]

    # Equalize volumes
    orig_mean = funct.mean(orig_wav)
    edit_mean = funct.mean(edit_wav)
    if(orig_mean < edit_mean):
        edit_wav = edit_wav*(orig_mean/edit_mean)
    else:
        orig_wav = orig_wav*(edit_mean/orig_mean)

    # Consider third fourth of mel filterbanks
    lower_mel = int(orig_wav.shape[1]*0.5)
    upper_mel = int(orig_wav.shape[1]*0.75)
    orig_wav = orig_wav[:,lower_mel:upper_mel,:]
    edit_wav = edit_wav[:,lower_mel:upper_mel,:]
    orig_db = funct.mean(orig_wav)
    edit_db = funct.mean(edit_wav)
    del orig_wav
    del edit_wav

    diff_db = orig_db-edit_db
    eq_db = diff_db

    if(eq_db > 1):
        if(eq_db > thres):
            print(f"Difference of {diff_db}dB too large, reducing to {thres}dB")
            eq_db = thres
        #q=0.707*math.log2(eq_db)
        q=0.707
        eq_freqs = [4,8,12,16,20]
        print(f"EQing {eq_freqs[0]}kHz to {eq_freqs[len(eq_freqs)-1]}kHz by {eq_db}dB")
        eq_freqs = [f * 1000 for f in eq_freqs]
        for f in eq_freqs:
            tensor = F.equalizer_biquad(tensor, sample_rate, f, eq_db, Q=q)
    else:
        print(f"Difference only {diff_db}dB, skipping EQ")

    return tensor

with torch.no_grad():
    curr_filename = None
    part_num = 1
    for tensor, fileinfo in dataset:
        if(curr_filename != fileinfo[0]):
            if(curr_filename != None):
                curr_tensor = crossfade(filename=curr_filename, path=path, output_path=output_path, part_time=part_time, overlap_factor=overlap_factor, extra_factor=extra_factor, fade_shape=fade_shape, norm_thres=norm_thres)
                # Equalize file to same frequency response as original
                if(eq):
                    # Save whold waveform before EQ
                    if(save_noeq_wav):
                        funct.save_wav(curr_tensor, output_path+f"noeq_"+curr_filename)

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
    tensor = crossfade(filename=curr_filename, path=path, output_path=output_path, part_time=part_time, overlap_factor=overlap_factor, extra_factor=extra_factor, fade_shape=fade_shape, norm_thres=norm_thres)

    # Equalize file to same frequency response as original
    if(eq):
        # Save whold waveform before EQ
        if(save_noeq_wav):
            funct.save_wav(tensor, output_path+f"noeq_"+curr_filename)

        tensor = equalize(tensor, filename=curr_filename, save_noeq_wav=save_noeq_wav)

    # Save complete waveform
    funct.save_wav(tensor, output_path+curr_filename)
