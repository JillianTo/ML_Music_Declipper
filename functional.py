import os
import math
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

class Functional():

    def __init__(self, sample_rate, max_time, device, n_fft=4096):
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.device = device
        self.n_fft = n_fft

    def wav_to_mel_db(self, tensor, n_mels=64, top_db=80):
        mel = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, f_max=self.sample_rate>>1, n_mels=n_mels)
        mel = mel.to(self.device)
        amp_to_db = T.AmplitudeToDB("power", top_db=top_db)
        amp_to_db = amp_to_db.to(self.device)
        
        return amp_to_db(mel(tensor))

    def wav_to_pow_db(self, tensor, top_db=80):
        pow_spec = T.Spectrogram(n_fft=self.n_fft, power=2)
        pow_spec = pow_spec.to(self.device)
        amp_to_db = T.AmplitudeToDB("power", top_db=top_db)
        amp_to_db = amp_to_db.to(self.device)
        
        return amp_to_db(pow_spec(tensor))

    def wav_to_complex(self, tensor):
        complex_spec = T.Spectrogram(n_fft=self.n_fft, power=None)
        complex_spec = complex_spec.to(self.device)
        return  complex_spec(tensor)

    def wav_to_db(self, tensor, top_db=80):
        amp_to_db = T.AmplitudeToDB(stype="amplitude", top_db=top_db)
        amp_to_db = amp_to_db.to(self.device)
        return amp_to_db(tensor)

    def compute_std_mean(self, tensor, top_db=80):
        # Convert input to complex spectrogram
        x = self.wav_to_complex(tensor)

        # Calculate magnitude of complex spectrogram
        x = torch.sqrt(torch.pow(x.real, 2)+torch.pow(x.imag,2))
        amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=top_db)
        amp_to_db = amp_to_db.to(self.device)
        x = amp_to_db(x)

        # Calculate standard deviation and mean
        return torch.std_mean(x)

    def find_max_min(self, tensor):
        # Convert input to complex spectrogram
        x = self.wav_to_complex(tensor, n_fft=self.n_fft)

        # Calculate phase of complex spectrogram
        phase = torch.atan(x.imag/(x.real+1e-7))
        phase[x.real < 0] += 3.14159265358979323846264338

        # Calculate magnitude of complex spectrogram
        x = torch.sqrt(torch.pow(x.real, 2)+torch.pow(x.imag,2))
        amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=80)
        amp_to_db = amp_to_db.to(self.device)
        x = amp_to_db(x)

        # Calculate maximum and minimum
        return torch.max(x), torch.min(x)

    def upsample(self, tensor, time):
        upsampler = nn.Upsample(time)
        upsampler = upsampler.to(self.device)
        return upsampler(tensor)

    def resample(self, tensor, old_sample_rate):
        resampler = T.Resample(old_sample_rate, self.sample_rate, dtype=tensor.dtype)
        resampler = resampler.to(self.device)
        return resampler(tensor)

    def pad(self, tensor):
        w_padding = self.max_time - tensor.shape[1]
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        return F.pad(tensor, (left_pad, right_pad))

    def get_filelist(self, input_path, pad_thres, filelist_path):
        # Filelist was saved previously, load it
        if(filelist_path != None and os.path.isfile(filelist_path)):
            with open(filelist_path, 'rb') as f:
                filelist = pickle.load(f)
            print(f"Loaded filelist from \"{filelist_path}\"")
        # Filelist has not been generated yet, create it
        else:
            filelist = []

            for filename in tqdm(os.listdir(input_path)):
                time = torchaudio.info(input_path+filename).num_frames
                # Waveform is longer than time cut-off, needs to be split
                if(time > self.max_time):
                    remainder = time % self.max_time
                    # Round down number of parts to split
                    num_parts = int(time/self.max_time)
                    # Append filename and time start of each split part
                    for i in range(0, num_parts):
                        filelist.append([filename, self.max_time*i])
                    # If remainder is greater than pad threshold, add waveform ending at end of file with length max_time
                    if(remainder > (self.max_time/pad_thres)):
                        filelist.append([filename, time-self.max_time])
                # Waveform is shorter than or equal to time cut off and long enough for FFT, does not need to be split
                elif(time > self.n_fft):
                    filelist.append([filename, -1])
                # Waveform not long enough for FFT, skip
                else:
                    print(f"Skipping \"{input_path}{filename}\" because too short for FFT")

            if(filelist_path != None):
                with open(filelist_path, 'wb') as f:
                    pickle.dump(filelist, f)
                print(f"Saved filelist in \"{filelist_path}\'")

        return filelist

    # Tensor should be waveform output of torchaudio.load()
    def split(self, tensor, split_start, same_time):
        # Waveform does not need to be split
        if(split_start < 0):
            # Pad if all waveforms need to be same time
            if(same_time):
                return self.pad(tensor)
            # Else, return waveform as-is
            else:
                return tensor
        # Crop long waveform
        else:
            end_time = split_start + self.max_time
            # If split waveform length would be less than max_time
            if(end_time > tensor.shape[1]):
                # Pad if all waveforms need to be same time
                if(same_time):
                    return self.pad(tensor[:, split_start:])
                # Return split part as-is if time is enough for FFT
                elif(end_time > self.n_fft):
                    return tensor[:, split_start:]
            # Split waveform fits in max_time, return waveform starting at time split_start with length max_time
            else:
                return tensor[:, split_start:split_start+self.max_time]

    # Return waveform 
    def process_wav(self, path, fileinfo, same_time, is_input=True):
        # Load waveform
        filename = fileinfo[0]
        if(not is_input):
            filename = filename.replace(".flac-compressed-", "")

        wav_path = path + filename
        wav, sample_rate = torchaudio.load(wav_path)

        # Move waveform to device
        wav = wav.to(self.device)

        # Resample if not expected sample rate
        if(sample_rate != self.sample_rate):
            #print(f"\"{wav_path}\" has sample rate of {sample_rate}Hz, resampling")
            wav = self.resample(wav, sample_rate)

        # Split waveform
        wav = self.split(wav, fileinfo[1], same_time)

        # Return waveform
        return wav
    
    # Return input and label waveforms
    def process_wavs(self, input_path, label_path, fileinfo, same_time):
        # Load waveforms
        filename = fileinfo[0]
        input_wav_path = input_path + filename
        input_wav, input_sample_rate = torchaudio.load(input_wav_path)
        label_wav_path = label_path + filename.replace(".flac-compressed-", "")
        label_wav, label_sample_rate = torchaudio.load(label_wav_path)

        # Move waveforms to device
        input_wav = input_wav.to(self.device)
        label_wav = label_wav.to(self.device)

        # Resample if not expected sample rate
        if(input_sample_rate != self.sample_rate):
            #print(f"\"{input_wav_path}\" has sample rate of {input_sample_rate}Hz, resampling")
            input_wav = self.resample(input_wav, input_sample_rate)

        if(label_sample_rate != self.sample_rate):
            #print(f"\"{label_wav_path}\" has sample rate of {label_sample_rate}Hz, resampling")
            label_wav = self.resample(label_wav, label_sample_rate)

        # All DataLoader tensors should have the same time length if batch size > 1
        split_start = fileinfo[1]
        input_wav = self.split(input_wav, split_start, same_time)
        label_wav = self.split(label_wav, split_start, same_time)

        # Return input and label waveforms
        return input_wav, label_wav

    def save_wav(self, tensor, output_path):
        if(tensor.dim() > 2):
            tensor = torch.squeeze(tensor)

        # Input tensor must be on CPU to be saved
        out = tensor.to("cpu")

        # Save the output as a WAV file
        torchaudio.save(output_path, out, self.sample_rate)

