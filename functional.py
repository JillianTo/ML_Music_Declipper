import os
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

class Functional():

    def __init__(self, sample_rate, max_time, device):
        self.sample_rate = sample_rate
        self.max_time = max_time
        self.device = device

    def wav_to_mel_db(self, tensor, n_fft=1024, n_mels=64, top_db=80):
        mel = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=n_fft, f_max=self.sample_rate>>1, n_mels=n_mels)
        mel = mel.to(self.device)
        amp_to_db = T.AmplitudeToDB("power", top_db=top_db)
        amp_to_db = amp_to_db.to(self.device)
        
        return amp_to_db(mel(tensor))

    def wav_to_pow_db(self, tensor, n_fft=4096, top_db=80):
        pow_spec = T.Spectrogram(n_fft=n_fft, power=2)
        pow_spec = pow_spec.to(self.device)
        amp_to_db = T.AmplitudeToDB("power", top_db=top_db)
        amp_to_db = amp_to_db.to(self.device)
        
        return amp_to_db(pow_spec(tensor))

    def wav_to_complex(self, tensor, n_fft=4096):
        complex_spec = T.Spectrogram(n_fft=n_fft, power=None)
        complex_spec = complex_spec.to(self.device)
        return  complex_spec(tensor)

    def wav_to_db(self, tensor, top_db=80):
        amp_to_db = T.AmplitudeToDB(stype="amplitude", top_db=top_db)
        amp_to_db = amp_to_db.to(self.device)
        return amp_to_db(tensor)

    def compute_std_mean(self, tensor, n_fft=4096):
        # Convert input to complex spectrogram
        x = self.wav_to_complex(tensor, n_fft=n_fft)

        # Calculate phase of complex spectrogram
        phase = torch.atan(x.imag/(x.real+1e-7))
        phase[x.real < 0] += 3.14159265358979323846264338

        # Calculate magnitude of complex spectrogram
        x = torch.sqrt(torch.pow(x.real, 2)+torch.pow(x.imag,2))
        amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=80)
        amp_to_db = amp_to_db.to(self.device)
        x = amp_to_db(x)

        # Calculate standard deviation and mean
        return torch.std_mean(x)

    def find_max_min(self, tensor, n_fft=4096):
        # Convert input to complex spectrogram
        x = self.wav_to_complex(tensor, n_fft=n_fft)

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

    def get_filelist(self, comp_path, same_time):
        filelist = []
        for filename in tqdm(os.listdir(comp_path)):
            # Determine if file will be split up
            time = torchaudio.info(comp_path+filename).num_frames
            if((not same_time) or time <= self.max_time):
                filelist.append([filename, None])
            else:
                #remainder = time % self.max_time
                #if(remainder > (self.max_time/1)):
                #    num_parts = int(math.ceil(time/self.max_time))
                #else:
                #    num_parts = int(time/self.max_time)
                num_parts = int(time/self.max_time)
                for i in range(0, num_parts+1):
                    filelist.append([filename, self.max_time*i])

        return filelist

    # Tensor should be output of torchaudio.load()
    def split(self, tensor, split_start):
        # pad wav if shorter than max_time
        if(split_start == None):
            return self.pad(tensor)
        # crop long tensor
        else:
            end_time = split_start + self.max_time
            # pad split tensor if it's length would be less than max_time
            if(end_time > tensor.shape[1]):
                return self.pad(tensor[:, split_start:])
            # return tensor starting at time split_start with length max_time
            else:
                return tensor[:, split_start:split_start+self.max_time]

    # Return input waveform and filename 
    def process_wav(self, comp_path, fileinfo, same_time):
        # Load waveforms
        filename = fileinfo[0]
        comp_wav_path = comp_path + filename
        comp_wav, comp_sample_rate = torchaudio.load(comp_wav_path)

        # Move waveforms to device
        comp_wav = comp_wav.to(self.device)

        # Resample if not expected sample rate
        if(comp_sample_rate != self.sample_rate):
            #print(f"\"{comp_wav_path}\" has sample rate of {comp_sample_rate}Hz, resampling")
            comp_wav = self.resample(comp_wav, comp_sample_rate)

        # All DataLoader tensors should have the same time length if batch size > 1
        if(same_time):
            split_start = fileinfo[1]
            comp_wav = self.split(comp_wav, split_start)

        # Return input waveforms and filename
        return comp_wav, filename
    
    # Return input and label waveforms
    def process_wavs(self, comp_path, uncomp_path, fileinfo, same_time):
        # Load waveforms
        filename = fileinfo[0]
        comp_wav_path = comp_path + filename
        comp_wav, comp_sample_rate = torchaudio.load(comp_wav_path)
        uncomp_wav_path = uncomp_path + filename.replace(".flac-compressed-", "")
        uncomp_wav, uncomp_sample_rate = torchaudio.load(uncomp_wav_path)

        # Move waveforms to device
        comp_wav = comp_wav.to(self.device)
        uncomp_wav = uncomp_wav.to(self.device)

        # Resample if not expected sample rate
        if(comp_sample_rate != self.sample_rate):
            #print(f"\"{comp_wav_path}\" has sample rate of {comp_sample_rate}Hz, resampling")
            comp_wav = self.resample(comp_wav, comp_sample_rate)

        if(uncomp_sample_rate != self.sample_rate):
            #print(f"\"{uncomp_wav_path}\" has sample rate of {uncomp_sample_rate}Hz, resampling")
            uncomp_wav = self.resample(uncomp_wav, uncomp_sample_rate)

        # All DataLoader tensors should have the same time length if batch size > 1
        if(same_time):
            split_start = fileinfo[1]
            comp_wav = self.split(comp_wav, split_start)
            uncomp_wav = self.split(uncomp_wav, split_start)

        # Return input and label waveforms
        return comp_wav, uncomp_wav

    def save_wav(self, tensor, output_path):
        if(tensor.dim() > 2):
            tensor = torch.squeeze(tensor)

        # Input tensor must be on CPU to be saved
        out = tensor.to("cpu")

        # Save the output as a WAV file
        torchaudio.save(output_path, out, self.sample_rate)

