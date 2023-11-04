import os
import math
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

class Functional():

    # Class data fields
    n_fft = 4096

    def __init__(self, sample_rate, max_time):
        self.sample_rate = sample_rate
        self.max_time = max_time

    def transform(self, tensor):
        spectrogram = T.Spectrogram(n_fft=self.n_fft, power=None)
        return spectrogram(tensor)

    def resample(self, tensor, old_sample_rate):
        resampler = T.Resample(old_sample_rate, self.sample_rate, dtype=waveform.dtype)
        return resampler(tensor)

    def pad(self, tensor):
        w_padding = self.max_time - tensor.shape[1]
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        return F.pad(tensor, (left_pad, right_pad))

    def get_filelist(self, comp_path):
        filelist = []
        for filename in os.listdir(comp_path):
            wav, sr = torchaudio.load(comp_path+filename)

            # Resample if not expected sample rate
            if(sr != self.sample_rate):
                self.resample(wav, sr, sample_rate)

            time = wav.shape[1]
            if(time <= self.max_time):
                filelist.append([filename, None])
            else:
                remainder = time % self.max_time
                if(remainder > (self.max_time/3)):
                    num_parts = int(math.ceil(time/self.max_time))
                else:
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

    def process_wavs(self, comp_path, uncomp_path, fileinfo, train):
        # Load wavs
        filename = fileinfo[0]
        comp_wav_path = comp_path + filename
        comp_wav, comp_sample_rate = torchaudio.load(comp_wav_path)
        uncomp_wav_path = uncomp_path + filename.replace(".flac-compressed-", "")
        uncomp_wav, uncomp_sample_rate = torchaudio.load(uncomp_wav_path)

        # Resample if not expected sample rate
        if(comp_sample_rate != self.sample_rate):
            print(f"\"{comp_wav_path}\" has sample rate of {comp_sample_rate}Hz, resampling")
            self.resample(comp_wav, comp_sample_rate)

        if(uncomp_sample_rate != self.sample_rate):
            print(f"\"{uncomp_wav_path}\" has sample rate of {uncomp_sample_rate}Hz, resampling")
            self.resample(uncomp_wav, uncomp_sample_rate)

        # All training tensors should have the same time length
        if(train):
            split_start = fileinfo[1]
            comp_wav = self.split(comp_wav, split_start)
            uncomp_wav = self.split(uncomp_wav, split_start)

        # Return input and label spectrograms
        return self.transform(comp_wav), self.transform(uncomp_wav)

    def spec_to_wav(self, tensor, output_path):

        inv_spec = T.InverseSpectrogram(n_fft=self.n_fft)
        tensor = inv_spec(tensor)

        # Save the output as a WAV file
        torchaudio.save(output_path, tensor, self.sample_rate)

