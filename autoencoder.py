import os
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.io import StreamReader
import torchaudio.transforms as T

n_fft = 4096
enc_channels = 16

def transform_tensor(tensor, sample_rate):
    transform = T.Spectrogram(n_fft=n_fft, power=None)
    tensor = transform(tensor)
    return tensor

def pad_tensor(tensor, max_w):
    w_padding = max_w - tensor.shape[1]
    left_pad = w_padding // 2
    right_pad = w_padding - left_pad
    return F.pad(tensor, (left_pad, right_pad))

# Tensor should be output of torchaudio.load()
def split_tensor(tensor, max_w):
    tensor_w = tensor.shape[1]
    threshold = max_w/3 # minimum size tensor to add to list
    # pad compressed wav if short tensor
    if(tensor_w < max_w and tensor_w > threshold):
        tensor_parts = [pad_tensor(tensor, max_w)]
    # crop long tensor
    elif(tensor_w > max_w):
        pos = 0
        end_pos = max_w
        tensor_parts = []
        while(end_pos < tensor_w):
            tensor_parts.append(tensor[:, pos:end_pos])
            pos += max_w
            end_pos += max_w
        if(tensor_w-pos > threshold):
            tensor_parts.append(pad_tensor(tensor[:, pos:], max_w))
    else:
        tensor_parts = [tensor]
    return tensor_parts

# Tensor should be output of torchaudio.load()
def resample_tensor(tensor, old_sample_rate, new_sample_rate):
    resampler = T.Resample(old_sample_rate, new_sample_rate, dtype=waveform.dtype)
    return resampler(tensor)

## Credit: https://github.com/pytorch/pytorch/issues/52983
def combine_complex(r, i):
    return torch.view_as_complex(torch.cat([r.unsqueeze(-1), i.unsqueeze(-1)], dim=-1))

def upsample_tensor(tensor, input_size):
    transform = nn.Upsample(input_size)
    real = transform(tensor.real)
    imag = transform(tensor.imag)
    return combine_complex(real, imag)

def process_wavs(comp_folder, uncomp_folder, max_w, sample_rate, inputs, labels, train, device):
    for filename in os.listdir(comp_folder):
        # Load wavs
        comp_wav_path = comp_folder + filename
        comp_wav, comp_sample_rate = torchaudio.load(comp_wav_path)
        uncomp_wav_path = uncomp_folder + filename.replace(".flac-compressed-", "")
        uncomp_wav, uncomp_sample_rate = torchaudio.load(uncomp_wav_path)

        # Resample if not expected sample rate
        if(comp_sample_rate != sample_rate):
            print(f"\"{comp_wav_path}\" has sample rate of {comp_sample_rate}Hz, resampling")
            resample_tensor(comp_wav, comp_sample_rate, sample_rate)

        if(uncomp_sample_rate != sample_rate):
            print(f"\"{uncomp_wav_path}\" has sample rate of {uncomp_sample_rate}Hz, resampling")
            resample_tensor(uncomp_wav, uncomp_sample_rate, sample_rate)

        # All training tensors should have the same time length
        if(train):
            input = split_tensor(comp_wav, max_w)
            label = split_tensor(uncomp_wav, max_w)
        # Don't need to modify time length of test tensors
        else:
            input = [comp_wav]
            label = [uncomp_wav]

        # transform wavs to Mel spectrograms
        for i, wav in enumerate(input):
            input[i] = transform_tensor(wav, sample_rate)
            input[i] = input[i].to(device)
        for i, wav in enumerate(label):
            label[i] = transform_tensor(wav, sample_rate)
            label[i] = label[i].to(device)

        # Add wavs to lists
        inputs.extend(input)
        labels.extend(label)

def spec_to_wav(tensor, sample_rate, device):
    transform = T.InverseSpectrogram(n_fft=n_fft)

    tensor = transform(tensor)

    # Set the desired output path and filename
    output_path = "/home/jto/Documents/AIDeclip/AIDeclipper/"
    filename = "output_audio.wav"

    # Save the output as a WAV file
    torchaudio.save(output_path+filename, tensor, sample_rate)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=enc_channels, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU()
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=enc_channels, out_channels=enc_channels*2, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )

        # First decoder layer 
        self.fc1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=enc_channels*2, out_channels=enc_channels, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU()
        )

        # Second fully-connected layer
        self.fc2 = nn.ConvTranspose2d(in_channels=enc_channels, out_channels=2, kernel_size=(5, 5), stride=(2, 2))

    def forward(self, x):
        real = self.enc1(x.real)
        imag = self.enc1(x.imag)
        real = self.enc2(real)
        imag = self.enc2(imag)
        real = self.fc1(real)
        imag = self.fc1(imag)
        real = self.fc2(real)
        imag = self.fc2(imag)
        x = combine_complex(real, imag)
        return x
