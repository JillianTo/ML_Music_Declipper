import os
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio.io import StreamReader
import torchaudio.transforms as T

def transform_tensor(tensor, sample_rate):
    transform = T.MelSpectrogram(sample_rate)
    return transform(tensor)

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

def upsample_tensor(tensor, input_size):
    transform = nn.Upsample(input_size)
    return transform(tensor)

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

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder layers
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Decoder layers
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(5, 5), stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
