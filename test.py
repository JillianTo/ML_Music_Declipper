import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.init import constant
from torch.nn.parameter import Parameter
import torchaudio.functional as F
import torchaudio.transforms as T
import torchaudio
from functional import Functional

wav, sr = torchaudio.load("/mnt/PC801/declip/other/tdoyam.wav")

n_fft = [256]
fbin_idxs = [(0, 129)]
hop_length = 32
    
funct = Functional(device="cpu", n_fft=n_fft, hop_length=hop_length, fbin_idxs=fbin_idxs)

# If no batch dimension, add one
if wav.ndim < 3:
    wav = wav.unsqueeze(0)
    added_batch_ch = True
                
wav, phases = funct.wav_to_mr_spec(wav)
wav = funct.mr_spec_to_wav(wav, phases)
if added_batch_ch:
    wav = wav.squeeze(0)
torchaudio.save("/mnt/PC801/declip/edit.wav", wav, sr)
