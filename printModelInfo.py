import sys
import pickle
import torch
from functional import Functional
from autoencoder import AutoEncoder

preload_weights_path = "/mnt/PC801/declip/results/model01.pth"
stats_path = "db_stats.txt"

device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
device = "cpu"

with open(stats_path, 'rb') as f:
    db_stats = pickle.load(f)
    mean = db_stats[0]
    std = db_stats[1]

hparams = Functional.get_hparams(sys.argv)
n_fft = hparams["n_fft"]
hop_length = hparams["hop_length"]
sample_rate = hparams["expected_sample_rate"]

funct = Functional(sys.argv)
model = AutoEncoder(mean=mean, std=std, n_fft=n_fft, hop_length=hop_length, 
                    sample_rate=sample_rate)
model.load_state_dict(torch.load(preload_weights_path, 
                                 map_location=torch.device(device)))

i = 0
num_params = 0
for param in model.parameters():
    if param.requires_grad:
        i = i + 1
        curr_params = param.numel()
        num_params = num_params + curr_params
        print(f"Layer {i} has {curr_params} parameters")

print(f"Model has {num_params} total parameters")
    
