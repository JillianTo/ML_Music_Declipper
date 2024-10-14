import sys
import pickle
from model import LSTMModel, TransformerModel
from functional import Functional

hparams = Functional.get_hparams(sys.argv)

# I:etialize variables for relevant setup parameters
n_fft = hparams["n_fft"]
hop_length = hparams["hop_length"]
top_db = hparams["top_db"]
first_out_channels = hparams["first_out_channels"]
stats_path = hparams["stats_path"]

# Load stats from file if available
with open(stats_path, 'rb') as f:
    db_stats = pickle.load(f)
    mean = db_stats[0]
    std = db_stats[1]
    print(f"Loaded stats from \"{stats_path}\" with mean {mean} and std "
          f"{std}")

# Initialize model
if hparams["transformer"]:
    model = TransformerModel(mean=mean, std=std, n_fft=n_fft, 
                             hop_length=hop_length, top_db=top_db, 
                             first_out_channels=first_out_channels, 
                             tf_layers=hparams["n_layers"])
    print("Using Transformer Encoder")
else:
    model = RNNModel(mean=mean, std=std, n_fft=n_fft, 
                     hop_length=hop_length, top_db=top_db, 
                     first_out_channels=first_out_channels,
                     lstm_layers=hparams["n_layers"])
    print("Using LSTM")

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters = {pytorch_total_params}")
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters = {pytorch_total_params}")
