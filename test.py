import os
import sys
from tqdm import tqdm
import pickle
import datetime

from audiodataset import AudioDataset
from functional import Functional
from model import LSTMModel, TransformerModel

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Set up DDP process groups
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size, 
                            timeout=datetime.timedelta(seconds=36000))

# Destroy DDP process groups
def cleanup():
    dist.destroy_process_group()

# Process data and run training
def main(rank, world_size):
    # If using multi-GPU, set up DDP process groups
    if(world_size != None):
        # Set up process groups
        setup(rank, world_size)

    # Load hyperparameters
    hparams = Functional.get_hparams(sys.argv)

    # Save individual hyperparameters to variables for easier access
    sample_rate = hparams["expected_sample_rate"]
    batch_size = hparams["test_batch_size"]
    stats_path = hparams["stats_path"]
    filelist_path = hparams["test_filelist_path"]
    label_path = hparams["test_label_data_path"]
    input_path = hparams["input_data_path"]
    augmentation_lbls = hparams["augmentation_labels"]
    max_time = hparams["max_time"]
    short_thres = hparams["short_threshold"]
    overlap_factor = hparams["overlap_factor"]
    num_workers = hparams["num_workers"]
    pin_memory = hparams["pin_memory"]
    prefetch_factor = hparams["prefetch_factor"]
    first_out_channels = hparams["first_out_channels"]
    n_layers = hparams["n_layers"]
    preload_weights_path = hparams["preload_weights_path"]
    n_fft = hparams["n_fft"]
    hop_length = hparams["hop_length"]
    loss_n_ffts = hparams["loss_n_ffts"]
    n_mels = hparams["n_mels"]
    top_db = hparams["top_db"]
    use_amp = hparams["use_amp"]

    # Add inputs and labels to test dataset
    funct = Functional(max_time=max_time, device=rank, n_fft=n_fft, 
                       hop_length=hop_length, 
                       augmentation_lbls=augmentation_lbls)

    test_data = AudioDataset(funct, input_path, label_path, filelist_path, 
                             sample_rate=sample_rate, pad_short=False, 
                             short_thres=short_thres,
                             overlap_factor=overlap_factor,
                             rtn_lbl_wav=batch_size>1)
    print(f"Added {len(train_data)} file pairs to testing data")

    # If using multi-GPU, set up samplers and dataloaders accordingly
    if(world_size != None):
        # Create samplers
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, 
                                          rank=rank, shuffle=False, 
                                          drop_last=False)
        print("Using distributed samplers for testing data")

        # Create data loader for testing data 
        testloader = DataLoader(test_data, batch_size=batch_size, 
                                shuffle=False, sampler=test_sampler, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor)
    # Else, set up dataloaders for single device
    else:
        # Create data loader for validation data 
        testloader = DataLoader(test_data, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor) 

    # Load stats from file
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
                                 tf_layers=n_layers)
        print("Using Transformer Encoder")
    else:
        model = RNNModel(mean=mean, std=std, n_fft=n_fft, 
                         hop_length=hop_length, top_db=top_db, 
                         first_out_channels=first_out_channels,
                         lstm_layers=n_layers)
        print("Using LSTM")
    #model = torch.compile(model, mode='default')

    # Load model weights from path if given
    if preload_weights_path != None:
        model.load_state_dict(torch.load(preload_weights_path, 
                                         map_location="cpu",
                                         weights_only=True))
        print(f"Preloaded weights from \"{preload_weights_path}\"")
    else:
        sys.exit("Model checkpoint not specified, force quitting")

    # Send model to device
    model = model.to(rank)

    # If using multi-GPU, set up DDP
    if world_size != None:
        model = DDP(model, device_ids=[rank], output_device=rank)
        print("Using DDP")

    # Set model to validation state
    model.eval()
    # Don't update gradients during test
    with torch.no_grad():
        # Initialize total and average test loss to zero
        tot_loss = 0
        avg_loss = 0
        # Initialize tqdm progress bar for test epoch
        pbar = tqdm(enumerate(testloader), total=len(testloader), 
                    desc=f"Test Epoch {epoch+1}")

        # Validation phase
        for i, (comp_wav, fileinfo) in pbar:
            with torch.autocast(device_type="cuda", enabled=use_amp, 
                                dtype=torch.float16):
                # Forward pass
                comp_wav = comp_wav.to(rank)
                comp_wav = model(comp_wav)

                # Get target waveform
                uncomp_wav = getTgt(batch_size, fileinfo, funct, 
                                    label_path)
                uncomp_wav = uncomp_wav.to(rank)

                # Calculate loss
                loss = Functional.calc_loss(comp_wav, uncomp_wav, 
                                            sample_rate, 
                                            n_ffts=loss_n_ffts, 
                                            n_mels=n_mels, top_db=top_db) 

            # Add loss to total
            loss_item = loss.item()
            tot_loss += loss_item

            # Calculate average loss so far
            avg_loss = tot_loss / (i+1)

            # Update tqdm progress bar 
            pbar.set_postfix({"Loss": f"{loss_item:.4f}", 
                              "Avg Loss": f"{avg_loss:.4f}"}) 

# Process data and run training
if __name__ == "__main__":
    # Print Torch version
    print("Torch version: " + torch.__version__)
   
    # Load hyperparameters
    hparams = Functional.get_hparams(sys.argv)

    cuda_device = f"cuda:{hparams['cuda_device']}"
    device = (
            cuda_device
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    print(f"Using {device} device")
   
    # Check whether to use multi-GPU processing 
    if(hparams["multigpu"] and device == cuda_device):
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPUs")
        mp.spawn(main, args=[n_gpus], nprocs=n_gpus, join=True)
    # Else, run on one device
    else:
        main(device, None)
