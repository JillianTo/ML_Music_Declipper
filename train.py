import os
import sys
from aim import Run, Audio
import auraloss
import numpy as np
from tqdm import tqdm
import pickle
from autoclip.torch import QuantileClip

from audiodataset import AudioDataset
from autoencoder import AutoEncoder
from functional import Functional

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
print("Torch version: " + torch.__version__)

# Initialize a new run
run = Run(experiment="declipper")

# Log run parameters
run["hparams"] = {
    "learning_rate": 0.00001,
    "batch_size": 1,
    "test_batch_size": 1,
    "num_epochs": 100, 
    "expected_sample_rate": 44100,
    "uncompressed_data_path": "/mnt/MP600/data/uncomp/",
    "uncompressed_data_path": "/mnt/MP600/data/small/uncomp/",
    "compressed_data_path": "/mnt/MP600/data/comp/train/",
    "compressed_data_path": "/mnt/MP600/data/small/comp/train/",
    "test_compressed_data_path": "/mnt/MP600/data/comp/test/",
    "test_compressed_data_path": "/mnt/MP600/data/small/comp/test/",
    "results_path": "/mnt/PC801/declip/results/",
    "augmentation_labels": ["--01--","--10--","--11--","--20--"],
    #"max_time": 1880064,
    "max_time": 417792,
    #"test_max_time": 4751360,
    "test_max_time": 1638400,
    "num_workers": 0,
    "prefetch_factor": None,
    "pin_memory": False,
    "spectrogram_autoencoder": True,
    "preload_weights_path": None,
    #"preload_weights_path": "/mnt/PC801/declip/results/04-08/model08.pth",
    #"preload_weights_path": "/mnt/PC801/declip/results/model01.pth",
    "preload_optimizer_path": None,
    #"preload_optimizer_path": "/mnt/PC801/declip/results/04-08/optimizer08.pth",
    #"preload_optimizer_path": "/mnt/PC801/declip/results/optimizer01.pth",
    "preload_scheduler_path": None,
    #"preload_scheduler_path": "/mnt/PC801/declip/results/04-08/scheduler08.pth",
    #"preload_scheduler_path": "/mnt/PC801/declip/results/scheduler01.pth",
    #"n_ffts": [510, 2046, 8190],
    "n_ffts": [8190],
    #"hop_lengths": [64, 256, 1024],
    "hop_lengths": [256],
    "stats_n_fft": 8192,
    "stats_hop_length": 4096,
    "run_small_test": False,
    "eps": 0.00000001,
    "scheduler_state": 0,
    "scheduler_factors": [0.1, 0.1, 0.1],
    "scheduler_patiences": [0, 2, 4],
    "test_points": [0.125, 0.25,  0.5],
    "overwrite_preloaded_scheduler_values": False,
    "test_first": False,
    "autoclip": True,
    "multigpu": False, # TODO: Does not work yet
    "cuda_device": 0, # Choose which single GPU to use 
}

# Set up DDP process groups
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Destroy DDP process groups
def cleanup():
    dist.destroy_process_group()

# Process data and run training
def main(rank, world_size):
    # Save run parameters to variables for easier access
    learning_rate = run["hparams", "learning_rate"]
    sample_rate = run["hparams", "expected_sample_rate"]
    batch_size = run["hparams", "batch_size"]
    test_batch_size = run["hparams", "test_batch_size"]
    input_path = run["hparams", "compressed_data_path"]
    label_path = run["hparams", "uncompressed_data_path"]
    results_path = run["hparams", "results_path"]
    augmentation_lbls = run["hparams", "augmentation_labels"]
    max_time = run["hparams", "max_time"]
    test_max_time = run["hparams", "test_max_time"]
    num_workers = run["hparams", "num_workers"]
    prefetch_factor = run["hparams", "prefetch_factor"]
    pin_memory = run["hparams", "pin_memory"]
    preload_weights_path = run["hparams", "preload_weights_path"]
    preload_opt_path = run["hparams", "preload_optimizer_path"]
    preload_sch_path = run["hparams", "preload_scheduler_path"]
    n_ffts = run["hparams", "n_ffts"]
    hop_lengths = run["hparams", "hop_lengths"]
    stats_n_fft = run["hparams", "stats_n_fft"]
    stats_hop_length = run["hparams", "stats_hop_length"]
    run_small_test = run["hparams", "run_small_test"]
    sch_state = run["hparams", "scheduler_state"]
    factors = run["hparams", "scheduler_factors"]
    patiences = run["hparams", "scheduler_patiences"]
    overwrite_preload_sch = run["hparams",
                                "overwrite_preloaded_scheduler_values"]
    test_points = run["hparams", "test_points"]
    test_first = run["hparams", "test_first"]
    autoclip = run["hparams", "autoclip"]

    if(world_size != None):
        # Set up process groups
        setup(rank, world_size)

    # Load wavs into training data
    print("\nLoading data...")

    # Add inputs and labels to training dataset
    funct = Functional(sample_rate=sample_rate, max_time=max_time, device=rank, 
                       n_fft=stats_n_fft, hop_length=stats_hop_length, 
                       top_db=106, max_n_fft=max(n_ffts)+2, 
                       augmentation_lbls=augmentation_lbls)
    if(batch_size > 1):
        train_data = AudioDataset(funct, 
                                  input_path, 
                                  "filelist_train.txt", 
                                  label_path=label_path, 
                                  short_thres=2)
    else:
        train_data = AudioDataset(funct, 
                                  input_path, 
                                  "filelist_train.txt", 
                                  pad_short=False, 
                                  short_thres=2)
    print(f"Added {len(train_data)} file pairs to training data")

    # Add inputs and labels to test dataset
    test_funct = Functional(sample_rate=sample_rate, max_time=test_max_time, 
                            device=rank, n_fft=stats_n_fft, 
                            hop_length=stats_hop_length, top_db=106,
                            max_n_fft=max(n_ffts)+2, 
                            augmentation_lbls=augmentation_lbls)
    if(test_batch_size > 1):
        test_data = AudioDataset(test_funct, 
                                 run["hparams", "test_compressed_data_path"], 
                                 "filelist_test.txt", 
                                 label_path=label_path, 
                                 short_thres=2)
    else:
        test_data = AudioDataset(test_funct, 
                                 run["hparams", "test_compressed_data_path"], 
                                 "filelist_test.txt", 
                                 pad_short=False, 
                                 short_thres=2)
    print(f"Added {len(test_data)} file pairs to test data")

    # If using multi-GPU, set up sampleres and dataloaders accordingly
    if(world_size != None):
        # Create samplers
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, 
                                           rank=rank, shuffle=False, 
                                           drop_last=False)
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, 
                                          rank=rank, shuffle=False, 
                                          drop_last=False)
        print("Using distributed samplers for training/testing data")

        # Create data loader for training data 
        trainloader = DataLoader(train_data, batch_size=batch_size, 
                                 shuffle=False, num_workers=num_workers, 
                                 pin_memory=pin_memory, 
                                 prefetch_factor=prefetch_factor, 
                                 sampler=train_sampler)
        # Create data loader for test data 
        testloader = DataLoader(test_data, batch_size=test_batch_size, 
                                shuffle=False, num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor, 
                                sampler=test_sampler)
    # Else, set up dataloaders for single device
    else:
        # Create data loader for training data, enable shuffle if not using DDP
        trainloader = DataLoader(train_data, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers, 
                                 pin_memory=pin_memory, 
                                 prefetch_factor=prefetch_factor)
        # Create data loader for test data 
        testloader = DataLoader(test_data, batch_size=test_batch_size, 
                                shuffle=False, num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor)

    # Load stats from file if available
    stats_path = "db_stats.txt"
    if(os.path.isfile(stats_path)):
        with open(stats_path, 'rb') as f:
            db_stats = pickle.load(f)
            mean = db_stats[0]
            std = db_stats[1]
            print(f"Loaded stats from \"stats_path\" with mean {mean} and std "
                  f"{std}")
    # Else, calculate stats for training data
    else:
        print("Calculating mean and std...")
        mean, std = funct.db_stats(input_path)
        mean = mean.item()
        std = std.item()
        print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")
        with open(stats_path, 'wb') as f:
            pickle.dump([mean,std], f)
        print(f"Saved training set stats in \"{stats_path}\'")

    print("\nInitializing model, loss function, and optimizer...")

    # Initialize model
    if run["hparams", "spectrogram_autoencoder"]:
        model = AutoEncoder(mean=mean, std=std, n_ffts=n_ffts, 
                            hop_lengths=hop_lengths, 
                            sample_rate=sample_rate)
        print("Using spectrogram autoencoder")
    #else:
        #model = WavAutoEncoder(rank)
        #print("Using waveform autoencoder")
    model = model.to(rank)
    # If using multi-GPU, set up DDP
    if(world_size != None):
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=True)

    # Load model weights from path if given
    if preload_weights_path != None:
        model.load_state_dict(torch.load(preload_weights_path, 
                                         map_location=torch.device(rank)))
        print(f"Preloaded weights from \"{preload_weights_path}\"")

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # If enabled, add AutoClip to optimizer
    if autoclip:
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, 
                                              quantile=0.9, 
                                              history_length=1000)
        print("Using autoclip")
    # Load saved optimizer from path if given
    if preload_opt_path != None:
        optimizer.load_state_dict(torch.load(preload_opt_path))
        if(overwrite_preload_sch):
            if(sch_state > 0):
                for i, factor in enumerate(factors):
                    if(i < sch_state):
                        learning_rate = learning_rate*factor
                    else:
                        break
            optimizer.param_groups[0]['lr'] = learning_rate
        print(f"Preloaded optimizer from \"{preload_opt_path}\" "
              f"with learning rate {optimizer.param_groups[0]['lr']}")

    # Initialize scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                     factor=factors[sch_state], 
                                                     patience=patiences[sch_state],
                                                     eps=run["hparams", "eps"])
    # Load saved scheduler from path if given
    if preload_sch_path != None:
        scheduler.load_state_dict(torch.load(preload_sch_path))
        if(overwrite_preload_sch):
            scheduler.factor = factors[sch_state]
            scheduler.patience = patiences[sch_state]
        print(f"Preloaded scheduler from \"{preload_sch_path}\" with factor "
              f"{scheduler.factor} and patience {scheduler.patience}")
    
    # Do not run small test if learning rate will not be reduced by more than 
    # scheduler EPS
    step_sch = (optimizer.param_groups[0]['lr'] * (1-scheduler.factor) 
                > scheduler.eps)
    
    # Save when scheduler parameters will last be changed
    sch_change = len(factors) - 1

    # Define local function for saving model, optimizer, and scheduler states
    def saveStates(epoch):
        # Save model and optimizer states
        torch.save(model.state_dict(), results_path 
                   +f"model{(epoch+1):02d}.pth")
        torch.save(optimizer.state_dict(), results_path 
                   +f"optimizer{(epoch+1):02d}.pth")
        torch.save(scheduler.state_dict(), results_path 
                   +f"scheduler{(epoch+1):02d}.pth")

    # Initialize loss functions
    fft_sizes = [256, 512, 1024]
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, 
                                                   hop_sizes=[fft_size//4 for 
                                                              fft_size in 
                                                              fft_sizes], 
                                                   win_lengths=fft_sizes,
                                                   sample_rate=sample_rate, 
                                                   perceptual_weighting=True, 
                                                   scale=None)
    mrstft.to(rank)
    fft_sizes = [2048, 4096, 8192]
    mel_mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, 
                                                       hop_sizes=[fft_size//4 for 
                                                                  fft_size in 
                                                                  fft_sizes], 
                                                       win_lengths=fft_sizes, 
                                                       sample_rate=sample_rate, 
                                                       scale='mel', n_bins=256)
    mel_mrstft.to(rank)

    # Define function for calculating loss
    def calcLoss(pred, tgt):
        # Make sure input waveform is same length as label
        #if(tgt.shape[2] != pred.shape[2]):
        #    print(f"{fileinfo[0][0]} mismatched size, input is " 
        #          f"{pred.shape[2]} and label is "
        #          f"{tgt.shape[2]}")
        #    pred = funct.upsample(pred, tgt.shape[2])

        # Scale estimate to label mean
        pred = pred*(funct.mean(abs(tgt))/funct.mean(abs(pred)))

        # Calculate individual channel loss
        lr_loss = mrstft(pred, tgt)

        # Sum waveforms to one channel
        pred = funct.sum(pred)
        tgt = funct.sum(tgt)

        # Calculate sum loss
        sum_loss = mel_mrstft(pred, tgt)

        # Average sum and individual losses
        loss = (lr_loss*(2/3)) + (sum_loss/3)
       
        # Return loss
        return loss

    # Steps needed for data plotting in Aim
    train_step = 0
    small_test_step = 0
    test_step = 0

    # After how many train batches to run small test
    test_point = int(len(trainloader) * test_points[sch_state])

    # Do not run small test when about to finish training
    last_test_point = int(len(trainloader) * 0.96)

    # Number of batches to check in small test
    small_test_end = int(((len(trainloader)*(max_time/test_max_time))
                          + len(testloader))*0.01)
    
    # Get target waveform from fileinfo
    def getTgt(batch_size, fileinfo, funct):
        if(batch_size > 1):
            tgt = fileinfo
        else:
            fileinfo = (fileinfo[0][0], fileinfo[1].item())
            tgt = funct.process_wav(label_path, fileinfo, batch_size != 1, 
                                    is_input=False)  
            tgt = torch.unsqueeze(tgt, dim=0)
        return tgt

    # Training Loop
    print("\nStarting training loop...")
    for epoch in range(run["hparams", "num_epochs"]):
        # Clear CUDA cache
        #print(torch.cuda.max_memory_allocated(device=rank))
        #torch.cuda.empty_cache()

        # Check if test should be run first
        if(not (epoch < 1 and test_first)):
            # Initialize total and average training loss to zero
            tot_loss = 0
            avg_loss = 0
            # Initialize tqdm progress bar for training epoch
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), 
                        desc=f"Train Epoch {epoch+1}")

            # Training phase
            for i, (comp_wav, fileinfo) in pbar:
                # Set model to train state
                model.train()

                # Zero the gradients of all optimized variables. This is to 
                # ensure that we don't accumulate gradients from previous 
                # batches, as gradients are accumulated by default in PyTorch
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                comp_wav = model(comp_wav)

                # Get target waveform
                uncomp_wav = getTgt(batch_size, fileinfo, funct)
                
                # Calculate loss
                loss = calcLoss(comp_wav, uncomp_wav)

                # Force stop if loss is nan
                if(loss != loss):
                    sys.exit(f"Training loss is {loss} during \"{fileinfo[0]}"
                             f"\", force exiting")

                # Backward pass: compute the gradient of the loss with respect
                # to model parameters
                loss.backward()

                # Update the model's parameters using the optimizer's step 
                # method
                optimizer.step()

                # Add loss to total
                tot_loss += loss.item()

                # Calculate average loss so far
                avg_loss = tot_loss / (i+1)

                # Log loss to Aim
                run.track(loss.item(), name='loss', step=train_step, 
                          epoch=epoch+1, context={"subset":"train"})

                # Update tqdm progress bar 
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", 
                                  "Avg Loss": f"{avg_loss:.4f}"})

                # Increment step for next Aim log
                train_step = train_step + 1

                # If training has completed for test_point number of batches
                if(i%test_point == 0 and i < last_test_point and i > 0):
                    # Run small test if enabled
                    if(run_small_test):
                        # Run small test if learning rate can still be reduced
                        if(step_sch):
                            # Set model to validation state
                            model.eval()
                            # Don't update gradients during validation
                            with torch.no_grad():
                                # Initialize total and average small test loss 
                                # to zero
                                tot_test_loss = 0
                                avg_test_loss = 0
                                # Initialize tqdm progress bar for small test
                                test_pbar = tqdm(enumerate(testloader), 
                                                 total=small_test_end+1, 
                                                 desc=f"Small Test Epoch "
                                                 f"{scheduler.last_epoch+1}")
                                
                                # Small test phase
                                for i, (comp_wav, fileinfo) in test_pbar:
                                    # Stop small test after small_test_end 
                                    # batches
                                    if(i > small_test_end):
                                        break

                                    # Forward pass
                                    comp_wav = model(comp_wav)

                                    # Get target waveform
                                    uncomp_wav = getTgt(test_batch_size, 
                                                        fileinfo, test_funct)

                                    # Calculate loss
                                    loss = calcLoss(comp_wav, uncomp_wav)

                                    # Add loss to total
                                    tot_test_loss += loss.item()

                                    # Calculate current average loss
                                    avg_test_loss = tot_test_loss / (i+1)

                                    # Log loss to Aim
                                    run.track(loss.item(), name='loss', 
                                              step=small_test_step, 
                                              epoch=epoch+1, 
                                              context=
                                              {"subset":"small_test"})
                                
                                    # Increment step for next Aim log
                                    small_test_step += 1

                                    # Update tqdm progress bar
                                    test_pbar.set_postfix(
                                            {"Loss": f"{loss.item():.4f}", 
                                             "Avg Loss": f"{avg_test_loss:.4f}"})

                            # Send average loss from this small test to 
                            # scheduler
                            scheduler.step(avg_test_loss)

                            # Log average test loss and to Aim and CLI
                            run.track(avg_test_loss, name='avg_loss', 
                                      step=scheduler.last_epoch, epoch=epoch+1, 
                                      context={"subset":"small_test"})
                            
                            # Check if learning rate was changed
                            if(optimizer.param_groups[0]['lr'] 
                               < learning_rate):
                                # Save current learning rate
                                learning_rate = optimizer.param_groups[0]['lr']
                                # Change scheduler parameters after learning 
                                # rate reduction
                                if(sch_state < sch_change):
                                    sch_state += 1
                                    scheduler.factor = factors[sch_state]
                                    scheduler.patience = patiences[sch_state]
                                    test_point = int(len(train_data) 
                                                     * test_points[sch_state])
                                # If learning rate will not be reduced by more 
                                # than scheduler EPS, then do not run small 
                                # test anymore
                                if(not (learning_rate * (1-scheduler.factor) 
                                        > scheduler.eps)):
                                    step_sch = False
            
                    # Save current model, optimizer, and scheduler states
                    saveStates(epoch)

            # Log average training loss to Aim
            run.track(avg_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, 
                      context={"subset":"train"})
            
            # Clear CUDA cache
            print(torch.cuda.max_memory_allocated(device=rank))
            torch.cuda.empty_cache()

        # Set model to validation state
        model.eval()
        # Don't update gradients during validation
        with torch.no_grad():
            # Initialize total and average testing loss to zero
            tot_loss = 0
            avg_loss = 0
            # Initialize tqdm progress bar for testing epoch
            pbar = tqdm(enumerate(testloader), total=len(testloader), 
                        desc=f"Test Epoch {epoch+1}")

            # Testing phase
            for i, (comp_wav, fileinfo) in pbar:
                # Forward pass
                comp_wav = model(comp_wav)

                # Get target waveform
                uncomp_wav = getTgt(test_batch_size, fileinfo, test_funct)

                # Calculate loss
                loss = calcLoss(comp_wav, uncomp_wav)

                # Add loss to total
                tot_loss += loss.item()

                # Calculate average loss so far
                avg_loss = tot_loss / (i+1)

                # Log loss to Aim
                run.track(loss.item(), name='loss', step=test_step, 
                          epoch=epoch+1, context={"subset":"test"})
                if(not (i > small_test_end) and run_small_test):
                    run.track(loss.item(), name='loss', step=small_test_step, 
                              epoch=epoch+1, context={"subset":"small_test"})

                # Increment step for next Aim log
                test_step += 1

                # Update tqdm progress bar 
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", 
                                  "Avg Loss": f"{avg_loss:.4f}"}) 

                # Step scheduler after small_test_end batches
                if(i == small_test_end and run_small_test and step_sch):
                    # Send average small test loss to scheduler
                    scheduler.step(avg_loss)
                    # Log average loss to Aim
                    run.track(avg_loss, name='avg_loss', 
                              step=scheduler.last_epoch, epoch=epoch+1, 
                              context={"subset":"small_test"})

                torch.cuda.empty_cache()

        # Step scheduler at the end of full test phase if not set to step in
        # small tests
        if(not run_small_test and step_sch):
            # Send average testing loss to scheduler
            scheduler.step(avg_loss)

            # Change out scheduler parameters if necessary
            if(optimizer.param_groups[0]['lr'] < learning_rate):
                # Save current learning rate
                learning_rate = optimizer.param_groups[0]['lr']
                # Change scheduler parameters after learning rate reduction
                if(sch_state < sch_change):
                    sch_state += 1
                    scheduler.factor = factors[sch_state]
                    scheduler.patience = patiences[sch_state]
                    test_point = int(len(train_data) 
                                     * test_points[sch_state])
                # If learning rate will not be reduced by more than 
                # scheduler EPS, then do not run small test anymore
                if(not (learning_rate * (1-scheduler.factor) 
                        > scheduler.eps)):
                    step_sch = False


        # Check if test was run without training  
        if(test_first and epoch < 1):
            # Decrement epoch number so training starts at first epoch
            epoch -= 1
        else:
            # Save current model, optimizer, and scheduler states
            saveStates(epoch)

        # Log average test loss to Aim
        run.track(avg_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, 
                  context={"subset":"test"})
        
    # If multi-GPU was used, cleanup before exiting 
    if(world_size != None):
        cleanup()

    print("Finished Training")

if __name__ == "__main__":
    # Get CPU, GPU, or MPS device for training
    cuda_device = f"cuda:{run['hparams', 'cuda_device']}"
    device = (
            cuda_device
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    print(f"Using {device} device")
   
    # Check whether to use multi-GPU processing 
    if(run["hparams", "multigpu"] and device == cuda_device):
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPUs")
        mp.spawn(main, args=[n_gpus], nprocs=n_gpus, join=True)
    # Else, run on one device
    else:
        main(device, None)

