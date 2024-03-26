import os
import sys
from aim import Run, Audio
import auraloss
import numpy as np
from tqdm import tqdm
import pickle

from audiodataset import AudioDataset
from autoencoder import SpecAutoEncoder, WavAutoEncoder
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
    #"learning_rate": 0.00005,
    "learning_rate": 0.00001,
    "batch_size": 1,
    "test_batch_size": 1,
    "num_epochs": 100, 
    "expected_sample_rate": 44100,
    "uncompressed_data_path": "/mnt/MP600/data/uncomp/",
    #"uncompressed_data_path": "/mnt/MP600/data/small/uncomp/",
    "compressed_data_path": "/mnt/MP600/data/comp/train/",
    #"compressed_data_path": "/mnt/MP600/data/small/comp/train/",
    "test_compressed_data_path": "/mnt/MP600/data/comp/test/",
    #"test_compressed_data_path": "/mnt/MP600/data/small/comp/test/",
    "results_path": "/mnt/PC801/declip/results/",
    "max_time": 1907500,
    "test_max_time": 4250000,
    "num_workers": 0,
    "prefetch_factor": None,
    "pin_memory": False,
    "spectrogram_autoencoder": True,
    #"preload_weights_path": "/mnt/PC801/declip/results/model01.pth",
    "preload_weights_path": None,
    #"preload_optimizer_path": "/mnt/PC801/declip/results/optimizer01.pth",
    "preload_optimizer_path": None,
    #"preload_scheduler_path": "/mnt/PC801/declip/results/scheduler01.pth",
    "preload_scheduler_path": None,
    "run_small_test": False,
    #"eps": 0.0000000001,
    "eps": 0.00000001,
    "scheduler_state": 0,
    #"scheduler_factors": [(1/3), 0.1],
    #"scheduler_factors": [0.5, 0.4, 0.1],
    "scheduler_factors": [0.1, 0.1, 0.1],
    #"scheduler_patiences": [2, 4],
    #"scheduler_patiences": [2, 3, 4],
    "scheduler_patiences": [0, 1, 2],
    "test_points": [0.05, 0.125,  0.25],
    "overwrite_preloaded_scheduler_values": False,
    "test_first": False,
    "autoclip": True,
    "multigpu": False, # Does not work yet
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
    label_path = run["hparams", "uncompressed_data_path"]
    results_path = run["hparams", "results_path"]
    max_time = run["hparams", "max_time"]
    test_max_time = run["hparams", "test_max_time"]
    num_workers = run["hparams", "num_workers"]
    prefetch_factor = run["hparams", "prefetch_factor"]
    pin_memory = run["hparams", "pin_memory"]
    preload_weights_path = run["hparams", "preload_weights_path"]
    preload_opt_path = run["hparams", "preload_optimizer_path"]
    preload_sch_path = run["hparams", "preload_scheduler_path"]
    run_small_test = run["hparams", "run_small_test"]
    sch_state = run["hparams", "scheduler_state"]
    factors = run["hparams", "scheduler_factors"]
    patiences = run["hparams", "scheduler_patiences"]
    overwrite_preload_sch = run["hparams", "overwrite_preloaded_scheduler_values"]
    test_points = run["hparams", "test_points"]
    test_first = run["hparams", "test_first"]
    autoclip = run["hparams", "autoclip"]

    if(world_size != None):
        # Set up process groups
        setup(rank, world_size)

    # Load wavs into training data
    print("\nLoading data...")

    # Add inputs and labels to training dataset
    funct = Functional(sample_rate, max_time, rank)
    if(batch_size > 1):
        train_data = AudioDataset(funct, 
                                     run["hparams", "compressed_data_path"], 
                                     "filelist_train.txt", 
                                     label_path=label_path, 
                                     pad_thres=2)
    else:
        train_data = AudioDataset(funct, 
                                     run["hparams", "compressed_data_path"], 
                                     #None,
                                     "filelist_train.txt", 
                                     same_time=False, 
                                     pad_thres=2)
    print(f"Added {len(train_data)} file pairs to training data")

    # Add inputs and labels to test dataset
    test_funct = Functional(sample_rate, test_max_time, rank)
    if(test_batch_size > 1):
        test_data = AudioDataset(test_funct, 
                                 run["hparams", "test_compressed_data_path"], 
                                 "filelist_test.txt", 
                                 label_path=label_path, 
                                 pad_thres=2)
    else:
        test_data = AudioDataset(test_funct, 
                                 run["hparams", "test_compressed_data_path"], 
                                 #None,
                                 "filelist_test.txt", 
                                 same_time=False, 
                                 pad_thres=2)
    print(f"Added {len(test_data)} file pairs to test data")

    if(world_size != None):
        # Create samplers
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        print("Using distributed samplers for training/testing data")

        # Create data loader for training data 
        trainloader = DataLoader(train_data, batch_size=batch_size, 
                                 shuffle=False, num_workers=num_workers, 
                                 pin_memory=pin_memory, 
                                 prefetch_factor=prefetch_factor, sampler=train_sampler)
        # Create data loader for test data 
        testloader = DataLoader(test_data, batch_size=test_batch_size, 
                                shuffle=False, num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor, sampler=test_sampler)

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

    # Calculate mean and std for training set if not given
    stats_path = "db_stats.txt"
    if(os.path.isfile(stats_path)):
        with open(stats_path, 'rb') as f:
            db_stats = pickle.load(f)
            mean = db_stats[0].to(device)
            std = db_stats[1].to(device)
    else:
        print("Calculating mean and std...")
        mean = 0
        std = 0
        total = len(trainloader)
        pbar = tqdm(enumerate(trainloader), total=total, 
                    desc=f"Train Stats")

        for i, (comp_wav, _) in pbar:
            curr_std, curr_mean = funct.db_stats(comp_wav)
            mean = mean + curr_mean
            std = std + curr_std

            pbar.set_postfix({"Mean": f"{curr_mean:.4f}", 
                              "Std Dev": f"{curr_std:.4f}"})

        mean = mean / total
        std = std / total
        print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")
        with open(stats_path, 'wb') as f:
            pickle.dump([mean,std], f)
        print(f"Saved training dataset stats in \"{stats_path}\'")

    print("\nInitializing model, loss function, and optimizer...")

    # Initialize model
    if run["hparams", "spectrogram_autoencoder"]:
        model = SpecAutoEncoder(mean, std, n_fft=4096, hop_length=1024)
        print("Using spectrogram autoencoder")
    else:
        model = WavAutoEncoder(rank)
        print("Using waveform autoencoder")
    model = model.to(rank)

    if(world_size != None):
        model = DDP(model, device_ids=[rank],output_device=rank,find_unused_parameters=True)

    if preload_weights_path != None:
        model.load_state_dict(torch.load(preload_weights_path, 
                                         map_location=torch.device(rank)))
        print(f"Preloaded weights from \"{preload_weights_path}\"")

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
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
                                                     eps=run["hparams", "eps"],
                                                     verbose=True)
    if preload_sch_path != None:
        scheduler.load_state_dict(torch.load(preload_sch_path))
        if(overwrite_preload_sch):
            scheduler.factor = factors[sch_state]
            scheduler.patience = patiences[sch_state]
        print(f"Preloaded scheduler from \"{preload_sch_path}\" with factor "
              f"{scheduler.factor} and patience {scheduler.patience}")

    # Initialize loss function
    fft_sizes = [2048, 4096, 8192]
    #fft_sizes = [4096, 8192, 16384]
    #fft_sizes = [8192, 16384, 32768]
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, 
                                                   hop_sizes=[fft_size//8 for 
                                                              fft_size in 
                                                              fft_sizes], 
                                                   win_lengths=fft_sizes,
                                                   sample_rate=sample_rate, 
                                                   perceptual_weighting=True, 
                                                   scale=None, n_bins=64)
    mrstft.to(rank)
    mel_mrstft = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, 
                                                       hop_sizes=[fft_size//8 for 
                                                                  fft_size in 
                                                                  fft_sizes], 
                                                       win_lengths=fft_sizes, 
                                                       sample_rate=sample_rate, 
                                                       scale='mel', n_bins=64)
    mel_mrstft.to(rank)
    #sisdr = auraloss.time.SISDRLoss()

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

    # Do not run small test if learning rate will not be reduced by more than 
    # scheduler EPS
    step_sch = (optimizer.param_groups[0]['lr'] * (1-scheduler.factor) 
                > scheduler.eps)

    # Last time scheduler parameters will be changed
    sch_change = len(factors) - 1

    # Get normalized gradient history
    if(autoclip):
        grad_history = []

    # Training Loop
    print("\nStarting training loop...")
    for epoch in range(run["hparams", "num_epochs"]):
        # Training phase
        if(not (epoch < 1 and test_first)):
            tot_loss = 0
            #tot_sdr = 0
            avg_loss = 0
            #avg_sdr = 0
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), 
                        desc=f"Train Epoch {epoch+1}")
            for i, (comp_wav, fileinfo) in pbar:
                # Set model to train state
                model.train()

                # Zero the gradients of all optimized variables. This is to ensure 
                # that we don't accumulate gradients from previous batches, as 
                # gradients are accumulated by default in PyTorch
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                comp_wav = model(comp_wav)

                # Get uncompressed waveform
                if(batch_size > 1):
                    uncomp_wav = fileinfo
                else:
                    fileinfo = (fileinfo[0][0], fileinfo[1].item())
                    uncomp_wav = funct.process_wav(label_path, fileinfo, 
                                                   batch_size != 1, is_input=False)
                    uncomp_wav = torch.unsqueeze(uncomp_wav, dim=0)
                    # Make sure input waveform is same length as label
                    if(uncomp_wav.shape[2] != comp_wav.shape[2]):
                        print(f"{fileinfo[0]} mismatched size, input is " 
                              f"{comp_wav.shape[2]} and label is "
                              f"{uncomp_wav.shape[2]}")
                        comp_wav = funct.upsample(comp_wav, uncomp_wav.shape[2])

                # Calculate individual channel loss
                loss = mrstft(comp_wav, uncomp_wav)

                # Calculate SI-SDR
                #sdr = sisdr(comp_wav, uncomp_wav)

                # Sum waveforms to one channel
                comp_wav = funct.sum(comp_wav)
                uncomp_wav = funct.sum(uncomp_wav)

                # Calculate sum loss
                sum_loss = mel_mrstft(comp_wav, uncomp_wav)

                # Average sum and individual losses
                loss = (loss*(2/3)) + (sum_loss/3)

                # Force stop if loss is nan
                if(loss != loss):
                    sys.exit(f"Training loss is {loss}, force exiting")

                # Backward pass: compute the gradient of the loss with respect to 
                # model parameters
                #loss.backward(retain_graph=True)
                loss.backward()
                #sdr.backward()

                # Adapted from 
                # https://github.com/pseeth/autoclip/blob/master/autoclip.py, 
                # removes Pytorch Ignite dependency
                # Clip gradients
                if(autoclip):
                    tot_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            tot_norm += p.grad.data.norm(2).item() ** 2
                    tot_norm = tot_norm ** (1. / 2)
                    grad_history.append(tot_norm)
                    clip_val = np.percentile(grad_history, 10)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_val)

                # Update the model's parameters using the optimizer's step method
                optimizer.step()

                # Explicitly delete tensors so they don't stay in memory
                del comp_wav
                del uncomp_wav

                # Add loss and SDR to totals
                tot_loss += loss.item()
                #tot_sdr -= sdr.item()

                # Calculate average loss and SDR so far
                avg_loss = tot_loss / (i+1)
                #avg_sdr = tot_sdr / (i+1)

                # Log loss to Aim
                run.track(loss.item(), name='loss', step=train_step, epoch=epoch+1, 
                          context={"subset":"train"})
                #run.track(-sdr.item(), name='sdr', step=train_step, epoch=epoch+1, 
                #          context={"subset":"train"})

                # Update tqdm progress bar with fixed number of decimals for loss
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", 
                                  "Avg Loss": f"{avg_loss:.4f}"})
                #                  "SDR": f"{-sdr.item():.4f}", 
                #                  "Avg SDR": f"{avg_sdr:.4f}"})

                # Increment step for next Aim log
                train_step = train_step + 1

                # If training has completed for test_point number of batches
                if(i%test_point == 0 and i < last_test_point and i > 0):
                    if(run_small_test):
                        if(step_sch):
                            # Small testing phase
                            model.eval()
                            with torch.no_grad():
                                tot_test_loss = 0
                                #tot_test_sdr = 0
                                avg_test_loss = 0
                                #avg_test_sdr = 0
                                test_pbar = tqdm(enumerate(testloader), 
                                                 total=small_test_end+1, 
                                                 desc=f"Small Test Epoch "
                                                 f"{scheduler.last_epoch+1}")
                                for i, (comp_wav, fileinfo) in test_pbar:
                                    # Stop small test after small_test_end batches
                                    if(i > small_test_end):
                                        break

                                    # Forward pass
                                    comp_wav = model(comp_wav)

                                    # Get uncompressed waveform
                                    if(test_batch_size > 1):
                                        uncomp_wav = fileinfo
                                    else:
                                        fileinfo = (fileinfo[0][0], fileinfo[1].item())
                                        uncomp_wav = test_funct.process_wav(
                                                label_path, fileinfo, 
                                                test_batch_size != 1, 
                                                is_input=False)  
                                        uncomp_wav = torch.unsqueeze(uncomp_wav, 
                                                                     dim=0)
                                        # Make sure input waveform is same 
                                        # length as label
                                        if(uncomp_wav.shape[2] != 
                                           comp_wav.shape[2]):
                                            print(f"{fileinfo[0][0]} mismatched size, "
                                                  f"input is {comp_wav.shape[2]} and "
                                                  f"label is {uncomp_wav.shape[2]}")
                                            comp_wav = funct.upsample(comp_wav, 
                                                    uncomp_wav.shape[2])

                                    # Calculate individual channel loss
                                    loss = mrstft(comp_wav, uncomp_wav)

                                    # Calculate SDR 
                                    #sdr = sisdr(comp_wav, uncomp_wav)

                                    # Sum waveforms to one channel
                                    comp_wav = funct.sum(comp_wav)
                                    uncomp_wav = funct.sum(uncomp_wav)

                                    # Calculate sum loss
                                    sum_loss = mel_mrstft(comp_wav, uncomp_wav)

                                    # Average losses
                                    loss = (loss*(2/3)) + (sum_loss/3)

                                    # Explicitly delete tensors so theydon't stay in 
                                    # memory
                                    del comp_wav
                                    del uncomp_wav

                                    # Add loss and SDR to totals
                                    tot_test_loss += loss.item()
                                    #tot_test_sdr -= sdr.item()

                                    # Calculate current average loss and SDR
                                    avg_test_loss = tot_test_loss / (i+1)
                                    #avg_test_sdr = tot_test_sdr / (i+1)

                                    # Log loss to Aim
                                    run.track(loss.item(), name='loss', 
                                              step=small_test_step, 
                                              epoch=epoch+1, 
                                              context=
                                              {"subset":"small_test"})
                                    #run.track(-sdr.item(), name='sdr', 
                                    #          step=small_test_step, 
                                    #          epoch=epoch+1, 
                                    #          context=
                                    #          {"subset":"small_test"})
                                
                                    # Increment step for next Aim log
                                    small_test_step += 1

                                    # Update tqdm progress bar with fixed 
                                    # number of decimals for loss
                                    test_pbar.set_postfix(
                                            {"Loss": f"{loss.item():.4f}", 
                                    #         "SDR": f"{-sdr.item():.4f}",
                                             "Avg Loss": f"{avg_test_loss:.4f}"})
                                    #         "Avg SDR": f"{avg_test_sdr:.4f}"})

                            # Send average training loss to scheduler
                            scheduler.step(avg_test_loss)

                            # Log average test loss and SDR to Aim and 
                            # terminal
                            run.track(avg_test_loss, name='avg_loss', 
                                      step=scheduler.last_epoch, epoch=epoch+1, 
                                      context={"subset":"small_test"})
                            #run.track(avg_test_sdr, name='avg_sdr', 
                            #          step=scheduler.last_epoch, epoch=epoch+1, 
                            #          context={"subset":"small_test"})

                            # Check if learning rate was changed
                            if(optimizer.param_groups[0]['lr'] < learning_rate):
                                # Save current learning rate
                                learning_rate = optimizer.param_groups[0]['lr']
                                # Change scheduler parameters after learning rate 
                                # reduction
                                if(sch_state < sch_change):
                                    sch_state += 1
                                    scheduler.factor = factors[sch_state]
                                    scheduler.patience = patiences[sch_state]
                                    test_point = int(len(train_data)
                                                     * test_points[sch_state])
                                # If learning rate will not be reduced by more than 
                                # scheduler EPS, then do not run small test anymore
                                if(not (learning_rate*(1-scheduler.factor)
                                        > scheduler.eps)):
                                    step_sch = False

                    # Save model and optimizer states
                    torch.save(model.state_dict(), results_path 
                               +f"model{(epoch+1):02d}.pth")
                    torch.save(optimizer.state_dict(), results_path 
                               +f"optimizer{(epoch+1):02d}.pth")
                    torch.save(scheduler.state_dict(), results_path 
                               +f"scheduler{(epoch+1):02d}.pth")

            # Log average training loss to Aim
            run.track(avg_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, 
                      context={"subset":"train"})
            #run.track(avg_sdr, name='avg_sdr', step=epoch+1, epoch=epoch+1, 
            #          context={"subset":"train"})

        # Full testing phase
        model.eval()
        with torch.no_grad():
            tot_loss = 0
            #tot_sdr = 0
            avg_loss = 0
            #avg_sdr = 0
            pbar = tqdm(enumerate(testloader), total=len(testloader), 
                        desc=f"Test Epoch {epoch+1}")
            for i, (comp_wav, fileinfo) in pbar:
                # Forward pass
                comp_wav = model(comp_wav)

                # Get uncompressed waveform
                if(test_batch_size > 1):
                    uncomp_wav = fileinfo
                else:
                    fileinfo = (fileinfo[0][0], fileinfo[1].item())
                    uncomp_wav = test_funct.process_wav(label_path, fileinfo, 
                                                        test_batch_size != 1, 
                                                        is_input=False)  
                    uncomp_wav = torch.unsqueeze(uncomp_wav, dim=0)
                    # Make sure input waveform is same length as label
                    if(uncomp_wav.shape[2] != comp_wav.shape[2]):
                        print(f"{fileinfo[0][0]} mismatched size, input is " 
                              f"{comp_wav.shape[2]} and label is "
                              f"{uncomp_wav.shape[2]}")
                        comp_wav = funct.upsample(comp_wav, uncomp_wav.shape[2])

                # Calculate individual channel loss
                loss = mrstft(comp_wav, uncomp_wav)

                # Calculate SDR
                #sdr = sisdr(comp_wav, uncomp_wav)

                # Sum waveforms to one channel
                comp_wav = funct.sum(comp_wav)
                uncomp_wav = funct.sum(uncomp_wav)

                # Calculate sum loss
                sum_loss = mel_mrstft(comp_wav, uncomp_wav)

                # Average sum and individual losses
                loss = (loss*(2/3)) + (sum_loss/3)

                # Explicitly delete tensors so they don't stay in memory
                del comp_wav
                del uncomp_wav

                # Add loss and SDR to totals
                tot_loss += loss.item()
                #tot_sdr -= sdr.item()

                # Calculate average loss and SDR so far
                avg_loss = tot_loss / (i+1)
                #avg_sdr = tot_sdr / (i+1)

                # Log loss to Aim
                run.track(loss.item(), name='loss', step=test_step, epoch=epoch+1, 
                          context={"subset":"test"})
                #run.track(-sdr.item(), name='sdr', step=test_step, epoch=epoch+1, 
                #          context={"subset":"test"})
                if(not (i > small_test_end) and run_small_test):
                    run.track(loss.item(), name='loss', step=small_test_step, 
                              epoch=epoch+1, context={"subset":"small_test"})
                    #run.track(-sdr.item(), name='sdr', step=small_test_step, epoch=
                    #          epoch+1, context={"subset":"small_test"})

                # Increment step for next Aim log
                test_step += 1

                # Update tqdm progress bar with fixed number of decimals for loss
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", 
                                  "Avg Loss": f"{avg_loss:.4f}"}) 
                                  #"SDR": f"{-sdr.item():.4f}", 
                                  #"Avg SDR": f"{avg_sdr:.4f}"})

                # Step scheduler after small_test_end batches
                if(i == small_test_end and run_small_test and step_sch):
                    # Send average small test loss to scheduler
                    scheduler.step(avg_loss)
                    # Check if learning rate was changed
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
                    # Log average loss and SDR to Aim
                    run.track(avg_loss, name='avg_loss', step=scheduler.last_epoch, 
                              epoch=epoch+1, context={"subset":"small_test"})
                    #run.track(avg_sdr, name='avg_sdr', step=scheduler.last_epoch, 
                    #          epoch=epoch+1, context={"subset":"small_test"})

       # Step scheduler at the end of full test phase if not set to step in small 
       # tests
        if(not run_small_test and step_sch):
            scheduler.step(avg_loss)
            # Check if learning rate was changed
            if(optimizer.param_groups[0]['lr'] < learning_rate):
                # Save current learning rate
                learning_rate = optimizer.param_groups[0]['lr']
                # Change scheduler parameters after learning rate reduction
                if(sch_state < sch_change):
                    sch_state += 1
                    scheduler.factor = factors[sch_state]
                    scheduler.patience = patiences[sch_state]
                    test_point = int(len(train_data)*test_points[sch_state])
                # If learning rate will not be reduced by more than scheduler EPS, 
                # then do not run small test anymore
                if(not (learning_rate * (1-scheduler.factor) > scheduler.eps)):
                    step_sch = False

        # Save model and optimizer states
        if(test_first and epoch < 1):
            epoch -= 1
        else:
            torch.save(model.state_dict(), 
                       results_path+f"model{(epoch+1):02d}.pth")
            torch.save(optimizer.state_dict(), 
                       results_path+f"optimizer{(epoch+1):02d}.pth")
            torch.save(scheduler.state_dict(), 
                       results_path+f"scheduler{(epoch+1):02d}.pth")

        # Log average test loss and SDR and last model output to Aim
        run.track(avg_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, 
                  context={"subset":"test"})
        #run.track(avg_sdr, name='avg_sdr', step=epoch+1, epoch=epoch+1, 
        #          context={ "subset":"test" })
    
    if(world_size != None):
        cleanup()

    print("Finished Training")

if __name__ == "__main__":
    # Get CPU, GPU, or MPS device for training
    device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    print(f"Using {device} device")

    if(run["hparams", "multigpu"] and device == "cuda:0"):
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPUs")
        mp.spawn(main, args=[n_gpus], nprocs=n_gpus, join=True)
    else:
        main(device, None)

