import os
import sys
from aim import Run, Audio
import auraloss
import numpy as np
from tqdm import tqdm
import pickle
from autoclip.torch import QuantileClip
import time

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

# Set up DDP process groups
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# Destroy DDP process groups
def cleanup():
    dist.destroy_process_group()

# Get hyperparameters from file
def getHparams():
    # If no input argument, use default hyperparameter file path
    if len(sys.argv) < 2:
        hparams_path = "hparams.txt"    
    else:
        hparams_path = sys.argv[1] 
    
    # Load hyperparameters
    with open(hparams_path, 'rb') as f:
        hparams = pickle.load(f)

    return hparams

# Process data and run training
def main(rank, world_size):
    # If using multi-GPU, set up DDP process groups
    if(world_size != None):
        # Set up process groups
        setup(rank, world_size)

    # Load hyperparameters
    hparams = getHparams()

    # Initialize Aim run on master process
    if world_size == None or rank == 0:
        # Initialize a new Aim run
        run = Run(experiment="declipper")

        # Log run parameters
        run["hparams"] = {
            "learning_rate": hparams["learning_rate"],
            "batch_size": hparams["batch_size"],
            "test_batch_size": hparams["test_batch_size"],
            "num_epochs": hparams["num_epochs"], 
            "expected_sample_rate": hparams["expected_sample_rate"],
            "stats_path": hparams["stats_path"],
            "train_filelist_path": hparams["train_filelist_path"],
            "test_filelist_path": hparams["test_filelist_path"],
            "label_data_path": hparams["label_data_path"],
            "input_data_path": hparams["input_data_path"],
            "test_input_data_path": hparams["test_input_data_path"],
            "checkpoint_path": hparams["checkpoint_path"],
            "augmentation_labels": hparams["augmentation_labels"],
            "max_time": hparams["max_time"],
            "test_max_time": hparams["test_max_time"],
            "short_threshold": hparams["short_threshold"],
            "overlap_factor": hparams["overlap_factor"],
            "num_workers": hparams["num_workers"],
            "pin_memory": hparams["pin_memory"],
            "prefetch_factor": hparams["prefetch_factor"],
            "spectrogram_autoencoder": hparams["spectrogram_autoencoder"],
            "preload_weights_path": hparams["preload_weights_path"],
            "preload_optimizer_path": hparams["preload_optimizer_path"],
            "preload_scheduler_path": hparams["preload_scheduler_path"],
            "preload_scaler_path": hparams["preload_scaler_path"],
            "n_fft": hparams["n_fft"],
            "hop_length": hparams["hop_length"],
            "top_db": hparams["top_db"],
            "eps": hparams["eps"],
            "scheduler_state": hparams["scheduler_state"],
            "scheduler_factors": hparams["scheduler_factors"],
            "scheduler_patiences": hparams["scheduler_patiences"],
            "save_points": hparams["save_points"],
            "overwrite_preloaded_scheduler_values": 
                hparams["overwrite_preloaded_scheduler_values"],
            "test_first": hparams["test_first"],
            "autoclip": hparams["autoclip"],
            "multigpu": hparams["multigpu"], 
            "cuda_device": hparams["cuda_device"], 
            "use_amp": hparams["use_amp"],
        }

    # Save individual hyperparameters to variables for easier access
    learning_rate = hparams["learning_rate"]
    sample_rate = hparams["expected_sample_rate"]
    batch_size = hparams["batch_size"]
    test_batch_size = hparams["test_batch_size"]
    stats_path = hparams["stats_path"]
    train_filelist_path = hparams["train_filelist_path"]
    test_filelist_path = hparams["test_filelist_path"]
    input_path = hparams["input_data_path"]
    test_input_path = hparams["test_input_data_path"]
    label_path = hparams["label_data_path"]
    checkpoint_path = hparams["checkpoint_path"]
    augmentation_lbls = hparams["augmentation_labels"]
    max_time = hparams["max_time"]
    test_max_time = hparams["test_max_time"]
    short_thres = hparams["short_threshold"]
    overlap_factor = hparams["overlap_factor"]
    num_workers = hparams["num_workers"]
    pin_memory = hparams["pin_memory"]
    prefetch_factor = hparams["prefetch_factor"]
    preload_weights_path = hparams["preload_weights_path"]
    preload_opt_path = hparams["preload_optimizer_path"]
    preload_sch_path = hparams["preload_scheduler_path"]
    preload_scaler_path = hparams["preload_scaler_path"]
    n_fft = hparams["n_fft"]
    hop_length = hparams["hop_length"]
    top_db = hparams["top_db"]
    sch_state = hparams["scheduler_state"]
    factors = hparams["scheduler_factors"]
    patiences = hparams["scheduler_patiences"]
    overwrite_preload_sch = hparams["overwrite_preloaded_scheduler_values"]
    save_points = hparams["save_points"]
    test_first = hparams["test_first"]
    autoclip = hparams["autoclip"]
    use_amp = hparams["use_amp"]

    # Load wavs into training data
    print("\nLoading data...")

    # Add inputs and labels to training dataset
    funct = Functional(sample_rate=sample_rate, max_time=max_time, device=rank, 
                       n_fft=n_fft, hop_length=hop_length, 
                       top_db=top_db, augmentation_lbls=augmentation_lbls)
    if(batch_size > 1):
        train_data = AudioDataset(funct, input_path, train_filelist_path, 
                                  label_path=label_path, 
                                  short_thres=short_thres,
                                  overlap_factor=overlap_factor)
    else:
        train_data = AudioDataset(funct, input_path, train_filelist_path, 
                                  pad_short=False, short_thres=short_thres,
                                  overlap_factor=overlap_factor)
    print(f"Added {len(train_data)} file pairs to training data")

    # Add inputs and labels to test dataset
    test_funct = Functional(sample_rate=sample_rate, max_time=test_max_time, 
                            device=rank, n_fft=n_fft, 
                            hop_length=hop_length, top_db=top_db,
                            augmentation_lbls=augmentation_lbls)
    if(test_batch_size > 1):
        test_data = AudioDataset(test_funct, test_input_path, 
                                 test_filelist_path, label_path=label_path, 
                                 short_thres=short_thres, 
                                 overlap_factor=overlap_factor)
    else:
        test_data = AudioDataset(test_funct, test_input_path, 
                                 test_filelist_path, pad_short=False, 
                                 short_thres=short_thres, 
                                 overlap_factor=overlap_factor)
    print(f"Added {len(test_data)} file pairs to test data")

    # If using multi-GPU, set up samplers and dataloaders accordingly
    if(world_size != None):
        # Create samplers
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, 
                                           rank=rank, shuffle=True, 
                                           drop_last=False)
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, 
                                          rank=rank, shuffle=False, 
                                          drop_last=False)
        print("Using distributed samplers for training/testing data")

        # Create data loader for training data 
        trainloader = DataLoader(train_data, batch_size=batch_size, 
                                 shuffle=False, sampler=train_sampler, 
                                 num_workers=num_workers, 
                                 pin_memory=pin_memory, 
                                 prefetch_factor=prefetch_factor)
        # Create data loader for test data 
        testloader = DataLoader(test_data, batch_size=test_batch_size, 
                                shuffle=False, sampler=test_sampler, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor)
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
    with open(stats_path, 'rb') as f:
        db_stats = pickle.load(f)
        mean = db_stats[0]
        std = db_stats[1]
        print(f"Loaded stats from \"path\" with mean {mean} and std "
              f"{std}")

    print("\nInitializing model, loss function, and optimizer...")

    # Initialize model
    if hparams["spectrogram_autoencoder"]:
        model = AutoEncoder(mean=mean, std=std, n_fft=n_fft, 
                            hop_length=hop_length, 
                            sample_rate=sample_rate)
        print("Using spectrogram autoencoder")
    #model = torch.compile(model, mode='default')
    model = model.to(rank)

    # Load model weights from path if given
    if preload_weights_path != None:
        model.load_state_dict(torch.load(preload_weights_path, 
                                         map_location=torch.device(rank)))
        print(f"Preloaded weights from \"{preload_weights_path}\"")

    # If using multi-GPU, set up DDP
    if(world_size != None):
        model = DDP(model, device_ids=[rank], output_device=rank,
                    find_unused_parameters=True)

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
                                                     eps=hparams["eps"])
    # Load saved scheduler from path if given
    if preload_sch_path != None:
        scheduler.load_state_dict(torch.load(preload_sch_path))
        if(overwrite_preload_sch):
            scheduler.factor = factors[sch_state]
            scheduler.patience = patiences[sch_state]
        print(f"Preloaded scheduler from \"{preload_sch_path}\" with factor "
              f"{scheduler.factor} and patience {scheduler.patience}")

    # Initialize gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) 
    if preload_scaler_path != None:
        scaler.load_state_dict(torch.load(preload_scaler_path))
        print(f"Preloaded scaler from \"{preload_scaler_path}\"")
    
    # Save when scheduler parameters will last be changed
    sch_change = len(factors) - 1

    # Define local function for saving model, optimizer, and scheduler states
    def saveStates(epoch):
        # Save model and optimizer states
        torch.save(model.state_dict(), checkpoint_path 
                   +f"model{(epoch+1):02d}.pth")
        torch.save(optimizer.state_dict(), checkpoint_path 
                   +f"optimizer{(epoch+1):02d}.pth")
        torch.save(scheduler.state_dict(), checkpoint_path 
                   +f"scheduler{(epoch+1):02d}.pth")
        torch.save(scaler.state_dict(), checkpoint_path
                   +f"scaler{(epoch+1):02d}.pth")

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
                                                       hop_sizes=[fft_size//4 
                                                                  for fft_size 
                                                                  in 
                                                                  fft_sizes], 
                                                       win_lengths=fft_sizes, 
                                                       sample_rate=sample_rate, 
                                                       scale='mel', n_bins=256)
    mel_mrstft.to(rank)

    # Define function for calculating loss
    def calcLoss(pred, tgt):
        # Make sure input waveform is same length as label
        if(tgt.shape[2] != pred.shape[2]):
            print(f"{fileinfo[0][0]} mismatched size, input is " 
                  f"{pred.shape[2]} and label is "
                  f"{tgt.shape[2]}")
            pred = funct.upsample(pred, tgt.shape[2])

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
    test_step = 0

    # After how many train batches to run small test
    save_point = int(len(trainloader) * save_points[sch_state])

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
    for epoch in range(hparams["num_epochs"]):
        # Clear CUDA cache
        torch.cuda.empty_cache()

        # If using multi-GPU, set epoch in training data sampler
        if(world_size != None):
            train_sampler.set_epoch(epoch)

        # Check if test should be run first
        if(not (epoch < 1 and test_first)):
            # Initialize total and average training loss to zero
            tot_loss = 0
            avg_loss = 0
            # Initialize tqdm progress bar for training epoch
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), 
                        desc=f"Train Epoch {epoch+1}")
            # Make sure optimizer gradients are zeroed out
            optimizer.zero_grad(set_to_none=True)

            # Training phase
            for i, (comp_wav, fileinfo) in pbar:
                # Set model to train state
                model.train()


                # Forward pass
                with torch.autocast(device_type="cuda", enabled=use_amp, 
                                    dtype=torch.float16):
                    comp_wav = comp_wav.to(rank)
                    comp_wav = model(comp_wav)

                    # Get target waveform
                    uncomp_wav = getTgt(batch_size, fileinfo, funct)
                    uncomp_wav = uncomp_wav.to(rank)
                    
                    # Calculate loss
                    loss = calcLoss(comp_wav, uncomp_wav)

                # Force stop if loss is nan
                if(loss != loss):
                    sys.exit(f"Training loss is {loss} during \"{fileinfo[0]}"
                             f"\", force exiting")

                # Backward pass: compute the gradient of the loss with respect
                # to model parameters
                scaler.scale(loss).backward()
                #print(funct.mean(abs(model.enc1[3].weight.grad)))
                #print(funct.mean(abs(model.enc2[4].weight.grad)))
                #print(funct.mean(abs(model.enc3[4].weight.grad)))
                #print(funct.mean(abs(model.enc4[4].weight.grad)))
                #print(funct.mean(abs(model.enc5[4].weight.grad)))
                #print(funct.mean(abs(model.enc6[4].weight.grad)))
                #print(funct.mean(abs(model.enc7[4].weight.grad)))
                #print(funct.mean(abs(model.lstm.weight_ih_l0.grad)))
                #print(funct.mean(abs(model.lstm.weight_ih_l1.grad)))
                #print(funct.mean(abs(model.lstm.weight_ih_l2.grad)))
                #print(funct.mean(abs(model.lstm.weight_ih_l3.grad)))
                #print(funct.mean(abs(model.lstm.weight_ih_l4.grad)))
                #print(funct.mean(abs(model.lstm.weight_ih_l5.grad)))
                
                # Update the model's parameters using the optimizer's step 
                # method
                #optimizer.step()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()

                # Zero the gradients of all optimized variables. This is to 
                # ensure that we don't accumulate gradients from previous 
                # batches, as gradients are accumulated by default in 
                # PyTorch
                optimizer.zero_grad(set_to_none=True)

                # Add loss to total
                loss_item = loss.item()
                tot_loss += loss_item

                # Calculate average loss so far
                avg_loss = tot_loss / (i+1)

                # Log loss to Aim
                if world_size == None or rank == 0:
                    run.track(loss_item, name='loss', step=train_step, 
                              epoch=epoch+1, context={"subset":f"train"})

                # Update tqdm progress bar 
                pbar.set_postfix({"Loss": f"{loss_item:.4f}", 
                                  "Avg Loss": f"{avg_loss:.4f}"})

                # Increment step for next Aim log
                train_step = train_step + 1

                # If training has completed for save_point number of batches
                if(i%save_point == 0 and i > 0):
                    # If multi-GPU is enabled, make sure you are saving from 
                    # the first device
                    if world_size == None or rank == 0:
                        # Save current model, optimizer, and scheduler states
                        saveStates(epoch)

            # Log average training loss to Aim
            if world_size == None or rank == 0:
                run.track(avg_loss, name='avg_loss', step=epoch+1, 
                          epoch=epoch+1, context={"subset":f"train"})
            
            # Clear CUDA cache
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
                with torch.autocast(device_type="cuda", enabled=use_amp, 
                                    dtype=torch.float16):
                    # Forward pass
                    comp_wav = comp_wav.to(rank)
                    comp_wav = model(comp_wav)

                    # Get target waveform
                    uncomp_wav = getTgt(test_batch_size, fileinfo, test_funct)
                    uncomp_wav = uncomp_wav.to(rank)

                    # Calculate loss
                    loss = calcLoss(comp_wav, uncomp_wav)

                # Add loss to total
                loss_item = loss.item()
                tot_loss += loss_item

                # Calculate average loss so far
                avg_loss = tot_loss / (i+1)

                # Log loss to Aim
                if world_size == None or rank == 0:
                    run.track(loss_item, name='loss', step=test_step, 
                              epoch=epoch+1, context={"subset":"test"})

                # Increment step for next Aim log
                test_step += 1

                # Update tqdm progress bar 
                pbar.set_postfix({"Loss": f"{loss_item:.4f}", 
                                  "Avg Loss": f"{avg_loss:.4f}"}) 

        # Calculate average loss from both GPU if multi-GPU was used
        if world_size != None:
            tot_loss = torch.tensor(tot_loss, device=rank)
            dist.barrier()
            dist.all_reduce(tot_loss, dist.ReduceOp.SUM, async_op=False)
            avg_loss = tot_loss / (len(testloader)*world_size)

        if world_size == None or rank == 0:
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
                    save_point = int(len(trainloader) 
                                     * save_points[sch_state])

            # Log average test loss to Aim
            run.track(avg_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, 
                      context={"subset":f"test"})

        # Check if test was run without training  
        if(test_first and epoch < 1):
            # Decrement epoch number so training starts at first epoch
            epoch -= 1
        else:
            # Save current model, optimizer, and scheduler states
            if world_size == None or rank == 0:
                saveStates(epoch)

    # If multi-GPU was used, cleanup before exiting 
    if(world_size != None):
        cleanup()

    print("Finished Training")

if __name__ == "__main__":
    # Print Torch version
    print("Torch version: " + torch.__version__)
   
    # Load hyperparameters
    hparams = getHparams()

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get CPU, GPU, or MPS device for training
    cuda_device = f"cuda:{hparams['cuda_device']}"
    device = (
            cuda_device
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    print(f"Using {device} device")
   
    # Initialize variables for creating stats and filelists if needed
    train_filelist_path = hparams["train_filelist_path"]
    test_filelist_path = hparams["test_filelist_path"]
    max_time = hparams["max_time"]
    short_thres = hparams["short_threshold"]
    overlap_factor = hparams["overlap_factor"]
    input_path = hparams["input_data_path"]
    test_input_path = hparams["test_input_data_path"]
    stats_path = hparams["stats_path"]
    n_fft = hparams["n_fft"]
    #hop_length = hparams["hop_length"]
    hop_length = n_fft>>1
    top_db = hparams["top_db"]
    funct = Functional(max_time=max_time, device=device, n_fft=n_fft, 
                       hop_length=hop_length, top_db=top_db)

    # Generate training data filelist if it has not been
    if(not os.path.isfile(train_filelist_path)):
        funct.get_filelist(input_path, short_thres, train_filelist_path, 
                           overlap_factor)

    # Generate testing data filelist if it has not been
    if(not os.path.isfile(test_filelist_path)):
        funct.get_filelist(test_input_path, short_thres, test_filelist_path, 
                           overlap_factor)

    # Calculate training data stats if it has not been
    if(not os.path.isfile(stats_path)):
        print("Calculating mean and std...")
        # Calculate mean and std for training data
        mean, std = funct.db_stats(input_path)
        mean = mean.item()
        std = std.item()
        print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")
        with open(stats_path, 'wb') as f:
            pickle.dump([mean,std], f)
        print(f"Saved training set stats in \"{stats_path}\'")

    # Check whether to use multi-GPU processing 
    if(hparams["multigpu"] and device == cuda_device):
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPUs")
        mp.spawn(main, args=[n_gpus], nprocs=n_gpus, join=True)
    # Else, run on one device
    else:
        main(device, None)

