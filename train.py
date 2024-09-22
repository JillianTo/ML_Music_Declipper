import os
import sys
from aim import Run, Audio
import auraloss
import numpy as np
from tqdm import tqdm
import pickle
from autoclip.torch import QuantileClip
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
            "short_threshold": hparams["short_threshold"],
            "overlap_factor": hparams["overlap_factor"],
            "num_workers": hparams["num_workers"],
            "pin_memory": hparams["pin_memory"],
            "prefetch_factor": hparams["prefetch_factor"],
            "first_out_channels": hparams["first_out_channels"],
            "transformer": hparams["transformer"],
            "preload_weights_path": hparams["preload_weights_path"],
            "preload_optimizer_path": hparams["preload_optimizer_path"],
            "preload_scheduler_path": hparams["preload_scheduler_path"],
            "preload_scaler_path": hparams["preload_scaler_path"],
            "n_fft": hparams["n_fft"],
            "hop_length": hparams["hop_length"],
            "loss_n_ffts": hparams["loss_n_ffts"],
            "n_mels": hparams["n_mels"],
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
            "use_tf32": hparams["use_tf32"],
            "grad_accum": hparams["grad_accum"],
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
    short_thres = hparams["short_threshold"]
    overlap_factor = hparams["overlap_factor"]
    num_workers = hparams["num_workers"]
    pin_memory = hparams["pin_memory"]
    prefetch_factor = hparams["prefetch_factor"]
    first_out_channels = hparams["first_out_channels"]
    preload_weights_path = hparams["preload_weights_path"]
    preload_opt_path = hparams["preload_optimizer_path"]
    preload_sch_path = hparams["preload_scheduler_path"]
    preload_scaler_path = hparams["preload_scaler_path"]
    n_fft = hparams["n_fft"]
    hop_length = hparams["hop_length"]
    loss_n_ffts = hparams["loss_n_ffts"]
    n_mels = hparams["n_mels"]
    top_db = hparams["top_db"]
    sch_state = hparams["scheduler_state"]
    factors = hparams["scheduler_factors"]
    patiences = hparams["scheduler_patiences"]
    overwrite_preload_sch = hparams["overwrite_preloaded_scheduler_values"]
    save_points = hparams["save_points"]
    test_first = hparams["test_first"]
    autoclip = hparams["autoclip"]
    use_amp = hparams["use_amp"]
    grad_accum = hparams["grad_accum"]

    # Load wavs into training data
    print("\nLoading data...")

    # Add inputs and labels to training dataset
    funct = Functional(max_time=max_time, device=rank, n_fft=n_fft, 
                       hop_length=hop_length, 
                       augmentation_lbls=augmentation_lbls)
    if(batch_size > 1):
        train_data = AudioDataset(funct, input_path, train_filelist_path, 
                                  sample_rate=sample_rate,
                                  label_path=label_path, 
                                  short_thres=short_thres,
                                  overlap_factor=overlap_factor)
    else:
        train_data = AudioDataset(funct, input_path, train_filelist_path, 
                                  sample_rate=sample_rate, pad_short=False, 
                                  short_thres=short_thres,
                                  overlap_factor=overlap_factor)
    print(f"Added {len(train_data)} file pairs to training data")

    # Add inputs and labels to test dataset
    if(test_batch_size > 1):
        test_data = AudioDataset(funct, test_input_path, 
                                 test_filelist_path, sample_rate=sample_rate,
                                 label_path=label_path, 
                                 short_thres=short_thres, 
                                 overlap_factor=overlap_factor)
    else:
        test_data = AudioDataset(funct, test_input_path, 
                                 test_filelist_path, sample_rate=sample_rate,
                                 pad_short=False, short_thres=short_thres, 
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
        print(f"Loaded stats from \"{stats_path}\" with mean {mean} and std "
              f"{std}")

    print("\nInitializing model, loss function, and optimizer...")

    # Initialize model
    if hparams["transformer"]:
        model = TransformerModel(mean=mean, std=std, n_fft=n_fft, 
                                 hop_length=hop_length, 
                                 sample_rate=sample_rate, 
                                 first_out_channels=first_out_channels, 
                                 tf_layers=hparams["transformer_n_layers"])
        print("Using Transformer Encoder")
    else:
        model = RNNModel(mean=mean, std=std, n_fft=n_fft, 
                         hop_length=hop_length, 
                         sample_rate=sample_rate,
                         first_out_channels=first_out_channels)
        print("Using LSTM")
    #model = torch.compile(model, mode='default')

    # Load model weights from path if given
    if preload_weights_path != None:
        model.load_state_dict(torch.load(preload_weights_path, 
                                         map_location="cpu",
                                         weights_only=True))
        print(f"Preloaded weights from \"{preload_weights_path}\"")

    # Send model to device
    model = model.to(rank)

    # If using multi-GPU, set up DDP
    if world_size != None:
        model = DDP(model, device_ids=[rank], output_device=rank)
        print("Using DDP")

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
        optimizer.load_state_dict(torch.load(preload_opt_path, 
                                             weights_only=True))
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
        scheduler.load_state_dict(torch.load(preload_sch_path, 
                                             weights_only=True))
        if(overwrite_preload_sch):
            scheduler.factor = factors[sch_state]
            scheduler.patience = patiences[sch_state]
        print(f"Preloaded scheduler from \"{preload_sch_path}\" with factor "
              f"{scheduler.factor} and patience {scheduler.patience}")

    # Initialize gradient scaler
    if rank == 'cpu':
        scaler_device = rank
    else:
        scaler_device = 'cuda' 
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_amp) 
    if preload_scaler_path != None:
        scaler.load_state_dict(torch.load(preload_scaler_path, 
                                          weights_only=True))
        print(f"Preloaded scaler from \"{preload_scaler_path}\"")
    
    # Save when scheduler parameters will last be changed
    sch_change = len(factors) - 1

    # Define local function for saving model, optimizer, and scheduler states
    def save_states(epoch):
        # If using multi-GPU, save model module 
        if world_size != None: 
            torch.save(model.module.state_dict(), checkpoint_path 
                       +f"model{(epoch+1):02d}.pth")
        # Else, save model directly
        else:
            torch.save(model.state_dict(), checkpoint_path 
                       +f"model{(epoch+1):02d}.pth")
        # Save optimizer states
        torch.save(optimizer.state_dict(), checkpoint_path 
                   +f"optimizer{(epoch+1):02d}.pth")
        torch.save(scheduler.state_dict(), checkpoint_path 
                   +f"scheduler{(epoch+1):02d}.pth")
        torch.save(scaler.state_dict(), checkpoint_path
                   +f"scaler{(epoch+1):02d}.pth")

    # Define function for calculating how many train batches until checkpoint
    def calc_save_pt():
        save_point = int(len(trainloader) * save_points[sch_state])
        if save_point < 1:
            return 999
        else:
            return save_point

    # Get target waveform from fileinfo
    def getTgt(batch_size, fileinfo, funct):
        if(batch_size > 1):
            tgt = fileinfo
        else:
            fileinfo = (fileinfo[0][0], fileinfo[1].item(), fileinfo[2].item())
            tgt = funct.process_wav(label_path, fileinfo, sample_rate, 
                                    batch_size != 1, is_input=False)  
            tgt = torch.unsqueeze(tgt, dim=0)
        return tgt
    
    # Steps needed for data plotting in Aim
    train_step = 0
    test_step = 0

    # Determine when to checkpoint
    save_point = calc_save_pt()

    # Define autocast parameters depending on device
    if rank != "cpu":
        autocast_device = "cuda"
        autocast_dtype = torch.float16
    else:
        if rank == "mps":
            print("WARNING: Autocast functionality not tested for MPS")
        autocast_device = rank
        autocast_dtype = torch.bfloat16

    # Training Loop
    print("\nStarting training loop...")
    for epoch in range(hparams["num_epochs"]):
        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Check if test should be run first
        if(not (epoch < 1 and test_first)):
            # If using multi-GPU, set epoch in training data sampler
            if(world_size != None):
                train_sampler.set_epoch(epoch)

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
                with torch.autocast(device_type=autocast_device, 
                                    enabled=use_amp, dtype=autocast_dtype):
                    comp_wav = comp_wav.to(rank)
                    comp_wav = model(comp_wav)

                    # Get target waveform
                    uncomp_wav = getTgt(batch_size, fileinfo, funct)
                    uncomp_wav = uncomp_wav.to(rank)
                    
                    # Calculate loss
                    loss = Functional.calc_loss(comp_wav, uncomp_wav, 
                                                sample_rate, n_ffts=loss_n_ffts, 
                                                n_mels=n_mels, top_db=top_db) / grad_accum

                # Force stop if loss is nan
                if(loss != loss):
                    sys.exit(f"Training loss is {loss} during \"{fileinfo[0]}"
                             f"\", force exiting")

                # Backward pass: compute the gradient of the loss with respect
                # to model parameters
                scaler.scale(loss).backward()
                #print(funct.mean(abs(model.enc1[2].weight.grad)))
                #print(funct.mean(abs(model.enc2[3].weight.grad)))
                #print(funct.mean(abs(model.enc3[3].weight.grad)))
                #print(funct.mean(abs(model.enc4[3].weight.grad)))
                #print(funct.mean(abs(model.enc5[3].weight.grad)))
                #print(funct.mean(abs(model.enc6[4].weight.grad)))
                #print(funct.mean(abs(model.enc7[4].weight.grad)))
                #print(funct.mean(abs(model.lstm[1].weight_ih_l0.grad)))
                #print(funct.mean(abs(model.lstm[1].weight_ih_l1.grad)))
                #print(funct.mean(abs(model.lstm[1].weight_ih_l2.grad)))
                #print(funct.mean(abs(model.lstm[1].weight_ih_l3.grad)))
                #print(funct.mean(abs(model.lstm[1].weight_ih_l4.grad)))
                #print(funct.mean(abs(model.lstm[1].weight_ih_l5.grad)))
                
                if (i + 1) % grad_accum == 0:
                    # Unscale to allow for gradient clipping
                    scaler.unscale_(optimizer)

                    # Update the model's parameters using the optimizer's step 
                    # method
                    scaler.step(optimizer)
                    scaler.update()

                    # Zero the gradients of all optimized variables. This is to 
                    # ensure that we don't accumulate gradients from previous 
                    # batches, as gradients are accumulated by default in 
                    # PyTorch
                    optimizer.zero_grad(set_to_none=True)

                # Add loss to total
                loss_item = loss.item() * grad_accum
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
                        save_states(epoch)

            # Calculate average loss from both GPU if multi-GPU was used
            if world_size != None:
                tot_loss = torch.tensor(tot_loss, device=rank)
                dist.all_reduce(tot_loss, dist.ReduceOp.SUM, async_op=False)
                avg_loss = tot_loss / (len(trainloader)*world_size)

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
                    uncomp_wav = getTgt(test_batch_size, fileinfo, funct)
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
            dist.all_reduce(tot_loss, dist.ReduceOp.SUM, async_op=False)
            avg_loss = tot_loss / (len(testloader)*world_size)

        # Save current learning rate from scheduler
        prev_lr = scheduler.get_last_lr()

        # Send average testing loss to scheduler
        scheduler.step(avg_loss)

        # Save current learning rate from scheduler
        curr_lr = scheduler.get_last_lr()

        # Change out scheduler parameters if necessary
        if(curr_lr < prev_lr):
            # Change scheduler parameters after learning rate reduction
            if(sch_state < sch_change):
                sch_state += 1
                scheduler.factor = factors[sch_state]
                scheduler.patience = patiences[sch_state]
                save_point = calc_save_pt()
                print(f"Scheduler changed to state {sch_state} on rank {rank}")

        if world_size == None or rank == 0:
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
                save_states(epoch)

    # If multi-GPU was used, cleanup before exiting 
    if(world_size != None):
        cleanup()

    print("Finished Training")

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
   
    # Initialize variables for relevant setup parameters
    train_filelist_path = hparams["train_filelist_path"]
    test_filelist_path = hparams["test_filelist_path"]
    max_time = hparams["max_time"]
    short_thres = hparams["short_threshold"]
    overlap_factor = hparams["overlap_factor"]
    input_path = hparams["input_data_path"]
    test_input_path = hparams["test_input_data_path"]
    stats_path = hparams["stats_path"]
    n_fft = hparams["n_fft"]
    hop_length = hparams["stats_hop_length"]
    top_db = hparams["top_db"]
    use_tf32 = hparams["use_tf32"]
   
    # Initialize instance of functional with relevant parameters
    funct = Functional(max_time=max_time, device=device, n_fft=n_fft, 
                       hop_length=hop_length) 

    # Enable CUDA optimizations
    if device == cuda_device:
        #torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Get CPU, GPU, or MPS device for training
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
        if hparams["multigpu"]:
            print(f"WARNING: DDP not supported for stats calculations, likely "
                  f"to OOM when training starts. Recommended to restart after "
                  f"stats are calculated.")
        # Calculate mean and std for training data
        mean, std = funct.db_stats(input_path, hparams["stats_time"])
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

