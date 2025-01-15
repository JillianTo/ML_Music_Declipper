import os
import sys
from aim import Run, Audio
from tqdm import tqdm
import pickle
from autoclip.torch import QuantileClip
import datetime

from audiodataset import AudioDataset
from functional import Functional
from model import Model

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
        hparam_dict = {}
        for hparam in hparams:
            hparam_dict[hparam] = hparams[hparam]
        run["hparams"] = hparam_dict

    # Save individual hyperparameters to variables for easier access
    learning_rate = hparams["learning_rate"]
    sample_rate = hparams["expected_sample_rate"]
    batch_size = hparams["batch_size"]
    val_batch_size = hparams["val_batch_size"]
    stats_path = hparams["stats_path"]
    train_filelist_path = hparams["train_filelist_path"]
    val_filelist_path = hparams["val_filelist_path"]
    train_label_path = hparams["train_label_data_path"]
    val_label_path = hparams["val_label_data_path"]
    input_path = hparams["input_data_path"]
    checkpoint_path = hparams["checkpoint_path"]
    augmentation_lbls = hparams["augmentation_labels"]
    max_time = hparams["max_time"]
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
    loss_n_ffts = hparams["loss_n_ffts"]
    n_mels = hparams["n_mels"]
    top_db = hparams["top_db"]
    sch_state = hparams["scheduler_state"]
    factors = hparams["scheduler_factors"]
    patiences = hparams["scheduler_patiences"]
    overwrite_preload_sch = hparams["overwrite_preloaded_scheduler_values"]
    save_points = hparams["save_points"]
    val_first = hparams["val_first"]
    autoclip = hparams["autoclip"]
    use_amp = hparams["use_amp"]
    grad_accum = hparams["grad_accum"]

    # Load wavs into training data
    print("\nLoading data...")

    # Add inputs and labels to training dataset
    funct = Functional(max_time=max_time, device=rank, n_fft=n_fft, 
                       hop_length=hop_length, 
                       augmentation_lbls=augmentation_lbls)

    train_data = AudioDataset(funct, input_path, train_filelist_path,
                              lbl_path=train_label_path, 
                              sample_rate=sample_rate, pad_short=False, 
                              short_thres=short_thres,
                              overlap_factor=overlap_factor,
                              rtn_lbl_wav=batch_size>1)
    print(f"Added {len(train_data)} file pairs to training data")

    # Add inputs and labels to validation dataset
    val_data = AudioDataset(funct, input_path, val_filelist_path, 
                            lbl_path=val_label_path, sample_rate=sample_rate,
                            short_thres=short_thres, 
                            overlap_factor=overlap_factor,
                            rtn_lbl_wav=val_batch_size>1)
    print(f"Added {len(val_data)} file pairs to validation data")

    # If using multi-GPU, set up samplers and dataloaders accordingly
    if(world_size != None):
        # Create samplers
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, 
                                           rank=rank, shuffle=True, 
                                           drop_last=False)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, 
                                          rank=rank, shuffle=False, 
                                          drop_last=False)
        print("Using distributed samplers for training/validation data")

        # Create data loader for training data 
        trainloader = DataLoader(train_data, batch_size=batch_size, 
                                 shuffle=False, sampler=train_sampler, 
                                 num_workers=num_workers, 
                                 pin_memory=pin_memory, 
                                 prefetch_factor=prefetch_factor)
        # Create data loader for validation data 
        valloader = DataLoader(val_data, batch_size=val_batch_size, 
                                shuffle=False, sampler=val_sampler, 
                                num_workers=num_workers, pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor)
    # Else, set up dataloaders for single device
    else:
        # Create data loader for training data, enable shuffle if not using DDP
        trainloader = DataLoader(train_data, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers, 
                                 pin_memory=pin_memory, 
                                 prefetch_factor=prefetch_factor) 
        # Create data loader for validation data 
        valloader = DataLoader(val_data, batch_size=val_batch_size, 
                                shuffle=False, num_workers=num_workers, 
                                pin_memory=pin_memory, 
                                prefetch_factor=prefetch_factor) 

    # Load stats from file
    if hparams["mean_normalization"]:
        with open(stats_path, 'rb') as f:
            log_stats = pickle.load(f)
            mean = log_stats[0]
            std = log_stats[1]
            print(f"Loaded stats from \"{stats_path}\" with mean {mean} and "
                  f"std {std}")
    else:
        mean = None
        std = None

    print("\nInitializing model, loss function, and optimizer...")

    # Initialize model
    model = Model(n_fft=n_fft, hop_length=hop_length, top_db=top_db, 
                  first_out_channels=hparams["first_out_channels"], 
                  bn_layers=hparams["n_layers"], nhead=hparams["n_heads"], 
                  mean=mean, std=std)
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
    def getTgt(batch_size, fileinfo, funct, label_path):
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
    val_step = 0

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

        # Check if validation should be run first
        if(not (epoch < 1 and val_first)):
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
                    uncomp_wav = getTgt(batch_size, fileinfo, funct, 
                                        train_label_path)
                    uncomp_wav = uncomp_wav.to(rank)
                    
                    # Calculate loss
                    loss = Functional.calc_loss(comp_wav, uncomp_wav, 
                                                sample_rate, n_ffts=loss_n_ffts, 
                                                n_mels=n_mels, top_db=top_db) / grad_accum

                # Force stop if loss is nan
                if(loss != loss):
                    sys.exit(f"Training loss is {loss} during \""
                             f"{fileinfo[0][0]}\", force exiting")

                # Backward pass: compute the gradient of the loss with respect
                # to model parameters
                scaler.scale(loss).backward()
                
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
            # Initialize total and average validation loss to zero
            tot_loss = 0
            avg_loss = 0
            # Initialize tqdm progress bar for validation epoch
            pbar = tqdm(enumerate(valloader), total=len(valloader), 
                        desc=f"Validation Epoch {epoch+1}")

            # Validation phase
            for i, (comp_wav, fileinfo) in pbar:
                with torch.autocast(device_type="cuda", enabled=use_amp, 
                                    dtype=torch.float16):
                    # Forward pass
                    comp_wav = comp_wav.to(rank)
                    comp_wav = model(comp_wav)

                    # Get target waveform
                    uncomp_wav = getTgt(val_batch_size, fileinfo, funct, 
                                        val_label_path)
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
                    run.track(loss_item, name='loss', step=val_step, 
                              epoch=epoch+1, context={"subset":"val"})

                # Increment step for next Aim log
                val_step += 1

                # Update tqdm progress bar 
                pbar.set_postfix({"Loss": f"{loss_item:.4f}", 
                                  "Avg Loss": f"{avg_loss:.4f}"}) 

        # Calculate average loss from both GPU if multi-GPU was used
        if world_size != None:
            tot_loss = torch.tensor(tot_loss, device=rank)
            dist.all_reduce(tot_loss, dist.ReduceOp.SUM, async_op=False)
            avg_loss = tot_loss / (len(valloader)*world_size)

        # Save current learning rate from scheduler
        prev_lr = scheduler.get_last_lr()

        # Send average validation loss to scheduler
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
            # Log average validation loss to Aim
            run.track(avg_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, 
                      context={"subset":f"val"})

        # Check if validation was run without training  
        if(val_first and epoch < 1):
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
    val_filelist_path = hparams["val_filelist_path"]
    max_time = hparams["max_time"]
    short_thres = hparams["short_threshold"]
    overlap_factor = hparams["overlap_factor"]
    input_path = hparams["input_data_path"]
    train_label_path = hparams["train_label_data_path"]
    val_label_path = hparams["val_label_data_path"]
    stats_path = hparams["stats_path"]
    augmentation_lbls = hparams["augmentation_labels"]
    n_fft = hparams["n_fft"]
    hop_length = hparams["stats_hop_length"]
    top_db = hparams["top_db"]
    use_tf32 = hparams["use_tf32"]
   
    # Initialize instance of functional with relevant parameters
    funct = Functional(max_time=max_time, device=device, n_fft=n_fft, 
                       hop_length=hop_length, 
                       augmentation_lbls=augmentation_lbls) 

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
        funct.get_filelist(train_label_path, short_thres, train_filelist_path, 
                           overlap_factor)

    # Generate validation data filelist if it has not been
    if(not os.path.isfile(val_filelist_path)):
        funct.get_filelist(val_label_path, short_thres, val_filelist_path, 
                           overlap_factor)

    # Calculate training data stats if it has not been
    multigpu_stats = False
    if(not os.path.isfile(stats_path) and hparams["mean_normalization"]):
        print("Calculating mean and std...")
        if hparams["multigpu"]:
            multigpu_stats = True
            print(f"WARNING: DDP not supported for stats calculations, rerun "
                  f"training script after stats are calculated.")
        # Calculate mean and std for training data
        mean, std = funct.log_stats(input_path, train_label_path, 
                                   hparams["stats_time"], hparams["top_db"])
        mean = mean.item()
        std = std.item()
        print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")
        with open(stats_path, 'wb') as f:
            pickle.dump([mean,std], f)
        print(f"Saved training set stats in \"{stats_path}\'")
    
    if not multigpu_stats:
        # Check whether to use multi-GPU processing 
        if(hparams["multigpu"] and device == cuda_device):
            n_gpus = torch.cuda.device_count()
            print(f"Using {n_gpus} GPUs")
            mp.spawn(main, args=[n_gpus], nprocs=n_gpus, join=True)
        # Else, run on one device
        else:
            main(device, None)

