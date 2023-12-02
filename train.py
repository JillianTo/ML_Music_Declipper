import os
import sys
from aim import Run, Audio
import auraloss
from tqdm import tqdm

from audiodataset import AudioDataset
from autoencoder import SpecAutoEncoder, WavAutoEncoder
from functional import Functional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
print("Torch version: " + torch.__version__)

# Initialize a new run
run = Run(experiment="declipper")

# Log run parameters
run["hparams"] = {
    "learning_rate": 0.00002,
    "batch_size": 2,
    "test_batch_size": 1,
    "num_epochs": 100, 
    #"mean": -7.5930,
    #"std": 16.4029, 
    #"mean": -8.9372,
    #"std": 14.4010, 
    "mean": -9.0133,
    "std": 14.3514, 
    #"mean": None,
    #"std": None, 
    "stats_compute_stop": None,
    "expected_sample_rate": 44100,
    "uncompressed_data_path": "/mnt/MP600/data/uncomp/",
    "compressed_data_path": "/mnt/MP600/data/comp/train/",
    #"compressed_data_path": "/mnt/MP600/data/comp/small/train/",
    #"compressed_data_path": "/mnt/MP600/data/comp/test/",
    "test_compressed_data_path": "/mnt/MP600/data/comp/test/",
    #"test_compressed_data_path": "/mnt/MP600/data/comp/small/test/",
    #"test_compressed_data_path": "/mnt/MP600/data/comp/smallTest/",
    "results_path": "/mnt/MP600/data/results/",
    #"max_time": 900000,
    "max_time": 1250000,
    #"max_time": 14000000,
    #"test_max_time": 1800000,
    "test_max_time": 3500000,
    #"test_max_time": 20000000,
    "num_workers": 0,
    "prefetch_factor": None,
    "pin_memory": False,
    "spectrogram_autoencoder": True,
    #"preload_weights_path": "/mnt/MP600/data/results/model01.pth",
    "preload_weights_path": None,
    #"preload_optimizer_path": "/mnt/MP600/data/results/optimizer01.pth",
    "preload_optimizer_path": None,
    "preload_scheduler_path": None,
    "accuracy_bound": 0.01,
    "save_partial_epoch": True,
    "scheduler_state": 0,
    "scheduler_changes": 1,
    "scheduler_factors": [(1/2), 0.1],
    "scheduler_patiences": [1, 2],
    "test_points": [0.02, 0.05],
}

# Save run parameters to variables for easier access
learning_rate = run["hparams", "learning_rate"]
mean = run["hparams", "mean"]
std = run["hparams", "std"]
stats_compute_stop = run["hparams", "stats_compute_stop"]
sample_rate = run["hparams", "expected_sample_rate"]
batch_size = run["hparams", "batch_size"]
test_batch_size = run["hparams", "test_batch_size"]
label_path = run["hparams", "uncompressed_data_path"]
results_path = run["hparams", "results_path"]
num_workers = run["hparams", "num_workers"]
prefetch_factor = run["hparams", "prefetch_factor"]
pin_memory = run["hparams", "pin_memory"]
preload_weights_path = run["hparams", "preload_weights_path"]
preload_opt_path = run["hparams", "preload_optimizer_path"]
preload_sch_path = run["hparams", "preload_scheduler_path"]
acc_bound = run["hparams", "accuracy_bound"]
save_partial_epoch = run["hparams", "save_partial_epoch"]
sch_state = run["hparams", "scheduler_state"]
sch_change = run["hparams", "scheduler_changes"]
factors = run["hparams", "scheduler_factors"]
patiences = run["hparams", "scheduler_patiences"]
test_points = run["hparams", "test_points"]

# Get CPU, GPU, or MPS device for training
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
#device = "cpu"
print(f"Using {device} device")

# Load wavs into training data
print("\nLoading data...")

# Add inputs and labels to training dataset
funct = Functional(sample_rate, run["hparams", "max_time"], device)
if(batch_size > 1):
    training_data = AudioDataset(funct, run["hparams", "compressed_data_path"], "filelist_train.txt", label_path=label_path, same_time=batch_size!=1, pad_thres=2)
else:
    training_data = AudioDataset(funct, run["hparams", "compressed_data_path"], "filelist_train.txt", same_time=batch_size!=1, pad_thres=2)
    #training_data = AudioDataset(funct, run["hparams", "compressed_data_path"], None, same_time=batch_size!=1, pad_thres=2)
print(f"Added {len(training_data)} file pairs to training data")

# Add inputs and labels to test dataset
test_funct = Functional(sample_rate, run["hparams", "test_max_time"], device)
if(test_batch_size > 1):
    test_data = AudioDataset(test_funct, run["hparams", "test_compressed_data_path"], "filelist_test.txt", label_path=label_path, same_time=test_batch_size!=1, pad_thres=2)
else:
    test_data = AudioDataset(test_funct, run["hparams", "test_compressed_data_path"], "filelist_test.txt", same_time=test_batch_size!=1, pad_thres=2)
    #test_data = AudioDataset(test_funct, run["hparams", "test_compressed_data_path"], None, same_time=test_batch_size!=1, pad_thres=2)
print(f"Added {len(test_data)} file pairs to test data")

# Create data loaders
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    testloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

# Calculate mean and std for training set if not given
if(mean == None or std == None):
    print("Calculating mean and std...")
    mean = 0
    std = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Train Mean and Standard Deviation")

    for i, (comp_wav, _) in pbar:
        curr_std, curr_mean = funct.compute_std_mean(comp_wav)
        mean = mean + curr_mean
        std = std + curr_std

        # Update tqdm progress bar with fixed number of decimals for loss
        pbar.set_postfix({"Mean": f"{curr_mean:.4f}", "Standard Deviation": f"{curr_std:.4f}"})

        if(stats_compute_stop != None and stats_compute_stop < i-1):
            print(f"Stopping mean and std calculation after {i+1} samples")
            break

    mean = mean/(i+1)
    std = std/(i+1)
    print(f"Mean: {mean:.4f}, Standard Deviation: {std:.4f}")

# Initialize model
print("\nInitializing model, loss function, and optimizer...")
if run["hparams", "spectrogram_autoencoder"]:
    model = SpecAutoEncoder(device, mean, std)
    print("Using spectrogram autoencoder")
else:
    model = WavAutoEncoder(device)
    print("Using waveform autoencoder")
if preload_weights_path != None:
    model.load_state_dict(torch.load(preload_weights_path))
    print(f"Preloaded weights from \"{preload_weights_path}\"")
model.to(device)

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
if preload_opt_path != None:
    optimizer.load_state_dict(torch.load(preload_opt_path))
    print(f"Preloaded optimizer from \"{preload_opt_path}\"")

# Initialize scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factors[sch_state], patience=patiences[sch_state], verbose=True)
if preload_sch_path != None:
    scheduler.load_state_dict(torch.load(preload_sch_path))
    print(f"Preloaded scheduler from \"{preload_sch_path}\"")

# Initialize loss function
#criterion = nn.MSELoss()
#fft_sizes = [4096, 8192, 16384]
fft_sizes = [2048, 4096, 8192]
criterion = auraloss.freq.SumAndDifferenceSTFTLoss(fft_sizes=fft_sizes, hop_sizes=[fft_size//2 for fft_size in fft_sizes], win_lengths=fft_sizes, sample_rate=sample_rate, perceptual_weighting=True, scale='mel', n_bins=64)
criterion.to(device)

# Steps needed for loss and acc plotting in aim
train_step = 0
small_test_step = 0
test_step = 0

# After how many train batches to run small test
test_point = int(len(trainloader)*test_points[sch_state])

# Do not run small test when about to finish training
last_test_point = int(len(trainloader)*0.96)

# Number of batches to check in small test
small_test_size = int(len(testloader)*0.05)

# Do not run small test ff learning rate will not be reduced by more than scheduler EPS
run_small_test = (optimizer.param_groups[0]['lr']*(1-scheduler.factor) > scheduler.eps)

# Training Loop
print("\nStarting training loop...")
for epoch in range(run["hparams", "num_epochs"]):
    # Training phase
    tot_loss = 0
    tot_acc = 0
    avg_loss = 0
    avg_acc = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Train Epoch {epoch+1}")
    for i, (comp_wav, fileinfo) in pbar:
        # Set model to train state
        model.train()

        # Zero the gradients of all optimized variables. This is to ensure that we don't accumulate gradients from previous batches, as gradients are accumulated by default in PyTorch
        optimizer.zero_grad()

        # Forward pass
        comp_wav = model(comp_wav)

        # Get uncompressed waveform
        if(batch_size > 1):
            uncomp_wav = fileinfo
        else:
            fileinfo = (fileinfo[0][0], fileinfo[1].item())
            uncomp_wav = funct.process_wav(label_path, fileinfo, batch_size != 1, is_input=False)
            uncomp_wav = torch.unsqueeze(uncomp_wav, dim=0)

        # Convert input and label for loss calculation
        #comp_wav = funct.wav_to_mel_db(comp_wav)
        #uncomp_wav = funct.wav_to_mel_db(uncomp_wav)
        #comp_wav = funct.wav_to_pow_db(comp_wav)
        #uncomp_wav = funct.wav_to_pow_db(uncomp_wav)
        #comp_wav = funct.wav_to_db(comp_wav)
        #uncomp_wav = funct.wav_to_db(uncomp_wav)
        comp_wav = funct.upsample(comp_wav, uncomp_wav.shape[2])

        # Compute the loss value: this measures how well the predicted comp_wavs match the true labels
        loss = criterion(comp_wav, uncomp_wav)

        # Force stop if loss is nan
        if(loss != loss):
            sys.exit("Training loss is nan, force exiting")

        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()

        # Update the model's parameters using the optimizer's step method
        optimizer.step()

        # Calculate accuracy
        acc = torch.sum(torch.logical_and((comp_wav > uncomp_wav-acc_bound), (comp_wav < uncomp_wav+acc_bound)))/torch.numel(uncomp_wav)

        # Explicitly delete tensors so they don't stay in memory
        del comp_wav
        del uncomp_wav

        # Add loss and accuracy to totals
        tot_loss += loss.item()
        tot_acc += acc

        # Calculate average loss and accuracy so far
        avg_loss = tot_loss/(i+1)
        avg_acc = tot_acc/(i+1)

        # Log loss to aim
        run.track(loss, name='loss', step=train_step, epoch=epoch+1, context={ "subset":"train" })
        run.track(acc, name='acc', step=train_step, epoch=epoch+1, context={ "subset":"train" })

        # Update tqdm progress bar with fixed number of decimals for loss
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{avg_loss:.4f}", "Acc": f"{acc:.4f}", "Avg Acc": f"{avg_acc:.4f}"})

        # Increment step for next aim log
        train_step = train_step + 1

        # If training has completed for test_point number of batches
        if(save_partial_epoch):
            if(i%test_point == 0 and i < last_test_point and i > 0):
                if(run_small_test):
                    # Small testing phase
                    model.eval()
                    with torch.no_grad():
                        tot_test_loss = 0
                        tot_test_acc = 0
                        avg_test_loss = 0
                        avg_test_acc = 0
                        test_pbar = tqdm(enumerate(testloader), total=small_test_size, desc=f"Small Test Epoch {scheduler.last_epoch+1}")
                        for i, (comp_wav, fileinfo) in test_pbar:
                            # Stop small test after small_test_size batches
                            if(not i < small_test_size):
                                break
                            # Forward pass
                            comp_wav = model(comp_wav)

                            # Get uncompressed waveform
                            if(test_batch_size > 1):
                                uncomp_wav = fileinfo
                            else:
                                fileinfo = (fileinfo[0][0], fileinfo[1].item())
                                uncomp_wav = test_funct.process_wav(label_path, fileinfo, test_batch_size != 1, is_input=False)  
                                uncomp_wav = torch.unsqueeze(uncomp_wav, dim=0)

                            # Convert input and label for loss calculation
                            #comp_wav = torch.squeeze(comp_wav, dim=0)
                            #comp_wav = test_funct.wav_to_mel_db(comp_wav)
                            #uncomp_wav = test_funct.wav_to_mel_db(uncomp_wav)
                            #comp_wav = test_funct.wav_to_pow_db(comp_wav)
                            #uncomp_wav = test_funct.wav_to_pow_db(uncomp_wav)
                            #comp_wav = test_funct.wav_to_db(comp_wav)
                            #uncomp_wav = test_funct.wav_to_db(uncomp_wav)
                            comp_wav = funct.upsample(comp_wav, uncomp_wav.shape[2])

                            # Compute the loss value: this measures how well the predicted comp_wavs match the true labels
                            loss = criterion(comp_wav, uncomp_wav)

                            # Calculate accuracy 
                            acc = torch.sum(torch.logical_and((comp_wav > uncomp_wav-acc_bound), (comp_wav < uncomp_wav+acc_bound)))/torch.numel(uncomp_wav)

                            # Explicitly delete tensor so it doesn't stay in memory
                            del comp_wav
                            del uncomp_wav

                            # Add loss and accuracy to totals
                            tot_test_loss += loss.item()
                            tot_test_acc += acc

                            # Calculate average loss and accuracy so far
                            avg_test_loss = tot_test_loss/(i+1)
                            avg_test_acc = tot_test_acc/(i+1)

                            # Log loss to aim
                            run.track(loss, name='loss', step=small_test_step, epoch=epoch+1, context={ "subset":"small_test" })
                            run.track(acc, name='acc', step=small_test_step, epoch=epoch+1, context={ "subset":"small_test" })
                        
                            # Increment step for next aim log
                            small_test_step = small_test_step + 1

                            # Update tqdm progress bar with fixed number of decimals for loss
                            test_pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{avg_test_loss:.4f}", "Acc": f"{acc:.4f}", "Avg Acc": f"{avg_test_acc:.4f}"})

                        # Log average test loss and accuracy and last model output to Aim
                        step = scheduler.last_epoch + 1
                        run.track(avg_test_loss, name='avg_loss', step=step, epoch=epoch+1, context={ "subset":"small_test" })
                        run.track(avg_test_acc, name='avg_acc', step=step, epoch=epoch+1, context={ "subset":"small_test" })

                    # Send training loss to scheduler
                    scheduler.step(avg_test_loss)

                    # Check if learning rate was changed
                    if(optimizer.param_groups[0]['lr'] < learning_rate):
                        # Save current learning rate
                        learning_rate = optimizer.param_groups[0]['lr']
                        # Change scheduler parameters after learning rate reduction
                        if(sch_state < sch_change):
                            sch_state = sch_state + 1
                            scheduler.factor = factors[sch_state]
                            scheduler.patience = patiences[sch_state]
                            test_point = int(len(training_data)*test_points[sch_state])
                        # If learning rate will not be reduced by more than scheduler EPS, then do not run small test anymore
                        if(not (learning_rate*(1-scheduler.factor) > scheduler.eps)):
                            run_small_test = False

                # Save model and optimizer states
                torch.save(model.state_dict(), results_path+f"model{(epoch+1):02d}.pth")
                torch.save(optimizer.state_dict(), results_path+f"optimizer{(epoch+1):02d}.pth")
                torch.save(scheduler.state_dict(), results_path+f"scheduler{(epoch+1):02d}.pth")

    # Log average training loss to aim
    run.track(avg_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, context={ "subset":"train" })
    run.track(avg_acc, name='avg_acc', step=epoch+1, epoch=epoch+1, context={ "subset":"train" })

    # Full testing phase
    model.eval()
    with torch.no_grad():
        tot_test_loss = 0
        tot_test_acc = 0
        avg_test_loss = 0
        avg_test_acc = 0
        test_pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Test Epoch {epoch+1}")
        for i, (comp_wav, fileinfo) in test_pbar:
            # Forward pass
            comp_wav = model(comp_wav)

            # Get uncompressed waveform
            if(test_batch_size > 1):
                uncomp_wav = fileinfo
            else:
                fileinfo = (fileinfo[0][0], fileinfo[1].item())
                uncomp_wav = test_funct.process_wav(label_path, fileinfo, test_batch_size != 1, is_input=False)  
                uncomp_wav = torch.unsqueeze(uncomp_wav, dim=0)

            # Convert input and label for loss calculation
            #comp_wav = torch.squeeze(comp_wav, dim=0)
            #comp_wav = test_funct.wav_to_mel_db(comp_wav)
            #uncomp_wav = test_funct.wav_to_mel_db(uncomp_wav)
            #comp_wav = test_funct.wav_to_pow_db(comp_wav)
            #uncomp_wav = test_funct.wav_to_pow_db(uncomp_wav)
            #comp_wav = test_funct.wav_to_db(comp_wav)
            #uncomp_wav = test_funct.wav_to_db(uncomp_wav)
            comp_wav = test_funct.upsample(comp_wav, uncomp_wav.shape[2])

            # Compute the loss value: this measures how well the predicted comp_wavs match the true labels
            loss = criterion(comp_wav, uncomp_wav)

            # Calculate accuracy 
            acc = torch.sum(torch.logical_and((comp_wav > uncomp_wav-acc_bound), (comp_wav < uncomp_wav+acc_bound)))/torch.numel(uncomp_wav)

            # Step scheduler and log model output after small_test_size batches
            if(i == small_test_size-1 and save_partial_epoch):
                if(run_small_test):
                    scheduler.step(avg_test_loss)
                    # Check if learning rate was changed
                    if(optimizer.param_groups[0]['lr'] < learning_rate):
                        # Save current learning rate
                        learning_rate = optimizer.param_groups[0]['lr']
                        # Change scheduler parameters after learning rate reduction
                        if(sch_state < sch_change):
                            sch_state = sch_state + 1
                            scheduler.factor = factors[sch_state]
                            scheduler.patience = patiences[sch_state]
                            test_point = int(len(training_data)*test_points[sch_state])
                        # If learning rate will not be reduced by more than scheduler EPS, then do not run small test anymore
                        if(not (learning_rate*(1-scheduler.factor) > scheduler.eps)):
                            run_small_test = False

            # Explicitly delete tensors so they don't stay in memory
            del comp_wav
            del uncomp_wav

            # Add loss and accuracy to totals
            tot_test_loss += loss.item()
            tot_test_acc += acc

            # Calculate average loss and accuracy so far
            avg_test_loss = tot_test_loss/(i+1)
            avg_test_acc = tot_test_acc/(i+1)

            # Log loss to aim
            run.track(loss, name='loss', step=test_step, epoch=epoch+1, context={ "subset":"test" })
            run.track(acc, name='acc', step=test_step, epoch=epoch+1, context={ "subset":"test" })

            # Increment step for next aim log
            test_step = test_step + 1

            # Update tqdm progress bar with fixed number of decimals for loss
            test_pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Avg Loss": f"{avg_test_loss:.4f}", "Acc": f"{acc:.4f}", "Avg Acc": f"{avg_test_acc:.4f}"})

        # Log average test loss and accuracy and last model output to Aim
        run.track(avg_test_loss, name='avg_loss', step=epoch+1, epoch=epoch+1, context={ "subset":"test" })
        run.track(avg_test_acc, name='avg_acc', step=epoch+1, epoch=epoch+1, context={ "subset":"test" })
   
   # Step scheduler at the end of full test phase if not set to step in small tests
    if(not save_partial_epoch and run_small_test):
        scheduler.step(avg_test_loss)
        # Check if learning rate was changed
        if(optimizer.param_groups[0]['lr'] < learning_rate):
            # Save current learning rate
            learning_rate = optimizer.param_groups[0]['lr']
            # Change scheduler parameters after learning rate reduction
            if(sch_state < sch_change):
                sch_state = sch_state + 1
                scheduler.factor = factors[sch_state]
                scheduler.patience = patiences[sch_state]
                test_point = int(len(training_data)*test_points[sch_state])
            # If learning rate will not be reduced by more than scheduler EPS, then do not run small test anymore
            if(not (learning_rate*(1-scheduler.factor) > scheduler.eps)):
                run_small_test = False

    # Save model and optimizer states
    torch.save(model.state_dict(), results_path+f"model{(epoch+1):02d}.pth")
    torch.save(optimizer.state_dict(), results_path+f"optimizer{(epoch+1):02d}.pth")
    torch.save(scheduler.state_dict(), results_path+f"scheduler{(epoch+1):02d}.pth")

print("Finished Training")
