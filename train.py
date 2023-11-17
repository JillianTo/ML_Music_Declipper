import os
import sys
from aim import Run
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
    "learning_rate": 0.00001,
    "batch_size": 2,
    "test_batch_size": 1,
    "num_epochs": 1000, 
    "expected_sample_rate": 44100,
    "uncompressed_data_path": "/mnt/MP600/data/uncomp/",
    #"compressed_data_path": "/mnt/MP600/data/comp/train/",
    "compressed_data_path": "/mnt/MP600/data/comp/small/train/",
    #"test_compressed_data_path": "/mnt/MP600/data/comp/test/",
    "test_compressed_data_path": "/mnt/MP600/data/comp/small/test/",
    "results_path": "/home/jto/Documents/AIDeclip/results/",
    #"max_time": 14000000,
    #"max_time": 284700,
    "max_time": 1500000,
    "test_max_time": 3500000,
    "num_workers": 0,
    "prefetch_factor": None,
    "pin_memory": False,
    "weight_decay": 0,
    "spectrogram_autoencoder": True,
    #"preload_weights_path": "/home/jto/Documents/AIDeclip/results/spece/model07.pth"
    "preload_weights_path": None
}

# Save run parameters to variables for easier access
learning_rate = run["hparams", "learning_rate"]
sample_rate = run["hparams", "expected_sample_rate"]
batch_size = run["hparams", "batch_size"]
test_batch_size = run["hparams", "test_batch_size"]
uncomp_path = run["hparams", "uncompressed_data_path"]
results_path = run["hparams", "results_path"]
num_workers = run["hparams", "num_workers"]
prefetch_factor = run["hparams", "prefetch_factor"]
pin_memory = run["hparams", "pin_memory"]
preload_weights_path = run["hparams", "preload_weights_path"]
weight_decay = run["hparams", "weight_decay"]

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
training_data = AudioDataset(funct, run["hparams", "compressed_data_path"], uncomp_path, True)
print(f"Added {len(training_data)} file pairs to training data")

# Add inputs and labels to training dataset
test_funct = Functional(sample_rate, run["hparams", "test_max_time"], device)
#test_data = AudioDataset(funct, run["hparams", "test_compressed_data_path"], uncomp_path, not(batch_size != 1 and test_batch_size==1))
test_data = AudioDataset(test_funct, run["hparams", "test_compressed_data_path"], uncomp_path, True)
print(f"Added {len(test_data)} file pairs to test data")

# Create data loaders
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn', force=True)
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    testloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

# Initialize model, loss function, and optimizer
print("\nInitializing model, loss function, and optimizer...")
if run["hparams", "spectrogram_autoencoder"]:
    model = SpecAutoEncoder(device)
    print("Using spectrogram autoencoder")
else:
    model = WavAutoEncoder()
    print("Using waveform autoencoder")
if preload_weights_path != None:
    model.load_state_dict(torch.load(preload_weights_path))
    print(f"Preloading weights from \"{preload_weights_path}\"")

model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training Loop
print("Starting training loop...")
last_avg_loss = 9999
for epoch in range(run["hparams", "num_epochs"]):
    # Training phase
    model.train()
    tot_loss = 0
    avg_loss = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Train Epoch {epoch+1}")
    for i, (comp_wav, uncomp_wav) in pbar:
        # Zero the gradients of all optimized variables. This is to ensure that we don't accumulate gradients from previous batches, as gradients are accumulated by default in PyTorch
        optimizer.zero_grad()

        # Forward pass
        output = model(comp_wav)

        # Convert output and uncompressed wav to Mel spectrograms for loss calculation
        #output = funct.wav_to_mel_db(output)
        #uncomp_wav = funct.wav_to_mel_db(uncomp_wav)
        output = funct.wav_to_pow_db(output)
        uncomp_wav = funct.wav_to_pow_db(uncomp_wav)
        #output = funct.wav_to_db(output)
        #uncomp_wav = funct.wav_to_db(uncomp_wav)
        #output = funct.upsample(output, uncomp_wav.shape[2])

        # Compute the loss value: this measures how well the predicted outputs match the true labels
        loss = criterion(output, uncomp_wav)

        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()

        # Update the model's parameters using the optimizer's step method
        optimizer.step()

        # Add loss to total loss
        tot_loss += loss.item()

        # Calculate average loss so far
        avg_loss = tot_loss/(i+1)

        # Log loss to aim
        run.track(loss, name='loss', step=0.1, epoch=epoch+1, context={ "subset":"train" })

        # Update tqdm progress bar with fixed number of decimals for loss
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Average Loss": f"{avg_loss:.4f}"})

        # Explicitly delete tensors so they don't stay in used memory
        del uncomp_wav
        del output

        # Log average loss to aim
        run.track(avg_loss, name='avg_loss', step=0.1, epoch=epoch+1, context={ "subset":"test" })

    # Check for nan training loss
    if(avg_loss != avg_loss):
        print("Average training loss is nan, force quitting")
        break

    # Testing phase
    model.eval()
    with torch.no_grad():
        tot_loss = 0
        avg_loss = 0
        acc = 0
        pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Test Epoch {epoch+1}")
        for i, (comp_wav, uncomp_wav) in pbar:
            # Forward pass
            output = model(comp_wav)

            # Convert output and uncompressed wav to Mel spectrograms for loss calculation
            #output = funct.wav_to_mel_db(output)
            #uncomp_wav = funct.wav_to_mel_db(uncomp_wav)
            output = funct.wav_to_pow_db(output)
            uncomp_wav = funct.wav_to_pow_db(uncomp_wav)
            #output = funct.wav_to_db(output)
            #uncomp_wav = funct.wav_to_db(uncomp_wav)
            #output = funct.upsample(output, uncomp_wav.shape[2])

            # Compute the loss value: this measures how well the predicted outputs match the true labels
            loss = criterion(output, uncomp_wav)

            # Add loss to total loss
            tot_loss += loss.item()

            # Calculate average loss so far
            avg_loss = tot_loss/(i+1)

            # Log loss to aim
            run.track(loss, name='loss', step=0.1, epoch=epoch+1, context={ "subset":"test" })
        
            # Update tqdm progress bar with fixed number of decimals for loss
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Average Loss": f"{avg_loss:.4f}"})

            # Explicitly delete tensors so they don't stay in used memory
            del uncomp_wav
            del output

        # Log average loss to aim
        run.track(avg_loss, name='avg_loss', step=0.1, epoch=epoch+1, context={ "subset":"test" })

    # Reduced learning rate or stop training if average validation loss increased
    if(avg_loss > last_avg_loss):
        if(prev_model != None):
            learning_rate = learning_rate/10
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            print(f"Validation loss increased, reducing learning rate to {learning_rate} and loading \"{prev_model}\"")
            model.load_state_dict(torch.load(prev_model))
            prev_model=None
        else:
            print("Validation loss increased, force quitting training")
            break
    else:
        # Save average loss
        last_avg_loss = avg_loss

        # Save model after every epoch with reduced validation loss
        prev_model = results_path+f"model{(epoch+1):02d}.pth"
        torch.save(model.state_dict(), prev_model)

print("Finished Training")
