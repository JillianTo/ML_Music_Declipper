import os
from aim import Run
from tqdm import tqdm
from audiodataset import AudioDataset
from autoencoder import AutoEncoder, upsample 
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
    "learning_rate": 0.001,
    "batch_size": 4,
    "num_epochs": 20, 
    "expected_sample_rate": 44100,
    "uncompressed_data_path": "/mnt/Elements08/Music/ml/uncomp/wav/",
    "compressed_data_path": "/mnt/Elements08/Music/ml/comp/small/",
    "test_compressed_data_path": "/mnt/Elements08/Music/ml/comp/test/",
    "max_time": 14000000,
    "num_workers": 0,
    "prefetch_factor": None,
    "pin_memory": False
}

# Save run parameters to variables for easier access
uncomp_path = run["hparams", "uncompressed_data_path"]
num_workers = run["hparams", "num_workers"]
prefetch_factor = run["hparams", "prefetch_factor"]
pin_memory = run["hparams", "pin_memory"]

# Get CPU, GPU, or MPS device for training
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
print(f"Using {device} device")

# Load wavs into training data
print("\nLoading data...")

# Add inputs and labels to training dataset
funct = Functional(run["hparams", "expected_sample_rate"], run["hparams", "max_time"])
training_data = AudioDataset(funct, run["hparams", "compressed_data_path"], uncomp_path, True)
print(f"Added {len(training_data)} file pairs to training data")

# Save original frequency bins and time steps values to send to AutoEncoder upsampler
orig_train_shape = [training_data[0][0].shape[1], training_data[0][0].shape[2]]

# Add inputs and labels to training dataset
test_data = AudioDataset(funct, run["hparams", "test_compressed_data_path"], uncomp_path, False)
print(f"Added {len(test_data)} file pairs to test data")

# Create data loaders
trainloader = DataLoader(training_data, batch_size=run["hparams", "batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
testloader = DataLoader(test_data, batch_size=run["hparams", "batch_size"], shuffle=False, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)

# Initialize model, loss function, and optimizer
print("\nInitializing model, loss function, and optimizer...")
model = AutoEncoder()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=run["hparams", "learning_rate"])

# Training Loop
print("Starting training loop...")
for epoch in range(run["hparams", "num_epochs"]):
    # Training phase
    tot_loss = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Train Epoch {epoch+1}")
    for i, (comp_wav, uncomp_wav) in pbar:
        # Send tensors to device
        comp_wav = comp_wav.to(device)
        uncomp_wav = uncomp_wav.to(device)
        # Zero the gradients of all optimized variables. This is to ensure that we don't accumulate gradients from previous batches, as gradients are accumulated by default in PyTorch
        optimizer.zero_grad()
        # Forward pass
        output = model(comp_wav)
        # Upsample output to match uncompressed wav
        output = upsample(output, [uncomp_wav.shape[2], uncomp_wav.shape[3]])
        # Compute the loss value: this measures how well the predicted outputs match the true labels
        real_loss = criterion(output.real, uncomp_wav.real)
        imag_loss = criterion(output.imag, uncomp_wav.imag)
        loss = (real_loss+imag_loss)/2
        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()
        # Update the model's parameters using the optimizer's step method
        optimizer.step()

        # Log loss to aim
        run.track(loss, name='loss', step=10, epoch=epoch+1, context={ "subset":"train" })

        # Update tqdm progress bar with fixed number of decimals for loss and accuracy
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # Testing phase
    model.eval()
    with torch.no_grad():
        tot_loss = 0
        pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Test Epoch {epoch+1}")
        for i, (comp_wav, uncomp_wav) in pbar:
            # Send tensors to device
            comp_wav = comp_wav.to(device)
            uncomp_wav = uncomp_wav.to(device)

            output = model(comp_wav)
            output = upsample(output, [uncomp_wav.shape[2], uncomp_wav.shape[3]])

            real_loss = criterion(output.real, uncomp_wav.real)
            imag_loss = criterion(output.imag, uncomp_wav.imag)
            loss = (real_loss+imag_loss)/2

            # Log loss to aim
            run.track(loss, name='loss', step=10, epoch=epoch+1, context={ "subset":"test" })

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

print("Finished Training")
torch.save(model.state_dict(), './model.pth') 
