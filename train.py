import os
from aim import Run
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.io import StreamReader
import torchaudio.transforms as T
print("Torch version: " + torch.__version__)
print("TorchAudio version: " + torchaudio.__version__)

# Initialize a new run
run = Run(experiment="declipper")

# Log run parameters
run["hparams"] = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 3, 
    "expected_sample_rate": 44100,
    "uncompressed_data_path": "/mnt/Elements08/Music/ml/uncomp/wav/",
    "compressed_data_path": "/mnt/Elements08/Music/ml/comp/small/",
    "test_compressed_data_path": "/mnt/Elements08/Music/ml/comp/test/",
    "max_w": 17000000,
    "hidden_layer_size": 2
}

# Save run parameters to variables for easier access
max_w = run["hparams", "max_w"]

# Get CPU, GPU, or MPS device for training
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)
print(f"Using {device} device")

# Initialize input and label data to empty lists
inputs = []
labels = []
transform =  T.MelSpectrogram(run["hparams", "expected_sample_rate"])
print("\nLoading data")

# Load audio files into training data
for filename in os.listdir(run["hparams", "compressed_data_path"]):
    # Load wavs
    comp_wav_path = run["hparams", "compressed_data_path"] + filename
    comp_wav, comp_sample_rate = torchaudio.load(comp_wav_path)
    uncomp_wav_path = run["hparams", "uncompressed_data_path"] + filename.replace(".flac-compressed-", "")
    uncomp_wav, uncomp_sample_rate = torchaudio.load(uncomp_wav_path)

    # Resample if not expected sample rate
    if(comp_sample_rate != run["hparams", "expected_sample_rate"]):
        print(f"\"{wav_path}\" has sample rate of {sample_rate}Hz, resampling")
        resampler = T.Resample(comp_sample_rate, run["hparams", "expected_sample_rate"], dtype=waveform.dtype)
        comp_wav = resampler(comp_wav)

    if(uncomp_sample_rate != run["hparams", "expected_sample_rate"]):
        print(f"\"{uncomp_wav_path}\" has sample rate of {sample_rate}Hz, resampling")
        resampler = T.Resample(uncomp_sample_rate, run["hparams", "expected_sample_rate"], dtype=waveform.dtype)
        uncomp_wav = resampler(uncomp_wav)

    # pad compressed wav if short tensor
    if(comp_wav.shape[1] < max_w):
        w_padding = max_w - comp_wav.shape[1]
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        comp_wav = F.pad(comp_wav, (left_pad, right_pad))
    # crop long tensor
    elif(comp_wav.shape[1] > max_w):
        comp_wav = comp_wav[:max_w]

    # pad uncompressed wav if short tensor
    if(uncomp_wav.shape[1] < max_w):
        w_padding = max_w - uncomp_wav.shape[1]
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        uncomp_wav = F.pad(uncomp_wav, (left_pad, right_pad))
    # crop long tensor
    elif(uncomp_wav.shape[1] > max_w):
        uncomp_wav = uncomp_wav[:max_w]

    # transform wavs to Mel spectrograms, then reshape to [frequency_bins, channels, time_steps]
    comp_wav = transform(comp_wav)
    uncomp_wav = transform(uncomp_wav)

    # Add wavs to lists
    inputs.append(comp_wav)
    labels.append(uncomp_wav)

# Create dataset class
class AudioDataset(Dataset):
    def __init__(self, comp_wavs, uncomp_wavs):
        self.comp_wavs = comp_wavs
        self.uncomp_wavs = uncomp_wavs

    def __len__(self):
        return len(self.comp_wavs)
    
    def __getitem__(self, idx):
        return self.comp_wavs[idx], self.uncomp_wavs[idx]

# Add inputs and labels to training dataset
training_data = AudioDataset(inputs, labels)

# Clear inputs and labels lists to load test data into
inputs = []
labels = []

# Load audio files into testing data
for filename in os.listdir(run["hparams", "compressed_data_path"]):
    # Load wavs
    comp_wav_path = run["hparams", "compressed_data_path"] + filename
    comp_wav, comp_sample_rate = torchaudio.load(comp_wav_path)
    uncomp_wav_path = run["hparams", "uncompressed_data_path"] + filename.replace(".flac-compressed-", "")
    uncomp_wav, uncomp_sample_rate = torchaudio.load(uncomp_wav_path)

    # Resample if not expected sample rate
    if(comp_sample_rate != run["hparams", "expected_sample_rate"]):
        print(f"\"{wav_path}\" has sample rate of {sample_rate}Hz, resampling")
        resampler = T.Resample(comp_sample_rate, run["hparams", "expected_sample_rate"], dtype=waveform.dtype)
        comp_wav = resampler(comp_wav)

    if(uncomp_sample_rate != run["hparams", "expected_sample_rate"]):
        print(f"\"{uncomp_wav_path}\" has sample rate of {sample_rate}Hz, resampling")
        resampler = T.Resample(uncomp_sample_rate, run["hparams", "expected_sample_rate"], dtype=waveform.dtype)
        uncomp_wav = resampler(uncomp_wav)

    # pad compressed wav if short tensor
    if(comp_wav.shape[1] < max_w):
        w_padding = max_w - comp_wav.shape[1]
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        comp_wav = F.pad(comp_wav, (left_pad, right_pad))
    # crop long tensor
    elif(comp_wav.shape[1] > max_w):
        comp_wav = comp_wav[:max_w]

    # pad uncompressed wav if short tensor
    if(uncomp_wav.shape[1] < max_w):
        w_padding = max_w - uncomp_wav.shape[1]
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        uncomp_wav = F.pad(uncomp_wav, (left_pad, right_pad))
    # crop long tensor
    elif(uncomp_wav.shape[1] > max_w):
        uncomp_wav = uncomp_wav[:max_w]

    # transform wavs to Mel spectrograms, then reshape to [frequency_bins, channels, time_steps]
    comp_wav = transform(comp_wav)
    uncomp_wav = transform(uncomp_wav)

    # Add wavs to lists
    inputs.append(comp_wav)
    labels.append(uncomp_wav)

# Add inputs and labels to training dataset
test_data = AudioDataset(inputs, labels)

# Create data loaders
trainloader = DataLoader(training_data, batch_size=run["hparams", "batch_size"], shuffle=False)
testloader = DataLoader(test_data, batch_size=run["hparams", "batch_size"], shuffle=False)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # Encoder layers
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        # Decoder layers
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(5, 5), stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

# Initialize model, loss function, and optimizer
print("Initializing model, loss function, and optimizer...")
model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=run["hparams", "learning_rate"])

# Training Loop
print("Starting training loop...")
for epoch in range(run["hparams", "num_epochs"]):
    # Training phase
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Train Epoch {epoch+1}")
    for i, (comp_wav, uncomp_wav) in pbar:
        # Zero the gradients of all optimized variables. This is to ensure that we don't accumulate gradients from previous batches, as gradients are accumulated by default in PyTorch
        optimizer.zero_grad()
        # Forward pass
        output = model(comp_wav)
        # Interpolate uncomp_wav to compare to the output
        uncomp_wav = F.interpolate(uncomp_wav, size=(output.shape[2], output.shape[3]), mode='bilinear', align_corners=False)
        # Compute the loss value: this measures how well the predicted outputs match the true labels
        loss = criterion(output, uncomp_wav)
        # Backward pass: compute the gradient of the loss with respect to model parameters
        loss.backward()
        # Update the model's parameters using the optimizer's step method
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == uncomp_wav).sum().item()
        accuracy = correct / len(uncomp_wav)

        # Log loss and accuracy to aim
        run.track(loss, name='loss', step=10, epoch=epoch+1, context={ "subset":"train" })
        run.track(accuracy, name='acc', step=10, epoch=epoch+1, context={ "subset":"train" })

        # Update tqdm progress bar with fixed number of decimals for loss and accuracy
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{accuracy:.4f}"})

    # Testing phase
    model.eval()
    with torch.no_grad():
        mse = 0
        pbar = tqdm(enumerate(testloader), total=len(testloader), desc=f"Test Epoch {epoch+1}")
        for i, (comp_wav, uncomp_wav) in pbar:
            output = model(comp_wav)
            # Interpolate uncomp_wav to compare to the output
            uncomp_wav = F.interpolate(uncomp_wav, size=(output.shape[2], output.shape[3]), mode='bilinear', align_corners=False)
            mse += criterion(output, uncomp_wav)
        mse /= len(testloader)
        print(f"Mean Squared Error on testing set: {mse: .6f}")

print("Finished Training")
