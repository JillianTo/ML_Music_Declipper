from autoencoder import AutoEncoder
model = AutoEncoder(mean=-1, std=15, n_ffts=[8190], 
                    hop_lengths=256, 
                    sample_rate=44100)
for p in model.parameters():
    print(p.numel())

total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

