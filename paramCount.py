from autoencoder import AutoEncoder

model = AutoEncoder(mean=-1, std=15, n_ffts=[8190], hop_lengths=256, 
                    sample_rate=44100)

# Print individual layer parameters
for p in model.parameters():
    print(p.numel())

# Print total number of parameters in model
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

