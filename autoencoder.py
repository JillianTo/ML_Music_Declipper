import torch
import torch.nn as nn

## Credit: https://github.com/pytorch/pytorch/issues/52983
def combine_complex(r, i):
    return torch.view_as_complex(torch.cat([r.unsqueeze(-1), i.unsqueeze(-1)], dim=-1))

def upsample(tensor, input_size):
    upsample = nn.Upsample(input_size)
    real = upsample(tensor.real)
    imag = upsample(tensor.imag)
    return combine_complex(real, imag)

class AutoEncoder(nn.Module):
    def __init__(self, enc_channels=16):
        super(AutoEncoder, self).__init__()

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=enc_channels, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU()
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=enc_channels, out_channels=enc_channels*2, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU()
        )

        # First decoder layer 
        self.fc1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=enc_channels*2, out_channels=enc_channels, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU()
        )

        # Second fully-connected layer
        self.fc2 = nn.ConvTranspose2d(in_channels=enc_channels, out_channels=2, kernel_size=(5, 5), stride=(2, 2))

    def forward(self, x):
        real = self.enc1(x.real)
        imag = self.enc1(x.imag)
        real = self.enc2(real)
        imag = self.enc2(imag)
        real = self.fc1(real)
        imag = self.fc1(imag)
        real = self.fc2(real)
        imag = self.fc2(imag)
        x = combine_complex(real, imag)
        return x
