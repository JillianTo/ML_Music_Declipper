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
    def __init__(self, first_layer_channels=32):
        super(AutoEncoder, self).__init__()

        # Encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=first_layer_channels, kernel_size=(5, 5), stride=(2, 2)),
            #nn.BatchNorm2d(first_layer_channels),
            #nn.ReLU(),
            #nn.MaxPool2d((2,2))
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=first_layer_channels, out_channels=first_layer_channels*2, kernel_size=(3, 3), stride=(2, 2)),
            #nn.BatchNorm2d(first_layer_channels*2),
            #nn.ReLU(),
            #nn.MaxPool2d((2,2)),
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # First decoder layer 
        self.dec1 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=first_layer_channels*2, out_channels=first_layer_channels, kernel_size=(3, 3), stride=(2, 2)),
            #nn.ReLU()
        )

        # Decoder layer
        self.dec2 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=first_layer_channels, out_channels=2, kernel_size=(5, 5), stride=(2, 2)),
            #nn.Sigmoid()
        )

    def forward(self, x):
        real = self.enc1(x.real)
        imag = self.enc1(x.imag)
        real = self.dropout(real)
        imag = self.dropout(imag)
        real = self.dec2(real)
        imag = self.dec2(imag)
        x = combine_complex(real, imag)
        return x

class AutoEncoder2L(nn.Module):
    def __init__(self, first_layer_channels=16):
        super(AutoEncoder2L, self).__init__()

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=first_layer_channels, kernel_size=(5, 5), stride=(2, 2)),
            #nn.BatchNorm2d(first_layer_channels),
            #nn.ReLU(),
            #nn.MaxPool2d((2,2))
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=first_layer_channels, out_channels=first_layer_channels*2, kernel_size=(3, 3), stride=(2, 2)),
            #nn.BatchNorm2d(first_layer_channels*2),
            #nn.ReLU(),
            #nn.MaxPool2d((2,2)),
        )

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # First decoder layer 
        self.dec1 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=first_layer_channels*2, out_channels=first_layer_channels, kernel_size=(3, 3), stride=(2, 2)),
            #nn.ReLU()
        )

        # Second decoder layer
        self.dec2 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=first_layer_channels, out_channels=2, kernel_size=(5, 5), stride=(2, 2)),
            #nn.Sigmoid()
            #nn.Tanh()
        )

    def forward(self, x):
        real = self.enc1(x.real)
        imag = self.enc1(x.imag)
        real = self.enc2(real)
        imag = self.enc2(imag)
        real = self.dropout(real)
        imag = self.dropout(imag)
        real = self.dec1(real)
        imag = self.dec1(imag)
        real = self.dec2(real)
        imag = self.dec2(imag)
        x = combine_complex(real, imag)
        return x
