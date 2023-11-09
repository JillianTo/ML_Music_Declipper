import torch
import torch.nn as nn

class SpecAutoEncoder(nn.Module):
    def __init__(self, n_fft=1024, in_channels=2, first_out_channels=16, kernel_size_1=(5,5), kernel_size_2=(3,3), kernel_size_3=(3,3), stride_1=(2,2), stride_2=(3,3), stride_3=(3,3), dropout_p=0.5):
        super(SpecAutoEncoder, self).__init__()

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=first_out_channels, kernel_size=kernel_size_1, stride=stride_1),
            nn.BatchNorm2d(first_out_channels),
            #nn.Tanh(),
            #nn.MaxPool2d((2,2))
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels, out_channels=first_out_channels<<1, kernel_size=kernel_size_2, stride=stride_2),
            nn.BatchNorm2d(first_out_channels<<1),
            #nn.Tanh(),
            #nn.MaxPool2d((2,2)),
        )

        # Third encoder layer
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, out_channels=first_out_channels<<2, kernel_size=kernel_size_3, stride=stride_3),
            nn.BatchNorm2d(first_out_channels*4),
            #nn.Tanh(),
            #nn.MaxPool2d((2,2)),
        )

        # Dropout 
        #self.dropout = nn.Dropout(dropout_p)
        #self.dropout_imag = nn.Dropout(dropout_p)

        # First decoder layer 
        self.dec1 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<1, kernel_size=kernel_size_3, stride=stride_3),
            nn.BatchNorm2d(first_out_channels<<1),
            #nn.Tanh(),
        )

        # Second decoder layer 
        self.dec2 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=first_out_channels<<1, out_channels=first_out_channels, kernel_size=kernel_size_2, stride=stride_2),
            nn.BatchNorm2d(first_out_channels),
            #nn.Tanh(),
        )

        # Third decoder layer
        self.dec3 = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=first_out_channels, out_channels=in_channels, kernel_size=kernel_size_1, stride=stride_1),
            nn.BatchNorm2d(in_channels),
            #nn.Tanh(),
        )

        # Upsample
        self.upsample = nn.Upsampe

    def forward(self, x):
        upsample = nn.Upsample(x.shape[2])

        complex_spec = T.Spectrogram(n_fft=self.n_fft, power=None)
        x = complex_spec(x)

        magnitude = torch.sqrt(torch.pow(x.real, 2)+torch.pow(x.imag,2))
        phase = torch.atan(x.imag/x.real)
        del x

        magnitude = self.enc1(magnitude)
        magnitude = self.enc2(magnitude)
        magnitude = self.enc3(magnitude)

        #magnitude = self.dropout(magnitude)

        magnitude = self.dec1(magnitude)
        magnitude = self.dec2(magnitude)
        magnitude = self.dec3(magnitude)

        x = torch.polar(magnitude, phase)
        inv_spec = T.InverseSpectrogram(n_fft=self.n_fft)
        x = upsample(inv_spec(x))
        return x

class WavAutoEncoder(nn.Module):
    def __init__(self, in_channels=2, kernel_size=2, stride=1, dropout_p=0.5):
        super(WavAutoEncoder, self).__init__()

        self.enc1 = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels<<4, kernel_size=kernel_size, stride=stride),
                nn.BatchNorm1d(in_channels<<4),
                nn.Tanh(),
        )
        # Dropout 
        #self.dropout = nn.Dropout(dropout_p)

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels<<4, out_channels=in_channels<<3, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(in_channels<<3),
            nn.Tanh(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels<<3, out_channels = in_channels<<2, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(in_channels<<2),
            nn.Tanh(),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels<<2, out_channels = in_channels<<1, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(in_channels<<1),
            nn.Tanh(),
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_channels<<1, out_channels = in_channels, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(in_channels),
            nn.Tanh(),
        )

    def forward(self, x):
        upsample = nn.Upsample(x.shape[2])

        # Encoder
        x = self.enc1(x)

        #real = self.dropout(real)
        #imag = self.dropout_imag(imag)

        # Decoder 
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        # Upsample input to original time length
        x = upsample(x)

        return x
