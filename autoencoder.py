import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torchaudio.functional as F
import torchaudio.transforms as T

class SpecAutoEncoder(nn.Module):
    def __init__(self, device, mean, std, n_fft=4096, in_channels=2, first_out_channels=128, kernel_size=(3,3), up_kernel_size=(2,2), stride=(1,1), up_stride=(1,1), dropout=0.2):
        super(SpecAutoEncoder, self).__init__(),

        self.device=device
        self.mean=mean
        self.std=std
        self.n_fft=n_fft

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=first_out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels, out_channels=first_out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels, out_channels=first_out_channels<<1, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<1, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<1, out_channels=first_out_channels<<1, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<1, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Third encoder layer
        self.enc3 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<1, out_channels=first_out_channels<<2, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<2, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<2, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<2, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Fourth encoder layer
        self.enc4 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<3, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<3, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<3, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<3, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Fifth encoder layer
        self.enc5 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<4, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<4, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<4, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<4, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Sixth encoder layer
        self.enc6 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<5, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<5, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<5, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<5, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Seventh encoder layer
        self.enc7 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<6, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<6, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<6, out_channels=first_out_channels<<6, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<6, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Tanh activation
        self.tanh = nn.Tanh()

        # Set LSTM channels to output channels of last used encoder layer in forward
        lstm_channels=first_out_channels<<4

        # Long-short term memory
        self.lstm = nn.Sequential(
            nn.LayerNorm(lstm_channels),
            nn.LSTM(input_size=lstm_channels, hidden_size=lstm_channels>>1, num_layers=3, batch_first=True, dropout=dropout, bidirectional=True),
        )

        # First decoder layer 
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=lstm_channels, out_channels=lstm_channels>>1, kernel_size=up_kernel_size, stride=up_stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<4, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Second decoder layer 
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<6, out_channels=first_out_channels<<5, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<5, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<5, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<5, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<4, kernel_size=up_kernel_size, stride=up_stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<4, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Second decoder layer 
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<4, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<4, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<4, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<4, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<3, kernel_size=up_kernel_size, stride=up_stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<3, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Third decoder layer 
        self.dec4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<3, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<3, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<3, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<3, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<2, kernel_size=up_kernel_size, stride=up_stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<2, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Fourth decoder layer 
        self.dec5 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<2, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<2, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<2, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<2, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<1, kernel_size=up_kernel_size, stride=up_stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<1, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )
        
        # Fifth decoder layer
        self.dec6 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<1, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<1, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels<<1, out_channels=first_out_channels<<1, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels<<1, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<1, out_channels=first_out_channels, kernel_size=up_kernel_size, stride=up_stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
        )

        # Sixth decoder layer
        self.dec7 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, out_channels=first_out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU(),
            nn.Conv2d(in_channels=first_out_channels, out_channels=first_out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.InstanceNorm2d(first_out_channels, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU()
            nn.Conv2d(in_channels=first_out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=1),
            ##nn.InstanceNorm2d(in_channels, affine=affine, track_running_stats=track_running_stats),
            #nn.ReLU()
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(1,1), stride=(1,1), padding=1),
        )

    def forward(self, x):
        # Save time shape of unedited input for upsampling later
        upsample = nn.Upsample(x.shape[2])
        upsample = upsample.to(self.device)

        # Convert input to complex spectrogram
        complex_spec = T.Spectrogram(n_fft=self.n_fft, power=None)
        complex_spec = complex_spec.to(self.device)
        x = complex_spec(x)

        # Calculate phase of complex spectrogram
        phase = torch.atan(x.imag/(x.real+1e-7))
        phase[x.real < 0] += 3.14159265358979323846264338

        # Calculate magnitude of complex spectrogram
        x = torch.sqrt(torch.pow(x.real, 2)+torch.pow(x.imag,2))
        amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=80)
        amp_to_db = amp_to_db.to(self.device)
        x = amp_to_db(x)

        # Standardize magnitude
        x = (x-self.mean)/self.std

        # Save encoder layers for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        x = self.enc5(e4)
        #x = self.enc6(e5)
        #x = self.enc7(e6)

        # Tanh activation before going into LSTM
        x = self.tanh(x)

        # Flatten encoder output and rearrange to use with LSTM
        freq_dim = x.size(2)
        time_dim = x.size(3)
        x = x.reshape(x.size(0), freq_dim*time_dim, x.size(1))
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), x.size(2), freq_dim, time_dim)

        # Decoder
        x = self.dec1(x)
        #dec_upsample = nn.Upsample(size=[e6.shape[2], e6.shape[3]])
        #dec_upsample = dec_upsample.to(self.device)
        #x = dec_upsample(x)
        #x = torch.cat((x, e6), dim=1)
        #x = self.dec2(x)
        #dec_upsample = nn.Upsample(size=[e5.shape[2], e5.shape[3]])
        #dec_upsample = dec_upsample.to(self.device)
        #x = dec_upsample(x)
        #x = torch.cat((x, e5), dim=1)
        #x = self.dec3(x)
        dec_upsample = nn.Upsample(size=[e4.shape[2], e4.shape[3]])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e4), dim=1)
        x = self.dec4(x)
        dec_upsample = nn.Upsample(size=[e3.shape[2], e3.shape[3]])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e3), dim=1)
        x = self.dec5(x)
        dec_upsample = nn.Upsample(size=[e2.shape[2], e2.shape[3]])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e2), dim=1)
        x = self.dec6(x)
        dec_upsample = nn.Upsample(size=[e1.shape[2], e1.shape[3]])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e1), dim=1)
        x = self.dec7(x)
        
        # Upscale magnitude to match phase shape
        mag_upsample = nn.Upsample([phase.shape[2], phase.shape[3]])
        mag_upsample = mag_upsample.to(self.device)
        x = mag_upsample(x)

        # Unnormalize magnitude
        x = (x*self.std)+self.mean

        # Convert magnitude back to linear amplitude
        x = F.DB_to_amplitude(x, 1, 0.5)

        x = torch.polar(x, phase)
        inv_spec = T.InverseSpectrogram(n_fft=self.n_fft)
        inv_spec = inv_spec.to(self.device)
        x = inv_spec(x)
        x = upsample(x)
        return x

class WavAutoEncoder(nn.Module):
    def __init__(self, device, in_channels=2, first_out_channels=64, enc_kernel_size_1=8, enc_kernel_size_2=1, dec_kernel_size=1, stride_1=4, stride_2=1, dropout=0.2):
        super(WavAutoEncoder, self).__init__()

        self.device = device

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=first_out_channels, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels, out_channels=first_out_channels<<1, kernel_size=enc_kernel_size_2, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
        )

        # Max pooling layer
        self.max_pool = nn.MaxPool1d(2)

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels, out_channels=first_out_channels<<1, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<1, out_channels=first_out_channels<<2, kernel_size=enc_kernel_size_2, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
        )

        # Third encoder layer
        self.enc3 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<1, out_channels=first_out_channels<<2, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<3, kernel_size=enc_kernel_size_2, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
        )

        # Fourth encoder layer
        self.enc4 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<3, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<4, kernel_size=enc_kernel_size_2, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
        )

        # Fifth encoder layer
        self.enc5 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<4, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<5, kernel_size=enc_kernel_size_2, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
        )

        # Sixth encoder layer
        self.enc6 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<5, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<6, kernel_size=enc_kernel_size_2, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
        )

        # Tanh activation
        self.tanh = nn.Tanh()

        # Set LSTM channels to output channels of last used encoder layer in forward
        lstm_channels=first_out_channels<<5

        # Long-short term memory
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=lstm_channels, hidden_size=lstm_channels, num_layers=2, batch_first=True, dropout=dropout, bidirectional=True),
        )

        # Linear layer 
        self.linear = nn.Linear(in_features=lstm_channels<<1, out_features=lstm_channels)

        # First decoder layer
        self.dec1 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<6, out_channels=first_out_channels<<5, kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
            nn.ConvTranspose1d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<4, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
        )

        # Second decoder layer
        self.dec2 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<5, out_channels=first_out_channels<<4, kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
            nn.ConvTranspose1d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<3, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
        )

        # Third decoder layer
        self.dec3 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<4, out_channels=first_out_channels<<3, kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
            nn.ConvTranspose1d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<2, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
        )

        # Fourth decoder layer
        self.dec4 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<3, out_channels=first_out_channels<<2, kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
            nn.ConvTranspose1d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<1, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
        )

        # Fifth decoder layer
        self.dec5 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<2, out_channels=first_out_channels<<1, kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
            nn.ConvTranspose1d(in_channels=first_out_channels<<1, out_channels=first_out_channels, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
            #nn.ReLU(),
        )

        # Sixth decoder layer
        self.dec6 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<1, out_channels=first_out_channels, kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            #nn.GLU(dim=1),
            nn.ConvTranspose1d(in_channels=first_out_channels, out_channels=in_channels, kernel_size=enc_kernel_size_1, stride=stride_1, padding=1),
        )

    def forward(self, x):
        upsample = nn.Upsample(x.shape[2])

        # Encoder
        e1 = self.enc1(x)
        e1 = e1.reshape(e1.size(0), e1.size(2), e1.size(1))
        e1 = self.max_pool(e1)
        e1 = e1.reshape(e1.size(0), e1.size(2), e1.size(1))
        e2 = self.enc2(e1)
        e2 = e2.reshape(e2.size(0), e2.size(2), e2.size(1))
        e2 = self.max_pool(e2)
        e2 = e2.reshape(e2.size(0), e2.size(2), e2.size(1))
        e3 = self.enc3(e2)
        e3 = e3.reshape(e3.size(0), e3.size(2), e3.size(1))
        e3 = self.max_pool(e3)
        e3 = e3.reshape(e3.size(0), e3.size(2), e3.size(1))
        e4 = self.enc4(e3)
        e4 = e4.reshape(e4.size(0), e4.size(2), e4.size(1))
        e4 = self.max_pool(e4)
        e4 = e4.reshape(e4.size(0), e4.size(2), e4.size(1))
        e5 = self.enc5(e4)
        e5 = e5.reshape(e5.size(0), e5.size(2), e5.size(1))
        e5 = self.max_pool(e5)
        e5 = e5.reshape(e5.size(0), e5.size(2), e5.size(1))
        e6 = self.enc6(e5)
        e6 = e6.reshape(e6.size(0), e6.size(2), e6.size(1))
        e6 = self.max_pool(e6)
        e6 = e6.reshape(e6.size(0), e6.size(2), e6.size(1))

        # BLSTM
        x = self.tanh(e6)
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x, _ = self.lstm(x)

        # Linear layer
        x = self.linear(x)
        x = x.reshape(x.size(0), x.size(2), x.size(1))

        # Decoder 
        x = torch.cat((x, e6), dim=1)
        x = self.dec1(x)
        dec_upsample = nn.Upsample(size=e5.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e5), dim=1)
        x = self.dec2(x)
        dec_upsample = nn.Upsample(size=e4.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e4), dim=1)
        x = self.dec3(x)
        dec_upsample = nn.Upsample(size=e3.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e3), dim=1)
        x = self.dec4(x)
        dec_upsample = nn.Upsample(size=e2.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e2), dim=1)
        x = self.dec5(x)
        dec_upsample = nn.Upsample(size=e1.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = torch.cat((x, e1), dim=1)
        x = self.dec6(x)

        # Upsample input to original time length
        x = upsample(x)

        return x
