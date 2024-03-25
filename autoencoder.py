import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
import torchaudio.functional as F
import torchaudio.transforms as T

class SpecAutoEncoder(nn.Module):
    def __init__(self, mean, std, n_fft=4096, hop_length=512, 
                 in_channels=2, first_out_channels=64, kernel_size=(3,3), 
                 up_kernel_size=(2,2), stride=(1,1), up_stride=(1,1), 
                 dropout=0.2):
        super(SpecAutoEncoder, self).__init__(),

        self.mean = mean
        self.std = std
        self.n_fft = int(n_fft)
        self.hop_length = hop_length

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )

        # Third encoder layer
        self.enc3 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )

        # Fourth encoder layer
        self.enc4 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, 
                      stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )

        # Fifth encoder layer
        self.enc5 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )

        # Sixth encoder layer
        self.enc6 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )

        # Seventh encoder layer
        self.enc7 = nn.Sequential(
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<6, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )

        # Tanh activation
        self.tanh = nn.Tanh()

        # Set LSTM channels to output channels of last used encoder layer 
        lstm_channels=first_out_channels<<4

        # Long-short term memory
        self.lstm = nn.Sequential(
            nn.LayerNorm(lstm_channels),
            nn.LSTM(input_size=lstm_channels, hidden_size=lstm_channels>>1, 
                    num_layers=3, batch_first=True, dropout=dropout, 
                    bidirectional=True),
        )

        # First decoder layer 
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=lstm_channels, 
                      out_channels=lstm_channels>>1, 
                      kernel_size=up_kernel_size, stride=up_stride, padding=1),
        )

        # Second decoder layer 
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<6, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=up_kernel_size, stride=up_stride, padding=1),
        )

        # Second decoder layer 
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=up_kernel_size, stride=up_stride, padding=1),
        )

        # Third decoder layer 
        self.dec4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=up_kernel_size, stride=up_stride, padding=1),
        )

        # Fourth decoder layer 
        self.dec5 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=up_kernel_size, stride=up_stride, padding=1),
        )
        
        # Fifth decoder layer
        self.dec6 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=up_kernel_size, stride=up_stride, padding=1),
        )

        # Sixth decoder layer
        self.dec7 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=in_channels, kernel_size=kernel_size, 
                      stride=stride, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=(1,1), stride=(1,1), padding=1),
        )

    def forward(self, x):
        device = x.device

        # Save time shape of unedited input for upsampling later
        upsample = nn.Upsample(x.shape[2])
        upsample = upsample.to(device)

        # Convert input to complex spectrogram
        complex_spec = T.Spectrogram(n_fft=self.n_fft,
                                     hop_length=self.hop_length, power=None)
        complex_spec = complex_spec.to(device)
        x = complex_spec(x)

        # Calculate phase of complex spectrogram
        phase = torch.atan(x.imag/(x.real+1e-7))
        phase[x.real < 0] += 3.14159265358979323846264338

        # Calculate magnitude of complex spectrogram
        x = torch.sqrt(torch.pow(x.real, 2)+torch.pow(x.imag,2))
        amp_to_db = T.AmplitudeToDB(stype='amplitude')
        amp_to_db = amp_to_db.to(device)
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
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, e4), dim=1)
        x = self.dec4(x)
        dec_upsample = nn.Upsample(size=[e3.shape[2], e3.shape[3]])
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, e3), dim=1)
        x = self.dec5(x)
        dec_upsample = nn.Upsample(size=[e2.shape[2], e2.shape[3]])
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, e2), dim=1)
        x = self.dec6(x)
        dec_upsample = nn.Upsample(size=[e1.shape[2], e1.shape[3]])
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, e1), dim=1)
        x = self.dec7(x)
        
        # Upscale magnitude to match phase shape
        mag_upsample = nn.Upsample([phase.shape[2], phase.shape[3]])
        mag_upsample = mag_upsample.to(device)
        x = mag_upsample(x)

        # Unnormalize magnitude
        x = (x*self.std)+self.mean

        # Convert magnitude back to linear amplitude
        x = F.DB_to_amplitude(x, 1, 0.5)

        x = torch.polar(x, phase)
        inv_spec = T.InverseSpectrogram(n_fft=self.n_fft, 
                                        hop_length=self.hop_length)
        inv_spec = inv_spec.to(device)
        x = inv_spec(x)

        # Set output to same time length as original input
        x = upsample(x)
        
        # Return modified waveform
        return x

# TODO: Needs work
class WavAutoEncoder(nn.Module):
    def __init__(self, device, in_channels=2, first_out_channels=64, 
                 enc_kernel_size_1=8, enc_kernel_size_2=1, dec_kernel_size=1, 
                 stride_1=4, stride_2=1, dropout=0.2):
        super(WavAutoEncoder, self).__init__()

        self.device = device

        # First encoder layer
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=first_out_channels, 
                      kernel_size=enc_kernel_size_1, stride=stride_1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=enc_kernel_size_2, stride=stride_2, 
                      padding=1),
            nn.GLU(dim=1),
        )

        # Second encoder layer
        self.enc2 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=enc_kernel_size_1, stride=stride_1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=enc_kernel_size_2, stride=stride_2, 
                      padding=1),
            nn.GLU(dim=1),
        )

        # Third encoder layer
        self.enc3 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=enc_kernel_size_1, stride=stride_1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=enc_kernel_size_2, stride=stride_2, 
                      padding=1),
            nn.GLU(dim=1),
        )

        # Fourth encoder layer
        self.enc4 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=enc_kernel_size_1, stride=stride_1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=enc_kernel_size_2, stride=stride_2, 
                      padding=1),
            nn.GLU(dim=1),
        )

        # Fifth encoder layer
        self.enc5 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=enc_kernel_size_1, stride=stride_1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=enc_kernel_size_2, stride=stride_2, 
                      padding=1),
            nn.GLU(dim=1),
        )

        # Sixth encoder layer
        self.enc6 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=enc_kernel_size_1, stride=stride_1, 
                      padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=enc_kernel_size_2, stride=stride_2, 
                      padding=1),
            nn.GLU(dim=1),
        )

        # Set LSTM channels to output channels of last used encoder layer
        lstm_channels=first_out_channels<<3

        # Long-short term memory
        self.lstm = nn.Sequential(
            nn.LSTM(input_size=lstm_channels, hidden_size=lstm_channels, 
                    num_layers=2, batch_first=True, dropout=dropout, 
                    bidirectional=True),
        )

        # Linear layer 
        self.linear = nn.Linear(in_features=lstm_channels<<1, 
                                out_features=lstm_channels)

        # First decoder layer
        self.dec1_1 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            nn.GLU(dim=1),
        )
        self.dec1_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=first_out_channels<<5, 
                               out_channels=first_out_channels<<4, 
                               kernel_size=enc_kernel_size_1, stride=stride_1, 
                               padding=1),
            nn.ReLU(),
        )

        # Second decoder layer
        self.dec2_1 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            nn.GLU(dim=1),
        )
        self.dec2_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=first_out_channels<<4, 
                               out_channels=first_out_channels<<3, 
                               kernel_size=enc_kernel_size_1, 
                               stride=stride_1, padding=1),
            nn.ReLU(),
        )

        # Third decoder layer
        self.dec3_1 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            nn.GLU(dim=1),
        )
        self.dec3_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=first_out_channels<<3, 
                               out_channels=first_out_channels<<2, 
                               kernel_size=enc_kernel_size_1, stride=stride_1, 
                               padding=1),
            nn.ReLU(),
        )

        # Fourth decoder layer
        self.dec4_1 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            nn.GLU(dim=1),
        )
        self.dec4_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=first_out_channels<<2, 
                               out_channels=first_out_channels<<1, 
                               kernel_size=enc_kernel_size_1, stride=stride_1, 
                               padding=1),
            nn.ReLU(),
        )

        # Fifth decoder layer
        self.dec5_1 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            nn.GLU(dim=1),
        )
        self.dec5_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=first_out_channels<<1, 
                               out_channels=first_out_channels, 
                               kernel_size=enc_kernel_size_1, stride=stride_1, 
                               padding=1),
            nn.ReLU(),
        )

        # Sixth decoder layer
        self.dec6_1 = nn.Sequential(
            nn.Conv1d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=dec_kernel_size, stride=stride_2, padding=1),
            nn.GLU(dim=1),
        )
        self.dec6_2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=first_out_channels, 
                               out_channels=in_channels, 
                               kernel_size=enc_kernel_size_1, stride=stride_1, 
                               padding=1),
        )

    def forward(self, x):
        upsample = nn.Upsample(x.shape[2])

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        #e5 = self.enc5(e4)
        #e6 = self.enc6(e5)

        # BLSTM
        x = e4
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x, _ = self.lstm(x)

        # Linear layer
        x = self.linear(x)
        x = x.reshape(x.size(0), x.size(2), x.size(1))

        # Decoder 
        #x = x + e6
        #x = self.dec1_1(x)
        #x = self.dec1_2(x)
        #dec_upsample = nn.Upsample(size=e5.shape[2])
        #dec_upsample = dec_upsample.to(self.device)
        #x = dec_upsample(x)
        #x = x + e5
        #x = self.dec2_1(x)
        #x = self.dec2_2(x)
        #dec_upsample = nn.Upsample(size=e4.shape[2])
        #dec_upsample = dec_upsample.to(self.device)
        #x = dec_upsample(x)
        x = x + e4
        x = self.dec3_1(x)
        x = self.dec3_2(x)
        dec_upsample = nn.Upsample(size=e3.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = x + e3
        x = self.dec4_1(x)
        x = self.dec4_2(x)
        dec_upsample = nn.Upsample(size=e2.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = x + e2
        x = self.dec5_1(x)
        x = self.dec5_2(x)
        dec_upsample = nn.Upsample(size=e1.shape[2])
        dec_upsample = dec_upsample.to(self.device)
        x = dec_upsample(x)
        x = x + e1
        x = self.dec6_1(x)
        x = self.dec6_2(x)

        # Upsample input to original time length
        x = upsample(x)

        return x
