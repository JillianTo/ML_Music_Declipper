import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.init import constant
from torch.nn.parameter import Parameter
import torchaudio.functional as F
import torchaudio.transforms as T
import time

class AutoEncoder(nn.Module):
    def __init__(self, mean, std, n_fft, hop_length, sample_rate, top_db=106,
                 in_channels=2, first_out_channels=32, kernel_size=(3,3), 
                 stride_1=(1,1), stride_2=(2,2), dropout=0.4):
        super(AutoEncoder, self).__init__(),

        self.mean = mean
        self.std = std
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.top_db = top_db
        self.in_channels = in_channels

        #tanh_lims = [16.0, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0]

        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride_2, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride_2, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride_2, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride_2, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride_2, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
        )
        self.enc7 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride_2, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[6], max_val=tanh_lims[6]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<5, first_out_channels<<6),
            nn.Conv2d(in_channels=first_out_channels<<6, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[6], max_val=tanh_lims[6]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<5, first_out_channels<<6),
        )

        # Set HardTanh limit to same as last used encoder layer
        lstm_idx = 5-1

        # Set LSTM channels to output channels of last used encoder layer 
        lstm_channels = first_out_channels<<lstm_idx

        # Long-short term memory
        self.lstm = nn.LSTM(input_size=lstm_channels, 
                            hidden_size=lstm_channels>>1, num_layers=4, 
                            batch_first=True, dropout=dropout, 
                            bidirectional=True)

        # Decoder layers
        self.up_conv_lstm = nn.Sequential(
            nn.ConvTranspose2d(in_channels=lstm_channels, 
                               out_channels=lstm_channels,
                               kernel_size=kernel_size, stride=stride_2, 
                               padding=1),
            nn.Conv2d(in_channels=lstm_channels, 
                      out_channels=lstm_channels>>1, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[lstm_idx], max_val=tanh_lims[lstm_idx]),
            #nn.GroupNorm(lstm_channels>>2, lstm_channels>>1),
        )
        self.dec6 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<6, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
        )
        self.up_conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<5, 
                               out_channels=first_out_channels<<5,
                               kernel_size=kernel_size, stride=stride_2, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
        )
        self.up_conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<4, 
                               out_channels=first_out_channels<<4,
                               kernel_size=kernel_size, stride=stride_2, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
        )
        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<3, 
                               out_channels=first_out_channels<<3,
                               kernel_size=kernel_size, stride=stride_2, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<2, 
                               out_channels=first_out_channels<<2,
                               kernel_size=kernel_size, stride=stride_2, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<1, 
                               out_channels=first_out_channels<<1,
                               kernel_size=kernel_size, stride=stride_2, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=in_channels, kernel_size=kernel_size, 
                      stride=stride_1, padding=1),
            #nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(max(int(in_channels>>1), 1), in_channels),
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
                                     hop_length=self.hop_length, 
                                     power=None)
        complex_spec = complex_spec.to(device)
        x = complex_spec(x)

        # Calculate phase of complex spectrogram
        phase = torch.atan(x.imag/(x.real+1e-7))
        phase[x.real < 0] += 3.14159265358979323846264338

        # Calculate magnitude of complex spectrogram
        x = torch.sqrt(torch.pow(x.real,2)+torch.pow(x.imag,2))

        # Convert magnitude from linear amplitude to dB
        amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=self.top_db)
        amp_to_db = amp_to_db.to(device)
        x = amp_to_db(x)

        # Standardize magnitude
        x = (x-self.mean)/self.std

        # Save encoder layers for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        #x = self.enc4(e3)
        e4 = self.enc4(e3)
        x = self.enc5(e4)
        #e5 = self.enc5(e4)
        #x = self.enc6(e5)
        #e6 = self.enc6(e5)
        #x = self.enc7(e6)

        # Apply tanh to last encoder layer
        x = torch.tanh(x)

        # Flatten encoder output and rearrange to use with LSTM
        freq_dim = x.size(2)
        time_dim = x.size(3)
        x = x.reshape(x.size(0), freq_dim*time_dim, x.size(1))
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), x.size(2), freq_dim, time_dim)

        # Decoder
        x = self.up_conv_lstm(x)
        #dec_upsample = nn.Upsample(size=[e6.shape[2], e6.shape[3]])
        #dec_upsample = dec_upsample.to(device)
        #x = dec_upsample(x)
        #x = torch.cat((x, e6), dim=1)
        #x = self.up_conv6(self.dec6(x))
        #dec_upsample = nn.Upsample(size=[e5.shape[2], e5.shape[3]])
        #dec_upsample = dec_upsample.to(device)
        #x = dec_upsample(x)
        #x = torch.cat((x, e5), dim=1)
        #x = self.up_conv5(self.dec5(x))
        dec_upsample = nn.Upsample(size=[e4.shape[2], e4.shape[3]])
        dec_upsample = dec_upsample.to(device)
        #print(e4.shape)
        #print(x.shape)
        x = dec_upsample(x)
        x = torch.cat((x, e4), dim=1)
        x = self.up_conv4(self.dec4(x))
        dec_upsample = nn.Upsample(size=[e3.shape[2], e3.shape[3]])
        dec_upsample = dec_upsample.to(device)
        #print(e3.shape)
        #print(x.shape)
        x = dec_upsample(x)
        x = torch.cat((x, e3), dim=1)
        x = self.up_conv3(self.dec3(x))
        dec_upsample = nn.Upsample(size=[e2.shape[2], e2.shape[3]])
        dec_upsample = dec_upsample.to(device)
        #print(e2.shape)
        #print(x.shape)
        x = dec_upsample(x)
        x = torch.cat((x, e2), dim=1)
        x = self.up_conv2(self.dec2(x))
        dec_upsample = nn.Upsample(size=[e1.shape[2], e1.shape[3]])
        dec_upsample = dec_upsample.to(device)
        #print(e1.shape)
        #print(x.shape)
        x = dec_upsample(x)
        x = torch.cat((x, e1), dim=1)
        x = self.dec1(x)

        # Upscale magnitude to match phase shape
        mag_upsample = nn.Upsample([phase.shape[2], phase.shape[3]])
        mag_upsample = mag_upsample.to(device)
        #print(phase.shape)
        #print(x.shape)
        x = mag_upsample(x)

        # Unnormalize magnitude
        x = (x*self.std)+self.mean

        # Convert magnitude back to linear amplitude
        x = F.DB_to_amplitude(x, 1, 0.5)
        
        # Combine magnitude and phase
        x = torch.polar(x, phase)

        # Convert spectrogram to waveform
        inv_spec = T.InverseSpectrogram(n_fft=self.n_fft, 
                                        hop_length=self.hop_length)
        inv_spec = inv_spec.to(device)
        x = inv_spec(x)

        # Set output to same time length as original input
        x = upsample(x)
   
        return x
