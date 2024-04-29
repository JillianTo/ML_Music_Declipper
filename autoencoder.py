import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.init import constant
from torch.nn.parameter import Parameter
import torchaudio.functional as F
import torchaudio.transforms as T

class AutoEncoder(nn.Module):
    def __init__(self, mean, std, n_ffts, hop_lengths, sample_rate, top_db=106,
                 in_channels=2, first_out_channels=32, kernel_size_1=(3,3), 
                 kernel_size_2=(3,3), kernel_size_3=(3,3), stride_1=(1,1), 
                 stride_2=(1,1), stride_3=(2,2), dropout=0.0):
        super(AutoEncoder, self).__init__(),

        self.mean = mean
        self.std = std
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.sample_rate = sample_rate
        self.top_db = top_db
        self.in_channels = in_channels
        self.num_n_fft = len(n_ffts)
        self.center_idx = int(self.num_n_fft/2)

        tanh_lims = [8, 8, 8, 8, 8, 8, 8]

        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
        )
        self.enc1b = nn.Sequential(
            nn.Conv2d(in_channels=in_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
        )
        self.enc1c = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*3, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_3, stride=stride_3, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size_3, stride=stride_3, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size_3, stride=stride_3, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size_1, 
                      stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size_3, stride=stride_3, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size_3, stride=stride_3, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
        )
        self.enc7 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size_3, stride=stride_3, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[6], max_val=tanh_lims[6]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<5, first_out_channels<<6),
            nn.Conv2d(in_channels=first_out_channels<<6, 
                      out_channels=first_out_channels<<6, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[6], max_val=tanh_lims[6]),
            nn.Dropout(dropout),
            #nn.GroupNorm(first_out_channels<<5, first_out_channels<<6),
        )

        # Set HardTanh limit to same as last used encoder layer
        lstm_idx = 4-1 

        # Set LSTM channels to output channels of last used encoder layer 
        lstm_channels = first_out_channels<<lstm_idx

        # Long-short term memory
        self.lstm = nn.Sequential(
            nn.LayerNorm(lstm_channels),
            nn.LSTM(input_size=lstm_channels, hidden_size=lstm_channels>>1, 
                    num_layers=3, batch_first=True, dropout=dropout, 
                    bidirectional=True),
        )

        # Decoder layers
        self.up_conv_lstm = nn.Sequential(
            nn.ConvTranspose2d(in_channels=lstm_channels, 
                               out_channels=lstm_channels,
                               kernel_size=kernel_size_3, stride=stride_3, 
                               padding=1),
            nn.Conv2d(in_channels=lstm_channels, 
                      out_channels=lstm_channels>>1, 
                      kernel_size=kernel_size_2, stride=stride_2, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[lstm_idx], max_val=tanh_lims[lstm_idx]),
            #nn.GroupNorm(lstm_channels>>2, lstm_channels>>1),
        )
        self.dec6 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<6, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<5, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            #nn.GroupNorm(first_out_channels<<4, first_out_channels<<5),
        )
        self.up_conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<5, 
                               out_channels=first_out_channels<<5,
                               kernel_size=kernel_size_3, stride=stride_3, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size_2, stride=stride_2, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[5], max_val=tanh_lims[5]),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<5, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<4, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            #nn.GroupNorm(first_out_channels<<3, first_out_channels<<4),
        )
        self.up_conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<4, 
                               out_channels=first_out_channels<<4,
                               kernel_size=kernel_size_3, stride=stride_3, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size_2, stride=stride_2, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            #nn.GroupNorm(first_out_channels<<2, first_out_channels<<3),
        )
        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<3, 
                               out_channels=first_out_channels<<3,
                               kernel_size=kernel_size_3, stride=stride_3, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size_2, stride=stride_2, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            #nn.GroupNorm(first_out_channels<<1, first_out_channels<<2),
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<2, 
                               out_channels=first_out_channels<<2,
                               kernel_size=kernel_size_3, stride=stride_3, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size_2, stride=stride_2, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[2], max_val=tanh_lims[2]),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            #nn.GroupNorm(first_out_channels, first_out_channels<<1),
        )
        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=first_out_channels<<1, 
                               out_channels=first_out_channels<<1,
                               kernel_size=kernel_size_3, stride=stride_3, 
                               padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_2, stride=stride_2, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[1], max_val=tanh_lims[1]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=in_channels, kernel_size=kernel_size_1, 
                      stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(max(int(in_channels>>1), 1), in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=(1,1), stride=(1,1), padding=1),
        )
        self.dec1b = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=in_channels<<1, kernel_size=kernel_size_1, 
                      stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(in_channels, in_channels<<1),
            nn.Conv2d(in_channels=in_channels<<1, out_channels=in_channels<<1, 
                      kernel_size=(1,1), stride=(1,1), padding=1),
        )
        self.dec1c = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size_1, stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(first_out_channels>>1, first_out_channels),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=in_channels*3, kernel_size=kernel_size_1, 
                      stride=stride_1, padding=1),
            nn.Hardtanh(min_val=-tanh_lims[0], max_val=tanh_lims[0]),
            #nn.GroupNorm(in_channels, in_channels*3),
            nn.Conv2d(in_channels=in_channels*3, out_channels=in_channels*3, 
                      kernel_size=(1,1), stride=(1,1), padding=0),
        )

    def forward(self, x):
        device = x.device

        # Save time shape of unedited input for upsampling later
        upsample = nn.Upsample(x.shape[2])
        upsample = upsample.to(device)
        
        ffts = [torch.zeros(0).to(device)] * self.num_n_fft
        phases = [torch.zeros(0).to(device)] * self.num_n_fft
       
        for i in range(self.num_n_fft):
            # Convert input to complex spectrogram
            complex_spec = T.Spectrogram(n_fft=self.n_ffts[i],
                                         hop_length=self.hop_lengths[i], 
                                         power=None)
            complex_spec = complex_spec.to(device)
            ffts[i] = complex_spec(x)

            # Calculate phase of complex spectrogram
            phases[i] = torch.atan(ffts[i].imag/(ffts[i].real+1e-7))
            phases[i][ffts[i].real < 0] += 3.14159265358979323846264338

            # Calculate magnitude of complex spectrogram
            ffts[i] = torch.sqrt(torch.pow(ffts[i].real,2)+torch.pow(ffts[i].imag,2))

            # Convert magnitude from linear amplitude to dB
            amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=self.top_db)
            amp_to_db = amp_to_db.to(device)
            ffts[i] = amp_to_db(ffts[i])

            # Standardize magnitude
            ffts[i] = (ffts[i]-self.mean)/self.std
            
        if(self.num_n_fft > 1):
            num_batch = ffts[0].size(0)
            last_idx = self.num_n_fft-1
            last_fbins = ffts[last_idx].size(2)
            center_fbins = ffts[self.center_idx].size(2)
            for i in range(self.num_n_fft):
                if i < self.center_idx:
                    fbins = ffts[i].size(2)
                    # Upscale time so it is evenly divisible by largest FFT bin number
                    fft_resample = nn.Upsample(size=(fbins, int((int(((fbins*ffts[i].size(3))/center_fbins)+0.5)*center_fbins)/fbins)))
                    fft_resample = fft_resample.to(device)
                    ffts[i] = fft_resample(ffts[i])
                    # Scale spectrograms to match one stored in middle of list
                    ffts[i] = torch.reshape(ffts[i], (num_batch, self.in_channels, center_fbins, -1))
                else:
                    if i != self.center_idx:
                        ffts[i] = torch.reshape(ffts[i], (num_batch, self.in_channels, center_fbins, -1))
                    fft_resample = nn.Upsample(size=(center_fbins, ffts[0].size(3)))
                    fft_resample = fft_resample.to(device)
                    ffts[i] = fft_resample(ffts[i])
                
            # Concatenate all tensors along channel dimension
            x = torch.cat(ffts, dim=1)
            # Run corresponding first encoder layer
            match(self.num_n_fft):
                case 2:
                    e1 = self.enc1b(x)
                case 3:
                    e1 = self.enc1c(x)
                case _:
                    sys.exit("Number of FFT resolutions not supported!")
        else:
            e1 = self.enc1(ffts[0])

        # Save encoder layers for skip connections
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        x = self.enc4(e3)
        #e4 = self.enc4(e3)
        #x = self.enc5(e4)
        #e5 = self.enc5(e4)
        #x = self.enc6(e5)
        #e6 = self.enc6(e5)
        #x = self.enc7(e6)

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
        #dec_upsample = nn.Upsample(size=[e4.shape[2], e4.shape[3]])
        #dec_upsample = dec_upsample.to(device)
        #x = dec_upsample(x)
        #x = torch.cat((x, e4), dim=1)
        #x = self.up_conv4(self.dec4(x))
        dec_upsample = nn.Upsample(size=[e3.shape[2], e3.shape[3]])
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, e3), dim=1)
        x = self.up_conv3(self.dec3(x))
        dec_upsample = nn.Upsample(size=[e2.shape[2], e2.shape[3]])
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, e2), dim=1)
        x = self.up_conv2(self.dec2(x))
        dec_upsample = nn.Upsample(size=[e1.shape[2], e1.shape[3]])
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, e1), dim=1)
        
        # Run corresponding last decoder layer
        match(self.num_n_fft):
            case 1:
                x = self.dec1(x)
            case 2:
                x = self.dec1b(x)
            case 3:
                x = self.dec1c(x)
            case _:
                sys.exit("Number of FFT resolutions not supported!")
      
        # Split decoder output into the different FFTs and do inverse FFT 
        start_ch = 0
        end_ch = 2 
        for i in range(self.num_n_fft):
            ffts[i] = x[:, start_ch:end_ch, :, :]
            start_ch = end_ch
            end_ch += 2
            fbins =  phases[i].shape[2]
            if i != self.center_idx:
                if fbins > center_fbins: 
                    fft_resample = nn.Upsample(size=(center_fbins, int((int(((ffts[i].size(2)*ffts[i].size(3))/fbins)+0.5)*fbins)/center_fbins)))
                    fft_resample = fft_resample.to(device)
                    ffts[i] = fft_resample(ffts[i])
                     
                ffts[i] = torch.reshape(ffts[i], (num_batch, self.in_channels, fbins, -1))

            # Upscale magnitude to match phase shape
            mag_upsample = nn.Upsample([fbins, phases[i].shape[3]])
            mag_upsample = mag_upsample.to(device)
            ffts[i] = mag_upsample(ffts[i])

            # Unnormalize magnitude
            ffts[i] = (ffts[i]*self.std)+self.mean

            # Convert magnitude back to linear amplitude
            ffts[i] = F.DB_to_amplitude(ffts[i], 1, 0.5)
            
            # Combine magnitude and phase
            ffts[i] = torch.polar(ffts[i], phases[i])

            # Convert spectrogram to waveform
            inv_spec = T.InverseSpectrogram(n_fft=self.n_ffts[i], 
                                            hop_length=self.hop_lengths[i])
            inv_spec = inv_spec.to(device)
            ffts[i] = inv_spec(ffts[i])

        # Combine waveforms
        if(self.num_n_fft > 1):
            # Set outputs to same time length as original input
            for i in range(self.num_n_fft):
                ffts[i] = upsample(ffts[i])
            
            # Combine waveforms from all the FFTs
            x = ffts[0]/self.num_n_fft
            for i in range(1,self.num_n_fft):
                x = x + ffts[i]/self.num_n_fft
        else:
            x = ffts[0]
            # Set output to same time length as original input
            x = upsample(x)
   
        return x
