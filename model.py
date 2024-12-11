import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.init import constant
from torch.nn.parameter import Parameter
import torchaudio.functional as F
import torchaudio.transforms as T
from functional import Functional

class LSTMModel(nn.Module):
    def __init__(self, mean, std,  n_fft, hop_length, top_db, 
                 first_out_channels, lstm_layers, in_channels=2, 
                 kernel_size=(3,3), stride=(1,1), norm_groups=32, 
                 lstm_dropout=0.5):
        super(LSTMModel, self).__init__(),

        self.mean = mean
        self.std = std
        self.top_db = top_db
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        #self.enc5 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<3, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #)
        #self.enc6 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.Conv2d(in_channels=first_out_channels<<5, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #)

        # Tanh activation
        self.tanh = nn.Tanh()        

        # Pool layer
        self.pool = nn.MaxPool2d((2,2), stride=2)

        # Last used encoder layer
        lstm_idx = 4

        # Set LSTM channels to output channels of last used encoder layer 
        lstm_channels = first_out_channels<<lstm_idx

        # Long-short term memory
        self.enc_lstm = nn.Sequential(
            nn.Conv2d(in_channels=lstm_channels>>1, 
                      out_channels=lstm_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        
        self.lstm = nn.Sequential(
            nn.LayerNorm(lstm_channels),
            nn.LSTM(input_size=lstm_channels, 
                    hidden_size=lstm_channels>>1, num_layers=lstm_layers, 
                    batch_first=True, dropout=lstm_dropout, 
                    bidirectional=True)
        )
        
        self.up_conv_lstm = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=lstm_channels, 
                      out_channels=lstm_channels>>1, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )

        # Decoder layers
        #self.dec6 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<6, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.Conv2d(in_channels=first_out_channels<<5, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #)
        #self.up_conv6 = nn.Sequential(
        #    nn.Upsample(scale_factor=(2,2)),
        #    nn.Conv2d(in_channels=first_out_channels<<6, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=(2,2), stride=stride, padding=1),
        #)
        #self.dec5 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<5, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #)
        #self.up_conv5 = nn.Sequential(
        #    nn.Upsample(scale_factor=(2,2)),
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<3, 
        #              kernel_size=(2,2), stride=stride, padding=1),
        #)
        self.dec4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        self.up_conv4 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        self.up_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )
        self.dec1 = nn.Sequential(
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

        # Get magnitude and phase spectrograms from waveform
        x, phase = Functional.wav_to_mag_phase(x, self.n_fft, self.hop_length)

        # Convert magnitude from linear amplitude to dB
        amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=self.top_db)
        amp_to_db = amp_to_db.to(device)
        x = amp_to_db(x)

        # Standardize magnitude
        x = (x-self.mean)/self.std

        # Save encoder layers for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.tanh(self.enc4(self.pool(e3)))
        #e5 = self.enc5(self.pool(e4))
        #e6 = self.enc6(self.pool(e5))
        x = self.enc_lstm(self.pool(e4))

        # Flatten encoder output and rearrange to use with LSTM
        ch_dim = x.size(1)
        freq_dim = x.size(2)
        time_dim = x.size(3)
        x = x.reshape(x.size(0), freq_dim*time_dim, ch_dim)
        x, _ = self.lstm(x)
        # Rearrange input to original shape
        x = x.reshape(x.size(0), ch_dim, freq_dim, time_dim)
        # Upsample after LSTM
        x = self.up_conv_lstm(x)

        # Decoder
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
        x = dec_upsample(x)
        x = torch.cat((x, e4), dim=1)
        x = self.up_conv4(self.dec4(x))
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
        x = self.dec1(x)

        # Unnormalize magnitude
        x = (x*self.std)+self.mean

        # Clamp minimum value
        min_db = torch.max(x)-self.top_db
        x = torch.clamp(x, min=min_db)

        # Convert magnitude back to linear amplitude
        x = torch.pow(10, torch.div(x, 20))
        
        # Convert spectrogram to waveform
        x = Functional.mag_phase_to_wav(x, phase, self.n_fft, self.hop_length) 

        # Set output to same time length as original input
        x = upsample(x)
   
        return x

class TransformerModel(nn.Module):
    def __init__(self, mean, std, n_fft, hop_length, top_db,
                 first_out_channels, tf_layers, in_channels=2, 
                 kernel_size=(3,3), stride=(1,1), norm_groups=32, nhead=8):
        super(TransformerModel, self).__init__(),

        self.mean = mean
        self.std = std
        self.top_db = top_db
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GroupNorm(norm_groups, first_out_channels<<1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GroupNorm(norm_groups, first_out_channels<<2),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GroupNorm(norm_groups, first_out_channels<<3),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
        )
        #self.enc5 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<3, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.GroupNorm(norm_groups, first_out_channels<<4),
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.GELU(),
        #)
        #self.enc6 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.GroupNorm(norm_groups, first_out_channels<<5),
        #    nn.Conv2d(in_channels=first_out_channels<<5, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.GELU(),
        #)

        # Pool layer
        self.pool = nn.MaxPool2d((2,2), stride=2)

        # Last used encoder layer
        tf_idx = 4

        # Set transformer channels to output channels of last used encoder 
        # layer 
        tf_channels = first_out_channels<<tf_idx

        # Create transformer encoder layer with defined parameters
        encoder_layer = nn.TransformerEncoderLayer(d_model=tf_channels, 
                                                   nhead=nhead, 
                                                   batch_first=True)

        # Transformer encoder
        self.enc_tf = nn.Sequential(
            nn.Conv2d(in_channels=tf_channels>>1, 
                      out_channels=tf_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        
        self.tf = nn.Sequential(
            nn.LayerNorm(tf_channels),
            nn.TransformerEncoder(encoder_layer, num_layers=tf_layers),
        )
        
        self.up_conv_tf = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=tf_channels, 
                      out_channels=tf_channels>>1, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )

        # Decoder layers
        #self.dec6 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<6, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.GroupNorm(norm_groups, first_out_channels<<5),
        #    nn.Conv2d(in_channels=first_out_channels<<5, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.Hardtanh(min_val=-tanh_lims[4], max_val=tanh_lims[4]),
        #)
        #self.up_conv6 = nn.Sequential(
        #    nn.Upsample(scale_factor=(2,2)),
        #    nn.Conv2d(in_channels=first_out_channels<<6, 
        #              out_channels=first_out_channels<<5, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #)
        #self.dec5 = nn.Sequential(
        #    nn.Conv2d(in_channels=first_out_channels<<5, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.GroupNorm(norm_groups, first_out_channels<<4),
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<4, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #    nn.Hardtanh(min_val=-tanh_lims[3], max_val=tanh_lims[3]),
        #)
        #self.up_conv5 = nn.Sequential(
        #    nn.Upsample(scale_factor=(2,2)),
        #    nn.Conv2d(in_channels=first_out_channels<<4, 
        #              out_channels=first_out_channels<<3, 
        #              kernel_size=kernel_size, stride=stride, padding=1),
        #)
        self.dec4 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<4, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GroupNorm(norm_groups, first_out_channels<<3),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<3, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
        )
        self.up_conv4 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<3, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GroupNorm(norm_groups, first_out_channels<<2),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<2, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
        )
        self.up_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<2, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GroupNorm(norm_groups, first_out_channels<<1),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels<<1, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=first_out_channels<<1, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.Conv2d(in_channels=first_out_channels, 
                      out_channels=first_out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
            nn.GELU(),
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

        # Get magnitude and phase spectrograms from waveform
        x, phase = Functional.wav_to_mag_phase(x, self.n_fft, self.hop_length)

        # Convert magnitude from linear amplitude to dB
        amp_to_db = T.AmplitudeToDB(stype='amplitude', top_db=self.top_db)
        amp_to_db = amp_to_db.to(device)
        x = amp_to_db(x)

        # Standardize magnitude
        x = (x-self.mean)/self.std

        # Save encoder layers for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        #e5 = self.enc5(self.pool(e4))
        #e6 = self.enc6(self.pool(e5))
        x = self.enc_tf(self.pool(e4))

        # Flatten encoder output and rearrange to use with transformer
        ch_dim = x.size(1)
        freq_dim = x.size(2)
        time_dim = x.size(3)
        x = x.reshape(x.size(0), freq_dim*time_dim, ch_dim)
        x = self.tf(x)
        # Rearrange input to original shape
        x = x.reshape(x.size(0), ch_dim, freq_dim, time_dim)
        # Upsample after LSTM
        x = self.up_conv_tf(x)

        # Decoder
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
        x = dec_upsample(x)
        x = torch.cat((x, e4), dim=1)
        x = self.up_conv4(self.dec4(x))
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
        x = self.dec1(x)

        # Unnormalize magnitude
        x = (x*self.std)+self.mean

        # Clamp minimum value
        min_db = torch.max(x)-self.top_db
        x = torch.clamp(x, min=min_db)

        # Convert magnitude back to linear amplitude
        x = torch.pow(10, torch.div(x, 20))
        
        # Convert spectrogram to waveform
        x = Functional.mag_phase_to_wav(x, phase, self.n_fft, self.hop_length) 

        # Set output to same time length as original input
        x = upsample(x)
   
        return x
