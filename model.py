import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.init import constant
from torch.nn.parameter import Parameter
import torchaudio.functional as F
import torchaudio.transforms as T
from functional import Functional

class Model(nn.Module):
    def __init__(self, n_fft, hop_length, top_db, first_out_channels, 
                 bn_layers, in_channels=2, first_kernel_size=(3,3), 
                 kernel_size=(3,3), stride=(1,1), activation='elu', 
                 cnn_layers=4, norm_groups=None, nhead=8, lstm_dropout=0.5, 
                 mean=None, std=None):
        super(Model, self).__init__(),

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cnn_layers = cnn_layers
        self.nhead = nhead
        self.mean = mean
        self.std = std

        # Determine log range if top_db is provided
        if top_db != None:
            self.top_log = top_db/20
        else:
            self.top_log = top_db

        # Determine which activation function to use
        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'elu':
            act = nn.ELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'identity':
            act = nn.Identity()

        # Encoder layers
        self.encs = nn.ModuleList([nn.Sequential(
                                    nn.Conv2d(in_channels=in_channels, 
                                              out_channels=first_out_channels, 
                                              kernel_size=first_kernel_size, 
                                              stride=stride, padding=1),
                                    nn.Conv2d(in_channels=first_out_channels, 
                                              out_channels=first_out_channels, 
                                              kernel_size=first_kernel_size, 
                                              stride=stride, padding=1),
                                    act,
                                 )])
        for i in range(1, cnn_layers):
            self.encs.append(nn.Sequential())
            self.encs[i].append(nn.Conv2d(in_channels=first_out_channels<<i-1, 
                                          out_channels=first_out_channels<<i, 
                                          kernel_size=kernel_size, 
                                          stride=stride, padding=1))
            if norm_groups != None:
                self.encs[i].append(nn.GroupNorm(norm_groups, 
                                                 first_out_channels<<i))

            self.encs[i].append(nn.Conv2d(in_channels=first_out_channels<<i, 
                                          out_channels=first_out_channels<<i, 
                                          kernel_size=kernel_size, 
                                          stride=stride, padding=1))
            self.encs[i].append(act)

        # Pool layer
        self.pool = nn.MaxPool2d((2,2), stride=2)

        # Bottleneck layer
        # Set transformer channels to output channels of last used encoder 
        # layer 
        bn_channels = first_out_channels<<cnn_layers

        # Last CNN layer before bottleneck
        self.enc_bn = nn.Sequential(
            nn.Conv2d(in_channels=bn_channels>>1, 
                      out_channels=bn_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1),
        )
        
        # Start bottleneck with layer normalization
        self.bn = nn.Sequential(nn.LayerNorm(bn_channels))

        # Use Transformer encoder as bottleneck
        if nhead != None:
            encoder_layer = nn.TransformerEncoderLayer(d_model=bn_channels, 
                                                       nhead=nhead, 
                                                       batch_first=True)
            self.bn.append(nn.TransformerEncoder(encoder_layer, 
                                                 num_layers=bn_layers))
        # Use LSTM as bottleneck
        else:
            self.bn.append(nn.LSTM(input_size=bn_channels,
                                   hidden_size=bn_channels>>1, 
                                   num_layers=bn_layers, batch_first=True, 
                                   dropout=lstm_dropout, bidirectional=True))
       
        # Upsample after bottleneck 
        self.up_conv_bn = nn.Sequential(
            nn.Upsample(scale_factor=(2,2)),
            nn.Conv2d(in_channels=bn_channels, 
                      out_channels=bn_channels>>1, 
                      kernel_size=(2,2), stride=stride, padding=1),
        )

        # Decoder layers
        self.decs = nn.ModuleList([nn.Sequential(
                                    nn.Conv2d(in_channels=first_out_channels<<1, 
                                              out_channels=first_out_channels, 
                                              kernel_size=kernel_size, 
                                              stride=stride, padding=1),
                                    nn.Conv2d(in_channels=first_out_channels, 
                                              out_channels=first_out_channels, 
                                              kernel_size=first_kernel_size, 
                                              stride=stride, padding=1),
                                    act,
                                    nn.Conv2d(in_channels=first_out_channels, 
                                              out_channels=in_channels, 
                                              kernel_size=first_kernel_size, 
                                              stride=stride, padding=1),
                                    nn.Conv2d(in_channels=in_channels, 
                                              out_channels=in_channels, 
                                              kernel_size=(1,1), stride=(1,1), padding=1),
                                 )])

        for i in range(1, cnn_layers):
            # Decoder section
            self.decs.append(nn.Sequential())
            self.decs[i].append(nn.Conv2d(in_channels=first_out_channels<<i+1, 
                                          out_channels=first_out_channels<<i, 
                                          kernel_size=kernel_size, 
                                          stride=stride, padding=1))
            if norm_groups != None:
                nn.GroupNorm(norm_groups, first_out_channels<<i),
            self.decs[i].append(nn.Conv2d(in_channels=first_out_channels<<i, 
                                          out_channels=first_out_channels<<i, 
                                          kernel_size=kernel_size, 
                                          stride=stride, padding=1))
            self.decs[i].append(act)
            
            # Upsample section
            self.decs[i].append(nn.Upsample(scale_factor=(2,2)))
            self.decs[i].append(nn.Conv2d(in_channels=first_out_channels<<i, 
                                          out_channels=first_out_channels<<i-1, 
                                          kernel_size=(2,2), stride=stride, 
                                          padding=1))

    def forward(self, x):
        # Get device that input is on
        device = x.device
        
        # Save time shape of unedited input for upsampling later
        upsample = nn.Upsample(x.shape[2])
        upsample = upsample.to(device)

        # Get magnitude and phase spectrograms from waveform
        x, phase = Functional.wav_to_mag_phase(x, self.n_fft, self.hop_length)

        # Convert magnitude from linear to log10 scale
        x = torch.log10(x)

        # Clamp minimum value
        if self.top_log != None:
            floor = torch.max(x)-self.top_log
        else:
            floor = -10
        x = torch.clamp(x, min=floor)

        # Standardize magnitude
        if self.mean != None:
            x = (x-self.mean)/self.std
        
        # Flip sign
        x = -x

        # Apply first encoder layer
        skips = [self.encs[0](x)]
        
        # Apply encoder layers after first
        for i in range(1, self.cnn_layers):
            skips.append(self.encs[i](self.pool(skips[i-1])))

        # Apply final encoder layer before bottleneck
        x = self.enc_bn(self.pool(skips[self.cnn_layers-1]))

        # Flatten encoder output and rearrange to use with transformer or LSTM
        ch_dim = x.size(1)
        freq_dim = x.size(2)
        time_dim = x.size(3)
        x = x.reshape(x.size(0), freq_dim*time_dim, ch_dim)
        if self.nhead != None:
            x = self.bn(x)
        else:
            x, _ = self.bn(x)

        # Rearrange input to original shape
        x = x.reshape(x.size(0), ch_dim, freq_dim, time_dim)

        # Upsample after LSTM
        x = self.up_conv_bn(x)

        # Apply first decoder layer
        curr_idx = self.cnn_layers-1
        curr_skip = skips[curr_idx]
        dec_upsample = nn.Upsample(size=[curr_skip.shape[2], 
                                         curr_skip.shape[3]])
        dec_upsample = dec_upsample.to(device)
        x = dec_upsample(x)
        x = torch.cat((x, curr_skip), dim=1)
        x = self.decs[curr_idx](x)

        # Apply decoder layers after first
        for i in range(1, self.cnn_layers):
            curr_idx = curr_idx-1
            curr_skip = skips[curr_idx]
            dec_upsample = nn.Upsample(size=[curr_skip.shape[2], 
                                             curr_skip.shape[3]])
            dec_upsample = dec_upsample.to(device)
            x = dec_upsample(x)
            x = torch.cat((x, curr_skip), dim=1)
            x = self.decs[curr_idx](x)

        # Flip sign
        x = -x

        # Unnormalize magnitude
        if self.mean != None:
            x = (x*self.std)+self.mean

        # Clamp minimum value
        if self.top_log != None:
            floor = torch.max(x)-self.top_log
        x = torch.clamp(x, min=floor)

        # Convert magnitude back to linear amplitude
        x = torch.pow(10, x)
        
        # Convert spectrogram to waveform
        x = Functional.mag_phase_to_wav(x, phase, self.n_fft, self.hop_length) 

        # Set output to same time length as original input
        x = upsample(x)
        return x
