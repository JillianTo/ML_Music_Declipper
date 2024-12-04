import os
import math
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

class Functional():

    def __init__(self, max_time=None, device=None, n_fft=None, hop_length=None, 
                 augmentation_lbls=None):
        self.max_time = max_time
        self.device = device
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augmentation_lbls = augmentation_lbls

    def mean(tensor):
        return torch.sum(tensor)/torch.numel(tensor)
    
    def upsample(tensor, time):
        upsampler = nn.Upsample(time)
        upsampler = upsampler.to(tensor.device)
        return upsampler(tensor)

    def resample(tensor, old_sample_rate, new_sample_rate):
        resampler = T.Resample(old_sample_rate, new_sample_rate, dtype=
                               tensor.dtype)
        resampler = resampler.to(tensor.device)
        return resampler(tensor)

    def wav_to_spec_db(tensor, n_fft, hop_length, top_db=None):
        mag_spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, 
                                 power=1)
        mag_spec = mag_spec.to(tensor.device)
        amp_to_db = T.AmplitudeToDB("amplitude", top_db=top_db)
        amp_to_db = amp_to_db.to(tensor.device)
        
        return amp_to_db(mag_spec(tensor))  
    
    def wav_to_mel_db(tensor, sample_rate, n_fft, hop_length, n_mels, 
                      top_db=None):      
        mel_spec = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, 
                                    hop_length=hop_length, n_mels=n_mels, 
                                    power=1)
        mel_spec = mel_spec.to(tensor.device)
        amp_to_db = T.AmplitudeToDB("amplitude", top_db=top_db)
        amp_to_db = amp_to_db.to(tensor.device)

        return amp_to_db(mel_spec(tensor))

    def calc_loss(pred, tgt, sample_rate, n_ffts, n_mels=None, 
                  hop_lengths=None, top_db=None):
        loss = 0
        # Calculate loss for all defined FFT sizes
        for i in range(len(n_ffts)):
            # If not defined, set hop length to 1/8th of FFT size
            if hop_lengths == None:
                hop_length = n_ffts[i]>>3
            else:
                hop_length = hop_lengths[i]
            
            # Convert stereo waveform to spectrogram
            pred_spec = Functional.wav_to_spec_db(pred, n_ffts[i], hop_length, 
                                                  top_db=top_db)
            tgt_spec = Functional.wav_to_spec_db(tgt, n_ffts[i], hop_length, 
                                                 top_db=top_db)

            # Calculate L1 loss for left+right channels
            lr_loss = F.l1_loss(pred_spec, tgt_spec)
            
            # Calculate L1 loss for mel spectrogram
            if n_mels != None:
                pred_spec = Functional.wav_to_mel_db(pred, sample_rate, 
                                                     n_ffts[i], hop_length, 
                                                     n_mels[i], top_db=top_db)
                tgt_spec = Functional.wav_to_mel_db(tgt, sample_rate, 
                                                    n_ffts[i], hop_length, 
                                                    n_mels[i], top_db=top_db)
                lr_loss = lr_loss + F.l1_loss(pred_spec, tgt_spec)

            # Combine stereo waveforms to mono
            pred_mono = torch.div(pred[:,0,:]+pred[:,1,:], 2)
            tgt_mono = torch.div(tgt[:,0,:]+tgt[:,1,:], 2)

            # Convert mono waveform to spectrogram
            pred_spec = Functional.wav_to_spec_db(pred_mono, n_ffts[i], 
                                                  hop_length, top_db=top_db)
            tgt_spec = Functional.wav_to_spec_db(tgt_mono, n_ffts[i], 
                                                 hop_length, top_db=top_db)
            # Calculate L1 loss for mono spectrogram
            sum_loss = F.l1_loss(pred_spec, tgt_spec)

            # Calculate L1 loss for mono mel spectrogram
            if n_mels != None:
                pred_spec = Functional.wav_to_mel_db(pred_mono, sample_rate, 
                                                     n_ffts[i], hop_length, 
                                                     n_mels[i], top_db=top_db)
                tgt_spec = Functional.wav_to_mel_db(tgt_mono, sample_rate, 
                                                    n_ffts[i], hop_length, 
                                                    n_mels[i], top_db=top_db)
                sum_loss = sum_loss + F.l1_loss(pred_spec, tgt_spec)

            # Subtract right channel from left channel
            #pred_mono = pred[:,0,:]-pred[:,1,:]
            #tgt_mono = tgt[:,0,:]-tgt[:,1,:]

            # Convert difference waveform to spectrogram
            #pred_spec = Functional.wav_to_spec_db(pred_mono, n_ffts[i], 
            #                                      hop_length, top_db=top_db)
            #tgt_spec = Functional.wav_to_spec_db(tgt_mono, n_ffts[i], 
            #                                     hop_length, top_db=top_db)

            # Calculate L1 loss for difference spectrogram
            #diff_loss = F.l1_loss(pred_spec, tgt_spec)

            # Calculate L1 loss for difference spectrogram
            #if n_mels != None:
            #    pred_spec = Functional.wav_to_mel_db(pred_mono, sample_rate, 
            #                                         n_ffts[i], hop_length, 
            #                                         n_mels[i], top_db=top_db)
            #    tgt_spec = Functional.wav_to_mel_db(tgt_mono, sample_rate, 
            #                                        n_ffts[i], hop_length, 
            #                                        n_mels[i], top_db=top_db)
            #    diff_loss = diff_loss + F.l1_loss(pred_spec, tgt_spec)
            
            # Add loss term for this FFT size to total 
            #loss = loss + (1/len(n_ffts))*(lr_loss*(2/3)+(sum_loss+diff_loss)*(1/3))
            loss = loss + (1/len(n_ffts))*(lr_loss*(2/3)+sum_loss*(1/3))

        # Return loss term averaged across all defined FFT sizes
        return loss
                

    def wav_to_complex(tensor, n_fft, hop_length):
        complex_spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, 
                                     power=None)
        complex_spec = complex_spec.to(tensor.device)
        return complex_spec(tensor)

    def complex_to_wav(tensor, n_fft, hop_length):
        inv_spec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        inv_spec = inv_spec.to(tensor.device)
        return inv_spec(tensor)

    def wav_to_mag_phase(tensor, n_fft, hop_length):
        # Convert input to complex spectrogram
        x = Functional.wav_to_complex(tensor, n_fft, hop_length)

        # Return loss term averaged across all defined FFT sizes
        return loss
                

    def wav_to_complex(tensor, n_fft, hop_length):
        complex_spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, 
                                     power=None)
        complex_spec = complex_spec.to(tensor.device)
        return complex_spec(tensor)

    def complex_to_wav(tensor, n_fft, hop_length):
        inv_spec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        inv_spec = inv_spec.to(tensor.device)
        return inv_spec(tensor)

    def wav_to_mag_phase(tensor, n_fft, hop_length):
        # Convert input to complex spectrogram
        x = Functional.wav_to_complex(tensor, n_fft, hop_length)

        # Calculate phase of complex spectrogram
        phase = torch.atan(x.imag/(x.real+1e-7))
        phase[x.real < 0] += 3.14159265358979323846264338

        # Calculate magnitude of complex spectrogram
        #x = torch.sqrt(torch.pow(x.real,2)+torch.pow(x.imag,2))
        x = x.abs()
        
        # Return magnitude and phase spectrograms 
        return (x, phase)

    def mag_phase_to_wav(mag, phase, n_fft, hop_length):
        # Upsample magnitude to match phase shape
        x = Functional.upsample(mag, [phase.shape[2], phase.shape[3]])
        # Combine magnitude and phase
        x = torch.polar(x, phase)
        # Invert spectrogram
        x = Functional.complex_to_wav(x, n_fft, hop_length)
        
        # Return waveform
        return x

    # Get hyperparameters from file
    def get_hparams(args):
        # If no input argument, use default hyperparameter file path
        if len(args) < 2:
            hparams_path = "hparams.txt"    
        else:
            hparams_path = args[1] 
        
        # Load hyperparameters
        with open(hparams_path, 'rb') as f:
            hparams = pickle.load(f)

        return hparams

    # Get label filename from input filepath by removing augmentation label
    def input_to_label_filepath(self, filepath):
        file_ext = ".wav"
        for lbl in self.augmentation_lbls:
            filepath = filepath.replace(lbl+file_ext,file_ext)
        return filepath

    # Calculate mean and standard deviation in dB for training data spectrograms
    def db_stats(self, input_path, train_lbl_path, stats_time):
        mean = 0
        std = 0
        num_files = 0

        for filename in tqdm(os.listdir(input_path)):
            if filename.endswith('.wav'):
                # Remove augmentation label from filename
                lbl_filename = self.input_to_label_filepath(filename)
                
                # Calculate stats if current file's label exists in training set
                if os.path.isfile(train_lbl_path+lbl_filename):
                    # Load waveform
                    x, _ = torchaudio.load(input_path+filename)
                    x = x.to(self.device)

                    # If no batch dimension, add one
                    if x.ndim < 4:
                        x = x.unsqueeze(0)
                    
                    # If file is above time threshold, trim it
                    if x.shape[2] > stats_time:
                        print(f"\"{filename}\" too long, trimming")
                        x = x[:, :, :stats_time]

                    # Convert input to complex spectrogram
                    x, _ = Functional.wav_to_mag_phase(x, self.n_fft, self.hop_length)

                    # Calculate stats
                    curr_std, curr_mean = torch.std_mean(x) 

                    # Add stats to totals
                    mean = mean + curr_mean
                    std = std + curr_std

                    # Increment file count
                    num_files = num_files+1

        # Calculate average stats
        mean = mean/num_files
        std = std/num_files
        
        # Return stats
        return mean, std

    def pad(self, tensor):
        w_padding = self.max_time - tensor.shape[1]
        left_pad = w_padding // 2
        right_pad = w_padding - left_pad
        return F.pad(tensor, (left_pad, right_pad))

    def get_augmentation_pts(self, filename, pt_time, tot_time):
        if self.augmentation_lbls == None:
            return [[filename, pt_time, tot_time]]
        else:
            augm_pts = []
            file_ext = ".wav"
            for lbl in self.augmentation_lbls:
                augm_pts.append([filename.replace(file_ext, lbl+file_ext), pt_time, tot_time])
            return augm_pts

    def get_filelist(self, input_path, short_thres, filelist_path, 
                     overlap_factor):
        # Filelist was saved previously, load it
        if(filelist_path != None and os.path.isfile(filelist_path)):
            with open(filelist_path, 'rb') as f:
                filelist = pickle.load(f)
            print(f"Loaded filelist from \"{filelist_path}\"")
        # Filelist has not been generated yet, create it
        else:
            filelist = []

            for filename in tqdm(os.listdir(input_path)):
                if filename.endswith('.wav'):
                    # TODO: this does not work for files not 44.1kHz samplerate
                    time = torchaudio.info(input_path+filename).num_frames
                    # Waveform is longer than time cut-off, needs to be split
                    if(time > self.max_time):
                        # Calculate time that parts will overlap
                        overlap_time = int(self.max_time*overlap_factor)
                        # Calculate leftover of waveform after being parted out
                        remainder = time % (self.max_time-overlap_time)
                        # Round down number of parts to split
                        num_parts = int(time/(self.max_time-overlap_time))
                        # Append filename and time start of each split part
                        filelist.extend(self.get_augmentation_pts(filename, 0, time))
                        for i in range(1, num_parts):
                            filelist.extend(self.get_augmentation_pts(filename, ((self.max_time*i)-(overlap_time*i)), time))
                        # If remainder is greater than threshold, add 
                        # waveform ending at end of file with length max_time
                        if(remainder > (self.max_time*short_thres)):
                            filelist.extend(self.get_augmentation_pts(filename, time-self.max_time, time))
                    # Waveform is shorter than or equal to time cut off and 
                    # long enough for FFT, does not need to be split
                    elif (time > self.n_fft):
                        filelist.extend(self.get_augmentation_pts(filename, -1, time))
                    # Waveform not long enough for FFT, skip
                    else:
                        print(f"Skipping \"{input_path}{filename}\" because "
                        "too short for FFT")

            if(filelist_path != None):
                with open(filelist_path, 'wb') as f:
                    pickle.dump(filelist, f)
                print(f"Saved filelist in \"{filelist_path}\'")

        return filelist

    def get_split_wav(self, wav_path, sample_rate, split_start, total_n_frames, pad_short):
        # Waveform does not need to be split
        if(split_start < 0):
            wav, wav_sample_rate = torchaudio.load(wav_path)
            # Pad if all waveforms need to be same time
            if(pad_short):
                wav = self.pad(wav)
        # Crop long waveform
        else:
            end_time = split_start + self.max_time
            # If split waveform length would be less than max_time
            if(end_time > total_n_frames):
                # Pad if all waveforms need to be same time or if waveform is 
                # not long enough for FFT
                wav, wav_sample_rate = torchaudio.load(wav_path, frame_offset=split_start)
                if(end_time < self.n_fft or pad_short):
                    wav = self.pad(wav)
            # Split waveform fits in max_time, return waveform starting at 
            # time split_start with length max_time
            else:
                wav, wav_sample_rate = torchaudio.load(wav_path, frame_offset=split_start, num_frames=self.max_time)

        # Resample if not expected sample rate
        if(wav_sample_rate != sample_rate):
            print(f"\"{wav_path}\" has sample rate of {wav_sample_rate}Hz, "
                  f"resampling")
            wav = Functional.resample(wav, wav_sample_rate, sample_rate)
        
        return wav

    # Return waveform 
    def process_wav(self, path, fileinfo, sample_rate, pad_short, 
                    is_input=True):
        # Load waveform
        filename = fileinfo[0]
        if(not is_input):
            filename = self.input_to_label_filepath(filename)
        wav_path = path + filename
        wav = self.get_split_wav(wav_path, sample_rate, fileinfo[1], 
                                 fileinfo[2], pad_short)

        # Move waveform to device
        #wav = wav.to(self.device)

        # Return waveform
        return wav
    
    # Return input and label waveforms
    def process_wavs(self, input_path, label_path, fileinfo, sample_rate, 
                     pad_short):
        # Load waveforms
        filename = fileinfo[0]
        input_wav_path = input_path + filename
        label_wav_path = label_path + filename
        label_wav_path = self.input_to_label_filepath(label_wav_path)
        split_start = fileinfo[1]
        total_n_frames = fileinfo[2]
        input_wav = self.get_split_wav(input_wav_path, sample_rate, 
                                       split_start, total_n_frames, pad_short)
        label_wav = self.get_split_wav(label_wav_path, sample_rate, 
                                       split_start, total_n_frames, pad_short)

        # Move waveforms to device
        #input_wav = input_wav.to(self.device)
        #label_wav = label_wav.to(self.device)

        # Return input and label waveforms
        return input_wav, label_wav

    def save_wav(tensor, sample_rate, output_path, verbose=True):
        if(tensor.dim() > 2):
            tensor = torch.squeeze(tensor)

        # Input tensor must be on CPU to be saved
        out = tensor.to("cpu")

        # Save the output as a WAV file
        torchaudio.save(output_path, out, sample_rate)
        if verbose:
            print(f"Saved \"{output_path}\"")

