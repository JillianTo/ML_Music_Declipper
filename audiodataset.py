import functional
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, funct, input_path, filelist_path, sample_rate, lbl_path=None, pad_short=False, short_thres=0.5, overlap_factor=0, rtn_input_wav=True, rtn_lbl_wav=False):
        self.funct = funct
        self.input_path = input_path
        self.lbl_path = lbl_path
        self.sample_rate = sample_rate
        self.pad_short = pad_short
        self.short_thres = short_thres
        self.rtn_input_wav = rtn_input_wav
        self.rtn_lbl_wav = rtn_lbl_wav
        if lbl_path != None:
            self.filelist = funct.get_filelist(self.lbl_path, self.short_thres, filelist_path, overlap_factor)
        else:
            self.filelist = funct.get_filelist(self.input_path, self.short_thres, filelist_path, overlap_factor)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if self.rtn_lbl_wav:
            return self.funct.process_wavs(self.input_path, self.lbl_path, self.filelist[idx], self.sample_rate, self.pad_short)
        elif self.rtn_input_wav:
            return self.funct.process_wav(self.input_path, self.filelist[idx], self.sample_rate, self.pad_short), self.filelist[idx]
        else:
            return -1, self.filelist[idx]
