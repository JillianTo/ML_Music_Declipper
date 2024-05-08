import functional
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, funct, input_path, filelist_path, label_path=None, pad_short=True, short_thres=0.5, overlap_factor=0):
        self.funct = funct
        self.input_path = input_path
        self.label_path = label_path
        self.pad_short = pad_short
        self.short_thres = short_thres
        self.filelist = funct.get_filelist(self.input_path, self.short_thres, filelist_path, overlap_factor)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if(self.label_path != None):
            return self.funct.process_wavs(self.input_path, self.label_path, self.filelist[idx], self.pad_short)
        else:
            return self.funct.process_wav(self.input_path, self.filelist[idx], self.pad_short), self.filelist[idx]
