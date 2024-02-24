import functional
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, funct, input_path, filelist_path, label_path=None, same_time=True, pad_thres=2, overlap_factor=None):
        self.funct = funct
        self.input_path = input_path
        self.label_path = label_path
        self.same_time = same_time
        self.pad_thres = pad_thres
        self.filelist = funct.get_filelist(self.input_path, self.pad_thres, filelist_path, overlap_factor)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if(self.label_path != None):
            return self.funct.process_wavs(self.input_path, self.label_path, self.filelist[idx], self.same_time)
        else:
            return self.funct.process_wav(self.input_path, self.filelist[idx], self.same_time), self.filelist[idx]
