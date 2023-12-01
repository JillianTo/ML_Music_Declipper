import functional
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, funct, comp_path, filelist_path, uncomp_path=None, same_time=True, pad_thres=2):
        self.funct = funct
        self.comp_path = comp_path
        self.uncomp_path = uncomp_path
        self.same_time = same_time
        self.pad_thres = pad_thres
        self.filelist = funct.get_filelist(self.comp_path, self.pad_thres, filelist_path)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        if(self.uncomp_path != None):
            return self.funct.process_wavs(self.comp_path, self.uncomp_path, self.filelist[idx], self.same_time)
        else:
            return self.funct.process_wav(self.comp_path, self.filelist[idx], self.same_time), self.filelist[idx]
