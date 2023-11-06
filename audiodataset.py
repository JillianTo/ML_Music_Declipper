import functional
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, funct, comp_path, uncomp_path, same_time):
        self.funct = funct
        self.comp_path = comp_path
        self.uncomp_path = uncomp_path
        self.same_time = same_time
        self.filelist = funct.get_filelist(self.comp_path, self.same_time)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        return self.funct.process_wavs(self.comp_path, self.uncomp_path, self.filelist[idx], self.same_time)
