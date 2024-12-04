import os
import sys
from tqdm import tqdm
import torch
import torchaudio
from functional import Functional
 
# Load hyperparameters
hparams = Functional.get_hparams(sys.argv)

input_path = hparams["input_data_path"]
#label_paths = [hparams["train_label_data_path"], hparams["val_label_data_path"], hparams["test_label_data_path"]]
label_paths = [hparams["train_label_data_path"], hparams["val_label_data_path"]]
augmentation_lbls = hparams["augmentation_labels"]

funct = Functional(augmentation_lbls=augmentation_lbls) 

for label_path in label_paths:
    print(f"Checking \"{label_path}\"")
    for filename in tqdm(os.listdir(label_path)):
        if filename.endswith('.wav'):
            tgt_time = torchaudio.info(label_path+filename).num_frames
            augmentation_filenames = funct.get_augmentation_pts(filename, None, None)
            for augmentation_filename in augmentation_filenames:
                input_time = torchaudio.info(input_path+augmentation_filename[0]).num_frames
                # Verify prediction and target have same shape
                if(input_time != tgt_time):
                    print(f"WARNING: Input time ({input_time}) does not match label time ({tgt_time}) for \"{filename}\"")

