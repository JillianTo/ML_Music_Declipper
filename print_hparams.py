import os
import sys
import pickle
from functional import Functional

# Load hyperparameters
hparams = Functional.get_hparams(sys.argv)

for hparam in hparams:
    hparam_val = hparams[hparam]
    print(f"{hparam}: {hparam_val}")

preload_train_info_path = hparams["preload_train_info_path"]
if os.path.isfile(preload_train_info_path):
    with open(preload_train_info_path, 'rb') as f:
        train_info = pickle.load(f)
        print(f"start epoch: {train_info[0]}")
        print(f"start idx: {train_info[1]}")
        print(f"run hash: {train_info[2]}")

filelist_path = hparams["train_filelist_path"]
if(os.path.isfile(filelist_path)):
    with open(filelist_path, 'rb') as f:
        filelist = pickle.load(f)
        print(filelist[4][1]/(1-hparams["overlap_factor"]))
        
