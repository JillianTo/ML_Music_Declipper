import os
import sys
import pickle
from functional import Functional


hparams = Functional.get_hparams(sys.argv)
filelist_path = hparams["train_filelist_path"]

# Filelist was saved previously, load it
if(filelist_path != None and os.path.isfile(filelist_path)):
    with open(filelist_path, 'rb') as f:
        filelist = pickle.load(f)
    print(f"Loaded filelist from \"{filelist_path}\"")

print(filelist)
print(filelist[0])
