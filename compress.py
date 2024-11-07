import subprocess
import os
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
from functional import Functional

xo_freqs = [100, 400, 1600, 6400]
attack_delay = ["0.005,0.1", "0.003,0.05", "0.000625,0.0125", "0.0001,0.025", "0,0.025"]
#mcomp_lvls = [0, 1, 2, 3, 4, 5]
#mcomp_lvls = [2, 3, 4]
mcomp_lvls = [2, 4]
#mcomp_lvls = [3]
#comp_lvls = [[5], [4], [3], [2], [1], [0]]
#comp_lvls = [[3], [2], [1]]
comp_lvls = [[3], [1]]
#comp_lvls = [[2]]
input_path = "/mnt/MP600/data/uncomp/"
#input_path = "/mnt/PC801/declip/test/uncomp/"
#tmp_path = "/mnt/MP600/data/comp/tmp/"
tmp_path = "/mnt/PC801/declip/tmp/"
#output_path = "/mnt/MP600/data/comp/"
#output_path = "/mnt/PC801/declip/test/comp/"
output_path = "/mnt/Ultrastar14-05/Music/New Folder/declip/comp/"

# Store parameters of user-defined variables
num_bands = len(attack_delay)
num_mcomp_lvls = len(mcomp_lvls)

for filename in tqdm(os.listdir(input_path)):
    if filename.endswith('.wav'):
        file_exists = 0
        for i in range(0, num_mcomp_lvls):
            if os.path.isfile(output_path+os.path.splitext(filename)[0]+f"--{mcomp_lvls[i]}{comp_lvls[i][0]}--.wav"):
                file_exists = file_exists + 1

        if file_exists == num_mcomp_lvls:
            print(f"{filename} already compressed, skipping")
        else:
            sample_rate = torchaudio.info(input_path+filename).sample_rate

            args = ["./create_multiband_splits.sh", input_path + filename, tmp_path]
            for f in xo_freqs:
                args.append(str(f))
            subprocess.check_call(args)

            band_means = []
            for band_idx in range(0, num_bands):
                wav, sample_rate = torchaudio.load(tmp_path+"soxtmp_"+str(band_idx)+".wav")
                band_means.append(Functional.mean(torch.abs(wav)))

            for i in range(0, num_mcomp_lvls):
                mcomp_lvl = mcomp_lvls[i]

                args = ["./compand_bands.sh", tmp_path, str(mcomp_lvl)]
                for ad in attack_delay:
                    args.append(ad)
                subprocess.check_call(args)
                
                new_wav = torch.zeros(wav.shape)
                for band_idx in range(0, num_bands):
                    wav, sample_rate = torchaudio.load(tmp_path+"soxtmpcomp_"+str(band_idx)+".wav")
                    wav = wav * (band_means[band_idx]/Functional.mean(torch.abs(wav)))
                    new_wav = new_wav + wav

                new_wav = new_wav * (1/torch.max(torch.abs(new_wav)))

                tmp_file = tmp_path+f"pytmp_{mcomp_lvl}.wav"
                Functional.save_wav(new_wav, sample_rate, tmp_file)

                for comp_lvl in comp_lvls[i]:
                    args = ["./compand.sh", tmp_file, output_path+os.path.splitext(filename)[0]+f"--{mcomp_lvl}", str(comp_lvl)]
                    subprocess.check_call(args)

                os.remove(tmp_file)




