import subprocess
import os
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
from functional import Functional

xo_freqs = [100, 400, 1600, 6400]
attack_delay = ["0.005,0.1", "0.003,0.05", "0.000625,0.0125", "0.0001,0.025", "0,0.025"]
mcomp_lvls = [3, 5, 7, 9]
comp_lvls = [[0, 0], [0, 0], [0, 0], [0, 0]]
#buffer = 524288
#buffer = 1048576
#buffer = 2097152
buffer = 4194304
input_path = "/mnt/XS70/declip/uncomp/train/"
#input_path = "/mnt/PC801/declip/test/uncomp/"
#tmp_path = "/mnt/PC801/declip/tmp/"
#tmp_path = "/mnt/MP600/declip/tmp/"
tmp_path = "/mnt/XS70/declip/tmp/"
#output_path = "/mnt/PC801/declip/test/comp/"
output_path = "/mnt/XS70/declip/comp/"
#output_path = "/mnt/MP600/declip/comp/"
#output_path = "/mnt/Ultrastar14-05/Music/New Folder/declip/comp/"
file_check = True

# Store parameters of user-defined variables
num_bands = len(attack_delay)
num_mcomp_lvls = len(mcomp_lvls)

# Save strings for master compression level
comp_strs = []
for i in range(num_mcomp_lvls):
    comp_lvl = comp_lvls[i]
    comp_str = ''
    num_subcomp_lvls = len(comp_lvl)
    for j in range(num_subcomp_lvls):
        comp_str = comp_str + str(comp_lvl[j])
    comp_strs.append(comp_str)

for filename in tqdm(os.listdir(input_path)):
    if filename.endswith('.wav'):
        if file_check:
            file_exists = 0
            for i in range(num_mcomp_lvls):
                if os.path.isfile(output_path+os.path.splitext(filename)[0]+f"--{mcomp_lvls[i]}-{comp_strs[i]}--.wav"):
                    file_exists = file_exists + 1

        if file_check and file_exists == num_mcomp_lvls:
            print(f"{filename} already compressed, skipping")
        else:
            sample_rate = torchaudio.info(input_path+filename).sample_rate

            args = ["./create_multiband_splits.sh", input_path + filename, tmp_path]
            for f in xo_freqs:
                args.append(str(f))
            subprocess.check_call(args)

            band_means = []
            for band_idx in range(num_bands):
                wav, sample_rate = torchaudio.load(tmp_path+"soxtmp_"+str(band_idx)+".wav")
                band_means.append(Functional.mean(torch.abs(wav)))

            for i in range(num_mcomp_lvls):
                mcomp_lvl = mcomp_lvls[i]

                args = ["./compand_bands.sh", tmp_path, str(mcomp_lvl), str(buffer)]
                for ad in attack_delay:
                    args.append(ad)
                subprocess.check_call(args)
                
                new_wav = torch.zeros(wav.shape)
                for band_idx in range(num_bands):
                    wav, sample_rate = torchaudio.load(tmp_path+"soxtmpcomp_"+str(band_idx)+".wav")
                    wav = wav * (band_means[band_idx]/Functional.mean(torch.abs(wav)))
                    new_wav = new_wav + wav

                tmp_file = tmp_path+f"pytmp_{mcomp_lvl}.wav"
                Functional.save_wav(new_wav, sample_rate, tmp_file, verbose=False)

                comp_lvl = comp_lvls[i]
                num_subcomp_lvls = len(comp_lvl)
                for j in range(num_subcomp_lvls):
                    subcomp_lvl = comp_lvl[j]
                    if j == (num_subcomp_lvls-1):
                        output_comp = output_path+f"{os.path.splitext(filename)[0]}--{mcomp_lvl}-{comp_strs[i]}--.wav"
                    else:
                        output_comp = tmp_path+f"pytmp_{mcomp_lvl}{j}.wav"
                    args = ["./compand.sh", tmp_file, output_comp, str(subcomp_lvl), str(buffer)]
                    subprocess.check_call(args)
                    os.remove(tmp_file)
                    tmp_file = output_comp




