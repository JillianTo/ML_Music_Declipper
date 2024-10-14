import subprocess
import os
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
from functional import Functional

xo_freqs = [100, 400, 1600, 6400]
attack_delay = ["0.005,0.1", "0.003,0.05", "0.000625,0.0125", "0.0001,0.025", "0,0.025"]
input_path = "/mnt/MP600/data/test/"
tmp_path = "/mnt/MP600/data/comp/tmp/"
output_path = "/mnt/MP600/data/test/"
num_comps = 3

last_comp = num_comps-1
for filename in tqdm(os.listdir(input_path)):
    if filename.endswith('.wav'):
        sample_rate = torchaudio.info(input_path+filename).sample_rate

        args = ["./create_multiband_splits.sh", input_path + filename, tmp_path]
        for f in xo_freqs:
            args.append(str(f))
        subprocess.check_call(args)

        band_means = []
        num_bands = len(attack_delay)
        for i in range(0, num_bands):
            wav, sample_rate = torchaudio.load(tmp_path+"soxtmp_"+str(i)+".wav")
            band_means.append(Functional.mean(torch.abs(wav)))

        for comp_lvl in range(0,num_comps):
            args = ["./compand_bands.sh", tmp_path, str(comp_lvl)]
            for ad in attack_delay:
                args.append(ad)
            subprocess.check_call(args)
            
            new_wav = torch.zeros(wav.shape)
            for i in range(0, num_bands):
                wav, sample_rate = torchaudio.load(tmp_path+"soxtmpcomp_"+str(i)+".wav")
                wav = wav * (band_means[i]/Functional.mean(torch.abs(wav)))
                new_wav = new_wav + wav

            new_wav = new_wav * (1/torch.max(torch.abs(new_wav)))

            tmp_file = tmp_path+f"pytmp_{comp_lvl}.wav"
            Functional.save_wav(new_wav, sample_rate, tmp_file)

            if(comp_lvl == 0):
                comp_op = 0
            elif(comp_lvl == last_comp):
                comp_op = 1
            else:
                comp_op = 2
            args = ["./compand.sh", tmp_file, output_path+os.path.splitext(filename)[0]+f"--{comp_lvl}", str(comp_op)]
            subprocess.check_call(args)




