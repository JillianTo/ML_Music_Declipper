import sys
from functional import Functional

# Load hyperparameters
hparams = Functional.get_hparams(sys.argv)

for hparam in hparams:
    hparam_val = hparams[hparam]
    print(f"{hparam}: {hparam_val}")

