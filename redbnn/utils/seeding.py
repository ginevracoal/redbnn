import torch
import numpy as np
import random
import pyro

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    pyro.set_rng_seed(seed)

