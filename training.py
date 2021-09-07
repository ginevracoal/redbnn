"""
Loads an architecture from torchvision library and trains a deterministic Neural Network (baseNN) or a r
educed Bayesian Neural Network (redBNN) using Stochastic Variational Inference (SVI) or Hamiltonian Monte Carlo (HMC). 
redBNN computes a MAP estimate and then performs Bayesian inference on a chosen layer (--reduction=layers) or block 
(--reduction=blocks). 
"""


import torch
from argparse import ArgumentParser

import redbnn
from redbnn.nn.base import baseNN
from redbnn.nn.reduced import redBNN
from redbnn.utils.data import load_data
from redbnn.utils.pickling import load_from_pickle
import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="imagenette", help="Dataset name.")
parser.add_argument("--architecture", type=str, default="resnet18", help="Torchvision model name.")
parser.add_argument("--model", type=str, default="redBNN", help="Choose the model: 'baseNN', 'redBNN'")
parser.add_argument("--n_inputs", type=int, default=None, help="Number of images. None loads all the available ones.")
parser.add_argument("--n_iters", type=int, default=2, help="Number of training iterations for baseNN.")
parser.add_argument("--inference", type=str, default="svi", help="Inference method: 'svi', 'hmc'.")
parser.add_argument("--reduction", type=str, default="blocks", help="Choose 'blocks' or 'layers' mode.")
parser.add_argument("--bayesian_idx", type=eval, default=4, help="Idx for Bayesian block or layer.")
parser.add_argument("--n_samples", type=int, default=10, help="Number of posterior samples for testing.")
parser.add_argument("--svi_iters", type=int, default=5, help="Number of iterations for SVI.")
parser.add_argument("--hmc_samples", type=int, default=50, help="Number of samples for HMC.")
parser.add_argument("--hmc_warmup", type=int, default=100, help="Number of warmup steps for HMC.")
parser.add_argument("--device", type=str, default="cuda", help="Choose 'cuda' or 'cpu'.")
args = parser.parse_args()

dataloaders, num_classes = load_data(dataset_name=args.dataset, 
                                                 data_dir='data/imagenette2-320',
                                                 subset_size=args.n_inputs)

if args.model=='baseNN':

    model = baseNN(architecture=args.architecture, num_classes=num_classes)

    model.train(dataloaders=dataloaders, num_iters=args.n_iters, device=args.device)
    model.save(filename='baseNN', savedir='data/trained_models/')
    model.evaluate(dataloaders['test'], device=args.device)

elif args.model=='redBNN':

    if args.inference=='hmc':
        assert args.n_samples <= args.hmc_samples

    model = redBNN(architecture=args.architecture, num_classes=num_classes, 
                    inference=args.inference, reduction=args.reduction, bayesian_idx=args.bayesian_idx)
    model.train(dataloaders=dataloaders, num_iters=args.n_iters, device=args.device, 
                eval_samples=1, svi_iters=args.svi_iters, 
                hmc_samples=args.hmc_samples, hmc_warmup=args.hmc_warmup)
    model.save(filename='redBNN_'+str(args.inference), savedir='data/trained_models/', hmc_samples=args.hmc_samples)
    model.evaluate(dataloaders['test'], device=args.device, n_samples=args.n_samples)

else:
    raise AttributeError