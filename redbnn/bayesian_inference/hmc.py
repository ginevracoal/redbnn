import os
import time
import copy 
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as nnf
softplus = torch.nn.Softplus()

import pyro
from pyro import poutine
from pyro.infer.mcmc import MCMC, HMC, NUTS
from pyro.distributions import Normal, Categorical
from collections import OrderedDict

from redbnn.utils.pickling import save_to_pickle, load_from_pickle


def model(redbnn, x_data, y_data):
    """ Stochastic function that implements the generative process and is conditioned on the observations. 

    Parameters:
        x_data (torch.tensor): Observed data points.
        y_data (torch.tensor): Labels of the observed data.

    """
    priors = {}
    for key, value in redbnn.network.named_parameters():
        if key in redbnn.bayesian_weights.keys():

            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

    lifted_module = pyro.random_module("module", redbnn.network, priors)()   

    with pyro.plate("data", len(x_data)):
        out = lifted_module(x_data)
        obs = pyro.sample("obs", Categorical(logits=out), obs=y_data)

def train(redbnn, dataloaders, device, n_samples, warmup, is_inception):
    """ Freezes the deterministic parameters and infers the Bayesian paramaters using Hamiltonian Monte Carlo.

        Parameters:
            dataloaders (dict): Dictionary containing training and validation torch dataloaders.
            device (str): Device chosen for training.
            n_samples (int): Number of Hamiltonian Monte Carlo samples.
            warmup (int): Number of Hamiltonian Monte Carlo warmup samples.

    """
    print("\n == HMC ==")

    if is_inception:
        raise NotImplementedError
    
    device = torch.device(device)
    redbnn.to(device)

    pyro.clear_param_store()    
    num_batches = len(dataloaders['train'])
    n_batch_samples = int(n_samples/num_batches)+1
    print("\nn_batches =",num_batches,"\tbatch_samples =", n_batch_samples)

    kernel = NUTS(redbnn.model, adapt_step_size=True)
    mcmc = MCMC(kernel=kernel, num_samples=n_batch_samples, warmup_steps=warmup, num_chains=1)

    state_dict_keys = list(redbnn.network.state_dict().keys())
    start = time.time()

    posterior_samples=[]
    for phase in ['train']:

        for inputs, labels in tqdm(dataloaders[phase]):
            inputs, labels = inputs.to(device), labels.to(device)

            mcmc_run = mcmc.run(inputs, labels)
            batch_samples = mcmc.get_samples(n_batch_samples)

            for sample_idx in range(n_batch_samples):

                net_copy = copy.deepcopy(redbnn.network)

                weights_dict = {}
                for key, value in redbnn.network.state_dict().items():
                    weights_dict.update({str(key):value})

                for param_name, param in redbnn.bayesian_weights.items():
                    w = batch_samples['module$$$'+param_name][sample_idx]
                    weights_dict.update({param_name:w})
                    assert w.shape==param.shape

                net_copy.load_state_dict(weights_dict)
                posterior_samples.append(net_copy)

    redbnn.posterior_samples = posterior_samples

    print("\nLearned variational params:\n")
    print(list(batch_samples.keys()))

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return posterior_samples

def forward(redbnn, inputs, n_samples, sample_idxs=None, softmax=True):
    """ Forward pass of the inputs through the network using the chosen number of samples.

    Parameters:
        inputs (torch.tensor): Input images.
        n_samples (int, optional): Number of samples drawn during the evaluation.
        samples_idxs (list, optional): Random seeds used for drawing samples. If `samples_idxs` is None it is \
                                        defined as the range of integers from 0 to the maximum number of samples.
        softmax (bool, optional): If True computes the softmax of each output tensor.

    Returns: 
        (torch.Tensor): Output predictions

    """
    if n_samples>len(redbnn.posterior_samples):
        raise ValueError("Too many samples. Max available samples =", len(redbnn.posterior_samples))

    preds = []
    for sample_idx in sample_idxs:
        net = redbnn.posterior_samples[sample_idx]
        net = net.to(inputs.device)
        preds.append(net.forward(inputs))

    preds = torch.stack(preds)
    return nnf.softmax(preds, dim=-1) if softmax else preds
    
def save(redbnn, savedir, filename, hmc_samples):
    """ Saves the learned parameters as torch.tensors on the CPU.

    Parameters:
        savedir (str): Output directory.
        filename (str): Filename.
        hmc_samples (str): Number of samples drawn during HMC inference, needed for saving models \
                            trained with HMC.

    """

    savedir=os.path.join(savedir, filename+"_weights")
    os.makedirs(savedir, exist_ok=True)

    print(hmc_samples)
    for sample_idx in range(hmc_samples):

        weights_dict={}
        for param_name in redbnn.bayesian_weights.keys():
            w = redbnn.posterior_samples[sample_idx].state_dict()[param_name]
            weights_dict.update({param_name:w})

        save_to_pickle(data=weights_dict, path=savedir, filename=filename+"_"+str(sample_idx))

def load(redbnn, savedir, filename, hmc_samples):
    """ Loads the learned parameters.

    Parameters:
        savedir (str): Output directory.
        filename (str): Filename.
        hmc_samples (str): Number of samples drawn during HMC inference, needed for loading models \
                            trained with HMC.

    """    
    savedir=os.path.join(savedir, filename+"_weights")

    posterior_samples=[]
 
    for sample_idx in range(hmc_samples):

        weights_dict = {}
        for key, value in redbnn.network.state_dict().items():
            weights_dict.update({str(key):value})
        weights_dict.update(load_from_pickle(path=savedir, filename=filename+"_"+str(sample_idx)))

        net_copy = copy.deepcopy(redbnn.network)
        net_copy.load_state_dict(weights_dict)
        posterior_samples.append(net_copy)

    redbnn.posterior_samples = posterior_samples

def to(device):
    """ Sends pyro parameters to the chosen device.
    
    Parameters:
        device (str): Name of the chosen device.
    """    
    for k, v in pyro.get_param_store().items():
        pyro.get_param_store()[k] = v.to(device)