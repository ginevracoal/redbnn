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

def train(redbnn, dataloaders, device, n_samples, warmup, is_inception=False):
    print("\n == HMC ==")
    
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
    for k, v in pyro.get_param_store().items():
        pyro.get_param_store()[k] = v.to(device)