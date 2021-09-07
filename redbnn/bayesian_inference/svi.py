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
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.distributions import Normal, Categorical


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

def guide(redbnn, x_data, y_data=None):

    dists = {}
    for key, value in redbnn.network.named_parameters():

        if key in redbnn.bayesian_weights.keys():

            loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
            scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))
            distr = Normal(loc=loc, scale=softplus(scale))
            dists.update({str(key):distr})

    lifted_module = pyro.random_module("module", redbnn.network, dists)()
    out = lifted_module(x_data)
    return out


def train(redbnn, dataloaders, device, num_iters, lr=0.01, is_inception=False):
    print("\n == SVI ==")

    device = torch.device(device)
    redbnn.to(device)

    pyro.clear_param_store()
    elbo = Trace_ELBO()
    optimizer = pyro.optim.Adam({"lr":lr})
    svi = SVI(redbnn.model, redbnn.guide, optimizer, loss=elbo)

    val_acc_history = []
    start = time.time()

    for epoch in range(num_iters):

        loss=0.0

        print('\nEpoch {}/{}'.format(epoch, num_iters - 1))
        print('-' * 10)

        for phase in ['train','val']:

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs, labels  = inputs.to(device), labels.to(device)

                loss = svi.step(x_data=inputs, y_data=labels)
                out = redbnn.forward(inputs, n_samples=10)
                preds = out.argmax(dim=-1)
                    
                running_loss += loss * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), end="\t")

            if phase == 'val':
                print()

            val_acc_history.append(epoch_acc)

        print()

    print("\nLearned variational params:\n")
    print(pyro.get_param_store().get_all_param_names())

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return val_acc_history

def forward(redbnn, inputs, n_samples, sample_idxs=None, softmax=True):

    old_state_dict = redbnn.network.state_dict()

    preds = []  
    for seed in sample_idxs: 
        pyro.set_rng_seed(seed)
        guide_trace = poutine.trace(redbnn.guide).get_trace(inputs) 

        weights_dict = {}
        for key, value in redbnn.network.state_dict().items():
            weights_dict.update({str(key):value})

        for param_name, param in redbnn.bayesian_weights.items():

            dist = Normal(loc=guide_trace.nodes[param_name+"_loc"]["value"], 
                          scale=softplus(guide_trace.nodes[param_name+"_scale"]["value"]))
            w = pyro.sample(param_name, dist)
            weights_dict.update({param_name:w})

        net_copy = copy.deepcopy(redbnn.network)
        net_copy.load_state_dict(weights_dict)
        out = net_copy.forward(inputs)
        out = nnf.softmax(out, dim=-1) if softmax else out
        preds.append(out)

    preds = torch.stack(preds)
    return nnf.softmax(preds, dim=-1) if softmax else preds

def save(redbnn, savedir, filename):
    os.makedirs(savedir, exist_ok=True)
    torch.save(redbnn.network.state_dict(), os.path.join(savedir, filename+"_weights.pt"))
    param_store = pyro.get_param_store()
    param_store.save(os.path.join(savedir, filename+"_weights.pt"))

def load(redbnn, savedir, filename):
    saved_param_store = torch.load(os.path.join(savedir, filename+"_weights.pt"))
    param_store = pyro.get_param_store()
    param_store.load(os.path.join(savedir, filename+"_weights.pt"))
    for key, value in param_store.items():
        param_store.replace_param(key, value, value)

    return redbnn

def to(device):
    for k, v in pyro.get_param_store().items():
        pyro.get_param_store()[k] = v.to(device)