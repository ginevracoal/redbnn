import os
import time
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as nnf
softplus = torch.nn.Softplus()

import pyro
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.distributions import Normal, Categorical


def model(network, x_data, y_data):

    priors = {}
    for key, value in network.network.named_parameters():

        if key in network.bayesian_weights.keys():
            loc = torch.zeros_like(value)
            scale = torch.ones_like(value)
            prior = Normal(loc=loc, scale=scale)
            priors.update({str(key):prior})

    lifted_module = pyro.random_module("module", network.network, priors)()   

    with pyro.plate("data", len(x_data)):
        out = lifted_module(x_data)
        obs = pyro.sample("obs", Categorical(logits=out), obs=y_data)

def guide(network, x_data, y_data=None):

    dists = {}
    for key, value in network.network.named_parameters():

        if key in network.bayesian_weights.keys():

            loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
            scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))
            distr = Normal(loc=loc, scale=softplus(scale))
            dists.update({str(key):distr})

    lifted_module = pyro.random_module("module", network.network, dists)()
    out = lifted_module(x_data)

    print(out.shape)

    # if network.bayesian_idx == network.n_layers-1:
    #     out = nnf.softmax(out, dim=-1)

    return out


def train(network, dataloaders, eval_samples, device, num_iters, lr=0.001, is_inception=False):
    print("\n == SVI training ==")

    device = torch.device(device)
    network.to(device)

    pyro.clear_param_store()
    elbo = Trace_ELBO()
    optimizer = pyro.optim.Adam({"lr":lr})
    svi = SVI(network.model, network.guide, optimizer, loss=elbo)

    val_acc_history = []
    since = time.time()

    for epoch in range(num_iters):

        loss=0.0

        print('\nEpoch {}/{}'.format(epoch, num_iters - 1))
        print('-' * 10)

        for phase in ['train']:

            network.network.train() 

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels  = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):

                    loss = svi.step(x_data=inputs, y_data=labels)
                    out = network.forward(inputs, n_samples=eval_samples)
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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return val_acc_history

def forward(network, inputs, n_samples, sample_idxs=None, softmax=False):

    preds = []  
    for seed in sample_idxs:
        pyro.set_rng_seed(seed)
        guide_trace = poutine.trace(network.guide).get_trace(inputs) 

        print(guide_trace.nodes.keys())

        weights = {}
        for key, value in network.network.state_dict().items():
            weights.update({str(key):value})

        for param_name, param in network.bayesian_weights.items():

            dist = Normal(loc=guide_trace.nodes[param_name+"_loc"]["value"], 
                          scale=softplus(guide_trace.nodes[param_name+"_scale"]["value"]))
            w = pyro.sample(param_name, dist)
            weights.update({param_name:w})

        basenet_copy = copy.deepcopy(network.network)
        basenet_copy.load_state_dict(weights)
        preds.append(basenet_copy.forward(inputs))

    preds = torch.stack(preds)

    if torch.all(preds[0,0,:]==preds[1,0,:]):
        raise ValueError("Same prediction from independent samples.")

    return preds

def save(bayesian_network, path, filename):
    os.makedirs(path, exist_ok=True)

    param_store = pyro.get_param_store()
    param_store.save(os.path.join(path, filename+".pt"))

def load(bayesian_network, path, filename):
    param_store = pyro.get_param_store()
    param_store.load(os.path.join(path, filename+".pt"))
    for key, value in param_store.items():
        param_store.replace_param(key, value, value)
    
    print("\nLoading: ", os.path.join(path, filename+".pt"))

def to(device):
    for k, v in pyro.get_param_store().items():
        pyro.get_param_store()[k] = v.to(device)