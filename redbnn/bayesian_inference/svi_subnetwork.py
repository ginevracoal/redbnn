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

from torchsummary import summary

DEBUG = False

def model(nn, x_data, y_data):
    priors = {}

    for key, value in nn.bayesian_subnetwork.named_parameters():
        # if key in nn.bayesian_weights.keys():

        loc = torch.zeros_like(value)
        scale = 10*torch.ones_like(value)
        prior = Normal(loc=loc, scale=scale)
        priors.update({str(key):prior})

    # lifted_module = pyro.random_module("module", redbnn.network, priors)()   
    lifted_module = pyro.random_module("module", nn.bayesian_subnetwork, priors)()   

    with pyro.plate("data", len(x_data)):

        if hasattr(nn, 'activation_subnetwork'):
            x_data = activation_subnetwork(x_data)

        out = lifted_module.forward(x_data)

        if hasattr(nn, 'output_subnetwork'):
            out = output_subnetwork(out)
            obs = pyro.sample("obs", Categorical(logits=out), obs=y_data)
        else:
            obs = pyro.sample("obs", Normal(out, 1.), obs=y_data)

def guide(nn, x_data, y_data=None):

    dists = {}

    for key, value in nn.bayesian_subnetwork.named_parameters():
        # if key in nn.bayesian_weights.keys():
    # for key, value in nn.network.named_parameters():

        loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
        scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))
        distr = Normal(loc=loc, scale=softplus(scale))
        dists.update({str(key):distr})

    lifted_module = pyro.random_module("module", nn.bayesian_subnetwork, dists)()

    if hasattr(nn, 'activation_subnetwork'):
        x_data = nn.activation_subnetwork(x_data)

    # print(x_data.shape)
    # # print(lifted_module)

    # exit()
    out = lifted_module.forward(x_data)

    if hasattr(nn, 'output_subnetwork'):
        out = nn.output_subnetwork(out)

    return out


def train(nn, model, guide, dataloaders, device, num_iters, is_inception=False, lr=0.01):

    print("\n == SVI ==")

    device = torch.device(device)
    nn.network.to(device)

    pyro.clear_param_store()
    elbo = Trace_ELBO()
    optimizer = pyro.optim.Adam({"lr":lr})
    svi = SVI(nn.model, nn.guide, optimizer, loss=elbo)

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

                if DEBUG:
                    nn.network.cuda()
                    summary(nn.network, inputs[0].shape)

                loss = svi.step(x_data=inputs, y_data=labels)
                exit()
                out = nn.network.forward(inputs)
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

def forward(nn, model, guide, inputs, n_samples, sample_idxs=None, softmax=False):

    old_state_dict = redbnn.state_dict()

    if sample_idxs is None:
        sample_idxs = list(range(n_samples))
    else:
        if len(sample_idxs) != n_samples:
            raise ValueError("The number of sample idxs should match the number of posterior samples.")

    preds = []  
    for seed in sample_idxs: 
        pyro.set_rng_seed(seed)

        if hasattr(nn, 'activation_subnetwork'):
            inputs = nn.activation_subnetwork(inputs)

        guide_trace = poutine.trace(guide).get_trace(inputs) 

        weights_dict = {}

        for key, value in network.state_dict().items():
            dist = Normal(loc=guide_trace.nodes[param_name+"_loc"]["value"], 
                          scale=softplus(guide_trace.nodes[param_name+"_scale"]["value"]))
            w = pyro.sample(param_name, dist)
            weights_dict.update({param_name:w})

        net_copy = copy.deepcopy(network)
        net_copy.load_state_dict(weights_dict)
        out = net_copy.forward(inputs)

        #  if hasattr(self, activation_subnetwork):

        # out = output_subnetwork(out)
        out = nnf.softmax(out, dim=-1) if softmax else out
        preds.append(out)

    preds = torch.stack(preds)
    return nnf.softmax(preds, dim=-1) if softmax else preds

def save(network, savedir, filename):
    
    os.makedirs(savedir, exist_ok=True)
    # torch.save(network.state_dict(), os.path.join(savedir, filename+"_weights.pt"))
    param_store = pyro.get_param_store()
    param_store.save(os.path.join(savedir, filename+"_weights.pt"))

def load(savedir, filename):

    param_store = pyro.get_param_store()
    param_store.load(os.path.join(savedir, filename+"_weights.pt"))
    for key, value in param_store.items():
        param_store.replace_param(key, value, value)

    return self

def to(device):
    for k, v in pyro.get_param_store().items():
        pyro.get_param_store()[k] = v.to(device)