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

DEBUG=False  

def model(bayesian_network, x_data, y_data):

    priors = {}
    for key, value in bayesian_network.bayesian_layers_dict.items():
        loc = torch.zeros_like(value)
        scale = torch.ones_like(value)
        prior = Normal(loc=loc, scale=scale)
        priors.update({str(key):prior})

    lifted_module = pyro.random_module("module", bayesian_network.bayesian_subnetwork, priors)()   

    with pyro.plate("data", len(x_data)):
        out = lifted_module(x_data)

        if bayesian_network.bayesian_layers_idxs[-1]==bayesian_network.n_layers-1:
            obs = pyro.sample("obs", Categorical(logits=out), obs=y_data)
        else:
            obs = pyro.sample("obs", Normal(out, 1.), obs=y_data)

        if DEBUG:
            print("obs", obs.shape)


def guide(bayesian_network, x_data, y_data=None):

    dists = {}
    for key, value in bayesian_network.bayesian_layers_dict.items():
        loc = pyro.param(str(f"{key}_loc"), torch.randn_like(value)) 
        scale = pyro.param(str(f"{key}_scale"), torch.randn_like(value))
        distr = Normal(loc=loc, scale=softplus(scale))
        dists.update({str(key):distr})

    if DEBUG:
        print("\nguide dists:", dists)

    lifted_module = pyro.random_module("module", bayesian_network.bayesian_subnetwork, dists)()

    out = lifted_module(x_data)
    preds = nnf.softmax(out, dim=-1)
    return preds


def train(bayesian_network, dataloaders, eval_samples, device, num_iters, is_inception=False):
    print("\n == SVI training ==")
    pyro.clear_param_store()
    bayesian_network.to(device)

    elbo = TraceMeanField_ELBO()
    optimizer = pyro.optim.Adam({"lr":0.001})
    svi = SVI(bayesian_network.model, bayesian_network.guide, optimizer, loss=elbo)

    val_acc_history = []
    since = time.time()

    for epoch in range(num_iters):

        loss=0.0

        print('Epoch {}/{}'.format(epoch, num_iters - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs, labels  = inputs.to(device), labels.to(device)

                lidx = bayesian_network.bayesian_layers_idxs[0]
                ridx = bayesian_network.bayesian_layers_idxs[-1]
                bay_inp = bayesian_network.get_activation(x=inputs, layer_idx=lidx-1).to(device)

                if ridx==bayesian_network.n_layers-1:
                    bay_out = labels
                else:
                    bay_out = bayesian_network.get_activation(x=inputs, layer_idx=ridx).to(device)

                if DEBUG:
                    print("lidx=", lidx, " ridx=", ridx)
                    print("bay_inp", bay_inp.shape, " bay_out", bay_out.shape)

                with torch.set_grad_enabled(phase == 'train'):

                    loss = svi.step(x_data=bay_inp.squeeze(), y_data=bay_out.squeeze())
                    out = bayesian_network.forward(inputs, n_samples=eval_samples)
                    preds = out.argmax(dim=-1)
                    
                running_loss += loss * bay_inp.size(0)

                if DEBUG:
                    print("out", out.shape, "preds", preds.shape, "labels", labels.shape)

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

def forward(bayesian_network, inputs, n_samples, sample_idxs=None):

    if DEBUG:
        print("\nn_samples", n_samples)

    bay_inp = bayesian_network.get_activation(x=inputs, layer_idx=bayesian_network.bayesian_layers_idxs[0]-1)
    guide = bayesian_network.guide

    preds = []  

    for seed in sample_idxs:
        pyro.set_rng_seed(seed)
        guide_trace = poutine.trace(guide).get_trace(bay_inp) 

        weights = {}
        for key, value in bayesian_network.network.state_dict().items():
            weights.update({str(key):value})

        for param_idx, (param_name, param) in enumerate(bayesian_network.bayesian_subnetwork.state_dict().items()):
            loc = guide_trace.nodes["module$$$"+param_name]["value"]
            scale = guide_trace.nodes["module$$$"+param_name]["value"]
            w = Normal(loc=loc, scale=softplus(scale)).sample()

            key = list(bayesian_network.bayesian_layers_dict.keys())[param_idx]
            weights.update({key:w})

            if DEBUG:
                print("\nupdate: ", key, param_name)

        basenet_copy = copy.deepcopy(bayesian_network.network)
        basenet_copy.load_state_dict(weigshts)
        preds.append(basenet_copy.forward(inputs))

    preds = torch.stack(preds)

    if torch.all(preds[0,0,:]==preds[1,0,:]):
        # print(preds.shape)
        # print(preds[:,0,:])
        raise ValueError("Same prediction from independent samples.")

    if DEBUG:   
        print("fwd out", preds.shape)

    return preds
    
def set_params_updates():
    for weights_name in pyro.get_param_store():
        if weights_name not in ["w_mu","w_sigma","b_mu","b_sigma"]:
            pyro.get_param_store()[weights_name].requires_grad=False

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